"""Complete agent-teardown protocol for the delete-while-ingest hardening.

The delete flow writes the persistent marker ``agents:{agent_id}:ingestion:delete``
(no TTL) at the START of the agent teardown, so in-flight ingestion workers
(EffING's ``ingestion_canceled``) see it and self-abort while the teardown is
still running. Invariant: marker present <=> teardown incomplete.

This module owns the whole protocol, not just the marker:

1. wait (bounded) for the agent's ingestion rows to reach a terminal state;
2. run the per-source deletion cascade for the files captured *before* the
   wipe (``before_file_manager_file_delete`` while the point metadata is still
   present -> delete the source's points -> ``after_file_manager_file_deleted``),
   then purge all remaining vector memory as the catch-all;
3. remove the file storage folder;
4. wipe the agent's Redis namespace, preserving the marker;
5. fire ``after_cheshire_cat_destroy`` once the namespace is already wiped;
6. clear the marker LAST, so a crash at any point leaves a resumable marker
   that ``complete_pending_deletes`` can pick up.

The cascade runs BEFORE the wipe: the wipe deletes the stored file settings, so
the sources are captured first, and a crash between the wipe and the cascade
would leave only a marker while points and files still exist — which the
marker-only short-circuit would then silently clear. With this order, "only the
marker remains" genuinely means the cascade already ran and there is nothing
left to purge.

The destroy hook fires AFTER the wipe: a plugin handler may delete and
re-store the marker (EffING's ``clear_agent``), which is harmless once the
namespace holds nothing else. Firing it before the wipe would let a crash
between the plugin's delete and re-store lose the marker while the master
key, settings and conversations still exist.

The marker key format mirrors the ``ingestion_status`` plugin registry
(EffING's ``registry.delete_marker_key``): the plugin owns the single source
of truth, but the core MUST NOT import ``cat.core_plugins`` (core must not
depend on plugins), so the key literal and the marker helpers are re-declared
here and must be kept in sync with the plugin registry.

Import-safe: this module is imported by the core at startup; it performs no
network/Redis/filesystem work at import time.
"""
import asyncio
import contextvars
import time

from cat.db import crud
from cat.db.database import DEFAULT_AGENTS_KEY
from cat.db.models import generate_timestamp
from cat.env import get_env_int
from cat.log import log

#: Terminal ingestion statuses (see ``IngestionStatus`` in the ingestion
#: plugin). Anything else — including an unknown/future status — keeps the
#: agent busy: terminal-first is the safe choice.
_TERMINAL_INGESTION_STATUSES = frozenset({"completed", "error"})

#: Default quiesce/poll windows.
_DEFAULT_QUIESCE_SECONDS = 180
_DEFAULT_POLL_SECONDS = 5

#: The file-manager scope passed to the deletion hooks for agent-root files
#: (mirrors ``DELETE /file_manager/files`` for an agent-level path).
_AGENT_FILE_SCOPE = "agent"

#: Task-local set of ``(operation, agent_id)`` pairs currently executing.
#: ``ContextVar`` is per-task (and copied into child tasks), so a plugin hook
#: that re-enters create/destroy for the SAME id is caught immediately instead
#: of self-blocking on the nested distributed lock until its timeout. A
#: create -> teardown transition for the same id is legitimate: they are
#: different operations. delete -> delete and create -> create are not.
_ACTIVE_OPERATIONS: contextvars.ContextVar[frozenset[tuple[str, str]]] = contextvars.ContextVar(
    "agent_deletion_active_operations", default=frozenset()
)


def _enter_operation(operation: str, agent_id: str) -> contextvars.Token[frozenset[tuple[str, str]]]:
    """Register ``(operation, agent_id)`` and reject a re-entrant same pair.

    Returns the token to pass to :func:`_exit_operation` in a ``finally``.
    Raises ``RuntimeError`` immediately (no lock wait) when the same operation
    is already running for this agent in the current context.
    """
    active = _ACTIVE_OPERATIONS.get()
    pair = (operation, agent_id)
    if pair in active:
        raise RuntimeError(
            f"Re-entrant `{operation}` for agent `{agent_id}`: a plugin hook "
            "fired during create/destroy must not call the same operation for "
            "the same agent id."
        )
    return _ACTIVE_OPERATIONS.set(active | {pair})


def _exit_operation(token: contextvars.Token[frozenset[tuple[str, str]]]) -> None:
    """Undo the registration performed by :func:`_enter_operation`."""
    _ACTIVE_OPERATIONS.reset(token)


def _teardown_lock_budget(quiesce: int) -> int:
    """Worst-case wall-clock budget for one full agent teardown.

    Phases covered: the ingestion quiesce window (``quiesce``), the namespace
    wipe, the per-source memory/file cascade, the blanket memory purge, the
    light-cat boot/shutdown, and the plugin hooks. Doubled to leave headroom
    over a slow-but-healthy run, plus a fixed 300s for the phases that are not
    governed by the quiesce window.
    """
    return 2 * quiesce + 300


def _delete_marker_key(agent_id: str) -> str:
    """Redis key of the deletion marker for an agent."""
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:ingestion:delete"


def _escape_glob(agent_id: str) -> str:
    """Escape Redis glob metacharacters in a user-controlled agent id.

    ``agent_id`` comes straight from ``POST /agents/create``, so it is
    untrusted: an id like ``a*`` must not turn a teardown pattern into a
    wildcard that wipes unrelated agents. Every pattern built from an agent id
    MUST go through this helper. Backslash is escaped first so it cannot
    neutralize the escapes added afterwards.
    """
    escaped = agent_id
    for char in ("\\", "*", "?", "[", "]"):
        escaped = escaped.replace(char, "\\" + char)
    return escaped


async def set_delete_marker(agent_id: str) -> None:
    """Write the deletion marker for an agent (idempotent).

    Called at the START of the teardown so in-flight ingestion workers see
    the marker and self-abort. The marker is persistent (no TTL) and is
    removed only once the memory and files are gone.
    """
    await crud.store(
        _delete_marker_key(agent_id),
        {"delete_marker": True, "started_at": generate_timestamp()},
    )


async def has_delete_marker(agent_id: str) -> bool:
    """Whether the deletion marker is present (teardown incomplete).

    Presence-only: tolerates both this module's schema (``delete_marker`` +
    ``started_at``) and the plugin's ``{"delete_marker": True}``-only rows.
    """
    return await crud.read(_delete_marker_key(agent_id)) is not None


#: Alias kept for callers that phrase the question as "is a delete pending?".
has_pending_delete = has_delete_marker


async def clear_delete_marker(agent_id: str) -> None:
    """Remove the deletion marker (teardown completion), and nothing else."""
    await crud.delete(_delete_marker_key(agent_id))


def _unwrap_doc(doc):
    """Unwrap the RedisJSON array wrapper returned by ``crud.read``.

    ``json().get(key, "$")`` wraps a document in a one-element list; older
    pipeline writes can wrap it twice. Unwrap defensively.
    """
    while isinstance(doc, list) and len(doc) == 1 and isinstance(doc[0], (dict, list)):
        doc = doc[0]
    return doc


async def wait_for_ingestion_quiesce(
    agent_id: str,
    quiesce_seconds: int = _DEFAULT_QUIESCE_SECONDS,
    poll_seconds: float = _DEFAULT_POLL_SECONDS,
) -> None:
    """Wait (bounded) for the agent's in-flight ingestion workers to settle.

    Scans ``agents:{escaped_agent_id}:ingestion:*``, skips the marker (which
    would otherwise keep the loop running for the whole timeout), and treats
    a row as busy only when it has a ``status`` outside the terminal set.
    Sleeps ``poll_seconds`` between polls until settled or the deadline; on
    timeout it logs a warning and returns. The marker is the broadcast that
    makes workers self-abort, so this is bounded assurance, not a hard gate.
    """
    marker_key = _delete_marker_key(agent_id)
    pattern = f"{DEFAULT_AGENTS_KEY}:{_escape_glob(agent_id)}:ingestion:*"
    deadline = time.monotonic() + quiesce_seconds
    poll = max(float(poll_seconds), 0.01)

    while True:
        busy = False
        for key in await crud.scan_keys(pattern):
            if key == marker_key:
                continue

            doc = _unwrap_doc(await crud.read(key))
            if not isinstance(doc, dict):
                continue

            # Terminal-first: any status not explicitly terminal is still busy.
            if "status" in doc and doc["status"] not in _TERMINAL_INGESTION_STATUSES:
                busy = True
                break

        if not busy:
            return

        if time.monotonic() >= deadline:
            log.warning(
                f"Agent {agent_id}: ingestion still active after {quiesce_seconds}s, "
                "proceeding with teardown (marker broadcast; workers self-abort)"
            )
            return

        await asyncio.sleep(poll)


async def wipe_agent_namespace_excluding_marker(agent_id: str) -> int:
    """Delete every ``agents:{agent_id}:*`` key except the delete marker.

    Removes the master key (``agents:<id>:agent``) and every settings,
    conversation, plugin and user key under the agent namespace. Returns the
    number of keys actually deleted.
    """
    return await crud.destroy(
        f"{DEFAULT_AGENTS_KEY}:{_escape_glob(agent_id)}:*",
        exclude={_delete_marker_key(agent_id)},
    )


async def _boot_light_cat(agent_id: str):
    """Boot just enough of a ``CheshireCat`` to reach memory and files."""
    # Imported dynamically: ``cheshire_cat`` lazily imports this module from
    # ``destroy``, so a static import here would be a cycle (and widen the
    # core startup import graph).
    from importlib import import_module

    module = import_module("cat.looking_glass.cheshire_cat")
    return await module.CheshireCat.create(agent_id)


async def _teardown_memory_and_files(ccat, sources) -> None:
    """Cascade per source, purge remaining memory, remove the file folder.

    ``sources`` is the file list captured BEFORE the namespace wipe: the wipe
    deletes the stored file settings, so it must not be read from storage at
    this point.

    For each source the order is: ``before_file_manager_file_delete`` (the
    point metadata is still present, so plugins can e.g. drop extracted images
    tied to the source) -> ``delete_tenant_points`` for that source ->
    ``after_file_manager_file_deleted``. Only afterwards is ``destroy_memory``
    run as the catch-all purge for everything not covered by a source; the
    ``before`` hook must fire while the points still exist for the cascade to
    have any effect.

    This whole function runs BEFORE the namespace wipe: a crash here leaves the
    marker plus the namespace, so recovery boots a cat and retries the cascade.
    Reaching the marker-only state is therefore only possible once this has
    completed. The ``after_cheshire_cat_destroy`` hook is NOT fired here; the
    caller fires it after the wipe so a plugin that deletes and re-stores the
    marker cannot leave a half-deleted agent.
    """
    # Lazy imports keep the startup import graph small and avoid cycles.
    from cat.routes.routes_utils import sanitize_source_name
    from cat.services.memory.models import VectorMemoryType

    for source in sources:
        sanitized_source = sanitize_source_name(source, ccat.agent_key)

        # Plugins see the file gone from the list but its memory points still
        # present (e.g. multimodal_ingestion removes extracted images).
        await ccat.plugin_manager.execute_hook(
            "before_file_manager_file_delete",
            sanitized_source,
            _AGENT_FILE_SCOPE,
            caller=ccat,
        )

        await ccat.vector_memory_handler.delete_tenant_points(
            str(VectorMemoryType.DECLARATIVE), {"source": sanitized_source}
        )

        # Points are gone: plugins can drop per-source rows (ingestion status).
        await ccat.plugin_manager.execute_hook(
            "after_file_manager_file_deleted",
            sanitized_source,
            _AGENT_FILE_SCOPE,
            caller=ccat,
        )

    # Catch-all purge for any memory not tied to a captured source. Runs after
    # the per-source cascade, and always, even when no sources were captured.
    await ccat.destroy_memory()

    ccat.file_manager.remove_folder(ccat.agent_key)


def _env_int(name: str, default: int) -> int:
    value = get_env_int(name)
    return default if value is None else value


async def _namespace_has_only_marker(agent_id: str) -> bool:
    """Whether the agent namespace is empty apart from the delete marker."""
    keys = await crud.scan_keys(f"{DEFAULT_AGENTS_KEY}:{_escape_glob(agent_id)}:*")
    return keys == [_delete_marker_key(agent_id)]


async def complete_agent_teardown(agent_id: str, ccat=None) -> None:
    """Finish an agent teardown, resumably and idempotently.

    Serialized per agent by ``agent-delete:<id>``. Returns immediately when
    the marker is already gone (a previous run completed, or a duplicate call
    arrived late), so re-running on a half-torn-down agent is safe.

    ``ccat`` may be a live cat owned by the caller (it is left running) or
    ``None``, in which case a light cat is booted and shut down here — unless
    the namespace already holds only the marker (a previous run crashed after
    the cascade AND the wipe), in which case there is nothing left to purge and
    the marker is simply cleared without booting a cat.

    Re-entrant calls for the same agent are rejected with ``RuntimeError``
    before touching the lock (see :func:`_enter_operation`).
    """
    token = _enter_operation("teardown", agent_id)
    try:
        quiesce_seconds = _env_int("CAT_INGESTION_DELETE_QUIESCE_SECONDS", _DEFAULT_QUIESCE_SECONDS)
        poll_seconds = _env_int("CAT_INGESTION_DELETE_POLL_SECONDS", _DEFAULT_POLL_SECONDS)
        lock_budget = _teardown_lock_budget(quiesce_seconds)

        async with crud.distributed_lock(
            f"agent-delete:{agent_id}",
            timeout=lock_budget,
            blocking_timeout=lock_budget,
        ):
            if not await has_delete_marker(agent_id):
                return None

            # Crash recovery: the cascade and the wipe already ran, so only the
            # marker is left. This is only reachable AFTER the cascade, i.e. the
            # memory purge and folder removal already completed; clear the
            # marker without booting a light cat, otherwise a crash at the very
            # end would leave a permanently stuck marker.
            if ccat is None and await _namespace_has_only_marker(agent_id):
                await clear_delete_marker(agent_id)
                return None

            booted = False
            if ccat is None:
                ccat = await _boot_light_cat(agent_id)
                booted = True

            try:
                # ORDERING FIX: the per-file cascade needs the file list, but the
                # namespace wipe deletes the stored file settings. Capture the
                # sources from the live cat BEFORE anything is destroyed; read
                # them afterwards and they are gone. (MyCAT captured after the
                # wipe and lost the cascade on every fully-provisioned agent.)
                sources = [f.name for f in ccat.file_manager.list_files(ccat.agent_key)]

                await wait_for_ingestion_quiesce(agent_id, quiesce_seconds, poll_seconds)

                # Cascade BEFORE the wipe: memory and files must be gone while
                # the marker still marks the teardown incomplete. If this raises,
                # the marker stays and the namespace is untouched, so recovery
                # can retry the cascade.
                await _teardown_memory_and_files(ccat, sources)

                # Wipe the remaining Redis namespace, preserving the marker, only
                # once memory and files are actually gone.
                await wipe_agent_namespace_excluding_marker(agent_id)

                # Fire the destroy hook AFTER the wipe: a plugin handler may
                # delete and re-store the marker, which is harmless once nothing
                # else remains. Firing it earlier could lose the marker mid-way
                # while the master key/settings/conversations still existed.
                await ccat.plugin_manager.execute_hook(
                    "after_cheshire_cat_destroy",
                    ccat.agent_key,
                    caller=ccat,
                )

                # Last: the marker means "teardown incomplete", so it can only go
                # away once memory, files and the namespace are gone. A crash
                # before this line leaves a resumable marker.
                await clear_delete_marker(agent_id)
            finally:
                if booted:
                    # Only a cat booted here is ours to shut down: the caller owns
                    # a live ``ccat``.
                    await ccat.shutdown()
    finally:
        _exit_operation(token)

    return None


async def complete_pending_deletes() -> None:
    """Resume every agent whose teardown left a marker behind.

    Never raises for a single agent: one broken teardown must not stop the
    others from completing.
    """
    prefix = f"{DEFAULT_AGENTS_KEY}:"
    suffix = ":ingestion:delete"
    try:
        keys = await crud.scan_keys(f"{DEFAULT_AGENTS_KEY}:*:ingestion:delete")
    except Exception as e:  # noqa: BLE001 - background recovery must not crash the caller
        log.error(f"Could not scan pending agent deletions: {e}")
        return

    for key in keys:
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue

        agent_id = key[len(prefix):-len(suffix)]
        if not agent_id:
            continue

        try:
            await complete_agent_teardown(agent_id)
        except Exception as e:  # noqa: BLE001 - one failed agent must not block the rest
            log.error(f"Failed to complete pending deletion for agent {agent_id}: {e}")
