"""Complete agent-teardown protocol for the delete-while-ingest hardening.

The delete flow writes the persistent marker ``agents:{agent_id}:ingestion:delete``
(no TTL) at the START of the agent teardown, so in-flight ingestion workers
(EffING's ``ingestion_canceled``) see it and self-abort while the teardown is
still running. Invariant: marker present <=> teardown incomplete.

This module owns the whole protocol, not just the marker:

1. wait (bounded) for the agent's ingestion rows to reach a terminal state;
2. wipe the agent's Redis namespace, preserving the marker;
3. purge vector memory and run the per-source deletion cascade for the files
   captured *before* the wipe;
4. remove the file storage folder and fire ``after_cheshire_cat_destroy``;
5. clear the marker LAST, so a crash at any point leaves a resumable marker
   that ``complete_pending_deletes`` can pick up.

The marker key format mirrors the ``ingestion_status`` plugin registry
(EffING's ``registry.delete_marker_key``): the plugin owns the single source
of truth, but the core MUST NOT import ``cat.core_plugins`` (core must not
depend on plugins), so the key literal and the marker helpers are re-declared
here and must be kept in sync with the plugin registry.

Import-safe: this module is imported by the core at startup; it performs no
network/Redis/filesystem work at import time.
"""
import asyncio
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


async def purge_agent_memory(agent_id: str) -> None:
    """Purge an agent's vector memory without an existing cat instance."""
    ccat = await _boot_light_cat(agent_id)
    try:
        await ccat.destroy_memory()
    finally:
        await ccat.shutdown()


async def _teardown_memory_and_files(ccat, sources) -> None:
    """Purge vector memory, cascade per source, remove files, notify plugins.

    ``sources`` is the file list captured BEFORE the namespace wipe: the wipe
    deletes the stored file settings, so it must not be read from storage at
    this point.
    """
    # Lazy imports keep the startup import graph small and avoid cycles.
    from cat.routes.routes_utils import sanitize_source_name
    from cat.services.memory.models import VectorMemoryType

    await ccat.destroy_memory()

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

    ccat.file_manager.remove_folder(ccat.agent_key)

    # Plugin-owned namespaces get their final cleanup while the plugin manager
    # is still alive; fired even when no files were captured.
    await ccat.plugin_manager.execute_hook(
        "after_cheshire_cat_destroy",
        ccat.agent_key,
        caller=ccat,
    )


def _env_int(name: str, default: int) -> int:
    value = get_env_int(name)
    return default if value is None else value


async def complete_agent_teardown(agent_id: str, ccat=None) -> None:
    """Finish an agent teardown, resumably and idempotently.

    Serialized per agent by ``agent-delete:<id>``. Returns immediately when
    the marker is already gone (a previous run completed, or a duplicate call
    arrived late), so re-running on a half-torn-down agent is safe.

    ``ccat`` may be a live cat owned by the caller (it is left running) or
    ``None``, in which case a light cat is booted and shut down here.
    """
    quiesce_seconds = _env_int("CAT_INGESTION_DELETE_QUIESCE_SECONDS", _DEFAULT_QUIESCE_SECONDS)
    poll_seconds = _env_int("CAT_INGESTION_DELETE_POLL_SECONDS", _DEFAULT_POLL_SECONDS)
    lock_budget = quiesce_seconds + 120

    async with crud.distributed_lock(
        f"agent-delete:{agent_id}",
        timeout=lock_budget,
        blocking_timeout=lock_budget,
    ):
        if not await has_delete_marker(agent_id):
            return None

        booted = False
        if ccat is None:
            ccat = await _boot_light_cat(agent_id)
            booted = True

        try:
            # ORDERING FIX: the per-file cascade needs the file list, but the
            # namespace wipe deletes the stored file settings. Capture the
            # sources from the live cat BEFORE the wipe; read them afterwards
            # and they are gone. (MyCAT captured after the wipe and lost the
            # cascade on every fully-provisioned agent.)
            sources = [f.name for f in ccat.file_manager.list_files(ccat.agent_key)]

            await wait_for_ingestion_quiesce(agent_id, quiesce_seconds, poll_seconds)
            await wipe_agent_namespace_excluding_marker(agent_id)
            await _teardown_memory_and_files(ccat, sources)

            # Last: the marker means "teardown incomplete", so it can only go
            # away once memory and files are actually gone. A crash before
            # this line leaves a resumable marker.
            await clear_delete_marker(agent_id)
        finally:
            if booted:
                # Only a cat booted here is ours to shut down: the caller owns
                # a live ``ccat``.
                await ccat.shutdown()

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
