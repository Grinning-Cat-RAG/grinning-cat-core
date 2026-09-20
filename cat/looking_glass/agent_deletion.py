"""Delete-marker protocol for the delete-while-ingest hardening.

The delete flow writes the persistent marker ``agents:{agent_id}:ingestion:delete``
(no TTL) at the START of the agent teardown, so in-flight ingestion workers
(EffING's ``ingestion_canceled``) see it and self-abort while the teardown is
still running. Invariant: marker present <=> teardown incomplete.

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
from cat.db.cruds.settings import get_agents_main_keys
from cat.db.database import DEFAULT_AGENTS_KEY
from cat.log import log

#: Non-terminal ingestion statuses (see ``IngestionStatus`` in the
#: ``ingestion_status`` plugin): while any of these is present the teardown
#: waits for the workers to self-abort (the marker is a broadcast, so the wait
#: is bounded assurance, not a hard gate).
_NON_TERMINAL_INGESTION_STATUSES = frozenset(
    {"uploaded", "downloading", "downloaded", "processing"}
)

#: Default quiesce/poll windows.
_DEFAULT_QUIESCE_SECONDS = 180
_DEFAULT_POLL_SECONDS = 5


def _delete_marker_key(agent_id: str) -> str:
    """Redis key of the deletion marker for an agent.

    Mirrors ``ingestion_status.registry.delete_marker_key``: the plugin owns
    the single source of truth, but the core needs the same literal and must
    not import the plugin. Keep in sync.
    """
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:ingestion:delete"


async def set_delete_marker(agent_id: str) -> None:
    """Write the deletion marker for an agent (idempotent).

    Called at the START of ``CheshireCat.destroy`` so in-flight ingestion
    workers (EffING's ``ingestion_canceled``) see the marker and self-abort
    while the teardown is still running. The marker is persistent (no TTL)
    and is removed only at teardown completion.
    """
    await crud.store(_delete_marker_key(agent_id), {"delete_marker": True})


async def has_delete_marker(agent_id: str) -> bool:
    """Whether the deletion marker is present (teardown incomplete).

    Read through the official ``cat.db.crud`` API (``crud.read(...) is not
    None``), never a raw client — EffING's ``ingestion_canceled`` relies on
    this.
    """
    return await crud.read(_delete_marker_key(agent_id)) is not None


async def clear_delete_marker(agent_id: str) -> None:
    """Remove the deletion marker (teardown completion).

    Removes ONLY the marker key — never any other key of the agent.
    """
    await crud.delete(_delete_marker_key(agent_id))


async def wait_for_ingestion_quiesce(
    agent_id: str,
    quiesce_seconds: int = _DEFAULT_QUIESCE_SECONDS,
    poll_seconds: int = _DEFAULT_POLL_SECONDS,
) -> bool:
    """Wait (bounded) for the agent's in-flight ingestion workers to finish.

    Scans ``agents:{agent_id}:ingestion:*`` through the crud-documented scan
    primitive ``cat.db.cruds.settings.get_agents_main_keys`` (the only
    non-destructive pattern scan ``cat.db.crud`` exposes; on upstream it
    returns the agent ids matching the pattern, so a non-empty result means at
    least one ingestion row exists). While any row exists it sleeps
    ``poll_seconds`` and retries, up to ``quiesce_seconds`` total. Then it
    returns regardless: the marker is a broadcast that makes workers
    self-abort, so this is only assurance, not a hard gate.

    Note: unlike MyCAT, the marker key itself cannot be excluded from the
    scan (upstream's scan primitive does not enumerate keys), so a marker
    alone keeps the wait running until the timeout. Bounded and safe either
    way.

    Args:
        agent_id: The agent (chatbot) id.
        quiesce_seconds: Total budget for the wait.
        poll_seconds: Interval between polls.

    Returns:
        True if the ingestion namespace settled (no rows), False on timeout.
    """
    deadline = time.monotonic() + quiesce_seconds
    while True:
        rows = await get_agents_main_keys(
            pattern=f"{DEFAULT_AGENTS_KEY}:{agent_id}:ingestion:*"
        )
        if not rows:
            return True
        if time.monotonic() >= deadline:
            log.warning(
                f"Agent {agent_id}: ingestion still active after {quiesce_seconds}s, "
                "proceeding with teardown (marker broadcast; workers self-abort)"
            )
            return False
        await asyncio.sleep(poll_seconds)