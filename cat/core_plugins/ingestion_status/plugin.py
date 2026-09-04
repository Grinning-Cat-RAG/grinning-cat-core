"""Ingestion-status lifecycle hooks.

Observes the RabbitHole ingestion pipeline and writes per-source status docs
into Redis via the plugin's registry module.

Lifecycle written by these hooks:
    files:  uploaded -> processing -> completed | error
    urls:   uploaded -> downloading -> downloaded -> processing -> completed | error

Phase diary (``doc["phase"]``), set at the START of each phase so a crash
leaves the in-flight phase and the resume re-runs exactly that phase:
    downloading -> parsing_chunking -> embedding -> (completed clears phase)
"""
import asyncio

from cat import hook
from cat.core_plugins.ingestion_status.registry import (
    PHASE_DOWNLOADING,
    PHASE_PARSING_CHUNKING,
    IngestionStatus,
    clear_agent,
    delete_status,
    get_status,
    set_status,
)


def _scope_and_chat(cat):
    """Resolve ``(scope, chat_id)`` from the hook caller.

    A StrayCat carries a conversation ``id``; a CheshireCat does not, so the
    scope is the agent KB. Mirrors ``chat_id = self.stray.id if self.stray else None``.
    """
    if hasattr(cat, "id"):
        return cat.id, cat.id
    return "agent", None


def _source_type(source: str, is_url: bool = False) -> str:
    if is_url or source.startswith("http"):
        return "url"
    return "file"


def _chunker_name(cat) -> str | None:
    """Best-effort name of the active chunker, read from the caller.

    The chunker is resolved lazily by the core (``cat.chunker`` on
    BotMixin); a failure here must never break a status write.
    """
    try:
        chunker = getattr(cat, "chunker", None)
        if chunker is None:
            return None
        name = getattr(chunker, "name", None)
        return str(name) if name else None
    except Exception:  # noqa: BLE001 - status writes must never fail
        return None


_heartbeat_tasks: dict = {}


def _heartbeat_key(agent_id: str, scope: str, source: str) -> str:
    return f"{agent_id}:{scope}:{source}"


async def _cancel_heartbeat(agent_id: str, scope: str, source: str) -> None:
    task = _heartbeat_tasks.pop(_heartbeat_key(agent_id, scope, source), None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _heartbeat_status(agent_id: str, scope: str, source: str, interval: float) -> None:
    """Periodically bump ``updated_at`` of a PROCESSING row while it is being handled.

    A long parse/embed can otherwise make the row look stale and get re-claimed
    by another worker (see ``claim_source_for_resume``). Stops as soon as the
    row leaves the PROCESSING state.
    """
    while True:
        await asyncio.sleep(interval)
        current = await get_status(agent_id, scope, source)
        if current and current.get("status") == IngestionStatus.PROCESSING.value:
            await set_status(
                agent_id,
                scope,
                source,
                type_=current.get("type", "file"),
                status=IngestionStatus.PROCESSING,
                chat_id=current.get("chat_id"),
            )
        else:
            return


@hook(priority=0)
async def rabbithole_ingestion_start(source, metadata, is_url, cat) -> None:
    """Record that an ingestion is about to begin (source known, nothing stored yet)."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source, is_url),
        status=IngestionStatus.UPLOADED,
        chat_id=chat_id,
    )


@hook(priority=0)
async def rabbithole_url_downloading(url, filename, cat) -> None:
    """Record that a URL download is about to start."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        url,
        type_="url",
        status=IngestionStatus.DOWNLOADING,
        chat_id=chat_id,
        phase=PHASE_DOWNLOADING,
    )


@hook(priority=0)
async def rabbithole_url_download_completed(url, filename, cat) -> None:
    """Record that a URL download completed successfully."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        url,
        type_="url",
        status=IngestionStatus.DOWNLOADED,
        chat_id=chat_id,
        phase=PHASE_DOWNLOADING,
    )


@hook(priority=0)
async def rabbithole_ingestion_processing(source, cat) -> None:
    """Record that the source is being parsed, chunked and embedded."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source),
        status=IngestionStatus.PROCESSING,
        chat_id=chat_id,
        phase=PHASE_PARSING_CHUNKING,
        chunker_name=_chunker_name(cat),
    )


@hook(priority=0)
async def after_rabbithole_stored_documents(source, stored_points, cat) -> None:
    """Record that the source was stored successfully.

    The hook fires in the ``finally`` block of ``ingest_file``, so it also runs
    on the error path: never overwrite an already-recorded ERROR state, and
    ignore the unresolved empty source. The phase diary is cleared on success:
    a ``completed`` row proves all phases finished, so no phase is pending.
    """
    if not source:
        return
    scope, chat_id = _scope_and_chat(cat)
    current = await get_status(cat.agent_key, scope, source)
    if current and current.get("status") == IngestionStatus.ERROR.value:
        return
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source),
        status=IngestionStatus.COMPLETED,
        chat_id=chat_id,
        clear_phase=True,
    )


@hook(priority=0)
async def rabbithole_ingestion_error(source, error, cat) -> None:
    """Record that the ingestion failed, with the error message."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source),
        status=IngestionStatus.ERROR,
        chat_id=chat_id,
        error=str(error),
    )

@hook(priority=0)
def rabbithole_processing_heartbeat_start(source: str, scope: str, interval: float, cat) -> None:
    """Spawn the heartbeat task that keeps the PROCESSING row fresh."""
    if cat is None:
        return
    agent_id = getattr(cat, "agent_key", None)
    if not agent_id:
        return
    key = _heartbeat_key(agent_id, scope, source)
    # safety: stop any pre-existing heartbeat for the same row
    if key in _heartbeat_tasks:
        _heartbeat_tasks[key].cancel()
    _heartbeat_tasks[key] = asyncio.ensure_future(
        _heartbeat_status(agent_id, scope, source, interval)
    )


@hook(priority=0)
async def rabbithole_processing_heartbeat_stop(source: str, scope: str, cat) -> None:
    """Cancel the heartbeat task spawned on processing start."""
    if cat is None:
        return
    agent_id = getattr(cat, "agent_key", None)
    if not agent_id:
        return
    await _cancel_heartbeat(agent_id, scope, source)


@hook(priority=0)
async def after_file_manager_file_deleted(filename: str, scope: str, cat) -> None:
    """Drop the per-source status row when the file is deleted."""
    if cat is None:
        return
    agent_id = getattr(cat, "agent_key", None)
    if not agent_id:
        return
    await delete_status(agent_id, scope, filename)


@hook(priority=0)
async def after_cheshire_cat_destroy(agent_id: str, cat) -> None:
    """Drop every ingestion-status row of a destroyed agent.

    Fired by ``CheshireCat.destroy()`` once the agent's resources are gone;
    the whole ``agents:<agent_id>:ingestion:*`` namespace belongs to this
    plugin, so the cleanup lives here (the core only fires the hook).
    """
    await clear_agent(agent_id)
