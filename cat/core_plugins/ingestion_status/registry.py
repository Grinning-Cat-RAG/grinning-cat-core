"""Redis-backed ingestion-status registry for the ``ingestion_status`` core plugin.

Key namespace (owned by this plugin; consumed by the destroy purge and the
re-embed flow): ``agents:{agent_id}:ingestion:{scope}:{sha256(source)}`` where
``scope`` is ``"agent"`` for agent-KB ingestion or a conversation ``chat_id``.

Every status transition is written as a Redis-JSON document via ``cat.db.crud``
(no raw Redis client is used here).
"""
import hashlib
from typing import Dict, List, Optional

from cat.db import crud
from cat.db.database import get_async_db
from cat.db.models import generate_timestamp
from cat.utils import Enum


class IngestionStatus(Enum):
    """Lifecycle states of a source ingestion.

    Files: ``uploaded -> processing -> completed | error``.
    URLs add ``downloading -> downloaded`` between ``uploaded`` and ``processing``.
    """
    UPLOADED = "uploaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


#: Work phases (diary of what the source is doing), stored in ``doc["phase"]``.
#: ``parsing_chunking`` and ``embedding`` are the recovery-relevant phases: a
#: crash while ``phase=embedding`` means chunks are intact and only the vectors
#: must be recomputed; ``phase=parsing_chunking`` means chunks must be
#: regenerated (full re-ingest). ``downloading`` applies to URLs.
PHASE_DOWNLOADING = "downloading"
PHASE_PARSING_CHUNKING = "parsing_chunking"
PHASE_EMBEDDING = "embedding"


def status_key(agent_id: str, scope: str, source: str) -> str:
    """Redis key for a source's ingestion status.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` for the agent KB, or a conversation ``chat_id``.
        source: The ingested file name or URL.

    Returns:
        ``agents:{agent_id}:ingestion:{scope}:{sha256(source)}``
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return f"agents:{agent_id}:ingestion:{scope}:{digest}"


async def _read_doc(key: str) -> Optional[Dict]:
    """Read a status doc, unwrapping the RedisJSON array wrapper."""
    value = await crud.read(key)
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


async def set_status(
    agent_id: str,
    scope: str,
    source: str,
    *,
    type_: str,
    status: IngestionStatus,
    chat_id: Optional[str] = None,
    error: Optional[str] = None,
    phase: str | None = None,
    embedder_name: str | None = None,
    chunker_name: str | None = None,
    clear_phase: bool = False,
    extra: Optional[Dict] = None,
) -> Dict:
    """Write (or update) the ingestion-status doc for a source.

    ``created_at`` is preserved across updates; ``updated_at`` is bumped on
    every write; ``error_at`` is set when ``status`` is ERROR.

    **Merge semantics (important)**: ``phase`` / ``embedder_name`` /
    ``chunker_name`` are updated ONLY when explicitly provided (not None).
    Existing callers that write status without these fields (heartbeat,
    ``after_rabbithole_stored_documents``, ``mark_file_missing``) therefore
    never clobber the engine-written phase/embedder/chunker of a row.
    ``clear_phase`` is the explicit way to drop the phase when the source
    reaches a terminal state (e.g. ``completed``).

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` for the agent KB, or a conversation ``chat_id``.
        source: The ingested file name or URL.
        type_: ``"file"`` or ``"url"``.
        status: The new lifecycle state.
        chat_id: The conversation id, when the ingestion is chat-scoped.
        error: The error message, when ``status`` is ERROR.
        phase: The current work phase (``downloading``, ``parsing_chunking``,
            ``embedding``) — see :func:`set_phase`. Updated only when provided.
        embedder_name: The embedder that produced the stored vectors. Updated
            only when provided.
        chunker_name: The chunker that produced the stored chunks. Updated
            only when provided.
        clear_phase: When True, drop the ``phase`` field from the doc.
        extra: Optional extra fields merged into the stored doc.

    Returns:
        The stored status document.
    """
    key = status_key(agent_id, scope, source)
    now = generate_timestamp()

    existing = await _read_doc(key)
    created_at = existing.get("created_at", now) if existing else now

    # Start from a copy of the existing doc so fields NOT provided in this
    # write (phase/embedder_name/chunker_name/extra) survive — the merge
    # semantics that lets a plain lifecycle write coexist with engine-written
    # metadata. Fields explicitly overwritten below always win.
    doc = dict(existing or {})
    doc.update({
        "source": source,
        "scope": scope,
        "chat_id": chat_id,
        "type": type_,
        "status": status,
        "error": error,
        "error_at": now if status == IngestionStatus.ERROR else None,
        "created_at": created_at,
        "updated_at": now,
    })
    # Merge semantics: only update phase/embedder/chunker when explicitly given
    if phase is not None:
        doc["phase"] = phase
    if embedder_name is not None:
        doc["embedder_name"] = embedder_name
    if chunker_name is not None:
        doc["chunker_name"] = chunker_name
    if clear_phase:
        doc.pop("phase", None)
    if extra:
        doc.update(extra)
    await crud.store(key, doc)
    return doc


async def set_phase(
    agent_id: str,
    scope: str,
    source: str,
    phase: str,
    *,
    embedder_name: str | None = None,
    chunker_name: str | None = None,
    type_: str = "file",
    chat_id: str | None = None,
) -> dict:
    """Transition a source into a work phase.

    Atomic ``status=PROCESSING`` + ``phase=<phase>`` write, bumping
    ``updated_at`` so the row is restartable on crash and never looks stale
    to another worker mid-phase. Records ``embedder_name`` (for the
    ``embedding`` phase) or ``chunker_name`` (for the ``parsing_chunking``
    phase) at the START of the phase that consumes them.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` or a conversation ``chat_id``.
        source: The ingested file name or URL.
        phase: ``downloading`` | ``parsing_chunking`` | ``embedding``.
        embedder_name: Recorded when provided (used at ``embedding`` start).
        chunker_name: Recorded when provided (used at ``parsing_chunking`` start).
        type_: ``"file"`` or ``"url"``.
        chat_id: The conversation id, when chat-scoped.

    Returns:
        The stored status document.
    """
    return await set_status(
        agent_id,
        scope,
        source,
        type_=type_,
        status=IngestionStatus.PROCESSING,
        chat_id=chat_id,
        phase=phase,
        embedder_name=embedder_name,
        chunker_name=chunker_name,
    )


async def get_status(agent_id: str, scope: str, source: str) -> Optional[Dict]:
    """Read the ingestion-status doc for a source, or None if absent."""
    return await _read_doc(status_key(agent_id, scope, source))


async def claim_source_for_resume(
    agent_id: str,
    scope: str,
    source: str,
    *,
    stale_after: float,
    owner: str,
    claim_completed: bool = False,
) -> Optional[Dict]:
    """Atomically claim a source for (re)ingestion under a per-source lock.

    Lock granularity is ``<agent>:<scope>:<sha256(source)>``, so many sources
    of the same agent can be (re)ingested concurrently by different workers,
    while two workers can never claim the SAME source at the same time.

    A source is claimable only when its current status allows a (re)start and
    its last update is older than ``stale_after`` (seconds): a fresh
    ``uploaded``/``processing`` row means another worker is already handling
    it, so this call returns None instead of double-ingesting.

    When ``claim_completed`` is True a ``completed`` row is ALSO claimable
    (regardless of staleness): used by the ingestion engine to re-embed a
    source whose embedder/chunker changed. The engine must re-check the
    mismatch itself before claiming. ``resume.py`` keeps the default False.

    On success the row is reset to PROCESSING with ``resume_owner`` /
    ``resume_at`` / bumped ``updated_at`` so other workers see it as taken.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` or a conversation ``chat_id``.
        source: The ingested file name or URL.
        stale_after: Minimum age of ``updated_at`` (seconds) for the entry to
            be claimable — protects in-flight work from being double-processed.
        owner: Identifier of the claiming worker (e.g. ``pid``).
        claim_completed: Whether a ``completed`` row may be claimed for
            re-embedding (engine passes True after its own mismatch check).

    Returns:
        The claimed (updated) status doc, or None when not claimable.
    """
    import time

    key = status_key(agent_id, scope, source)
    lock_pattern = f"ingestion-resume:{agent_id}:{scope}:{source}"
    async with crud.distributed_lock(lock_pattern, timeout=30, blocking_timeout=15):
        doc = await _read_doc(key)
        if doc is None:
            return None

        status = doc.get("status")
        if status == IngestionStatus.COMPLETED.value:
            # Completed rows are never re-claimed by the resume sweep; the
            # engine may claim them for re-embedding only when it already
            # verified a real embedder/chunker mismatch.
            if not claim_completed:
                return None
        elif status not in (
            IngestionStatus.UPLOADED.value,
            IngestionStatus.PROCESSING.value,
            IngestionStatus.ERROR.value,
        ):
            return None

        # Fresh in-flight work (uploaded/processing) is only claimable when
        # stale; completed rows (engine re-embed) have no in-flight counterpart,
        # so they bypass the staleness gate: a completed row is never "in flight".
        if status != IngestionStatus.COMPLETED.value:
            try:
                updated_at = float(doc.get("updated_at") or 0)
            except (TypeError, ValueError):
                updated_at = 0
            if time.time() - updated_at < stale_after:
                # fresh: another worker is handling it right now
                return None

        now = generate_timestamp()
        doc.update({
            "status": IngestionStatus.PROCESSING.value,
            "error": None,
            "error_at": None,
            "resume_owner": owner,
            "resume_at": now,
            "updated_at": now,
        })
        await crud.store(key, doc)
        return doc


async def release_resume_claim(agent_id: str, scope: str, source: str) -> None:
    """Clear the resume-owner markers after a re-ingestion attempt.

    The lifecycle hooks (``rabbithole_ingestion_start`` etc.) will later
    overwrite the row with the new state, so clearing is best-effort: it only
    removes the transient claim markers.
    """
    key = status_key(agent_id, scope, source)
    async with crud.distributed_lock(f"ingestion-resume:{agent_id}:{scope}:{source}", timeout=30, blocking_timeout=15):
        doc = await _read_doc(key)
        if doc is None:
            return
        doc.pop("resume_owner", None)
        doc.pop("resume_at", None)
        await crud.store(key, doc)


async def delete_status(agent_id: str, scope: str, source: str) -> None:
    """Delete the ingestion-status doc for a source."""
    await crud.delete(status_key(agent_id, scope, source))


async def list_statuses(agent_id: str, chat_id: Optional[str] = None) -> List[Dict]:
    """List the ingestion-status docs for an agent.

    With no ``chat_id`` only agent-scope entries are returned; with a
    ``chat_id`` only that conversation's entries are returned.
    """
    db = get_async_db()
    results: List[Dict] = []
    async for key in db.scan_iter(f"agents:{agent_id}:ingestion:*"):
        doc = await _read_doc(key)
        if not doc:
            continue
        scope = doc.get("scope")
        if chat_id is None:
            if scope != "agent":
                continue
        elif scope != chat_id:
            continue
        results.append(doc)
    return results


async def clear_agent(agent_id: str) -> int:
    """Delete every ingestion-status key for an agent (all scopes).

    Returns:
        The number of keys deleted.
    """
    return await crud.destroy(f"agents:{agent_id}:ingestion:*")


async def clear_chat(agent_id: str, chat_id: str) -> int:
    """Delete every ingestion-status key for one conversation scope.

    Returns:
        The number of keys deleted.
    """
    return await crud.destroy(f"agents:{agent_id}:ingestion:{chat_id}:*")