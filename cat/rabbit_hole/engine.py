"""The dispatcher of the phases of a source: one worker at a time per source, with its state in the registry.

Every operation on a source (ingestion, run of the stale phases, deletion) holds the lock of the source, renewed by a
heartbeat that also refreshes the ``updated_at`` of the row: a row whose ``updated_at`` is older than two heartbeats
has no live worker, and can be taken over. The diary of a phase is written only when the phase ends.
"""
import asyncio
import contextlib
import contextvars
import os
import socket
import time
import uuid
from typing import Dict, List

from redis.exceptions import LockError, LockNotOwnedError

from cat.db import crud
from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds import settings as crud_settings
from cat.db.cruds.ingestions import PHASE_ID, IngestionStatus
from cat.db.database import get_async_db
from cat.env import get_env_int
from cat.log import log
from cat.services.memory.models import VectorMemoryType
from cat.rabbit_hole.phases import (
    PHASE_EMBEDDING,
    PHASE_PARSING_CHUNKING,
    EmbeddingPhase,
    IngestionPhase,
    IngestionSource,
    ParsingChunkingPhase,
    PhaseGraphError,
    PhaseResult,
    compute_tags,
    execution_order,
)

#: Identity of this process: the owner written in the rows it works on.
BOOT_ID = uuid.uuid4().hex
OWNER = {"pod": os.environ.get("HOSTNAME") or socket.gethostname(), "boot": BOOT_ID}

STATE_COMPLETE = "complete"
STATE_ERROR = "error"
STATE_SKIPPED = "skipped"


class IngestionCanceled(Exception):
    """The source is gone while it was processed (its agent, its chat or its row was deleted): what the run wrote is
    removed, nothing more is written."""


class LockLost(Exception):
    """The lock of the source expired (e.g. Redis restarted): another worker may own the source now, so this one stops
    at once, without writing or removing anything."""


#: Set by the heartbeat of the lock held in the running task when the lock is lost.
_lock_lost: contextvars.ContextVar[asyncio.Event | None] = contextvars.ContextVar("ingestion_lock_lost", default=None)


def heartbeat_seconds() -> int:
    return max(1, get_env_int("CAT_INGESTION_HEARTBEAT_SECONDS") or 30)


def lock_timeout_seconds() -> int:
    return max(heartbeat_seconds() * 2, get_env_int("CAT_INGESTION_LOCK_TIMEOUT_SECONDS") or 300)


def not_ready_budget_seconds() -> int:
    value = get_env_int("CAT_INGESTION_NOT_READY_MAX_SECONDS")
    return 3600 if value is None else value


def lock_name(source: IngestionSource) -> str:
    return f"ingestion-source:{crud_ingestions.format_key(source.agent_id, source.chat_id, source.name)}"


@contextlib.asynccontextmanager
async def source_lock(source: IngestionSource, blocking: bool = True, wait: float | None = None):
    """Hold the lock of the source, with a heartbeat renewing it and refreshing the row while the body runs.

    It waits for the lock as long as needed, or ``wait`` seconds; ``blocking=False`` does not wait. ``LockError`` when
    another worker still holds it.
    """
    timeout = lock_timeout_seconds()
    blocking_timeout = wait if wait is not None else (None if blocking else 0.1)
    async with crud.distributed_lock(lock_name(source), timeout=timeout, blocking_timeout=blocking_timeout) as lock:
        stop = asyncio.Event()
        lost = asyncio.Event()
        token = _lock_lost.set(lost)

        async def beat():
            while not stop.is_set():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=heartbeat_seconds())
                if stop.is_set():
                    return
                try:
                    await lock.extend(timeout, replace_ttl=True)
                except LockNotOwnedError as e:
                    log.warning(f"Heartbeat of {source.name}: the lock is lost ({e}), the worker stops")
                    lost.set()
                    return
                except Exception as e:  # noqa: BLE001 - a missed beat is retried at the next one
                    log.warning(f"Heartbeat of {source.name} failed: {e}")
                    continue
                try:
                    await crud_ingestions.touch_status(source.agent_id, source.chat_id, source.name)
                except Exception as e:  # noqa: BLE001 - a missed beat is retried at the next one
                    log.warning(f"Heartbeat of {source.name} failed: {e}")

        task = asyncio.create_task(beat())
        try:
            yield lock
        finally:
            stop.set()
            await task
            _lock_lost.reset(token)


async def agent_exists(agent_id: str) -> bool:
    return bool(await get_async_db().exists(crud_settings.format_key(agent_id)))


async def collect_phases(rabbit_hole, cat) -> List[IngestionPhase]:
    """The built-in phases and the ones of the plugins of the agent, in execution order.

    The plugins return ``IngestionPhase`` subclasses (the hooks deep-copy what they pass along: a class is passed as
    it is), instantiated here without arguments. A class that is not an ``IngestionPhase`` subclass, cannot be
    instantiated, has an invalid id or an id already taken is left out."""
    phases: List[IngestionPhase] = [ParsingChunkingPhase(rabbit_hole), EmbeddingPhase(rabbit_hole)]
    extra = await cat.plugin_manager.execute_hook("rabbithole_ingestion_phases", [], caller=cat) or []
    for phase_class in extra:
        if not (isinstance(phase_class, type) and issubclass(phase_class, IngestionPhase)):
            log.warning(f"Ingestion phase ignored, not an IngestionPhase subclass: {phase_class!r}")
            continue
        try:
            phase = phase_class()
        except Exception as e:  # noqa: BLE001 - a broken plugin phase must not stop the ingestion
            log.warning(f"Ingestion phase ignored, {phase_class.__name__} cannot be instantiated: {e}")
            continue
        if not isinstance(phase.id, str) or not PHASE_ID.match(phase.id):
            log.warning(f"Ingestion phase ignored, invalid id {phase.id!r}: lowercase letters, digits and underscores")
            continue
        if any(phase.id == known.id for known in phases):
            log.warning(f"Ingestion phase ignored, id already taken: {phase.id}")
            continue
        phases.append(phase)
    return execution_order(phases)


def _entry(state: str, tag: str | None, result: PhaseResult | None = None, reason: str | None = None) -> Dict:
    entry = {"state": state, "tag": tag, "reason": reason, "finished_at": round(time.time(), 3)}
    if result is not None:
        entry.update({"reason": result.reason or reason, "processed": result.processed, "expected": result.expected})
        if result.warnings:
            entry["warnings"] = result.warnings
    return entry


async def _run_phase(phase: IngestionPhase, source: IngestionSource, cat) -> PhaseResult:
    """Run the phase until it is done, skipped or failed: ``not_ready`` is retried within the budget of the phase."""
    budget = phase.not_ready_budget_seconds if phase.not_ready_budget_seconds is not None else not_ready_budget_seconds()
    started = time.monotonic()
    while True:
        try:
            result = await phase.process(source, cat)
        except Exception as e:  # noqa: BLE001 - a failing phase is an error state, never a crash of the worker
            log.error(f"Agent id: {source.agent_id}. Phase {phase.id} failed on {source.name}: {e}")
            return PhaseResult(status="error", reason=str(e) or type(e).__name__)
        if not isinstance(result, PhaseResult):
            return PhaseResult(status="error", reason=f"The phase {phase.id} returned no result")
        if result.status != "not_ready":
            return result
        retry_after = result.retry_after if result.retry_after is not None and result.retry_after >= 0 else 5
        if time.monotonic() - started + retry_after > budget:
            return PhaseResult(status="error", reason=f"Not ready after {budget}s: {result.reason or 'no reason'}")
        await asyncio.sleep(retry_after)


async def _stored_points(cat, source: IngestionSource) -> int | None:
    """The points stored for the source (``None`` when the store cannot be read: never force a re-processing on it)."""
    collection = str(VectorMemoryType.DECLARATIVE if source.chat_id is None else VectorMemoryType.EPISODIC)
    metadata = {"source": source.name}
    if source.chat_id is not None:
        metadata["chat_id"] = source.chat_id
    try:
        points, _ = await cat.vector_memory_handler.get_all_tenant_points(
            collection, metadata=metadata, with_vectors=False,
        )
    except Exception as e:  # noqa: BLE001 - a store failure is not proof that the points are gone
        log.warning(f"Agent id: {source.agent_id}. Cannot check the stored points of {source.name}: {e}")
        return None
    return len(points)


async def store_grounded_freshness(ordered, tags, diary, cat, source: IngestionSource) -> bool:
    """Ground the freshness of a source in the store: a fresh diary over zero stored points is stale.

    When every phase is up to date (no work) and the two producing phases completed, but the source has no stored
    point, their entries are dropped so ``parsing_chunking`` and ``embedding`` run again. Returns whether it forced
    them.
    """
    if any(_decide(phase, tags, diary, set()) is not None for phase in ordered):
        return False
    producing = (PHASE_PARSING_CHUNKING, PHASE_EMBEDDING)
    if any((diary.get(phase_id) or {}).get("state") != STATE_COMPLETE for phase_id in producing):
        return False
    stored = await _stored_points(cat, source)
    if stored == 0:
        log.warning(
            f"Agent id: {source.agent_id}. The source {source.name} is up to date but has no stored points: "
            f"processing it again"
        )
        for phase_id in producing:
            diary.pop(phase_id, None)
        return True
    return False


async def run_phases(rabbit_hole, cat, source: IngestionSource) -> Dict | None:
    """Run the stale phases of the source (the lock of the source must be held) and write its final state.

    Returns the final row, or None when the row does not exist anymore (the source was deleted meanwhile).
    """
    agent, chat, name = source.agent_id, source.chat_id, source.name

    async def check() -> None:
        lost = _lock_lost.get()
        if lost is not None and lost.is_set():
            raise LockLost(name)
        if not await agent_exists(agent):
            raise IngestionCanceled(agent)

    async def write(**fields) -> None:
        await check()
        if not await crud_ingestions.update_status(agent, chat, name, **fields):
            raise IngestionCanceled(name)  # the row is gone: the chat, or the source, was deleted

    async def record(phase_id: str, entry: Dict) -> None:
        await check()
        if not await crud_ingestions.record_phase(agent, chat, name, phase_id, entry):
            raise IngestionCanceled(name)

    row = await crud_ingestions.get_status(agent, chat, name)
    if row is None:
        return None
    if row["status"] == IngestionStatus.DELETING.value:
        return row  # an interrupted deletion: only a deletion resumes it, the source is never brought back
    try:
        ordered = await collect_phases(rabbit_hole, cat)
        tags = await compute_tags(ordered, cat)
    except PhaseGraphError as e:
        await write(status=IngestionStatus.ERROR, phase=None, error=str(e), owner=None)
        return await crud_ingestions.get_status(agent, chat, name)

    diary: Dict[str, Dict] = dict(row.get("completed_phases") or {})
    ran: set = set()
    terminal = row["status"] in (IngestionStatus.COMPLETED.value, IngestionStatus.ERROR.value)
    if terminal and not _has_work(ordered, tags, diary):
        # the diary may be fresh while the store lost the source (e.g. its collection was recreated): the producing
        # phases are forced again, so a source never looks up to date without its points
        await store_grounded_freshness(ordered, tags, diary, cat, source)
        if not _has_work(ordered, tags, diary):
            return row  # nothing changed: nothing is written (the periodic checks run on every finished source)
    try:
        await _run_ordered(ordered, tags, diary, ran, source, cat, write, record)
        # the error of the source is the one of its first failed phase in execution order: the cause, not the
        # dependents failing after it
        errors = [diary[p.id] for p in ordered if diary.get(p.id, {}).get("state") == STATE_ERROR]
        if errors:
            await write(status=IngestionStatus.ERROR, phase=None, owner=None, error=errors[0].get("reason"))
        else:
            await write(status=IngestionStatus.COMPLETED, phase=None, owner=None, error=None)
    except IngestionCanceled:
        # the source is gone meanwhile (its agent, its chat or its row deleted): everything its phases produced goes,
        # of this run and of the previous ones, as in a deletion (the deletions are idempotent: its folder and points
        # may be gone already)
        for phase in reversed(ordered):
            try:
                await phase.delete(source, cat)
            except Exception as e:  # noqa: BLE001 - best effort, the agent is gone
                log.warning(f"Cleanup of {phase.id} for {name} failed: {e}")
        raise
    return await crud_ingestions.get_status(agent, chat, name)


def _decide(phase: IngestionPhase, tags: Dict[str, str], diary: Dict[str, Dict], ran: set) -> str | None:
    """What a phase needs: ``"error"``/``"skipped"`` (a dependency failed or was skipped, and its entry does not say so
    yet), ``"run"`` (stale), or None (nothing)."""
    states = [diary.get(dep, {}).get("state") for dep in phase.depends_on]
    entry = diary.get(phase.id)
    if STATE_ERROR in states or STATE_SKIPPED in states:
        state = STATE_ERROR if STATE_ERROR in states else STATE_SKIPPED
        current = entry is not None and entry.get("state") == state and entry.get("tag") == tags[phase.id]
        return None if current else state
    if entry is None or entry.get("tag") != tags[phase.id] or any(dep in ran for dep in phase.depends_on):
        return "run"
    return None


def _has_work(ordered: List[IngestionPhase], tags: Dict[str, str], diary: Dict[str, Dict]) -> bool:
    return any(_decide(phase, tags, diary, set()) is not None for phase in ordered)


async def _run_ordered(ordered, tags, diary, ran, source, cat, write, record) -> None:
    """Run the stale phases in order, recording each one when it ends (``diary`` and ``ran`` are updated)."""
    await write(status=IngestionStatus.PROCESSING, owner=OWNER, error=None)
    for phase in ordered:
        decision = _decide(phase, tags, diary, ran)
        if decision is None:
            continue
        if decision != "run":
            cause = next(dep for dep in phase.depends_on if diary.get(dep, {}).get("state") == decision)
            reason = f"Depends on the {'failed' if decision == STATE_ERROR else 'skipped'} phase {cause}"
            diary[phase.id] = _entry(decision, tags[phase.id], reason=reason)
            await record(phase.id, diary[phase.id])
            ran.add(phase.id)
            continue

        await write(phase=phase.id)
        applies, reason = await phase.applies_to(source, cat)
        if not applies:
            result = PhaseResult(status="skipped", reason=reason)
        else:
            result = await _run_phase(phase, source, cat)
        state = {"done": STATE_COMPLETE, "skipped": STATE_SKIPPED}.get(result.status, STATE_ERROR)
        ran.add(phase.id)  # before recording: a cancellation while recording still cleans it up
        diary[phase.id] = _entry(state, tags[phase.id], result)
        await record(phase.id, diary[phase.id])


async def delete_source(rabbit_hole, cat, source: IngestionSource) -> None:
    """Remove the source from the memory of the agent: what every phase produced (in reverse order), the files kept by
    the RabbitHole and the row. An interrupted deletion resumes from the phases not yet removed (the row stays
    ``deleting`` meanwhile). The lock of the source must be held."""
    agent, chat, name = source.agent_id, source.chat_id, source.name
    row = await crud_ingestions.get_status(agent, chat, name)
    deleted = list(row.get("deleted_phases") or []) if row else []
    if row is not None:
        await crud_ingestions.update_status(agent, chat, name, status=IngestionStatus.DELETING, phase=None)
    try:
        ordered = await collect_phases(rabbit_hole, cat)
    except PhaseGraphError:
        ordered = execution_order([ParsingChunkingPhase(rabbit_hole), EmbeddingPhase(rabbit_hole)])
    for phase in reversed(ordered):
        if phase.id in deleted:
            continue
        await phase.delete(source, cat)  # a failure leaves the row deleting, resumable
        deleted.append(phase.id)
        if row is not None:
            await crud_ingestions.update_status(agent, chat, name, deleted_phases=deleted)
    await crud_ingestions.delete_status(agent, chat, name)


__all__ = [
    "BOOT_ID", "OWNER", "IngestionCanceled", "LockError", "LockLost", "agent_exists", "collect_phases",
    "delete_source",
    "heartbeat_seconds", "lock_timeout_seconds", "run_phases", "source_lock", "store_grounded_freshness",
]
