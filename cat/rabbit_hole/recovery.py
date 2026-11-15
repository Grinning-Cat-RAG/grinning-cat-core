"""The automatic resume of the ingestions: what a crash, a restart or a change of settings left behind.

Every pod runs a pass at the start and then every ``CAT_INGESTION_RESUME_INTERVAL_SECONDS`` (``<= 0`` disables the
resume: a single pass at the start could not take over the sources of the previous boot, whose locks are still alive).
For every agent not in maintenance (e.g. a change of the embedder, or of its vector database, running), holding the lock
of the agent (contended: another pod is passing, the agent is left to it), a source is brought up to date
(``RabbitHole.rerun_source``, without waiting for a live worker) when:

- it is in flight (uploaded, downloading, downloaded, processing) or being deleted: the lock of its source is free, so its
  worker is dead (a live worker holds it);
- it completed, but a tag of its diary is not the current one: the chunker, the embedder or the settings of a phase
  changed, or a phase was added;
- its diary names a phase no longer installed (a plugin was uninstalled): the source is re-ingested from scratch;
- it failed in a phase whose tag is not the current one anymore: the settings of that phase, or of a phase it depends on
  (a tag includes the tags of the phases it depends on), changed since, e.g. to fix the failure. The failed phase runs
  again, with the ones depending on it;
- it failed, but a phase was never tried while the phases it depends on completed (it was left to do by a failed change
  of the vector database, or added later): it runs once.

A change of the vector database whose pod died (the heartbeat of its record, ``crud_ingestions.get_vector_transfer``,
older than a lock timeout) is taken over as a failed copy, under a heartbeat of its own: every source is left to be
embedded again (and is, in the same pass), the procedures are embedded again, and
``after_vector_memory_transfer_on_agent`` runs with ``success=False``. While the heartbeat is alive the agent is left
alone.

Any other source in error is left alone (no endless retries of a permanent failure, e.g. paid LLM calls): POST
/ingestion/retry, a new upload, or the maintenance, runs it again.
"""
import asyncio
import copy
import time
from typing import Dict

from cat.db import crud
from cat.db.cruds import conversations as crud_conversations
from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds import settings as crud_settings
from cat.db.cruds.ingestions import IngestionStatus
from cat.env import get_env_int
from cat.log import log
from cat.rabbit_hole import chunks, engine
from cat.rabbit_hole.phases import compute_tags
from cat.services.memory.models import VectorMemoryType

IN_FLIGHT = {
    IngestionStatus.UPLOADED.value,
    IngestionStatus.DOWNLOADING.value,
    IngestionStatus.DOWNLOADED.value,
    IngestionStatus.PROCESSING.value,
    IngestionStatus.DELETING.value,
}


def _needs_attention(row: dict, tags: Dict[str, str], depends_on: Dict[str, tuple]) -> bool:
    if row.get("status") in IN_FLIGHT:
        return True
    diary = row.get("completed_phases") or {}
    if set(diary) - set(tags):
        return True  # its diary names a phase no longer installed: rerun_source prunes it and re-ingests the source
    if row.get("status") == IngestionStatus.ERROR.value:
        # the phase that failed itself, not the ones inheriting its failure: its tag includes the tags of the phases it
        # depends on, so a change upstream shows there too
        def failed_itself(phase_id: str) -> bool:
            return not any(
                (diary.get(dep) or {}).get("state") in (engine.STATE_ERROR, engine.STATE_SKIPPED)
                for dep in depends_on.get(phase_id, ())
            )

        # a phase never tried (e.g. left to do by a failed change of the vector database, or added later) whose
        # dependencies completed: work to do, not a failure to leave alone
        def to_do(phase_id: str) -> bool:
            deps = depends_on.get(phase_id, ())
            return phase_id not in diary and bool(deps) and all(
                (diary.get(dep) or {}).get("state") == engine.STATE_COMPLETE for dep in deps
            )

        return any(to_do(phase_id) for phase_id in tags) or any(
            entry.get("state") == engine.STATE_ERROR and phase_id in tags and entry.get("tag") != tags[phase_id]
            and failed_itself(phase_id)
            for phase_id, entry in diary.items()
        )
    # completed (every other state is in flight or failed)
    return any((diary.get(phase_id) or {}).get("tag") != tag for phase_id, tag in tags.items())


async def _purge_orphaned_chat_source(ccat, row: Dict) -> None:
    """Remove the points, the kept files and the row of a source whose chat no longer exists."""
    chat_id, name = row["chat_id"], row["source"]
    try:
        await ccat.vector_memory_handler.delete_tenant_points(
            str(VectorMemoryType.EPISODIC), {"source": name, "chat_id": chat_id},
        )
    except Exception as e:  # noqa: BLE001 - the row is kept so the next pass retries
        log.warning(f"Agent id: {ccat.agent_key}. Cannot remove the points of the orphaned chat source {name}: {e}")
        raise
    try:
        await asyncio.to_thread(chunks.delete_all, ccat.file_manager, ccat.agent_key, chat_id, name)
    except Exception as e:  # noqa: BLE001 - the points are gone; the row is kept so the next pass retries
        log.warning(f"Agent id: {ccat.agent_key}. Cannot remove the files of the orphaned chat source {name}: {e}")
        raise
    await crud_ingestions.delete_status(ccat.agent_key, chat_id, name)


async def resume_agent(lizard, agent_id: str) -> None:
    """One pass on the sources of an agent (see the module)."""
    if await crud_ingestions.is_in_maintenance(agent_id):
        return
    try:
        async with crud.distributed_lock(
            f"ingestion-resume:{agent_id}", timeout=engine.lock_timeout_seconds(), blocking_timeout=0.1,
        ):
            rows = await crud_ingestions.list_statuses(agent_id)
            transfer = await crud_ingestions.get_vector_transfer(agent_id)
            if (not rows and transfer is None) or (ccat := await lizard.get_cheshire_cat(agent_id)) is None:
                return
            if transfer is not None:
                if time.time() - transfer["beat"] <= engine.lock_timeout_seconds():
                    return  # the vector database is being changed (on any pod), even if its marker was cleared
                await ccat.take_over_dead_vector_transfer(transfer["since"])
                rows = await crud_ingestions.list_statuses(agent_id)
            try:
                purged = await crud_ingestions.reconcile_statuses(agent_id, ccat)
            except Exception as e:  # noqa: BLE001 - a reconcile failure must not stop the resume
                log.error(f"Agent id: {agent_id}. Reconcile of the ingestion states failed: {e}")
                purged = []
            if purged:
                rows = await crud_ingestions.list_statuses(agent_id)
            rabbit_hole = copy.copy(lizard.rabbit_hole)
            await rabbit_hole.setup(ccat)
            phases = await engine.collect_phases(rabbit_hole, ccat)
            tags = await compute_tags(phases, ccat)
            depends_on = {phase.id: phase.depends_on for phase in phases}
            for row in rows:
                if row.get("chat_id") is not None and not await crud_conversations.get_user_id_from_conversation_keys(
                    agent_id, row["chat_id"],
                ):
                    # the chat was destroyed but its source lingers (e.g. a worker died before the chat teardown): its
                    # artifacts and row go, without reviving a chat that no longer exists
                    try:
                        await _purge_orphaned_chat_source(ccat, row)
                        log.warning(
                            f"Agent id: {agent_id}. Removed the orphaned chat source {row['source']} "
                            f"({row['chat_id']})"
                        )
                    except Exception as e:  # noqa: BLE001 - the next pass retries
                        log.error(f"Agent id: {agent_id}. Cleanup of the orphaned source {row['source']} failed: {e}")
                    continue
                if not _needs_attention(row, tags, depends_on):
                    continue
                if await crud_ingestions.is_in_maintenance(agent_id):
                    return  # e.g. a change of the embedder started meanwhile: it brings the sources up to date
                try:
                    await lizard.rabbit_hole.rerun_source(ccat, row["source"], row["chat_id"], blocking=False)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - a failing source does not stop the others
                    log.error(f"Agent id: {agent_id}. Resume of {row['source']} failed: {e}")
    except engine.LockError:
        return  # another pod is passing on the agent


async def resume_all(lizard) -> None:
    """One pass on every agent; the failure of an agent does not stop the others."""
    for agent_id in await crud_settings.get_agents_main_keys():
        try:
            await resume_agent(lizard, agent_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - the resume goes on with the other agents
            log.error(f"Agent id: {agent_id}. Resume of the ingestions failed: {e}")


async def run(lizard) -> None:
    """The resume loop of the pod: a pass at the start, then one every interval (see the module)."""
    interval = get_env_int("CAT_INGESTION_RESUME_INTERVAL_SECONDS")
    if interval is None or interval <= 0:
        return
    while True:
        try:
            await resume_all(lizard)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - the next pass retries
            log.error(f"Resume of the ingestions failed: {e}")
        await asyncio.sleep(interval)
