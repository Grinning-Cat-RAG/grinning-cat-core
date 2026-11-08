"""The automatic resume of the ingestions: what a crash, a restart or a change of settings left behind.

Every pod runs a pass at the start and then every ``CAT_INGESTION_RESUME_INTERVAL_SECONDS`` (``<= 0`` disables the
resume: a single pass at the start could not take over the sources of the previous boot, whose locks are still alive).
For every agent not in maintenance (e.g. a change of the embedder running), holding the lock of the agent (contended:
another pod is passing, the agent is left to it), a source is brought up to date (``RabbitHole.rerun_source``, without
waiting for a live worker) when:

- it is in flight (uploaded, downloading, downloaded, processing) or being deleted: the lock of its source is free, so its
  worker is dead (a live worker holds it);
- it completed, but a tag of its diary is not the current one: the chunker, the embedder or the settings of a phase
  changed, or a phase was added;
- it failed in a phase whose tag is not the current one anymore: the settings of that phase, or of a phase it depends on
  (a tag includes the tags of the phases it depends on), changed since, e.g. to fix the failure. The failed phase runs
  again, with the ones depending on it.

Any other source in error is left alone (no endless retries of a permanent failure, e.g. paid LLM calls): POST
/ingestion/retry, a new upload, or the maintenance, runs it again.
"""
import asyncio
import copy
from typing import Dict

from cat.db import crud
from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds import settings as crud_settings
from cat.db.cruds.ingestions import IngestionStatus
from cat.env import get_env_int
from cat.log import log
from cat.rabbit_hole import engine
from cat.rabbit_hole.phases import compute_tags

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
    if row.get("status") == IngestionStatus.ERROR.value:
        # the phase that failed itself, not the ones inheriting its failure: its tag includes the tags of the phases it
        # depends on, so a change upstream shows there too
        def failed_itself(phase_id: str) -> bool:
            return not any(
                (diary.get(dep) or {}).get("state") in (engine.STATE_ERROR, engine.STATE_SKIPPED)
                for dep in depends_on.get(phase_id, ())
            )

        return any(
            entry.get("state") == engine.STATE_ERROR and phase_id in tags and entry.get("tag") != tags[phase_id]
            and failed_itself(phase_id)
            for phase_id, entry in diary.items()
        )
    # completed (every other state is in flight or failed)
    return any((diary.get(phase_id) or {}).get("tag") != tag for phase_id, tag in tags.items())


async def resume_agent(lizard, agent_id: str) -> None:
    """One pass on the sources of an agent (see the module)."""
    if await crud_ingestions.is_in_maintenance(agent_id):
        return
    try:
        async with crud.distributed_lock(
            f"ingestion-resume:{agent_id}", timeout=engine.lock_timeout_seconds(), blocking_timeout=0.1,
        ):
            rows = await crud_ingestions.list_statuses(agent_id)
            if not rows or (ccat := await lizard.get_cheshire_cat(agent_id)) is None:
                return
            rabbit_hole = copy.copy(lizard.rabbit_hole)
            await rabbit_hole.setup(ccat)
            phases = await engine.collect_phases(rabbit_hole, ccat)
            tags = await compute_tags(phases, ccat)
            depends_on = {phase.id: phase.depends_on for phase in phases}
            for row in rows:
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
