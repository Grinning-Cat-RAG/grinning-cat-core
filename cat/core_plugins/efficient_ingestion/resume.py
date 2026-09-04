"""Recovery loop for the ``efficient_ingestion`` plugin — ONE state machine.

The recovery/recovery loop of the ingestion lifecycle lives ONLY in this plugin:
the ``after_lizard_bootstrap`` hook schedules a fire-and-forget background pass
(per agent) that:

  (a) resumes stale ``uploaded``/``processing``/``error`` ingestions by handing
      each candidate to the SAME phase machine used by the re-embed pass
      (:func:`reembed_sources <cat.core_plugins.efficient_ingestion.reembed.reembed_sources>`);
      the machine re-reads the file from disk (or re-downloads the URL), claims
      the per-source work, records the phase, cleans up orphan images and
      performs the full re-ingest; and
  (b) purges status entries whose source is absent from the canonical lists
      (files on disk, URLs in the vector store, existing conversations) via the
      ``ingestion_status`` shared helper :func:`reconcile_agent`.

The same pass is re-run periodically (every
``CAT_INGESTION_RESUME_INTERVAL_SECONDS``, default 60) so a stale entry is
recovered without a manual restart; set the interval to ``0`` to disable the
periodic sweep.

The core and ``ingestion_status`` stay untouched by this loop: the core keeps
its upstream ingestion flow (plain ``ingest_file``, no state machine) and is
"replaced" only when this plugin drives the lifecycle (recovery, re-embed,
re-ingest all reuse the one machine).

Importing this module has zero side effects: the loop only starts from the
``after_lizard_bootstrap`` hook.
"""

import asyncio
import os
from io import BytesIO
from typing import Any

from cat import BillTheLizard, hook, log
from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
from cat.core_plugins.ingestion_status.reconcile import reconcile_agent
from cat.core_plugins.ingestion_status.registry import (
    IngestionStatus,
    list_statuses,
    set_status,
)
from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.env import get_env_int
from cat.services.memory.models import VectorMemoryType
from cat.utils import is_url


def _resume_enabled() -> bool:
    """``CAT_INGESTION_RESUME_ON_STARTUP`` flag, defaulting to true."""
    value = os.getenv("CAT_INGESTION_RESUME_ON_STARTUP")
    return value is None or value.lower() in ("1", "true", "yes", "on")


def _gc_enabled() -> bool:
    """``CAT_INGESTION_STATUS_GC_ON_STARTUP`` flag, defaulting to true."""
    value = os.getenv("CAT_INGESTION_STATUS_GC_ON_STARTUP")
    return value is None or value.lower() in ("1", "true", "yes", "on")


def _stale_seconds() -> int:
    """Staleness threshold for resume, defaulting to 300 seconds."""
    value = get_env_int("CAT_INGESTION_RESUME_STALE_SECONDS")
    return value if value is not None and value > 0 else 300


def _boot_stale_seconds() -> int:
    """Staleness threshold for the resume pass right after a container restart.

    A freshly restarted container inherits rows left in ``processing`` by a
    worker that died with the previous process: those must be recoverable
    quickly, not after the full ``_stale_seconds()`` (default 300s) budget.
    Any *live* worker (another replica still ingesting) keeps its rows fresh
    with the heartbeat (``CAT_INGESTION_HEARTBEAT_SECONDS``, default 30s), so
    requiring the row to be older than ``2 x heartbeat`` is enough to never
    steal live work while making dead-worker rows claimable at the first boot
    pass. Configurable with ``CAT_INGESTION_RESUME_BOOT_STALE_SECONDS``.
    """
    value = get_env_int("CAT_INGESTION_RESUME_BOOT_STALE_SECONDS")
    if value is not None and value > 0:
        return value
    heartbeat = get_env_int("CAT_INGESTION_HEARTBEAT_SECONDS")
    if heartbeat is not None and heartbeat > 0:
        return max(30, 2 * heartbeat)
    return 60


async def mark_file_missing(agent_id: str, scope: str, source: str) -> None:
    """Mark a source whose file is missing on disk as ``error``.

    A stale ``uploaded``/``processing`` entry whose source file no longer
    exists on disk cannot be resumed; instead of leaving it ``processing``
    forever, transition it to ``error`` so the teacher sees why and can
    re-upload (or remove the file to abandon it).
    """
    await set_status(
        agent_id,
        scope,
        source,
        type_="file",
        status=IngestionStatus.ERROR,
        chat_id=None if scope == "agent" else scope,
        error="Source file does not exist on disk; cannot resume. Remove the file to abandon it.",
    )


def _collection_for_scope(scope: str) -> VectorMemoryType:
    """Vector-memory collection for an ingestion scope ("agent" -> declarative, chat -> episodic)."""
    return VectorMemoryType.DECLARATIVE if scope == "agent" else VectorMemoryType.EPISODIC


def _source_from_entry(ccat, agent_id: str, scope: str, source: str, type_: str = "file"):
    """Build a ``StoredSourceWithMetadata`` for an in-flight entry, re-reading the file.

    Files are re-read from disk (persisted bytes wrapped in ``BytesIO``, the
    same shape ``get_stored_sources_with_metadata`` produces), so the phase
    machine performs the full re-ingest from the actual file content. URLs are
    passed through with ``content=None`` (the machine re-downloads them).
    Returns None (and logs) when the file is missing.
    """
    from cat.looking_glass.models import StoredSourceWithMetadata

    if (type_ or "") == "url" or is_url(source):
        return StoredSourceWithMetadata(
            name=source, path=source, content=None,
            metadata={"chat_id": scope} if scope != "agent" else {},
        )

    path = agent_id
    if scope != "agent":
        path = os.path.join(path, str(scope))
    file_bytes = ccat.file_manager.read_file(source, path)
    if file_bytes is None:
        log.warning(f"Ingestion resume: file {source} missing for agent {agent_id}; skipping")
        return None
    metadata = {}
    if scope != "agent":
        metadata["chat_id"] = scope
    return StoredSourceWithMetadata(
        name=source, path=path, content=BytesIO(file_bytes), metadata=metadata,
    )


async def _resume_agent(lizard: BillTheLizard, agent_id: str, ccat: Any = None) -> None:
    """Re-trigger stale ingestions for one agent through the ONE state machine.

    Only entries in ``uploaded``/``processing``/``error`` older than the stale
    threshold are candidates; each is handed to ``reembed_sources`` (the same
    phase machine used for re-embed): it decides the phase from the status doc,
    claims the per-source work atomically (so two workers never double-process),
    records the phase, cleans up any orphan images and performs the full re-ingest
    from disk (or URL re-download). Fresh entries and ``completed`` entries are
    left untouched.
    """
    if ccat is None:
        ccat = await lizard.get_cheshire_cat(agent_id)
    if ccat is None:
        return

    boot_stale = _boot_stale_seconds()
    entries = await list_statuses(agent_id)
    for entry in entries:
        status = entry.get("status")
        if status not in (
            IngestionStatus.UPLOADED.value,
            IngestionStatus.PROCESSING.value,
            IngestionStatus.ERROR.value,
        ):
            continue

        source = entry.get("source")
        scope = entry.get("scope")
        type_ = entry.get("type")
        if not source:
            continue

        # Re-read the file from disk (or pass the URL through); skip entries
        # whose file is gone on disk (they are marked error so the teacher can
        # re-upload or abandon).
        source_obj = _source_from_entry(ccat, agent_id, str(scope), source, type_)
        if source_obj is None:
            if type_ != "url" and not is_url(source):
                await mark_file_missing(agent_id, str(scope), source)
            continue

        log.info(
            f"Ingestion resume: handing {source} (status {status}, phase {entry.get('phase')!r}) "
            f"to the ingestion phase machine for {agent_id}"
        )
        # The phase machine claims the per-source work itself, using the boot
        # staleness so a freshly-crashed row is recoverable at restart.
        await reembed_sources(
            ccat, _collection_for_scope(str(scope)), [source_obj],
            stale_after=boot_stale,
        )


async def _pass_for_agent(lizard: BillTheLizard, agent_id: str) -> None:
    """Run the recovery + GC pass for one agent.

    Only the enumeration is guarded by a short agent-level distributed lock;
    actual (re)ingestion of each candidate happens under its own per-source
    lock (inside the phase machine's claim), so different sources of the same
    agent are processed concurrently while the same source can never be
    double-processed by another worker.
    """
    try:
        async with crud.distributed_lock(f"ingestion-sweep:{agent_id}", timeout=30, blocking_timeout=5):
            ccat = await lizard.get_cheshire_cat(agent_id)
            if ccat is None:
                return
            if _resume_enabled():
                await _resume_agent(lizard, agent_id, ccat=ccat)
            if _gc_enabled():
                await reconcile_agent(agent_id, ccat=ccat)
    except crud.LockError:
        # Another replica is already running this sweep: expected contention
        # across workers, not a fault. Log as info and move on.
        log.info(f"Ingestion sweep for agent {agent_id} skipped: lock held by another worker")
    except Exception as e:
        log.error(f"Ingestion startup pass failed for agent {agent_id}: {e}")


async def _startup_pass(lizard: BillTheLizard) -> None:
    """Enumerate agents and run the recovery + GC pass for each."""
    try:
        agent_ids = await crud_settings.get_agents_main_keys()
    except Exception as e:
        log.error(f"Ingestion startup pass: failed to enumerate agents: {e}")
        return
    for agent_id in agent_ids:
        await _pass_for_agent(lizard, agent_id)


async def _periodic_sweep_loop(lizard: BillTheLizard) -> None:
    """Re-run the recovery + GC pass every ``CAT_INGESTION_RESUME_INTERVAL_SECONDS``.

    Fire-and-forget: never blocks bootstrap. Each pass is safe against
    double-processing across replicas because ``_pass_for_agent`` is guarded by
    a per-agent distributed lock and each source claim by the phase machine.
    """
    interval = get_env_int("CAT_INGESTION_RESUME_INTERVAL_SECONDS")
    if interval is None or interval <= 0:
        return
    while True:
        await _startup_pass(lizard)
        await asyncio.sleep(interval)


@hook
async def after_lizard_bootstrap(lizard: BillTheLizard):
    """Schedule the fire-and-forget startup pass (never blocks bootstrap)."""
    asyncio.ensure_future(_startup_pass(lizard))
    interval = get_env_int("CAT_INGESTION_RESUME_INTERVAL_SECONDS")
    if interval is not None and interval > 0:
        asyncio.ensure_future(_periodic_sweep_loop(lizard))