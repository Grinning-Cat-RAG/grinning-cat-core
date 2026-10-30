"""Final-wave findings (B2/N1/N3/N4) for the efficient-ingestion slice.

Covers:

- the chunker phase fingerprint through the REAL ``PUT /chunking/settings``
  route (agent-scoped, like the route writes) plus a real engine re-run whose
  point count changes;
- the ``downloading`` lease/heartbeat that keeps an active download from being
  stolen and still lets a stale one be reclaimed;
- ``crud.store(nx=True)`` reporting a lost create (N1);
- the time-bounded ``resume_created`` lease when the claim gate is disabled (N3);
- the batch/recovery heartbeat around ``reembed_sources`` (N4).

Redis keys live under ``agents:effingtest_<uuid>:`` and are destroyed in
``finally``. The host ``tests/conftest.py`` forces ``CAT_REDIS_DB=1``.
"""
import asyncio
import time
import uuid
from io import BytesIO
from types import SimpleNamespace

import pytest

from cat.core_plugins.cat_efficient_ingestion import plugin as ingestion_plugin
from cat.core_plugins.cat_efficient_ingestion import reembed
from cat.core_plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    claim_source_for_resume,
    get_status,
    set_status,
    status_key,
)
from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.db.models import Setting
from cat.services.memory.models import VectorMemoryType


def _agent_id() -> str:
    return f"effingtest_{uuid.uuid4().hex}"


async def _seed_master_key(agent_id: str) -> None:
    await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])


async def _drop(agent_id: str) -> None:
    await crud.destroy(f"agents:{agent_id}:*")


async def _age_doc(agent_id: str, scope: str, source: str, *, age: float, **fields) -> None:
    """Rewrite a status row's ``updated_at`` (and extra fields) to a chosen age."""
    key = status_key(agent_id, scope, source)
    doc = await crud.read(key)
    if isinstance(doc, list):
        doc = doc[0]
    doc = dict(doc)
    doc["updated_at"] = time.time() - age
    doc.update(fields)
    await crud.store(key, doc)


# ---------------------------------------------------------------------------
# B2 — chunker change through the REAL route re-runs parsing (point count)
# ---------------------------------------------------------------------------

_LONG_TEXT = ("The quick brown fox jumps over the lazy dog. " * 120).strip()


async def test_chunker_change_through_the_real_route_reruns_parsing(
    secure_client, secure_client_headers, cheshire_cat, lizard, monkeypatch
):
    """Change the chunker via ``PUT /chunking/settings`` and re-run the engine.

    The route writes the agent-scoped chunker; the phase marker must read that
    same scope, otherwise the rewrite is invisible and the source is skipped.
    The engine really re-parses (the real chunker splits the real text) and the
    stored point count changes.
    """
    # Do not let the route fire the engine in the background: the test drives it
    # explicitly below.
    monkeypatch.setattr("cat.routes.chunker.run_background_task", lambda *a, **k: None)

    # The default file manager is the no-op Dummy one; a real LocalFileManager is
    # needed for the engine's re-embed pass to find the stored source on disk.
    await crud_settings.upsert_setting_by_category(
        cheshire_cat.agent_key,
        Setting(name="LocalFileManagerConfig", value={}, category="file_manager"),
    )
    ccat = await lizard.get_cheshire_cat(cheshire_cat.agent_key)

    engine = reembed.EfficientIngestionEngine()
    source = "final-wave-chunker.txt"

    async def point_count() -> int:
        points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
            str(VectorMemoryType.DECLARATIVE), with_vectors=False
        )
        return sum(
            1
            for p in points
            if (p.payload or {}).get("metadata", {}).get("source") == source
        )

    async def put_chunk_size(size: int) -> None:
        res = await secure_client.put(
            "/chunking/settings/RecursiveTextChunkerSettings",
            headers=secure_client_headers,
            json={"encoding_name": "cl100k_base", "chunk_size": size, "chunk_overlap": 0},
        )
        assert res.status_code == 200, res.text
        current = await crud_settings.get_settings_by_category(ccat.agent_key, "chunker")
        assert current["value"]["chunk_size"] == size

    await put_chunk_size(32)
    # Re-resolve the cat: its services are snapshotted at creation, so the
    # chunker must be re-read after the route changed the stored config.
    ccat = await lizard.get_cheshire_cat(cheshire_cat.agent_key)
    assert ccat.chunker._chunk_size == 32
    await engine.ingest_file(ccat, BytesIO(_LONG_TEXT.encode()), source, content_type="text/plain")
    first = await point_count()
    assert first > 1, f"expected several chunks at chunk_size=32, got {first}"

    doc = await get_status(ccat.agent_key, "agent", source)
    assert doc is not None and doc.get("status") == IngestionStatus.COMPLETED.value
    first_marker = doc["completed_phases"]["parsing_chunking"]["marker"]
    assert first_marker is not None

    # Real route change -> parsing phase must be re-run by the engine.
    await put_chunk_size(128)
    assert await engine.run(lizard) is True

    second = await point_count()
    assert second < first, (
        f"a larger chunk_size must produce fewer points ({second} !< {first}); "
        "the chunker change was not detected through the real route"
    )

    doc = await get_status(ccat.agent_key, "agent", source)
    assert doc.get("status") == IngestionStatus.COMPLETED.value
    assert doc["completed_phases"]["parsing_chunking"]["marker"] != first_marker


# ---------------------------------------------------------------------------
# B2 — active download lease/heartbeat
# ---------------------------------------------------------------------------


async def test_active_download_heartbeat_prevents_stealing_but_stale_row_is_claimed():
    """A live download writer keeps the row fresh; a dead one ages out."""
    agent = _agent_id()
    source = "https://example.com/big.pdf"
    await _seed_master_key(agent)
    try:
        await set_status(agent, "agent", source, type_="url", status=IngestionStatus.DOWNLOADING)
        await _age_doc(agent, "agent", source, age=10_000)
        stale_before = (await get_status(agent, "agent", source))["updated_at"]

        task = asyncio.ensure_future(
            ingestion_plugin._heartbeat_status(agent, "agent", source, 0.01)
        )
        try:
            await asyncio.sleep(0.08)
            live = await get_status(agent, "agent", source)
            assert live["status"] == IngestionStatus.DOWNLOADING.value
            assert live["updated_at"] > stale_before, "the writer never refreshed updated_at"

            claimed = await claim_source_for_resume(
                agent, "agent", source, stale_after=300, owner="live-writer"
            )
            assert claimed is None, "a freshly heartbeat-refreshed download was stolen"
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # Writer died: the row ages out and is reclaimed.
        await _age_doc(agent, "agent", source, age=10_000)
        reclaimed = await claim_source_for_resume(
            agent, "agent", source, stale_after=300, owner="recovery"
        )
        assert reclaimed is not None
        assert reclaimed["status"] == IngestionStatus.PROCESSING.value
    finally:
        await _drop(agent)


# ---------------------------------------------------------------------------
# N1 — lost NX create is detected
# ---------------------------------------------------------------------------


async def test_store_nx_detects_a_lost_create():
    """The second ``nx=True`` store must report the lost create as ``None``."""
    key = f"agents:{_agent_id()}:nx-probe"
    try:
        assert await crud.store(key, {"v": 1}, nx=True) is not None
        assert await crud.store(key, {"v": 2}, nx=True) is None
        stored = await crud.read(key)
        assert (stored[0] if isinstance(stored, list) else stored)["v"] == 1
    finally:
        await crud.delete(key)


# ---------------------------------------------------------------------------
# N3 — resume_created lease is finite when the claim gate is disabled
# ---------------------------------------------------------------------------


async def test_resume_created_row_is_reclaimed_after_its_lease():
    """With ``stale_after=0`` a crashed no-doc claim must not be refused forever."""
    agent = _agent_id()
    source = "doc.pdf"
    await _seed_master_key(agent)
    try:
        await set_status(agent, "agent", source, type_="file", status=IngestionStatus.PROCESSING)
        # A fresh row created by the no-doc path: still owned by its writer.
        await _age_doc(agent, "agent", source, age=0, resume_created=True)
        assert (
            await claim_source_for_resume(agent, "agent", source, stale_after=0, owner="second")
        ) is None

        # The creator crashed long ago: the lease expired, reclaim must succeed.
        await _age_doc(agent, "agent", source, age=10_000)
        claimed = await claim_source_for_resume(
            agent, "agent", source, stale_after=0, owner="second"
        )
        assert claimed is not None
        assert claimed["status"] == IngestionStatus.PROCESSING.value
    finally:
        await _drop(agent)


# ---------------------------------------------------------------------------
# N4 — batch/recovery pass starts and stops the heartbeat
# ---------------------------------------------------------------------------


class _RecordingPluginManager:
    def __init__(self):
        self.started = []
        self.stopped = []

    async def execute_hook(self, name, *args, caller=None):
        if name == "rabbithole_processing_heartbeat_start":
            self.started.append(args)
        elif name == "rabbithole_processing_heartbeat_stop":
            self.stopped.append(args)
        return None


async def test_reembed_sources_heartbeats_the_claimed_source(monkeypatch):
    """``reembed_sources`` (batch/recovery path) brackets the work with a heartbeat."""
    agent = _agent_id()
    await _seed_master_key(agent)
    plugin_manager = _RecordingPluginManager()

    async def fake_parse(*args, **kwargs):
        return "doc.pdf", []

    async def fake_record(*args, **kwargs):
        return {}

    async def fake_probe(*args, **kwargs):
        return []

    async def fake_complete(*args, **kwargs):
        return None

    async def _empty_points(*args, **kwargs):
        return [], None

    monkeypatch.setattr(reembed, "_run_parsing_phase", fake_parse)
    monkeypatch.setattr(reembed, "_record_phase", fake_record)
    monkeypatch.setattr(reembed, "_probe_pending", fake_probe)
    monkeypatch.setattr(reembed, "_complete_source", fake_complete)

    async def embedder():
        return SimpleNamespace(name="test-embedder")

    ccat = SimpleNamespace(
        _id=agent,
        agent_key=agent,
        vector_memory_handler=SimpleNamespace(
            get_all_tenant_points=_empty_points,
            add_points_to_tenant=_empty_points,
            delete_tenant_points=_empty_points,
        ),
        embedder=embedder,
        chunker=SimpleNamespace(name="test-chunker"),
        rabbit_hole=SimpleNamespace(),
        plugin_manager=plugin_manager,
    )
    source = SimpleNamespace(name="doc.pdf", path="doc.pdf", content=None, metadata={})
    try:
        await reembed.reembed_sources(
            ccat, VectorMemoryType.DECLARATIVE, [source], caller_cat=ccat
        )
        assert plugin_manager.started, "batch re-embed never started a heartbeat"
        assert plugin_manager.stopped, "batch re-embed never stopped its heartbeat"
        assert plugin_manager.started[0][0] == "doc.pdf"
    finally:
        await _drop(agent)
