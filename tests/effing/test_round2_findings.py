"""Round-2 findings for the efficient-ingestion slice (BN1, NN1-NN4).

Covers:

- **BN1/NN4**: two agents ingesting concurrently must each be chunked with
  their OWN agent chunker. The lizard owns ONE shared ``RabbitHole``; the
  phase machine takes a per-call copy instead of mutating that shared object,
  so concurrent re-embeds never borrow another agent's chunker and the shared
  object is left untouched.
- **NN1**: the ``resume_created`` lease covers URL rows that flip out of
  ``PROCESSING`` (to ``DOWNLOADING``/``DOWNLOADED``) while the claim gate is
  disabled.
- **NN2**: the download heartbeat bound at the HOOK level (not only via the
  registry), proving a failed download stops refreshing and ages out.
- **NN3**: replacing a heartbeat awaits the cancelled task, and a raising tick
  never kills the heartbeat.

Redis keys live under ``agents:effingtest_<uuid>:`` and are destroyed in
``finally``. The host ``tests/conftest.py`` forces ``CAT_REDIS_DB=1``.
"""
import asyncio
import time
import uuid

import pytest

from cat.core_plugins.cat_efficient_ingestion import hooks
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
from cat.services.factory.embedder import embedding_size
from cat.services.memory.models import VectorMemoryType


def _agent_id() -> str:
    return f"effingtest_{uuid.uuid4().hex}"


async def _seed_master_key(agent_id: str) -> None:
    await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])


async def _drop(agent_id: str) -> None:
    await crud.destroy(f"agents:{agent_id}:*")


async def _age_doc(agent_id: str, scope: str, source: str, *, age: float, **fields) -> None:
    key = status_key(agent_id, scope, source)
    doc = await crud.read(key)
    if isinstance(doc, list):
        doc = doc[0]
    doc = dict(doc)
    doc["updated_at"] = time.time() - age
    doc.update(fields)
    await crud.store(key, doc)


# ---------------------------------------------------------------------------
# BN1 / NN4 — concurrent two-agent re-embed uses each agent's own chunker
# ---------------------------------------------------------------------------

_LONG_TEXT = ("The quick brown fox jumps over the lazy dog. " * 100).strip()


async def _put_chunk_size(agent_id: str, size: int) -> None:
    await crud_settings.upsert_setting_by_category(
        agent_id,
        Setting(
            name="RecursiveTextChunkerSettings",
            value={"encoding_name": "cl100k_base", "chunk_size": size, "chunk_overlap": 0},
            category="chunker",
        ),
    )


async def _point_count(ccat, source: str) -> int:
    points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
        str(VectorMemoryType.DECLARATIVE), with_vectors=False
    )
    return sum(
        1 for p in points
        if (p.payload or {}).get("metadata", {}).get("source") == source
    )


async def _marker(phase: str, cat) -> object:
    result = hooks.ingestion_phase_settings_marker.function(phase, cat)
    return await result if hasattr(result, "__await__") else result


async def test_concurrent_two_agent_ingestion_uses_each_agents_chunker(lizard):
    """Both agents ingest at once; each source must be chunked by its own agent.

    The two chunkers are held at an ``asyncio.Barrier`` inside the split step,
    so both phase machines are provably inside chunking at the same time. With
    the old shared-``RabbitHole`` mutation whichever ``setup`` ran last would
    win and BOTH sources would get the same chunker.
    """
    agent_a = _agent_id()
    agent_b = _agent_id()
    source_a = "concurrent-a.txt"
    source_b = "concurrent-b.txt"
    engine = reembed.EfficientIngestionEngine()
    ccat_a = None
    ccat_b = None
    try:
        ccat_a = await lizard.create_cheshire_cat(agent_a)
        ccat_b = await lizard.create_cheshire_cat(agent_b)
        await _put_chunk_size(agent_a, 32)
        await _put_chunk_size(agent_b, 256)
        # Services are snapshotted at creation: re-resolve to apply the chunker
        # settings written above, then re-initialize the (fresh) vector handler
        # so it sees the collections created by ``create_cheshire_cat``.
        ccat_a = await lizard.get_cheshire_cat(agent_a)
        ccat_b = await lizard.get_cheshire_cat(agent_b)
        embedder = await lizard.embedder()
        size = await embedding_size(embedder)
        await ccat_a.vector_memory_handler.initialize(embedder.name, size)
        await ccat_b.vector_memory_handler.initialize(embedder.name, size)
        assert ccat_a.chunker._chunk_size == 32
        assert ccat_b.chunker._chunk_size == 256

        # Hold both split steps at the same barrier: the old code has already
        # mutated the shared RabbitHole by the time it reaches the split.
        barrier = asyncio.Barrier(2)

        def _barriered(chunker):
            original = chunker.split_documents

            async def split_documents(docs):
                await barrier.wait()
                return await original(docs)

            chunker.split_documents = split_documents

        _barriered(ccat_a.chunker)
        _barriered(ccat_b.chunker)

        await asyncio.gather(
            engine.ingest_file(ccat_a, _LONG_TEXT.encode(), source_a, content_type="text/plain"),
            engine.ingest_file(ccat_b, _LONG_TEXT.encode(), source_b, content_type="text/plain"),
        )

        count_a = await _point_count(ccat_a, source_a)
        count_b = await _point_count(ccat_b, source_b)
        assert count_a > 1 and count_b > 1
        # The 32-token chunker must produce many more chunks than the 256 one.
        assert count_a > count_b * 2, (
            f"agent A (32) and agent B (256) produced {count_a}/{count_b} chunks: "
            "an agent was chunked with the other agent's chunker"
        )

        doc_a = await get_status(agent_a, "agent", source_a)
        doc_b = await get_status(agent_b, "agent", source_b)
        assert doc_a["status"] == IngestionStatus.COMPLETED.value
        assert doc_b["status"] == IngestionStatus.COMPLETED.value
        assert (
            doc_a["completed_phases"]["parsing_chunking"]["marker"]
            == await _marker("parsing_chunking", ccat_a)
        )
        assert (
            doc_b["completed_phases"]["parsing_chunking"]["marker"]
            == await _marker("parsing_chunking", ccat_b)
        )
        # NN4: the shared RabbitHole was never mutated.
        assert lizard.rabbit_hole.cat is not ccat_a
        assert lizard.rabbit_hole.cat is not ccat_b
    finally:
        for cat in (ccat_a, ccat_b):
            if cat is not None:
                try:
                    await cat.destroy()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
        await _drop(agent_a)
        await _drop(agent_b)


# ---------------------------------------------------------------------------
# NN1 — the resume_created lease covers URL rows that flip out of PROCESSING
# ---------------------------------------------------------------------------


async def test_resume_created_lease_covers_a_url_row_out_of_processing():
    """A live URL row (DOWNLOADING) created by the no-doc path is not stolen."""
    agent = _agent_id()
    source = "https://example.com/doc.pdf"
    await _seed_master_key(agent)
    try:
        await set_status(agent, "agent", source, type_="url", status=IngestionStatus.PROCESSING)
        await _age_doc(agent, "agent", source, age=0, resume_created=True)

        # The phase machine flips the row to DOWNLOADING (marker is preserved).
        await set_status(agent, "agent", source, type_="url", status=IngestionStatus.DOWNLOADING)
        assert (await get_status(agent, "agent", source))["resume_created"] is True

        # Live writer, claim gate disabled: the lease must still refuse.
        assert (
            await claim_source_for_resume(agent, "agent", source, stale_after=0, owner="second")
        ) is None

        # The writer crashed long ago: the lease expired, reclaim succeeds.
        await _age_doc(agent, "agent", source, age=10_000)
        claimed = await claim_source_for_resume(
            agent, "agent", source, stale_after=0, owner="second"
        )
        assert claimed is not None
        assert claimed["status"] == IngestionStatus.PROCESSING.value
    finally:
        await _drop(agent)


# ---------------------------------------------------------------------------
# NN2 — the download heartbeat bound at the HOOK level
# ---------------------------------------------------------------------------


async def test_download_hook_heartbeat_stops_at_its_bound(monkeypatch):
    """``rabbithole_url_downloading`` bounds the heartbeat so a dead download ages out."""
    agent = _agent_id()
    source = "https://example.com/big.pdf"
    await _seed_master_key(agent)
    monkeypatch.setattr(ingestion_plugin, "_heartbeat_interval", lambda: 0.01)
    monkeypatch.setattr(ingestion_plugin, "_download_heartbeat_max_seconds", lambda: 0.05)
    cat = _HookCat(agent)
    try:
        await ingestion_plugin.rabbithole_url_downloading.function(source, source, cat)
        key = ingestion_plugin._heartbeat_key(agent, "agent", source)
        task = ingestion_plugin._heartbeat_tasks[key]

        # While inside the bound the task is alive and refreshing the row.
        await asyncio.sleep(0.02)
        assert not task.done()
        first = (await get_status(agent, "agent", source))["updated_at"]

        # Past the bound the task returns by itself (no completed hook needed).
        await asyncio.wait_for(task, timeout=1.0)
        stopped_at = (await get_status(agent, "agent", source))["updated_at"]
        await asyncio.sleep(0.05)
        after = (await get_status(agent, "agent", source))["updated_at"]
        assert after == stopped_at, "the bounded download heartbeat kept refreshing after its bound"
        assert stopped_at >= first

        # The row now ages out and becomes resumable.
        await _age_doc(agent, "agent", source, age=10_000)
        claimed = await claim_source_for_resume(
            agent, "agent", source, stale_after=300, owner="recovery"
        )
        assert claimed is not None
    finally:
        ingestion_plugin._heartbeat_tasks.pop(
            ingestion_plugin._heartbeat_key(agent, "agent", source), None
        )
        await _drop(agent)


class _HookCat:
    """Minimal agent-scoped hook caller (no ``id`` -> agent scope)."""

    def __init__(self, agent_key: str):
        self.agent_key = agent_key


# ---------------------------------------------------------------------------
# NN3 — replacing a heartbeat awaits the cancelled task; a raising tick is safe
# ---------------------------------------------------------------------------


async def test_start_heartbeat_awaits_the_replaced_task(monkeypatch):
    """The old heartbeat must be fully unwound before the replacement is installed."""
    agent = _agent_id()
    source = "doc.pdf"
    state = {"unwound": False}

    async def fake_heartbeat(agent_id, scope, src, interval, *, max_seconds=None):
        try:
            await asyncio.sleep(3600)
        finally:
            state["unwound"] = True

    monkeypatch.setattr(ingestion_plugin, "_heartbeat_status", fake_heartbeat)
    try:
        await ingestion_plugin._start_heartbeat(agent, "agent", source, 0.01)
        first = ingestion_plugin._heartbeat_tasks[
            ingestion_plugin._heartbeat_key(agent, "agent", source)
        ]
        # Let the first task actually start so its ``finally`` unwind is real.
        await asyncio.sleep(0)
        await ingestion_plugin._start_heartbeat(agent, "agent", source, 0.01)

        assert first.cancelled() or first.done()
        assert state["unwound"] is True, "the replaced heartbeat was not awaited"
    finally:
        await ingestion_plugin._cancel_heartbeat(agent, "agent", source)


async def test_heartbeat_tick_never_raises(monkeypatch):
    """A transient store error in one tick must not kill the heartbeat."""
    agent = _agent_id()
    source = "doc.pdf"
    calls = {"n": 0}

    async def flaky_get_status(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("redis blip")

    monkeypatch.setattr(ingestion_plugin, "get_status", flaky_get_status)
    task = asyncio.ensure_future(
        ingestion_plugin._heartbeat_status(agent, "agent", source, 0.01)
    )
    try:
        await asyncio.sleep(0.06)
        assert not task.done(), "a raising tick killed the heartbeat"
        assert calls["n"] >= 2, "the heartbeat did not retry after a failed tick"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
