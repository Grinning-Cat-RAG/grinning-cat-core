"""Round-2 concurrent two-agent finding (BN1/NN4) kept in core.

The case below needs the real ``lizard`` (two real ``CheshireCat`` instances,
real chunker services and a barrier inside both split steps), so it is a
cross-repo integration test.

The pure heartbeat/lease cases (NN1-NN3) were handed to the plugin's own
suite: ``cat-efficient-ingestion/test_round2_findings.py``.

Redis keys live under ``agents:effingtest_<uuid>:`` and are destroyed in
``finally``. The host ``tests/conftest.py`` forces ``CAT_REDIS_DB=1``.
"""
import asyncio
import uuid

from cat.core_plugins.cat_efficient_ingestion import hooks
from cat.core_plugins.cat_efficient_ingestion import reembed
from cat.core_plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    get_status,
)
from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.db.models import Setting
from cat.services.factory.embedder import embedding_size
from cat.services.memory.models import VectorMemoryType


def _agent_id() -> str:
    return f"effingtest_{uuid.uuid4().hex}"


async def _drop(agent_id: str) -> None:
    await crud.destroy(f"agents:{agent_id}:*")


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
