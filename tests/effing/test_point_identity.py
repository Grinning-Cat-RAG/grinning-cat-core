"""Point identity through the efficient-ingestion store/embed phases.

A Document-level id (e.g. the deterministic catalogue-card uuid5) must survive
the phase machine: the parsing/chunking store and the embedding re-store both
use it as the point id, so a re-ingest upserts instead of duplicating. Plain
chunks (no Document id) keep the historical random uuid4.
"""
import uuid
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from cat.core_plugins.cat_efficient_ingestion import reembed
from cat.services.memory.models import PointStruct, VectorMemoryType


class _FakeEmbedder:
    name = "test-embedder"
    size = 384

    def embed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]


def _fake_cat():
    stored: dict = {}

    async def add_points(collection_name, points):
        stored.setdefault("added", []).extend(points)

    async def delete_points(collection_name, metadata=None):
        stored["deleted"] = metadata

    cat = SimpleNamespace(
        agent_key="agent",
        vector_memory_handler=SimpleNamespace(
            add_points_to_tenant=add_points,
            delete_tenant_points=delete_points,
        ),
        plugin_manager=SimpleNamespace(execute_hook=_noop_hook),
    )
    return cat, stored


async def _noop_hook(*args, **kwargs):
    return None


@pytest.fixture(autouse=True)
def _no_cancel(monkeypatch):
    async def _not_canceled(agent_key):
        return False

    monkeypatch.setattr(reembed, "ingestion_canceled", _not_canceled)


async def test_store_empty_vectors_honours_document_id():
    cat, stored = _fake_cat()
    card_id = uuid.uuid5(uuid.NAMESPACE_URL, "catalogue_card:test.txt").hex
    docs = [Document(page_content="card", metadata={"source": "test.txt"}, id=card_id)]

    points = await reembed._store_empty_vectors(
        cat, VectorMemoryType.DECLARATIVE, "test.txt", None, docs, [],
    )

    assert [p.id for p in points] == [card_id]
    assert [p.id for p in stored["added"]] == [card_id]


async def test_store_empty_vectors_generates_uuid4_without_document_id():
    cat, stored = _fake_cat()
    docs = [Document(page_content="chunk", metadata={"source": "test.txt"})]

    points = await reembed._store_empty_vectors(
        cat, VectorMemoryType.DECLARATIVE, "test.txt", None, docs, [],
    )

    assert len(points) == 1
    assert len(points[0].id) == 32
    assert uuid.UUID(points[0].id).version == 4
    assert stored["added"][0].id == points[0].id


async def test_embed_phase_preserves_stored_point_id():
    cat, stored = _fake_cat()
    card_id = uuid.uuid5(uuid.NAMESPACE_URL, "catalogue_card:test.txt").hex
    source_points = [
        PointStruct(
            id=card_id,
            payload={"id": card_id, "page_content": "card", "metadata": {"source": "test.txt"}},
            vector={},
        ),
    ]

    points = await reembed._embed_phase(
        cat, VectorMemoryType.DECLARATIVE, "test.txt", source_points,
        _FakeEmbedder(), None, cat, rabbit_hole=None,
    )

    assert [p.id for p in points] == [card_id]
    assert [p.id for p in stored["added"]] == [card_id]


async def test_embed_phase_reuses_existing_point_id():
    """Re-embedding keeps the id already stored on the point (idempotent pass).

    ``PointStruct`` always carries its id in the payload, so the rebuild picks
    it up and the re-store is an upsert rather than a fresh random point.
    """
    cat, stored = _fake_cat()
    source_points = [
        PointStruct(
            id="existing-point-id", payload={"page_content": "chunk", "metadata": {"source": "test.txt"}}, vector={},
        ),
    ]

    points = await reembed._embed_phase(
        cat, VectorMemoryType.DECLARATIVE, "test.txt", source_points,
        _FakeEmbedder(), None, cat, rabbit_hole=None,
    )

    assert [p.id for p in points] == ["existing-point-id"]
    assert [p.id for p in stored["added"]] == ["existing-point-id"]
