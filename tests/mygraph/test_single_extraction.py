"""One document -> one entity extraction under the efficient ingestion engine.

The efficient engine (EffING) stores every chunk TWICE: first with empty
vectors in the ``parsing_chunking`` phase (``_store_empty_vectors``), then with
the real vectors after the ``embedding`` phase deletes and re-adds the points.
Entity extraction is vector-independent, so an ungated ``add_point_to_tenant``
runs the whole graph build on BOTH passes — twice per document.

The fix gates graph work on the PASS, not on vector validity: the staging pass
is the recorded ``parsing_chunking`` phase (``_is_empty_vector_staging_pass``),
while the ``embedding`` pass owns the graph. Vector validity would wrongly drop
legitimate payload-only / zero-vector documents (e.g. an image under a
non-multimodal embedder). These tests count the extraction calls per pass:

- efficient flow (staging pass + embedding pass): exactly 1 per document;
- classic flow (a single pass, no ingestion status): exactly 1;
- payload-only document in the embedding pass: MUST build the graph once;
- the discriminator reads the recorded phase AND the phase-machine claim
  (parsing_chunking + `resume_owner` -> staging; classic is unclaimed -> build).

Only ``add_point_to_tenant`` / ``add_points_to_tenant`` are exercised; Neo4j is
replaced by an accepting fake session and the spaCy extractor by a counter, so
no database/spaCy is required.
"""

import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
PKG = "_mygraph_single_extraction"


def _load_handler_module():
    """Import the real GraphRAG handler with only its hard deps stubbed.

    ``neo4j`` is not installed in the core test env and the spaCy-backed
    ``entity_extractor`` is replaced by a counter-friendly stub. Everything
    else (cat, pydantic, langchain_core, epoch/versioning/models) is real. The
    temporary stubs are restored afterwards so the rest of the test session is
    unaffected.
    """
    saved = {name: sys.modules.get(name) for name in ("neo4j", "neo4j.exceptions", PKG)}

    neo4j = types.ModuleType("neo4j")
    neo4j.AsyncGraphDatabase = type("AsyncGraphDatabase", (), {})
    neo4j.AsyncDriver = type("AsyncDriver", (), {})
    neo4j.AsyncSession = type("AsyncSession", (), {})
    sys.modules["neo4j"] = neo4j
    neo4j_exc = types.ModuleType("neo4j.exceptions")
    neo4j_exc.Neo4jError = type("Neo4jError", (), {})
    sys.modules["neo4j.exceptions"] = neo4j_exc

    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(PLUGIN_DIR)]
    sys.modules[PKG] = pkg

    entity_extractor = types.ModuleType(PKG + ".entity_extractor")

    class _EntityExtractor:
        @staticmethod
        def get_entity_hash(name, entity_type, tenant_id):
            return f"{tenant_id}:{name}"

    entity_extractor.EntityExtractor = _EntityExtractor
    sys.modules[PKG + ".entity_extractor"] = entity_extractor

    try:
        return importlib.import_module(PKG + ".graphrag_handler")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


handler_module = _load_handler_module()
GraphRAGHandler = handler_module.GraphRAGHandler


class _Point:
    """Minimal PointStruct stand-in: id + payload + vector."""

    def __init__(self, point_id, content="doc text", vector=None, source="file.pdf"):
        self.id = point_id
        self.vector = {} if vector is None else vector
        self.payload = {"page_content": content, "metadata": {"source": source}}


class _Result:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record


class _Session:
    """Accepts every Cypher query; the tests only count extraction calls."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        return _Result({"id": params.get("id")})


def _make_handler(enabled_derived_graph=False):
    handler = GraphRAGHandler.__new__(GraphRAGHandler)
    handler.agent_id = "tenant_test"
    handler._generation = "v1"
    handler._names = {"embedding_prop": "embedding_v1"}
    handler._enable_entity_extraction = True
    handler._entity_extractor = object()
    handler._enable_derived_graph = enabled_derived_graph
    handler._pending_entity_tasks = []
    handler._pending_source_tasks = {}
    handler.extraction_calls = 0
    handler.derived_calls = 0
    # Sources whose write belongs to the empty-vector staging pass. The tests
    # toggle this instead of relying on vector validity (the production
    # discriminator reads the recorded EffING phase).
    handler.staging_sources = set()

    async def _is_staging(source, metadata):
        return source in handler.staging_sources

    handler._is_empty_vector_staging_pass = _is_staging

    async def _ensure_connected():
        return None

    async def _read_generation():
        return "v1"

    async def _extract(point_id, content, metadata):
        handler.extraction_calls += 1

    async def _similarity(*args, **kwargs):
        return None

    async def _derived(source, stored_points, *args, **kwargs):
        handler.derived_calls += 1

    handler._ensure_connected = _ensure_connected
    handler._read_generation = _read_generation
    handler._extract_and_link_entities = _extract
    handler._create_similarity_relationships = _similarity
    handler.create_derived_graph_for_source = _derived
    handler._get_session = lambda: _Session()
    return handler


async def _drain(handler):
    tasks = [t for t in handler._pending_entity_tasks if not t.done()]
    if tasks:
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_efficient_flow_extracts_exactly_once_per_document():
    """Empty-vector staging pass does nothing; the embedding pass extracts
    once. Ungated, the same counters would be 1 + 1 = 2."""
    handler = _make_handler()

    handler.staging_sources.add("file.pdf")
    await handler.add_points_to_tenant(
        "declarative", [_Point("p-empty", content="same document chunk", vector={})]
    )
    await _drain(handler)
    assert handler.extraction_calls == 0  # staging pass: no graph work

    handler.staging_sources.clear()
    await handler.add_points_to_tenant(
        "declarative", [_Point("p-real", content="same document chunk", vector=[0.1, 0.2, 0.3])]
    )
    await _drain(handler)
    assert handler.extraction_calls == 1  # embedding pass: exactly once


@pytest.mark.asyncio
async def test_classic_flow_extraction_count_is_unchanged():
    """Classic single real-vector pass: baseline count is exactly 1, identical
    to the pre-fix behaviour (the fix only removes the empty staging pass)."""
    handler = _make_handler()

    await handler.add_points_to_tenant(
        "declarative", [_Point("p-classic", content="classic chunk", vector=[1.0, 0.0, 0.0])]
    )
    await _drain(handler)

    assert handler.extraction_calls == 1


@pytest.mark.asyncio
async def test_efficient_flow_builds_derived_graph_exactly_once():
    """The derived graph (incl. step-8 LLM concept extraction) also runs once:
    skipped for the staging batch, run for the embedding batch."""
    handler = _make_handler(enabled_derived_graph=True)

    handler.staging_sources.add("file.pdf")
    await handler.add_points_to_tenant(
        "declarative", [_Point("p-empty", content="chunk", vector={})]
    )
    await _drain(handler)
    assert handler.derived_calls == 0

    handler.staging_sources.clear()
    await handler.add_points_to_tenant(
        "declarative", [_Point("p-real", content="chunk", vector=[0.1, 0.2, 0.3])]
    )
    await _drain(handler)
    assert handler.derived_calls == 1


@pytest.mark.asyncio
async def test_payload_only_document_still_builds_the_graph():
    """B2: a legitimate payload-only point (empty/zero vector) in the embedding
    pass MUST build the derived graph and extract once — vector validity is no
    longer a gate."""
    handler = _make_handler(enabled_derived_graph=True)

    await handler.add_points_to_tenant(
        "declarative", [_Point("p-image", content="[Image] file.pdf", vector={})]
    )
    await _drain(handler)

    assert handler.extraction_calls == 1, "payload-only document must extract once"
    assert handler.derived_calls == 1, "payload-only document must build the graph once"


@pytest.mark.asyncio
async def test_staging_discriminator_reads_recorded_ingestion_phase(monkeypatch):
    """B2: the real discriminator is the recorded phase, not vector validity."""
    from cat.core_plugins.cat_efficient_ingestion import registry

    handler = _make_handler()
    del handler._is_empty_vector_staging_pass  # reveal the real class method

    async def _claimed_staging(agent_id, scope, source):
        assert agent_id == "tenant_test"
        assert scope == "agent"
        # the phase machine claimed the source before staging
        return {"phase": "parsing_chunking", "resume_owner": "reembed-1"}

    monkeypatch.setattr(registry, "get_status", _claimed_staging)
    assert await handler._is_empty_vector_staging_pass(
        "file.pdf", {"source": "file.pdf"}
    ) is True

    async def _classic_unclaimed(agent_id, scope, source):
        # classic core also has a parsing_chunking row, but no phase-machine claim
        return {"phase": "parsing_chunking"}

    monkeypatch.setattr(registry, "get_status", _classic_unclaimed)
    assert await handler._is_empty_vector_staging_pass(
        "file.pdf", {"source": "file.pdf"}
    ) is False

    async def _embedding(agent_id, scope, source):
        return {"phase": "embedding", "resume_owner": "reembed-1"}

    monkeypatch.setattr(registry, "get_status", _embedding)
    assert await handler._is_empty_vector_staging_pass(
        "file.pdf", {"source": "file.pdf"}
    ) is False
