"""One document -> one entity extraction under the efficient ingestion engine.

The efficient engine (EffING) stores every chunk TWICE: first with empty
vectors in the ``parsing_chunking`` phase (``_store_empty_vectors``), then with
the real vectors after the ``embedding`` phase deletes and re-adds the points.
Entity extraction is vector-independent, so an ungated ``add_point_to_tenant``
runs the whole graph build on BOTH passes — twice per document.

The fix gates all graph work on a REAL vector (the handler's single normalized
``_is_valid_vector`` check). These tests count the extraction calls per pass:

- efficient flow (empty pass + real pass): exactly 1 per document;
- classic flow (a single real-vector pass): exactly 1, unchanged from before
  the fix (pinned baseline).

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
    """Empty-vector staging pass does nothing; the real-vector pass extracts
    once. Ungated, the same counters would be 1 + 1 = 2."""
    handler = _make_handler()

    await handler.add_points_to_tenant(
        "declarative", [_Point("p-empty", content="same document chunk", vector={})]
    )
    await _drain(handler)
    assert handler.extraction_calls == 0  # empty-vector pass: no graph work

    await handler.add_points_to_tenant(
        "declarative", [_Point("p-real", content="same document chunk", vector=[0.1, 0.2, 0.3])]
    )
    await _drain(handler)
    assert handler.extraction_calls == 1  # real-vector pass: exactly once


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
    skipped for the empty-vector batch, run for the real-vector batch."""
    handler = _make_handler(enabled_derived_graph=True)

    await handler.add_points_to_tenant(
        "declarative", [_Point("p-empty", content="chunk", vector={})]
    )
    await _drain(handler)
    assert handler.derived_calls == 0

    await handler.add_points_to_tenant(
        "declarative", [_Point("p-real", content="chunk", vector=[0.1, 0.2, 0.3])]
    )
    await _drain(handler)
    assert handler.derived_calls == 1
