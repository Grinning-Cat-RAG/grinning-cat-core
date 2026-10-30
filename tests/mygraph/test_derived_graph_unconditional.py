"""The document-structure graph is built unconditionally.

``enable_derived_graph`` was retired: the SourceFile/PART_OF/NEXT/HAS_SUMMARY
structure must be created on every ingest, with no toggle able to disable it.
The retired key is accepted-but-IGNORED in saved configs (it no longer exists
on the config model), so a full ``model_dump()`` still instantiates the
handler without a ``TypeError`` and the factory cannot silently fall back to
Qdrant.

The graph writes are exercised at the unit level (no Neo4j): the real handler
is instantiated from a full config dump, its driver/session replaced by a
recorder, and ``create_derived_graph_for_source`` is called directly.
"""

import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

PLUGIN_SRC = Path("/root/git/plugins/cat-graphrag.my")
PACKAGE = "_mygraph_unconditional"

_QUERY_LOG = []


def _load_graphrag_handler():
    """Import the real plugin handler under a synthetic package name."""
    for name in list(sys.modules):
        if name == PACKAGE or name.startswith(PACKAGE + "."):
            sys.modules.pop(name, None)
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(PLUGIN_SRC)]
    package.__package__ = PACKAGE
    sys.modules[PACKAGE] = package
    return importlib.import_module(PACKAGE + ".graphrag_handler")


class _FakeResult:
    async def data(self):
        return []


class _RecordingSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        _QUERY_LOG.append((query, params))
        return _FakeResult()


class _Point:
    def __init__(self, pid, metadata):
        self.id = pid
        self.vector = [0.1, 0.2, 0.3]
        self.payload = {"page_content": pid, "metadata": metadata}


@pytest.fixture
def handler():
    gh = _load_graphrag_handler()
    # Mimic a saved config that still carries the retired key.
    dumped = gh.Neo4jGraphRAGConfig.parse_config({
        "neo4j_uri": "bolt://fake",
        "neo4j_user": "u",
        "neo4j_password": None,
        "enable_derived_graph": False,
    })
    instance = gh.GraphRAGHandler(**dumped)
    instance.agent_id = "agent_unconditional"

    async def _ensure_connected():
        return None

    instance._ensure_connected = _ensure_connected
    instance._get_session = lambda: _RecordingSession()
    _QUERY_LOG.clear()
    return instance


def test_retired_key_is_not_a_config_field_and_is_ignored(handler):
    gh = _load_graphrag_handler()
    assert "enable_derived_graph" not in gh.Neo4jGraphRAGConfig.model_fields
    assert not hasattr(handler, "_enable_derived_graph")


def test_factory_instantiates_from_full_model_dump_without_type_error(handler):
    """The factory path (``parse_config`` -> ``model_dump()`` -> ``pyclass``)
    must not raise ``TypeError`` — a model field without a matching ``__init__``
    argument would make it fall back to Qdrant silently."""
    gh = _load_graphrag_handler()
    dumped = gh.Neo4jGraphRAGConfig.parse_config({
        "neo4j_uri": "bolt://fake",
        "neo4j_user": "u",
        "neo4j_password": None,
    })
    assert "enable_derived_graph" not in dumped
    rebuilt = gh.GraphRAGHandler(**dumped)
    assert isinstance(rebuilt, gh.GraphRAGHandler)


def test_create_derived_graph_always_writes_part_of_next_and_summary(handler):
    points = [
        _Point("chunk-0", {"source": "file.pdf", "chunk_index": 0}),
        _Point("chunk-1", {"source": "file.pdf", "chunk_index": 1}),
        _Point("card", {"source": "file.pdf", "is_catalogue_card": True}),
    ]

    asyncio.run(handler.create_derived_graph_for_source("file.pdf", points))

    all_queries = "\n".join(q for q, _ in _QUERY_LOG)
    assert "MERGE (sf:SourceFile" in all_queries
    assert "MERGE (d)-[:PART_OF]->(sf)" in all_queries
    assert "MERGE (a)-[:NEXT]->(b)" in all_queries
    assert "MERGE (sf)-[:HAS_SUMMARY]->(card)" in all_queries


def test_next_edges_follow_start_index_when_present(handler):
    points = [
        _Point("A", {"source": "file.pdf", "chunk_index": 2, "start_index": 10}),
        _Point("B", {"source": "file.pdf", "chunk_index": 0, "start_index": 30}),
        _Point("C", {"source": "file.pdf", "chunk_index": 1, "start_index": 20}),
    ]

    asyncio.run(handler.create_derived_graph_for_source("file.pdf", points))

    pairs = next(
        params["pairs"]
        for query, params in _QUERY_LOG
        if "MERGE (a)-[:NEXT]->(b)" in query
    )
    assert pairs == [
        {"curr": "A", "next": "C"},
        {"curr": "C", "next": "B"},
    ]


def test_next_edges_fall_back_to_chunk_index_without_start_index(handler):
    points = [
        _Point("A", {"source": "file.pdf", "chunk_index": 2}),
        _Point("B", {"source": "file.pdf", "chunk_index": 0}),
        _Point("C", {"source": "file.pdf", "chunk_index": 1}),
    ]

    asyncio.run(handler.create_derived_graph_for_source("file.pdf", points))

    pairs = next(
        params["pairs"]
        for query, params in _QUERY_LOG
        if "MERGE (a)-[:NEXT]->(b)" in query
    )
    assert pairs == [
        {"curr": "B", "next": "C"},
        {"curr": "C", "next": "A"},
    ]
