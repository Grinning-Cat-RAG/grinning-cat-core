"""MyGRAPH embedder injection on the hook BOTH ingestion engines emit.

The embedder used to be injected only from ``before_rabbithole_stores_documents``,
which the efficient engine (EffING) NEVER emits, so under EffING the handler ran
its extraction phases with ``handler.embedder is None`` and skipped entity
embeddings. The injection now also lives on ``rabbithole_ingestion_processing``,
which BOTH engines emit (classic: ``cat/rabbit_hole.py:247``; EffING:
``cat/core_plugins/cat_efficient_ingestion/reembed.py:1105``).

These tests load the real MyGRAPH ``main.py``; its heavy relative modules are
replaced by local fakes, so no Neo4j/spaCy/Redis is needed.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
MAIN_PATH = PLUGIN_DIR / "main.py"
EFFING = "cat.core_plugins.cat_efficient_ingestion"

EFFICIENT = "EfficientIngestionConfiguration"
CORE = "BaseIngestionConfiguration"


class FakeEmbedder:
    """Stand-in for the agent-resolved embedder instance (identity matters)."""


class FakeEntityExtractor:
    def __init__(self):
        self.ensure_initialized_calls = 0

    async def ensure_initialized(self):
        self.ensure_initialized_calls += 1


class FakeGraphRAGHandler:
    """Mirrors the real handler's embedder property + once-only alignment."""

    def __init__(self):
        self._enable_derived_graph = True
        self._embedder = None
        self._alignment_done = False
        self.align_calls = 0
        self.classic_calls = 0
        self.spacy_calls = 0
        self.llm_calls = 0
        self.seen_embedder = "UNSET"
        self._entity_extractor = FakeEntityExtractor()

    @property
    def embedder(self):
        return self._embedder

    @embedder.setter
    def embedder(self, value):
        self._embedder = value

    @property
    def entity_extractor(self):
        return self._entity_extractor

    async def _align_embedder_lazy(self):
        if self._alignment_done or self._embedder is None:
            return
        self._alignment_done = True
        self.align_calls += 1

    async def create_derived_graph_for_source(self, source, stored_points, cat):
        self.classic_calls += 1
        self.seen_embedder = self._embedder

    async def run_spacy_extraction_phase(self, source, cat=None):
        self.spacy_calls += 1
        self.seen_embedder = self._embedder
        return "done"

    async def run_llm_extraction_phase(self, source, cat=None):
        self.llm_calls += 1
        self.seen_embedder = self._embedder
        return "done"


def _cleanup(package_name):
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            sys.modules.pop(name, None)


def _load_plugin_main(package_name):
    package = types.ModuleType(package_name)
    package.__path__ = [str(PLUGIN_DIR)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    handler = types.ModuleType(package_name + ".graphrag_handler")
    setattr(handler, "GraphRAGHandler", FakeGraphRAGHandler)
    setattr(handler, "Neo4jGraphRAGConfig", type("Neo4jGraphRAGConfig", (), {}))
    sys.modules[package_name + ".graphrag_handler"] = handler

    extractor = types.ModuleType(package_name + ".entity_extractor")
    setattr(extractor, "EntityExtractor", type("EntityExtractor", (), {}))
    sys.modules[package_name + ".entity_extractor"] = extractor

    spec = importlib.util.spec_from_file_location(package_name + ".main", MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name + ".main"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _cleanup(package_name)
        raise
    return module


def _install_resolver(monkeypatch, name):
    ingestion = importlib.import_module("cat.services.factory.ingestion")

    async def resolved_config_name(lizard):
        return name

    monkeypatch.setattr(ingestion, "resolved_config_name", resolved_config_name, raising=False)


def _cat(handler, embedder):
    async def resolve_embedder():
        return embedder

    return types.SimpleNamespace(
        vector_memory_handler=handler,
        agent_key="agent_test",
        embedder=resolve_embedder,
    )


@pytest.mark.asyncio
async def test_efficient_flow_has_embedder_before_extraction(monkeypatch):
    """EffING emits `rabbithole_ingestion_processing` (never the classic hook):
    the handler already holds the agent embedder when a graph phase runs."""
    _install_resolver(monkeypatch, EFFICIENT)
    module = _load_plugin_main("_mygraph_embedder_effing")
    try:
        handler = FakeGraphRAGHandler()
        embedder = FakeEmbedder()
        cat = _cat(handler, embedder)

        # EffING: processing hook, then the phase dispatcher.
        await module.rabbithole_ingestion_processing.function("file.pdf", cat)

        assert handler.embedder is embedder
        assert handler.align_calls == 1

        await module.ingestion_phase_run.function(
            "graph_spacy_extraction", "file.pdf", [], cat
        )
        await module.ingestion_phase_run.function(
            "graph_llm_extraction", "file.pdf", [], cat
        )

        assert handler.spacy_calls == 1
        assert handler.llm_calls == 1
        assert handler.seen_embedder is embedder
    finally:
        _cleanup("_mygraph_embedder_effing")


@pytest.mark.asyncio
async def test_classic_flow_keeps_identical_behaviour(monkeypatch):
    """Classic RabbitHole emits processing AND stores_documents then the
    callback: the handler/extractor still get the agent embedder."""
    _install_resolver(monkeypatch, CORE)
    module = _load_plugin_main("_mygraph_embedder_classic")
    try:
        handler = FakeGraphRAGHandler()
        embedder = FakeEmbedder()
        cat = _cat(handler, embedder)

        await module.rabbithole_ingestion_processing.function("file.pdf", cat)
        docs = [types.SimpleNamespace(metadata={})]
        returned = await module.before_rabbithole_stores_documents.function(docs, cat)
        await module.after_rabbithole_stored_documents.function("file.pdf", [], cat)

        assert returned is docs
        assert handler.embedder is embedder
        assert handler.entity_extractor.ensure_initialized_calls == 1
        assert docs[0].metadata["chunk_index"] == 0
        assert handler.classic_calls == 1
        assert handler.seen_embedder is embedder
    finally:
        _cleanup("_mygraph_embedder_classic")


@pytest.mark.asyncio
async def test_injection_twice_is_harmless(monkeypatch):
    """Re-injecting (classic emits both hooks; retries) changes no state."""
    _install_resolver(monkeypatch, EFFICIENT)
    module = _load_plugin_main("_mygraph_embedder_idempotent")
    try:
        handler = FakeGraphRAGHandler()
        embedder = FakeEmbedder()
        cat = _cat(handler, embedder)

        await module.rabbithole_ingestion_processing.function("file.pdf", cat)
        first = handler.embedder
        await module.rabbithole_ingestion_processing.function("file.pdf", cat)
        await module.rabbithole_ingestion_processing.function("file.pdf", cat)

        assert handler.embedder is first is embedder
        assert handler.align_calls == 1

        docs = [types.SimpleNamespace(metadata={})]
        await module.before_rabbithole_stores_documents.function(docs, cat)
        await module.before_rabbithole_stores_documents.function(docs, cat)

        assert handler.embedder is embedder
        assert handler.align_calls == 1
        assert docs[0].metadata["chunk_index"] == 0
    finally:
        _cleanup("_mygraph_embedder_idempotent")
