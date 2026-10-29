"""Tri-state contract of MyGRAPH's ``ingestion_phase_run`` (B1).

EffING's phase machine treats a ``None`` return as a hard ERROR (fail-hard for
an unimplemented phase). MyGRAPH must therefore return the documented tri-state:

======================  ============================================
situation               return
======================  ============================================
handler not GraphRAG    ``{"status": "done"}`` (nothing to do)
derived graph disabled  ``{"status": "done"}`` (nothing to do)
points not stored yet   ``{"status": "not_ready", "retry_after": 5}``
phase done              ``{"status": "done"}``
unknown phase           ``None`` (genuinely unimplemented -> error)
======================  ============================================

The real plugin ``main.py`` is loaded (its heavy relative modules stubbed) and
driven through the resolved hook function, exactly like the host does.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
MAIN_PATH = PLUGIN_DIR / "main.py"


def _cleanup(package_name: str) -> None:
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            sys.modules.pop(name, None)


def _load_plugin_main(package_name: str):
    assert MAIN_PATH.is_file(), f"MyGRAPH plugin not found at {PLUGIN_DIR}"

    package = types.ModuleType(package_name)
    package.__path__ = [str(PLUGIN_DIR)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    class _GraphRAGHandler:
        def __init__(self, *, enable_derived_graph=True):
            self._enable_derived_graph = enable_derived_graph

        async def run_spacy_extraction_phase(self, source, cat=None):
            return {"status": "done"}

        async def run_llm_extraction_phase(self, source, cat=None):
            return {"status": "done"}

    handler = types.ModuleType(package_name + ".graphrag_handler")
    setattr(handler, "GraphRAGHandler", _GraphRAGHandler)
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
    return module, _GraphRAGHandler


def _cat(handler):
    return types.SimpleNamespace(vector_memory_handler=handler)


def test_done_when_handler_is_not_graphrag():
    module, _ = _load_plugin_main("_mygraph_run_handler_absent")
    try:
        result = asyncio.run(module.ingestion_phase_run.function(
            "graph_spacy_extraction", "file.pdf", [], _cat(object())
        ))
        assert result == {"status": "done"}, result
    finally:
        _cleanup("_mygraph_run_handler_absent")


def test_done_when_derived_graph_disabled():
    module, handler_cls = _load_plugin_main("_mygraph_run_derived_off")
    try:
        handler = handler_cls(enable_derived_graph=False)
        result = asyncio.run(module.ingestion_phase_run.function(
            "graph_spacy_extraction", "file.pdf", [], _cat(handler)
        ))
        assert result == {"status": "done"}, result
    finally:
        _cleanup("_mygraph_run_derived_off")


def test_passes_through_not_ready_when_points_absent():
    module, handler_cls = _load_plugin_main("_mygraph_run_not_ready")
    try:
        class _NotReady(handler_cls):
            async def run_spacy_extraction_phase(self, source, cat=None):
                return {"status": "not_ready", "retry_after": 5}

        result = asyncio.run(module.ingestion_phase_run.function(
            "graph_spacy_extraction", "file.pdf", [], _cat(_NotReady())
        ))
        assert result == {"status": "not_ready", "retry_after": 5}, result
    finally:
        _cleanup("_mygraph_run_not_ready")


def test_unknown_phase_is_fail_hard_none():
    module, handler_cls = _load_plugin_main("_mygraph_run_unknown")
    try:
        result = asyncio.run(module.ingestion_phase_run.function(
            "embedding", "file.pdf", [], _cat(handler_cls())
        ))
        assert result is None, result
    finally:
        _cleanup("_mygraph_run_unknown")
