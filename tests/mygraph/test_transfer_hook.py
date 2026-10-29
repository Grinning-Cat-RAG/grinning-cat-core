"""MyGRAPH tenant cleanup must match the host's hook contract.

The host calls ``after_vector_memory_transfer_on_agent(success, cat)``
(``cat/core_plugins/base_plugin/hooks/flow.py``, dispatched from
``CheshireCat.transfer_vector_points_from``). The plugin used to declare a
third ``old_handler_name`` parameter, so ``_run_hooks`` raised ``TypeError``,
logged it and continued: the Neo4j tenant wipe NEVER ran.

These tests load the real MyGRAPH ``main.py`` with the heavy relative modules
(``.graphrag_handler`` / ``.entity_extractor``) stubbed, invoke the hook with
the host's exact argument shape, and assert the wipe policy:

* success + new handler NOT GraphRAG -> wipe the tenant (graphrag -> other);
* failure -> no wipe;
* new handler IS GraphRAG -> no wipe (the graph phases own the work).

An adjacent test pins the classic ingestion callback to a no-op when the
EffING phase path is active.
"""

import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
MAIN_PATH = PLUGIN_DIR / "main.py"
EFFICIENT = "EfficientIngestionConfiguration"


class FakeGraphRAGHandler:
    """Current handler stand-in; counts classic graph writes."""

    def __init__(self):
        self._enable_derived_graph = True
        self.classic_calls = 0

    async def create_derived_graph_for_source(self, source, stored_points, cat):
        self.classic_calls += 1


def _wipe_spy():
    calls = []

    async def wipe_tenant_graph_data(agent_key):
        calls.append(agent_key)

    return calls, wipe_tenant_graph_data


def _cleanup(package_name):
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            sys.modules.pop(name, None)


def _load_plugin_main(package_name, wipe_spy):
    package = types.ModuleType(package_name)
    package.__path__ = [str(PLUGIN_DIR)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    handler = types.ModuleType(package_name + ".graphrag_handler")
    setattr(handler, "GraphRAGHandler", FakeGraphRAGHandler)
    setattr(handler, "Neo4jGraphRAGConfig", type("Neo4jGraphRAGConfig", (), {}))
    setattr(handler, "wipe_tenant_graph_data", wipe_spy)
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


def _cat(handler):
    return types.SimpleNamespace(vector_memory_handler=handler, agent_key="agent_test")


def _install_resolver(monkeypatch, name):
    ingestion = importlib.import_module("cat.services.factory.ingestion")

    async def resolved_config_name(lizard):
        return name

    monkeypatch.setattr(ingestion, "resolved_config_name", resolved_config_name, raising=False)


def test_hook_signature_matches_host_contract():
    """Exactly (success, cat) — a third parameter makes _run_hooks raise TypeError."""
    calls, wipe = _wipe_spy()
    module = _load_plugin_main("_mygraph_transfer_signature", wipe)
    try:
        params = list(inspect.signature(module.after_vector_memory_transfer_on_agent.function).parameters)
        assert params == ["success", "cat"]
    finally:
        _cleanup("_mygraph_transfer_signature")


@pytest.mark.asyncio
async def test_host_shaped_success_non_graphrag_wipes_tenant():
    """Host calls (success, cat) with a non-GraphRAG new handler: the wipe runs
    with no TypeError (the exact bug: the old 3-arg signature never wiped)."""
    calls, wipe = _wipe_spy()
    module = _load_plugin_main("_mygraph_transfer_wipes", wipe)
    try:
        cat = _cat(None)

        await module.after_vector_memory_transfer_on_agent.function(True, cat)

        assert calls == ["agent_test"]
    finally:
        _cleanup("_mygraph_transfer_wipes")


@pytest.mark.asyncio
async def test_no_wipe_on_failed_transfer():
    """A failed transfer must never wipe the old store, even if the new handler
    is not GraphRAG."""
    calls, wipe = _wipe_spy()
    module = _load_plugin_main("_mygraph_transfer_failure", wipe)
    try:
        cat = _cat(None)

        await module.after_vector_memory_transfer_on_agent.function(False, cat)

        assert calls == []
    finally:
        _cleanup("_mygraph_transfer_failure")


@pytest.mark.asyncio
async def test_no_wipe_when_new_handler_is_graphrag():
    """A non-graphrag -> GraphRAG switch keeps the old graph: the new handler IS
    GraphRAG, so the revalidation sweep + graph phases own the work."""
    calls, wipe = _wipe_spy()
    module = _load_plugin_main("_mygraph_transfer_new_graphrag", wipe)
    try:
        cat = _cat(FakeGraphRAGHandler())

        await module.after_vector_memory_transfer_on_agent.function(True, cat)

        assert calls == []
    finally:
        _cleanup("_mygraph_transfer_new_graphrag")


@pytest.mark.asyncio
async def test_classic_callback_noop_when_phase_path_active(monkeypatch):
    """With EffING active (same gate as the transfer hook's sibling), the
    classic ``after_rabbithole_stored_documents`` does zero graph work."""
    calls, wipe = _wipe_spy()
    module = _load_plugin_main("_mygraph_transfer_classic_gate", wipe)
    try:
        _install_resolver(monkeypatch, EFFICIENT)
        handler = FakeGraphRAGHandler()
        cat = _cat(handler)

        await module.after_rabbithole_stored_documents.function("file.pdf", [], cat)

        assert handler.classic_calls == 0
    finally:
        _cleanup("_mygraph_transfer_classic_gate")
