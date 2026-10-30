"""MyGRAPH phase registration must use the canonical EffING module identity.

The host loads ``cat/core_plugins/`` in preference to ``cat/plugins/``, so the
canonical ``PhaseSpec`` lives at
``cat.core_plugins.cat_efficient_ingestion.phases``. A consumer that imports the
same file through ``cat.plugins.cat_efficient_ingestion.phases`` creates a
SECOND module identity: ``isinstance(spec, PhaseSpec)`` is then False and
EffING's ``merged_phases`` silently drops the specs. These tests load the real
MyGRAPH ``main.py`` (its heavy relative modules are stubbed) and pin the
canonical identity plus the exactly-two-phase contract.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from cat.core_plugins.cat_efficient_ingestion import phases

PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
MAIN_PATH = PLUGIN_DIR / "main.py"
CANONICAL = "cat.core_plugins.cat_efficient_ingestion.phases"
LEGACY = "cat.plugins.cat_efficient_ingestion"
MISSING = object()

EXPECTED_IDS = ["graph_spacy_extraction", "graph_llm_extraction"]
EXPECTED_DEPS = ("parsing_chunking", "embedding")


def _cleanup(package_name: str) -> None:
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            sys.modules.pop(name, None)


def _load_plugin_main(package_name: str):
    """Load the real plugin ``main.py`` as ``package_name.main``.

    The plugin's relative imports (``.graphrag_handler`` / ``.entity_extractor``)
    pull the neo4j/spaCy stack, which the host test environment does not
    install; both are stubbed with the two names ``main.py`` binds at import
    time. The package is removed by the caller in ``finally``.
    """
    assert MAIN_PATH.is_file(), f"MyGRAPH plugin not found at {PLUGIN_DIR}"

    package = types.ModuleType(package_name)
    package.__path__ = [str(PLUGIN_DIR)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    handler = types.ModuleType(package_name + ".graphrag_handler")
    setattr(handler, "GraphRAGHandler", type("GraphRAGHandler", (), {}))
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


def test_hook_registers_exactly_two_canonical_specs():
    """The hook returns two specs in order, all of the canonical class."""
    module = _load_plugin_main("_mygraph_phase_present")
    try:
        specs = module.ingestion_phase_specs.function([], None)

        assert len(specs) == 2
        assert [spec.id for spec in specs] == EXPECTED_IDS
        for spec in specs:
            assert isinstance(spec, phases.PhaseSpec)
            assert type(spec).__module__ == CANONICAL
            assert type(spec).__module__.startswith("cat.core_plugins")
            assert spec.settings_category is None
            assert spec.depends_on == EXPECTED_DEPS

        assert LEGACY + ".phases" not in sys.modules
    finally:
        _cleanup("_mygraph_phase_present")


@pytest.mark.asyncio
async def test_merged_phases_keeps_both_graph_phases():
    """EffING's merged_phases accepts both specs (no silent isinstance drop)."""
    module = _load_plugin_main("_mygraph_phase_merged")
    try:
        class _PluginManager:
            hooks = {"ingestion_phase_specs": module.ingestion_phase_specs}

            async def execute_hook(self, name, *args, caller=None):
                assert name == "ingestion_phase_specs"
                return module.ingestion_phase_specs.function(*args, cat=caller)

        cat = types.SimpleNamespace(plugin_manager=_PluginManager())
        merged = await phases.merged_phases(cat, cat)

        assert len(merged) == len(phases.PHASES) + 2
        assert [phase for phase in merged if phase in EXPECTED_IDS] == EXPECTED_IDS
        for phase_id in EXPECTED_IDS:
            assert isinstance(merged[phase_id], phases.PhaseSpec)
    finally:
        _cleanup("_mygraph_phase_merged")


def test_module_imports_and_returns_specs_unchanged_when_effing_blocked():
    """Without EffING the plugin imports and the hook is an accumulator no-op."""
    saved = sys.modules.get(CANONICAL, MISSING)
    sys.modules[CANONICAL] = None  # type: ignore[assignment]
    try:
        module = _load_plugin_main("_mygraph_phase_absent")
        try:
            incoming = ["sentinel"]
            assert module.ingestion_phase_specs.function(incoming, None) is incoming
        finally:
            _cleanup("_mygraph_phase_absent")
    finally:
        if saved is MISSING:
            sys.modules.pop(CANONICAL, None)
        else:
            sys.modules[CANONICAL] = saved  # type: ignore[assignment]
