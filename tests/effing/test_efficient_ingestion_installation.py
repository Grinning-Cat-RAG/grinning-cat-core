"""Installation validation and phase-hook degradation tests.

Covers the system-level contract of the efficient ingestion engine:

- the ``cat_efficient_ingestion`` plugin must NOT be system-only (agents would
  lose its phase hooks);
- the BillTheLizard startup validation fires when the efficient engine is
  selected but the plugin is missing / system-only, and stays silent otherwise;
- the plugin's phase-hook guard degrades on a missing hook and still propagates
  errors raised by an existing hook.
"""

import pytest

from cat.core_plugins.cat_efficient_ingestion.hooks import execute_phase_hook
from cat.looking_glass.bill_the_lizard import (
    EFFICIENT_INGESTION_ENGINE,
    EFFICIENT_INGESTION_PLUGIN,
    _efficient_ingestion_installation_error,
    _validate_efficient_ingestion_installation,
)


class _FakePluginManager:
    def __init__(self, core=(), system_only=(), hooks=None):
        self.get_core_plugins_ids = list(core)
        self.get_system_only_plugin_ids = list(system_only)
        self.hooks = {} if hooks is None else hooks
        self.calls = []

    async def execute_hook(self, name, *args, caller=None):
        self.calls.append(name)
        return None


async def test_efficient_ingestion_plugin_is_not_system_only(lizard):
    """The plugin's phase hooks must reach every agent: it is a core plugin and
    never system-only / non-toggleable."""
    pm = lizard.plugin_manager
    assert EFFICIENT_INGESTION_PLUGIN not in pm.get_system_only_plugin_ids
    assert EFFICIENT_INGESTION_PLUGIN not in pm.get_non_toggleable_plugin_ids


async def test_validation_fires_when_plugin_absent(monkeypatch):
    logged = []
    monkeypatch.setattr(
        "cat.looking_glass.bill_the_lizard.log.error", logged.append
    )
    pm = _FakePluginManager(core=["base_plugin"])

    message = await _validate_efficient_ingestion_installation(
        pm, EFFICIENT_INGESTION_ENGINE
    )

    assert message is not None
    assert EFFICIENT_INGESTION_PLUGIN in message
    assert "not installed as a core plugin" in message
    assert logged == [message]


async def test_validation_fires_when_plugin_is_system_only(monkeypatch):
    logged = []
    monkeypatch.setattr(
        "cat.looking_glass.bill_the_lizard.log.error", logged.append
    )
    pm = _FakePluginManager(
        core=[EFFICIENT_INGESTION_PLUGIN], system_only=[EFFICIENT_INGESTION_PLUGIN]
    )

    message = await _validate_efficient_ingestion_installation(
        pm, EFFICIENT_INGESTION_ENGINE
    )

    assert message is not None
    assert "get_system_only_plugin_ids" in message
    assert logged == [message]


async def test_validation_stays_silent_when_plugin_present(monkeypatch):
    logged = []
    monkeypatch.setattr(
        "cat.looking_glass.bill_the_lizard.log.error", logged.append
    )
    pm = _FakePluginManager(core=[EFFICIENT_INGESTION_PLUGIN])

    message = await _validate_efficient_ingestion_installation(
        pm, EFFICIENT_INGESTION_ENGINE
    )

    assert message is None
    assert logged == []


async def test_validation_ignores_other_engines():
    pm = _FakePluginManager(core=["base_plugin"])
    assert _efficient_ingestion_installation_error(pm, "CoreIngestionConfiguration") is None


async def test_guard_returns_default_when_hook_missing(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "cat.core_plugins.cat_efficient_ingestion.hooks.log.warning",
        warnings.append,
    )
    pm = _FakePluginManager(hooks={})

    result = await execute_phase_hook(
        pm, "ingestion_phase_pending", [], "doc.pdf", [], default=list, caller=object()
    )

    assert result == []
    assert pm.calls == []
    assert warnings


async def test_guard_degrades_on_dispatcher_missing_hook_error(monkeypatch):
    class _NoRegistryPM:
        async def execute_hook(self, name, *args, caller=None):
            raise Exception(f"Hook {name} not present in any plugin")

    warnings = []
    monkeypatch.setattr(
        "cat.core_plugins.cat_efficient_ingestion.hooks.log.warning",
        warnings.append,
    )

    result = await execute_phase_hook(
        _NoRegistryPM(), "ingestion_phase_specs", [], default=list, caller=object()
    )

    assert result == []
    assert warnings


async def test_guard_propagates_real_hook_error(monkeypatch):
    class _RaisingPM(_FakePluginManager):
        async def execute_hook(self, name, *args, caller=None):
            raise ValueError("boom from the hook body")

    pm = _RaisingPM(hooks={"ingestion_phase_run": [object()]})

    with pytest.raises(ValueError, match="boom from the hook body"):
        await execute_phase_hook(
            pm, "ingestion_phase_run", "embedding", "doc.pdf", [], default=None, caller=object()
        )
