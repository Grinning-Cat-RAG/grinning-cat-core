import os
import pytest
from inspect import isfunction

import cat.utils as utils
from cat.looking_glass.mad_hatter.decorators.hook import CatHook
from cat.looking_glass.mad_hatter.plugin import Plugin

from tests.utils import create_mock_plugin_zip


async def _test_on_plugin_manager(plugin_manager):
    all_plugins = plugin_manager.get_core_plugins_ids

    assert len(plugin_manager.plugins.keys()) == len(all_plugins)

    for k in plugin_manager.plugins.keys():
        assert k in all_plugins
        assert isinstance(plugin_manager.plugins[k], Plugin)

    loaded_plugins = await plugin_manager.load_active_plugins_ids_from_db()
    for p in loaded_plugins:
        assert p in all_plugins
        assert plugin_manager.plugins[p].active

    # finds hooks
    assert len(plugin_manager.hooks.keys()) > 0
    for hook_name, hooks_list in plugin_manager.hooks.items():
        assert len(hooks_list) >= 1
        h = hooks_list[0]
        assert isinstance(h, CatHook)
        assert h.plugin_id in all_plugins
        assert isinstance(h.name, str)
        assert isfunction(h.function)
        assert h.priority >= 0.0

    # finds procedures
    assert len(plugin_manager.procedures_registry) == 0  # no procedure in core plugins, i.e., no tools or forms


async def test_instantiation_discovery_for_lizard(lizard):
    await _test_on_plugin_manager(lizard.plugin_manager)


async def test_instantiation_discovery_for_cheshirecat(cheshire_cat):
    await _test_on_plugin_manager(cheshire_cat.plugin_manager)


# installation tests will be run for both flat and nested plugin
@pytest.mark.parametrize("plugin_is_flat", [True, False])
async def test_plugin_install(lizard, plugin_is_flat):
    plugin_manager = lizard.plugin_manager
    core_plugins = plugin_manager.get_core_plugins_ids

    # install plugin
    new_plugin_zip_path = create_mock_plugin_zip(flat=plugin_is_flat)
    await lizard.install_plugin(new_plugin_zip_path)

    # archive extracted
    assert os.path.exists(os.path.join(utils.get_plugins_path(), "mock_plugin"))

    # plugins list updated
    assert "mock_plugin" in list(plugin_manager.plugins.keys())
    assert isinstance(plugin_manager.plugins["mock_plugin"], Plugin)
    assert (
        "mock_plugin" in await plugin_manager.load_active_plugins_ids_from_db()
    )  # plugin starts active

    # plugin is activated by default
    assert len(plugin_manager.plugins["mock_plugin"].hooks) == 3
    assert len(plugin_manager.plugins["mock_plugin"].procedures) == 4
    assert len(plugin_manager.plugins["mock_plugin"].tools) == 2
    assert len(plugin_manager.plugins["mock_plugin"].forms) == 1
    assert len(plugin_manager.plugins["mock_plugin"].mcp_clients) == 1

    # tool found
    new_tool = plugin_manager.plugins["mock_plugin"].tools[0]
    assert new_tool.plugin_id == "mock_plugin"
    assert id(new_tool) == id(plugin_manager.procedures_registry["mock_tool"])  # cached and same object in memory!
    # tool examples found
    assert len(new_tool.examples) == 2
    assert "mock tool example 1" in new_tool.examples
    assert "mock tool example 2" in new_tool.examples

    # hooks found
    new_hooks = plugin_manager.plugins["mock_plugin"].hooks
    hooks_ram_addresses = []
    for h in new_hooks:
        assert h.plugin_id == "mock_plugin"
        hooks_ram_addresses.append(id(h))

    # found tool and hook have been cached
    mock_hook_name = "before_cat_sends_message"
    assert len(plugin_manager.hooks[mock_hook_name]) == 6  # 3 in cores, 1 in analytics, two in mock plugin
    expected_priorities = [3, 2, 1, 1, 1, 0]
    for hook_idx, cached_hook in enumerate(plugin_manager.hooks[mock_hook_name]):
        assert cached_hook.name == mock_hook_name
        assert (
            cached_hook.priority == expected_priorities[hook_idx]
        )  # correctly sorted by priority
        if cached_hook.plugin_id not in core_plugins:
            assert cached_hook.plugin_id == "mock_plugin"
            assert id(cached_hook) in hooks_ram_addresses  # same object in memory!

    # list of active plugins in DB is correct
    active_plugins = await plugin_manager.load_active_plugins_ids_from_db()
    assert len(active_plugins) == len(core_plugins) + 1
    assert "mock_plugin" in active_plugins


async def test_plugin_uninstall_non_existent(lizard):
    plugin_manager = lizard.plugin_manager
    core_plugins = plugin_manager.get_core_plugins_ids

    # should not throw error
    assert len(plugin_manager.plugins) == len(core_plugins)
    await lizard.uninstall_plugin("wrong_plugin")
    assert len(plugin_manager.plugins) == len(core_plugins)

    # list of active plugins in DB is correct
    active_plugins = await plugin_manager.load_active_plugins_ids_from_db()
    assert len(active_plugins) == len(core_plugins)
    for p in active_plugins:
        assert p in core_plugins


@pytest.mark.parametrize("plugin_is_flat", [True, False])
async def test_plugin_uninstall(lizard, plugin_is_flat):
    plugin_manager = lizard.plugin_manager
    core_plugins = plugin_manager.get_core_plugins_ids

    # install plugin
    new_plugin_zip_path = create_mock_plugin_zip(flat=plugin_is_flat)
    await lizard.install_plugin(new_plugin_zip_path)

    # uninstall
    await lizard.uninstall_plugin("mock_plugin")

    # directory removed
    assert not os.path.exists(os.path.join(utils.get_plugins_path(), "mock_plugin"))

    # plugins list updated
    assert "mock_plugin" not in plugin_manager.plugins.keys()
    # plugin cache updated (only core plugins stuff)
    assert len(plugin_manager.hooks) > 0
    for h_name, h_list in plugin_manager.hooks.items():
        assert len(h_list) >= 1
        assert h_list[0].plugin_id in core_plugins
    assert len(plugin_manager.procedures_registry) == 0  # no procedure in core plugins, i.e., no tools or forms

    # list of active plugins in DB is correct
    active_plugins = await plugin_manager.load_active_plugins_ids_from_db()
    assert len(active_plugins) == len(core_plugins)
    for p in active_plugins:
        assert p in core_plugins


async def test_ingestion_status_is_untoggling_and_forced_active(lizard):
    """ingestion_status must be always-on: part of the untoggling set, so it is
    force-added to every agent's active_plugins (even ones that previously
    customized the list without it) and cannot be deactivated."""
    plugin_manager = lizard.plugin_manager

    assert "ingestion_status" in plugin_manager.get_untoggling_plugin_ids

    # Simulate an agent that customized active_plugins WITHOUT ingestion_status
    from cat.db.cruds import settings as crud_settings
    from cat.db.models import Setting

    customized = [p for p in plugin_manager.get_core_plugins_ids if p != "ingestion_status"]
    await crud_settings.upsert_setting_by_name(
        plugin_manager.agent_key, Setting(name="active_plugins", value=customized)
    )

    active_plugins = await plugin_manager.load_active_plugins_ids_from_db()
    assert "ingestion_status" in active_plugins

    # registered hooks are active
    from cat.core_plugins.ingestion_status.plugin import rabbithole_ingestion_start

    hook_plugins = {h.plugin_id for h in plugin_manager.hooks.get("rabbithole_ingestion_start", [])}
    assert "ingestion_status" in hook_plugins


async def test_ingestion_status_cannot_be_deactivated(lizard):
    plugin_manager = lizard.plugin_manager

    with pytest.raises(Exception, match="cannot be deactivated"):
        await plugin_manager.toggle_plugin("ingestion_status")


async def test_multimodal_ingestion_is_untoggling_and_forced_active(lizard):
    """multimodal_ingestion must be always-on so its file-deletion cascade is
    registered on pre-existing agents too (the ones whose active_plugins list
    was seeded before that core plugin existed). Without this, deleting a file
    from which images were extracted never removes the image files."""
    plugin_manager = lizard.plugin_manager

    assert "multimodal_ingestion" in plugin_manager.get_untoggling_plugin_ids

    # Simulate an agent that customized active_plugins WITHOUT multimodal_ingestion
    from cat.db.cruds import settings as crud_settings
    from cat.db.models import Setting

    customized = [p for p in plugin_manager.get_core_plugins_ids if p != "multimodal_ingestion"]
    await crud_settings.upsert_setting_by_name(
        plugin_manager.agent_key, Setting(name="active_plugins", value=customized)
    )

    active_plugins = await plugin_manager.load_active_plugins_ids_from_db()
    assert "multimodal_ingestion" in active_plugins

    # registered hooks are active (including the deletion cascade)
    hook_plugins = {h.plugin_id for h in plugin_manager.hooks.get("before_file_manager_file_delete", [])}
    assert "multimodal_ingestion" in hook_plugins


async def test_multimodal_ingestion_cannot_be_deactivated(lizard):
    plugin_manager = lizard.plugin_manager

    with pytest.raises(Exception, match="cannot be deactivated"):
        await plugin_manager.toggle_plugin("multimodal_ingestion")
