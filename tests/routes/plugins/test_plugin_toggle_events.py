"""A plugin is told when it is activated and when it is deactivated on an agent (``after_plugin_toggling_on_agent``),
e.g. to schedule its jobs and to remove them."""
from cat.db.database import get_async_db

from tests.utils import agent_id, just_installed_plugin

PLUGIN_ID = "mock_plugin_toggle_listener"


async def test_a_plugin_is_told_of_its_own_deactivation(secure_client, secure_client_headers, cheshire_cat):
    # regression: after the toggle the core ran only the hooks of the active plugins, so a plugin never knew it was
    # deactivated (e.g. its scheduled jobs went on running for the agent)
    await just_installed_plugin(secure_client, secure_client_headers, plugin_id=PLUGIN_ID)
    for _ in range(2):
        response = await secure_client.put(f"/plugins/toggle/{PLUGIN_ID}", headers=secure_client_headers)
        assert response.status_code == 200

    events = await get_async_db().lrange(f"test_toggle_events:{agent_id}", 0, -1)
    assert events == ["activated", "deactivated"]


async def test_the_deactivated_plugin_is_found_while_its_hook_runs(lizard, cheshire_cat):
    # e.g. the hooks of a plugin call `cat.mad_hatter.get_plugin()` to know their own id and settings
    from cat.looking_glass.mad_hatter.decorators.hook import CatHook

    manager = cheshire_cat.plugin_manager
    seen = []

    class Deactivated:
        id = "just_deactivated"
        hooks = [CatHook("after_plugin_toggling_on_agent", lambda plugin_id, cat: seen.append(
            (plugin_id, "just_deactivated" in manager.plugins, "just_deactivated" in manager.active_plugins)), 1)]

    result = await manager.execute_hook_of_deactivated_plugin(
        Deactivated(), "after_plugin_toggling_on_agent", "just_deactivated", caller=cheshire_cat,
    )
    assert seen == [("just_deactivated", True, False)]
    assert result == "just_deactivated"
    assert "just_deactivated" not in manager.plugins
    # a plugin without that hook: nothing to run
    Deactivated.hooks = []
    assert await manager.execute_hook_of_deactivated_plugin(
        Deactivated(), "after_plugin_toggling_on_agent", "just_deactivated", caller=cheshire_cat,
    ) == "just_deactivated"
