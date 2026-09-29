from cat import hook
from cat.db.database import get_async_db


@hook
async def after_plugin_toggling_on_agent(plugin_id, cat):
    # as a plugin scheduling jobs does: only its own toggles, with its own state
    if plugin_id != "mock_plugin_toggle_listener":
        return
    state = "activated" if plugin_id in cat.mad_hatter.active_plugins else "deactivated"
    await get_async_db().rpush(f"test_toggle_events:{cat.agent_key}", state)
