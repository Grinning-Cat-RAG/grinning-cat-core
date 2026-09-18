from pydantic import BaseModel

from cat import plugin, log
from cat.db.cruds import plugins as crud_plugins
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.db.models import Setting

# entry name under which this plugin persists its settings in the global
# ``system:agent`` settings list, exactly like the system-level embedder
# configuration (same ``{name, value, category, setting_id, updated_at}`` shape)
_MGMT_SETTING_NAME = "mgmt_message"


async def drop_agent_keys() -> bool:
    """Best-effort removal of the leftover ``agents:<id>:plugins:mgmt_message`` keys.

    The core writes one of those keys per agent at plugin activation
    (``activate_settings``); this plugin's single source of truth is the global
    ``system:agent`` settings list, so the leftovers are dropped whenever the
    settings are written.

    The removal goes through ``crud_plugins.destroy_plugin`` (a pattern-based
    SCAN + DEL): ``crud.delete`` targets a single exact key and would silently
    match nothing for a wildcard.
    """
    try:
        await crud_plugins.destroy_plugin(_MGMT_SETTING_NAME)
    except Exception as e:
        log.error(f"mgmt_message agent keys delete failed: {e}")
    return True


class PluginSettings(BaseModel):
    management_message: str = ""
    management_active: bool = False
    global_message: str = ""
    show_global_msg: bool = False


@plugin
def settings_schema() -> dict:
    return PluginSettings.model_json_schema()


@plugin
def settings_model():
    return PluginSettings


@plugin
async def load_settings(plugin_id: str, agent_id: str) -> dict:
    """Read the plugin settings from the global ``system:agent`` settings list.

    Async override using the official ``cat.db.crud`` API (the Plugin base now
    awaits async ``load_settings`` overrides); the per-agent ``agent_id`` is
    ignored on purpose: these settings are global for the whole instance.
    """
    setting = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, _MGMT_SETTING_NAME)
    value = (setting or {}).get("value")
    if isinstance(value, dict):
        return value

    return PluginSettings().model_dump()


@plugin
async def save_settings(plugin_id: str, settings: dict, agent_id: str) -> dict:
    """Upsert the plugin settings into the global ``system:agent`` list.

    Async override persisting with the official ``cat.db.crud`` API (same
    storage and the same category semantics as the system-level embedder).
    """
    validated = PluginSettings(**settings).model_dump()
    await crud_settings.upsert_setting_by_name(
        DEFAULT_SYSTEM_KEY,
        Setting(name=_MGMT_SETTING_NAME, value=validated),
    )
    return validated
