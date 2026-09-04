from pydantic import BaseModel

from cat import log, plugin
from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.db.models import Setting

# entry name under which this plugin persists its settings in the global
# ``system:agent`` settings list, exactly like the system-level embedder
# configuration (same ``{name, value, category, setting_id, updated_at}`` shape)
_MGMT_SETTING_NAME = "mgmt_message"
_MGMT_SETTING_CATEGORY = "mgmt_message"

# key written by the previous default-plugin-settings mechanism (and by the
# core activate_settings), which this plugin replaces with system:agent storage
_LEGACY_KEY = f"{DEFAULT_SYSTEM_KEY}:plugins:{_MGMT_SETTING_NAME}"


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


async def _read_legacy() -> dict | None:
    """Read the legacy key via the official crud API (read may wrap in a list)."""
    try:
        raw = await crud.read(_LEGACY_KEY)
    except Exception as e:  # noqa: BLE001 - cleanup/migration must never break settings I/O
        log.error(f"mgmt_message legacy key read failed: {e}")
        return None

    # ``crud.read`` returns the document wrapped in a list; normalize to the dict
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    return raw if isinstance(raw, dict) else None


async def _drop_legacy_key() -> bool:
    """Best-effort removal of the leftover ``system:plugins:mgmt_message`` key.

    The core writes that key at plugin activation (``activate_settings``); this
    plugin's single source of truth is the ``system:agent`` settings list, so
    the leftover is dropped whenever the settings are read or written.
    """
    if await _read_legacy() is None:
        return False

    try:
        await crud.delete(_LEGACY_KEY)
    except Exception as e:  # noqa: BLE001
        log.error(f"mgmt_message legacy key delete failed: {e}")
    return True


async def _migrate_legacy() -> dict | None:
    """One-off migration: move the old ``system:plugins:mgmt_message`` dict
    into the ``system:agent`` settings list, via the official crud API."""
    legacy = await _read_legacy()
    if legacy is None:
        return None

    try:
        validated = PluginSettings(**legacy).model_dump()
        await crud_settings.upsert_setting_by_name(
            DEFAULT_SYSTEM_KEY,
            Setting(name=_MGMT_SETTING_NAME, value=validated, category=_MGMT_SETTING_CATEGORY),
        )
        await crud.delete(_LEGACY_KEY)
        return validated
    except Exception as e:  # noqa: BLE001 - migration must never block the settings load
        log.error(f"mgmt_message legacy migration failed: {e}")
        return None


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
        # self-heal the activation leftover while we are here
        await _drop_legacy_key()
        return value

    # not stored: one-off migration of the legacy key, otherwise model defaults
    migrated = await _migrate_legacy()
    if migrated is not None:
        return migrated
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
        Setting(name=_MGMT_SETTING_NAME, value=validated, category=_MGMT_SETTING_CATEGORY),
    )
    # no longer used: drop the leftover key if it still exists
    await _drop_legacy_key()
    return validated
