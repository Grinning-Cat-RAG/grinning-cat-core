from typing import Any, Dict

from pydantic import ValidationError

from cat import endpoint, log
from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.db.cruds import plugins as crud_plugins
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.exceptions import CustomValidationException
from cat.routes.routes_utils import UpsertSettingResponse, GetSettingResponse

from .settings import MGMT_SETTING_NAME


@endpoint.get("/settings", prefix="/mgmt_message", tags=["Management Message"])
async def get_mgmt_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.READ),
) -> GetSettingResponse:
    """Authenticated read of the plugin's global settings, with their schema.

    The counterpart of ``PUT /mgmt_message/settings``: same storage, pinned to
    the system agent, and the value falls back to the settings model defaults
    when nothing has been saved yet. ``GET /plugins/system/settings/mgmt_message``
    returns the same content through the core's generic system-level route.
    """
    plugin = info.lizard.plugin_manager.plugins[MGMT_SETTING_NAME]

    final_settings = await plugin.load_settings(DEFAULT_SYSTEM_KEY)

    return GetSettingResponse(
        name=MGMT_SETTING_NAME, value=final_settings, scheme=plugin.settings_schema()
    )


@endpoint.put("/settings", prefix="/mgmt_message", tags=["Management Message"])
async def put_mgmt_settings(
    payload: Dict[str, Any],
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.WRITE),
) -> UpsertSettingResponse:
    """System-level write of the plugin's global settings (SYSTEM WRITE).

    Same body as the agent-level ``PUT /plugins/settings/{plugin_id}``: validate against the plugin's settings model,
    persist through the plugin's own ``save_settings``. The write is pinned to the system agent, so it lands on
    ``system:plugins:mgmt_message``: these settings are global for the whole instance.

    The plugin owns this route because the agent-level plugin routes cannot reach it: ``mgmt_message`` is in
    ``get_non_toggleable_plugin_ids``, so ``is_plugin_manageable`` rejects it, and the core exposes no system-level
    settings write.
    """
    plugin_manager = info.lizard.plugin_manager
    plugin = plugin_manager.plugins[MGMT_SETTING_NAME]

    try:
        plugin.settings_model().model_validate(payload)
    except ValidationError as e:
        raise CustomValidationException("\n".join(err["msg"] for err in e.errors())) from e

    final_settings = await plugin.save_settings(payload, DEFAULT_SYSTEM_KEY)

    return UpsertSettingResponse(name=MGMT_SETTING_NAME, value=final_settings)


@endpoint.get("/global_message", prefix="/mgmt_message", tags=["Management Message"])
async def get_global_message() -> dict[str, Any]:
    """Public, unauthenticated read of the plugin's global settings.

    Returns the 4-field settings dict stored on the system agent's plugin key
    (``system:plugins:mgmt_message``), the same storage every other system
    plugin uses. No authentication is required so that external consumers
    (e.g. the RITA widget) can show the global banner without holding
    SYSTEM/admin credentials.

    Example: ``{"management_message": "", "management_active": false,
    "global_message": "...", "show_global_msg": true}``

    Note: the read is intentionally limited to the 4 banner fields; the
    authenticated writes go through ``PUT /mgmt_message/settings``.
    """
    try:
        settings = await crud_plugins.get_setting(DEFAULT_SYSTEM_KEY, MGMT_SETTING_NAME)
        return settings if isinstance(settings, dict) else {}
    except Exception as e:  # noqa: BLE001 - endpoint must never 500 on a banner read
        log.error(f"mgmt_message global_message read failed: {e}")
        return {}
