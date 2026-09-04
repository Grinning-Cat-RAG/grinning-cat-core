from typing import Any

from pydantic import ValidationError

from cat import endpoint, log
from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.core_plugins.mgmt_message.settings import (
    _MGMT_SETTING_CATEGORY,
    _MGMT_SETTING_NAME,
    PluginSettings,
    _drop_legacy_key,
)
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.db.models import Setting
from cat.exceptions import CustomValidationException


def _validated_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return PluginSettings(**payload).model_dump()
    except ValidationError as e:
        raise CustomValidationException("\n".join(err["msg"] for err in e.errors())) from e


@endpoint.get("/settings", prefix="/mgmt_message", tags=["Management Message"])
async def get_mgmt_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.READ),
) -> dict[str, Any]:
    """System-level read of the plugin's 4 global settings (SYSTEM READ).

    Same storage and shape as the old core ``GET /plugins/system/settings``
    route: ``{name, value, scheme}`` inside the global ``system:agent`` list.
    """
    await _drop_legacy_key()
    setting = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, _MGMT_SETTING_NAME)
    value = (setting or {}).get("value")
    return {
        "name": _MGMT_SETTING_NAME,
        "value": value if isinstance(value, dict) else {},
        "scheme": PluginSettings.model_json_schema(),
    }


@endpoint.put("/settings", prefix="/mgmt_message", tags=["Management Message"])
async def put_mgmt_settings(
    payload: dict[str, Any],
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.WRITE),
) -> dict[str, Any]:
    """System-level write of the plugin's global settings (SYSTEM WRITE).

    Same storage as the old core ``PUT /plugins/system/settings/mgmt_message``
    (upsert inside the global ``system:agent`` list, embedder pattern).
    """
    validated = _validated_payload(payload)
    await crud_settings.upsert_setting_by_name(
        DEFAULT_SYSTEM_KEY,
        Setting(name=_MGMT_SETTING_NAME, value=validated, category=_MGMT_SETTING_CATEGORY),
    )
    await _drop_legacy_key()
    return {"name": _MGMT_SETTING_NAME, "value": validated}


@endpoint.get("/global_message", prefix="/mgmt_message", tags=["Management Message"])
async def get_global_message() -> dict[str, Any]:
    """Public, unauthenticated read of the plugin's global settings.

    Returns the 4-field settings dict stored inside the global ``system:agent``
    settings list under the ``mgmt_message`` entry — the same storage and the
    same ``crud_settings`` interface used for the system-level embedder
    configuration. No authentication is required so that external consumers
    (e.g. the RITA widget) can show the global banner without holding
    SYSTEM/admin credentials.

    Example: ``{"management_message": "", "management_active": false,
    "global_message": "...", "show_global_msg": true}``

    Note: the read is intentionally limited to the 4 banner fields; the
    authenticated writes go through the plugin's own
    ``PUT /mgmt_message/settings`` route (embedder pattern).
    """
    try:
        setting = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, _MGMT_SETTING_NAME)
        value = (setting or {}).get("value")
        return value if isinstance(value, dict) else {}
    except Exception as e:  # noqa: BLE001 - endpoint must never 500 on a banner read
        log.error(f"mgmt_message global_message read failed: {e}")
        return {}
