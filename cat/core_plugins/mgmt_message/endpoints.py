from typing import Any, Dict

from cat.routes.routes_utils import UpsertSettingResponse
from pydantic import ValidationError

from cat import endpoint, log
from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.db.models import Setting
from cat.exceptions import CustomValidationException

from .settings import _MGMT_SETTING_NAME, PluginSettings, drop_agent_keys


def _validated_payload(payload: dict[str, Any]) -> Dict[str, Any]:
    try:
        return PluginSettings(**payload).model_dump()
    except ValidationError as e:
        raise CustomValidationException("\n".join(err["msg"] for err in e.errors())) from e


@endpoint.put("/settings", prefix="/mgmt_message", tags=["Management Message"])
async def put_mgmt_settings(
    payload: Dict[str, Any],
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.WRITE),
) -> UpsertSettingResponse:
    """System-level write of the plugin's global settings (SYSTEM WRITE).

    Same storage as the old core ``PUT /plugins/system/settings/mgmt_message``
    (upsert inside the global ``system:agent`` list, embedder pattern).
    """
    validated = _validated_payload(payload)
    await crud_settings.upsert_setting_by_name(
        DEFAULT_SYSTEM_KEY,
        Setting(name=_MGMT_SETTING_NAME, value=validated),
    )
    await drop_agent_keys()
    return UpsertSettingResponse(name=_MGMT_SETTING_NAME, value=validated)


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
