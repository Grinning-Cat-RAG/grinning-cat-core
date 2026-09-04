from cat import hook
from cat.auth.permissions import AuthResource
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.exceptions import ManagementModeException


@hook(priority=1)
async def auth_request(local_user, agent_id, connection, **kwargs):
    # read global settings from the system:agent settings list (same mechanism
    # as the system-level embedder configuration)
    setting = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, "mgmt_message")
    value = (setting or {}).get("value") or {}
    if not value.get("management_active", False):
        return  # allow

    # allowed iff the principal has SYSTEM permission (admin system user or valid API-KEY)
    permissions = getattr(local_user, "permissions", None) or {}
    if str(AuthResource.SYSTEM) in permissions:
        return  # allow

    message = value.get("management_message", "Access denied")
    # a dedicated exception: keeps the 403 semantics for clients, lets the core
    # log this at INFO level and lets clients distinguish this gate from a
    # generic permission error
    raise ManagementModeException(message)
