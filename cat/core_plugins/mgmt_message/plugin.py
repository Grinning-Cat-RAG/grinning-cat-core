from cat import hook
from cat.auth.permissions import AuthResource
from cat.db.cruds import plugins as crud_plugins
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.exceptions import ManagementModeException

from .settings import MGMT_SETTING_NAME


@hook(priority=1)
async def auth_request(local_user, agent_id, connection, **kwargs):
    # these settings are global for the whole instance: they live on the system
    # agent's plugin key (system:plugins:mgmt_message), like every other system
    # plugin, so the per-agent agent_id is not used here
    value = await crud_plugins.get_setting(DEFAULT_SYSTEM_KEY, MGMT_SETTING_NAME) or {}
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
