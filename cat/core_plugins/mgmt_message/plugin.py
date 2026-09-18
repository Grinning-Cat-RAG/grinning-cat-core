from cat import hook
from cat.db.cruds import plugins as crud_plugins
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.exceptions import ManagementModeException

from .settings import MGMT_SETTING_NAME


def _own_endpoint_paths(lizard) -> set:
    """The paths this plugin registers, read from the plugin itself.

    Taking them from the loaded plugin instead of hardcoding them means an
    endpoint added to ``endpoints.py`` is reachable in management mode without
    touching this hook.
    """
    plugin = lizard.plugin_manager.plugins.get(MGMT_SETTING_NAME) if lizard else None
    return {endpoint.name.rstrip("/") for endpoint in plugin.endpoints} if plugin else set()


@hook(priority=1)
async def auth_request(local_user, agent_id, connection, **kwargs):
    # these settings are global for the whole instance: they live on the system
    # agent's plugin key (system:plugins:mgmt_message), like every other system
    # plugin, so the per-agent agent_id is not used here
    value = await crud_plugins.get_setting(DEFAULT_SYSTEM_KEY, MGMT_SETTING_NAME) or {}
    if not value.get("management_active", False):
        return  # allow

    # In management mode the instance behaves as if it had only this plugin:
    # its own endpoints answer, everything else is gone. They stay open to every
    # principal (they carry their own SYSTEM WRITE check) because they are the
    # only way to read the management message and to switch the mode back off.
    if connection is not None and connection.url.path.rstrip("/") in _own_endpoint_paths(kwargs.get("lizard")):
        return  # allow

    message = value.get("management_message", "Not Found")
    # a dedicated exception: a 404 like any missing route, but recognisable by
    # clients and logged at INFO level instead of ERROR
    raise ManagementModeException(message)
