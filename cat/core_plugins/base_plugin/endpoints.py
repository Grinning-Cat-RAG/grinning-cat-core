from typing import List

from cat import AuthResource, AuthPermission, AuthorizedInfo, check_permissions, endpoint


@endpoint.get("/", prefix="/admins/core_plugins", tags=["Admins - Plugins"])
async def get_core_plugins(
    info: AuthorizedInfo = check_permissions(AuthResource.PLUGIN, AuthPermission.READ),
) -> List[str]:
    """Get list of available core plugins"""
    return info.lizard.plugin_manager.get_core_plugins_ids


@endpoint.get("/", prefix="/admins/core_plugins/untoggling", tags=["Admins - Plugins"])
async def get_core_untoggling_plugins(
    info: AuthorizedInfo = check_permissions(AuthResource.PLUGIN, AuthPermission.READ),
) -> List[str]:
    """Get list of available core plugins which cannot be deactivated"""
    return info.lizard.plugin_manager.get_non_toggleable_plugin_ids
