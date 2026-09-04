import mimetypes
from typing import Dict
from fastapi import Body, APIRouter, UploadFile, BackgroundTasks
from pydantic import ValidationError
from slugify import slugify

from cat.log import log
from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.db.cruds import plugins as crud_plugins
from cat.exceptions import CustomValidationException, CustomNotFoundException
from cat.routes.routes_utils import (
    GetAvailablePluginsResponse,
    GetSettingResponse,
    PluginsSettingsResponse,
    TogglePluginResponse,
    get_available_plugins,
    get_plugins_settings,
    get_plugin_settings,
    has_write_permission,
    InstallPluginResponse,
    GetPluginDetailsResponse,
    DeletePluginResponse,
    InstallPluginFromRegistryResponse,
    create_plugin_manifest,
    run_background_task,
)
from cat.utils import get_allowed_plugins_mime_types, write_temp_file

router = APIRouter(tags=["Plugins"], prefix="/plugins")


# GET plugins
@router.get("/", response_model=GetAvailablePluginsResponse)
async def get_cheshirecat_available_plugins(
    query: str = None,  # type: ignore[assignment]
    info: AuthorizedInfo = check_permissions(AuthResource.PLUGIN, AuthPermission.READ),
    # author: str = None, to be activated in case of more granular search
    # tag: str = None, to be activated in case of more granular search
) -> GetAvailablePluginsResponse:
    """List available plugins"""
    if query is not None:
        query = slugify(query, separator="_")

    return await get_available_plugins(info.lizard.plugin_registry, info.cheshire_cat.plugin_manager, query)  # type: ignore[union-attr]


@router.put("/toggle/{plugin_id}", response_model=TogglePluginResponse)
async def toggle_plugin_cheshirecat(
    plugin_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.PLUGIN, AuthPermission.WRITE),
) -> TogglePluginResponse:
    """Enable or disable a single plugin"""
    plugin_id = slugify(plugin_id, separator="_")

    # access cat instance
    ccat = info.cheshire_cat

    # check if plugin exists
    if not ccat.plugin_manager.plugin_exists(plugin_id):  # type: ignore[union-attr]
        raise CustomNotFoundException("Plugin not found")

    # toggle plugin
    await ccat.toggle_plugin(plugin_id)  # type: ignore[union-attr]
    return TogglePluginResponse(info=f"Plugin {plugin_id} toggled")


@router.get("/settings", response_model=PluginsSettingsResponse)
async def get_cheshirecat_plugins_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.PLUGIN, AuthPermission.READ),
) -> PluginsSettingsResponse:
    """Returns the settings of all the plugins"""
    ccat = info.cheshire_cat
    reveal = has_write_permission(info.user.permissions, AuthResource.PLUGIN)
    return await get_plugins_settings(ccat.plugin_manager, ccat.agent_key, reveal=reveal)  # type: ignore[union-attr]


@router.get("/settings/{plugin_id}", response_model=GetSettingResponse)
async def get_cheshirecat_plugin_settings(
    plugin_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.PLUGIN, AuthPermission.READ),
) -> GetSettingResponse:
    """Returns the settings of a specific plugin"""
    plugin_id = slugify(plugin_id, separator="_")

    ccat = info.cheshire_cat
    if not ccat.plugin_exists(plugin_id):  # type: ignore[union-attr]
        raise CustomNotFoundException("Plugin not found")

    reveal = has_write_permission(info.user.permissions, AuthResource.PLUGIN)
    return await get_plugin_settings(ccat.plugin_manager, plugin_id, ccat.agent_key, reveal=reveal)  # type: ignore[union-attr]


@router.put("/settings/{plugin_id}", response_model=GetSettingResponse)
async def upsert_cheshirecat_plugin_settings(
    plugin_id: str,
    payload: Dict = Body({"setting_a": "some value", "setting_b": "another value"}),
    info: AuthorizedInfo = check_permissions(AuthResource.PLUGIN, AuthPermission.WRITE),
) -> GetSettingResponse:
    """Updates the settings of a specific plugin"""
    plugin_id = slugify(plugin_id, separator="_")

    # access cat instance
    ccat = info.cheshire_cat
    if not ccat.plugin_exists(plugin_id):  # type: ignore[union-attr]
        raise CustomNotFoundException("Plugin not found")

    # Get the plugin object
    plugin = ccat.plugin_manager.plugins[plugin_id]  # type: ignore[union-attr]
    try:
        # Load the plugin settings Pydantic model, and validate the settings
        plugin.settings_model().model_validate(payload)
    except ValidationError as e:
        raise CustomValidationException("\n".join(list(map(lambda x: x["msg"], e.errors()))))

    final_settings = await plugin.save_settings(payload, ccat.agent_key)  # type: ignore[union-attr]
    await ccat.plugin_manager.execute_hook("after_plugin_settings_update", plugin_id, final_settings, caller=ccat)  # type: ignore[union-attr]

    return GetSettingResponse(name=plugin_id, value=final_settings)


@router.post("/settings/{plugin_id}", response_model=GetSettingResponse)
async def reset_cheshirecat_plugin_settings(
    plugin_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.PLUGIN, AuthPermission.WRITE),
) -> GetSettingResponse:
    """Resets the settings of a specific plugin"""
    plugin_id = slugify(plugin_id, separator="_")

    # Get the factory settings of the plugin
    factory_settings = await crud_plugins.get_setting(info.lizard.agent_key, plugin_id)   # type: ignore[arg-type]
    if factory_settings is None:
        raise CustomNotFoundException("Plugin not found.")

    # access cat instance
    ccat = info.cheshire_cat
    if not ccat.plugin_exists(plugin_id):  # type: ignore[union-attr]
        raise CustomNotFoundException("Plugin not found")

    await crud_plugins.set_setting(ccat.agent_key, plugin_id, factory_settings)  # type: ignore[union-attr]
    await ccat.plugin_manager.execute_hook("after_plugin_settings_update", plugin_id, factory_settings, caller=ccat)  # type: ignore[union-attr]

    return GetSettingResponse(name=plugin_id, value=factory_settings)


@router.get("/installed", response_model=GetAvailablePluginsResponse)
async def get_lizard_available_plugins(
    query: str = None,  # type: ignore[assignment]
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.READ),
    # author: str = None, to be activated in case of more granular search
    # tag: str = None, to be activated in case of more granular search
) -> GetAvailablePluginsResponse:
    """List available plugins"""
    if query is not None:
        query = slugify(query, separator="_")

    return await get_available_plugins(info.lizard.plugin_registry, info.lizard.plugin_manager, query)


@router.post("/install/upload", response_model=InstallPluginResponse)
async def install_plugin(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.WRITE),
) -> InstallPluginResponse:
    """Install a new plugin from a zip file"""
    allowed_mime_types = get_allowed_plugins_mime_types()

    content_type, _ = mimetypes.guess_type(file.filename)   # type: ignore[arg-type]
    if content_type not in allowed_mime_types:
        raise CustomValidationException(
            f'MIME type `{file.content_type}` not supported. Admitted types: {", ".join(allowed_mime_types)}'
        )

    filename = file.filename
    log.info(f"Uploading {content_type} plugin {filename}")
    try:
        plugin_archive_path = await write_temp_file(filename, await file.read())  # type: ignore[union-attr]
        run_background_task(background_tasks, info.lizard.install_plugin, plugin_archive_path)
    except Exception as e:
        raise CustomValidationException(f"Could not install plugin from file: {e}")

    return InstallPluginResponse(
        filename=filename,  # type: ignore[arg-type]
        content_type=content_type,  # type: ignore[arg-type]
        info="Plugin is being installed asynchronously",
    )

@router.post("/install/registry", response_model=InstallPluginFromRegistryResponse)
async def install_plugin_from_registry(
    background_tasks: BackgroundTasks,
    payload: Dict = Body({"url": "https://github.com/plugin-dev-account/plugin-repo"}),
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.WRITE),
) -> InstallPluginFromRegistryResponse:
    """Install a new plugin from registry"""
    # download zip from registry
    try:
        tmp_plugin_path = await info.lizard.plugin_registry.download_plugin(payload["url"])
        run_background_task(background_tasks, info.lizard.install_plugin, tmp_plugin_path)
    except Exception as e:
        raise CustomValidationException(f"Could not install plugin from registry: {e}")

    return InstallPluginFromRegistryResponse(url=payload["url"], info="Plugin is being installed asynchronously")


@router.get("/system/settings", response_model=PluginsSettingsResponse)
async def get_lizard_plugins_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.READ),
) -> PluginsSettingsResponse:
    """Returns the default settings of all the plugins"""
    lizard = info.lizard
    reveal = has_write_permission(info.user.permissions, AuthResource.SYSTEM)
    return await get_plugins_settings(lizard.plugin_manager, lizard.agent_key, reveal=reveal)   # type: ignore[arg-type]


@router.get("/system/settings/{plugin_id}", response_model=GetSettingResponse)
async def get_lizard_plugin_settings(
    plugin_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.READ),
) -> GetSettingResponse:
    """Returns the default settings of a specific plugin"""
    lizard = info.lizard
    plugin_manager = lizard.plugin_manager
    if not plugin_manager.plugin_exists(plugin_id):
        raise CustomNotFoundException("Plugin not found")

    reveal = has_write_permission(info.user.permissions, AuthResource.SYSTEM)
    return await get_plugin_settings(plugin_manager, plugin_id, lizard.agent_key, reveal=reveal)   # type: ignore[arg-type]


@router.get("/system/details/{plugin_id}", response_model=GetPluginDetailsResponse)
async def get_lizard_plugin_details(
    plugin_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.READ),
) -> GetPluginDetailsResponse:
    """Returns information on a single plugin, at a system level"""
    plugin_manager = info.lizard.plugin_manager
    plugin_id = slugify(plugin_id, separator="_")

    if not plugin_manager.plugin_exists(plugin_id):
        raise CustomNotFoundException("Plugin not found")

    active_plugins = await plugin_manager.load_active_plugins_ids_from_db()
    plugin = plugin_manager.plugins[plugin_id]

    # get manifest and active True/False. We make a copy to avoid modifying the original obj
    return GetPluginDetailsResponse(data=create_plugin_manifest(plugin, active_plugins))


@router.delete("/uninstall/{plugin_id}", response_model=DeletePluginResponse)
async def uninstall_plugin(
    plugin_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.DELETE),
) -> DeletePluginResponse:
    """Physically remove plugin at a system level."""
    plugin_manager = info.lizard.plugin_manager
    plugin_id = slugify(plugin_id, separator="_")

    if not plugin_manager.plugin_exists(plugin_id):
        raise CustomNotFoundException("Plugin not found")

    try:
        # remove folder, hooks and tools
        await info.lizard.uninstall_plugin(plugin_id)
    except Exception as e:
        raise CustomValidationException(f"Could not uninstall plugin: {e}")

    return DeletePluginResponse(deleted=plugin_id)


@router.put("/system/toggle/{plugin_id}", response_model=TogglePluginResponse)
async def toggle_plugin_admin(
    plugin_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.SYSTEM, AuthPermission.DELETE),
) -> TogglePluginResponse:
    """Enable or disable a single plugin"""
    plugin_manager = info.lizard.plugin_manager
    plugin_id = slugify(plugin_id, separator="_")

    # check if plugin exists
    if not plugin_manager.plugin_exists(plugin_id):
        raise CustomNotFoundException("Plugin not found")

    # toggle plugin
    await info.lizard.toggle_plugin(plugin_id)
    return TogglePluginResponse(info=f"Plugin {plugin_id} toggled")
