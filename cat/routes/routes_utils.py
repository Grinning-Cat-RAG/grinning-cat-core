import asyncio
import json
from ast import literal_eval
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Type
from fastapi import Query, BackgroundTasks
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from pydantic import BaseModel

from cat import utils
from cat.auth.permissions import AuthPermission, AuthResource
from cat.db.database import get_async_db
from cat.env import get_env_float
from cat.exceptions import CustomValidationException, CustomUnauthorizedException
from cat.looking_glass.mad_hatter.mad_hatter import MadHatter
from cat.looking_glass.mad_hatter.plugin import Plugin
from cat.looking_glass.mad_hatter.registry import PluginRegistry
from cat.looking_glass.models import PluginManifest
from cat.services.redis_search import RedisSearchService
from cat.utils import SUFFIX_TO_CRYPT, safe_deepcopy


class Plugins(BaseModel):
    installed: List[PluginManifest]
    registry: List[PluginManifest]


class UserCredentials(BaseModel):
    username: str
    password: str


class JWTResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class GetAvailablePluginsFilter(BaseModel):
    query: str | None


class GetAvailablePluginsResponse(Plugins):
    filters: GetAvailablePluginsFilter


class TogglePluginResponse(BaseModel):
    info: str


class InstallPluginResponse(TogglePluginResponse):
    filename: str
    content_type: str


class InstallPluginFromRegistryResponse(TogglePluginResponse):
    url: str


class UpsertSettingResponse(BaseModel):
    name: str
    value: Dict


class GetSettingResponse(UpsertSettingResponse):
    scheme: Dict[str, Any] | None = None


class GetSettingsResponse(BaseModel):
    settings: List[GetSettingResponse]
    selected_configuration: str | None


class PluginsSettingsResponse(BaseModel):
    settings: List[GetSettingResponse]


def has_write_permission(permissions: Dict[str, List[str]] | None, resource: AuthResource) -> bool:
    """True if the caller may WRITE the given resource (and can therefore see real secret values)."""
    if not permissions:
        return False
    return str(AuthPermission.WRITE) in [str(p) for p in permissions.get(str(resource), [])]


def mask_secret_values(value: Any, reveal: bool) -> Any:
    """Return ``value`` as-is for settings writers; otherwise mask secret-bearing string values.

    A value is masked when its key ends with a known secret suffix (``SUFFIX_TO_CRYPT``,
    the same rule used to encrypt values at rest) and the value is a non-empty string.
    This mirrors the upstream behaviour of hiding ``*_key``/``*_secret``/``*_password``
    values, but only towards callers that lack WRITE permission on the settings resource,
    so read-modify-write clients (SDK, ADMIN, RITA) keep the real values.
    """
    if reveal or not isinstance(value, dict):
        return value
    return {
        k: "********" if isinstance(v, str) and v and any(suffix in k for suffix in SUFFIX_TO_CRYPT) else v
        for k, v in value.items()
    }


class GetPluginDetailsResponse(BaseModel):
    data: PluginManifest


class DeletePluginResponse(BaseModel):
    deleted: str


def create_plugin_manifest(
    plugin: Plugin,
    active_plugins: List[str],
    registry_plugins_index: Dict[str, PluginManifest] | None = None,
    query: str | None = None
) -> PluginManifest:
    # get manifest
    manifest: PluginManifest = safe_deepcopy(plugin.manifest)  # we make a copy to avoid modifying the plugin obj
    manifest.local_info = {
        "active": (plugin.id in active_plugins),  # pass along if plugin is active or not
        "hooks": [{"name": hook.name, "priority": hook.priority} for hook in plugin.hooks],
        "tools": [{"name": tool.name} for tool in plugin.tools],
        "forms": [{"name": form.name} for form in plugin.forms],
        "mcp_clients": [{"name": mcp_client.name} for mcp_client in plugin.mcp_clients],
        "endpoints": [{"name": endpoint.name, "tags": endpoint.tags} for endpoint in plugin.endpoints],
    }

    if registry_plugins_index is not None:
        manifest.local_info["upgrade"] = None

        # do not show already installed plugins among registry plugins
        r = registry_plugins_index.pop(manifest.plugin_url, None)
        # filter by query
        plugin_text = manifest.model_dump_json()
        if (
                (query is None or query.lower() in plugin_text)
                and r is not None
                and r.version is not None
                and r.version != plugin.manifest.version
        ):
            manifest.local_info["upgrade"] = r.version

    return manifest


async def get_available_plugins(
    plugin_registry: PluginRegistry,
    plugin_manager: MadHatter,
    query: str = None,
    # author: str = None, to be activated in case of more granular search
    # tag: str = None, to be activated in case of more granular search
) -> GetAvailablePluginsResponse:
    """
    Get the plugins related to the passed plugin manager instance and the query.

    Args:
        plugin_registry: the instance of PluginRegistry
        plugin_manager: the instance of MadHatter
        query: the query to look for

    Returns:
        The list of plugins
    """
    # retrieve plugins from official repo
    registry_plugins = await plugin_registry.search_plugins(query)
    # index registry plugins by url
    registry_plugins_index = {p.plugin_url: p for p in registry_plugins if p.plugin_url is not None}

    # get active plugins
    active_plugins_ids = await plugin_manager.load_active_plugins_ids_from_db()

    # list installed plugins' manifest
    installed_plugins = [
        create_plugin_manifest(p, active_plugins_ids, registry_plugins_index, query)
        for p in (await plugin_manager.available_plugins()).values()
    ]

    return GetAvailablePluginsResponse(
        filters=GetAvailablePluginsFilter(
            query=query,
            # "author": author, to be activated in case of more granular search
            # "tag": tag, to be activated in case of more granular search
        ),
        installed=installed_plugins,
        registry=list(registry_plugins_index.values()),
    )


async def get_plugins_settings(plugin_manager: MadHatter, agent_id: str, reveal: bool = True) -> PluginsSettingsResponse:
    settings = []

    # plugins are managed by the MadHatter class (and its inherits)
    for plugin in plugin_manager.plugins.values():
        try:
            plugin_settings = await plugin.load_settings(agent_id)
            plugin_schema = plugin.settings_schema()
            if plugin_schema["properties"] == {}:
                plugin_schema = {}
            settings.append(
                GetSettingResponse(
                    name=plugin.id,
                    value=mask_secret_values(plugin_settings, reveal),
                    scheme=plugin_schema,
                )
            )
        except Exception as e:
            raise CustomValidationException(
                f"Error loading {plugin} settings. The result will not contain the settings for this plugin. "
                f"Error details: {e}"
            )

    return PluginsSettingsResponse(settings=settings)


async def get_plugin_settings(plugin_manager: MadHatter, plugin_id: str, agent_id: str, reveal: bool = True) -> GetSettingResponse:
    """Returns the settings of a specific plugin"""
    settings = await plugin_manager.plugins[plugin_id].load_settings(agent_id)
    scheme = plugin_manager.plugins[plugin_id].settings_schema()

    if scheme["properties"] == {}:
        scheme = {}

    return GetSettingResponse(name=plugin_id, value=mask_secret_values(settings, reveal), scheme=scheme)


def create_dict_parser(param_name: str, description: str | None = None):
    def parser(
        param_value: str | None = Query(
            default=None,
            alias=param_name,
            description=description or f"{param_name} JSON filter."
        )
    ) -> Dict[str, Any]:
        if not param_value:
            return {}
        try:
            return literal_eval(param_value)
        except ValueError:
            return {}
    return parser


async def startup_app(app):
    from cat.looking_glass import BillTheLizard

    set_llm_cache(InMemoryCache())
    utils.pod_id()

    bill_the_lizard = BillTheLizard()
    bill_the_lizard.fastapi_app = app
    await bill_the_lizard.bootstrap()
    app.state.lizard = bill_the_lizard


async def shutdown_app(app):
    utils.singleton.instances.clear()

    # shutdown Manager
    await app.state.lizard.shutdown()
    del app.state.lizard

    await get_async_db().aclose()


def validate_permissions(permissions: Dict[str, List[str]], resources: Type[utils.Enum]):
    if not permissions:
        raise ValueError("Permissions cannot be empty")

    # Check if all permissions are empty
    all_items_empty = all([not p for p in permissions.values()])
    if all_items_empty:
        raise ValueError("At least one permission must be set")

    # Validate each resource and its permissions
    for k_, v_ in permissions.items():
        if k_ not in resources:
            raise ValueError(f"Invalid resource: {k_}")
        if any([p not in AuthPermission for p in v_]):
            raise ValueError(f"Invalid permissions for {k_}")

    return permissions


async def create_jwt_content(credentials: UserCredentials, redis_search_service: RedisSearchService) -> Dict[str, Any]:
    username = credentials.username
    password = credentials.password

    # search for user across all agents — async so the Lua script doesn't block the event loop
    valid_matches = await redis_search_service.search_user_by_credentials(username, password)
    if not valid_matches:
        # Invalid username or password
        # wait a little to avoid brute force attacks
        await asyncio.sleep(1)
        raise CustomUnauthorizedException("Invalid Credentials")

    final_valid_matches = []
    for valid_match in valid_matches:
        valid_match_json = json.loads(valid_match)
        # remove sensitive info
        if valid_match_json.get("user", {}).get("password"):
            del valid_match_json["user"]["password"]
        final_valid_matches.append(json.dumps(valid_match_json))

    # using seconds for easier testing
    expire_delta_in_seconds = get_env_float("CAT_JWT_EXPIRE_MINUTES") * 60  # type: ignore[operator]
    now = datetime.now(timezone.utc)

    expires = now + timedelta(seconds=expire_delta_in_seconds)

    return {
        "sub": username,  # Subject (the Username)
        "exp": expires,  # Expiry date as a Unix timestamp
        "iat": now,
        "agents": final_valid_matches,
    }

def sanitize_source_name(source_name: str, path: str) -> str:
    # Security: Validate and sanitize the source parameter
    if not source_name or (source_name := source_name.strip()) == "":
        raise CustomValidationException("Invalid filename")

    # map Windows paths to Unix
    source_name = source_name.replace('\\','/')

    # Validate path characters (allow only alphanumeric, dash, underscore, dot, slash)
    if not source_name.replace('.', '').replace('-', '').replace('_', '').replace('/','').isalnum():
        raise CustomValidationException("Filename contains invalid characters")

    source_path = Path(source_name)
    # Prevent hidden files and dirs and ensure reasonable length
    if any(part.startswith('.') for part in source_path.parts) or len(source_name) > 255:
        raise CustomValidationException("Invalid filename")

    # Additional validation: reject suspicious patterns ('\\', '.' and '..' handled above)
    forbidden_patterns = ['\x00']
    if any(pattern in piece for pattern in forbidden_patterns for piece in source_path.parts):
        raise CustomValidationException("Invalid filename")

    # Optional: Additional path validation using pathlib for extra safety
    try:
        # This ensures the resolved path doesn't escape the intended directory
        base_path = Path(path).resolve()
        requested_path = (base_path / source_path).resolve()

        # Ensure the resolved path is within the base directory
        if not requested_path.is_relative_to(base_path):
            raise CustomValidationException("Access denied")
    except (OSError, ValueError):
        raise CustomValidationException("Invalid file path")

    return str(source_path)


def run_background_task(background_tasks: BackgroundTasks, func, *args, **kwargs):
    background_tasks.add_task(func, *args, **kwargs)
