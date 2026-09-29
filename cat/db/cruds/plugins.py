from typing import Dict, Any, List
from redis.exceptions import RedisError

from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_AGENTS_KEY, DEFAULT_AGENT_KEY, DEFAULT_SYSTEM_KEY, DEFAULT_PLUGINS_KEY, get_async_db
from cat.log import log


def format_key(agent_id: str, plugin_id: str) -> str:
    """
    Format Redis key for a plugin's settings.

    Args:
        agent_id: ID of the chatbot.
        plugin_id: ID of the plugin.

    Returns:
        Formatted key (e.g., "agents:<agent_id>:plugins:<plugin_id>" or "system:plugins:<plugin_id>" for the system agent)
    """
    return (
        f"{DEFAULT_SYSTEM_KEY}:{DEFAULT_PLUGINS_KEY}:{plugin_id}"
        if agent_id == DEFAULT_SYSTEM_KEY
        else f"{DEFAULT_AGENTS_KEY}:{agent_id}:{DEFAULT_PLUGINS_KEY}:{plugin_id}"
    )


async def get_settings(agent_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve all plugin settings for a specific agent from Redis.

    Args:
        agent_id: ID of the chatbot.
    Returns:
        Dictionary of plugin settings, or empty dict if none found.
    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        all_settings = await crud.read(format_key(agent_id, "*"))
        if not all_settings:
            log.debug(f"No plugin settings found for agent {agent_id}")
            return {}

        if isinstance(all_settings, list):
            all_settings = all_settings[0]

        return all_settings
    except RedisError as e:
        log.error(f"Redis error getting all settings for agent {agent_id}: {e}")
        raise


async def get_setting(agent_id: str, plugin_id: str) -> Dict[str, Any] | None:
    """
    Retrieve plugin settings from Redis.

    Args:
        agent_id: ID of the chatbot.
        plugin_id: ID of the plugin.

    Returns:
        Dictionary of plugin settings, or None if not found.

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        settings = await crud.read(format_key(agent_id, plugin_id))
        if settings is None:
            log.debug(f"No settings found for {agent_id}:{plugin_id}")
            return None

        if isinstance(settings, list):
            settings = settings[0]

        return settings
    except RedisError as e:
        log.error(f"Redis error getting settings for {agent_id}:{plugin_id}: {e}")
        raise


async def set_setting(agent_id: str, plugin_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store plugin settings in Redis.

    Args:
        agent_id: ID of the chatbot.
        plugin_id: ID of the plugin.
        settings: Dictionary of plugin settings.

    Returns:
        Stored settings.

    Raises:
        RedisError: If Redis connection fails.
        ValueError: If settings serialization fails.
    """
    try:
        await crud.store(format_key(agent_id, plugin_id), settings)

        log.debug(f"Stored settings for {agent_id}:{plugin_id}")
        return settings
    except (RedisError, ValueError) as e:
        log.error(f"Error storing settings for {agent_id}:{plugin_id}: {e}")
        raise


async def update_setting(agent_id: str, plugin_id: str, updated_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update plugin settings in Redis atomically.

    Args:
        agent_id: ID of the chatbot.
        plugin_id: ID of the plugin.
        updated_settings: Dictionary with settings to update.

    Returns:
        Updated settings.

    Raises:
        RedisError: If Redis connection fails.
        ValueError: If serialization fails.
    """
    try:
        # read, merge and write under a lock: two updates at the same time (e.g. on two instances) never lose one
        async with crud.distributed_lock(format_key(agent_id, plugin_id)):
            settings_db = await get_setting(agent_id, plugin_id) or {}
            settings_db.update(updated_settings)

            return await set_setting(agent_id, plugin_id, settings_db)
    except (RedisError, ValueError) as e:
        log.error(f"Error updating settings for {agent_id}:{plugin_id}: {e}")
        raise


async def delete_setting(agent_id: str, plugin_id: str):
    """
    Delete plugin settings from Redis.

    Args:
        agent_id: ID of the chatbot.
        plugin_id: ID of the plugin.

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        await crud.delete(format_key(agent_id, plugin_id))
        log.debug(f"Deleted settings for {agent_id}:{plugin_id}")
    except RedisError as e:
        log.error(f"Redis error deleting settings for {agent_id}:{plugin_id}: {e}")
        raise


async def destroy_all(agent_id: str):
    """
    Delete all plugin settings for a specific agent.

    Args:
        agent_id: ID of the chatbot.

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        await crud.destroy(format_key(agent_id, "*"))
        log.debug(f"Destroyed plugin settings for {agent_id}")
    except RedisError as e:
        log.error(f"Redis error destroying settings for {agent_id}: {e}")
        raise


async def destroy_plugin(plugin_id: str):
    """
    Delete all settings for a specific plugin across all agents.

    Args:
        plugin_id: ID of the plugin.

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        await crud.destroy(format_key("*", plugin_id))
        log.debug(f"Destroyed plugin settings for plugin {plugin_id}")
    except RedisError as e:
        log.error(f"Redis error destroying settings for plugin {plugin_id}: {e}")
        raise


async def get_agents_plugin_keys(plugin_id: str) -> List[str]:
    """
    Get all unique agent IDs where the plugin_id is listed in the active_plugins setting.

    Uses a single SCAN on `agents:*:agent` keys followed by one JSON.MGET command to retrieve active_plugins from all
    agents at once, then filters locally.
    Total cost: 1 SCAN + 1 Redis command, regardless of the number of agents.

    Args:
        plugin_id: The name of the plugin to filter by.

    Returns:
        List of unique agent IDs for which the plugin is active.

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        db = get_async_db()

        # Scan directly for agent settings keys — no intermediate reconstruction needed
        keys = [
            k.decode() if isinstance(k, bytes) else k
            async for k in db.scan_iter(f"{DEFAULT_AGENTS_KEY}:*:{DEFAULT_AGENT_KEY}")
        ]
        if not keys:
            return []

        # Single Redis command: read active_plugins entry from every key at once
        results = await db.json().mget(keys, '$[?(@.name=="active_plugins")]')

        active_agents = [
            key.split(":")[1] for key, result in zip(keys, results) if result and plugin_id in (result[0].get("value") or [])
        ]
        return active_agents
    except RedisError as e:
        log.error(f"Redis error in get_agents_plugin_keys for plugin {plugin_id}: {e}")
        raise


async def get_active_plugins_from_db(agent_id: str) -> List[str]:
    """
    Retrieve the list of active plugins for a specific agent from settings.

    Args:
        agent_id: ID of the chatbot.

    Returns:
        List of active plugin IDs.
    """
    active_plugins_from_db = await crud_settings.get_setting_by_name(agent_id, "active_plugins")
    active_plugins: List[str] = [] if active_plugins_from_db is None else active_plugins_from_db["value"]
    return active_plugins
