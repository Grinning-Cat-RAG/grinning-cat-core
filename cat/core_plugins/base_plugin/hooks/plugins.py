"""
Hook to handle operations for plugins.

This module provides functionality to define hooks that can be triggered after operations on plugins.
"""

from typing import Dict, Any

from cat import hook, BillTheLizard, CheshireCat


@hook(priority=0)
def lizard_notify_plugin_installation(plugin_id: str, plugin_path: str, lizard: BillTheLizard) -> None:
    """
    Handles the notification process when a plugin is installed.

    This function is triggered when a plugin is installed and a hook is activated. The notification ensures that
    relevant stakeholders, systems, or components are informed about the installation event.

    Args:
        plugin_id: The ID of the plugin that is being installed.
        plugin_path: The path to the plugin's installation directory.
        lizard: The specific category or context associated with the installed plugin.
    """
    pass


@hook(priority=0)
def lizard_notify_plugin_uninstallation(plugin_id: str, lizard: BillTheLizard) -> None:
    """
    Handles the notification process when a plugin is installed.

    This function is triggered when a plugin is installed and a hook is de-activated. The notification ensures that
    relevant stakeholders, systems, or components are informed about the uninstallation event.

    Args:
        plugin_id: The ID of the plugin that is being uninstalled.
        lizard: The specific category or context associated with the uninstalled plugin.
    """
    pass


@hook(priority=0)
def after_plugin_settings_update(plugin_id: str, settings: Dict[str, Any], cat) -> None:
    """
    Hook triggered after plugin settings are updated.

    This function is executed after the plugin's settings have been updated to allow any post-update operations to be
    performed.

    Args:
        plugin_id: str
            The unique identifier of the plugin whose settings were updated.
        settings: Dict[str, Any]
            The updated plugin settings.
        cat:
            A contextual object or dependency required for post-update processing.
    """
    pass


@hook(priority=0)
def after_plugin_toggling_on_system(plugin_id: str, lizard: BillTheLizard) -> None:
    """
    Hook that gets executed immediately after toggling a plugin on a system level.

    It runs only on the instance (POD) that received the toggle: the other PODs align their plugins with the
    database (March Hare) without running it. For a per-POD reaction, use the `activated` / `deactivated` overrides of
    `@plugin`, which run on every POD.

    Args:
        plugin_id (str): The unique identifier of the plugin being toggled.
        lizard: The specific category or context associated with the toggled plugin.

    Returns:
        None
    """
    pass


@hook(priority=0)
def after_plugin_toggling_on_agent(plugin_id: str, cat: CheshireCat) -> None:
    """
    Hook that gets executed immediately after toggling a plugin on an agent level.

    Args:
        plugin_id (str): The unique identifier of the plugin being toggled.
        cat: The CheshireCat instance associated with the agent where the plugin is toggled.

    Returns:
        None
    """
    pass
