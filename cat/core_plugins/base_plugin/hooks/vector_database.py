"""
Hook to handle operations for vector databases.

This module provides functionality to define hooks that can be triggered after operations on vector databases.
"""

from typing import Dict, Any

from cat import hook, CheshireCat


@hook(priority=0)
def after_vector_database_settings_update(
    vector_database_name: str, previous_config: Dict[str, Any], new_config: Dict[str, Any], cat: CheshireCat
) -> None:
    """
    Hook triggered after vector database settings are updated.

    This function is executed after the vector database settings have been updated to allow any post-update
    operations to be performed.

    Args:
        vector_database_name: str
            The name of the vector database whose settings were updated.
        previous_config: Dict[str, Any]
            The previous vector database settings.
        new_config: Dict[str, Any]
            The updated vector database settings.
        cat: CheshireCat
            A contextual object or dependency required for post-update processing.
    """
    pass