import json
from pathlib import Path
from typing import Callable, List, Dict, Any
from unittest.mock import patch
import pytest

from cat.looking_glass.mad_hatter.mad_hatter import MadHatter


@pytest.fixture
def mock_plugins_env(tmp_path: Path):
    """
    Isolated Pytest fixture that creates a temporary file system structure for plugins.

    Structure generated:
    tmp_path/
      ├── core_plugins/
      │     └── base_plugin/
      └── custom_plugins/
            ├── plugin_a/ (no dependencies)
            ├── plugin_b/ (depends on plugin_a)
            └── plugin_c/ (depends on plugin_b)

    Returns:
        Dict containing paths and a helper function to dynamically construct custom plugins.
    """
    core_plugins_dir = tmp_path / "core_plugins"
    custom_plugins_dir = tmp_path / "custom_plugins"
    core_plugins_dir.mkdir(parents=True, exist_ok=True)
    custom_plugins_dir.mkdir(parents=True, exist_ok=True)

    def create_plugin(
        base_dir: Path, plugin_id: str, dependencies: List[str] = None, priority: int = 1
    ) -> Path:
        """
        Helper function to create a plugin directory with a manifest and a main.py entrypoint.
        """
        plugin_dir = base_dir / plugin_id
        plugin_dir.mkdir(parents=True, exist_ok=True)

        manifest_data = {
            "name": plugin_id.replace("_", " ").title(),
            "description": f"Test plugin {plugin_id}",
            "author": "Test Suite",
            "version": "0.1.0",
            "dependencies": dependencies or [],
        }

        manifest_path = plugin_dir / "plugin.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        main_py = plugin_dir / "main.py"
        main_py_content = f"""
from cat.looking_glass.mad_hatter.decorators import hook

@hook(priority={priority})
def before_cat_sends_message(message, cat):
    return message
"""
        with open(main_py, "w", encoding="utf-8") as f:
            f.write(main_py_content.strip())

        return plugin_dir

    # Create the default non-toggleable base plugin
    create_plugin(core_plugins_dir, "base_plugin")

    # Create default test plugins with a linear dependency chain: C -> B -> A
    create_plugin(custom_plugins_dir, "plugin_a", dependencies=[])
    create_plugin(custom_plugins_dir, "plugin_b", dependencies=["plugin_a"])
    create_plugin(custom_plugins_dir, "plugin_c", dependencies=["plugin_b"])

    with patch("cat.utils.get_core_plugins_path", return_value=str(core_plugins_dir)), \
         patch("cat.utils.get_plugins_path", return_value=str(custom_plugins_dir)):
        yield {
            "tmp_path": tmp_path,
            "core_plugins_dir": core_plugins_dir,
            "custom_plugins_dir": custom_plugins_dir,
            "create_plugin": create_plugin,
        }


@pytest.fixture
def test_agent_key() -> str:
    """Fixture providing a mock agent key string for tests."""
    return "test_agent_key"
