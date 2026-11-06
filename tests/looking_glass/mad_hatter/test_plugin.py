import os
import pytest
from inspect import isfunction
from pathlib import Path

from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.looking_glass.mad_hatter.decorators.experimental.form import CatForm
from cat.looking_glass.mad_hatter.decorators.experimental.mcp_client import CatMcpClient
from cat.looking_glass.mad_hatter.decorators.hook import CatHook
from cat.looking_glass.mad_hatter.decorators.tool import CatTool
from cat.looking_glass.mad_hatter.plugin import Plugin, PluginManifest

from tests.utils import mock_plugin_path


def test_create_plugin_wrong_folder():
    with pytest.raises(Exception) as e:
        Plugin("/non/existent/folder")

    assert "Cannot create" in str(e.value)


def test_create_plugin_empty_folder():
    path = "tests/mocks/empty_folder"

    os.mkdir(path)

    with pytest.raises(Exception) as e:
        Plugin(path)

    assert "Cannot create" in str(e.value)


def _write_plugin(root, rel_paths):
    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        # test modules only import under pytest: importing them at load time would break the plugin
        path.write_text("raise ImportError('test-only module')\n" if "test" in rel else "\n")


def test_plugin_test_files_are_not_loaded(tmp_path):
    root = tmp_path / "plugin_with_tests"
    _write_plugin(root, [
        "hook.py", "sub/module.py", "contests.py", "my_tests.py", "sub/latest/module.py",
        "conftest.py", "tests/__init__.py", "tests/conftest.py", "tests/test_a.py",
        "tests/unit/test_b.py", "sub/tests/test_c.py", "sub/conftest.py",
    ])

    plugin = Plugin(str(root))

    loaded = sorted(Path(p).relative_to(root).as_posix() for p in plugin._py_files)
    assert loaded == ["contests.py", "hook.py", "my_tests.py", "sub/latest/module.py", "sub/module.py"]


def test_plugin_path_inside_a_tests_folder_is_still_loaded(tmp_path):
    """Only folders *inside* the plugin are skipped: the plugin may itself live under a `tests` path."""
    root = tmp_path / "tests" / "mocks" / "plugin"
    _write_plugin(root, ["hook.py"])

    assert [Path(p).name for p in Plugin(str(root))._py_files] == ["hook.py"]


def test_plugin_path_with_glob_metacharacters(tmp_path):
    """`[1]` in the folder name must not be read as a pattern matching a sibling `plug1` folder."""
    _write_plugin(tmp_path / "plug1", ["other_plugin_hook.py"])
    root = tmp_path / "plug[1]"
    _write_plugin(root, ["hook.py"])

    assert [Path(p).name for p in Plugin(str(root))._py_files] == ["hook.py"]


def test_create_plugin_with_only_test_files(tmp_path):
    root = tmp_path / "only_tests"
    _write_plugin(root, ["conftest.py", "tests/test_a.py"])

    with pytest.raises(Exception) as e:
        Plugin(str(root))

    assert "Cannot create" in str(e.value)


def test_create_plugin(plugin):
    assert not plugin.active

    assert plugin.path == mock_plugin_path
    assert plugin.id == "mock_plugin"

    # manifest
    assert isinstance(plugin.manifest, PluginManifest)
    assert plugin.manifest.id == plugin.id
    assert plugin.manifest.name == "MockPlugin"
    assert "Description not found" in plugin.manifest.description

    # hooks and tools
    assert plugin.hooks == []
    assert plugin.tools == []


async def test_activate_plugin(plugin):
    # activate it
    await plugin.activate(DEFAULT_SYSTEM_KEY)

    assert plugin.active is True

    # hooks
    assert len(plugin.hooks) == 3
    for hook in plugin.hooks:
        assert isinstance(hook, CatHook)
        assert hook.plugin_id == "mock_plugin"
        assert hook.name in [
            "factory_allowed_llms",
            "before_cat_sends_message",
        ]
        assert isfunction(hook.function)

        if hook.name == "before_cat_sends_message":
            assert hook.priority > 1
        else:
            assert hook.priority == 1  # default priority

    # tools
    assert len(plugin.tools) == 2
    tool = plugin.tools[0]
    assert isinstance(tool, CatTool)
    assert tool.name == "mock_tool"
    assert tool.description == "Used to test mock tools. Input is the topic."
    assert isfunction(tool.func)
    # tool examples found
    assert len(tool.examples) == 2
    assert "mock tool example 1" in tool.examples
    assert "mock tool example 2" in tool.examples

    # forms
    assert len(plugin.forms) == 1
    form = plugin.forms[0]
    assert isinstance(form, CatForm)

    # mcp_clients
    assert len(plugin.mcp_clients) == 1
    mcp_client = plugin.mcp_clients[0]
    assert isinstance(mcp_client, CatMcpClient)


async def test_deactivate_plugin(plugin):
    # The plugin is non active by default
    await plugin.activate(DEFAULT_SYSTEM_KEY)

    # deactivate it
    await plugin.deactivate(DEFAULT_SYSTEM_KEY)

    assert plugin.active is False

    # hooks and tools
    assert len(plugin.hooks) == 0
    assert len(plugin.procedures) == 0


def test_settings_schema(plugin):
    settings_schema = plugin.settings_schema()
    assert isinstance(settings_schema, dict)
    assert settings_schema["properties"] == {}
    assert settings_schema["title"] == "PluginSettingsModel"
    assert settings_schema["type"] == "object"


async def test_load_settings(plugin):
    settings = await plugin.load_settings(DEFAULT_SYSTEM_KEY)
    assert settings == {}


async def test_save_settings(plugin):
    fake_settings = {"a": 42}
    await plugin.save_settings(fake_settings, DEFAULT_SYSTEM_KEY)

    settings = await plugin.load_settings(DEFAULT_SYSTEM_KEY)
    assert settings["a"] == fake_settings["a"]
