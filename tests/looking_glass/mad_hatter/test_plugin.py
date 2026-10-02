import os
import pytest
from inspect import isfunction

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
