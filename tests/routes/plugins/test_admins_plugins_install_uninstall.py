import os

from cat.db import crud
from cat.db.cruds import plugins as crud_plugins
from cat.db.database import DEFAULT_SYSTEM_KEY

from tests.utils import agent_core_plugins, api_key, agent_id, just_installed_plugin


# NOTE: here we test zip upload and install
# installation from registry is in `./test_plugins_registry.py`
async def test_plugin_install_from_zip(lizard, secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)
    # only the agent-visible core plugins: system plugins are hidden at an agent level
    core_plugins = agent_core_plugins(lizard.plugin_manager)

    # during tests, the cat uses a different folder for plugins
    mock_plugin_final_folder = "tests/mocks/mock_plugin_folder/mock_plugin"

    # GET plugin endpoint responds
    response = await secure_client.get("/plugins/system/details/mock_plugin", headers=secure_client_headers)
    assert response.status_code == 200
    json = response.json()
    assert json["data"]["id"] == "mock_plugin"
    assert isinstance(json["data"]["local_info"]["active"], bool)
    assert json["data"]["local_info"]["active"]

    # GET plugins endpoint lists the plugin
    response = await secure_client.get("/plugins/installed", headers=secure_client_headers)
    installed_plugins = response.json()["installed"]
    installed_plugins_names = list(map(lambda p_: p_["id"], installed_plugins))
    assert "mock_plugin" in installed_plugins_names
    # core plugins and mock_plugin are active
    for p in installed_plugins:
        assert isinstance(p["local_info"]["active"], bool)
        assert p["local_info"]["active"]

    # now, lists the plugins as an agent (new plugins are installed but deactivated, initially)
    response = await secure_client.get(
        "/plugins/", headers={"X-Agent-ID": cheshire_cat.agent_key, "Authorization": f"Bearer {api_key}"}
    )
    installed_plugins = response.json()["installed"]
    installed_plugins_names = list(map(lambda p_: p_["id"], installed_plugins))
    assert "mock_plugin" in installed_plugins_names
    assert len(installed_plugins_names) == len(core_plugins) + 1
    # core plugins are active, mock_plugin is not at an agent level
    for p in installed_plugins:
        assert isinstance(p["local_info"]["active"], bool)
        assert p["local_info"]["active"] == (p["id"] in core_plugins)

    # plugin has been actually extracted in (mock) plugins folder
    assert os.path.exists(mock_plugin_final_folder)

    # GET single plugin info, plugin is active
    response = await secure_client.get("/plugins/system/details/mock_plugin", headers=secure_client_headers)
    assert isinstance(response.json()["data"]["local_info"]["active"], bool)
    assert response.json()["data"]["local_info"]["active"]


async def test_plugin_install_after_cheshire_cat_creation(lizard, secure_client, secure_client_headers, cheshire_cat):
    # create a new agent
    ccat = await lizard.create_cheshire_cat("agent_test_test")
    # only the agent-visible core plugins: system plugins are hidden at an agent level
    core_plugins = agent_core_plugins(lizard.plugin_manager)

    # list the plugins as an agent: mock_plugin is not installed yet
    response = await secure_client.get(
        "/plugins/", headers={"X-Agent-ID": ccat.agent_key, "Authorization": f"Bearer {api_key}"}
    )
    installed_plugins = response.json()["installed"]
    installed_plugins_names = list(map(lambda p_: p_["id"], installed_plugins))
    assert "mock_plugin" not in installed_plugins_names

    await just_installed_plugin(secure_client, secure_client_headers, plugin_id="mock_plugin")

    # now, lists the plugins as an agent (new plugins are installed but deactivated, initially)
    response = await secure_client.get(
        "/plugins/", headers={"X-Agent-ID": ccat.agent_key, "Authorization": f"Bearer {api_key}"}
    )
    installed_plugins = response.json()["installed"]
    installed_plugins_names = list(map(lambda p_: p_["id"], installed_plugins))
    assert "mock_plugin" in installed_plugins_names
    assert len(installed_plugins_names) == len(core_plugins) + 1
    # core plugins are active, mock_plugin is not at an agent level
    for p in installed_plugins:
        assert isinstance(p["local_info"]["active"], bool)
        assert p["local_info"]["active"] == (p["id"] in core_plugins)


async def test_create_cheshire_cat_after_plugin_install(lizard, secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers, plugin_id="mock_plugin")

    # create a new agent
    ccat = await lizard.create_cheshire_cat("agent_test_test")
    # only the agent-visible core plugins: system plugins are hidden at an agent level
    core_plugins = agent_core_plugins(lizard.plugin_manager)

    # now, lists the plugins as an agent (new plugins are installed but deactivated, initially)
    response = await secure_client.get(
        "/plugins/", headers={"X-Agent-ID": ccat.agent_key, "Authorization": f"Bearer {api_key}"}
    )
    installed_plugins = response.json()["installed"]
    installed_plugins_names = list(map(lambda p_: p_["id"], installed_plugins))
    assert "mock_plugin" in installed_plugins_names
    assert len(installed_plugins_names) == len(core_plugins) + 1
    # core plugins are active, mock_plugin is not at an agent level
    for p in installed_plugins:
        assert isinstance(p["local_info"]["active"], bool)
        assert p["local_info"]["active"] == (p["id"] in core_plugins)


async def test_plugin_uninstall(secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)

    # The plugin is active, now let's activate for the agent too
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)
    agent_settings = await crud.read(crud_plugins.format_key(agent_id, "mock_plugin"))
    system_settings = await crud.read(crud_plugins.format_key(DEFAULT_SYSTEM_KEY, "mock_plugin"))

    assert agent_settings is not None
    assert system_settings is not None

    # during tests, the cat uses a different folder for plugins
    mock_plugin_final_folder = "tests/mocks/mock_plugin_folder/mock_plugin"

    # remove plugin via endpoint (will delete also plugin folder in mock_plugin_folder)
    response = await secure_client.delete("/plugins/uninstall/mock_plugin", headers=secure_client_headers)
    assert response.status_code == 200

    # mock_plugin is not installed in the cat (check both via endpoint and filesystem)
    response = await secure_client.get("/plugins/installed", headers=secure_client_headers)
    installed_plugins_names = list(map(lambda p: p["id"], response.json()["installed"]))
    assert "mock_plugin" not in installed_plugins_names
    assert not os.path.exists(mock_plugin_final_folder)  # plugin folder removed from disk

    # GET single plugin info, plugin is not active
    response = await secure_client.get("/plugins/system/details/mock_plugin", headers=secure_client_headers)
    assert response.json()["detail"] == "Plugin not found"

    agent_settings = await crud.read(crud_plugins.format_key(agent_id, "mock_plugin"))
    system_settings = await crud.read(crud_plugins.format_key(DEFAULT_SYSTEM_KEY, "mock_plugin"))

    assert agent_settings is None
    assert system_settings is None


async def test_plugin_recurrent_installs(lizard, secure_client, secure_client_headers, cheshire_cat):
    # create a new agent
    ccat = await lizard.create_cheshire_cat("agent_test_test")

    await just_installed_plugin(secure_client, secure_client_headers, plugin_id="mock_plugin")

    # activate for the new agent
    ccat_headers = {"X-Agent-ID": ccat.agent_key, "Authorization": f"Bearer {api_key}"}
    await secure_client.put("/plugins/toggle/mock_plugin", headers=ccat_headers)
    response = await secure_client.get("/plugins/", headers=ccat_headers)
    installed_plugins = response.json()["installed"]
    installed_plugins_names = list(map(lambda p_: p_["id"], installed_plugins))
    assert "mock_plugin" in installed_plugins_names

    for p in installed_plugins:
        assert isinstance(p["local_info"]["active"], bool)
        assert p["local_info"]["active"]

    await just_installed_plugin(secure_client, secure_client_headers, plugin_id="mock_plugin")

    # now, re-list the plugins as an agent: all the plugins should be still active
    response = await secure_client.get("/plugins/", headers=ccat_headers)
    installed_plugins = response.json()["installed"]
    installed_plugins_names = list(map(lambda p_: p_["id"], installed_plugins))
    assert "mock_plugin" in installed_plugins_names

    for p in installed_plugins:
        assert isinstance(p["local_info"]["active"], bool)
        assert p["local_info"]["active"]


async def test_plugin_incremental_settings_on_recurrent_installs(
    lizard, secure_client, secure_client_headers, cheshire_cat,
):
    # create a new agent
    ccat = await lizard.create_cheshire_cat("agent_test_test")
    ccat_headers = {"X-Agent-ID": ccat.agent_key, "Authorization": f"Bearer {api_key}"}

    await just_installed_plugin(secure_client, secure_client_headers, plugin_id="mock_plugin")

    agent_settings = await crud_plugins.get_setting(ccat.agent_key, "mock_plugin")
    assert not agent_settings

    # activate for the new agent
    await secure_client.put("/plugins/toggle/mock_plugin", headers=ccat_headers)
    agent_settings = await crud_plugins.get_setting(ccat.agent_key, "mock_plugin")
    assert agent_settings["a"] == "a"
    assert agent_settings["b"] == 0

    # manually change the configuration of `mock_plugin` for the agent in the Redis database (mock a first manual
    # update)
    agent_settings["a"] = "value_a"
    agent_settings["b"] = 10
    await crud_plugins.update_setting(ccat.agent_key, "mock_plugin", agent_settings)

    # now, use a `mock_plugin_overrides.py.new` into the plugin folder to emulate a second update, with a new value in
    # the `a` and the removal of `b` in the settings
    os.replace("tests/mocks/mock_plugin/mock_plugin_overrides.py", "tests/mocks/mock_plugin_overrides.py")
    os.replace(
        "tests/mocks/mock_plugin_override/mock_plugin_overrides.py",
        "tests/mocks/mock_plugin/mock_plugin_overrides.py",
    )

    await just_installed_plugin(secure_client, secure_client_headers, plugin_id="mock_plugin")

    # check that the configuration of `mock_plugin` for the agent has changed according to the new mock_plugin_overrides.py
    agent_settings = await crud_plugins.get_setting(ccat.agent_key, "mock_plugin")

    try:
        assert "b" not in agent_settings
        assert agent_settings["a"] == "value_a"
    finally:
        os.replace(
            "tests/mocks/mock_plugin/mock_plugin_overrides.py",
            "tests/mocks/mock_plugin_override/mock_plugin_overrides.py",
        )
        os.replace("tests/mocks/mock_plugin_overrides.py", "tests/mocks/mock_plugin/mock_plugin_overrides.py")


async def test_plugin_install_from_zip_with_missing_dependencies(
    lizard, secure_client, secure_client_headers, cheshire_cat,
):
    plugin_name = "mock_plugin_with_dependencies"

    # check that "missing dependencies" and "mock_plugin" is within the error message
    try:
        await just_installed_plugin(secure_client, secure_client_headers, plugin_id=plugin_name)
    except Exception as e:
        assert "missing dependencies" in str(e)
        assert plugin_name in str(e)

    # GET plugin endpoint responds
    response = await secure_client.get(f"/plugins/system/details/{plugin_name}", headers=secure_client_headers)
    assert response.status_code == 404
