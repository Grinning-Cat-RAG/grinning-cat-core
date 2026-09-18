from tests.utils import agent_core_plugins, just_installed_plugin
from tests.mocks.mock_plugin.mock_plugin_overrides import MockPluginSettings


async def test_get_all_plugin_settings(lizard, secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)

    response = await secure_client.get("/plugins/settings", headers=secure_client_headers)
    json = response.json()

    plugin_manager = lizard.plugin_manager
    available_plugins = agent_core_plugins(plugin_manager) + ["mock_plugin"]

    assert response.status_code == 200
    assert isinstance(json["settings"], list)
    assert len(json["settings"]) == len(available_plugins)

    # system plugins are not manageable at an agent level: they are not listed here
    assert plugin_manager.get_untoggling_plugin_ids
    for system_plugin in plugin_manager.get_untoggling_plugin_ids:
        assert system_plugin not in [s["name"] for s in json["settings"]]

    for setting in json["settings"]:
        assert setting["name"] in available_plugins
        if setting["name"] == "mock_plugin":
            assert setting["value"] == {"a": "a", "b": 0}
            assert setting["scheme"] == MockPluginSettings.model_json_schema()
        elif setting["name"] == "memory":
            assert setting["value"] == {
                "enable_llm_knowledge": True,
                "fast_reply_message": "Sorry, I have no memories about that.",
            }
        else:
            assert setting["value"] == {}
            assert setting["scheme"] == {}


async def test_get_plugin_settings_non_existent(secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)

    non_existent_plugin = "ghost_plugin"
    response = await secure_client.get(f"/plugins/settings/{non_existent_plugin}", headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 404
    assert "not found" in json["detail"]


# endpoint to get settings and settings schema
async def test_get_plugin_settings(secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)

    response = await secure_client.get("/plugins/settings/mock_plugin", headers=secure_client_headers)
    response_json = response.json()

    assert response.status_code == 200
    assert response_json["name"] == "mock_plugin"
    assert response_json["value"] == {"a": "a", "b": 0}
    assert response_json["scheme"] == MockPluginSettings.model_json_schema()


async def test_save_wrong_plugin_settings(secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)

    # save settings (wrong schema)
    fake_settings = {"a": "a", "c": 1}
    response = await secure_client.put("/plugins/settings/mock_plugin", json=fake_settings, headers=secure_client_headers)
    assert response.status_code == 400

    # check default settings did not change
    response = await secure_client.get("/plugins/settings/mock_plugin", headers=secure_client_headers)
    assert response.status_code == 200
    json = response.json()
    assert json["name"] == "mock_plugin"
    assert json["value"] == {"a": "a", "b": 0}


async def test_save_plugin_settings(secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)

    # save settings
    fake_settings = {"a": "a", "b": 1}
    response = await secure_client.put("/plugins/settings/mock_plugin", json=fake_settings, headers=secure_client_headers)

    # check immediate response
    assert response.status_code == 200
    json = response.json()
    assert json["name"] == "mock_plugin"
    assert json["value"] == fake_settings

    # get settings back for this specific plugin
    response = await secure_client.get("/plugins/settings/mock_plugin", headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    assert json["name"] == "mock_plugin"
    assert json["value"] == fake_settings

    # retrieve all plugins settings to check if it was saved in DB
    response = await secure_client.get("/plugins/settings", headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    saved_config = [c for c in json["settings"] if c["name"] == "mock_plugin"]
    assert saved_config[0]["value"] == fake_settings


async def test_base_plugin_settings(secure_client, secure_client_headers, cheshire_cat):
    """base_plugin is a system plugin: an agent can neither read nor write its settings.

    They stay reachable on the system routes (see test_admins_plugin_settings.py).
    """
    fake_settings = {"a": "a", "b": 1}

    response = await secure_client.put(
        "/plugins/settings/base_plugin", json=fake_settings, headers=secure_client_headers
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Plugin not found"

    response = await secure_client.get("/plugins/settings/base_plugin", headers=secure_client_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "Plugin not found"


async def test_reset_plugin_settings(secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)

    # save settings
    fake_settings = {"a": "a", "b": 1}
    response = await secure_client.put("/plugins/settings/mock_plugin", json=fake_settings, headers=secure_client_headers)
    assert response.status_code == 200

    # reset settings
    response = await secure_client.post("/plugins/settings/mock_plugin", headers=secure_client_headers)
    assert response.status_code == 200
    json = response.json()
    assert json["name"] == "mock_plugin"
    assert json["value"] == {"a": "a", "b": 0}

    # get settings back for this specific plugin
    response = await secure_client.get("/plugins/settings/mock_plugin", headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    assert json["name"] == "mock_plugin"
    assert json["value"] == {"a": "a", "b": 0}

    # retrieve all plugins settings to check if it was saved in DB
    response = await secure_client.get("/plugins/settings", headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    saved_config = [c for c in json["settings"] if c["name"] == "mock_plugin"]
    assert saved_config[0]["value"] == {"a": "a", "b": 0}
