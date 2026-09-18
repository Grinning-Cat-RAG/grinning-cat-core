from tests.utils import agent_core_plugins, just_installed_plugin


async def _check_activation(secure_client, secure_client_headers):
    # GET plugins endpoint lists the plugin
    response = await secure_client.get("/plugins/", headers=secure_client_headers)
    available_plugins = response.json()["installed"]
    mock_plugin = [p for p in available_plugins if p["id"] == "mock_plugin"][0]
    assert isinstance(mock_plugin["local_info"]["active"], bool)
    assert mock_plugin["local_info"]["active"]  # plugin active

    assert mock_plugin["id"] == "mock_plugin"
    assert len(mock_plugin["local_info"]["hooks"]) == 3
    assert len(mock_plugin["local_info"]["tools"]) == 2
    assert len(mock_plugin["local_info"]["forms"]) == 1
    assert len(mock_plugin["local_info"]["endpoints"]) == 7


async def test_toggle_non_existent_plugin(secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)

    response = await secure_client.put("/plugins/toggle/no_plugin", headers=secure_client_headers)
    response_json = response.json()

    assert response.status_code == 404
    assert response_json["detail"] == "Plugin not found"


async def test_activate_plugin(secure_client, secure_client_headers, cheshire_cat):
    # install and activate
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)

    await _check_activation(secure_client, secure_client_headers)


async def test_deactivate_plugin(lizard, secure_client, secure_client_headers, cheshire_cat):
    # install and activate
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)
    core_plugins = agent_core_plugins(lizard.plugin_manager)

    # verify that the plugin is active
    response = await secure_client.get("/plugins/", headers=secure_client_headers)
    available_plugins = response.json()["installed"]
    assert len(available_plugins) == len(core_plugins) + 1  # core plugins and mock_plugin

    # deactivate
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    # the mock_plugin is installed but no more active
    response = await secure_client.get("/plugins/", headers=secure_client_headers)
    available_plugins = response.json()["installed"]
    assert len(available_plugins) == len(core_plugins) + 1  # core plugins and mock_plugin
    for p in available_plugins:
        assert p["local_info"]["active"] == (p["id"] in core_plugins)


async def test_reactivate_plugin(secure_client, secure_client_headers, cheshire_cat):
    # install and activate
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)

    # deactivate
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    # re-activate
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)
    await _check_activation(secure_client, secure_client_headers)


async def test_deactivate_base_plugin(lizard, secure_client, secure_client_headers, cheshire_cat):
    """System plugins are not manageable at an agent level: the route reports them as
    not found, so an agent can never deactivate one."""
    response = await secure_client.put("/plugins/toggle/base_plugin", headers=secure_client_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "Plugin not found"
