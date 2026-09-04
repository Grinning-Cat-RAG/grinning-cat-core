import os
import shutil
import pytest

from tests.utils import create_mock_plugin_zip


async def mock_registry_download_plugin(url: str):
    return create_mock_plugin_zip(True)


async def test_list_registry_plugins(secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.get("/plugins/installed", headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 200
    assert "registry" in json.keys()

    # registry (see more registry tests in `./test_plugins_registry.py`)
    assert isinstance(json["registry"], list)
    assert len(json["registry"]) >= 0

    if len(json["registry"]) > 0:
        # query
        for key in ["query"]:  # ["query", "author", "tag"]:
            assert key in json["filters"].keys()


async def test_list_registry_plugins_by_query(secure_client, secure_client_headers, cheshire_cat):
    params = {"query": "podcast"}
    response = await secure_client.get("/plugins/installed", params=params, headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 200
    assert json["filters"]["query"] == params["query"]
    assert len(json["registry"]) > 0  # found registry plugins with text
    for p in json["registry"]:
        plugin_text = p["name"] + p["description"]
        assert params["query"] in plugin_text  # verify searched text


async def test_plugin_install_from_registry(secure_client, secure_client_headers, monkeypatch, lizard, cheshire_cat):
    # Mock the download from the registry creating a zip on-the-fly
    monkeypatch.setattr(lizard.plugin_registry, "download_plugin", mock_registry_download_plugin)

    # during tests, the cat uses a different folder for plugins
    new_plugin_final_folder = "tests/mocks/mock_plugin_folder/mock_plugin"

    if os.path.exists(new_plugin_final_folder):
        shutil.rmtree(new_plugin_final_folder)
    assert not os.path.exists(new_plugin_final_folder)

    # install plugin from registry
    payload = {"url": "https://mockup_url.com"}
    response = await secure_client.post("/plugins/install/registry", json=payload, headers=secure_client_headers)

    assert response.status_code == 200
    assert response.json()["url"] == payload["url"]
    assert response.json()["info"] == "Plugin is being installed asynchronously"

    # GET plugin endpoint responds
    response = await secure_client.get("/plugins/system/details/mock_plugin", headers=secure_client_headers)
    assert response.status_code == 200
    json = response.json()
    assert json["data"]["id"] == "mock_plugin"
    assert isinstance(json["data"]["local_info"]["active"], bool)
    assert json["data"]["local_info"]["active"]

    # GET plugins endpoint lists the plugin
    response = await secure_client.get("/plugins/installed", headers=secure_client_headers)
    assert response.status_code == 200
    installed_plugins = response.json()["installed"]
    installed_plugins_names = list(map(lambda p: p["id"], installed_plugins))
    assert "mock_plugin" in installed_plugins_names
    # core plugins and new_plugin are active
    for p in installed_plugins:
        assert isinstance(p["local_info"]["active"], bool)
        assert p["local_info"]["active"]

    # plugin has been actually extracted in (mock) plugins folder
    assert os.path.exists(new_plugin_final_folder)


# take away from the list of available registry plugins, the ones that are already installed
async def test_list_registry_plugins_without_duplicating_installed_plugins(
    secure_client, secure_client_headers, cheshire_cat,
):
    # 1. install plugin from registry
    # TODO !!!

    # 2. get available plugins searching for the one just installed
    params = {"query": "podcast"}
    response = await secure_client.get("/plugins/installed", params=params, headers=secure_client_headers)
    #json = response.json()

    # 3. plugin should show up among installed by not among registry ones
    assert response.status_code == 200
    # TODO plugin compares in installed!!!
    # TODO plugin does not appear in registry!!!


@pytest.mark.skip("This test has to be activated when also search by tag and author is activated in core")
async def test_list_registry_plugins_by_author(secure_client, secure_client_headers, cheshire_cat):
    params = {
        "author": "Nicola Corbellini"
    }
    response = await secure_client.get("/plugins/installed", params=params, headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 200
    assert json["filters"]["author"] == params["query"]
    assert len(json["registry"]) > 0 # found registry plugins with author
    for p in json["registry"]:
        assert params["author"] in p["author_name"] # verify author


@pytest.mark.skip("This test has to be activated when also search by tag and author is activated in core")
async def test_list_registry_plugins_by_tag(secure_client, secure_client_headers, cheshire_cat):
    params = {
        "tag": "llm"
    }
    response = await secure_client.get("/plugins/installed", params=params, headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 200
    assert json["filters"]["tag"] == params["tag"]
    assert len(json["registry"]) > 0 # found registry plugins with tag
    for p in json["registry"]:
        plugin_tags = p["tags"].split(", ")
        assert params["tag"] in plugin_tags # verify tag
