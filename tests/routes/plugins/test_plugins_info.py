from tests.routes.plugins.conftest import REGISTRY_CATALOGUE


async def test_list_plugins(lizard, secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.get("/plugins/", headers=secure_client_headers)
    json = response.json()

    core_plugins = lizard.plugin_manager.get_core_plugins_ids

    assert response.status_code == 200
    for key in ["filters", "installed", "registry"]:
        assert key in json.keys()

    # query
    for key in ["query"]:  # ["query", "author", "tag"]:
        assert key in json["filters"].keys()

    # installed
    for idx in range(len(json["installed"])):
        assert "id" in json["installed"][idx].keys()
        assert json["installed"][idx]["id"] in core_plugins

        assert "local_info" in json["installed"][idx].keys()
        assert isinstance(json["installed"][idx]["local_info"], dict)
        assert "active" in json["installed"][idx]["local_info"].keys()
        assert isinstance(json["installed"][idx]["local_info"]["active"], bool)
        assert json["installed"][idx]["local_info"]["active"]

    # registry (see more registry tests in `./test_plugins_registry.py`)
    assert isinstance(json["registry"], list)
    assert [p["plugin_url"] for p in json["registry"]] == [p["plugin_url"] for p in REGISTRY_CATALOGUE]
