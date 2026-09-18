from cat.looking_glass.mad_hatter.decorators.endpoint import CatEndpoint


def get_endpoint(plugin_manager, uri, method = None):
    for e in plugin_manager.endpoints:
        condition = e.name == uri
        if method:
            condition = condition and method in e.methods
        if condition:
            return e

    return None


def test_endpoints_discovery(plugin_manager):
    mock_plugin_endpoints = plugin_manager.plugins["mock_plugin"].endpoints
    for e in mock_plugin_endpoints:
        assert isinstance(e, CatEndpoint)
        assert e.plugin_id == "mock_plugin"
        assert e in plugin_manager.endpoints

    # discovered endpoints
    assert len(mock_plugin_endpoints) == 7

    # basic properties
    for e in mock_plugin_endpoints:
        assert isinstance(e, CatEndpoint)
        assert e.plugin_id == "mock_plugin"


def test_endpoint_decorator(plugin_manager):
    endpoint = get_endpoint(plugin_manager, "/custom/endpoint")

    assert endpoint.name == "/custom/endpoint"
    assert endpoint.prefix == "/custom"
    assert endpoint.path == "/endpoint"
    assert endpoint.methods == {"GET"}  # fastapi stores http verbs as a set
    assert endpoint.tags == ["Custom Endpoints"]
    assert endpoint.function() == {"result": "endpoint default prefix"}


def test_endpoint_decorator_prefix(plugin_manager):
    endpoint = get_endpoint(plugin_manager, "/tests/endpoint")

    assert endpoint.name == "/tests/endpoint"
    assert endpoint.prefix == "/tests"
    assert endpoint.path == "/endpoint"
    assert endpoint.methods == {"GET"}
    assert endpoint.tags == ["Tests"]
    assert endpoint.function() == {"result": "endpoint prefix tests"}


def test_get_endpoint(plugin_manager):
    endpoint = get_endpoint(plugin_manager, "/tests/crud", "GET")

    assert endpoint.name == "/tests/crud"
    assert endpoint.prefix == "/tests"
    assert endpoint.path == "/crud"
    assert endpoint.methods == {"GET"}
    assert endpoint.tags == ["Tests"]


def test_post_endpoint(plugin_manager):
    endpoint = get_endpoint(plugin_manager, "/tests/crud", "POST")

    assert endpoint.name == "/tests/crud"
    assert endpoint.prefix == "/tests"
    assert endpoint.path == "/crud"
    assert endpoint.methods == {"POST"}
    assert endpoint.tags == ["Tests"]


def test_put_endpoint(plugin_manager):
    endpoint = get_endpoint(plugin_manager, "/tests/crud/{item_id}", "PUT")

    assert endpoint.name == "/tests/crud/{item_id}"
    assert endpoint.prefix == "/tests"
    assert endpoint.path == "/crud/{item_id}"
    assert endpoint.methods == {"PUT"}
    assert endpoint.tags == ["Tests"]


def test_delete_endpoint(plugin_manager):
    endpoint = get_endpoint(plugin_manager, "/tests/crud/{item_id}", "DELETE")

    assert endpoint.name == "/tests/crud/{item_id}"
    assert endpoint.prefix == "/tests"
    assert endpoint.path == "/crud/{item_id}"
    assert endpoint.methods == {"DELETE"}
    assert endpoint.tags == ["Tests"]


async def test_endpoints_deactivation_or_uninstall(lizard):
    # custom endpoints are registered in mad_hatter, mock_plugin is installed into the plugin_manager fixture
    for e in lizard.plugin_manager.endpoints:
        assert isinstance(e, CatEndpoint)
        assert e.plugin_id in lizard.plugin_manager.get_core_plugins_ids + ["mock_plugin"]

    await lizard.uninstall_plugin("mock_plugin")

    # no more custom endpoints
    for e in lizard.plugin_manager.endpoints:
        assert isinstance(e, CatEndpoint)
        assert e.plugin_id != "mock_plugin"
        assert e.plugin_id in lizard.plugin_manager.get_core_plugins_ids


# The admin UI calls both of these on every Plugins page load, and nothing was
# exercising them: a rename of the plugin-manager property behind the second one
# turned it into an AttributeError without a single test noticing.
async def test_core_plugins_endpoints(lizard, secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.get("/admins/core_plugins/", headers=secure_client_headers)

    assert response.status_code == 200
    assert sorted(response.json()) == sorted(lizard.plugin_manager.get_core_plugins_ids)


async def test_core_plugins_untoggling_endpoint(lizard, secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.get("/admins/core_plugins/untoggling/", headers=secure_client_headers)

    assert response.status_code == 200
    body = response.json()
    assert sorted(body) == sorted(lizard.plugin_manager.get_non_toggleable_plugin_ids)
    assert body  # the list has content: every entry must be a real plugin
    for plugin_id in body:
        assert plugin_id in lizard.plugin_manager.get_core_plugins_ids
