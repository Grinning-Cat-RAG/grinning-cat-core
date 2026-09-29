import pytest

from tests.utils import just_installed_plugin, create_new_user, new_user_password, agent_id, get_client_admin_headers


async def test_custom_endpoint_base(client, secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)
    # activate the plugin
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    response = await client.get("/custom/endpoint")
    assert response.status_code == 200
    assert response.json()["result"] == "endpoint default prefix"


async def test_custom_endpoint_prefix(client, secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)
    # activate the plugin
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    response = await client.get("/tests/endpoint")
    assert response.status_code == 200
    assert response.json()["result"] == "endpoint prefix tests"


async def test_custom_endpoint_get(secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)
    # activate the plugin
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    response = await secure_client.get("/tests/crud", headers=secure_client_headers)

    assert response.status_code == 200
    assert response.json()["result"] == "ok"
    assert isinstance(response.json()["user_id"], str)
    assert len(response.json()["user_id"]) == 36


async def test_custom_endpoint_get_admin_not_found(client, secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    response = await client.get("/tests/admin/crud", headers=await get_client_admin_headers(client) | {"X-Agent-ID": agent_id})
    assert response.status_code == 200


async def test_custom_endpoint_get_not_found(secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)

    response = await secure_client.get("/tests/crud", headers=secure_client_headers)
    assert response.status_code == 404


async def test_custom_endpoint_post(client, secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)
    # activate the plugin
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    payload = {"name": "the cat", "description" : "it's magic"}
    response = await client.post("/tests/crud", json=payload)

    assert response.status_code == 200
    assert response.json()["name"] == "the cat"
    assert response.json()["description"] == "it's magic"


async def test_custom_endpoint_put(client, secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)
    # activate the plugin
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    payload = {"name": "the cat", "description": "it's magic"}
    response = await client.put("/tests/crud/123", json=payload)
    
    assert response.status_code == 200
    assert response.json()["id"] == 123
    assert response.json()["name"] == "the cat"
    assert response.json()["description"] == "it's magic"

async def test_custom_endpoint_delete(client, secure_client, secure_client_headers, cheshire_cat):
    await just_installed_plugin(secure_client, secure_client_headers)
    # activate the plugin
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    response = await client.delete("/tests/crud/123")
    
    assert response.status_code == 200
    assert response.json()["result"] == "ok"
    assert response.json()["id"] == 123


@pytest.mark.parametrize("switch_type", ["deactivate", "uninstall"])
async def test_custom_endpoints_on_plugin_deactivation_or_uninstall(
    switch_type, secure_client, secure_client_headers, cheshire_cat,
):
    # install and activate the plugin
    await just_installed_plugin(secure_client, secure_client_headers, activate=True)

    # endpoints added via mock_plugin (verb, endpoint, payload)
    custom_endpoints = [
        ("GET", "/custom/endpoint", None, False),
        ("GET", "/tests/endpoint", None, False),
        ("GET", "/tests/crud", None, True),
        ("POST", "/tests/crud", {"name": "the cat", "description": "it's magic"}, False),
        ("PUT", "/tests/crud/123", {"name": "the cat", "description": "it's magic"}, False),
        ("DELETE", "/tests/crud/123", None, False),
        # regression: the path with parameters was compared with the template of the endpoint, and the endpoint
        # behind the authentication was still reachable for the agent without the plugin
        ("DELETE", "/tests/crud/secure/123", None, True),
    ]

    # custom endpoints are active
    for verb, endpoint, payload, _ in custom_endpoints:
        response = await secure_client.request(verb, endpoint, json=payload, headers=secure_client_headers)
        assert response.status_code == 200

    if switch_type == "deactivate":
        # deactivate plugin
        response = await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)
        assert response.status_code == 200
    else:
        # uninstall plugin
        response = await secure_client.delete("/plugins/uninstall/mock_plugin", headers=secure_client_headers)
        assert response.status_code == 200

    # no more custom endpoints
    for verb, endpoint, payload, upon_auth in custom_endpoints:
        # the endpoint is still reachable, unless it is behind the authentication, on deactivation
        response = await secure_client.request(verb, endpoint, json=payload, headers=secure_client_headers)
        assert response.status_code == (404 if switch_type == "uninstall" or upon_auth else 200)


@pytest.mark.parametrize("resource", ["PLUGIN", "LLM"])
@pytest.mark.parametrize("permission", ["READ", "DELETE"])
async def test_custom_endpoint_permissions(
    resource, permission, client, secure_client, secure_client_headers, cheshire_cat,
):
    await just_installed_plugin(secure_client, secure_client_headers)
    # activate the plugin
    await secure_client.put("/plugins/toggle/mock_plugin", headers=secure_client_headers)

    # create user with permissions
    data = await create_new_user(
        secure_client, headers=secure_client_headers, permissions={resource: [permission]}
    )
    # get jwt for user
    response = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = response.json()["access_token"]

    # use endpoint (requires PLUGIN resource and READ permission)
    response = await client.get("/tests/crud", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id})
    if resource == "PLUGIN" and permission == "READ":
        assert response.status_code == 200
    else:
        assert response.status_code == 403
