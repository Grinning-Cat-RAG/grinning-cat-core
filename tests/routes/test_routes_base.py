import pytest

from cat import AuthPermission, log
from cat.auth.permissions import AuthResource

from cat.auth.permissions import get_full_permissions

from tests.utils import (
    create_new_user,
    get_client_admin_headers,
    new_user_password,
    agent_id,
)


async def test_ping_success(client):
    response = await client.get("/")
    assert response.status_code == 200

    json_response = response.json()
    assert isinstance(json_response, str)


@pytest.mark.parametrize("path", ["/health/readiness", "/health/liveness", "/mgmt_message/global_message"])
async def test_probe_routes_log_the_user_agent(client, monkeypatch, path):
    """Health probes and the public banner read are unauthenticated: the caller's
    User-Agent is logged to tell which client polls them."""
    messages = []
    monkeypatch.setattr(log, "info", messages.append)

    response = await client.get(path, headers={"User-Agent": "probe-under-test/1.0"})
    assert response.status_code == 200

    assert any(path in m and "probe-under-test/1.0" in m for m in messages)


async def test_ping_non_admin_endpoint_with_admin(secure_client, secure_client_headers, client, cheshire_cat):
    permissions = get_full_permissions()
    permissions[str(AuthResource.PLUGIN)].append(str(AuthPermission.READ))

    new_admin = await create_new_user(
        client, headers=await get_client_admin_headers(client), permissions=permissions
    )

    creds = {
        "username": new_admin["username"],
        "password": new_user_password,
    }

    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 200

    res_json = res.json()
    received_token = res_json["access_token"]

    # check the access to the get LLMs endpoint
    response = await client.get("/plugins/", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id})
    assert response.status_code == 200
