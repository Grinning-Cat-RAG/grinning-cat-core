import jwt

from cat.auth.permissions import get_base_permissions
from cat.auth.tokens import JWT_ALGORITHM

from tests.utils import agent_id, chat_id, create_new_user, api_key, new_user_password, http_message


async def _login(secure_client, client):
    creds = {"username": "user", "password": new_user_password}
    await create_new_user(
        secure_client,
        creds["username"],
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
        password=creds["password"],
    )
    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    return res.json()


async def test_token_returns_refresh_token(secure_client, client, cheshire_cat):
    body = await _login(secure_client, client)
    assert body["refresh_token"] and body["expires_in"] > 0 and body["refresh_expires_in"] > 0


async def test_refresh_rotates_and_detects_reuse(secure_client, client, cheshire_cat):
    body = await _login(secure_client, client)
    old_refresh = body["refresh_token"]

    res = await client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert res.status_code == 200
    new_body = res.json()
    assert new_body["refresh_token"] != old_refresh

    # the new access token works
    headers = {"Authorization": f"Bearer {new_body['access_token']}", "X-Agent-ID": agent_id, "X-Chat-ID": chat_id}
    status_code, _ = await http_message(client, {"text": "hey"}, headers)
    assert status_code == 200

    # reuse of the rotated token is refused and kills the whole session
    assert (await client.post("/auth/refresh", json={"refresh_token": old_refresh})).status_code == 401
    assert (await client.post("/auth/refresh", json={"refresh_token": new_body["refresh_token"]})).status_code == 401


async def test_logout_revokes_refresh(secure_client, client, cheshire_cat):
    body = await _login(secure_client, client)
    assert (await client.post("/auth/logout", json={"refresh_token": body["refresh_token"]})).status_code == 204
    assert (await client.post("/auth/refresh", json={"refresh_token": body["refresh_token"]})).status_code == 401


async def test_forged_jwt_is_rejected_by_me(secure_client, client, cheshire_cat):
    forged = jwt.encode({"sub": "admin", "agents": []}, "not_the_secret", algorithm=JWT_ALGORITHM)
    res = await client.get("/me", headers={"Authorization": f"Bearer {forged}"})
    assert res.status_code == 401


async def test_jwt_is_bound_to_the_agent_it_was_issued_for(secure_client, client, cheshire_cat):
    body = await _login(secure_client, client)
    # a token issued for `agent_id` must not authenticate a same-username user of another agent
    headers = {"Authorization": f"Bearer {body['access_token']}", "X-Agent-ID": "another_agent", "X-Chat-ID": chat_id}
    status_code, _ = await http_message(client, {"text": "hey"}, headers)
    assert status_code == 401


async def test_jwt_is_bound_to_an_existing_other_agent(secure_client, client, lizard, cheshire_cat):
    body = await _login(secure_client, client)
    other_agent_id = "another_agent"
    await lizard.create_cheshire_cat(other_agent_id)
    await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": other_agent_id},
        permissions=get_base_permissions(),
        password=new_user_password,
    )
    headers = {"Authorization": f"Bearer {body['access_token']}", "X-Agent-ID": other_agent_id, "X-Chat-ID": chat_id}
    status_code, _ = await http_message(client, {"text": "hey"}, headers)
    assert status_code == 401
