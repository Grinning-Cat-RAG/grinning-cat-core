"""
Credentials must NEVER be accepted from the querystring: URLs end up in proxy/server access logs, browser history
and Referer headers. These tests lock that behaviour for HTTP and WebSocket, with JWTs and with the API key.
"""
import pytest
from httpx_ws import WebSocketDisconnect

from cat.auth.permissions import get_base_permissions

from tests.utils import (
    agent_id,
    chat_id,
    api_key,
    create_new_user,
    new_user_password,
    send_websocket_message,
)

QUERY_PARAM_NAMES = ["token", "access_token", "jwt", "api_key", "apikey", "key", "Authorization"]


async def _jwt(secure_client, client) -> str:
    await create_new_user(
        secure_client, "user", headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(), password=new_user_password,
    )
    res = await client.post("/auth/token", json={"username": "user", "password": new_user_password})
    return res.json()["access_token"]


async def test_http_ignores_credentials_in_querystring(secure_client, client, cheshire_cat):
    token = await _jwt(secure_client, client)
    for credential in (token, api_key):
        for name in QUERY_PARAM_NAMES:
            res = await client.post(
                f"/message?{name}={credential}",
                json={"text": "hey"},
                headers={"X-Agent-ID": agent_id, "X-Chat-ID": chat_id},
            )
            assert res.status_code == 401, name

            res = await client.get(f"/me?{name}={credential}")
            assert res.status_code == 401, name


async def test_websocket_ignores_credentials_in_querystring(secure_client, client, cheshire_cat):
    token = await _jwt(secure_client, client)
    for credential in (token, api_key):
        for name in QUERY_PARAM_NAMES:
            with pytest.raises((WebSocketDisconnect, ExceptionGroup)):
                # no Authorization header: the credential is only in the URL
                await send_websocket_message({"text": "hey"}, secure_client, None, query_params={name: credential})

    # sanity check: the same token works in the Authorization header
    res = await send_websocket_message({"text": "hey"}, secure_client, token)
    assert res is not None
