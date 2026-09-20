import os
import pytest

from cat.auth.permissions import get_base_permissions

from tests.utils import agent_id, create_new_user, api_key, new_user_password


@pytest.fixture
def low_limits():
    saved = {k: os.environ.get(k) for k in ("CAT_AUTH_MAX_FAILURES_PER_USER", "CAT_AUTH_MAX_ATTEMPTS_PER_IP")}
    os.environ["CAT_AUTH_MAX_FAILURES_PER_USER"] = "3"
    os.environ["CAT_AUTH_MAX_ATTEMPTS_PER_IP"] = "6"
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


async def test_login_rate_limited_per_username(secure_client, client, cheshire_cat, low_limits):
    await create_new_user(
        secure_client, "user", headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(), password=new_user_password,
    )
    for _ in range(3):
        assert (await client.post("/auth/token", json={"username": "user", "password": "wrong"})).status_code == 401

    # locked, even with the right password
    res = await client.post("/auth/token", json={"username": "user", "password": new_user_password})
    assert res.status_code == 429
    assert int(res.headers["retry-after"]) > 0


async def test_login_rate_limited_per_ip(secure_client, client, cheshire_cat, low_limits):
    codes = [
        (await client.post("/auth/token", json={"username": f"nobody{i}", "password": "x"})).status_code
        for i in range(7)
    ]
    assert codes[:6] == [401] * 6 and codes[6] == 429


async def _token(secure_client, client):
    await create_new_user(
        secure_client, "user", headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(), password=new_user_password,
    )
    res = await client.post("/auth/token", json={"username": "user", "password": new_user_password})
    return res.json()["access_token"]


async def test_cookie_auth_refused_from_foreign_origin(secure_client, client, cheshire_cat):
    token = await _token(secure_client, client)
    client.cookies.set("jwt", token)
    try:
        assert (await client.get("/me")).status_code == 200  # no Origin (same-origin / non browser)
        assert (await client.get("/me", headers={"Origin": "http://test"})).status_code == 200  # same origin
        assert (await client.get("/me", headers={"Origin": "https://evil.example"})).status_code == 401
    finally:
        client.cookies.clear()


async def test_bearer_not_affected_by_origin(secure_client, client, cheshire_cat):
    token = await _token(secure_client, client)
    res = await client.get("/me", headers={"Authorization": f"Bearer {token}", "Origin": "https://evil.example"})
    assert res.status_code == 200
