from cat.auth.permissions import AuthPermission, AuthResource
from cat.routes.routes_utils import has_write_permission, mask_secret_values
from tests.utils import create_new_user, new_user_password, agent_id


def test_has_write_permission():
    # WRITE on the same resource reveals secrets
    granted = {"PLUGIN": ["READ", "WRITE"]}
    assert has_write_permission(granted, AuthResource.PLUGIN)

    # stringified enum keys/values work too (that's how permissions are stored in Redis)
    perms = {str(AuthResource.PLUGIN): [str(AuthPermission.READ), str(AuthPermission.WRITE)]}
    assert has_write_permission(perms, AuthResource.PLUGIN)

    # READ-only, other resources, empty dicts and None never reveal
    readonly = {"PLUGIN": ["READ"]}
    assert not has_write_permission(readonly, AuthResource.PLUGIN)
    other = {"LLM": ["WRITE"]}
    assert not has_write_permission(other, AuthResource.PLUGIN)
    assert not has_write_permission({}, AuthResource.PLUGIN)
    assert not has_write_permission(None, AuthResource.PLUGIN)

    # stringly-fied enum keys/values are equivalent to raw strings
    assert has_write_permission(
        {str(AuthResource.PLUGIN): [str(AuthPermission.READ), str(AuthPermission.WRITE)]},
        AuthResource.PLUGIN,
    )


def test_mask_secret_values():
    value = {
        "model": "gpt-4o",
        "openai_api_key": "sk-123",
        "db_password": "pw",
        "other_secret": "s",
        "plain": "x",
        "empty_secret": "",
        "num": 1,
    }

    masked = mask_secret_values(value, reveal=False)

    # secret-suffixed non-empty strings are masked
    assert masked["openai_api_key"] == "********"
    assert masked["db_password"] == "********"
    assert masked["other_secret"] == "********"

    # non-secret keys, empty strings and non-strings are untouched
    assert masked["model"] == "gpt-4o"
    assert masked["plain"] == "x"
    assert masked["empty_secret"] == ""
    assert masked["num"] == 1

    # input dict is not mutated, writers see real values, non-dicts pass through
    assert value["openai_api_key"] == "sk-123"
    unmasked = mask_secret_values(value, reveal=True)
    assert unmasked["openai_api_key"] == "sk-123"
    assert mask_secret_values("not-a-dict", False) == "not-a-dict"


async def test_read_only_user_can_list_llm_settings(secure_client, secure_client_headers, client, cheshire_cat):
    # a user with LLM READ but no WRITE can still list settings (masking applies to secrets, GET keeps working)
    data = await create_new_user(
        secure_client,
        headers=secure_client_headers,
        permissions={str(AuthResource.LLM): [str(AuthPermission.READ)]},
    )
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]
    response = await client.get(
        "/llm/settings",
        headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id},
    )
    assert response.status_code == 200