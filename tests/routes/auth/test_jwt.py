import os
import time
import jwt

from cat.env import get_env
from cat.auth.permissions import AuthPermission, AuthResource, get_base_permissions
from cat.auth.auth_utils import is_jwt, DEFAULT_JWT_ALGORITHM
from cat.auth.tokens import JWT_AUDIENCE, JWT_ISSUER

from tests.utils import agent_id, chat_id, create_new_user, api_key, new_user_password, http_message


def test_is_jwt():
    assert not is_jwt("not_a_jwt.not_a_jwt.not_a_jwt")

    actual_jwt = jwt.encode(
        {"username": "Alice"},
        "some_secret",
        algorithm=DEFAULT_JWT_ALGORITHM,
    )
    assert is_jwt(actual_jwt)


async def test_refuse_issue_jwt(secure_client, client, cheshire_cat):
    creds = {"username": "user", "password": "wrong"}

    await create_new_user(
        secure_client,
        creds["username"],
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )

    res = await client.post("/auth/token", json=creds)

    # wrong credentials
    assert res.status_code == 401
    json = res.json()
    assert json["detail"] == "Invalid Credentials"


async def test_issue_jwt(secure_client, client, cheshire_cat):
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

    res_json = res.json()

    # did we obtain a JWT?
    assert res_json["token_type"] == "bearer"
    received_token = res_json["access_token"]
    assert is_jwt(received_token)

    # is the JWT correct for core auth handler?
    user_info = await cheshire_cat.custom_auth_handler.authorize_user_from_jwt(
        received_token, AuthResource.CHAT, AuthPermission.READ, key_id=agent_id
    )
    assert len(user_info.id) == 36 and len(user_info.id.split("-")) == 5  # uuid4
    assert user_info.name == "user"

    # manual JWT verification
    try:
        payload = jwt.decode(
            received_token,
            get_env("CAT_JWT_SECRET"),
            algorithms=[DEFAULT_JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
        )
        assert payload["sub"] == "user"
        assert (payload["exp"] - time.time() < 60 * 60 * 24)  # expires in less than 24 hours
    except jwt.exceptions.DecodeError:
        assert False


async def test_issue_jwt_for_new_user(client, secure_client, secure_client_headers, cheshire_cat):
    # create new user
    creds = {"username": "Alice", "password": "Alice"}

    # we should not obtain a JWT for this user
    # because it does not exist
    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 401
    assert res.json()["detail"] == "Invalid Credentials"

    # let's create the user
    res = await secure_client.post(
        "/users/",
        json=creds | {"permissions": {str(AuthResource.LLM): [str(AuthPermission.WRITE)]}},
        headers=secure_client_headers,
    )
    assert res.status_code == 200

    # now we should get a JWT
    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 200

    res_json = res.json()

    # did we obtain a JWT?
    assert res_json["token_type"] == "bearer"
    received_token = res_json["access_token"]
    assert is_jwt(received_token)


# test token expiration after successful login
# NOTE: here we are using the secure_client fixture (see conftest.py)
async def test_jwt_expiration(secure_client, client, cheshire_cat):
    message = {"text": "hey"}

    # set ultrashort JWT expiration time
    current_jwt_expire_minutes = os.getenv("CAT_JWT_EXPIRE_MINUTES")
    os.environ["CAT_JWT_EXPIRE_MINUTES"] = "0.05"  # 3 seconds

    # not allowed
    status_code, response_json = await http_message(client, message, {"X-Agent-ID": agent_id, "X-Chat-ID": chat_id})
    assert status_code == 401
    assert response_json["detail"] == "Unauthorized"

    # request JWT
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
    token = res.json()["access_token"]

    # allowed via JWT
    headers = {"Authorization": f"Bearer {token}", "X-Agent-ID": agent_id, "X-Chat-ID": chat_id}
    status_code, _ = await http_message(client, message, headers)
    assert status_code == 200

    # wait for expiration time
    time.sleep(3)

    # not allowed because JWT expired
    headers = {"Authorization": f"Bearer {token}", "X-Agent-ID": agent_id, "X-Chat-ID": chat_id}
    status_code, response_json = await http_message(client, message, headers)
    assert status_code == 401
    assert response_json["detail"] == "Unauthorized"

    # restore default env
    if current_jwt_expire_minutes:
        os.environ["CAT_JWT_EXPIRE_MINUTES"] = current_jwt_expire_minutes
    else:
        del os.environ["CAT_JWT_EXPIRE_MINUTES"]


# test ws and http endpoints can get user_id from JWT
# NOTE: here we are using the secure_client fixture (see conftest.py)
async def test_jwt_imposes_user_id(secure_client, client, cheshire_cat):
    message = {"text": "hey"}

    # not allowed
    status_code, response_json = await http_message(client, message, {"X-Agent-ID": agent_id, "X-Chat-ID": chat_id})
    assert status_code == 401
    assert response_json["detail"] == "Unauthorized"

    # request JWT
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
    token = res.json()["access_token"]

    # send user specific message via http
    headers = {"Authorization": f"Bearer {token}", "X-Agent-ID": agent_id, "X-Chat-ID": chat_id}
    status_code, _ = await http_message(client, message, headers)
    assert status_code == 200
