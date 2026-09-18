import pytest
from pydantic import ValidationError

from cat.auth.permissions import AuthResource, AuthPermission
from cat.db.cruds import conversations as crud_conversations
from cat.routes.users import UserBase, UserUpdate
from cat.services.memory.messages import ConversationMessage, UserMessage

from tests.utils import agent_id, api_key, create_new_user, check_user_fields, fake_timestamp, new_user_password


def test_validation_errors():
    with pytest.raises(ValidationError):
        UserBase(username="A", permissions={})

    user = UserUpdate(username="Alice")
    assert isinstance(user, UserUpdate)
    assert user.username == "Alice"

    with pytest.raises(ValidationError):
        UserUpdate(username="Alice", permissions={"READ": []})
    with pytest.raises(ValidationError):
        UserUpdate(username="Alice", permissions={"STATUS": ["WRITE", "WRONG"]})


async def test_create_user(secure_client, secure_client_headers, cheshire_cat):
    # create user
    data = await create_new_user(secure_client, headers=secure_client_headers)

    # assertions on user structure
    check_user_fields(data)

    assert data["username"] == "Alice"
    assert len(data["permissions"]) == 1


async def test_cannot_create_user_because_of_wrong_resource(client, secure_client, secure_client_headers, cheshire_cat):
    # create new user
    creds = {"username": "Alice", "password": "Alice"}

    # let's create the user
    res = await secure_client.post(
        "/users/",
        json=creds | {"permissions": {str(AuthResource.CHESHIRE_CAT): [str(AuthPermission.WRITE)]}},
        headers=secure_client_headers,
    )
    assert res.status_code == 400


async def test_cannot_create_duplicate_user(secure_client, secure_client_headers, cheshire_cat):
    # create user
    data = await create_new_user(secure_client, headers=secure_client_headers)

    # create user with same username
    response = await secure_client.post(
        "/users/", json={"username": data["username"], "password": "ecilA"}, headers=secure_client_headers
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Cannot duplicate user"


async def test_cannot_create_duplicate_user_by_id(secure_client, secure_client_headers, cheshire_cat):
    # create user
    data = await create_new_user(secure_client, headers=secure_client_headers)

    # create user with same username
    response = await secure_client.post(
        "/users/", json={"username": data["username"], "password": "2ecilA"}, headers=secure_client_headers
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Cannot duplicate user"


async def test_get_users(secure_client, secure_client_headers, cheshire_cat):
    # get list of users
    response = await secure_client.get("/users/", headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0

    # create user
    await create_new_user(secure_client, headers=secure_client_headers)

    # get updated list of users
    response = await secure_client.get("/users/", headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1

    # check users integrity and values
    for idx, d in enumerate(data):
        check_user_fields(d)
        assert d["username"] == "Alice"
        assert len(d["permissions"]) == 1


async def test_get_user(secure_client, secure_client_headers, cheshire_cat):
    # get unexisting user
    response = await secure_client.get("/users/wrong_user_id", headers=secure_client_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # create user and obtain id
    user_id = (await create_new_user(secure_client, headers=secure_client_headers))["id"]

    # get specific existing user
    response = await secure_client.get(f"/users/{user_id}", headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()

    # check user integrity and values
    check_user_fields(data)
    assert data["username"] == "Alice"
    assert len(data["permissions"]) == 1


async def test_update_user(secure_client, secure_client_headers, cheshire_cat):
    # update unexisting user
    response = await secure_client.put("/users/non_existent_id", json={"username": "Red Queen"}, headers=secure_client_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # create user and obtain id
    user_id = (await create_new_user(secure_client, headers=secure_client_headers))["id"]

    # update unexisting attribute (bad request)
    updated_user = {"username": "Alice", "something": 42}
    response = await secure_client.put(f"/users/{user_id}", json=updated_user, headers=secure_client_headers)
    assert response.status_code == 400

    response = await secure_client.get(f"/users/{user_id}", headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice"
    assert len(data["permissions"]) == 1
    
    # update password
    updated_user = {"username": data["username"], "password": "12345", "permissions": data["permissions"]}
    response = await secure_client.put(f"/users/{user_id}", json=updated_user, headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice"
    assert len(data["permissions"]) == 1
    assert "password" not in data # api will not send passwords around
    
    # change username
    updated_user = {"username": "Alice2", "permissions": data["permissions"]}
    response = await secure_client.put(f"/users/{user_id}", json=updated_user, headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice2"
    assert len(data["permissions"]) == 1

    # change permissions
    updated_user = {"username": data["username"], "permissions": {"MEMORY": ["READ"]}}
    response = await secure_client.put(f"/users/{user_id}", json=updated_user, headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice2"
    assert data["permissions"] == {"MEMORY": ["READ"]}

    # change username and permissions
    updated_user = {"username": "Alice3", "permissions": {"UPLOAD":["WRITE"]}}
    response = await secure_client.put(f"/users/{user_id}", json=updated_user, headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    check_user_fields(data)
    assert data["username"] == "Alice3"
    assert data["permissions"] == {"UPLOAD":["WRITE"]}

    # get list of users, should be admin, user and Alice3
    response = await secure_client.get("/users/", headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    for d in data:
        check_user_fields(d)
        assert d["username"] == "Alice3"
        assert d["permissions"] == {"UPLOAD": ["WRITE"]}


async def test_delete_user(secure_client, secure_client_headers, cheshire_cat):
    # delete unexisting user
    response = await secure_client.delete("/users/non_existent_id", headers=secure_client_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # create user and obtain id
    user_id = (await create_new_user(secure_client, headers=secure_client_headers))["id"]

    # delete user
    response = await secure_client.delete(f"/users/{user_id}", headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id

    # check that the user is not in the db anymore
    response = await secure_client.get(f"/users/{user_id}", headers=secure_client_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    # check user is no more in the list of users
    response = await secure_client.get("/users/", headers=secure_client_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0

    # check there is no conversation related to the user
    conversations = await crud_conversations.get_conversations_attributes(agent_id=agent_id, user_id=user_id)
    assert conversations == []


# note: using secure secure_client (api key set both for http and ws)
async def test_no_access_if_api_keys_active(secure_client, secure_client_headers, cheshire_cat):
    # create user (forbidden)
    response = await secure_client.post(
        "/users/",
        json={"username": "Alice", "password": new_user_password},
        headers={"X-Agent-ID": agent_id}
    )
    assert response.status_code == 401

    # read users (forbidden)
    response = await secure_client.get("/users/", headers={"X-Agent-ID": agent_id})
    assert response.status_code == 401

    # edit user (forbidden)
    response = await secure_client.put(
        "/users/non_existent_id", # it does not exist, but request should be blocked before the check
        json={"username": "Alice"},
        headers={"X-Agent-ID": agent_id}
    )
    assert response.status_code == 401

    # check default list giving the correct CAT_API_KEY
    headers = {"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id}
    response = await secure_client.get("/users/", headers=headers)
    assert response.status_code == 200
    assert len(response.json()) == 0


async def test_delete_user_deletes_their_conversations(secure_client, secure_client_headers, cheshire_cat):
    """Deleting a user must also drop every conversation of that user.

    The conversation keys are per-chat (``agents:<id>:conversations:<user>:<chat>``),
    so the removal has to be pattern-based: an exact-key delete with a literal
    ``*`` in it matches nothing and leaves the history behind.
    """
    user_id = (await create_new_user(secure_client, headers=secure_client_headers))["id"]
    other_user_id = (await create_new_user(secure_client, username="Bobby", headers=secure_client_headers))["id"]

    def _message(text: str) -> ConversationMessage:
        return ConversationMessage(who="user", when=fake_timestamp, content=UserMessage(text=text))

    # two conversations for the user to be deleted, one for another user
    for chat_id in ("chat_one", "chat_two"):
        await crud_conversations.set_messages(
            agent_id, user_id, chat_id, [_message(f"hello from {chat_id}")]
        )
    await crud_conversations.set_messages(
        agent_id, other_user_id, "chat_other", [_message("hello")]
    )

    assert len(await crud_conversations.get_conversations_attributes(agent_id=agent_id, user_id=user_id)) == 2

    response = await secure_client.delete(f"/users/{user_id}", headers=secure_client_headers)
    assert response.status_code == 200

    # every conversation of the deleted user is gone...
    assert await crud_conversations.get_conversations_attributes(agent_id=agent_id, user_id=user_id) == []
    for chat_id in ("chat_one", "chat_two"):
        assert await crud_conversations.get_conversation(agent_id, user_id, chat_id) is None

    # ...and the other user's conversation is untouched
    assert await crud_conversations.get_conversation(agent_id, other_user_id, "chat_other") is not None
