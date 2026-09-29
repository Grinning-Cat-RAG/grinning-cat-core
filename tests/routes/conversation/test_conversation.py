import asyncio

from cat import AuthResource, AuthPermission
from cat.auth.permissions import get_base_permissions
from cat.db.cruds import users as crud_users, conversations as crud_conversations

from tests.utils import (
    send_websocket_message,
    agent_id,
    api_key,
    create_new_user,
    new_user_password,
    chat_id,
)


async def test_convo_history_absent(secure_client, secure_client_headers, cheshire_cat):
    # no ws connection, so no convo history available
    response = await secure_client.get(f"/conversations/{chat_id}/history", headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    assert "history" in json
    assert len(json["history"]) == 0


async def test_convo_history_no_update_invalid_llm(secure_client, secure_client_headers, cheshire_cat):
    user = await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )
    message = "It's late! It's late!"

    # send websocket messages
    await send_websocket_message(
        {"text": message}, secure_client, api_key, ch_id=chat_id, query_params={"user_id": user["id"]}
    )

    # check conversation history update
    response = await secure_client.get(f"/conversations/{chat_id}/history", headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    assert "history" in json
    assert len(json["history"]) == 0


async def test_convo_history_update(secure_client, secure_client_headers, mocked_default_llm_answer_prompt, cheshire_cat):
    user = await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )
    message = "It's late! It's late!"

    # send websocket messages
    await send_websocket_message(
        {"text": message}, secure_client, api_key, ch_id=chat_id, query_params={"user_id": user["id"]}
    )
    user = await crud_users.get_user_by_username(agent_id, "user")

    # check conversation history update
    response = await secure_client.get(
        f"/conversations/{chat_id}/history", headers={**secure_client_headers, **{"X-User-ID": user["id"]}}
    )
    json = response.json()
    assert response.status_code == 200
    assert "history" in json
    assert len(json["history"]) == 2  # mex and reply

    for item in json["history"]:
        assert "who" in item
        assert item["who"] in ["user", "assistant"]

        assert "content" in item
        assert "text" in item["content"]

        if item["who"] == "user":
            assert item["content"]["text"] == message
            assert "why" not in item["content"]
        else:
            assert "You did not configure" in item["content"]["text"]
            assert "why" in item["content"]

        assert "when" in item
        assert isinstance(item["when"], float)  # timestamp


async def test_convo_delete(secure_client, secure_client_headers, mocked_default_llm_answer_prompt, cheshire_cat):
    user = await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )

    # send websocket messages
    await send_websocket_message(
        {"text": "It's late! It's late!"},
        secure_client,
        api_key,
        ch_id=chat_id,
        query_params={"user_id": user["id"]},
    )
    user = await crud_users.get_user_by_username(agent_id, "user")

    # delete convo history
    response = await secure_client.delete(
        f"/conversations/{chat_id}", headers={**secure_client_headers, **{"X-User-ID": user["id"]}}
    )
    assert response.status_code == 200
    assert response.json()["deleted"] is True

    # check conversation history is empty
    response = await secure_client.get(
        f"/conversations/{chat_id}/history", headers={**secure_client_headers, **{"X-User-ID": user["id"]}}
    )
    json = response.json()
    assert response.status_code == 200
    assert "history" in json
    assert len(json["history"]) == 0

    # check there is not conversation with name = chat_id
    conversation = await crud_conversations.get_conversation(agent_id, user["id"], chat_id)
    assert conversation is None


async def test_convo_history_by_user(secure_client, secure_client_headers, client, mocked_default_llm_answer_prompt, cheshire_cat):
    convos = {
        # user_id: n_messages
        "White Rabbit": 2,
        "Alice": 3,
    }

    # a chat id belongs to its first user: every user has its own conversation
    chats = {"White Rabbit": f"{chat_id}-rabbit", "Alice": chat_id}
    tokens = {}
    users = {}
    # send websocket messages
    for username, n_messages in convos.items():
        data = await create_new_user(
            secure_client,
            username=username,
            headers=secure_client_headers,
            permissions={
                str(AuthResource.CHAT): [str(AuthPermission.READ), str(AuthPermission.WRITE)],
                str(AuthResource.MEMORY): [str(AuthPermission.READ)]
            }
        )
        res = await client.post(
            "/auth/token",
            json={"username": data["username"], "password": new_user_password},
        )
        received_token = res.json()["access_token"]
        tokens[username] = received_token
        users[username] = data

        for m in range(n_messages):
            await asyncio.sleep(0.1)
            await send_websocket_message(
                {"text": f"Mex n.{m} from {username}"},
                client,
                received_token,
                ch_id=chats[username],
            )

    # check conversation history
    for username, n_messages in convos.items():
        response = await client.get(
            f"/conversations/{chats[username]}/history",
            headers={"X-Agent-ID": agent_id, "Authorization": f"Bearer {tokens[username]}"},
        )
        json = response.json()
        assert response.status_code == 200
        assert "history" in json
        assert len(json["history"]) == n_messages * 2  # mex and reply
        for m_idx, m in enumerate(json["history"]):
            assert "who" in m
            assert "content" in m
            assert "text" in m["content"]
            if m_idx % 2 == 0:  # even message
                m_number_from_user = int(m_idx / 2)
                assert m["who"] == "user"
                assert m["content"]["text"] == f"Mex n.{m_number_from_user} from {username}"
            else:
                assert m["who"] == "assistant"

    # a user never reaches the conversation of another one
    response = await client.get(
        f"/conversations/{chats['White Rabbit']}/history",
        headers={"X-Agent-ID": agent_id, "Authorization": f"Bearer {tokens['Alice']}"},
    )
    assert response.status_code == 401

    # delete White Rabbit convo
    response = await client.delete(
        f"/conversations/{chats['White Rabbit']}",
        headers={"X-Agent-ID": agent_id, "Authorization": f"Bearer {tokens['White Rabbit']}"},
    )
    assert response.status_code == 403  # user has no permission
    response = await secure_client.delete(
        f"/conversations/{chats['White Rabbit']}",
        headers={"X-User-ID": users["White Rabbit"]["id"], "X-Agent-ID": agent_id, "Authorization": f"Bearer {api_key}"},
    )
    assert response.status_code == 200

    # check convo deletion per user
    ### White Rabbit convo is empty
    response = await secure_client.get(
        f"/conversations/{chats['White Rabbit']}/history",
        headers=secure_client_headers | {"X-User-ID": users["White Rabbit"]["id"]},
    )
    json = response.json()
    assert len(json["history"]) == 0
    ### Alice convo still the same
    response = await secure_client.get(
        f"/conversations/{chats['Alice']}/history",
        headers=secure_client_headers | {"X-User-ID": users["Alice"]["id"]},
    )
    json = response.json()
    assert len(json["history"]) == convos["Alice"] * 2


async def test_change_attributes_to_conversation(secure_client, secure_client_headers, client, cheshire_cat):
    user = await create_new_user(
        secure_client,
        username="Alice",
        headers=secure_client_headers,
        permissions={
            str(AuthResource.CHAT): [str(AuthPermission.READ), str(AuthPermission.WRITE)],
            str(AuthResource.MEMORY): [str(AuthPermission.WRITE)]
        }
    )
    res = await client.post(
        "/auth/token",
        json={"username": user["username"], "password": new_user_password},
    )
    received_token = res.json()["access_token"]

    await send_websocket_message(
        {"text": "Hello, Alice!"},
        client,
        received_token,
        ch_id=chat_id,
    )

    conversation = await crud_conversations.get_conversation(agent_id, user["id"], chat_id)
    assert conversation["name"] == chat_id
    assert conversation["metadata"] == {}

    # change the name of the conversation
    response = await secure_client.put(
        f"/conversations/{chat_id}",
        headers={**secure_client_headers, **{"X-User-ID": user["id"]}},
        json={"name": "this_is_a_new_name"}
    )
    assert response.status_code == 200

    conversation = await crud_conversations.get_conversation(agent_id, user["id"], chat_id)
    assert conversation["name"] == "this_is_a_new_name"
    assert conversation["metadata"] == {}

    # change the metadata of the conversation
    response = await secure_client.put(
        f"/conversations/{chat_id}",
        headers={**secure_client_headers, **{"X-User-ID": user["id"]}},
        json={"metadata": {"topic": "greetings"}}
    )
    assert response.status_code == 200

    conversation = await crud_conversations.get_conversation(agent_id, user["id"], chat_id)
    assert conversation["name"] == "this_is_a_new_name"
    assert conversation["metadata"] == {"topic": "greetings"}

    # change nothing
    response = await secure_client.put(
        f"/conversations/{chat_id}",
        headers={**secure_client_headers, **{"X-User-ID": user["id"]}},
    )
    assert response.status_code == 400


async def test_get_empty_conversations(secure_client, secure_client_headers, cheshire_cat):
    await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )

    user = await crud_users.get_user_by_username(agent_id, "user")

    # check conversation history update
    response = await secure_client.get(
        "/conversations/", headers={**secure_client_headers, **{"X-User-ID": user["id"]}}
    )
    assert response.status_code == 200

    json_response = response.json()
    assert isinstance(json_response, list)
    assert len(json_response) == 0  # no chats for this user


async def test_get_conversations(secure_client, secure_client_headers, mocked_default_llm_answer_prompt, cheshire_cat):
    await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )
    user = await crud_users.get_user_by_username(agent_id, "user")

    message = "It's late! It's late!"
    # send 3 messages to 3 different chats for the same user
    for _ in range(3):
        # send websocket messages
        await send_websocket_message(
            {"text": message}, secure_client, api_key, query_params={"user_id": user["id"]}
        )

    # check all the conversation histories
    response = await secure_client.get(
        "/conversations/", headers={**secure_client_headers, **{"X-User-ID": user["id"]}}
    )
    assert response.status_code == 200

    json_response = response.json()
    assert isinstance(json_response, list)
    assert len(json_response) == 3

    # sending two more messages to the `chat_id` chat
    for _ in range(2):
        # send websocket messages
        await send_websocket_message(
            {"text": message}, secure_client, api_key, query_params={"user_id": user["id"]}, ch_id=chat_id
        )

    # check again all the conversation histories
    response = await secure_client.get(
        "/conversations/", headers={**secure_client_headers, **{"X-User-ID": user["id"]}}
    )
    assert response.status_code == 200

    json_response = response.json()
    assert len(json_response) == 4

    # check that the `chat_id` chat has 4 messages (2 mex + 2 replies)
    for item in json_response:
        assert item["chat_id"] == item["name"]  # chat_id and name are the same
        ch_id = item["chat_id"]

        if ch_id != chat_id:
            assert item["num_messages"] == 2  # 1 mex + 1 reply
        else:
            assert item["num_messages"] == 4  # 2 mex + 2 replies


async def test_get_conversations_by_agent(secure_client, secure_client_headers, mocked_default_llm_answer_prompt, cheshire_cat):
    await test_get_conversations(secure_client, secure_client_headers, mocked_default_llm_answer_prompt, cheshire_cat)
    user = await crud_users.get_user_by_username(agent_id, "user")

    # check all the conversation histories by agent
    response = await secure_client.get(
        "/agents/conversations/", headers={**secure_client_headers, **{"X-User-ID": user["id"]}}
    )
    assert response.status_code == 200

    json_response = response.json()
    assert isinstance(json_response, dict)
    assert len(json_response) == 1  # only one agent
    assert user["id"] in json_response
    assert len(json_response[user["id"]]) == 4


async def test_get_conversation(secure_client, secure_client_headers, mocked_default_llm_answer_prompt, cheshire_cat):
    await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )
    user = await crud_users.get_user_by_username(agent_id, "user")

    message = "It's late! It's late!"
    # sending two messages to the `chat_id` chat
    for _ in range(2):
        # send websocket messages
        await send_websocket_message(
            {"text": message}, secure_client, api_key, query_params={"user_id": user["id"]}, ch_id=chat_id
        )

    # check again all the conversation histories
    response = await secure_client.get(
        f"/conversations/{chat_id}", headers={**secure_client_headers, **{"X-User-ID": user["id"]}}
    )
    json_response = response.json()

    assert response.status_code == 200
    assert json_response["chat_id"] == json_response["name"]  # chat_id and name are the same
    assert json_response["num_messages"] == 4  # 2 mex + 2 reply
