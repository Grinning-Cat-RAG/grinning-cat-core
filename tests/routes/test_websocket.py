import json
import time
import uuid

import httpx
import pytest
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport
from starlette.websockets import WebSocketDisconnect

from cat.auth.permissions import get_base_permissions
from cat.db.cruds import users as crud_users

from tests.utils import (
    send_websocket_message,
    send_n_websocket_messages,
    agent_id,
    api_key,
    create_new_user,
)


def check_correct_websocket_reply(reply):
    for k in ["type", "text", "why"]:
        assert k in reply.keys()

    assert reply["type"] != "error"
    assert isinstance(reply["text"], str)
    assert "You did not configure" in reply["text"]

    # why
    why = reply["why"]
    assert {"input", "intermediate_steps", "memory"} == set(why.keys())
    assert isinstance(why["input"], str)
    assert isinstance(why["intermediate_steps"], list)
    assert isinstance(why["memory"], list)


async def test_websocket(secure_client, secure_client_headers, cheshire_cat):
    user = await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )

    msg = {"text": "It's late! It's late", "image": "tests/mocks/sample.png"}
    # send a websocket message
    res = await send_websocket_message(msg, secure_client, api_key, query_params={"user_id": user["id"]})
    content = json.loads(res["content"])

    assert content["agent_id"] == agent_id
    assert content["user_id"] is not None
    assert content["chat_id"] is not None
    check_correct_websocket_reply(content["message"])

    # check analytics
    response = await secure_client.get("/analytics/llm", headers=secure_client_headers)
    analytics = response.json()
    #{agent_id: {user_id: {chat_id: {llm_id: <content>}}}}
    assert isinstance(analytics, dict)
    assert len(analytics.keys()) == 1
    assert agent_id in analytics.keys()
    assert analytics[agent_id] is not None

    assert len(analytics[agent_id].keys()) == 1
    user_id = list(analytics[agent_id].keys())[0]
    assert user_id is not None
    assert isinstance(analytics[agent_id][user_id], dict)

    chat_id = list(analytics[agent_id][user_id].keys())[0]
    assert chat_id is not None
    assert isinstance(analytics[agent_id][user_id][chat_id], dict)

    llm_id = list(analytics[agent_id][user_id][chat_id].keys())[0]
    assert llm_id == "default_llm"

    # no tokens used since no valid LLM was configured and, then, no LLM call was made
    info = analytics[agent_id][user_id][chat_id][llm_id]
    assert "input_tokens" in info.keys()
    assert info["input_tokens"] == 0
    assert "output_tokens" in info.keys()
    assert info["output_tokens"] == 0
    assert "total_tokens" in info.keys()
    assert info["total_tokens"] == 0
    assert "total_calls" in info.keys()
    assert info["total_calls"] == 1


async def test_websocket_with_additional_items_in_message(secure_client, cheshire_cat):
    user = await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )

    msg = {
        "text": "It's late! It's late",
        "image": "tests/mocks/sample.png",
        "prompt_settings": {"temperature": 0.5}
    }
    # send websocket message
    res = await send_websocket_message(msg, secure_client, api_key, query_params={"user_id": user["id"]})
    content = json.loads(res["content"])

    assert content["agent_id"] == agent_id
    assert content["user_id"] is not None
    assert content["chat_id"] is not None
    check_correct_websocket_reply(content["message"])


async def test_websocket_with_non_saved_user(secure_client, cheshire_cat):
    with pytest.raises((WebSocketDisconnect, ExceptionGroup)):
        mocked_user_id = uuid.uuid4()
        user = await crud_users.get_user(agent_id, str(mocked_user_id))
        assert user is None

        msg = {"text": "It's late! It's late", "image": "tests/mocks/sample.png"}
        await send_websocket_message(msg, secure_client, api_key, {"user_id": mocked_user_id})


async def test_websocket_multiple_messages(secure_client, cheshire_cat):
    user = await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )

    # send websocket message
    replies = await send_n_websocket_messages(3, secure_client, query_params={"user_id": user["id"]})

    for res in replies:
        content = json.loads(res["content"])
        check_correct_websocket_reply(content["message"])


async def test_websocket_multiple_connections(secure_client, secure_client_headers, lizard, cheshire_cat):
    mex = {"text": "It's late!"}

    data = await create_new_user(secure_client, username="Alice", headers=secure_client_headers)
    data2 = await create_new_user(secure_client, username="Caterpillar", headers=secure_client_headers)

    chat_id = str(uuid.uuid4())
    chat_id2 = str(uuid.uuid4())

    headers = {"Authorization": f"Bearer {api_key}"}
    app = secure_client._fastapi_test_app


    async with httpx.AsyncClient(transport=ASGIWebSocketTransport(app=app)) as ws_client:
        async with aconnect_ws(
                f"http://server/ws/{agent_id}/{chat_id}?user_id={data['id']}", ws_client, headers=headers
        ) as websocket:
            # send ws message
            await websocket.send_json(mex)

            async with aconnect_ws(
                    f"http://server/ws/{agent_id}/{chat_id2}?user_id={data2['id']}", ws_client, headers=headers
            ) as websocket2:
                # send ws message
                await websocket2.send_json(mex)
                # get reply
                reply2 = await websocket2.receive_json()
                content2 = json.loads(reply2["content"])

                # two connections open
                ws_chats = lizard.websocket_manager.connections.keys()
                assert set(ws_chats) == {chat_id, chat_id2}

            # one connection open
            time.sleep(0.5)
            ws_chats = lizard.websocket_manager.connections.keys()
            assert set(ws_chats) == {chat_id}

            # get reply
            reply = await websocket.receive_json()
            content = json.loads(reply["content"])

    check_correct_websocket_reply(content["message"])
    check_correct_websocket_reply(content2["message"])

    # websocket connection is closed
    time.sleep(0.5)
    assert lizard.websocket_manager.connections == {}


async def test_websocket_with_not_existing_agent(secure_client, cheshire_cat):
    app = secure_client._fastapi_test_app
    with pytest.raises(ExceptionGroup) as exc_info:
        async with httpx.AsyncClient(transport=ASGIWebSocketTransport(app=app)) as ws_client:
            async with aconnect_ws(
                "http://server/ws/not_existing_agent", ws_client, headers={"Authorization": f"Bearer {api_key}"},
            ) as websocket:
                await websocket.send_json({"text": "It's late! It's late"})
                await websocket.receive_json()

    # same answer as for invalid credentials: the existence of the agent is not disclosed
    assert [(e.code, e.reason) for e in exc_info.value.exceptions] == [(1008, "Unauthorized")]
