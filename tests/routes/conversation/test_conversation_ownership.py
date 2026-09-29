"""The chat id is chosen by the client: a conversation belongs to the first user of its id, and the other users of the
agent can never use it (files, deletion, episodic memory), whatever id they send."""
import asyncio

import pytest

from cat import AuthPermission, AuthResource
from cat.auth.permissions import get_base_permissions
from cat.db.cruds import conversations as crud_conversations, settings as crud_settings
from cat.db.database import DEFAULT_CONVERSATIONS_KEY, get_async_db
from cat.services.memory.messages import ConversationMessage

from tests.utils import agent_id, api_key, create_new_user, send_file, send_websocket_message

CHAT = "shared-looking-chat"


def permissions():
    granted = get_base_permissions()
    granted[str(AuthResource.MEMORY)] = [str(p) for p in AuthPermission]
    granted[str(AuthResource.UPLOAD)] = [str(AuthPermission.WRITE)]
    return granted


async def two_users(client):
    admin = {"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id}
    response = await client.put("/file_manager/settings/LocalFileManagerConfig", headers=admin, json={})
    assert response.status_code == 200
    alice = await create_new_user(client, "alice", headers=admin, permissions=permissions())
    bob = await create_new_user(client, "bobby", headers=admin, permissions=permissions())
    return admin, {**admin, "X-User-ID": alice["id"]}, {**admin, "X-User-ID": bob["id"]}, alice, bob


async def test_another_user_cannot_use_the_conversation(secure_client, cheshire_cat):
    admin, alice, bob, _, bob_user = await two_users(secure_client)

    # alice uploads a file in her conversation, before writing any message in it
    response, _ = await send_file("sample.txt", "text/plain", secure_client, alice, ch_id=CHAT)
    assert response.status_code == 200
    alice_chat = {**alice, "X-Chat-ID": CHAT}
    files = (await secure_client.get("/file_manager/", headers=alice_chat)).json()["files"]
    assert [f["name"] for f in files] == ["sample.txt"]

    # bob sends the same chat id: every access is refused, as for credentials that are not valid (as for an agent
    # that does not exist: nothing tells that the conversation exists)
    bob_chat = {**bob, "X-Chat-ID": CHAT}
    for method, url, kwargs in (
        ("GET", "/file_manager/", {}),
        ("GET", "/file_manager/files/sample.txt", {}),
        ("DELETE", "/file_manager/files/sample.txt", {}),
        ("DELETE", "/file_manager/files", {}),
        ("DELETE", f"/conversations/{CHAT}", {}),
        ("GET", f"/conversations/{CHAT}/history", {}),
        ("POST", "/message", {"json": {"text": "what did alice upload?"}}),
    ):
        response = await secure_client.request(method, url, headers=bob_chat, **kwargs)
        assert response.status_code == 401, (method, url, response.status_code)
    with pytest.raises(Exception):
        await send_websocket_message(
            {"text": "hello"}, secure_client, api_key, ch_id=CHAT, query_params={"user_id": bob_user["id"]}
        )

    # nothing of alice was touched; the owner and the system administrators still access the conversation
    for headers in (alice_chat, {**admin, "X-Chat-ID": CHAT}):
        files = (await secure_client.get("/file_manager/", headers=headers)).json()["files"]
        assert [f["name"] for f in files] == ["sample.txt"]
    assert await crud_conversations.get_owner(agent_id, CHAT) == alice["X-User-ID"]


async def test_the_conversation_is_free_again_once_deleted(secure_client, cheshire_cat):
    _, alice, bob, _, _ = await two_users(secure_client)
    assert (await secure_client.get("/file_manager/", headers={**alice, "X-Chat-ID": CHAT})).status_code == 200
    assert (await secure_client.delete(f"/conversations/{CHAT}", headers={**alice, "X-Chat-ID": CHAT})).status_code == 200
    assert await crud_conversations.get_owner(agent_id, CHAT) is None
    assert (await secure_client.get("/file_manager/", headers={**bob, "X-Chat-ID": CHAT})).status_code == 200
    assert await crud_conversations.get_owner(agent_id, CHAT) == bob["X-User-ID"]


async def test_system_administrators_do_not_take_the_conversation(secure_client, cheshire_cat):
    admin, alice, _, _, _ = await two_users(secure_client)
    assert (await secure_client.get("/file_manager/", headers={**admin, "X-Chat-ID": CHAT})).status_code == 200
    assert await crud_conversations.get_owner(agent_id, CHAT) is None
    assert (await secure_client.get("/file_manager/", headers={**alice, "X-Chat-ID": CHAT})).status_code == 200


async def test_the_first_user_wins(client):
    owners = await asyncio.gather(*(crud_conversations.claim_conversation(agent_id, "race", f"user-{i}") for i in range(20)))
    assert len(set(owners)) == 1
    assert await crud_conversations.get_owner(agent_id, "race") == owners[0]


async def test_chat_ids_are_not_patterns(client):
    # regression: the chat id went into a pattern of keys: "*" matched the conversations of every user
    message = ConversationMessage(who="user", when=1.0, content={"text": "hi"})
    await crud_conversations.update_messages(agent_id, "alice", "c1", message)
    await crud_conversations.update_messages(agent_id, "bob", "c2", message)
    for chat_id in ("*", "c?", "c[12]"):
        assert await crud_conversations.get_user_id_from_conversation_keys(agent_id, chat_id) is None
        assert await crud_conversations.claim_conversation(agent_id, chat_id, "carol") == "carol"


async def test_owners_go_with_the_agent_and_are_not_cloned(client):
    await crud_conversations.claim_conversation(agent_id, "c1", "alice")
    await crud_settings.clone_agent(agent_id, "clone_of_agent", [DEFAULT_CONVERSATIONS_KEY])
    assert await crud_conversations.get_owner("clone_of_agent", "c1") is None
    await crud_conversations.destroy_all(agent_id)
    assert await crud_conversations.get_owner(agent_id, "c1") is None
    assert [k async for k in get_async_db().scan_iter(f"agents:{agent_id}:*owners*")] == []


async def test_a_conversation_released_while_claiming_is_claimed_again(client, monkeypatch):
    await crud_conversations.claim_conversation(agent_id, "c1", "alice")
    db = get_async_db()
    real_get, released = db.get, []

    async def get(key):
        if not released:
            # the owner deletes the conversation right after the attempt of bob
            released.append(True)
            await db.delete(crud_conversations.owner_key(agent_id, "c1"))
            return None
        return await real_get(key)

    monkeypatch.setattr(crud_conversations, "get_async_db", lambda: type("Db", (), {"set": db.set, "get": staticmethod(get)})())
    assert await crud_conversations.claim_conversation(agent_id, "c1", "bob") == "bob"


async def test_an_administrator_deleting_a_conversation_deletes_the_history_of_its_owner(secure_client, cheshire_cat):
    admin, alice, _, _, _ = await two_users(secure_client)
    message = ConversationMessage(who="user", when=1.0, content={"text": "hi"})
    assert (await secure_client.get("/file_manager/", headers={**alice, "X-Chat-ID": CHAT})).status_code == 200
    await crud_conversations.update_messages(agent_id, alice["X-User-ID"], CHAT, message)

    assert (await secure_client.delete(f"/conversations/{CHAT}", headers={**admin, "X-Chat-ID": CHAT})).status_code == 200
    assert await crud_conversations.get_conversation(agent_id, alice["X-User-ID"], CHAT) is None
    assert await crud_conversations.get_owner(agent_id, CHAT) is None


async def test_the_chats_of_a_user(client):
    for chat_id, user_id in (("a1", "alice"), ("a*2", "alice"), ("b1", "bob")):
        await crud_conversations.claim_conversation(agent_id, chat_id, user_id)
    assert sorted(await crud_conversations.get_owned_chats(agent_id, "alice")) == ["a*2", "a1"]
    assert await crud_conversations.get_owned_chats(agent_id, "carol") == []
    assert await crud_conversations.get_owned_chats("agent-without-chats", "alice") == []
