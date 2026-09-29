from cat import AgenticWorkflowTask
from cat.db.cruds import users as crud_users
from cat.looking_glass import StrayCat
from cat.services.memory.messages import MessageWhy
from cat.services.memory.models import VectorMemoryType, RecallSettings
from cat.services.memory.working_memory import WorkingMemory

from tests.utils import api_key, send_file, http_message, just_installed_plugin


def test_stray_initialization(stray_no_memory):
    assert isinstance(stray_no_memory, StrayCat)
    # check that stray_no_memory.user.id is uuid4
    assert len(stray_no_memory.user.id) == 36
    assert isinstance(stray_no_memory.working_memory, WorkingMemory)


async def test_stray_nlp(lizard, stray_no_memory):
    agent_input = AgenticWorkflowTask(
        user_prompt="hey",
    )
    res = await stray_no_memory.agentic_workflow.run(
        task=agent_input,
        llm=stray_no_memory.large_language_model,
    )
    assert "You did not configure" in res.output

    embedding = (await lizard.embedder()).embed_documents(["hey"])
    assert isinstance(embedding[0], list)
    assert isinstance(embedding[0][0], float)


async def test_stray_call(secure_client, stray_no_memory):
    await crud_users.create_user(
        stray_no_memory.agent_key,
        {
            "id": stray_no_memory.user.id,
            "username": stray_no_memory.user.name,
            "password": "password123",
            "permissions": stray_no_memory.user.permissions
        },
    )

    ccat_headers = {
        "X-Agent-ID": stray_no_memory.agent_key,
        "Authorization": f"Bearer {api_key}",
        "X-Chat-ID": stray_no_memory.id,
        "X-User-ID": stray_no_memory.user.id,
    }

    # send message
    message = "what time is it?"
    status_code, response_json = await http_message(secure_client,{"text": message}, ccat_headers)

    assert status_code == 200
    assert response_json["agent_id"] == stray_no_memory.agent_key
    assert response_json["user_id"] == stray_no_memory.user.id
    assert response_json["chat_id"] == stray_no_memory.id
    assert response_json["message"]["type"] == "chat"
    assert "You did not configure" in response_json["message"]["text"]

    assert response_json["message"]["why"]["input"] == message
    assert response_json["message"]["why"]["intermediate_steps"] == []
    assert response_json["message"]["why"]["memory"] == []

    why = MessageWhy(**response_json["message"]["why"])
    assert isinstance(why, MessageWhy)


async def test_stray_recall_all_memories(secure_client, secure_client_headers, stray, lizard):
    await send_file("sample.pdf", "application/pdf", secure_client, secure_client_headers)

    query = (await lizard.embedder()).embed_query("")
    memories = await stray.context_retriever.run(
        VectorMemoryType.DECLARATIVE, RecallSettings(embedding=query, k=None)
    )

    assert len(memories) > 0
    for mem in memories:
        assert mem.score is None
        assert isinstance(mem.vector, list)


async def test_stray_recall_by_metadata(secure_client, secure_client_headers, stray, lizard):
    content_type = "application/pdf"
    query = (await lizard.embedder()).embed_query("late")

    file_name = "sample.pdf"
    _, file_path = await send_file(file_name, content_type, secure_client, secure_client_headers)
    memories = await stray.context_retriever.run(
        VectorMemoryType.DECLARATIVE,
        RecallSettings(threshold=0.1, embedding=query, metadata={"source": file_name}),
    )
    assert len(memories) > 0
    for mem in memories:
        assert mem.document.metadata["source"] == file_name

    with open(file_path, "rb") as f:
        files = {"file": ("sample2.pdf", f, content_type)}
        _ = await secure_client.post("/rabbithole/", files=files, headers=secure_client_headers)

    memories = await stray.context_retriever.run(
        VectorMemoryType.DECLARATIVE,
        RecallSettings(threshold=0.1, embedding=query, metadata={"source": file_name}),
    )
    assert len(memories) > 0
    for mem in memories:
        assert mem.document.metadata["source"] == file_name


async def test_stray_fast_reply_hook(secure_client, secure_client_headers, cheshire_cat, stray):
    await crud_users.create_user(
        stray.agent_key,
        {
            "id": stray.user.id,
            "username": stray.user.name,
            "password": "password123",
            "permissions": stray.user.permissions
        },
    )

    await just_installed_plugin(secure_client, secure_client_headers, plugin_id="mock_plugin_fast_reply")

    # activate for the new agent
    ccat_headers = {"X-Agent-ID": stray.agent_key, "Authorization": f"Bearer {api_key}"}
    await secure_client.put("/plugins/toggle/mock_plugin_fast_reply", headers=ccat_headers)

    # send message
    status_code, response_json = await http_message(
        secure_client, {"text": "hello"}, ccat_headers | {"X-Chat-ID": stray.id, "X-User-ID": stray.user.id},
    )
    assert status_code == 200
    assert response_json["agent_id"] == stray.agent_key
    assert response_json["user_id"] == stray.user.id
    assert response_json["chat_id"] == stray.id
    assert response_json["message"]["type"] == "chat"
    assert response_json["message"]["text"] == "This is a fast reply"

    # there should be NO side effects
    upd_stray = await StrayCat.from_cat(user_data=stray.user, cat=cheshire_cat, stray_id=stray.id)
    assert len(upd_stray.working_memory.history) == 0


async def test_stray_agent_fast_reply_goes_through_the_plugins(secure_client, secure_client_headers, cheshire_cat, stray):
    # an agent fast reply skips the agent, not the plugins: they see (and e.g. trace) the answer, and the conversation
    # history gets it (otherwise the user message stays without an answer)
    await crud_users.create_user(
        stray.agent_key,
        {
            "id": stray.user.id,
            "username": stray.user.name,
            "password": "password123",
            "permissions": stray.user.permissions
        },
    )

    await just_installed_plugin(secure_client, secure_client_headers, plugin_id="mock_plugin_agent_fast_reply")

    ccat_headers = {"X-Agent-ID": stray.agent_key, "Authorization": f"Bearer {api_key}"}
    await secure_client.put("/plugins/toggle/mock_plugin_agent_fast_reply", headers=ccat_headers)

    status_code, response_json = await http_message(
        secure_client, {"text": "hello"}, ccat_headers | {"X-Chat-ID": stray.id, "X-User-ID": stray.user.id},
    )
    assert status_code == 200
    assert response_json["message"]["text"] == "This is an agent fast reply (seen by the plugins)"

    upd_stray = await StrayCat.from_cat(user_data=stray.user, cat=cheshire_cat, stray_id=stray.id)
    history = [(m.who, m.content.text) for m in upd_stray.working_memory.history]
    assert history == [("user", "hello"), ("assistant", "This is an agent fast reply (seen by the plugins)")]
