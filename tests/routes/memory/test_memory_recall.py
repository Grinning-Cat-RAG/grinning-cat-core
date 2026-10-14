from cat.auth.permissions import get_base_permissions
from cat.db.cruds import users as crud_users
from cat.services.memory.models import VectorMemoryType

from tests.utils import send_n_websocket_messages, agent_id, create_new_user, api_key


# search on default startup memory
async def test_memory_recall_default_success(secure_client, secure_client_headers, cheshire_cat):
    params = {"text": "Red Queen"}
    response = await secure_client.get("/memory/recall", params=params, headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200

    # query was received and embedded
    assert json["query"]["text"] == params["text"]
    assert isinstance(json["query"]["vector"], list)
    assert isinstance(json["query"]["vector"][0], float)

    # results are grouped by collection
    assert len(json["vectors"]["collections"]) == 1
    assert str(VectorMemoryType.DECLARATIVE) == list(json["vectors"]["collections"].keys())[0]

    # initial collections contents
    assert isinstance(json["vectors"]["collections"][str(VectorMemoryType.DECLARATIVE)], list)
    assert len(json["vectors"]["collections"][str(VectorMemoryType.DECLARATIVE)]) == 0


# search without query should throw error
async def test_memory_recall_without_query_error(secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.get("/memory/recall", headers=secure_client_headers)
    assert response.status_code == 422


# search with query
async def test_memory_recall_success(secure_client, secure_client_headers, mocked_default_llm_answer_prompt, cheshire_cat):
    user = await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )
    # send a few messages via chat
    num_messages = 3
    await send_n_websocket_messages(num_messages, secure_client, query_params={"user_id": user["id"]})

    user_db = await crud_users.get_user_by_username(agent_id, "user")
    assert user_db["username"] == user["username"]

    # recall
    params = {"text": "Red Queen"}
    response = await secure_client.get(
        "/memory/recall", params=params, headers={**secure_client_headers, **{"X-User-ID": user_db["id"]}}
    )
    assert response.status_code == 200


# search with query and k
async def test_memory_recall_with_k_success(
    secure_client, secure_client_headers, mocked_default_llm_answer_prompt, cheshire_cat,
):
    user = await create_new_user(
        secure_client,
        "user",
        headers={"Authorization": f"Bearer {api_key}", "X-Agent-ID": agent_id},
        permissions=get_base_permissions(),
    )
    # send a few messages via chat
    num_messages = 6
    await send_n_websocket_messages(num_messages, secure_client, query_params={"user_id": user["id"]})

    user_db = await crud_users.get_user_by_username(agent_id, "user")
    assert user_db["username"] == user["username"]

    # recall at max k memories
    max_k = 2
    params = {"k": max_k, "text": "Red Queen"}
    response = await secure_client.get(
        "/memory/recall", params=params, headers={**secure_client_headers, **{"X-User-ID": user_db["id"]}}
    )
    assert response.status_code == 200


async def test_memory_recall_with_metadata(
    secure_client, secure_client_headers, mocked_default_llm_answer_prompt, cheshire_cat,
):
    messages = [
        {
            "content": "MEOW_ONE",
            "metadata": {"key_first": "value_one", "key_second": "value_two"},
        },
        {
            "content": "MEOW_TWO",
            "metadata": {"key_first": "value_one", "key_second": "value_three"},
        },
        {
            "content": "MEOW_THREE",
            "metadata": {},
        }
    ]

    # insert a new points with metadata
    for req_json in messages:
        await secure_client.post("/memory/collections/declarative/points", json=req_json, headers=secure_client_headers)

    # recall with metadata
    params = {"text": "MEOW", "metadata": {"key_first": "value_one"}}
    response = await secure_client.get("/memory/recall", params=params, headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    declarative_memories = json["vectors"]["collections"][str(VectorMemoryType.DECLARATIVE)]
    assert len(declarative_memories) == 2

    # recall with metadata multiple keys in metadata
    params = {"text": "MEOW", "metadata": {"key_first": "value_one", "key_second": "value_two"}}
    response = await secure_client.get("/memory/recall", params=params, headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    declarative_memories = json["vectors"]["collections"][str(VectorMemoryType.DECLARATIVE)]
    assert len(declarative_memories) == 1
    assert declarative_memories[0]["page_content"] == "MEOW_ONE"
