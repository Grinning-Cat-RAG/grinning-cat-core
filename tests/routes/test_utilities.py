import uuid
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from cat.db.cruds import (
    settings as crud_settings,
    conversations as crud_conversations,
    plugins as crud_plugins,
    users as crud_users,
)
from cat.db.database import get_async_db
from cat.env import get_env
from cat.services.memory.models import VectorMemoryType

from tests.utils import create_new_user, get_client_admin_headers, new_user_password, api_key


async def checks_on_agent_create(lizard, new_agent_id):
    settings = await crud_settings.get_settings(new_agent_id)
    assert len(settings) > 0

    histories = await get_async_db().get(crud_conversations.format_key(new_agent_id, "*", "*"))
    assert histories is None

    plugins = await crud_plugins.get_settings(new_agent_id)
    assert plugins == {}

    users = await crud_users.get_users(new_agent_id)
    assert len(users) == 0

    ccat = await lizard.get_cheshire_cat(new_agent_id)
    num_vectors = await ccat.vector_memory_handler.get_tenant_vectors_count(str(VectorMemoryType.DECLARATIVE))
    points, _ = await ccat.vector_memory_handler.get_all_tenant_points(str(VectorMemoryType.DECLARATIVE))
    assert num_vectors == 0
    assert len(points) == 0


async def checks_on_agent_reset(res, client, ccat_id, lizard):
    assert res.status_code == 200

    received_token = res.json()["access_token"]
    response = await client.post(
        "/utils/agents/reset", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": ccat_id}
    )

    assert response.status_code == 200
    assert response.json() == {"deleted_settings": True, "deleted_memories": True, "deleted_plugin_folders": False}

    await checks_on_agent_create(lizard, ccat_id)


async def checks_on_agent_destroy(cheshire_cat):
    settings = await crud_settings.get_settings(cheshire_cat.agent_key)
    assert len(settings) > 0

    collections = await cheshire_cat.vector_memory_handler._client.get_collections()
    assert len(collections.collections) > 0

    num_vectors = await cheshire_cat.vector_memory_handler.get_tenant_vectors_count(str(VectorMemoryType.DECLARATIVE))
    points, _ = await cheshire_cat.vector_memory_handler.get_all_tenant_points(str(VectorMemoryType.DECLARATIVE))
    assert num_vectors == 0
    assert len(points) == 0


async def test_factory_reset_success(client, lizard, cheshire_cat):
    # check that the vector database is not empty
    c = await cheshire_cat.vector_memory_handler._client.get_collections()
    assert len(c.collections) > 0

    creds = {
        "username": "admin",
        "password": get_env("CAT_ADMIN_DEFAULT_PASSWORD"),
    }

    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 200

    received_token = res.json()["access_token"]
    response = await client.post(
        "/utils/factory/reset", headers={"Authorization": f"Bearer {received_token}"}
    )

    assert response.status_code == 200
    assert response.json() == {"deleted_settings": True, "deleted_memories": True, "deleted_plugin_folders": True}

    settings = await crud_settings.get_settings(cheshire_cat.agent_key)
    assert len(settings) == 0

    # check that the Lizard has been correctly recreated from scratch
    settings = await crud_settings.get_settings(lizard.agent_key)
    assert len(settings) > 0

    # check that the vector database is not empty
    c = await cheshire_cat.vector_memory_handler._client.get_collections()
    assert len(c.collections) == 3


async def test_agent_destroy_success(client, lizard, cheshire_cat):
    creds = {
        "username": "admin",
        "password": get_env("CAT_ADMIN_DEFAULT_PASSWORD"),
    }

    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 200

    received_token = res.json()["access_token"]
    response = await client.post(
        "/utils/agents/destroy",
        headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": cheshire_cat.agent_key},
    )

    assert response.status_code == 200
    assert response.json() == {"deleted_settings": True, "deleted_memories": True, "deleted_plugin_folders": False}

    settings = await crud_settings.get_settings(cheshire_cat.agent_key)
    assert len(settings) == 0

    conversations = await get_async_db().get(crud_conversations.format_key(cheshire_cat.agent_key, "*", "*"))
    assert conversations is None

    plugins = await crud_plugins.get_settings(cheshire_cat.agent_key)
    assert plugins == {}

    users = await crud_users.get_users(cheshire_cat.agent_key)
    assert users == {}

    qdrant_filter = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=cheshire_cat.agent_key))])
    count_response = await cheshire_cat.vector_memory_handler._client.count(
        collection_name=str(VectorMemoryType.DECLARATIVE), count_filter=qdrant_filter
    )
    assert count_response.count == 0


async def test_agent_reset_success(client, lizard, cheshire_cat):
    creds = {
        "username": "admin",
        "password": get_env("CAT_ADMIN_DEFAULT_PASSWORD"),
    }

    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 200

    await checks_on_agent_reset(res, client, cheshire_cat.agent_key, lizard)


async def test_agent_reset_by_new_admin_success(secure_client, client, lizard, cheshire_cat):
    data = await create_new_user(
        secure_client,
        headers={"Authorization": f"Bearer {api_key}"},
        permissions={"CHESHIRE_CAT": ["WRITE"]}
    )
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})

    await checks_on_agent_reset(res, client, cheshire_cat.agent_key, lizard)


async def test_agent_destroy_error_because_of_lack_of_permissions(client, lizard, cheshire_cat):
    # create new admin with wrong permissions
    data = await create_new_user(
        client, headers=await get_client_admin_headers(client), permissions={"EMBEDDER": ["READ"]}
    )

    creds = {"username": data["username"], "password": new_user_password}
    res = await client.post("/auth/token", json=creds)
    received_token = res.json()["access_token"]

    response = await client.post(
        "/utils/agents/destroy",
        headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": cheshire_cat.agent_key}
    )

    assert response.status_code == 403

    await checks_on_agent_destroy(cheshire_cat)


async def test_agent_destroy_error_because_of_not_existing_agent(client, lizard, cheshire_cat):
    creds = {
        "username": "admin",
        "password": get_env("CAT_ADMIN_DEFAULT_PASSWORD"),
    }

    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 200

    received_token = res.json()["access_token"]
    response = await client.post(
        "/utils/agents/destroy", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": "wrong_id"}
    )

    assert response.status_code == 401

    await checks_on_agent_destroy(cheshire_cat)


async def test_agent_create_success(client, lizard):
    creds = {
        "username": "admin",
        "password": get_env("CAT_ADMIN_DEFAULT_PASSWORD"),
    }

    new_agent_id = str(uuid.uuid4())

    res = await client.post("/auth/token", json=creds)
    assert res.status_code == 200

    received_token = res.json()["access_token"]
    response = await client.post(
        "/utils/agents/create",
        headers={"Authorization": f"Bearer {received_token}"},
        json={"agent_id": new_agent_id},
    )

    assert response.status_code == 200
    assert response.json() == {"created": True}

    await checks_on_agent_create(lizard, new_agent_id)


async def test_clone_agent(secure_client, secure_client_headers, lizard, cheshire_cat):
    # First, create a new agent to clone into
    new_agent_id = "test_clone_agent_2"

    # Prepare the request payload
    payload = {"agent_id": new_agent_id}

    # Make the POST request to clone the agent
    response = await secure_client.post(
        "/utils/agents/clone",
        json=payload,
        headers=secure_client_headers,
    )

    # Assert the response status code
    assert response.status_code == 200

    # Assert the response content
    response_data = response.json()
    assert response_data["cloned"] is True

    # check that settings were cloned (settings_id excluded)
    settings = await crud_settings.get_settings(cheshire_cat.agent_key)
    cloned_settings = await crud_settings.get_settings(new_agent_id)
    assert len(cloned_settings) == len(settings)
    for s, cs in zip(settings, cloned_settings):
        s.pop("setting_id", None)
        s.pop("updated_at", None)
        cs.pop("setting_id", None)
        cs.pop("updated_at", None)

        if "active_plugins" in s:
            assert s["active_plugins"].sorted() == cs["active_plugins"].sorted()

    # check that the users were cloned
    users = await crud_users.get_users(cheshire_cat.agent_key)
    cloned_users = await crud_users.get_users(new_agent_id)
    assert len(cloned_users) == len(users)
    assert users == cloned_users
    # check that the plugins were cloned
    plugins = await crud_plugins.get_settings(cheshire_cat.agent_key)
    cloned_plugins = await crud_plugins.get_settings(new_agent_id)
    assert len(cloned_plugins) == len(plugins)
    assert set(plugins) == set(cloned_plugins)

    # check that the vector memory points were cloned
    cloned_ccat = await lizard.get_cheshire_cat(new_agent_id)
    original_points, _ = await cheshire_cat.vector_memory_handler.get_all_tenant_points(
        str(VectorMemoryType.DECLARATIVE), with_vectors=True
    )
    cloned_points, _ = await cloned_ccat.vector_memory_handler.get_all_tenant_points(
        str(VectorMemoryType.DECLARATIVE), with_vectors=True
    )
    assert len(original_points) == len(cloned_points)
    original_payloads = sorted([p.payload for p in original_points], key=lambda x: x["id"])
    cloned_payloads = sorted([p.payload for p in cloned_points], key=lambda x: x["id"])
    assert original_payloads == cloned_payloads
    # assert the vectors are the same
    original_vectors = sorted([p.vector for p in original_points], key=lambda x: x[0])
    cloned_vectors = sorted([p.vector for p in cloned_points], key=lambda x: x[0])
    assert original_vectors == cloned_vectors


async def test_reclone_agent(secure_client, secure_client_headers, lizard, cheshire_cat):
    await test_clone_agent(secure_client, secure_client_headers, lizard, cheshire_cat)

    # First, create a new agent to clone into
    new_agent_id = "test_clone_agent_2"

    # Prepare the request payload
    payload = {"agent_id": new_agent_id}

    # Make the POST request to clone the agent
    response = await secure_client.post(
        "/utils/agents/clone",
        json=payload,
        headers=secure_client_headers,
    )

    # Assert the response status code
    assert response.status_code == 200

    # Assert the response content
    response_data = response.json()
    assert response_data["cloned"] is False


async def test_get_agents(client, cheshire_cat):
    response = await client.get("/utils/agents", headers=await get_client_admin_headers(client))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert cheshire_cat.agent_key in [agent["agent_id"] for agent in data]
    for agent in data:
        assert agent["metadata"] == {}


async def test_update_agent(client, cheshire_cat):
    new_metadata = {"key": "value"}

    response = await client.put(
        "/utils/agents",
        headers=await get_client_admin_headers(client) | {"X-Agent-ID": cheshire_cat.agent_key},
        json={"metadata": new_metadata},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["updated"]

    response = await client.get("/utils/agents", headers=await get_client_admin_headers(client))
    data = response.json()
    for agent in data:
        if agent["agent_id"] == cheshire_cat.agent_key:
            assert agent["metadata"] == new_metadata
        else:
            assert agent["metadata"] == {}
