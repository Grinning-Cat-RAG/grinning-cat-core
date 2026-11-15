from json import dumps
from fastapi.encoders import jsonable_encoder

from cat.services.service_factory import ServiceFactory

from tests.utils import create_new_user, new_user_password, agent_id


async def test_get_all_chunker_settings(secure_client, secure_client_headers, cheshire_cat):
    sf = ServiceFactory(
        agent_key=cheshire_cat.agent_key,
        hook_manager=cheshire_cat.plugin_manager,
        factory_allowed_handler_name="factory_allowed_chunkers",
        setting_category="chunker",
        schema_name="chunkerName",
    )
    chunkers_schemas = await sf.get_schemas()

    response = await secure_client.get("/chunking/settings", headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 200
    assert isinstance(json["settings"], list)
    assert len(json["settings"]) == len(chunkers_schemas)

    for setting in json["settings"]:
        assert setting["name"] in chunkers_schemas.keys()
        expected_schema = chunkers_schemas[setting["name"]]
        assert dumps(jsonable_encoder(expected_schema)) == dumps(setting["scheme"])

    assert json["selected_configuration"] == "RecursiveTextChunkerSettings"


async def test_get_chunker_settings_non_existent(secure_client, secure_client_headers, cheshire_cat):
    non_existent_chunker_name = "ChunkerNonExistentConfig"
    response = await secure_client.get(f"/chunking/settings/{non_existent_chunker_name}", headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 400
    assert f"{non_existent_chunker_name} not supported" in json["detail"]


async def test_get_chunker_settings(secure_client, secure_client_headers, cheshire_cat):
    chunker_name = "RecursiveTextChunkerSettings"
    response = await secure_client.get(f"/chunking/settings/{chunker_name}", headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 200
    assert json["name"] == chunker_name
    assert json["value"] == {"chunk_overlap": 64, "chunk_size": 256, "encoding_name": "cl100k_base"}
    assert json["scheme"]["chunkerName"] == chunker_name
    assert json["scheme"]["type"] == "object"


async def test_upsert_chunker_settings_success(secure_client, secure_client_headers, cheshire_cat):
    invented_model_name = "this_should_be_a_model"

    # set a different chunker
    new_chunker = "RecursiveTextChunkerSettings"
    payload = {"model_name": invented_model_name}
    response = await secure_client.put(f"/chunking/settings/{new_chunker}", json=payload, headers=secure_client_headers)

    # check immediate response
    json = response.json()
    assert response.status_code == 200
    assert json["name"] == new_chunker
    assert json["value"]["model_name"] == invented_model_name

    # retrieve all Chunker settings to check if it was saved in DB
    response = await secure_client.get("/chunking/settings", headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    assert json["selected_configuration"] == new_chunker
    saved_config = [c for c in json["settings"] if c["name"] == new_chunker]
    assert saved_config[0]["value"]["model_name"] == invented_model_name

    # check also specific Chunker endpoint
    response = await secure_client.get(f"/chunking/settings/{new_chunker}", headers=secure_client_headers)
    assert response.status_code == 200
    json = response.json()
    assert json["name"] == new_chunker
    assert json["value"]["model_name"] == invented_model_name
    assert json["scheme"]["chunkerName"] == new_chunker


async def test_forbidden_access_no_auth(client):
    response = await client.get("/chunking/settings")
    assert response.status_code == 401


async def test_granted_access_on_permissions(secure_client, secure_client_headers, client, cheshire_cat):
    # create user
    data = await create_new_user(secure_client, headers=secure_client_headers, permissions={"CHUNKER": ["READ"]})

    creds = {"username": data["username"], "password": new_user_password}

    res = await client.post("/auth/token", json=creds)
    received_token = res.json()["access_token"]

    response = await client.get("/chunking/settings", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id})
    assert response.status_code == 200


async def test_forbidden_access_no_permission(secure_client, secure_client_headers, client, cheshire_cat):
    # create user
    data = await create_new_user(secure_client, headers=secure_client_headers)
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]

    response = await client.get("/chunking/settings", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id})
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"


async def test_forbidden_access_wrong_permissions(secure_client, secure_client_headers, client, cheshire_cat):
    # create user
    data = await create_new_user(secure_client, headers=secure_client_headers, permissions={"CHUNKER": ["DELETE"]})
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]

    response = await client.get("/chunking/settings", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id})
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"


async def test_upsert_chunker_setting_material_change_schedules_resume(
    secure_client, secure_client_headers, cheshire_cat, monkeypatch
):
    scheduled = []

    def fake_run_background_task(background_tasks, func, *args, **kwargs):
        scheduled.append((func, args, kwargs))

    monkeypatch.setattr("cat.routes.chunker.run_background_task", fake_run_background_task)

    response = await secure_client.put(
        "/chunking/settings/RecursiveTextChunkerSettings",
        json={"chunk_size": 128},
        headers=secure_client_headers,
    )

    assert response.status_code == 200
    assert len(scheduled) == 1
    from cat.routes import chunker
    func, args, kwargs = scheduled[0]
    assert func is chunker.resume_agent
    assert args[1] == cheshire_cat.agent_key


async def test_upsert_chunker_setting_without_material_change_does_not_schedule_resume(
    secure_client, secure_client_headers, cheshire_cat, monkeypatch
):
    scheduled = []

    def fake_run_background_task(background_tasks, func, *args, **kwargs):
        scheduled.append((func, args, kwargs))

    monkeypatch.setattr("cat.routes.chunker.run_background_task", fake_run_background_task)

    response = await secure_client.get("/chunking/settings", headers=secure_client_headers)
    json = response.json()
    chunker_name = json["selected_configuration"]
    current_value = [s for s in json["settings"] if s["name"] == chunker_name][0]["value"]

    response = await secure_client.put(
        f"/chunking/settings/{chunker_name}",
        json=current_value,
        headers=secure_client_headers,
    )

    assert response.status_code == 200
    assert scheduled == []
