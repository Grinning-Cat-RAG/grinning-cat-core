from json import dumps
from typing import Type
from unittest.mock import patch

from fastapi.encoders import jsonable_encoder

from cat.looking_glass.mad_hatter.mad_hatter import MadHatter
from cat.services.factory.ingestion import (
    BaseIngestionConfiguration,
    BaseIngestionEngine,
    CoreIngestionEngine,
)
from cat.services.service_factory import ServiceFactory

from tests.utils import agent_id, create_new_user, new_user_password


class FakeIngestionEngine(BaseIngestionEngine):
    """Engine contributed by a plugin, used to check the factory wiring."""
    def __init__(self, marker: str = "fake"):
        self.marker = marker

    async def run(self, lizard) -> bool:
        return True

    async def ingest_file(
        self,
        cat,
        file,
        filename: str | None = None,
        metadata: dict | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ) -> None:
        pass


class FakeIngestionConfiguration(BaseIngestionConfiguration):
    marker: str = "fake"

    @classmethod
    def pyclass(cls) -> Type:
        return FakeIngestionEngine


def ingestion_factory(lizard) -> ServiceFactory:
    return ServiceFactory(
        agent_key=lizard.agent_key,
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_ingestions",
        setting_category="ingestion",
        schema_name="ingestionName",
    )


def with_plugin_engine():
    """Patch the ``factory_allowed_ingestions`` hook as a plugin would extend it."""
    original = MadHatter.execute_hook

    async def patched(self, hook_name: str, *args, caller=None):
        allowed = await original(self, hook_name, *args, caller=caller)
        if hook_name == "factory_allowed_ingestions":
            return allowed + [FakeIngestionConfiguration]
        return allowed

    return patch.object(MadHatter, "execute_hook", patched)


async def test_get_all_ingestion_settings(secure_client, secure_client_headers, lizard, cheshire_cat):
    ingestion_schemas = await ingestion_factory(lizard).get_schemas()
    response = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 200
    assert isinstance(json["settings"], list)
    assert len(json["settings"]) == len(ingestion_schemas)

    for setting in json["settings"]:
        assert setting["name"] in ingestion_schemas.keys()
        assert setting["value"] == {}
        expected_schema = ingestion_schemas[setting["name"]]
        assert dumps(jsonable_encoder(expected_schema)) == dumps(setting["scheme"])

    # the core engine is the one bootstrapped for the orchestrator
    assert json["selected_configuration"] == "CoreIngestionConfiguration"


async def test_get_ingestion_settings_non_existent(secure_client, secure_client_headers, cheshire_cat):
    non_existent_ingestion_name = "IngestionNonExistentConfiguration"
    response = await secure_client.get(
        f"/ingestion/settings/{non_existent_ingestion_name}", headers=secure_client_headers
    )
    json = response.json()

    assert response.status_code == 400
    assert f"{non_existent_ingestion_name} not supported" in json["detail"]


async def test_get_ingestion_setting(secure_client, secure_client_headers, cheshire_cat):
    ingestion_name = "CoreIngestionConfiguration"
    response = await secure_client.get(f"/ingestion/settings/{ingestion_name}", headers=secure_client_headers)
    json = response.json()

    assert response.status_code == 200
    assert json["name"] == ingestion_name
    assert json["value"] == {}  # the core engine has indeed an empty config (no options)
    assert json["scheme"]["ingestionName"] == ingestion_name
    assert json["scheme"]["type"] == "object"


async def test_upsert_ingestion_setting_non_existent(secure_client, secure_client_headers, cheshire_cat):
    non_existent_ingestion_name = "IngestionNonExistentConfiguration"
    response = await secure_client.put(
        f"/ingestion/settings/{non_existent_ingestion_name}", json={}, headers=secure_client_headers
    )
    json = response.json()

    assert response.status_code == 400
    assert f"{non_existent_ingestion_name} not supported" in json["detail"]


async def test_upsert_ingestion_setting(secure_client, secure_client_headers, cheshire_cat):
    ingestion_name = "CoreIngestionConfiguration"
    response = await secure_client.put(
        f"/ingestion/settings/{ingestion_name}", json={}, headers=secure_client_headers
    )
    json = response.json()

    # verify success
    assert response.status_code == 200
    assert json["name"] == ingestion_name
    assert json["value"] == {}

    # retrieve all ingestion settings to check if it was saved in DB
    response = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    json = response.json()
    assert response.status_code == 200
    assert json["selected_configuration"] == ingestion_name


async def test_ingestion_engine_is_the_configured_one(secure_client, secure_client_headers, lizard, cheshire_cat):
    """The engine used by the upload routes is the one selected in the factory."""
    engine = await lizard.ingestion()
    assert isinstance(engine, CoreIngestionEngine)


async def test_get_ingestion_settings_with_plugin_engine(secure_client, secure_client_headers, lizard, cheshire_cat):
    """An engine contributed through ``factory_allowed_ingestions`` is listed."""
    with with_plugin_engine():
        response = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
        json = response.json()

        assert response.status_code == 200
        names = [s["name"] for s in json["settings"]]
        assert "CoreIngestionConfiguration" in names
        assert "FakeIngestionConfiguration" in names

        # and it is retrievable on its own, with its own schema
        response = await secure_client.get(
            "/ingestion/settings/FakeIngestionConfiguration", headers=secure_client_headers
        )
        json = response.json()
        assert response.status_code == 200
        assert json["name"] == "FakeIngestionConfiguration"
        assert json["scheme"]["ingestionName"] == "FakeIngestionConfiguration"
        assert json["scheme"]["properties"]["marker"]["default"] == "fake"


async def test_upsert_ingestion_setting_selects_plugin_engine(
    secure_client, secure_client_headers, lizard, cheshire_cat
):
    """Upserting a plugin configuration makes it the engine the core resolves."""
    with with_plugin_engine():
        response = await secure_client.put(
            "/ingestion/settings/FakeIngestionConfiguration",
            json={"marker": "from_plugin"},
            headers=secure_client_headers,
        )
        json = response.json()

        assert response.status_code == 200
        assert json["name"] == "FakeIngestionConfiguration"
        assert json["value"]["marker"] == "from_plugin"

        # the selection is persisted and it is a single-entry category
        response = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
        json = response.json()
        assert response.status_code == 200
        assert json["selected_configuration"] == "FakeIngestionConfiguration"
        saved_config = [c for c in json["settings"] if c["name"] == "FakeIngestionConfiguration"]
        assert saved_config[0]["value"]["marker"] == "from_plugin"
        core_config = [c for c in json["settings"] if c["name"] == "CoreIngestionConfiguration"]
        assert core_config[0]["value"] == {}

        # the core resolves the plugin engine, with its stored settings
        engine = await lizard.ingestion()
        assert isinstance(engine, FakeIngestionEngine)
        assert engine.marker == "from_plugin"


async def test_forbidden_access_no_auth(client, cheshire_cat):
    response = await client.get("/ingestion/settings")
    assert response.status_code == 401


async def test_granted_access_on_permissions(secure_client, secure_client_headers, client, cheshire_cat):
    data = await create_new_user(secure_client, headers=secure_client_headers, permissions={"INGESTION": ["READ"]})
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]

    response = await client.get(
        "/ingestion/settings", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id}
    )
    assert response.status_code == 200


async def test_forbidden_access_no_permission(secure_client, secure_client_headers, client, cheshire_cat):
    data = await create_new_user(secure_client, headers=secure_client_headers)
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]

    response = await client.get(
        "/ingestion/settings", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"


async def test_forbidden_access_wrong_permissions(secure_client, secure_client_headers, client, cheshire_cat):
    data = await create_new_user(secure_client, headers=secure_client_headers, permissions={"INGESTION": ["DELETE"]})
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]

    response = await client.get(
        "/ingestion/settings", headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"


async def test_upsert_ingestion_setting_forbidden_on_read_only(
    secure_client, secure_client_headers, client, cheshire_cat
):
    """WRITE is required to change the engine; READ alone is not enough."""
    data = await create_new_user(secure_client, headers=secure_client_headers, permissions={"INGESTION": ["READ"]})
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]

    response = await client.put(
        "/ingestion/settings/CoreIngestionConfiguration",
        json={},
        headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id},
    )
    assert response.status_code == 403
