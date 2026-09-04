from unittest.mock import patch

from cat.looking_glass.mad_hatter.mad_hatter import MadHatter


async def test_same_payload_does_not_fire_hook(secure_client, secure_client_headers, cheshire_cat):
    payload = {"host": "localhost", "port": 6333, "client_timeout": 10}
    calls = []
    real_execute_hook = MadHatter.execute_hook

    async def recording_execute_hook(self, hook_name, *args, **kwargs):
        if hook_name == "after_vector_database_settings_update":
            calls.append((hook_name, args, kwargs))
        return await real_execute_hook(self, hook_name, *args, **kwargs)

    with patch.object(MadHatter, "execute_hook", new=recording_execute_hook):
        # first PUT establishes the config (may fire hook vs the previous default)
        await secure_client.put(
            "/vector_database/settings/QdrantConfig", json=payload, headers=secure_client_headers
        )
        calls.clear()
        # identical payload → hook must NOT fire
        response = await secure_client.put(
            "/vector_database/settings/QdrantConfig", json=payload, headers=secure_client_headers
        )
        assert response.status_code == 200
        assert calls == []


async def test_changed_payload_fires_hook_once(secure_client, secure_client_headers, cheshire_cat):
    payload1 = {"host": "localhost", "port": 6333, "client_timeout": 10}
    payload2 = {"host": "otherhost", "port": 6333, "client_timeout": 10}
    calls = []
    real_execute_hook = MadHatter.execute_hook

    async def recording_execute_hook(self, hook_name, *args, **kwargs):
        if hook_name == "after_vector_database_settings_update":
            calls.append((hook_name, args, kwargs))
        return await real_execute_hook(self, hook_name, *args, **kwargs)

    with patch.object(MadHatter, "execute_hook", new=recording_execute_hook):
        await secure_client.put(
            "/vector_database/settings/QdrantConfig", json=payload1, headers=secure_client_headers
        )
        calls.clear()
        response = await secure_client.put(
            "/vector_database/settings/QdrantConfig", json=payload2, headers=secure_client_headers
        )
        assert response.status_code == 200
        assert len(calls) == 1
        hook_name, args, kwargs = calls[0]
        assert hook_name == "after_vector_database_settings_update"
        assert args == ("QdrantConfig", payload1, payload2)
        assert kwargs["caller"] == cheshire_cat


async def test_type_switch_does_not_fire_hook(secure_client, secure_client_headers, cheshire_cat):
    payload = {"host": "localhost", "port": 6333, "client_timeout": 10}
    calls = []
    real_execute_hook = MadHatter.execute_hook

    async def recording_execute_hook(self, hook_name, *args, **kwargs):
        if hook_name == "after_vector_database_settings_update":
            calls.append((hook_name, args, kwargs))
        return await real_execute_hook(self, hook_name, *args, **kwargs)

    with patch("cat.services.factory.vector_db.BaseVectorDatabaseHandler.__eq__", return_value=False), patch.object(
        MadHatter, "execute_hook", new=recording_execute_hook
    ):
        response = await secure_client.put(
            "/vector_database/settings/QdrantConfig", json=payload, headers=secure_client_headers
        )
        assert response.status_code == 200
        # type-switch path (transfer) → hook must NOT fire
        assert calls == []


async def test_hook_registered(cheshire_cat):
    assert "after_vector_database_settings_update" in cheshire_cat.plugin_manager.hooks