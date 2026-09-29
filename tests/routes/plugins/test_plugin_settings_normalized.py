"""The settings of a plugin are stored as its settings model reads them: e.g. "false" is False, not a (true) string."""
import asyncio

from cat.db.cruds import plugins as crud_plugins

from tests.utils import agent_id


async def test_saved_settings_are_normalized(secure_client, secure_client_headers, cheshire_cat):
    # regression: the payload was validated, then stored as it was sent: "false" stayed a string, and the core memory
    # plugin (`if settings["enable_llm_knowledge"]`) read it as true
    response = await secure_client.put(
        "/plugins/settings/memory", headers=secure_client_headers, json={"enable_llm_knowledge": "false"},
    )
    assert response.status_code == 200
    assert response.json()["value"]["enable_llm_knowledge"] is False
    assert (await crud_plugins.get_setting(agent_id, "memory"))["enable_llm_knowledge"] is False

    response = await secure_client.get("/plugins/settings/memory", headers=secure_client_headers)
    assert response.json()["value"]["enable_llm_knowledge"] is False


async def test_settings_stored_as_sent_are_read_normalized(secure_client, secure_client_headers, cheshire_cat):
    # the settings stored by the previous versions of the core are read as the model reads them
    await crud_plugins.set_setting(agent_id, "memory", {"enable_llm_knowledge": "false", "not_declared": "kept"})
    settings = await cheshire_cat.plugin_manager.plugins["memory"].load_settings(agent_id)
    assert settings["enable_llm_knowledge"] is False
    assert settings["not_declared"] == "kept"


async def test_partial_updates_are_never_lost(client):
    # regression: an update read, merged and wrote the settings: two updates at the same time (e.g. on two instances)
    # lost one of them
    await crud_plugins.set_setting(agent_id, "concurrent", {})
    await asyncio.gather(*(crud_plugins.update_setting(agent_id, "concurrent", {f"k{i}": i}) for i in range(20)))
    assert await crud_plugins.get_setting(agent_id, "concurrent") == {f"k{i}": i for i in range(20)}
