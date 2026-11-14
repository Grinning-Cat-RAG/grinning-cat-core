"""A namespace without the master agent key is not an agent (a ghost), and factory reset tolerates it."""
import types

from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.exceptions import UnknownAgentException


async def test_a_namespace_without_the_master_key_is_not_listed_as_an_agent(lizard, cheshire_cat):
    ghost = "ghost-agent"
    await crud.store(f"agents:{ghost}:users:someone", {"id": "someone"})
    await crud.store(f"agents:{ghost}:ingestion:agent:x", {"status": "uploaded"})

    agents = await crud_settings.get_agents_main_keys()

    assert ghost not in agents
    assert cheshire_cat.agent_key in agents


async def test_factory_reset_skips_a_ghost_agent(monkeypatch, cheshire_cat, tmp_path):
    destroyed = []
    real = "real-agent"

    class FakeCat:
        async def destroy(self):
            destroyed.append("real-agent")

    class FakeLizard:
        async def get_cheshire_cat(self, agent_id):
            if agent_id == "ghost-agent":
                raise UnknownAgentException(agent_id)
            return FakeCat()

    class FakeDB:
        async def flushdb(self):
            return True

    async def noop(*args, **kwargs):
        return None

    async def agents_main_keys(pattern=None):
        return ["ghost-agent", real]

    monkeypatch.setattr("cat.routes.utilities.crud_settings.get_agents_main_keys", agents_main_keys)
    monkeypatch.setattr("cat.routes.utilities.shutdown_app", noop)
    monkeypatch.setattr("cat.routes.utilities.startup_app", noop)
    monkeypatch.setattr("cat.routes.utilities.get_async_db", lambda: FakeDB())
    monkeypatch.setattr("cat.routes.utilities.utils.get_plugins_path", lambda: str(tmp_path))

    from cat.routes.utilities import factory_reset

    info = types.SimpleNamespace(lizard=FakeLizard())
    request = types.SimpleNamespace(app=types.SimpleNamespace())

    response = await factory_reset(request, info)

    assert destroyed == [real]
    assert response.deleted_memories is True
