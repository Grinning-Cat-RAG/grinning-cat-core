"""The rabbit hole of the instance serves every agent at the same time: an ingestion never writes in the memory of
another agent, whatever runs meanwhile."""
import asyncio
from io import BytesIO

from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.rabbit_hole import RabbitHole
from cat.rabbit_hole.phases import PhaseResult


class Agent:
    def __init__(self, agent_key):
        self.agent_key = agent_key
        self.stored = []

        async def execute_hook(name, *args, caller=None):
            return args[0] if args else None

        self.plugin_manager = type("PM", (), {"execute_hook": staticmethod(execute_hook)})()


async def test_concurrent_ingestions_of_different_agents_stay_apart(monkeypatch):
    # regression: the rabbit hole kept the cat of the ingestion in its own attributes: an ingestion of another agent,
    # started meanwhile, replaced it, and the documents went to the memory of the other agent
    async def setup(self, cat):
        self.cat, self.stray = cat, None

    async def keep_source_bytes(self, source, file_bytes, content_type, store_file):
        return None

    async def parsing(self, source, cat):
        await asyncio.sleep(0.01)  # e.g. parsing: the other ingestions run meanwhile
        return PhaseResult(status="done", processed=1)

    async def embedding(self, source, cat):
        await asyncio.sleep(0.01)
        self.cat.stored.append(source.name)
        return PhaseResult(status="done", processed=1)

    async def notify(self, message):
        return None

    for name, function in (("setup", setup), ("_keep_source_bytes", keep_source_bytes),
                           ("run_parsing_chunking", parsing), ("run_embedding", embedding),
                           ("_send_notification_message", notify)):
        monkeypatch.setattr(RabbitHole, name, function)

    shared = RabbitHole()
    agents = [Agent(f"agent-{i}") for i in range(5)]
    for agent in agents:
        await crud.store(crud_settings.format_key(agent.agent_key), [])  # the agents exist

    rows = await asyncio.gather(*(
        shared.ingest_file(agent, BytesIO(b"bytes"), {}, filename=f"{agent.agent_key}.txt", store_file=False)
        for agent in agents
    ))

    assert [agent.stored for agent in agents] == [[f"{agent.agent_key}.txt"] for agent in agents]
    assert [row["status"] for row in rows] == ["completed"] * 5
