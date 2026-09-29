"""The rabbit hole of the instance serves every agent at the same time: an ingestion never writes in the memory of
another agent, whatever runs meanwhile."""
import asyncio

from langchain_core.documents import Document

from cat.rabbit_hole import RabbitHole


class Agent:
    def __init__(self, agent_key):
        self.agent_key = agent_key
        self.stored = []


async def test_concurrent_ingestions_of_different_agents_stay_apart(monkeypatch):
    # regression: the rabbit hole kept the cat of the ingestion in its own attributes: an ingestion of another agent,
    # started meanwhile, replaced it, and the documents went to the memory of the other agent
    async def setup(self, cat):
        self.cat, self.stray = cat, None

    async def resolve_source_bytes(self, file, filename, content_type=None):
        return filename, b"bytes", "text/plain", False

    async def parse_to_docs(self, source, file_bytes, content_type):
        await asyncio.sleep(0.01)  # e.g. parsing: the other ingestions run meanwhile
        return [Document(page_content=source)], []

    async def store_documents(self, docs, source, **kwargs):
        await asyncio.sleep(0.01)
        self.cat.stored.append(source)
        return []

    async def notify(self, message):
        return None

    async def execute_hook(*args, **kwargs):
        return args[1] if len(args) > 1 else None

    for name, function in (("setup", setup), ("_resolve_source_bytes", resolve_source_bytes),
                           ("_parse_to_docs", parse_to_docs), ("store_documents", store_documents),
                           ("_send_notification_message", notify)):
        monkeypatch.setattr(RabbitHole, name, function)

    shared = RabbitHole()
    agents = [Agent(f"agent-{i}") for i in range(5)]
    for agent in agents:
        agent.plugin_manager = type("PM", (), {"execute_hook": staticmethod(execute_hook)})()

    await asyncio.gather(*(
        shared.ingest_file(agent, b"bytes", {}, filename=f"{agent.agent_key}.txt", store_file=False) for agent in agents
    ))
    assert [agent.stored for agent in agents] == [[f"{agent.agent_key}.txt"] for agent in agents]
