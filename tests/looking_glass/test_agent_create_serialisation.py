"""Tests for agent-create serialisation (policy A: the delete wins, create waits).

Covers:

* a create that lands on a pending delete completes the teardown FIRST and then
  builds the agent;
* two concurrent creates of the same id yield exactly one agent and two
  successful returns;
* the ``destroy()``-then-``create()`` reset path still works.
"""

import asyncio
import uuid

from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.looking_glass import agent_deletion, bill_the_lizard
from cat.looking_glass.agent_deletion import has_pending_delete, set_delete_marker


def _agent_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


async def _cleanup_namespace(agent_id: str) -> None:
    """Delete every key under the agent's namespace (and its marker)."""
    await crud.delete(agent_deletion._delete_marker_key(agent_id))
    for key in await crud.scan_keys(f"agents:{agent_deletion._escape_glob(agent_id)}:*"):
        await crud.delete(key)


class _StubVectorMemoryHandler:
    async def initialize(self, name, size):
        return None


class _StubCat:
    def __init__(self, agent_id: str):
        self.agent_key = agent_id
        self.vector_memory_handler = _StubVectorMemoryHandler()

    async def embed_procedures(self):
        return None


class _StubCatFactory:
    """Stand-in for ``CheshireCat`` that records the create calls."""

    def __init__(self, log, store_master_key: bool = True):
        self.log = log
        self.store_master_key = store_master_key

    async def create(self, agent_id):
        self.log.append(f"create:{agent_id}")
        if self.store_master_key:
            await crud.store(f"agents:{agent_id}:agent", {"name": agent_id})
        return _StubCat(agent_id)


async def test_create_completes_pending_delete_first(lizard, monkeypatch):
    agent_id = _agent_id("agent_pending_create")
    log = []

    # Teardown spy: records the call and clears the marker (what the real,
    # idempotent teardown does when it completes).
    async def fake_teardown(target_id):
        log.append(f"teardown:{target_id}")
        await crud.delete(agent_deletion._delete_marker_key(target_id))

    async def fake_embedder():
        return type("E", (), {"name": "stub-embedder"})()

    async def fake_embedding_size(embedder):
        return 3

    async def fake_execute_hook(name, *args, caller=None):
        return None

    monkeypatch.setattr(bill_the_lizard, "complete_agent_teardown", fake_teardown)
    monkeypatch.setattr(bill_the_lizard, "CheshireCat", _StubCatFactory(log))
    monkeypatch.setattr(bill_the_lizard, "embedding_size", fake_embedding_size)
    monkeypatch.setattr(lizard, "embedder", fake_embedder)
    monkeypatch.setattr(lizard.plugin_manager, "execute_hook", fake_execute_hook)

    try:
        await set_delete_marker(agent_id)
        assert await has_pending_delete(agent_id) is True

        cat = await lizard.create_cheshire_cat(agent_id)

        assert cat.agent_key == agent_id
        # The delete won: teardown ran, then the (single) create.
        assert log == [f"teardown:{agent_id}", f"create:{agent_id}"]
        assert await has_pending_delete(agent_id) is False
        assert agent_id in await crud_settings.get_agents_main_keys()
    finally:
        await _cleanup_namespace(agent_id)


async def test_duplicate_concurrent_creates_yield_one_agent(lizard):
    agent_id = _agent_id("agent_concurrent_create")
    try:
        first, second = await asyncio.gather(
            lizard.create_cheshire_cat(agent_id),
            lizard.create_cheshire_cat(agent_id),
        )

        assert first.agent_key == agent_id
        assert second.agent_key == agent_id
        keys = await crud_settings.get_agents_main_keys()
        assert keys.count(agent_id) == 1
    finally:
        cat = await lizard.get_cheshire_cat(agent_id)
        if cat is not None:
            await cat.destroy()
        await _cleanup_namespace(agent_id)


async def test_reset_path_destroy_then_create(lizard):
    agent_id = _agent_id("agent_reset_path")
    try:
        first = await lizard.create_cheshire_cat(agent_id)
        await first.destroy()
        assert agent_id not in await crud_settings.get_agents_main_keys()

        second = await lizard.create_cheshire_cat(agent_id)

        assert second.agent_key == agent_id
        assert agent_id in await crud_settings.get_agents_main_keys()
    finally:
        cat = await lizard.get_cheshire_cat(agent_id)
        if cat is not None:
            await cat.destroy()
        await _cleanup_namespace(agent_id)
