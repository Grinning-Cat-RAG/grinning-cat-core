"""Audit fixes for the efficient-ingestion plugin.

Covers the remaining findings:

- glob escaping: an agent id containing ``*`` must never match another
  agent's keys in the plugin-built ``crud.destroy``/scan patterns;
- mid-phase cancellation: a delete marker flipped while a parse/embed loop is
  running stops the loop BEFORE any point is written (not only at status
  boundaries);
- untracked sources: ``reembed_sources`` claims (creates) the status row before
  doing work, so two concurrent passes cannot duplicate an untracked source;
- one reconcile implementation: the endpoint delegates to ``reconcile_agent``.

All keys live under ``agents:effingtest_<uuid>:`` and are destroyed in
``finally``. The host ``tests/conftest.py`` forces ``CAT_REDIS_DB=1`` isolation.
"""
import asyncio
import uuid
from types import SimpleNamespace

import pytest

from cat.core_plugins.cat_efficient_ingestion import endpoints, reembed
from cat.core_plugins.cat_efficient_ingestion import reconcile as reconcile_mod
from cat.core_plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    clear_agent,
    clear_chat,
    get_status,
    set_delete_marker,
    set_status,
)
from cat.db import crud
from cat.services.memory.models import VectorMemoryType


def _agent_id() -> str:
    return f"effingtest_{uuid.uuid4().hex}"


async def _seed_master_key(agent_id: str) -> None:
    await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])


async def _drop(agent_id: str) -> None:
    await crud.destroy(f"agents:{agent_id}:*")


class _Handler:
    """Minimal vector-memory handler that records writes."""

    def __init__(self):
        self.added = []
        self.deleted = []

    async def get_all_tenant_points(self, *args, **kwargs):
        return [], None

    async def add_points_to_tenant(self, *, collection_name, points):
        self.added.append(points)

    async def delete_tenant_points(self, *args, **kwargs):
        self.deleted.append((args, kwargs))


# ---------- 4. glob escaping ----------


async def test_clear_agent_does_not_glob_across_agents():
    """An id with ``*`` must only wipe its own literal namespace."""
    wildcard = f"{_agent_id()}_*"
    victim = f"{_agent_id()}_victim"
    other = _agent_id()
    try:
        for agent in (wildcard, victim, other):
            await _seed_master_key(agent)
            await set_status(agent, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)

        await clear_agent(wildcard)

        assert await get_status(wildcard, "agent", "doc.pdf") is None
        assert await get_status(victim, "agent", "doc.pdf") is not None
        assert await get_status(other, "agent", "doc.pdf") is not None
    finally:
        for agent in (wildcard, victim, other):
            await _drop(agent)


async def test_clear_chat_does_not_glob_across_scopes():
    """Both the agent id and the chat id are escaped in the chat-scope wipe."""
    wildcard_agent = f"{_agent_id()}_*"
    wildcard_chat = f"chat_{uuid.uuid4().hex}_*"
    victim_agent = f"{_agent_id()}_victim"
    victim_chat = f"chat_{uuid.uuid4().hex}_victim"
    try:
        for agent in (wildcard_agent, victim_agent):
            await _seed_master_key(agent)
        await set_status(wildcard_agent, wildcard_chat, "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)
        await set_status(victim_agent, victim_chat, "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)

        await clear_chat(wildcard_agent, wildcard_chat)

        assert await get_status(wildcard_agent, wildcard_chat, "doc.pdf") is None
        assert await get_status(victim_agent, victim_chat, "doc.pdf") is not None
    finally:
        for agent in (wildcard_agent, victim_agent):
            await _drop(agent)


# ---------- 5. cancellation before vector writes ----------


async def test_parse_loop_canceled_mid_run_writes_no_points():
    """A marker flipped while saving images stops before ``add_points_to_tenant``."""
    agent = _agent_id()
    handler = _Handler()
    saved = []
    await _seed_master_key(agent)

    async def save_file(*args, **kwargs):
        saved.append(args[2] if len(args) > 2 else kwargs.get("filename"))
        # flip the cancellation marker AFTER the first loop iteration started
        await set_delete_marker(agent)

    ccat = SimpleNamespace(agent_key=agent, vector_memory_handler=handler, save_file=save_file)
    images = [
        {"image_bytes": b"img-1", "image_mime_type": "image/png"},
        {"image_bytes": b"img-2", "image_mime_type": "image/png"},
    ]
    try:
        with pytest.raises(reembed._IngestionCanceled):
            await reembed._store_empty_vectors(
                ccat, VectorMemoryType.DECLARATIVE, "doc.pdf", None, [], images,
            )
        assert saved, "the first image should have been reached before the cancel"
        assert handler.added == [], "canceled parse must not write points"
    finally:
        await _drop(agent)


async def test_embed_phase_canceled_during_embedding_writes_no_vectors(monkeypatch):
    """A marker flipped during embedding stops before delete/add."""
    agent = _agent_id()
    handler = _Handler()
    await _seed_master_key(agent)

    async def runner(func, *args, **kwargs):
        result = func(*args, **kwargs)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def embed_documents(texts):
        await set_delete_marker(agent)
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(reembed, "run_in_ingestion_executor", runner)
    monkeypatch.setattr(reembed, "split_oversized", lambda docs, embedder: docs)

    point = SimpleNamespace(payload={"page_content": "hello", "metadata": {"source": "doc.pdf"}})
    ccat = SimpleNamespace(agent_key=agent, vector_memory_handler=handler)
    try:
        with pytest.raises(reembed._IngestionCanceled):
            await reembed._embed_phase(
                ccat, VectorMemoryType.DECLARATIVE, "doc.pdf", [point],
                SimpleNamespace(embed_documents=embed_documents, name="e"),
                None, None, None,
            )
        assert handler.deleted == [], "canceled embed must not delete old points"
        assert handler.added == [], "canceled embed must not write new points"
    finally:
        await _drop(agent)


# ---------- 6. untracked source claimed once ----------


async def test_untracked_source_is_claimed_once_by_concurrent_passes(monkeypatch):
    """Two concurrent ``reembed_sources`` on a no-doc source: only one proceeds."""
    agent = _agent_id()
    await _seed_master_key(agent)
    handler = _Handler()
    ran = []

    async def fake_parse(*args, **kwargs):
        ran.append(kwargs.get("source_name") or args[3])
        return "doc.pdf", []

    async def fake_record(*args, **kwargs):
        return {}

    async def fake_probe(*args, **kwargs):
        return []

    async def fake_complete(*args, **kwargs):
        return None

    monkeypatch.setattr(reembed, "_run_parsing_phase", fake_parse)
    monkeypatch.setattr(reembed, "_record_phase", fake_record)
    monkeypatch.setattr(reembed, "_probe_pending", fake_probe)
    monkeypatch.setattr(reembed, "_complete_source", fake_complete)

    async def embedder():
        return SimpleNamespace(name="test-embedder")

    ccat = SimpleNamespace(
        _id=agent,
        agent_key=agent,
        vector_memory_handler=handler,
        embedder=embedder,
        chunker=SimpleNamespace(name="test-chunker"),
        rabbit_hole=SimpleNamespace(),
    )
    source = SimpleNamespace(name="doc.pdf", path="doc.pdf", content=None, metadata={})
    try:
        assert await get_status(agent, "agent", "doc.pdf") is None
        await asyncio.gather(
            reembed.reembed_sources(ccat, VectorMemoryType.DECLARATIVE, [source]),
            reembed.reembed_sources(ccat, VectorMemoryType.DECLARATIVE, [source]),
        )
        assert len(ran) == 1, "both concurrent passes processed the untracked source"
    finally:
        await _drop(agent)


# ---------- 1. one reconcile implementation ----------


def test_endpoint_uses_the_shared_reconcile_implementation():
    """``endpoints.py`` must not keep a second ``_reconcile`` copy."""
    assert not hasattr(endpoints, "_reconcile")
    assert endpoints.reconcile_agent is reconcile_mod.reconcile_agent
