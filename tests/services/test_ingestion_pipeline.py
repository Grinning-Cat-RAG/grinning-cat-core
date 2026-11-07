"""Contract tests for the core ingestion pipeline.

The decisive test pins the delete contract of ``CoreSinglePassPipeline``: it
must reproduce the file-manager route's exact sequence — storage removal,
``before_file_manager_file_delete`` hook, points deletion,
``after_file_manager_file_deleted`` hook — and remove the status row LAST.

Import-safe: the pipeline module must not touch Redis at import time.
"""

import os
import types
from typing import Any
from unittest.mock import AsyncMock

from cat.services.ingestion.models import PhaseResult
from cat.services.ingestion.phase import IngestionPhase
from cat.services.ingestion.pipeline import (
    CoreIngestionPhase,
    CoreSinglePassPipeline,
    IngestionPipeline,
)


def _fake_cat(sequence: list[Any]) -> types.SimpleNamespace:
    class FakeFileManager:
        def remove_file(self, file_path):
            sequence.append(("remove_file", file_path))
            return True

    class FakeVectorMemoryHandler:
        async def delete_tenant_points(self, collection_name, metadata):
            sequence.append(("delete_tenant_points", collection_name, metadata))

    class FakePluginManager:
        async def execute_hook(self, name, *args, caller=None):
            sequence.append(("hook", name, args))

    return types.SimpleNamespace(
        agent_key="agent",
        file_manager=FakeFileManager(),
        vector_memory_handler=FakeVectorMemoryHandler(),
        plugin_manager=FakePluginManager(),
    )


def test_core_single_pass_pipeline_implements_the_contract():
    assert issubclass(CoreSinglePassPipeline, IngestionPipeline)
    assert issubclass(CoreIngestionPhase, IngestionPhase)


def test_phases_returns_exactly_one_core_ingestion_phase():
    pipeline = CoreSinglePassPipeline(store=AsyncMock())

    phases = pipeline.phases()

    assert len(phases) == 1
    assert isinstance(phases[0], CoreIngestionPhase)
    assert phases[0].id == "ingestion"


async def test_delete_fires_the_two_hooks_in_order_and_clears_status_last():
    sequence: list[Any] = []
    cat = _fake_cat(sequence)
    store = AsyncMock()

    async def _clear(agent_id, scope, source):
        sequence.append(("clear", agent_id, scope, source))

    store.clear = _clear

    pipeline = CoreSinglePassPipeline(store=store)
    removed = await pipeline.delete("test.txt", cat)

    print("DELETE HOOK SEQUENCE:")
    for step in sequence:
        print(" ", step)

    assert removed is True
    assert sequence == [
        ("remove_file", os.path.join("agent", "test.txt")),
        ("hook", "before_file_manager_file_delete", ("test.txt", "agent")),
        ("delete_tenant_points", "declarative", {"source": "test.txt"}),
        ("hook", "after_file_manager_file_deleted", ("test.txt", "agent")),
        ("clear", "agent", "agent", "test.txt"),
    ]


async def test_delete_uses_the_chat_scope_for_episodic_sources():
    sequence: list[Any] = []
    cat = _fake_cat(sequence)
    store = AsyncMock()

    async def _clear(agent_id, scope, source):
        sequence.append(("clear", agent_id, scope, source))

    store.clear = _clear

    pipeline = CoreSinglePassPipeline(store=store)
    removed = await pipeline.delete("chat.txt", cat, scope="chat-1")

    print("DELETE HOOK SEQUENCE (chat scope):")
    for step in sequence:
        print(" ", step)

    assert removed is True
    assert sequence == [
        ("remove_file", os.path.join("agent", "chat-1", "chat.txt")),
        ("hook", "before_file_manager_file_delete", ("chat.txt", "chat-1")),
        (
            "delete_tenant_points",
            "episodic",
            {"source": "chat.txt", "chat_id": "chat-1"},
        ),
        ("hook", "after_file_manager_file_deleted", ("chat.txt", "chat-1")),
        ("clear", "agent", "chat-1", "chat.txt"),
    ]


async def test_execute_delegates_to_rabbit_hole_with_the_engine_arguments():
    cat = types.SimpleNamespace()
    cat.rabbit_hole = types.SimpleNamespace(ingest_file=AsyncMock())
    pipeline = CoreSinglePassPipeline(store=AsyncMock())

    await pipeline.execute("test.txt", cat)

    cat.rabbit_hole.ingest_file.assert_awaited_once_with(
        cat=cat,
        file="test.txt",
        filename=None,
        metadata={},
        store_file=True,
        content_type=None,
    )


async def test_phase_process_delegates_to_rabbit_hole_and_returns_done():
    cat = types.SimpleNamespace()
    cat.rabbit_hole = types.SimpleNamespace(ingest_file=AsyncMock())
    phase = CoreIngestionPhase()

    result = await phase.process("test.txt", cat)

    cat.rabbit_hole.ingest_file.assert_awaited_once_with(
        cat=cat,
        file="test.txt",
        filename=None,
        metadata={},
        store_file=True,
        content_type=None,
    )
    assert result == PhaseResult(status="done")


async def test_phase_delete_removes_the_source_points():
    sequence: list[Any] = []
    cat = _fake_cat(sequence)
    phase = CoreIngestionPhase()

    await phase.delete("test.txt", cat)

    assert sequence == [
        ("delete_tenant_points", "declarative", {"source": "test.txt"}),
    ]
