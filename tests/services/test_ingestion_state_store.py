"""Contract tests for the core-owned ingestion state store.

The store must write the SAME document shape the efficient-ingestion plugin
writes (``registry.set_status``) into the same key namespace, so the UI and the
agent-teardown quiesce reader see one contract. The decisive test stores a row
and calls the REAL quiesce reader: a terminal row must not block teardown.

Import-safe tests: the store module itself must not touch Redis at import time.
"""
import hashlib
import json
import time
import uuid

from cat.db import crud
from cat.looking_glass.agent_deletion import wait_for_ingestion_quiesce
from cat.services.ingestion import state as state_module
from cat.services.ingestion.models import IngestionStatus
from cat.services.ingestion.state import (
    CoreIngestionStateStore,
    IngestionStateStore,
    status_key,
)

#: The exact top-level document keys the plugin's ``set_status`` can write.
#: ``phase`` is optional (only present once a phase was provided).
_EXPECTED_KEYS = {
    "source",
    "scope",
    "chat_id",
    "type",
    "status",
    "error",
    "error_at",
    "created_at",
    "updated_at",
    "completed_phases",
    "warnings",
    "warning_count",
    "has_warnings",
}


def _agent_id() -> str:
    return f"test-state-{uuid.uuid4().hex}"


def test_core_store_implements_the_contract():
    assert issubclass(CoreIngestionStateStore, IngestionStateStore)


def test_key_format_matches_the_plugin():
    agent_id = _agent_id()
    digest = hashlib.sha256(b"doc.txt").hexdigest()

    assert status_key(agent_id, "agent", "doc.txt") == (
        f"agents:{agent_id}:ingestion:agent:{digest}"
    )


async def test_write_read_round_trip_dumps_the_exact_document():
    store = CoreIngestionStateStore()
    agent_id = _agent_id()

    doc = await store.set(
        agent_id,
        "agent",
        "doc.txt",
        type_="file",
        status=IngestionStatus.COMPLETED,
    )
    print("EXACT DOCUMENT WRITTEN:")
    print(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print("KEY:", status_key(agent_id, "agent", "doc.txt"))

    assert set(doc) == _EXPECTED_KEYS
    assert doc["source"] == "doc.txt"
    assert doc["scope"] == "agent"
    assert doc["chat_id"] is None
    assert doc["type"] == "file"
    assert doc["status"] == "completed"
    assert doc["error"] is None
    assert doc["error_at"] is None
    assert doc["created_at"] == doc["updated_at"]
    assert doc["completed_phases"] == {
        "ingestion": {"state": "complete", "tag": "", "reason": None}
    }
    assert doc["warnings"] == []
    assert doc["warning_count"] == 0
    assert doc["has_warnings"] is False

    fetched = await store.get(agent_id, "agent", "doc.txt")
    assert fetched == doc

    # The raw stored value is the same document (behind the RedisJSON wrapper).
    raw = await crud.read(status_key(agent_id, "agent", "doc.txt"))
    while isinstance(raw, list):
        raw = raw[0]
    assert raw == doc


async def test_processing_lifecycle_writes_a_processing_phase_entry():
    store = CoreIngestionStateStore()
    agent_id = _agent_id()

    doc = await store.set(
        agent_id,
        "agent",
        "doc.txt",
        type_="file",
        status=IngestionStatus.PROCESSING,
        phase="ingestion",
    )

    assert doc["status"] == "processing"
    assert doc["phase"] == "ingestion"
    assert doc["completed_phases"]["ingestion"]["state"] == "processing"
    assert not doc["has_warnings"]


async def test_update_preserves_created_at_and_bumps_updated_at(monkeypatch):
    store = CoreIngestionStateStore()
    agent_id = _agent_id()

    timestamps = iter([1000.0, 2000.0])
    monkeypatch.setattr(state_module, "generate_timestamp", lambda: next(timestamps))

    first = await store.set(agent_id, "agent", "doc.txt", status=IngestionStatus.PROCESSING)
    second = await store.set(agent_id, "agent", "doc.txt", status=IngestionStatus.COMPLETED)

    assert first["created_at"] == 1000.0
    assert second["created_at"] == 1000.0
    assert first["updated_at"] == 1000.0
    assert second["updated_at"] == 2000.0
    assert second["completed_phases"]["ingestion"]["state"] == "complete"
    assert second["error_at"] is None


async def test_error_sets_error_at_and_error_reason(monkeypatch):
    store = CoreIngestionStateStore()
    agent_id = _agent_id()

    monkeypatch.setattr(state_module, "generate_timestamp", lambda: 3000.0)
    doc = await store.set(
        agent_id,
        "agent",
        "bad.pdf",
        status=IngestionStatus.ERROR,
        error="parse failed",
    )

    assert doc["status"] == "error"
    assert doc["error"] == "parse failed"
    assert doc["error_at"] == 3000.0
    assert doc["completed_phases"]["ingestion"] == {
        "state": "error",
        "tag": "",
        "reason": "parse failed",
    }


async def test_clear_removes_the_row():
    store = CoreIngestionStateStore()
    agent_id = _agent_id()

    await store.set(agent_id, "agent", "doc.txt", status=IngestionStatus.COMPLETED)
    assert await store.get(agent_id, "agent", "doc.txt") is not None

    await store.clear(agent_id, "agent", "doc.txt")

    assert await store.get(agent_id, "agent", "doc.txt") is None
    assert await crud.read(status_key(agent_id, "agent", "doc.txt")) is None


async def test_quiesce_reader_treats_a_completed_row_as_terminal():
    store = CoreIngestionStateStore()
    agent_id = _agent_id()

    await store.set(agent_id, "agent", "done.txt", status=IngestionStatus.COMPLETED)

    started = time.monotonic()
    await wait_for_ingestion_quiesce(agent_id, quiesce_seconds=5, poll_seconds=0.2)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"quiesce waited on a completed row ({elapsed:.3f}s)"


async def test_quiesce_reader_treats_an_error_row_as_terminal():
    store = CoreIngestionStateStore()
    agent_id = _agent_id()

    await store.set(
        agent_id, "agent", "failed.txt", status=IngestionStatus.ERROR, error="boom"
    )

    started = time.monotonic()
    await wait_for_ingestion_quiesce(agent_id, quiesce_seconds=5, poll_seconds=0.2)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"quiesce waited on an error row ({elapsed:.3f}s)"


async def test_quiesce_reader_keeps_waiting_on_a_processing_row():
    """Control: the terminal assertions above are not vacuous."""
    store = CoreIngestionStateStore()
    agent_id = _agent_id()

    await store.set(agent_id, "agent", "busy.txt", status=IngestionStatus.PROCESSING)

    started = time.monotonic()
    await wait_for_ingestion_quiesce(agent_id, quiesce_seconds=1, poll_seconds=0.1)
    elapsed = time.monotonic() - started

    assert elapsed >= 1.0, f"quiesce returned early on a busy row ({elapsed:.3f}s)"
