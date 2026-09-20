"""Tests for the delete-marker protocol (cat/looking_glass/agent_deletion.py).

The protocol: ``CheshireCat.destroy`` writes the persistent marker
``agents:{agent_id}:ingestion:delete`` at the START of the teardown, so
in-flight ingestion workers (EffING's ``ingestion_canceled``) see it and
self-abort while the teardown is still running.
"""
import pytest

from cat.db import crud
from cat.db.database import DEFAULT_AGENTS_KEY
from cat.looking_glass.agent_deletion import (
    _delete_marker_key,
    clear_delete_marker,
    has_delete_marker,
    set_delete_marker,
    wait_for_ingestion_quiesce,
)


@pytest.fixture
async def marker_agent_id() -> str:
    return "agent_test"


async def test_set_delete_marker_writes_key(marker_agent_id):
    key = _delete_marker_key(marker_agent_id)
    try:
        await set_delete_marker(marker_agent_id)
        doc = await crud.read(key)
        # crud.read returns the RedisJSON array wrapper
        assert doc == [{"delete_marker": True}]
    finally:
        await crud.delete(key)


async def test_has_delete_marker_reads_marker(marker_agent_id):
    key = _delete_marker_key(marker_agent_id)
    try:
        assert await has_delete_marker(marker_agent_id) is False
        await set_delete_marker(marker_agent_id)
        assert await has_delete_marker(marker_agent_id) is True
    finally:
        await crud.delete(key)


async def test_clear_delete_marker_removes_only_marker(marker_agent_id):
    key = _delete_marker_key(marker_agent_id)
    other_key = f"{DEFAULT_AGENTS_KEY}:{marker_agent_id}:ingestion:some_source"
    try:
        await set_delete_marker(marker_agent_id)
        await crud.store(other_key, {"status": "done"})

        await clear_delete_marker(marker_agent_id)

        assert await has_delete_marker(marker_agent_id) is False
        assert await crud.read(key) is None
        # the marker removal must never touch other agent keys
        assert await crud.read(other_key) == [{"status": "done"}]
    finally:
        await crud.delete(key)
        await crud.delete(other_key)


async def test_destroy_writes_delete_marker(cheshire_cat):
    key = _delete_marker_key(cheshire_cat._id)
    try:
        await cheshire_cat.destroy()
        assert await crud.read(key) == [{"delete_marker": True}]
    finally:
        await crud.delete(key)


async def test_wait_for_ingestion_quiesce_no_rows(marker_agent_id):
    assert (
        await wait_for_ingestion_quiesce(
            marker_agent_id, quiesce_seconds=1, poll_seconds=0.1
        )
        is True
    )


async def test_wait_for_ingestion_quiesce_times_out_with_active_rows(marker_agent_id):
    key = _delete_marker_key(marker_agent_id)
    status_key = f"{DEFAULT_AGENTS_KEY}:{marker_agent_id}:ingestion:agent:deadbeef"
    try:
        await crud.store(status_key, {"status": "processing"})
        assert (
            await wait_for_ingestion_quiesce(
                marker_agent_id, quiesce_seconds=1, poll_seconds=0.1
            )
            is False
        )
    finally:
        await crud.delete(key)
        await crud.delete(status_key)