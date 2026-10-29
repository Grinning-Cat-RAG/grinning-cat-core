"""Tests for the complete agent-teardown protocol.

``cat/looking_glass/agent_deletion.py`` owns the full delete-marker lifecycle:
marker -> bounded ingestion quiesce -> namespace wipe (marker preserved) ->
memory/file cascade over the sources captured BEFORE the wipe -> marker clear
LAST. These tests cover the marker schema, the wipe, the end-to-end run on a
real cat, idempotency, the quiesce semantics, the resumable orphan path, and
the per-source cascade ordering.
"""
import time

import pytest

from cat.db import crud
from cat.db.database import DEFAULT_AGENTS_KEY
from cat.looking_glass import agent_deletion
from cat.looking_glass.agent_deletion import (
    _delete_marker_key,
    _escape_glob,
    complete_agent_teardown,
    complete_pending_deletes,
    has_delete_marker,
    has_pending_delete,
    set_delete_marker,
    wait_for_ingestion_quiesce,
    wipe_agent_namespace_excluding_marker,
)


def _unwrap(doc):
    """Unwrap the RedisJSON one-element list wrapper."""
    while isinstance(doc, list) and len(doc) == 1:
        doc = doc[0]
    return doc


class _FakeFileManager:
    def __init__(self, sources, log):
        self._sources = list(sources)
        self.log = log
        self.removed = []

    def list_files(self, remote_root_dir):
        self.log.append("list_files")
        return [_Source(name) for name in self._sources]

    def remove_folder(self, remote_root_dir):
        self.removed.append(remote_root_dir)
        self.log.append("remove_folder")


class _Source:
    def __init__(self, name):
        self.name = name


class _FakeVectorMemoryHandler:
    def __init__(self, log):
        self.log = log
        self.calls = []

    async def delete_tenant_points(self, collection_name, metadata=None):
        source = metadata.get("source") if metadata else None
        self.calls.append((collection_name, metadata))
        self.log.append(f"delete_tenant_points:{source}")


class _FakePluginManager:
    def __init__(self, log):
        self.log = log
        self.hooks = []

    async def execute_hook(self, name, *args, caller=None):
        filename = args[0] if args else None
        self.hooks.append((name, filename))
        self.log.append(f"hook:{name}:{filename}")


class _FakeCat:
    """Minimal stand-in for ``CheshireCat`` for unit-level teardown tests."""

    def __init__(self, agent_id, sources=None):
        self._id = agent_id
        self.agent_key = agent_id
        self.log = []
        self.file_manager = _FakeFileManager(sources or [], self.log)
        self.vector_memory_handler = _FakeVectorMemoryHandler(self.log)
        self.plugin_manager = _FakePluginManager(self.log)
        self.shutdown_calls = 0

    async def destroy_memory(self):
        self.log.append("destroy_memory")

    async def shutdown(self):
        self.shutdown_calls += 1
        self.log.append("shutdown")


@pytest.fixture
def marker_agent_id() -> str:
    return "agent_test"


@pytest.fixture
def unique_agent_id() -> str:
    # No glob metacharacters, so namespace patterns are exact.
    return "agent_test_7f3c9a"


# 1. marker schema -----------------------------------------------------------
async def test_set_delete_marker_writes_schema(marker_agent_id):
    key = _delete_marker_key(marker_agent_id)
    try:
        await set_delete_marker(marker_agent_id)

        doc = _unwrap(await crud.read(key))
        assert isinstance(doc, dict)
        assert doc["delete_marker"] is True
        assert isinstance(doc["started_at"], float)
        assert doc["started_at"] > 0
    finally:
        await crud.delete(key)


async def test_marker_helpers_are_presence_only(marker_agent_id):
    key = _delete_marker_key(marker_agent_id)
    try:
        assert await has_delete_marker(marker_agent_id) is False
        assert await has_pending_delete(marker_agent_id) is False

        # The plugin's schema (no started_at) must also count as present.
        await crud.store(key, {"delete_marker": True})

        assert await has_delete_marker(marker_agent_id) is True
        assert await has_pending_delete(marker_agent_id) is True
    finally:
        await crud.delete(key)


# 2. namespace wipe ----------------------------------------------------------
async def test_wipe_removes_namespace_but_preserves_marker(unique_agent_id):
    marker_key = _delete_marker_key(unique_agent_id)
    master_key = f"{DEFAULT_AGENTS_KEY}:{unique_agent_id}:agent"
    peripheral = [
        f"{DEFAULT_AGENTS_KEY}:{unique_agent_id}:settings:foo",
        f"{DEFAULT_AGENTS_KEY}:{unique_agent_id}:conversations:bar",
    ]
    try:
        await set_delete_marker(unique_agent_id)
        await crud.store(master_key, {"name": unique_agent_id})
        for key in peripheral:
            await crud.store(key, {"value": key})

        deleted = await wipe_agent_namespace_excluding_marker(unique_agent_id)

        assert deleted == 3
        assert await crud.read(master_key) is None
        for key in peripheral:
            assert await crud.read(key) is None
        assert await has_delete_marker(unique_agent_id) is True
        assert await crud.scan_keys(
            f"{DEFAULT_AGENTS_KEY}:{_escape_glob(unique_agent_id)}:*"
        ) == [marker_key]
    finally:
        await crud.delete(marker_key)


async def test_wipe_escaping_does_not_touch_glob_sibling():
    victim = "agent_glob_victim"
    attacker = "agent_glob_*"
    attacker_marker = _delete_marker_key(attacker)
    victim_key = f"{DEFAULT_AGENTS_KEY}:{victim}:agent"
    try:
        await crud.store(victim_key, {"name": victim})
        await set_delete_marker(attacker)

        await wipe_agent_namespace_excluding_marker(attacker)

        # The escaped pattern must only match the literal ``agent_glob_*``
        # namespace, never the sibling whose id happens to start the same way.
        assert await crud.read(victim_key) is not None
        assert await has_delete_marker(attacker) is True
    finally:
        await crud.delete(attacker_marker)
        await crud.delete(victim_key)


# 3. end-to-end on a real cat ------------------------------------------------
async def test_complete_teardown_end_to_end(cheshire_cat):
    agent_id = cheshire_cat._id
    try:
        await set_delete_marker(agent_id)

        await complete_agent_teardown(agent_id, ccat=cheshire_cat)

        assert await has_delete_marker(agent_id) is False
        assert await crud.scan_keys(f"{DEFAULT_AGENTS_KEY}:{_escape_glob(agent_id)}:*") == []
    finally:
        await crud.delete(_delete_marker_key(agent_id))


# 4. idempotency -------------------------------------------------------------
async def test_complete_teardown_is_idempotent(unique_agent_id):
    fake = _FakeCat(unique_agent_id, sources=["doc.txt"])
    try:
        await set_delete_marker(unique_agent_id)

        await complete_agent_teardown(unique_agent_id, ccat=fake)
        log_after_first = list(fake.log)
        assert await has_delete_marker(unique_agent_id) is False

        # Second run: marker gone -> early return, no work, no shutdown.
        await complete_agent_teardown(unique_agent_id, ccat=fake)

        assert fake.log == log_after_first
        assert fake.shutdown_calls == 0
        assert await has_delete_marker(unique_agent_id) is False
    finally:
        await crud.delete(_delete_marker_key(unique_agent_id))


# 5. quiesce semantics -------------------------------------------------------
async def test_quiesce_ignores_marker_and_completed_rows(marker_agent_id):
    marker_key = _delete_marker_key(marker_agent_id)
    completed_key = f"{DEFAULT_AGENTS_KEY}:{marker_agent_id}:ingestion:agent:done"
    try:
        await set_delete_marker(marker_agent_id)
        await crud.store(completed_key, {"status": "completed"})

        started = time.monotonic()
        result = await wait_for_ingestion_quiesce(
            marker_agent_id, quiesce_seconds=5, poll_seconds=0.1
        )
        elapsed = time.monotonic() - started

        assert result is None
        assert elapsed < 1.5
    finally:
        await crud.delete(marker_key)
        await crud.delete(completed_key)


async def test_quiesce_waits_on_processing_row(marker_agent_id):
    marker_key = _delete_marker_key(marker_agent_id)
    processing_key = f"{DEFAULT_AGENTS_KEY}:{marker_agent_id}:ingestion:agent:busy"
    try:
        await set_delete_marker(marker_agent_id)
        await crud.store(processing_key, {"status": "processing"})

        started = time.monotonic()
        result = await wait_for_ingestion_quiesce(
            marker_agent_id, quiesce_seconds=1, poll_seconds=0.1
        )
        elapsed = time.monotonic() - started

        assert result is None
        assert elapsed >= 0.9
        assert elapsed < 3.0
    finally:
        await crud.delete(marker_key)
        await crud.delete(processing_key)


# 6. resumable orphan marker (boot-light path) -------------------------------
async def test_complete_pending_deletes_recovers_orphan_marker(lizard):
    agent_id = "agent_orphan_5b1e"
    marker_key = _delete_marker_key(agent_id)
    try:
        await set_delete_marker(agent_id)

        await complete_pending_deletes()

        assert await has_delete_marker(agent_id) is False
        assert await crud.scan_keys(f"{DEFAULT_AGENTS_KEY}:{_escape_glob(agent_id)}:*") == []
    finally:
        await crud.delete(marker_key)


# 7. per-source cascade ordering ---------------------------------------------
async def test_teardown_cascades_each_source_after_wipe(unique_agent_id, monkeypatch):
    sources = ["doc_one.txt", "doc_two.pdf"]
    fake = _FakeCat(unique_agent_id, sources=sources)

    async def fake_wipe(agent_id):
        fake.log.append("wipe")
        return 0

    monkeypatch.setattr(agent_deletion, "wipe_agent_namespace_excluding_marker", fake_wipe)

    try:
        await set_delete_marker(unique_agent_id)

        await complete_agent_teardown(unique_agent_id, ccat=fake)

        log = fake.log
        wipe_index = log.index("wipe")
        assert log.index("list_files") < wipe_index
        assert log.index("destroy_memory") > wipe_index

        for source in sources:
            before = log.index(f"hook:before_file_manager_file_delete:{source}")
            delete = log.index(f"delete_tenant_points:{source}")
            after = log.index(f"hook:after_file_manager_file_deleted:{source}")

            # Per source, in order, and only after the namespace wipe.
            assert wipe_index < before < delete < after

        assert "hook:after_cheshire_cat_destroy:" + unique_agent_id in log
        assert fake.file_manager.removed == [unique_agent_id]
        assert len(fake.vector_memory_handler.calls) == len(sources)
        assert await has_delete_marker(unique_agent_id) is False
    finally:
        await crud.delete(_delete_marker_key(unique_agent_id))


# 8. CheshireCat.destroy() entry point ---------------------------------------
async def test_destroy_clears_marker_and_namespace(cheshire_cat, monkeypatch):
    """``CheshireCat.destroy()`` runs the whole protocol, not just the marker."""
    from cat.looking_glass.cheshire_cat import CheshireCat

    agent_id = cheshire_cat._id
    file_manager = cheshire_cat.file_manager
    plugin_manager = cheshire_cat.plugin_manager

    assert isinstance(cheshire_cat, CheshireCat)

    hooks = []
    original_execute_hook = plugin_manager.execute_hook

    async def spy_execute_hook(name, *args, caller=None):
        hooks.append(name)
        return await original_execute_hook(name, *args, caller=caller)

    memory_destroyed = []
    original_destroy_memory = cheshire_cat.destroy_memory

    async def spy_destroy_memory():
        memory_destroyed.append(True)
        return await original_destroy_memory()

    removed_folders = []
    original_remove_folder = file_manager.remove_folder

    def spy_remove_folder(remote_root_dir):
        removed_folders.append(remote_root_dir)
        return original_remove_folder(remote_root_dir)

    monkeypatch.setattr(plugin_manager, "execute_hook", spy_execute_hook)
    monkeypatch.setattr(cheshire_cat, "destroy_memory", spy_destroy_memory)
    monkeypatch.setattr(file_manager, "remove_folder", spy_remove_folder)

    try:
        await cheshire_cat.destroy()

        assert memory_destroyed == [True]
        assert "after_cheshire_cat_destroy" in hooks
        assert removed_folders == [agent_id]
        assert await has_delete_marker(agent_id) is False
        assert await crud.scan_keys(f"{DEFAULT_AGENTS_KEY}:{_escape_glob(agent_id)}:*") == []
    finally:
        await crud.delete(_delete_marker_key(agent_id))


async def test_destroy_is_idempotent():
    """Calling the entry point twice does not raise and leaves nothing behind."""
    from cat.looking_glass.cheshire_cat import CheshireCat

    agent_id = "agent_destroy_twice_4a7d"
    fake = _FakeCat(agent_id, sources=["doc.txt"])
    try:
        await CheshireCat.destroy(fake)
        assert await has_delete_marker(agent_id) is False

        # Second call: it writes the marker again, runs the (now empty)
        # teardown and clears it again — no error.
        await CheshireCat.destroy(fake)

        assert fake.shutdown_calls == 2
        assert await has_delete_marker(agent_id) is False
        assert await crud.scan_keys(f"{DEFAULT_AGENTS_KEY}:{_escape_glob(agent_id)}:*") == []
    finally:
        await crud.delete(_delete_marker_key(agent_id))


# 9. rollback resiliency / recovery ------------------------------------------
async def test_teardown_keeps_marker_when_wipe_raises(unique_agent_id, monkeypatch):
    """A failing wipe must leave the marker behind, so boot recovery can retry."""
    fake = _FakeCat(unique_agent_id, sources=[])

    async def boom(*args, **kwargs):
        raise RuntimeError("wipe failed")

    monkeypatch.setattr(agent_deletion.crud, "destroy", boom)

    try:
        await set_delete_marker(unique_agent_id)

        with pytest.raises(RuntimeError):
            await complete_agent_teardown(unique_agent_id, ccat=fake)

        assert await has_delete_marker(unique_agent_id) is True
    finally:
        await crud.delete(_delete_marker_key(unique_agent_id))


async def test_marker_only_recovery_clears_without_booting_cat(unique_agent_id, monkeypatch):
    """A crash after the wipe leaves only the marker: clear it, do not boot."""
    boot_calls = []

    async def fail_boot(agent_id):
        boot_calls.append(agent_id)
        raise AssertionError("light cat must not be booted for a marker-only namespace")

    monkeypatch.setattr(agent_deletion, "_boot_light_cat", fail_boot)

    try:
        await set_delete_marker(unique_agent_id)
        assert await has_delete_marker(unique_agent_id) is True

        await complete_agent_teardown(unique_agent_id)

        assert boot_calls == []
        assert await has_delete_marker(unique_agent_id) is False
        assert await crud.scan_keys(
            f"{DEFAULT_AGENTS_KEY}:{_escape_glob(unique_agent_id)}:*"
        ) == []
    finally:
        await crud.delete(_delete_marker_key(unique_agent_id))


# 10. lock-budget ordering ---------------------------------------------------
def test_create_lock_budget_outlives_teardown_budget():
    """The create lock wraps a nested teardown, so it must be strictly larger."""
    from cat.looking_glass import bill_the_lizard
    from cat.looking_glass.agent_deletion import _teardown_lock_budget

    for quiesce in (0, 180, 600):
        assert bill_the_lizard._create_lock_budget(quiesce) > _teardown_lock_budget(quiesce)
