"""Tests for the ingestion-status core plugin (registry + lifecycle hooks).

The registry tests exercise the Redis-JSON CRUD layer directly (Redis db=1 is
flushed by the autouse ``encapsulate_each_test`` fixture). The lifecycle tests
drive the plugin's hook handlers with fake cat/stray objects so the status
transitions are observable without booting the full app.
"""
import asyncio
import hashlib
import uuid

import pytest

from cat.core_plugins.cat_efficient_ingestion import plugin as ingestion_plugin
from cat.core_plugins.cat_efficient_ingestion.registry import (
    PHASE_DOWNLOADING,
    PHASE_PARSING_CHUNKING,
    IngestionStatus,
    clear_agent,
    clear_chat,
    clear_delete_marker,
    claim_source_for_resume,
    delete_marker_key,
    delete_status,
    get_status,
    has_delete_marker,
    ingestion_canceled,
    list_statuses,
    set_delete_marker,
    set_status,
    status_key,
)
from cat.db import crud
from cat.db.database import get_async_db
#: Unique, module-scoped agent id: every key this module seeds lives under
#: ``agents:effingtest_<uuid>:`` and is destroyed in the autouse cleanup below.
agent_id = f"effingtest_{uuid.uuid4().hex}"


@pytest.fixture(autouse=True)
async def _ensure_agent_master_key():
    """The ``ingestion_canceled`` guard makes ``set_status`` a no-op when the
    agent master key ``agents:<agent_id>:agent`` is absent. These registry
    tests call ``set_status`` directly, so create the master key first (the
    production contract: ``set_status`` is only called for existing agents)."""
    await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])
    yield
    await crud.destroy(f"agents:{agent_id}:*")


# ---------- registry ----------


async def test_status_key_format():
    key = status_key("agent_1", "agent", "doc.pdf")
    digest = hashlib.sha256(b"doc.pdf").hexdigest()
    assert key == f"agents:agent_1:ingestion:agent:{digest}"


async def test_status_key_chat_scope():
    key = status_key("agent_1", "chat_abc", "https://example.com/doc")
    digest = hashlib.sha256(b"https://example.com/doc").hexdigest()
    assert key == f"agents:agent_1:ingestion:chat_abc:{digest}"


async def test_set_status_creates_doc():
    doc = await set_status(
        agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.UPLOADED,
    )
    assert doc["source"] == "doc.pdf"
    assert doc["scope"] == "agent"
    assert doc["chat_id"] is None
    assert doc["type"] == "file"
    assert doc["status"] == IngestionStatus.UPLOADED
    assert doc["error"] is None
    assert doc["error_at"] is None
    assert doc["created_at"] == doc["updated_at"]

    # stored in Redis as JSON, status serialized to its string value
    stored = await get_status(agent_id, "agent", "doc.pdf")
    assert stored is not None
    assert stored["status"] == "uploaded"


async def test_set_status_preserves_created_at():
    first = await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.UPLOADED)
    second = await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.PROCESSING)
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]
    assert second["status"] == IngestionStatus.PROCESSING


async def test_set_status_error_sets_error_at():
    doc = await set_status(
        agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.ERROR, error="boom",
    )
    assert doc["status"] == IngestionStatus.ERROR
    assert doc["error"] == "boom"
    assert doc["error_at"] is not None


async def test_delete_status():
    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.UPLOADED)
    assert await get_status(agent_id, "agent", "doc.pdf") is not None
    await delete_status(agent_id, "agent", "doc.pdf")
    assert await get_status(agent_id, "agent", "doc.pdf") is None


async def test_list_statuses_filters_scope():
    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)
    await set_status(
        agent_id, "chat_abc", "chat.pdf", type_="file", status=IngestionStatus.COMPLETED, chat_id="chat_abc",
    )

    agent_only = await list_statuses(agent_id)
    assert [d["source"] for d in agent_only] == ["doc.pdf"]

    chat_only = await list_statuses(agent_id, chat_id="chat_abc")
    assert [d["source"] for d in chat_only] == ["chat.pdf"]


async def test_clear_agent():
    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)
    await set_status(
        agent_id, "chat_abc", "chat.pdf", type_="file", status=IngestionStatus.COMPLETED, chat_id="chat_abc",
    )
    deleted = await clear_agent(agent_id)
    assert deleted == 2
    db = get_async_db()
    remaining = [k async for k in db.scan_iter(f"agents:{agent_id}:ingestion:*")]
    assert remaining == []


async def test_clear_chat():
    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)
    await set_status(
        agent_id, "chat_abc", "chat.pdf", type_="file", status=IngestionStatus.COMPLETED, chat_id="chat_abc",
    )
    deleted = await clear_chat(agent_id, "chat_abc")
    assert deleted == 1
    assert await get_status(agent_id, "chat_abc", "chat.pdf") is None
    assert await get_status(agent_id, "agent", "doc.pdf") is not None


# ---------- delete marker / ingestion-canceled guards ----------


async def test_delete_marker_helpers_roundtrip():
    key = delete_marker_key(agent_id)
    assert key == f"agents:{agent_id}:ingestion:delete"
    assert await has_delete_marker(agent_id) is False
    assert await ingestion_canceled(agent_id) is False

    await set_delete_marker(agent_id)
    assert await has_delete_marker(agent_id) is True

    await clear_delete_marker(agent_id)
    assert await has_delete_marker(agent_id) is False


async def test_set_status_and_claim_noop_on_delete_marker():
    """A canceled agent (marker present) must never get a status write or claim."""
    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)
    await set_delete_marker(agent_id)

    # set_status returns {} and writes nothing: the pre-existing row is untouched
    doc = await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.UPLOADED)
    assert doc == {}
    stored = await get_status(agent_id, "agent", "doc.pdf")
    assert stored is not None
    assert stored["status"] == IngestionStatus.COMPLETED

    # claim_source_for_resume returns None even though a stale row exists
    claimed = await claim_source_for_resume(
        agent_id, "agent", "doc.pdf", stale_after=0, owner="test-worker",
    )
    assert claimed is None


async def test_ingestion_canceled_true_when_master_key_absent():
    """Master key absent (and marker absent) -> canceled (ghost agent)."""
    await crud.destroy(f"agents:{agent_id}:agent")
    assert await ingestion_canceled(agent_id) is True


async def test_ingestion_canceled_true_when_marker_present_even_with_master():
    """Marker present -> canceled even if the agent master key still exists."""
    await crud.store(f"agents:{agent_id}:agent", [{"name": "agent_settings"}])
    assert await ingestion_canceled(agent_id) is False

    await set_delete_marker(agent_id)
    assert await ingestion_canceled(agent_id) is True


async def test_list_statuses_skips_delete_marker():
    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)
    await set_delete_marker(agent_id)

    docs = await list_statuses(agent_id)
    assert [d["source"] for d in docs] == ["doc.pdf"]
    assert all(d.get("delete_marker") is not True for d in docs)


async def test_clear_agent_preserves_delete_marker():
    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)
    await set_delete_marker(agent_id)

    deleted = await clear_agent(agent_id)
    assert deleted == 1
    assert await get_status(agent_id, "agent", "doc.pdf") is None
    assert await has_delete_marker(agent_id) is True

    # the marker is the ONLY remaining key in the agent's ingestion namespace
    db = get_async_db()
    remaining = [k async for k in db.scan_iter(f"agents:{agent_id}:ingestion:*")]
    assert remaining == [delete_marker_key(agent_id)]


# ---------- lifecycle hooks ----------


class _FakePluginManager:
    """Minimal plugin_manager driving the phase-probe protocol.

    ``pending`` configures what ``ingestion_phase_pending`` reports; when it
    is ``None`` the default no-op semantics apply (nothing pending). ``gate``
    may be an exception that ``before_ingestion_status_completed`` raises.
    """

    def __init__(self, pending=None, gate=None):
        self.pending = pending if pending is not None else []
        self.gate = gate
        self.calls = []

    async def execute_hook(self, hook_name, *args, **kwargs):
        self.calls.append((hook_name, args, kwargs))
        if hook_name == "ingestion_phase_pending":
            return self.pending
        if hook_name == "before_ingestion_status_completed" and self.gate is not None:
            raise self.gate
        return None


class FakeCat:
    """Agent-scoped cat (no ``id`` attribute, like CheshireCat)."""

    agent_key = agent_id

    def __init__(self, plugin_manager=None):
        self.plugin_manager = (
            plugin_manager if plugin_manager is not None else _FakePluginManager()
        )


class FakeStray:
    """Chat-scoped cat (has ``id``, like StrayCat)."""

    agent_key = agent_id

    def __init__(self, chat_id="chat_abc", plugin_manager=None):
        self.id = chat_id
        self.plugin_manager = (
            plugin_manager if plugin_manager is not None else _FakePluginManager()
        )


async def test_file_lifecycle():
    cat = FakeCat()

    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)
    doc = await get_status(agent_id, "agent", "doc.pdf")
    # during processing the phase diary points at parsing_chunking
    assert doc["status"] == "processing"
    assert doc["phase"] == PHASE_PARSING_CHUNKING

    await ingestion_plugin.after_rabbithole_stored_documents.function("doc.pdf", [object()], cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "completed"
    assert doc["type"] == "file"
    assert doc["scope"] == "agent"
    assert doc["chat_id"] is None
    # the phase diary is cleared on completion
    assert "phase" not in doc


async def test_url_lifecycle():
    cat = FakeCat()
    url = "https://example.com/doc"

    await ingestion_plugin.rabbithole_ingestion_start.function(url, {}, True, cat)
    await ingestion_plugin.rabbithole_url_downloading.function(url, url, cat)
    doc = await get_status(agent_id, "agent", url)
    assert doc["status"] == "downloading"
    assert doc["phase"] == PHASE_DOWNLOADING
    await ingestion_plugin.rabbithole_url_download_completed.function(url, url, cat)
    await ingestion_plugin.rabbithole_ingestion_processing.function(url, cat)
    await ingestion_plugin.after_rabbithole_stored_documents.function(url, [object()], cat)

    doc = await get_status(agent_id, "agent", url)
    assert doc is not None
    assert doc["status"] == "completed"
    assert doc["type"] == "url"


async def test_error_lifecycle():
    cat = FakeCat()

    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_error.function("doc.pdf", "boom during parse", cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "boom during parse"
    assert doc["error_at"] is not None


async def test_after_stored_does_not_overwrite_error():
    cat = FakeCat()

    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_error.function("doc.pdf", "store failed", cat)
    # the finally-block hook fires after the error too
    await ingestion_plugin.after_rabbithole_stored_documents.function("doc.pdf", [], cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "store failed"


async def test_processing_records_chunker_name():
    cat = FakeCat()
    cat.chunker = type("C", (), {"name": "RecursiveTextChunker"})()

    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc["phase"] == PHASE_PARSING_CHUNKING
    assert doc["chunker_name"] == "RecursiveTextChunker"


async def test_processing_records_phase_without_diary_or_fingerprint():
    """Processing start must record only status/phase/chunker_name.

    The clock-free ``completed_phases`` diary is written by the dispatcher on
    phase COMPLETION, never at processing start: after
    ``rabbithole_ingestion_processing`` the doc must carry no diary and no
    marker/settings_version/deps fingerprint (stale-state protection — the
    hook must never pre-write a diary the dispatcher later owns).
    """
    cat = FakeCat()
    cat.chunker = type("C", (), {"name": "RecursiveTextChunker"})()

    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "processing"
    assert doc["phase"] == PHASE_PARSING_CHUNKING
    assert doc["chunker_name"] == "RecursiveTextChunker"
    # no diary, no fingerprint keys at processing start
    assert "completed_phases" not in doc
    for fingerprint_key in ("marker", "settings_version", "deps"):
        assert fingerprint_key not in doc


async def test_processing_chunker_resolution_failure_leaves_none():
    """A chunker-resolution failure must leave chunker_name=None, never raise."""
    cat = FakeCat()

    # chunker whose ``name`` access blows up mid-resolution
    def boom():
        raise RuntimeError("no chunker")

    cat.chunker = type("C", (), {"name": property(boom)})()

    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "processing"
    assert doc["phase"] == PHASE_PARSING_CHUNKING
    # failed resolution -> no chunker_name written (reads as None); must not
    # clobber an engine-written chunker_name, so the key is absent, not None
    assert doc.get("chunker_name") is None
    assert "completed_phases" not in doc


async def test_chat_scope_lifecycle():
    stray = FakeStray("chat_abc")

    await ingestion_plugin.rabbithole_ingestion_start.function("chat.pdf", {}, False, stray)
    await ingestion_plugin.rabbithole_ingestion_processing.function("chat.pdf", stray)
    await ingestion_plugin.after_rabbithole_stored_documents.function("chat.pdf", [object()], stray)

    doc = await get_status(agent_id, "chat_abc", "chat.pdf")
    assert doc is not None
    assert doc["status"] == "completed"
    assert doc["scope"] == "chat_abc"
    assert doc["chat_id"] == "chat_abc"


async def test_after_stored_ignores_empty_source():
    cat = FakeCat()

    # the finally-block hook fires with an unresolved (empty) source on early errors
    await ingestion_plugin.after_rabbithole_stored_documents.function("", [], cat)

    assert await list_statuses(agent_id) == []


# ---------- after_rabbithole_stored_documents phase-probe gating ----------


async def test_after_stored_pending_keeps_processing_sets_next_phase():
    """(a) Pending phases -> the row stays PROCESSING, advanced to the next
    pending phase; COMPLETED is never written while a phase is pending."""
    pm = _FakePluginManager(pending=[{"phase": "embedding"}])
    cat = FakeCat(plugin_manager=pm)

    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)
    await ingestion_plugin.after_rabbithole_stored_documents.function("doc.pdf", [object()], cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "processing"
    assert doc["phase"] == "embedding"
    # the probe was consulted, and no COMPLETED write happened
    assert ("ingestion_phase_pending",) == tuple(h for h, *_ in pm.calls)


async def test_after_stored_gate_raise_forces_error():
    """(b) ``before_ingestion_status_completed`` raises -> ERROR, phase cleared."""
    pm = _FakePluginManager(pending=[], gate=RuntimeError("gate failed"))
    cat = FakeCat(plugin_manager=pm)

    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)
    await ingestion_plugin.after_rabbithole_stored_documents.function("doc.pdf", [object()], cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "gate failed"
    assert "phase" not in doc


async def test_after_stored_empty_probe_completes_and_clears_phase():
    """(c) Empty probe (nothing pending) -> COMPLETED, phase diary cleared."""
    pm = _FakePluginManager(pending=[])
    cat = FakeCat(plugin_manager=pm)

    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)
    await ingestion_plugin.after_rabbithole_stored_documents.function("doc.pdf", [object()], cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "completed"
    assert "phase" not in doc
    # both probe and gate were consulted
    hook_names = [h for h, *_ in pm.calls]
    assert "ingestion_phase_pending" in hook_names
    assert "before_ingestion_status_completed" in hook_names


async def test_after_stored_never_resurrects_error_row():
    """(d) An ERROR row is never overwritten by the completion hook — the
    guard short-circuits BEFORE the phase probe even runs."""
    pm = _FakePluginManager(pending=[], gate=None)
    cat = FakeCat(plugin_manager=pm)

    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_error.function("doc.pdf", "store failed", cat)
    await ingestion_plugin.after_rabbithole_stored_documents.function("doc.pdf", [object()], cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "store failed"
    # the ERROR guard returned before consulting the probe
    assert pm.calls == []


# ---------- lifecycle cancellation guards (Todo 11) ----------


async def test_canceled_agent_hooks_write_no_row():
    """A canceled agent (delete marker set) must never get a status write:
    ``rabbithole_ingestion_start`` and ``rabbithole_ingestion_processing``
    early-return, so no row appears at all."""
    cat = FakeCat()
    await set_delete_marker(agent_id)

    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)

    assert await get_status(agent_id, "agent", "doc.pdf") is None
    assert await list_statuses(agent_id) == []


async def test_canceled_agent_after_stored_never_completes():
    """A canceled agent never gets a phase advance nor a terminal COMPLETED:
    a pre-existing PROCESSING row is left untouched by the completion hook."""
    cat = FakeCat()
    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)
    await set_delete_marker(agent_id)

    await ingestion_plugin.after_rabbithole_stored_documents.function("doc.pdf", [object()], cat)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "processing"  # never COMPLETED
    assert doc["phase"] == PHASE_PARSING_CHUNKING  # never advanced/cleared


async def test_heartbeat_fast_aborts_canceled_agent_to_error():
    """The heartbeat is the fast-abort path: on a canceled agent it writes
    ERROR via a DIRECT store (bypassing the guarded ``set_status``, which
    would no-op) and stops, so the row ends terminal instead of dangling
    as PROCESSING."""
    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.PROCESSING)
    await set_delete_marker(agent_id)

    await ingestion_plugin._heartbeat_status(agent_id, "agent", "doc.pdf", 0)

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == IngestionStatus.ERROR
    assert doc["error"] == "Agent deleted — ingestion aborted"


async def test_heartbeat_keeps_live_processing_row_fresh():
    """A live (non-canceled) agent's heartbeat keeps bumping ``updated_at``
    while the row is PROCESSING — the fast-abort path must not change that."""
    cat = FakeCat()
    await ingestion_plugin.rabbithole_ingestion_start.function("doc.pdf", {}, False, cat)
    await ingestion_plugin.rabbithole_ingestion_processing.function("doc.pdf", cat)
    before = (await get_status(agent_id, "agent", "doc.pdf"))["updated_at"]

    task = asyncio.ensure_future(
        ingestion_plugin._heartbeat_status(agent_id, "agent", "doc.pdf", 0.01)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    doc = await get_status(agent_id, "agent", "doc.pdf")
    assert doc is not None
    assert doc["status"] == "processing"
    assert doc["updated_at"] > before