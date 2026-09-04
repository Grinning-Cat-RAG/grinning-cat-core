"""Tests for the ingestion-status core plugin (registry + lifecycle hooks).

The registry tests exercise the Redis-JSON CRUD layer directly (Redis db=1 is
flushed by the autouse ``encapsulate_each_test`` fixture). The lifecycle tests
drive the plugin's hook handlers with fake cat/stray objects so the status
transitions are observable without booting the full app.
"""
import hashlib

from cat.core_plugins.ingestion_status import plugin as ingestion_plugin
from cat.core_plugins.ingestion_status.registry import (
    PHASE_DOWNLOADING,
    PHASE_PARSING_CHUNKING,
    IngestionStatus,
    clear_agent,
    clear_chat,
    delete_status,
    get_status,
    list_statuses,
    set_status,
    status_key,
)
from cat.db.database import get_async_db
from tests.utils import agent_id

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


# ---------- lifecycle hooks ----------


class FakeCat:
    """Agent-scoped cat (no ``id`` attribute, like CheshireCat)."""
    agent_key = agent_id


class FakeStray:
    """Chat-scoped cat (has ``id``, like StrayCat)."""
    agent_key = agent_id

    def __init__(self, chat_id="chat_abc"):
        self.id = chat_id


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