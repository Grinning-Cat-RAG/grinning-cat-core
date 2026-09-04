"""Tests for the ``/ingestion/status`` endpoint (read-time reconcile).

Uses the repo's route-test conventions: httpx AsyncClient + ASGITransport via
the ``secure_client`` fixture, agent header from ``tests.utils.agent_id``,
Redis db=1 (flushed by the autouse ``encapsulate_each_test`` fixture).

The endpoint builds a *fresh* ``CheshireCat`` per request (via
``lizard.get_cheshire_cat``), so the file manager is monkeypatched at the
``ServiceProvider`` level and files are written straight to the mocked storage
root (``tests/data/storage``) rather than through the fixture's cat.
"""
import os
import urllib.parse

from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
from cat.core_plugins.ingestion_status.registry import (
    IngestionStatus,
    get_status,
    set_status,
)
from cat.db import crud
from cat.db.database import DEFAULT_AGENTS_KEY, DEFAULT_CONVERSATIONS_KEY
from cat.services.service_provider import ServiceProvider
from tests.utils import agent_id, chat_id, create_new_user, new_user_password

STORAGE_ROOT = "tests/data/storage"


def _write_storage_file(rel_path: str, content: str = "hello") -> None:
    """Write a file directly into the mocked file-manager storage root."""
    full = os.path.join(STORAGE_ROOT, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)


def _use_local_file_manager(monkeypatch) -> None:
    """Make freshly-created CheshireCats use the on-disk LocalFileManager."""

    async def fake_get_file_manager(self, agent_key, plugin_manager):
        return LocalFileManager()

    monkeypatch.setattr(ServiceProvider._class, "get_file_manager", fake_get_file_manager)


async def test_ingestion_status_returns_seeded(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    _use_local_file_manager(monkeypatch)
    _write_storage_file(os.path.join(agent_id, "doc.pdf"))

    await set_status(agent_id, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    json = response.json()
    assert isinstance(json, list)
    assert len(json) == 1
    entry = json[0]
    assert entry["source"] == "doc.pdf"
    assert entry["scope"] == "agent"
    assert entry["chat_id"] is None
    assert entry["type"] == "file"
    assert entry["status"] == "completed"
    assert "created_at" in entry
    assert "updated_at" in entry


async def test_chat_scope_returns_only_chat_entries(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    _use_local_file_manager(monkeypatch)
    _write_storage_file(os.path.join(agent_id, "agent_doc.pdf"))
    _write_storage_file(os.path.join(agent_id, chat_id, "chat_doc.pdf"))

    # create the conversation so the chat-scope reconcile keeps the entry
    await crud.store(
        f"{DEFAULT_AGENTS_KEY}:{agent_id}:{DEFAULT_CONVERSATIONS_KEY}:user1:{chat_id}",
        {"history": []},
    )

    await set_status(agent_id, "agent", "agent_doc.pdf", type_="file", status=IngestionStatus.COMPLETED)
    await set_status(
        agent_id, chat_id, "chat_doc.pdf", type_="file", status=IngestionStatus.COMPLETED, chat_id=chat_id,
    )

    # agent scope returns only the agent entry
    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert [e["source"] for e in response.json()] == ["agent_doc.pdf"]

    # chat scope returns only that chat's entry
    response = await secure_client.get(f"/ingestion/status?chat_id={chat_id}", headers=secure_client_headers)
    assert response.status_code == 200
    assert [e["source"] for e in response.json()] == ["chat_doc.pdf"]


async def test_file_status_purged_when_file_missing(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    _use_local_file_manager(monkeypatch)
    # seed a status for a file that does not exist in the file manager
    await set_status(agent_id, "agent", "ghost.pdf", type_="file", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert response.json() == []

    # the Redis key is gone
    assert await get_status(agent_id, "agent", "ghost.pdf") is None


async def test_url_status_purged_when_no_web_points(secure_client, secure_client_headers, cheshire_cat):
    # seed a URL status with no corresponding web point in the vector store
    await set_status(agent_id, "agent", "https://example.com/doc", type_="url", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert response.json() == []

    # the Redis key is gone
    assert await get_status(agent_id, "agent", "https://example.com/doc") is None


async def test_url_status_survives_when_web_point_exists(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    """B1: ``_canonical_urls`` passes ``str(collection)`` to the vector handler.

    A fake handler that raises on a non-str collection proves the endpoint no
    longer crashes and URL statuses with live web points survive the reconcile.
    """
    class FakeVectorMemoryHandler:
        def __init__(self):
            self.calls = []

        async def get_all_tenant_points_from_web(self, collection):
            self.calls.append(collection)
            if not isinstance(collection, str):
                raise TypeError("bad argument type for built-in operation")
            point = type("P", (), {"payload": {"metadata": {"source": "https://example.com/doc"}}})()
            return [point], None

    handler = FakeVectorMemoryHandler()

    async def fake_get_vector_memory_handler(self, *args, **kwargs):
        return handler

    monkeypatch.setattr(ServiceProvider._class, "get_vector_memory_handler", fake_get_vector_memory_handler)

    await set_status(agent_id, "agent", "https://example.com/doc", type_="url", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    json = response.json()
    assert len(json) == 1
    assert json[0]["source"] == "https://example.com/doc"
    assert json[0]["status"] == "completed"
    # the handler was called with a str, never the VectorMemoryType enum
    assert handler.calls, "vector handler was never called"
    assert all(isinstance(c, str) for c in handler.calls)


async def test_error_status_not_purged_when_source_absent(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    """M3: an ``error`` entry for an absent source survives the reconcile.

    A failed upload never lands in the file manager, so without the carve-out
    it would be purged on first read and the error badge could never appear.
    A ``completed`` entry for an absent source is still purged.
    """
    _use_local_file_manager(monkeypatch)
    await set_status(agent_id, "agent", "failed.pdf", type_="file", status=IngestionStatus.ERROR, error="boom")
    await set_status(agent_id, "agent", "ghost.pdf", type_="file", status=IngestionStatus.COMPLETED)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert [e["source"] for e in response.json()] == ["failed.pdf"]

    # the completed entry is gone; the error entry survives
    assert await get_status(agent_id, "agent", "ghost.pdf") is None
    assert await get_status(agent_id, "agent", "failed.pdf") is not None


async def test_empty_registry_returns_empty_list(secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    assert response.json() == []


async def test_forbidden_without_memory_read(secure_client, secure_client_headers, client, cheshire_cat):
    # default user has only CHAT:WRITE, no MEMORY.READ
    data = await create_new_user(secure_client, headers=secure_client_headers)
    res = await client.post("/auth/token", json={"username": data["username"], "password": new_user_password})
    received_token = res.json()["access_token"]

    response = await client.get(
        "/ingestion/status",
        headers={"Authorization": f"Bearer {received_token}", "X-Agent-ID": agent_id},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"


async def test_inflight_status_survives_when_source_absent(
    secure_client, secure_client_headers, cheshire_cat, monkeypatch
):
    """N2: in-flight entries (uploaded/downloading/downloaded/processing) are
    NEVER purged by the read-time reconcile, even when the source is not (yet)
    in the canonical lists — the processing queue must stay visible across
    sessions/workers. Only completed (vanished) and error (kept) are special.
    """
    _use_local_file_manager(monkeypatch)
    for status in ("uploaded", "downloading", "downloaded", "processing"):
        await set_status(agent_id, "agent", f"inflight_{status}.pdf", type_="file", status=status)

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert response.status_code == 200
    sources = {e["source"] for e in response.json()}
    assert sources == {
        "inflight_uploaded.pdf",
        "inflight_downloading.pdf",
        "inflight_downloaded.pdf",
        "inflight_processing.pdf",
    }


async def test_delete_status_error_survives(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    """DELETE /ingestion/status removes an error row (dismissal)."""
    _use_local_file_manager(monkeypatch)
    await set_status(agent_id, "agent", "failed.pdf", type_="file", status=IngestionStatus.ERROR, error="boom")

    response = await secure_client.delete(
        f"/ingestion/status?source={urllib.parse.quote('failed.pdf')}",
        headers=secure_client_headers,
    )
    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert await get_status(agent_id, "agent", "failed.pdf") is None


async def test_delete_inflight_status_refused(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    """DELETE /ingestion/status refuses in-flight sources: deleting the row
    while work is running would orphan the pipeline (the hooks re-create a
    completed row) — the teacher must remove the file instead.
    """
    _use_local_file_manager(monkeypatch)
    await set_status(agent_id, "agent", "busy.pdf", type_="file", status=IngestionStatus.PROCESSING)

    response = await secure_client.delete(
        f"/ingestion/status?source={urllib.parse.quote('busy.pdf')}",
        headers=secure_client_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is False
    assert body["reason"] == "in_flight"
    assert await get_status(agent_id, "agent", "busy.pdf") is not None


async def test_delete_status_not_found(secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.delete(
        f"/ingestion/status?source={urllib.parse.quote('nope.pdf')}",
        headers=secure_client_headers,
    )
    assert response.status_code == 200
    assert response.json() == {"deleted": False, "reason": "not_found"}
