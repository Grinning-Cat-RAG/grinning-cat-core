import threading

from fastapi.responses import StreamingResponse
from loguru import logger

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.looking_glass import CheshireCat
from cat.mixins import BotMixin
from cat.services.factory.vector_db import QdrantHandler


# Captured at import time, before the autouse conftest fixture replaces ``QdrantHandler.close`` with a no-op.
_ORIGINAL_QDRANT_CLOSE = QdrantHandler.__dict__["close"]


class _FakeBot(BotMixin):
    @property
    def agent_key(self) -> str:
        return "fake_agent"

    async def embedder(self):
        return None

    async def toggle_plugin(self, plugin_id: str):
        return None


def _run_in_thread_without_loop(fn):
    """Run ``fn`` in a thread that has no running event loop, returning anything it raised."""
    errors = []

    def runner():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - surface anything raised by the destructor
            errors.append(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    return errors


def _capture_warnings():
    records = []
    sink_id = logger.add(lambda message: records.append(message.record["message"]), level="WARNING")
    return records, sink_id


# ---------------------------------------------------------------------------
# Point 1: explicit per-request shutdown
# ---------------------------------------------------------------------------

async def test_per_request_ccat_is_closed_once_after_response(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    calls = []
    original_shutdown = CheshireCat.shutdown

    async def recording_shutdown(self):
        calls.append(self)
        await original_shutdown(self)

    monkeypatch.setattr(CheshireCat, "shutdown", recording_shutdown)

    app = secure_client._fastapi_test_app

    @app.get("/_test/lifecycle")
    async def _lifecycle(info: AuthorizedInfo = check_permissions(AuthResource.CHAT, AuthPermission.READ)):
        return {"agent": info.cheshire_cat.agent_key}

    first = await secure_client.get("/_test/lifecycle", headers=secure_client_headers)
    assert first.status_code == 200
    assert len(calls) == 1

    second = await secure_client.get("/_test/lifecycle", headers=secure_client_headers)
    assert second.status_code == 200
    assert len(calls) == 2
    # every request builds its own CheshireCat
    assert calls[0] is not calls[1]


async def test_streaming_keeps_ccat_alive_until_stream_ends(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    events = []
    original_shutdown = CheshireCat.shutdown

    async def recording_shutdown(self):
        events.append("shutdown")
        await original_shutdown(self)

    monkeypatch.setattr(CheshireCat, "shutdown", recording_shutdown)

    app = secure_client._fastapi_test_app

    @app.get("/_test/stream")
    async def _stream(info: AuthorizedInfo = check_permissions(AuthResource.CHAT, AuthPermission.READ)):
        ccat = info.cheshire_cat

        async def gen():
            events.append("stream-start")
            assert ccat._closed is False
            yield b"a"
            events.append("stream-mid")
            assert ccat._closed is False
            yield b"b"
            events.append("stream-end")

        return StreamingResponse(gen())

    response = await secure_client.get("/_test/stream", headers=secure_client_headers)
    assert response.status_code == 200
    assert response.content == b"ab"
    # the cat is closed only after the whole stream has been sent
    assert events == ["stream-start", "stream-mid", "stream-end", "shutdown"]


# ---------------------------------------------------------------------------
# Point 3: destructor hardening
# ---------------------------------------------------------------------------

def test_bot_mixin_del_without_running_loop_does_not_warn():
    bot = _FakeBot()
    records, sink_id = _capture_warnings()
    try:
        errors = _run_in_thread_without_loop(bot.__del__)
    finally:
        logger.remove(sink_id)

    assert errors == []
    assert not any("Error while shutting down" in message for message in records)


def test_vector_handler_del_without_running_loop_does_not_warn(monkeypatch):
    monkeypatch.setattr(QdrantHandler, "close", _ORIGINAL_QDRANT_CLOSE)
    handler = QdrantHandler.__new__(QdrantHandler)

    records, sink_id = _capture_warnings()
    try:
        errors = _run_in_thread_without_loop(handler.__del__)
    finally:
        logger.remove(sink_id)

    assert errors == []
    assert not any("Error while closing the vector database handler" in message for message in records)


async def test_bot_mixin_shutdown_is_idempotent():
    bot = _FakeBot()
    await bot.shutdown()
    assert bot._closed is True
    await bot.shutdown()
    assert bot._closed is True


async def test_qdrant_close_is_idempotent(monkeypatch):
    monkeypatch.setattr(QdrantHandler, "close", _ORIGINAL_QDRANT_CLOSE)
    handler = QdrantHandler.__new__(QdrantHandler)

    class _Inner:
        closed = False

    class _Client:
        def __init__(self):
            self._client = _Inner()
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1
            self._client.closed = True

    fake_client = _Client()
    handler._client = fake_client

    await handler.close()
    await handler.close()

    assert fake_client.close_calls == 1
    assert handler._client is None
