"""Webhooks are notified without blocking the event loop, with a timeout, and concurrently."""
import asyncio
import time

import httpx
import pytest
from pydantic import ValidationError

import cat.core_plugins.webhooks.webhooks as webhooks_mod


def _webhook(url):
    return {"url": url, "event": "embedder_updated", "secret": webhooks_mod.crypto.encrypt("secret")}


def test_a_slow_webhook_does_not_block_the_event_loop(monkeypatch):
    seen_timeouts = []

    pending = {}

    async def slow_post(self, url, **kwargs):
        seen_timeouts.append(self.timeout)
        pending["start"] = time.monotonic()
        await asyncio.sleep(0.2)
        pending["end"] = time.monotonic()
        return httpx.Response(200)

    monkeypatch.setattr(httpx.AsyncClient, "post", slow_post)
    ticks = []

    async def ticker():
        for _ in range(10):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    async def run():
        await asyncio.gather(
            webhooks_mod.trigger_webhook(webhooks_mod.WebhookPayload(**_webhook("http://hook.invalid")), {"a": 1}),
            ticker(),
        )

    asyncio.run(run())

    # the loop kept serving other work while the webhook was pending
    assert [t for t in ticks if pending["start"] < t < pending["end"]]
    assert seen_timeouts and seen_timeouts[0].read == webhooks_mod.WEBHOOK_TIMEOUT_SECONDS


def test_the_webhooks_of_an_event_are_notified_concurrently_and_a_failure_does_not_stop_the_others(monkeypatch):
    called = []

    async def post(self, url, **kwargs):
        called.append(url)
        await asyncio.sleep(0.2)
        if "broken" in url:
            raise httpx.ConnectError("refused")
        return httpx.Response(200)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    hooks = [_webhook("http://broken.invalid"), _webhook("http://a.invalid"), _webhook("http://b.invalid")]

    start = time.monotonic()
    asyncio.run(webhooks_mod.trigger_webhooks(hooks, {"success": True}, "on embedder updated"))
    elapsed = time.monotonic() - start

    assert sorted(called) == ["http://a.invalid", "http://b.invalid", "http://broken.invalid"]
    assert elapsed < 0.5  # one round of 0.2 s, not three


def test_every_hook_awaits_its_webhooks(monkeypatch):
    # trigger_webhook is a coroutine: a hook calling it without awaiting would silently never notify
    import inspect

    notified = []

    async def record(webhooks, payload, what):
        notified.append(what)

    async def get_webhooks(agent_key, event):
        return [_webhook("http://hook.invalid")]

    monkeypatch.setattr(webhooks_mod, "trigger_webhooks", record)
    monkeypatch.setattr(webhooks_mod.crud_webhook, "get_webhooks", get_webhooks)
    owner = type("Owner", (), {"agent_key": "agent", "id": "chat", "plugin_manager": type("P", (), {"plugins": {}})()})()

    hooks = [
        obj for obj in vars(webhooks_mod).values()
        if type(obj).__name__ == "CatHook" and inspect.iscoroutinefunction(obj.function)
    ]
    for hook_ in hooks:
        params = list(inspect.signature(hook_.function).parameters)
        args = {"source": "s", "stored_points": [], "success": True, "plugin_id": "p", "plugin_path": "x"}
        kwargs = {name: (owner if name in ("cat", "lizard") else args[name]) for name in params}
        asyncio.run(hook_.function(**kwargs))

    assert len(hooks) == 6 and len(notified) == 6


def _recording_transport(received):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(308, headers={"Location": "http://hook.invalid/new"})
        received.append(request)
        return httpx.Response(200)

    return httpx.MockTransport(handler)


def test_the_body_received_is_exactly_the_one_signed_and_redirects_are_followed(monkeypatch):
    import hashlib
    import hmac

    received = []
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        webhooks_mod.httpx, "AsyncClient",
        lambda **kwargs: real_client(transport=_recording_transport(received), **kwargs),
    )
    payload = {"source": "Città di Roma.pdf", "success": True}

    asyncio.run(webhooks_mod.trigger_webhook(
        webhooks_mod.WebhookPayload(**_webhook("http://hook.invalid/old")), payload,
    ))

    assert [r.url.path for r in received] == ["/new"]  # a 308 (e.g. http -> https) is followed, as before
    request = received[0]
    expected = hmac.new(b"secret", msg=request.content, digestmod=hashlib.sha256).hexdigest()
    assert request.headers["X-CheshireCat-Signature"] == expected  # the receiver can verify what it got
    assert request.headers["Content-Type"] == "application/json"


def test_a_webhook_without_a_signature_header_cannot_be_declared():
    # a null header key could never be sent: the webhook would fail on every event
    with pytest.raises(ValidationError):
        webhooks_mod.WebhookPayload(url="http://hook.invalid", event="embedder_updated", secret="s", header_key=None)

    default = webhooks_mod.WebhookPayload(url="http://hook.invalid", event="embedder_updated", secret="s")
    assert default.header_key == "X-CheshireCat-Signature"
