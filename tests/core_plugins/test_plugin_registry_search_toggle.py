"""Tests for the ``CAT_PLUGINS_REGISTRY_SEARCH_ENABLED`` toggle.

The remote plugin registry is a network dependency. Air-gapped or
privacy-conscious deployments must be able to disable it without losing the
local plugin listing.
"""
import importlib
import types

import pytest

from cat.core_plugins.base_plugin.registry import CheshireCatPluginRegistry
from cat.exceptions import CustomValidationException

REGISTRY_MODULE = "cat.core_plugins.base_plugin.registry"
REGISTRY_PLUGIN = {
    "url": "https://registry.test/air-gapped-plugin",
    "name": "Air gapped",
    "description": "Plugin served by the fake registry",
    "version": "1.0.0",
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.content = b"zip-bytes"

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeAsyncClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, params=None):
        self.calls.append(("get", url, params))
        return _FakeResponse({"plugins": [REGISTRY_PLUGIN]})

    async def post(self, url, json=None):
        self.calls.append(("post", url, json))
        return _FakeResponse([REGISTRY_PLUGIN])


def _patch_registry_http_client(monkeypatch, client_cls):
    """Replace only the registry module's ``httpx`` reference (not global httpx)."""
    module = importlib.import_module(REGISTRY_MODULE)
    monkeypatch.setattr(module, "httpx", types.SimpleNamespace(AsyncClient=client_cls))


async def test_search_disabled_returns_empty_without_network(monkeypatch):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", "false")

    calls = []

    def _forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("httpx.AsyncClient must not be instantiated when search is disabled")

    _patch_registry_http_client(monkeypatch, _forbidden)

    assert await CheshireCatPluginRegistry().search_plugins() == []
    assert await CheshireCatPluginRegistry().search_plugins("air") == []
    assert calls == []


async def test_search_disabled_logs_once(monkeypatch):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", "false")
    module = importlib.import_module(REGISTRY_MODULE)
    monkeypatch.setattr(module, "_registry_search_disabled_logged", False)

    infos = []
    monkeypatch.setattr(module.log, "info", lambda message: infos.append(message))
    _patch_registry_http_client(monkeypatch, _FakeAsyncClient)

    registry = CheshireCatPluginRegistry()
    assert await registry.search_plugins() == []
    assert await registry.search_plugins("air") == []

    assert len(infos) == 1
    assert "CAT_PLUGINS_REGISTRY_SEARCH_ENABLED" in infos[0]
    assert _FakeAsyncClient.instances == []


async def test_search_enabled_uses_http_client(monkeypatch):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", "true")
    _FakeAsyncClient.instances.clear()
    _patch_registry_http_client(monkeypatch, _FakeAsyncClient)

    result = await CheshireCatPluginRegistry().search_plugins()

    assert len(_FakeAsyncClient.instances) == 1
    client = _FakeAsyncClient.instances[0]
    assert client.calls and client.calls[0][0] == "get"
    assert [m.id for m in result] == ["https://registry.test/air-gapped-plugin"]


async def test_download_disabled_raises_without_network(monkeypatch):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", "false")

    calls = []

    def _forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("httpx.AsyncClient must not be instantiated when the registry is disabled")

    _patch_registry_http_client(monkeypatch, _forbidden)

    with pytest.raises(CustomValidationException) as exc_info:
        await CheshireCatPluginRegistry().download_plugin("https://registry.test/air-gapped-plugin")

    message = str(exc_info.value)
    assert "disabled" in message
    assert "CAT_PLUGINS_REGISTRY_SEARCH_ENABLED" in message
    assert calls == []
