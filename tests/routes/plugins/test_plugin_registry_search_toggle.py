"""Tests for the ``CAT_PLUGINS_REGISTRY_SEARCH_ENABLED`` toggle.

The remote plugin registry is a network dependency. Air-gapped or
privacy-conscious deployments must be able to disable it without losing the
local plugin listing.
"""
import importlib
import types

import pytest

from cat.core_plugins.base_plugin.registry import CheshireCatPluginRegistry
from cat.env_check import validate_env
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


async def test_download_enabled_uses_http_client(monkeypatch):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", "true")
    _FakeAsyncClient.instances.clear()
    _patch_registry_http_client(monkeypatch, _FakeAsyncClient)

    await CheshireCatPluginRegistry().download_plugin("https://registry.test/air-gapped-plugin")

    assert len(_FakeAsyncClient.instances) == 1
    client = _FakeAsyncClient.instances[0]
    assert client.calls and client.calls[0][0] == "post"
    assert client.calls[0][1].endswith("/download")


async def test_install_from_registry_route_reports_disabled_setting(
    monkeypatch, lizard, secure_client, secure_client_headers, cheshire_cat
):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", "false")
    # the package conftest patches only search_plugins; restore the real download
    # so this test exercises the env toggle end-to-end.
    monkeypatch.setattr(
        lizard.plugin_registry,
        "download_plugin",
        types.MethodType(CheshireCatPluginRegistry.download_plugin, lizard.plugin_registry),
    )

    def _forbidden(*args, **kwargs):
        raise AssertionError("the remote registry must not be contacted")

    _patch_registry_http_client(monkeypatch, _forbidden)

    response = await secure_client.post(
        "/plugins/install/registry",
        json={"url": "https://registry.test/air-gapped-plugin"},
        headers=secure_client_headers,
    )

    assert response.status_code == 400
    message = response.json()["detail"]
    assert "disabled" in message
    assert "CAT_PLUGINS_REGISTRY_SEARCH_ENABLED" in message


async def test_available_plugins_route_lists_local_plugins_when_search_disabled(
    monkeypatch, lizard, secure_client, secure_client_headers, cheshire_cat
):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", "false")
    # the package conftest substitutes the registry with a fixed catalogue; restore the real
    # implementation so this test exercises the env toggle end-to-end.
    monkeypatch.setattr(
        lizard.plugin_registry,
        "search_plugins",
        types.MethodType(CheshireCatPluginRegistry.search_plugins, lizard.plugin_registry),
    )

    def _forbidden(*args, **kwargs):
        raise AssertionError("the remote registry must not be contacted")

    _patch_registry_http_client(monkeypatch, _forbidden)

    response = await secure_client.get("/plugins/", headers=secure_client_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["registry"] == []
    assert body["installed"]


@pytest.mark.parametrize("value", ["true", "false", "1", "0"])
def test_env_check_accepts_boolean_values(monkeypatch, value):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", value)

    problems = [e for e in validate_env().errors if "CAT_PLUGINS_REGISTRY_SEARCH_ENABLED" in e]

    assert problems == []


@pytest.mark.parametrize("value", ["maybe", "yes", "2", "TRUE", ""])
def test_env_check_reports_invalid_boolean(monkeypatch, value):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", value)

    problems = [e for e in validate_env().errors if "CAT_PLUGINS_REGISTRY_SEARCH_ENABLED" in e]

    assert problems
