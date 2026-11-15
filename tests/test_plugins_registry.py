from cat.core_plugins.base_plugin import registry as registry_module
from cat.core_plugins.base_plugin.registry import CheshireCatPluginRegistry


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


class _RecordingAsyncClient:
    calls: list[tuple] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, **kwargs):
        _RecordingAsyncClient.calls.append(("get", url))
        return _FakeResponse({"plugins": [{"url": "https://registry.test/plugin", "name": "Test"}]})

    async def post(self, url, **kwargs):
        _RecordingAsyncClient.calls.append(("post", url))
        return _FakeResponse([{"url": "https://registry.test/plugin", "name": "Test"}])


async def test_search_plugins_enabled_when_flag_unset(monkeypatch):
    monkeypatch.delenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", raising=False)
    monkeypatch.setattr(registry_module, "_registry_search_disabled_logged", False)
    monkeypatch.setattr(registry_module.httpx, "AsyncClient", _RecordingAsyncClient)
    _RecordingAsyncClient.calls = []

    manifests = await CheshireCatPluginRegistry().search_plugins()

    assert len(manifests) == 1
    assert _RecordingAsyncClient.calls


async def test_search_plugins_disabled_returns_empty_and_logs_once(monkeypatch):
    monkeypatch.setenv("CAT_PLUGINS_REGISTRY_SEARCH_ENABLED", "false")
    monkeypatch.setattr(registry_module, "_registry_search_disabled_logged", False)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("the registry must not be contacted when the search is disabled")

    monkeypatch.setattr(registry_module.httpx, "AsyncClient", _must_not_be_called)

    infos = []
    monkeypatch.setattr(registry_module.log, "info", lambda message: infos.append(message))

    registry = CheshireCatPluginRegistry()
    assert await registry.search_plugins() == []
    assert await registry.search_plugins() == []

    assert len(infos) == 1
