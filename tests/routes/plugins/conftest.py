"""Fixtures shared by the plugin route tests.

``GET /plugins/`` and ``GET /plugins/installed`` ask the public plugin registry
over the network, so every test asserting on the ``registry`` list was flaky
twice over: it failed when the remote service answered slower than the test
client's timeout (``httpx.ReadTimeout`` raised out of the request, not an
assertion), and it depended on the live catalogue still containing whatever
plugin the test searched for.

The registry is served from a fixed catalogue instead. The fixture is autouse
so that no test in this package reaches a third-party service by accident.
"""
import pytest

from cat.looking_glass.models import PluginManifest

# Plugin urls that no installed plugin can claim: `get_available_plugins` drops
# a registry entry whose plugin_url matches an installed manifest.
REGISTRY_CATALOGUE = [
    {
        "id": "https://registry.test/podcast-plugin",
        "name": "Podcast reader",
        "description": "Subscribes to a podcast feed and summarises the episodes",
        "plugin_url": "https://registry.test/podcast-plugin",
        "version": "1.0.0",
        "author_name": "Registry Test",
        "tags": "podcast, audio",
    },
    {
        "id": "https://registry.test/weather-plugin",
        "name": "Weather",
        "description": "Answers with the forecast for a given city",
        "plugin_url": "https://registry.test/weather-plugin",
        "version": "2.1.0",
        "author_name": "Registry Test",
        "tags": "weather",
    },
]


@pytest.fixture(autouse=True)
async def mock_plugin_registry(monkeypatch, lizard):
    """Serve the plugin registry from REGISTRY_CATALOGUE instead of the network.

    The registry instance is patched, not its class: the plugin that provides
    it is imported by the plugin loader, so the class object the lizard holds
    is not necessarily the one a plain import returns.

    The fake keeps the contract of the real implementation: it filters on the
    query the same way the registry does (case-insensitive, over name and
    description) and returns manifests, never raising.
    """
    async def fake_search_plugins(query: str = None):
        manifests = [PluginManifest(**entry) for entry in REGISTRY_CATALOGUE]
        if not query:
            return manifests

        needle = query.lower()
        return [m for m in manifests if needle in f"{m.name}{m.description}".lower()]

    monkeypatch.setattr(lizard.plugin_registry, "search_plugins", fake_search_plugins)
    return REGISTRY_CATALOGUE
