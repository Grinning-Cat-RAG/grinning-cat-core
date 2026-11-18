"""March Hare must deliver a plugin event to the sibling workers of the same pod.

``pod_id`` identifies the container, not the process: every uvicorn worker of a
container shares it. The event is therefore tagged with a process-scoped
``worker_id`` too, and only the process that published it skips it.
"""
import json
from types import SimpleNamespace

import pytest

import cat.core_plugins.march_hare.march_hare as march_hare_mod
from cat.core_plugins.march_hare.march_hare import MarchHare, MarchHareConfig

POD = "pod-a"
WORKER = "1234:5678"
OTHER_WORKER = "1234:9999"
OTHER_POD = "pod-b"

INSTALLATION = MarchHareConfig.events["PLUGIN_INSTALLATION"]
UNINSTALLATION = MarchHareConfig.events["PLUGIN_UNINSTALLATION"]
SYSTEM_TOGGLE = MarchHareConfig.events["PLUGIN_SYSTEM_TOGGLE"]


class FakePluginManager:
    def __init__(self):
        self.installed = []

    async def install_extracted_plugin(self, plugin_id, persist=True):
        self.installed.append((plugin_id, persist))


class FakeLizard:
    def __init__(self):
        self.plugin_manager = FakePluginManager()
        self.activated = []
        self.synced = 0
        self.uninstalled = []

    def activate_plugin_endpoints(self, plugin_id):
        self.activated.append(plugin_id)

    async def sync_system_plugins(self):
        self.synced += 1

    async def uninstall_plugin(self, plugin_id, dispatch_event=True):
        self.uninstalled.append((plugin_id, dispatch_event))


@pytest.fixture
def hare(monkeypatch):
    fake = SimpleNamespace(pod_id=POD, worker_id=WORKER)
    monkeypatch.setattr(march_hare_mod, "_march_hare", fake)
    return fake


@pytest.fixture
def lizard():
    return FakeLizard()


def event(event_type, source_pod=POD, source_worker=WORKER, **payload):
    data = {
        "event_type": event_type,
        "payload": json.dumps({"plugin_id": "my_plugin", **payload}),
        "source_pod": source_pod,
    }
    if source_worker is not None:
        data["source_worker"] = source_worker
    return data


async def test_same_pod_same_worker_is_skipped(hare, lizard):
    await march_hare_mod._handle_plugin_event(lizard, event(INSTALLATION))
    await march_hare_mod._handle_plugin_event(lizard, event(SYSTEM_TOGGLE))
    await march_hare_mod._handle_plugin_event(lizard, event(UNINSTALLATION))

    assert lizard.plugin_manager.installed == []
    assert lizard.activated == []
    assert lizard.synced == 0
    assert lizard.uninstalled == []


async def test_same_pod_other_worker_applies_installation(hare, lizard):
    await march_hare_mod._handle_plugin_event(lizard, event(INSTALLATION, source_worker=OTHER_WORKER))

    assert lizard.plugin_manager.installed == [("my_plugin", False)]
    assert lizard.activated == ["my_plugin"]


async def test_same_pod_other_worker_applies_system_toggle(hare, lizard):
    await march_hare_mod._handle_plugin_event(lizard, event(SYSTEM_TOGGLE, source_worker=OTHER_WORKER))

    assert lizard.synced == 1


async def test_same_pod_other_worker_applies_uninstallation(hare, lizard):
    await march_hare_mod._handle_plugin_event(lizard, event(UNINSTALLATION, source_worker=OTHER_WORKER))

    assert lizard.uninstalled == [("my_plugin", False)]


async def test_other_pod_applies(hare, lizard):
    await march_hare_mod._handle_plugin_event(lizard, event(INSTALLATION, source_pod=OTHER_POD))

    assert lizard.plugin_manager.installed == [("my_plugin", False)]


async def test_legacy_event_without_source_worker_applies(hare, lizard):
    await march_hare_mod._handle_plugin_event(lizard, event(INSTALLATION, source_worker=None))

    assert lizard.plugin_manager.installed == [("my_plugin", False)]


async def test_notify_event_publishes_pod_and_worker():
    published = {}

    class FakeRedis:
        async def xadd(self, name, fields, maxlen, approximate):
            published["name"] = name
            published["fields"] = fields

    real_class = MarchHare.__wrapped__
    hare = real_class.__new__(real_class)
    hare.pod_id = POD
    hare.worker_id = WORKER
    hare._async_redis_client = FakeRedis()

    await hare.notify_event(INSTALLATION, {"plugin_id": "my_plugin"}, "stream")

    assert published["name"] == "stream"
    assert published["fields"]["source_pod"] == POD
    assert published["fields"]["source_worker"] == WORKER
    assert json.loads(published["fields"]["payload"]) == {"plugin_id": "my_plugin"}
