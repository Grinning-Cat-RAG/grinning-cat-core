"""Lifecycle of a RabbitHole ingestion.

The ingestion states (start, URL download, processing, heartbeat, error) are RabbitHole helpers, not plugin hooks:
the hooks left to the plugins are the ones that act on the content (parsers, documents, chunks, images, stored
points). A stateful property test drives concurrent ingestions with injected faults and checks the lifecycle rules
at every run.
"""
import asyncio
import json
import os
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.documents import Document

import cat.rabbit_hole as rabbit_hole_module
from cat.core_plugins.base_plugin.hooks import file_manager as file_manager_hooks
from cat.core_plugins.base_plugin.hooks import rabbithole as rabbithole_hooks
from cat.looking_glass.mad_hatter.decorators.hook import CatHook
from cat.rabbit_hole import RabbitHole

BASE_PLUGIN_HOOKS = {
    name for module in (rabbithole_hooks, file_manager_hooks)
    for name, value in vars(module).items() if isinstance(value, CatHook)
}

REMOVED_STATE_HOOKS = [
    "rabbithole_ingestion_start",
    "rabbithole_url_downloading",
    "rabbithole_url_download_completed",
    "rabbithole_ingestion_processing",
    "rabbithole_ingestion_error",
    "rabbithole_processing_heartbeat_start",
    "rabbithole_processing_heartbeat_stop",
    "run_in_ingestion_executor",
    "factory_allowed_ingestions",
]


def test_ingestion_state_hooks_are_not_offered_to_plugins(cheshire_cat):
    for plugin_manager in (cheshire_cat.plugin_manager, cheshire_cat.lizard.plugin_manager):
        offered = set(plugin_manager.hooks)
        assert offered.isdisjoint(REMOVED_STATE_HOOKS), offered.intersection(REMOVED_STATE_HOOKS)


# --- fake world -------------------------------------------------------------------------------------------------

class RecordingPluginManager:
    """Like ``MadHatter.execute_hook``, fails on a hook that no plugin offers (here: the base plugin's hooks)."""

    def __init__(self, events):
        self.events = events

    async def execute_hook(self, hook_name, *args, caller):
        if hook_name not in BASE_PLUGIN_HOOKS:
            raise Exception(f"Hook {hook_name} not present in any plugin")
        if hook_name == "after_rabbithole_stored_documents":
            source, points = args
            self.events.append((source, "stored", {"points": points, "caller": caller}))
        return args[0] if args else None


class FakeCat:
    def __init__(self, events, failing_saves):
        self.agent_key = "agent_test"
        self.plugin_manager = RecordingPluginManager(events)
        self.events = events
        self.failing_saves = failing_saves

    async def save_file(self, file_bytes, content_type, source, chat_id=None):
        if source in self.failing_saves:
            raise OSError("disk full")
        self.events.append((source, "save", {"chat_id": chat_id}))


class FakeNotifier:
    def __init__(self):
        self.errors = []

    def has_ws_connection(self):
        return False

    async def send_error(self, message):
        self.errors.append(message)


class World:
    """Shared state of one example: the event log, the fault plan and the concurrency probe."""

    def __init__(self, plans):
        self.events = []
        self.plans = {plan["id"]: plan for plan in plans}
        self.cat = FakeCat(self.events, {p["id"] for p in plans if p["fault"] == "save"})
        self.stray = SimpleNamespace(id="chat_abc", notifier=FakeNotifier())
        self.active = 0
        self.max_active = 0

    def client_factory(self):
        world = self

        class FakeResponse:
            headers = {"Content-Type": "text/html; charset=utf-8"}
            content = b"<p>hello</p>"

            def raise_for_status(self):
                pass

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                await asyncio.sleep(0)
                if world.plans[url]["fault"] == "download":
                    raise httpx.ConnectError("unreachable")
                return FakeResponse()

        return FakeClient


class ScriptedRabbitHole(RabbitHole):
    """RabbitHole whose state helpers record the lifecycle and whose heavy steps follow the fault plan."""

    world: World

    async def setup(self, _cat):
        self.cat = self.world.cat
        self.stray = _cat if _cat is self.world.stray else None

    def _record(self, source, event, **data):
        self.world.events.append((source, event, data))

    async def _ingestion_start(self, source, metadata, is_url):
        self._record(source, "start", is_url=is_url, metadata=metadata)

    async def _ingestion_url_downloading(self, url):
        self._record(url, "downloading")

    async def _ingestion_url_downloaded(self, url):
        self._record(url, "downloaded")

    async def _ingestion_processing(self, source):
        self._record(source, "processing")

    async def _ingestion_processing_heartbeat_start(self, source):
        self._record(source, "hb_start")

    async def _ingestion_processing_heartbeat_stop(self, source):
        self._record(source, "hb_stop")

    async def _ingestion_error(self, source, error):
        self._record(source, "error", error=error)

    async def _parse_to_docs(self, source, file_bytes, content_type):
        world = self.world
        world.active += 1
        world.max_active = max(world.max_active, world.active)
        try:
            await asyncio.sleep(0)  # let the other ingestions run, if the guard allows it
            fault = world.plans[source]["fault"]
            if fault == "parse":
                raise ValueError("unreadable file")
            if fault == "empty":
                return [], []
            return [Document(page_content="hello")], []
        finally:
            world.active -= 1

    async def store_documents(self, docs, source, file_hash=None, metadata=None, images=None, source_bytes=None):
        if self.world.plans[source]["fault"] == "store":
            raise RuntimeError("vector db down")
        return [f"point-of-{source}"]

    async def _send_notification_message(self, message):
        if message.startswith("Finished reading"):
            self._record(message.split()[2].rstrip(","), "finished")


FAULTS_FOR_URL = [None, "download", "parse", "empty", "store"]
FAULTS_FOR_UPLOAD = [None, "no_bytes", "save", "parse", "empty", "store"]


@st.composite
def ingestion_plan(draw, index):
    is_url = draw(st.booleans())
    store_file = draw(st.booleans())
    faults = FAULTS_FOR_URL if is_url else FAULTS_FOR_UPLOAD
    fault = draw(st.sampled_from([f for f in faults if f != "save" or store_file]))
    return {
        "id": f"http://example.com/doc{index}" if is_url else f"doc{index}.txt",
        "is_url": is_url,
        "store_file": store_file,
        "on_stray": draw(st.booleans()),
        "fault": fault,
    }


plans_strategy = st.integers(1, 4).flatmap(lambda n: st.tuples(*[ingestion_plan(i) for i in range(n)]))


def _run(plans, max_concurrency):
    world = World(plans)
    ScriptedRabbitHole.world = world
    rabbit_hole = ScriptedRabbitHole()  # one shared instance, as lizard.rabbit_hole

    async def ingest(plan):
        file = plan["id"] if plan["is_url"] else BytesIO(b"" if plan["fault"] == "no_bytes" else b"content")
        await rabbit_hole.ingest_file(
            cat=world.stray if plan["on_stray"] else world.cat,
            file=file,
            metadata={"tag": plan["id"]},
            filename=None if plan["is_url"] else plan["id"],
            store_file=plan["store_file"],
            content_type="text/plain",
        )

    async def main():
        await asyncio.gather(*[ingest(plan) for plan in plans])
        return rabbit_hole_module._ingestion_semaphore

    with (
        patch.object(rabbit_hole_module, "_ingestion_semaphore", None),
        patch.object(rabbit_hole_module, "AsyncClient", world.client_factory()),
        patch.dict(os.environ, {"CAT_INGESTION_MAX_CONCURRENCY": str(max_concurrency)}),
    ):
        semaphore = asyncio.run(main())
    return world, semaphore


@settings(max_examples=300, deadline=None)
@given(plans=plans_strategy, max_concurrency=st.sampled_from([0, 1, 2]))
def test_every_ingestion_follows_the_lifecycle(plans, max_concurrency):
    world, semaphore = _run(plans, max_concurrency)

    # the guard bounds the heavy bodies and is released on every exit path
    if max_concurrency > 0:
        assert world.max_active <= max_concurrency
        assert semaphore._value == max_concurrency
    else:
        assert semaphore is None

    stray_errors = list(world.stray.notifier.errors)
    for plan in plans:
        source = plan["id"]
        run = [(event, data) for src, event, data in world.events if src == source]
        names = [event for event, _ in run]
        failed = plan["fault"] is not None

        # one start, first, with the metadata and the URL flag of the request
        assert names[0] == "start" and names.count("start") == 1
        assert run[0][1] == {"is_url": plan["is_url"], "metadata": {"tag": source}}

        # the heartbeat is always stopped once, then the stored points are notified once, last
        assert names[-2:] == ["hb_stop", "stored"]
        assert names.count("hb_stop") == 1 and names.count("stored") == 1
        stored = run[-1][1]
        assert stored["points"] == ([] if failed else [f"point-of-{source}"])
        assert stored["caller"] is (world.stray if plan["on_stray"] else world.cat)

        # the heartbeat starts at most once, after the processing state
        assert names.count("hb_start") <= 1
        if "hb_start" in names:
            assert names.index("processing") < names.index("hb_start") < names.index("hb_stop")
        reached_body = plan["fault"] in (None, "parse", "empty", "store")
        assert ("hb_start" in names) == reached_body
        assert ("processing" in names) == reached_body

        # URL download states only for URLs; "downloaded" only after a successful download
        assert ("downloading" in names) == plan["is_url"]
        assert ("downloaded" in names) == (plan["is_url"] and plan["fault"] != "download")
        if "downloaded" in names:
            assert names.index("downloading") < names.index("downloaded")

        # the file is persisted before processing, only for uploads asked to be stored
        saved = plan["store_file"] and not plan["is_url"] and plan["fault"] not in ("no_bytes", "save")
        assert ("save" in names) == saved
        if saved:
            assert names.index("save") < names.index("processing")
            assert run[names.index("save")][1]["chat_id"] == ("chat_abc" if plan["on_stray"] else None)

        # exactly one outcome: the error state or the success notification
        assert names.count("error") == (1 if failed else 0)
        assert names.count("finished") == (0 if failed else 1)

        # the chat of a failed ingestion is told about the error
        if failed and plan["on_stray"]:
            stray_errors.remove(next(e for e in stray_errors if e.startswith(f"Error processing {source}:")))
    assert stray_errors == []


async def test_ingest_memory_records_start_and_error():
    rabbit_hole = ScriptedRabbitHole()
    world = World([])
    ScriptedRabbitHole.world = world

    async def embedder():
        return SimpleNamespace()

    world.cat.lizard = SimpleNamespace(embedder=embedder)
    memories = {"embedder": "OtherEmbedder", "collections": {"declarative": []}}

    await rabbit_hole.ingest_memory(cat=world.cat, file=BytesIO(json.dumps(memories).encode()), filename="mem.json")

    events = [(src, event) for src, event, _ in world.events]
    assert events == [("mem.json", "start"), ("mem.json", "error")]
    assert "Embedder mismatch" in world.events[-1][2]["error"]
