"""Stateful test of the lifecycle of the sources: uploads, crashes of a worker, resumes, changes of the embedder and
deletions, with faults injected in the phases. The rules are checked after every operation:

- no point without its vector, and the points of a source come from one ingestion only;
- a ``completed`` source has its points (as many as the embedding made), its chunks and a diary of complete phases
  with the current tags;
- an ``error`` source says why, and has no owner;
- a deleted source leaves no row, no points, no chunks;
- a source whose worker crashed (its lock left to expire) stays ``processing`` until the automatic resume of a pod
  completes it; a change of the embedder caught by the resume re-embeds the stored chunks without parsing again;
- an interrupted deletion leaves the source ``deleting`` until the resume completes it: it is never brought back;
- a change of the embedder recreates the collections and re-embeds every source from its stored chunks, also while an
  upload runs;
- concurrent uploads and deletions of the same source run one at a time; a source whose row is deleted while it is
  processed (its chat deleted) leaves nothing behind;
- every upload notifies ``after_rabbithole_stored_documents`` exactly once (points, or none on failure);
- a source ingested before the states existed gets its state at a change of the embedder, unless it is deleted
  meanwhile: then it stays deleted;
- fixing the settings of a failed phase lets the resume run again the sources that failed there, and only those;
- a reingest runs the source again from scratch; only a failed ingestion can be dismissed or retried (409 otherwise),
  a retry runs once even when asked twice at once.
"""
import asyncio
import copy
import os
from io import BytesIO

from unittest.mock import patch

import pytest
from redis.asyncio.lock import Lock
from redis.exceptions import LockError

from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
from cat.db.cruds import ingestions as crud_ingestions
from cat.rabbit_hole import RabbitHole, SourceNotKept, chunks, engine, recovery
from cat.rabbit_hole.phases import IngestionPhase, PhaseResult, compute_tags
from cat.services.memory.models import VectorMemoryType

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, example, given, settings, strategies as st  # noqa: E402

SOURCES = ("alice.txt", "bob.txt")
FAULTS = (None, None, "parse", "embed", "plugin_error", "plugin_not_ready", "empty")
TEXT = ("Down, down, down. Would the fall never come to an end? " * 40).encode()


class GraphPhase(IngestionPhase):
    """A plugin phase after the embedding: fails or is not ready for the sources told so (instantiated for every run,
    as the RabbitHole does with the classes of the plugins)."""

    id = "graph"
    depends_on = ("embedding",)

    async def settings(self, cat):
        return CURRENT["world"].graph_settings

    async def process(self, source, cat):
        world = CURRENT["world"]
        fault = world.faults.get(source.name)
        if fault == "plugin_error":
            return PhaseResult(status="error", reason="graph failed")
        if fault == "plugin_not_ready" and world.not_ready_left.get(source.name, 0) > 0:
            world.not_ready_left[source.name] -= 1
            return PhaseResult(status="not_ready", retry_after=0)
        return PhaseResult(status="done")

    async def delete(self, source, cat):
        world = CURRENT["world"]
        if world.delete_fails.pop(source.name, False):
            raise ConnectionError("graph store down")
        world.graph_deleted.append(source.name)


class World:
    def __init__(self):
        self.faults = {}
        self.not_ready_left = {}
        self.graph_deleted = []
        self.stored_hook = []
        self.delete_fails = {}
        self.graph_settings = 1
        self.legacy = set()  # sources ingested before the states existed: points and file, no row
        self.parsed = []
        self.embedder = "embedder-1"


CURRENT = {}  # the world of the running example: the hypothesis examples share the monkeypatch of the test


def install(world: World, cheshire_cat, monkeypatch):
    CURRENT["world"] = world
    if getattr(monkeypatch, "_lifecycle_installed", False):
        return
    monkeypatch._lifecycle_installed = True
    manager_class = type(cheshire_cat.plugin_manager)
    original_hook = manager_class.execute_hook

    async def execute_hook(self, name, *args, caller):
        if name == "rabbithole_ingestion_phases":
            return list(args[0]) + [GraphPhase]
        if name == "after_rabbithole_stored_documents":
            CURRENT["world"].stored_hook.append((args[0], len(args[1])))
        return await original_hook(self, name, *args, caller=caller)

    real_parse = RabbitHole._parse_to_docs
    real_embed = RabbitHole._embed_documents

    async def parse(self, source, file_bytes, content_type):
        fault = CURRENT["world"].faults.get(source)
        if fault == "parse":
            raise ValueError("unreadable file")
        CURRENT["world"].parsed.append(source)
        parsed = await real_parse(self, source, file_bytes, content_type)
        if fault == "row_gone":  # e.g. the chat of the source deleted while it is processed
            await crud_ingestions.delete_status(self.cat.agent_key, None, source)
        return parsed

    async def embed(self, docs, source, *args):
        await asyncio.sleep(0)  # the crash lands here
        if CURRENT["world"].faults.get(source) == "embed":
            raise ConnectionError("embedder down")
        return await real_embed(self, docs, source, *args)

    from cat.rabbit_hole import phases as phases_module
    real_settings = phases_module.service_settings

    async def service_settings(key_id, category):
        if category == "embedder":
            return {"name": CURRENT["world"].embedder}
        return await real_settings(key_id, category)

    # a crashed worker does not release its lock: it expires (short timeout, the heartbeat renews the live ones)
    monkeypatch.setenv("CAT_INGESTION_HEARTBEAT_SECONDS", "1")
    monkeypatch.setenv("CAT_INGESTION_LOCK_TIMEOUT_SECONDS", "2")
    monkeypatch.setattr(manager_class, "execute_hook", execute_hook)
    monkeypatch.setattr(RabbitHole, "_parse_to_docs", parse)
    monkeypatch.setattr(RabbitHole, "_embed_documents", embed)
    monkeypatch.setattr(phases_module, "service_settings", service_settings)

    async def initialize(self, embedder_name, embedder_size):
        # a change of the embedder recreates the collections: every point is gone
        await self.delete_tenant_points(str(VectorMemoryType.DECLARATIVE))

    monkeypatch.setattr(type(cheshire_cat.vector_memory_handler), "initialize", initialize)
    # every CheshireCat of the agent (the resume loads its own) keeps its files
    local = LocalFileManager()
    monkeypatch.setattr(type(cheshire_cat), "file_manager", property(lambda self: local, lambda self, value: None))


operation = st.one_of(
    st.tuples(st.just("ingest"), st.sampled_from(SOURCES), st.sampled_from(FAULTS)),
    st.tuples(st.just("crash"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("crash_restart"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("resume"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("embedder"), st.none(), st.none()),
    st.tuples(st.just("delete"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("delete_fails"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("row_gone"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("ingest_and_delete"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("ingest_both"), st.none(), st.none()),
    st.tuples(st.just("reingest"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("dismiss"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("retry"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("fix_graph"), st.none(), st.none()),
    st.tuples(st.just("legacy"), st.sampled_from(SOURCES), st.none()),
    st.tuples(st.just("delete_during_embedder"), st.sampled_from(SOURCES), st.none()),
)


async def wipe(cheshire_cat):
    """A new world for every example: no state, no points, no files, no lock left by a crashed worker."""
    from cat.db.database import get_async_db

    locks = [k async for k in get_async_db().scan_iter("lock:ingestion-source*")]
    if locks:
        await get_async_db().delete(*locks)
    await crud_ingestions.destroy_all(cheshire_cat.agent_key)
    collection = str(VectorMemoryType.DECLARATIVE)
    if await cheshire_cat.vector_memory_handler.check_collection_existence(collection):
        await cheshire_cat.vector_memory_handler.delete_tenant_points(collection)
    LocalFileManager().remove_folder(cheshire_cat.agent_key)


async def prepared(cheshire_cat):
    rabbit_hole = copy.copy(cheshire_cat.rabbit_hole)
    await rabbit_hole.setup(cheshire_cat)
    return rabbit_hole


def restarted(expected):
    """The model after a resume (or a change of the embedder, which reruns every source): the interrupted ingestions
    are completed (the crash clears the faults of its source), the interrupted deletions too."""
    for source, state in list(expected.items()):
        if state == "crashed":
            expected[source] = "completed"
        elif state == "deleting":
            expected.pop(source)


async def restart(lizard, cheshire_cat, expected):
    """A pod starts again: the automatic resume passes on the agent, after the locks of the crashed workers expired."""
    if "crashed" in expected.values():
        await asyncio.sleep(engine.lock_timeout_seconds() + 0.2)
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)


@settings(max_examples=15, deadline=None, database=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@example(operations=[("ingest", "alice.txt", "parse"), ("dismiss", "alice.txt", None)])
@example(operations=[("ingest", "alice.txt", "plugin_error"), ("retry", "alice.txt", None)])
@example(operations=[("ingest", "alice.txt", "plugin_error"), ("ingest", "bob.txt", "parse"), ("fix_graph", None, None)])
@example(operations=[("legacy", "bob.txt", None), ("delete_during_embedder", "bob.txt", None)])
@example(operations=[("legacy", "bob.txt", None), ("embedder", None, None)])
@given(operations=st.lists(operation, min_size=1, max_size=6))
async def test_the_lifecycle_of_the_sources(lizard, cheshire_cat, monkeypatch, operations, secure_client, secure_client_headers):
    world = World()
    install(world, cheshire_cat, monkeypatch)
    await wipe(cheshire_cat)
    agent = cheshire_cat.agent_key
    expected = {}  # source -> "completed" | "error" | "crashed" | "deleting"

    async def upload(name):
        return await cheshire_cat.rabbit_hole.ingest_file(
            cat=cheshire_cat, file=BytesIO(TEXT), metadata={}, filename=name, content_type="text/plain",
        )

    async def call(method, path, name):
        # the client of the tests, with the API key it sets
        return await secure_client.request(
            method, path, params={"source": name}, headers=secure_client_headers | {"X-Agent-ID": agent},
        )

    async def old_source(name):
        from langchain_core.documents import Document

        cheshire_cat.file_manager.write_file(TEXT, name, agent)
        rabbit_hole = await prepared(cheshire_cat)
        await rabbit_hole.store_documents([Document(page_content="old knowledge")], name, metadata={"author": "alice"})

    async def dismiss(name):
        return await call("DELETE", "/ingestion/status", name)

    async def crash_source(name):
        """An upload whose worker dies during the embedding: its lock is never released (it expires)."""
        world.faults[name] = None
        task = asyncio.create_task(upload(name))
        for _ in range(500):  # until the embedding starts (the upload may wait for the lock of an earlier crash)
            row = await crud_ingestions.get_status(agent, None, name)
            if row and row.get("phase") == "embedding":
                break
            await asyncio.sleep(0.01)
        with patch.object(Lock, "release", side_effect=LockError("the worker is dead")):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        row = await crud_ingestions.get_status(agent, None, name)
        assert row["status"] == "processing" and row["owner"] is not None

    for kind, name, fault in operations:
        hooks_before = len(world.stored_hook)
        if kind == "ingest":
            world.faults[name] = fault
            world.not_ready_left[name] = 2
            data = b"" if fault == "empty" else TEXT
            row = await cheshire_cat.rabbit_hole.ingest_file(
                cat=cheshire_cat, file=BytesIO(data), metadata={}, filename=name, content_type="text/plain",
            )
            expected[name] = "completed" if fault in (None, "plugin_not_ready") else "error"
            assert row["status"] == expected[name], (kind, name, fault, row)
            assert [h[0] for h in world.stored_hook[hooks_before:]] == [name]
        elif kind in ("crash", "crash_restart"):
            await crash_source(name)
            expected[name] = "crashed"
            if kind == "crash_restart":  # the pod starts again: its resume completes every interrupted operation
                await restart(lizard, cheshire_cat, expected)
                restarted(expected)
        elif kind == "resume":
            await restart(lizard, cheshire_cat, expected)
            restarted(expected)
            assert len(world.stored_hook) - hooks_before <= len(SOURCES)
        elif kind == "embedder":
            # a change of the embedder (the collections are recreated) while an upload of a source runs
            world.embedder = "embedder-2" if world.embedder == "embedder-1" else "embedder-1"
            uploaded = SOURCES[0]
            world.faults[uploaded] = None
            parsed_before = len(world.parsed)
            legacy = set(world.legacy)
            await asyncio.gather(lizard.embed_all_in_cheshire_cats(), upload(uploaded))
            restarted(expected)
            expected[uploaded] = "completed"
            expected.update({source: "completed" for source in legacy})  # the old sources got their state
            world.legacy.clear()
            parsed = world.parsed[parsed_before:]
            assert set(parsed) <= {uploaded} | legacy  # the sources with a state reuse their chunks
            assert all(parsed.count(source) == 1 for source in legacy - {uploaded})
        elif kind == "legacy":
            # a source ingested before the states existed
            if name in expected or name in world.legacy:
                continue
            world.faults[name] = None
            world.not_ready_left[name] = 0
            await old_source(name)
            world.legacy.add(name)
        elif kind == "delete_during_embedder":
            # an old source deleted while the embedder changes: it stays deleted
            if name not in world.legacy:
                continue
            world.embedder = "embedder-2" if world.embedder == "embedder-1" else "embedder-1"
            legacy = set(world.legacy) - {name}

            handler_class = type(cheshire_cat.vector_memory_handler)
            recreate = handler_class.initialize

            async def deleted_meanwhile(self, *args):
                # the source is deleted while the collections are recreated: after its adoption, before its re-run
                await cheshire_cat.rabbit_hole.delete_source(cheshire_cat, name)
                return await recreate(self, *args)

            with patch.object(handler_class, "initialize", deleted_meanwhile):
                await lizard.embed_all_in_cheshire_cats()
            restarted(expected)
            expected.update({source: "completed" for source in legacy})
            world.legacy.clear()
        elif kind == "delete":
            await cheshire_cat.rabbit_hole.delete_source(cheshire_cat, name)
            expected.pop(name, None)
            world.legacy.discard(name)
        elif kind == "delete_fails":
            if name not in expected:
                continue
            world.delete_fails[name] = True
            with pytest.raises(ConnectionError):
                await cheshire_cat.rabbit_hole.delete_source(cheshire_cat, name)
            expected[name] = "deleting"
        elif kind == "row_gone":
            world.faults[name] = "row_gone"
            assert await upload(name) is None
            world.faults[name] = None
            expected.pop(name, None)
        elif kind == "ingest_and_delete":
            # two actors on the same source: one at a time, in either order
            world.faults[name] = None
            await asyncio.gather(upload(name), cheshire_cat.rabbit_hole.delete_source(cheshire_cat, name))
            if await crud_ingestions.get_status(agent, None, name):
                expected[name] = "completed"
            else:
                expected.pop(name, None)
        elif kind == "reingest":
            # maintenance: the source again from scratch (waits for a crashed worker's lock to expire)
            if name not in expected:
                continue
            if expected[name] == "error" and world.faults.get(name) == "empty":
                with pytest.raises(SourceNotKept):  # an empty upload kept no bytes: nothing is touched
                    await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, name)
                continue
            row = await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, name)
            if expected[name] == "deleting":
                assert row is None
                expected.pop(name)
            else:
                ok = expected[name] == "crashed" or world.faults.get(name) in (None, "plugin_not_ready")
                expected[name] = "completed" if ok else "error"
                assert row["status"] == expected[name]
        elif kind == "dismiss":
            # the client dismisses a failed ingestion: only a failed one goes, with the files kept for it
            if name not in expected:
                continue
            response = await dismiss(name)
            if expected[name] == "error":
                assert response.status_code == 200
                expected.pop(name)
                if await cheshire_cat.vector_memory_handler.get_all_tenant_points(
                    str(VectorMemoryType.DECLARATIVE), limit=1, metadata={"source": name}, with_vectors=False,
                ) != ([], None):
                    world.legacy.add(name)  # its memory stays: a source without a state
            else:
                assert response.status_code == 409
        elif kind == "retry":
            # the client retries a failed ingestion, twice at once (a double click), after the cause was fixed
            failed_in_phases = expected.get(name) == "error" and world.faults.get(name) not in ("empty",)
            if failed_in_phases:
                world.faults[name] = None
            real_rerun = RabbitHole.rerun_source

            async def started_later(self, *args, **kwargs):
                await asyncio.sleep(0.3)  # the background run starts after the response, as in production
                return await real_rerun(self, *args, **kwargs)

            with patch.object(RabbitHole, "rerun_source", started_later):
                responses = await asyncio.gather(
                    call("POST", "/ingestion/retry", name), call("POST", "/ingestion/retry", name),
                )
            codes = sorted(r.status_code for r in responses)
            if name not in expected:
                assert codes == [404, 404]
            elif failed_in_phases:
                assert codes == [200, 409]  # one retry runs, the other is refused
                expected[name] = "completed"
            else:
                assert codes == [409, 409]
        elif kind == "fix_graph":
            # the settings of the graph are fixed: the resume runs again the sources whose graph failed (and only
            # those among the failed ones: a failure of another phase is not retried by itself)
            failed_in_graph = {
                s for s, state in expected.items() if state == "error" and world.faults.get(s) == "plugin_error"
            }
            failed_elsewhere = {
                s: await crud_ingestions.get_status(agent, None, s)
                for s, state in expected.items() if state == "error" and s not in failed_in_graph
            }
            world.graph_settings += 1
            for source, fault in list(world.faults.items()):
                if fault == "plugin_error":
                    world.faults[source] = None
            await restart(lizard, cheshire_cat, expected)
            restarted(expected)
            for source in failed_in_graph:
                expected[source] = "completed"
            for source, row in failed_elsewhere.items():  # a failure elsewhere is left untouched
                assert await crud_ingestions.get_status(agent, None, source) == row
        elif kind == "ingest_both":
            for source in SOURCES:
                world.faults[source] = None
            rows = await asyncio.gather(*(upload(source) for source in SOURCES))
            assert [row["status"] for row in rows] == ["completed"] * len(SOURCES)
            expected.update({source: "completed" for source in SOURCES})

        world.legacy -= set(expected)  # a source with a row has its own state again
        await check_rules(cheshire_cat, world, expected)


async def check_rules(cheshire_cat, world, expected):
    agent = cheshire_cat.agent_key
    rows = {r["source"]: r for r in await crud_ingestions.list_statuses(agent)}
    assert set(rows) == set(expected)
    collection = str(VectorMemoryType.DECLARATIVE)
    points = []
    if await cheshire_cat.vector_memory_handler.check_collection_existence(collection):
        points, _ = await cheshire_cat.vector_memory_handler.get_all_tenant_points(collection, with_vectors=True)
    by_source = {}
    for point in points:
        assert point.vector, "a point without its vector"
        by_source.setdefault(point.payload["metadata"]["source"], []).append(point)
    for name in SOURCES:
        whens = {p.payload["metadata"]["when"] for p in by_source.get(name, [])}
        assert len(whens) <= 1, "points of two ingestions of the same source"

    rabbit_hole = await prepared(cheshire_cat)
    ordered = await engine.collect_phases(rabbit_hole, cheshire_cat)
    tags = await compute_tags(ordered, cheshire_cat)
    for name, state in expected.items():
        row = rows[name]
        if state == "completed":
            assert row["status"] == "completed" and row["owner"] is None and row["error"] is None
            diary = row["completed_phases"]
            assert {pid: e["state"] for pid, e in diary.items()} == {pid: "complete" for pid in tags}
            assert {pid: e["tag"] for pid, e in diary.items()} == tags
            assert len(by_source.get(name, [])) == diary["embedding"]["processed"] > 0
            docs, _ = chunks.read_chunks(cheshire_cat.file_manager, agent, None, name)
            assert docs
        elif state == "error":
            assert row["status"] == "error" and row["error"] and row["owner"] is None
        elif state == "deleting":
            assert row["status"] == "deleting"
        else:
            assert row["status"] == "processing"
    for name in set(SOURCES) - set(expected) - world.legacy:
        assert name not in by_source
        assert not os.path.exists(os.path.join(
            cheshire_cat.file_manager._root_dir, agent, chunks.FOLDER, chunks.source_id(name) + ".json",
        ))
