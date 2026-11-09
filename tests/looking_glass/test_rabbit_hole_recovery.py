"""Bringing the sources up to date: ``RabbitHole.rerun_source``, the automatic resume (``cat.rabbit_hole.recovery``) and the
re-embedding on a change of the embedder (``BillTheLizard.embed_all_in_cheshire_cats``), and the maintenance of an agent
while its vector database changes (``CheshireCat.transfer_vector_points_from``)."""
import asyncio
from io import BytesIO

import pytest

from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
from cat.db import crud
from cat.db.cruds import conversations as crud_conversations
from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds import users as crud_users
from cat.looking_glass import CheshireCat
from cat.rabbit_hole import RabbitHole, chunks, engine, recovery
from cat.rabbit_hole.phases import IngestionSource
from cat.services.memory.models import VectorMemoryType

TEXT = ("Curiouser and curiouser! cried Alice, she was so much surprised. " * 30).encode()


@pytest.fixture
def files(monkeypatch):
    """Every CheshireCat of the test keeps its files (the default file manager of the tests keeps none)."""
    local = LocalFileManager()
    monkeypatch.setattr(CheshireCat, "file_manager", property(lambda self: local, lambda self, value: None))
    return local


@pytest.fixture
def embedder_name(monkeypatch):
    """The name of the embedder in the material settings: changing it is a change of the embedder."""
    from cat.rabbit_hole import phases

    current = {"name": "embedder-1"}
    real = phases.service_settings

    async def service_settings(key_id, category):
        return {"name": current["name"]} if category == "embedder" else await real(key_id, category)

    monkeypatch.setattr(phases, "service_settings", service_settings)
    return current


@pytest.fixture
def parses(monkeypatch):
    calls = []
    real = RabbitHole._parse_to_docs

    async def counting(self, source, file_bytes, content_type):
        calls.append(source)
        return await real(self, source, file_bytes, content_type)

    monkeypatch.setattr(RabbitHole, "_parse_to_docs", counting)
    return calls


async def ingest(cat, name="story.txt", data=TEXT):
    return await cat.rabbit_hole.ingest_file(
        cat=cat, file=BytesIO(data), metadata={}, filename=name, content_type="text/plain",
    )


async def points_of(ccat, name, collection=VectorMemoryType.DECLARATIVE):
    points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
        str(collection), metadata={"source": name}, with_vectors=False,
    )
    return points


async def chat_of_a_user(ccat):
    """A chat owned by a stored user, as a client opening it does."""
    user = await crud_users.create_user(ccat.agent_key, {"username": "alice", "password": "alice-password"})
    await crud_conversations.claim_conversation(ccat.agent_key, "chat-1", user["id"])
    return await ccat._find_stray_cat("chat-1")


async def crash(ccat, name):
    """A worker that died in the middle of the source: the row in flight, its lock expired."""
    await crud_ingestions.update_status(ccat.agent_key, None, name, status="processing", owner=engine.OWNER)
    await crud_ingestions.reset_phase_entries(ccat.agent_key, None, name, ["embedding"])


# --- rerun_source ------------------------------------------------------------------------------------------------

async def test_a_source_held_by_a_live_worker_is_left_alone(cheshire_cat, files):
    await ingest(cheshire_cat)
    await crash(cheshire_cat, "story.txt")

    async with engine.source_lock(IngestionSource(cheshire_cat.agent_key, None, "story.txt")):
        assert await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt", blocking=False) is None
        row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")

    assert row["status"] == "processing" and "embedding" not in row["completed_phases"]


async def test_a_crashed_ingestion_is_completed(cheshire_cat, files, parses):
    await ingest(cheshire_cat)
    await crash(cheshire_cat, "story.txt")

    row = await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt", blocking=False)

    assert row["status"] == "completed" and parses == ["story.txt"]  # the chunks are reused
    assert len(await points_of(cheshire_cat, "story.txt")) == row["completed_phases"]["embedding"]["processed"]


async def test_an_interrupted_deletion_is_completed(cheshire_cat, files):
    await ingest(cheshire_cat)
    await crud_ingestions.update_status(cheshire_cat.agent_key, None, "story.txt", status="deleting")

    assert await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt") is None

    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") is None
    assert await points_of(cheshire_cat, "story.txt") == []


async def test_a_url_interrupted_while_downloading_is_downloaded_again(cheshire_cat, files, monkeypatch):
    downloads = []

    async def resolve(self, file, filename, content_type):
        downloads.append(file)
        return file, TEXT, "text/plain", True

    monkeypatch.setattr(RabbitHole, "_resolve_source_bytes", resolve)
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "https://x.org/page", type_="url")
    await crud_ingestions.update_status(cheshire_cat.agent_key, None, "https://x.org/page", status="downloading")

    row = await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "https://x.org/page", blocking=False)

    assert row["status"] == "completed" and downloads == ["https://x.org/page"]
    assert chunks.read_source_bytes(files, cheshire_cat.agent_key, None, "https://x.org/page") == TEXT


async def test_a_url_that_cannot_be_downloaded_again_fails(cheshire_cat, files, monkeypatch):
    async def unreachable(self, file, filename, content_type):
        return None, None, None, True

    monkeypatch.setattr(RabbitHole, "_resolve_source_bytes", unreachable)
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "https://x.org/page", type_="url")

    row = await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "https://x.org/page")

    assert row["status"] == "error" and "Cannot download" in row["error"]


async def test_the_states_of_a_download_belong_to_the_upload_only(cheshire_cat, files, monkeypatch):
    seen = []
    real_update = crud_ingestions.update_status

    async def update_status(agent_id, chat_id, source, **fields):
        seen.append(fields.get("status"))
        return await real_update(agent_id, chat_id, source, **fields)

    class Response:
        headers = {"Content-Type": "text/plain"}
        content = TEXT

        def raise_for_status(self):
            pass

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            return Response()

    monkeypatch.setattr("cat.rabbit_hole.core.AsyncClient", Client)
    monkeypatch.setattr(crud_ingestions, "update_status", update_status)

    await cheshire_cat.rabbit_hole.ingest_file(cat=cheshire_cat, file="https://x.org/page", metadata={})
    assert "downloading" in seen and "downloaded" in seen

    seen.clear()
    chunks.delete_all(files, cheshire_cat.agent_key, None, "https://x.org/page")  # nothing kept: downloaded again
    await crud_ingestions.reset_phase_entries(cheshire_cat.agent_key, None, "https://x.org/page", ["parsing_chunking"])
    row = await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "https://x.org/page")

    assert row["status"] == "completed"
    assert "downloading" not in seen and "downloaded" not in seen


# --- the automatic resume ----------------------------------------------------------------------------------------

async def test_the_resume_brings_every_source_up_to_date_and_leaves_the_others(
    lizard, cheshire_cat, files, embedder_name, parses,
):
    for name in ("crashed.txt", "fresh.txt", "failed.txt", "stale.txt"):
        await ingest(cheshire_cat, name)
    await crash(cheshire_cat, "crashed.txt")
    await crud_ingestions.update_status(cheshire_cat.agent_key, None, "failed.txt", status="error", error="boom")
    await crud_ingestions.record_phase(cheshire_cat.agent_key, None, "stale.txt", "embedding", {"state": "complete", "tag": "old"})
    fresh_before = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "fresh.txt")
    parses.clear()

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    rows = {r["source"]: r for r in await crud_ingestions.list_statuses(cheshire_cat.agent_key)}
    assert rows["crashed.txt"]["status"] == "completed"
    assert rows["stale.txt"]["completed_phases"]["embedding"]["tag"] != "old"
    assert rows["fresh.txt"] == fresh_before  # nothing written
    assert rows["failed.txt"]["status"] == "error"
    assert parses == []  # only embeddings ran


async def test_a_change_of_the_embedder_is_caught_by_the_resume(lizard, cheshire_cat, files, embedder_name, parses):
    await ingest(cheshire_cat)
    parses.clear()
    embedder_name["name"] = "embedder-2"

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert row["status"] == "completed" and parses == []


async def test_an_agent_in_maintenance_or_held_by_another_pod_is_skipped(lizard, cheshire_cat, files):
    await ingest(cheshire_cat)
    await crash(cheshire_cat, "story.txt")

    await crud_ingestions.set_maintenance(cheshire_cat.agent_key, ttl=60)
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)
    assert (await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt"))["status"] == "processing"

    await crud_ingestions.clear_maintenance(cheshire_cat.agent_key)
    from cat.db import crud
    async with crud.distributed_lock(f"ingestion-resume:{cheshire_cat.agent_key}", timeout=10):
        await recovery.resume_agent(lizard, cheshire_cat.agent_key)
    assert (await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt"))["status"] == "processing"


async def test_the_resume_of_an_agent_without_sources_or_gone_does_nothing(lizard, cheshire_cat):
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)
    await crud_ingestions.create_status("ghost-agent", None, "x.txt", type_="file")
    await recovery.resume_agent(lizard, "ghost-agent")
    assert (await crud_ingestions.get_status("ghost-agent", None, "x.txt"))["status"] == "uploaded"


async def test_a_failing_agent_does_not_stop_the_resume_of_the_others(lizard, monkeypatch):
    agents = []

    async def resume_agent(lizard_, agent_id):
        agents.append(agent_id)
        if agent_id == "a1":
            raise ConnectionError("redis blip")

    async def agents_main_keys():
        return ["a1", "a2"]

    monkeypatch.setattr(recovery, "resume_agent", resume_agent)
    monkeypatch.setattr(recovery.crud_settings, "get_agents_main_keys", agents_main_keys)

    await recovery.resume_all(lizard)

    assert agents == ["a1", "a2"]


@pytest.mark.parametrize("interval", ["-1", "0"])
async def test_the_resume_disabled_runs_no_pass(lizard, monkeypatch, interval):
    calls = []

    async def resume_all(lizard_):
        calls.append(1)

    monkeypatch.setenv("CAT_INGESTION_RESUME_INTERVAL_SECONDS", interval)
    monkeypatch.setattr(recovery, "resume_all", resume_all)

    await recovery.run(lizard)

    assert calls == []


async def test_the_periodic_resume_survives_a_failed_pass_and_stops_with_the_pod(lizard, monkeypatch):
    calls = []

    async def resume_all(lizard_):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("redis blip")

    monkeypatch.setenv("CAT_INGESTION_RESUME_INTERVAL_SECONDS", "1")
    monkeypatch.setattr(recovery, "resume_all", resume_all)

    task = asyncio.create_task(recovery.run(lizard))
    await asyncio.sleep(1.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) == 2


async def test_the_pod_starts_its_resume_and_stops_it(lizard):
    started = lizard._ingestion_resume  # created by the bootstrap of the app (disabled in the tests: already done)
    assert started is not None and started.done()

    pending = asyncio.create_task(asyncio.sleep(3600))
    lizard._ingestion_resume = pending
    await lizard._stop_ingestion_resume()

    assert pending.cancelled() and lizard._ingestion_resume is None
    await lizard._stop_ingestion_resume()  # nothing to stop


# --- the change of the embedder ----------------------------------------------------------------------------------

async def test_a_change_of_the_embedder_reembeds_the_stored_chunks_and_reingests_the_old_sources(
    lizard, cheshire_cat, files, embedder_name, parses,
):
    await ingest(cheshire_cat, "stateful.txt")
    # a source ingested before the states existed: points and file, no row
    files.write_file(TEXT, "legacy.txt", cheshire_cat.agent_key)
    rabbit_hole = __import__("copy").copy(cheshire_cat.rabbit_hole)
    await rabbit_hole.setup(cheshire_cat)
    from langchain_core.documents import Document
    await rabbit_hole.store_documents([Document(page_content="old knowledge")], "legacy.txt")
    parses.clear()
    embedder_name["name"] = "embedder-2"

    await lizard.embed_all_in_cheshire_cats()

    assert parses == ["legacy.txt"]  # the stateful source reuses its chunks
    stateful = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "stateful.txt")
    legacy = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "legacy.txt")
    assert stateful["status"] == "completed" and legacy["status"] == "completed"
    # the clearing of the collection for the old sources did not remove the re-embedded ones
    assert len(await points_of(cheshire_cat, "stateful.txt")) == stateful["completed_phases"]["embedding"]["processed"]
    assert await points_of(cheshire_cat, "legacy.txt")




async def test_a_chat_source_is_rerun_in_its_chat_with_its_owner(cheshire_cat, files, monkeypatch):
    stray = await chat_of_a_user(cheshire_cat)
    await ingest(stray)
    callers = []
    original = type(cheshire_cat.plugin_manager).execute_hook

    async def execute_hook(self, name, *args, caller):
        if name == "after_rabbithole_stored_documents":
            callers.append(getattr(caller, "id", None))
        return await original(self, name, *args, caller=caller)

    monkeypatch.setattr(type(cheshire_cat.plugin_manager), "execute_hook", execute_hook)

    row = await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt", "chat-1", invalidate=["embedding"])

    assert row["status"] == "completed" and callers == ["chat-1"]
    assert await points_of(cheshire_cat, "story.txt", VectorMemoryType.EPISODIC)


async def test_invalidated_phases_run_again_with_their_dependents(cheshire_cat, files, parses):
    await ingest(cheshire_cat)
    parses.clear()

    row = await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt", invalidate=["parsing_chunking"])

    assert parses == ["story.txt"] and row["status"] == "completed"
    assert row["completed_phases"]["embedding"]["finished_at"] >= row["completed_phases"]["parsing_chunking"]["finished_at"]


async def test_a_failing_source_does_not_stop_the_resume_of_the_others(lizard, cheshire_cat, files, monkeypatch):
    for name in ("broken.txt", "crashed.txt"):
        await ingest(cheshire_cat, name)
        await crash(cheshire_cat, name)
    real = RabbitHole.rerun_source

    async def rerun_source(self, cat, name, chat_id=None, blocking=True, invalidate=None):
        if name == "broken.txt":
            raise ConnectionError("graph store down")
        return await real(self, cat, name, chat_id, blocking, invalidate)

    monkeypatch.setattr(RabbitHole, "rerun_source", rerun_source)

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    assert (await crud_ingestions.get_status(cheshire_cat.agent_key, None, "crashed.txt"))["status"] == "completed"


async def test_the_resume_stops_when_the_agent_goes_in_maintenance(lizard, cheshire_cat, files, monkeypatch):
    for name in ("first.txt", "second.txt"):
        await ingest(cheshire_cat, name)
        await crash(cheshire_cat, name)
    real = RabbitHole.rerun_source

    async def rerun_then_maintenance(self, cat, name, chat_id=None, blocking=True, invalidate=None):
        row = await real(self, cat, name, chat_id, blocking, invalidate)
        await crud_ingestions.set_maintenance(cat.agent_key, ttl=60)  # e.g. a change of the embedder starts
        return row

    monkeypatch.setattr(RabbitHole, "rerun_source", rerun_then_maintenance)

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    statuses = sorted(r["status"] for r in await crud_ingestions.list_statuses(cheshire_cat.agent_key))
    assert statuses == ["completed", "processing"]


async def test_a_change_of_the_embedder_redoes_the_embeddings_lost_with_the_old_collections(
    lizard, cheshire_cat, files, embedder_name, monkeypatch,
):
    await ingest(cheshire_cat)
    embedder_name["name"] = "embedder-2"
    # the resume re-embeds with the new embedder before the collections are recreated: the diary looks up to date
    await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt")
    handler_class = type(cheshire_cat.vector_memory_handler)
    resumes = []

    async def initialize(self, embedder_name_, embedder_size):
        for collection in (VectorMemoryType.DECLARATIVE, VectorMemoryType.EPISODIC):
            await self.delete_tenant_points(str(collection))  # the collections are recreated: their points are gone
        resumes.append(await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key))

    monkeypatch.setattr(handler_class, "initialize", initialize)

    await lizard.embed_all_in_cheshire_cats()

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert row["status"] == "completed"
    assert len(await points_of(cheshire_cat, "story.txt")) == row["completed_phases"]["embedding"]["processed"]
    assert resumes == [True]  # the agent was in maintenance meanwhile
    assert not await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)


async def test_reingesting_old_sources_keeps_the_points_of_the_others(cheshire_cat, files):
    await ingest(cheshire_cat, "stateful.txt")
    files.write_file(TEXT, "legacy.txt", cheshire_cat.agent_key)
    from cat.looking_glass.models import StoredSourceWithMetadata

    await cheshire_cat.embed_stored_sources(
        VectorMemoryType.DECLARATIVE,
        [StoredSourceWithMetadata(name="legacy.txt", path=cheshire_cat.agent_key, content=BytesIO(TEXT), metadata={})],
    )

    assert await points_of(cheshire_cat, "stateful.txt")
    assert await points_of(cheshire_cat, "legacy.txt")


async def test_stopping_the_pod_during_a_resume_is_not_swallowed(lizard, cheshire_cat, files, monkeypatch):
    await ingest(cheshire_cat)
    await crash(cheshire_cat, "story.txt")

    async def stopped(self, *args, **kwargs):
        raise asyncio.CancelledError()  # the pod stops while the source is resumed

    monkeypatch.setenv("CAT_INGESTION_RESUME_INTERVAL_SECONDS", "60")
    monkeypatch.setattr(RabbitHole, "rerun_source", stopped)

    with pytest.raises(asyncio.CancelledError):
        await recovery.run(lizard)


async def test_a_failed_change_of_the_embedder_leaves_no_agent_in_maintenance(lizard, cheshire_cat, monkeypatch):
    outcomes = []
    original = type(lizard.plugin_manager).execute_hook

    async def initialize(self, embedder_name_, embedder_size):
        raise ConnectionError("vector db down")

    async def execute_hook(self, name, *args, caller):
        if name == "after_all_cheshire_cats_embedded":
            outcomes.append(args[0])
        return await original(self, name, *args, caller=caller)

    monkeypatch.setattr(type(cheshire_cat.vector_memory_handler), "initialize", initialize)
    monkeypatch.setattr(type(lizard.plugin_manager), "execute_hook", execute_hook)

    await lizard.embed_all_in_cheshire_cats()

    assert outcomes == [False]
    assert not await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)


# --- the resume and the failed sources -------------------------------------------------------------------------

class SettingsPhase:
    """A plugin phase whose settings and outcomes the test controls."""

    settings_value = 1
    outcomes: list = []
    runs = 0


def plug_settings_phase(monkeypatch, cheshire_cat):
    from cat.rabbit_hole.phases import IngestionPhase, PhaseResult

    class Graph(IngestionPhase):
        id = "graph"
        depends_on = ("embedding",)

        async def settings(self, cat):
            return SettingsPhase.settings_value

        async def process(self, source, cat):
            SettingsPhase.runs += 1
            status = SettingsPhase.outcomes.pop(0) if SettingsPhase.outcomes else "done"
            if status == "die":
                raise asyncio.CancelledError()  # the worker dies in the middle of the phase (e.g. a restart)
            return PhaseResult(status=status, reason="graph failed" if status == "error" else None)

        async def delete(self, source, cat):
            pass

    manager_class = type(cheshire_cat.plugin_manager)
    original = manager_class.execute_hook

    async def execute_hook(self, name, *args, caller):
        if name == "rabbithole_ingestion_phases":
            return list(args[0]) + [Graph]
        return await original(self, name, *args, caller=caller)

    monkeypatch.setattr(manager_class, "execute_hook", execute_hook)
    SettingsPhase.settings_value, SettingsPhase.outcomes, SettingsPhase.runs = 1, [], 0


async def test_a_failed_source_is_resumed_when_the_settings_of_its_failed_phase_change(
    lizard, cheshire_cat, files, monkeypatch, parses,
):
    plug_settings_phase(monkeypatch, cheshire_cat)
    SettingsPhase.outcomes = ["error"]
    failed = await ingest(cheshire_cat)
    assert failed["status"] == "error"

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)  # same settings: no retry, nothing written
    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") == failed

    SettingsPhase.settings_value = 2  # the settings of the graph fixed
    parses.clear()
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert row["status"] == "completed" and SettingsPhase.runs == 2 and parses == []


async def test_a_failed_source_is_resumed_when_the_settings_of_a_phase_it_depends_on_change(
    lizard, cheshire_cat, files, monkeypatch, embedder_name,
):
    plug_settings_phase(monkeypatch, cheshire_cat)
    SettingsPhase.outcomes = ["error"]
    await ingest(cheshire_cat)

    embedder_name["name"] = "embedder-2"  # the embedding, which the graph depends on, changes
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert row["status"] == "completed" and SettingsPhase.runs == 2


async def test_the_dependents_of_an_invalidated_phase_run_again_after_a_crash(
    lizard, cheshire_cat, files, monkeypatch,
):
    """``rerun_source`` invalidates the dependents too: a worker dying after the invalidated phase ran, before its
    dependents (e.g. a restart during a change of the embedder), leaves them to do and the resume runs them."""
    plug_settings_phase(monkeypatch, cheshire_cat)
    await ingest(cheshire_cat)
    SettingsPhase.outcomes = ["die"]

    with pytest.raises(asyncio.CancelledError):
        await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt", invalidate=["embedding"])
    assert SettingsPhase.runs == 2

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert row["status"] == "completed" and SettingsPhase.runs == 3, (row["status"], SettingsPhase.runs)


async def test_an_invalidation_with_an_invalid_graph_of_phases_records_the_error(cheshire_cat, files, monkeypatch):
    await ingest(cheshire_cat)
    from cat.rabbit_hole.phases import PhaseGraphError

    async def cycle(rabbit_hole, cat):
        raise PhaseGraphError("Dependency cycle: graph -> graph")

    monkeypatch.setattr(engine, "collect_phases", cycle)
    row = await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt", invalidate=["embedding"])

    assert row["status"] == "error" and "cycle" in row["error"]


async def test_a_source_that_failed_before_its_phases_is_not_resumed(lizard, cheshire_cat, files, embedder_name):
    failed = await ingest(cheshire_cat, data=b"")  # nothing kept: only a new upload can process it
    embedder_name["name"] = "embedder-2"

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") == failed


async def test_a_source_that_failed_upstream_is_left_alone_when_a_later_phase_changes(
    lizard, cheshire_cat, files, monkeypatch,
):
    plug_settings_phase(monkeypatch, cheshire_cat)

    async def unreadable(self, source, file_bytes, content_type):
        raise ValueError("unreadable file")

    monkeypatch.setattr(RabbitHole, "_parse_to_docs", unreadable)
    failed = await ingest(cheshire_cat)  # parsing failed: the embedding and the graph inherit the error
    assert failed["completed_phases"]["graph"]["state"] == "error"

    SettingsPhase.settings_value = 2  # only the graph changes: the failure is elsewhere
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") == failed


# --- sources without a state (ingested before the states existed) ----------------------------------------------

async def a_legacy_source(cheshire_cat, files, name="legacy.txt", chat=None, kept=True, data=TEXT):
    """A source ingested before the states existed: its points (with the custom metadata of its upload), its file."""
    from langchain_core.documents import Document

    rabbit_hole = __import__("copy").copy(cheshire_cat.rabbit_hole)
    await rabbit_hole.setup(chat or cheshire_cat)
    if kept and not name.startswith("http"):
        files.write_file(data, name, cheshire_cat.agent_key if chat is None else f"{cheshire_cat.agent_key}/{chat.id}")
    await rabbit_hole.store_documents([Document(page_content="old knowledge")], name, metadata={"author": "alice"})


async def test_a_change_of_the_embedder_gives_a_state_to_the_old_sources(
    lizard, cheshire_cat, files, embedder_name, monkeypatch,
):
    await a_legacy_source(cheshire_cat, files)
    await a_legacy_source(cheshire_cat, files, "https://x.org/page")

    async def download(self, file, filename, content_type):
        return file, TEXT, "text/html", True

    monkeypatch.setattr(RabbitHole, "_resolve_source_bytes", download)
    embedder_name["name"] = "embedder-2"

    await lizard.embed_all_in_cheshire_cats()

    for name in ("legacy.txt", "https://x.org/page"):
        row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, name)
        assert row["status"] == "completed" and row["metadata"] == {"author": "alice"}
        assert all(p.payload["metadata"]["author"] == "alice" for p in await points_of(cheshire_cat, name))
    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "legacy.txt")
    assert row["content_type"] == "text/plain"


async def test_an_old_source_deleted_during_a_change_of_the_embedder_stays_deleted(
    lizard, cheshire_cat, files, embedder_name, monkeypatch,
):
    await a_legacy_source(cheshire_cat, files)
    embedder_name["name"] = "embedder-2"
    handler_class = type(cheshire_cat.vector_memory_handler)

    async def initialize_while_deleted(self, embedder_name_, embedder_size):
        # the file is deleted while the collections are recreated (DELETE /file_manager/files/legacy.txt)
        files.remove_file(f"{cheshire_cat.agent_key}/legacy.txt")
        await cheshire_cat.rabbit_hole.delete_source(cheshire_cat, "legacy.txt")

    monkeypatch.setattr(handler_class, "initialize", initialize_while_deleted)

    await lizard.embed_all_in_cheshire_cats()

    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "legacy.txt") is None
    assert await points_of(cheshire_cat, "legacy.txt") == []


async def test_a_source_ingested_during_a_change_of_the_embedder_is_not_taken_for_an_old_one(
    lizard, cheshire_cat, files, embedder_name, monkeypatch,
):
    await ingest(cheshire_cat)  # its row exists: it has its own state
    row_before = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    adopted = []
    real_create = crud_ingestions.create_status

    async def create_status(agent_id, chat_id, source, **kwargs):
        adopted.append(source)
        return await real_create(agent_id, chat_id, source, **kwargs)

    monkeypatch.setattr(crud_ingestions, "create_status", create_status)
    embedder_name["name"] = "embedder-2"

    await lizard.embed_all_in_cheshire_cats()

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert adopted == [] and row["created_at"] == row_before["created_at"] and row["status"] == "completed"


async def test_an_old_source_whose_file_is_gone_is_not_brought_back(lizard, cheshire_cat, files, embedder_name):
    await a_legacy_source(cheshire_cat, files, kept=False)
    embedder_name["name"] = "embedder-2"

    await lizard.embed_all_in_cheshire_cats()

    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "legacy.txt") is None


async def test_an_old_chat_source_gets_its_state_in_its_chat(lizard, cheshire_cat, files, embedder_name):
    stray = await chat_of_a_user(cheshire_cat)
    await a_legacy_source(cheshire_cat, files, "notes.txt", chat=stray)
    embedder_name["name"] = "embedder-2"

    await lizard.embed_all_in_cheshire_cats()

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, "chat-1", "notes.txt")
    assert row["status"] == "completed" and row["metadata"] == {"author": "alice"}
    assert await points_of(cheshire_cat, "notes.txt", VectorMemoryType.EPISODIC)


async def test_reingesting_an_old_source_keeps_its_metadata_and_content_type(cheshire_cat, files):
    await a_legacy_source(cheshire_cat, files)

    row = await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, "legacy.txt")

    assert row["status"] == "completed" and row["metadata"] == {"author": "alice"}
    assert row["content_type"] == "text/plain"


async def test_an_adopted_source_keeps_the_metadata_of_its_upload_only(cheshire_cat, files):
    from langchain_core.documents import Document

    files.write_file(TEXT, "legacy.txt", cheshire_cat.agent_key)
    rabbit_hole = __import__("copy").copy(cheshire_cat.rabbit_hole)
    await rabbit_hole.setup(cheshire_cat)
    await rabbit_hole.store_documents(
        [Document(page_content="one", metadata={"page": 1, "chunk_index": 0}),
         Document(page_content="two", metadata={"page": 2, "chunk_index": 1}),
         Document(page_content="[Image]", metadata={"image_file": "legacy_img_0.png"})],
        "legacy.txt", metadata={"author": "alice"},
    )

    assert await cheshire_cat.rabbit_hole.adopt_source(cheshire_cat, "legacy.txt")

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "legacy.txt")
    assert row["metadata"] == {"author": "alice"}


@pytest.mark.parametrize("name, data, content_type", [
    ("table.csv", TEXT, "text/csv"), ("notes.md", TEXT, "text/markdown"), ("doc.txt", TEXT, "text/plain"),
    ("scan.bin", b"%PDF-1.4\n%binary", "application/pdf"),  # the bytes win when they tell a precise format
])
async def test_an_adopted_text_source_keeps_the_type_of_its_name(cheshire_cat, files, name, data, content_type):
    await a_legacy_source(cheshire_cat, files, name, data=data)

    assert await cheshire_cat.rabbit_hole.adopt_source(cheshire_cat, name)

    assert (await crud_ingestions.get_status(cheshire_cat.agent_key, None, name))["content_type"] == content_type


async def test_a_source_whose_upload_ends_after_the_listing_is_not_adopted(
    lizard, cheshire_cat, files, embedder_name, monkeypatch,
):
    await a_legacy_source(cheshire_cat, files)
    real_listing = type(lizard)._sources_without_state

    async def upload_ends_meanwhile(ccat):
        found = await real_listing(ccat)
        await ingest(cheshire_cat, "legacy.txt")  # its upload ends right after the listing: it has its own state now
        return found

    monkeypatch.setattr(type(lizard), "_sources_without_state", staticmethod(upload_ends_meanwhile))
    embedder_name["name"] = "embedder-2"
    adopted = []
    real_adopt = RabbitHole._adopt

    async def recording_adopt(self, source, require_points):
        result = await real_adopt(self, source, require_points)
        adopted.append(result)
        return result

    monkeypatch.setattr(RabbitHole, "_adopt", recording_adopt)

    await lizard.embed_all_in_cheshire_cats()

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "legacy.txt")
    assert adopted == [False] and row["status"] == "completed"


async def test_an_old_source_deleted_after_the_listing_is_not_adopted(
    lizard, cheshire_cat, files, embedder_name, monkeypatch,
):
    await a_legacy_source(cheshire_cat, files)
    real_listing = type(lizard)._sources_without_state

    async def deleted_meanwhile(ccat):
        found = await real_listing(ccat)
        await cheshire_cat.vector_memory_handler.delete_tenant_points(  # its points go, its file not yet
            str(VectorMemoryType.DECLARATIVE), {"source": "legacy.txt"},
        )
        return found

    monkeypatch.setattr(type(lizard), "_sources_without_state", staticmethod(deleted_meanwhile))
    embedder_name["name"] = "embedder-2"

    await lizard.embed_all_in_cheshire_cats()

    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "legacy.txt") is None


async def test_a_memory_without_a_source_is_not_taken_for_an_old_source(lizard, cheshire_cat, files):
    await a_legacy_source(cheshire_cat, files)
    point = (await cheshire_cat.vector_memory_handler.get_all_tenant_points(
        str(VectorMemoryType.DECLARATIVE), limit=1, with_vectors=True,
    ))[0][0]
    await cheshire_cat.vector_memory_handler.add_point_to_tenant(  # a memory stored without a source
        str(VectorMemoryType.DECLARATIVE), "loose", point.vector, {"when": 1.0},
    )

    assert await type(lizard)._sources_without_state(cheshire_cat) == [("legacy.txt", None)]


# --- the change of the vector database ---------------------------------------------------------------------------

COLLECTIONS = (str(VectorMemoryType.DECLARATIVE), str(VectorMemoryType.EPISODIC))


class PreviousVectorDatabase:
    """The vector database being left: it keeps the points it had when the switch started (``of``), read by the transfer
    collection by collection. ``gate`` holds every read until set (a long copy); ``failing`` is a collection whose read
    fails (the old database goes down halfway)."""

    def __init__(self, points: dict, gate: asyncio.Event | None = None, failing: str | None = None):
        self.points, self.gate, self.failing = points, gate, failing
        self.reads: list = []

    @classmethod
    async def of(cls, handler, **kwargs):
        points = {c: (await handler.get_all_tenant_points(c, with_vectors=True))[0] for c in COLLECTIONS}
        return cls(points, **kwargs)

    async def get_collection_names(self):
        return list(COLLECTIONS)

    async def get_all_tenant_points(self, collection_name, with_vectors=True):
        self.reads.append(collection_name)
        if self.gate is not None:
            await self.gate.wait()
        if collection_name == self.failing:
            raise ConnectionError("old vector db down")
        return list(self.points[collection_name]), None


def new_database_starts_empty(monkeypatch, cheshire_cat):
    """The new vector database has none of the points: its ``initialize`` finds the collections empty (the tests run on
    one database, so the points of the agent are removed there)."""

    async def initialize(self, embedder_name_, embedder_size):
        for collection in COLLECTIONS:
            await self.delete_tenant_points(collection)

    monkeypatch.setattr(type(cheshire_cat.vector_memory_handler), "initialize", initialize)


def record_transfer_hook(monkeypatch, cheshire_cat):
    """The outcomes of ``after_vector_memory_transfer_on_agent``, each with whether the agent was in maintenance."""
    outcomes = []
    manager_class = type(cheshire_cat.plugin_manager)
    original = manager_class.execute_hook

    async def execute_hook(self, name, *args, caller):
        if name == "after_vector_memory_transfer_on_agent":
            outcomes.append((args[0], await crud_ingestions.is_in_maintenance(caller.agent_key)))
        return await original(self, name, *args, caller=caller)

    monkeypatch.setattr(manager_class, "execute_hook", execute_hook)
    return outcomes


def record_reruns(monkeypatch):
    """The sources the transfer (or the resume) runs again, with the phases invalidated."""
    calls = []
    real = RabbitHole.rerun_source

    async def rerun_source(self, cat, name, chat_id=None, blocking=True, invalidate=None):
        calls.append((name, list(invalidate or [])))
        return await real(self, cat, name, chat_id, blocking, invalidate)

    monkeypatch.setattr(RabbitHole, "rerun_source", rerun_source)
    return calls


async def wait_until(predicate, seconds=5.0):
    async def poll():
        while not await predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), seconds)


async def make_stale(agent_id):
    """The heartbeat of the transfer stopped longer than a lock timeout ago (its pod died)."""
    record = await crud_ingestions.get_vector_transfer(agent_id)
    stale = record["beat"] - engine.lock_timeout_seconds() - 1
    await crud.store(crud_ingestions.vector_transfer_key(agent_id), {**record, "beat": stale})


async def is_up_to_date_in_the_new_database(cat, name):
    row = await crud_ingestions.get_status(cat.agent_key, None, name)
    embedding = row["completed_phases"].get("embedding")
    return row["status"] == "completed" and embedding is not None and (
        len(await points_of(cat, name)) == embedding["processed"] > 0
    )


async def test_the_resume_leaves_the_agent_alone_while_its_vector_database_changes(
    lizard, cheshire_cat, files, monkeypatch,
):
    await ingest(cheshire_cat)
    await ingest(cheshire_cat, "crashed.txt")
    await crash(cheshire_cat, "crashed.txt")
    plug_settings_phase(monkeypatch, cheshire_cat)  # the new database brings a phase: every tag changes
    outcomes = record_transfer_hook(monkeypatch, cheshire_cat)
    reruns = record_reruns(monkeypatch)
    previous = await PreviousVectorDatabase.of(cheshire_cat.vector_memory_handler, gate=asyncio.Event())
    new_database_starts_empty(monkeypatch, cheshire_cat)
    rows_before = await crud_ingestions.list_statuses(cheshire_cat.agent_key)
    transfer = asyncio.create_task(cheshire_cat.transfer_vector_points_from(previous))
    try:
        await wait_until(lambda: asyncio.sleep(0, bool(previous.reads)))  # the copy is running
        assert await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)

        await recovery.resume_agent(lizard, cheshire_cat.agent_key)

        assert await crud_ingestions.list_statuses(cheshire_cat.agent_key) == rows_before  # nothing written
        assert SettingsPhase.runs == 0 and outcomes == []
    finally:
        previous.gate.set()
        await asyncio.wait_for(transfer, 10)

    assert outcomes == [(True, True)]  # the plugins rebuild still in maintenance
    assert not await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is None
    # the source of the dead worker (in flight) is completed into the new database; the others keep the copied points
    assert reruns == [("crashed.txt", ["embedding"])]
    assert await is_up_to_date_in_the_new_database(cheshire_cat, "crashed.txt")
    assert len(await points_of(cheshire_cat, "story.txt")) == len(
        [p for p in previous.points[str(VectorMemoryType.DECLARATIVE)] if p.payload["metadata"]["source"] == "story.txt"]
    ) > 0


async def test_a_resume_pass_running_stops_at_the_next_source_when_the_vector_database_changes(
    lizard, cheshire_cat, files, monkeypatch,
):
    for name in ("first.txt", "second.txt"):
        await ingest(cheshire_cat, name)
        await crash(cheshire_cat, name)
    previous = await PreviousVectorDatabase.of(cheshire_cat.vector_memory_handler, gate=asyncio.Event())
    transfers = []
    real = RabbitHole.rerun_source

    async def rerun_then_transfer(self, cat, name, chat_id=None, blocking=True, invalidate=None):
        row = await real(self, cat, name, chat_id, blocking, invalidate)
        if not transfers:  # the vector database changes while the pass is on its first source
            transfers.append(asyncio.create_task(cat.transfer_vector_points_from(previous)))
            await wait_until(lambda: crud_ingestions.is_in_maintenance(cat.agent_key))
        return row

    monkeypatch.setattr(RabbitHole, "rerun_source", rerun_then_transfer)
    try:
        await recovery.resume_agent(lizard, cheshire_cat.agent_key)

        statuses = sorted(r["status"] for r in await crud_ingestions.list_statuses(cheshire_cat.agent_key))
        assert statuses == ["completed", "processing"]
    finally:
        previous.gate.set()
        await asyncio.wait_for(asyncio.gather(*transfers), 10)
    assert not await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)


async def test_a_source_ingested_during_the_copy_is_embedded_again_in_the_new_database(
    cheshire_cat, files, monkeypatch,
):
    """An upload started before the switch writes into the old database: once its worker ends, the transfer embeds it
    again in the new one. The sources not touched during the copy are not."""
    await ingest(cheshire_cat, "untouched.txt")
    await ingest(cheshire_cat, "uploading.txt")
    reruns = record_reruns(monkeypatch)
    previous = await PreviousVectorDatabase.of(cheshire_cat.vector_memory_handler)
    worker_done, worker_ready = asyncio.Event(), asyncio.Event()

    async def live_worker():
        # the worker of the upload holds its source beyond the copy, in the middle of its embedding: its heartbeat
        # (every 30 seconds) last touched the row before the switch
        async with engine.source_lock(IngestionSource(cheshire_cat.agent_key, None, "uploading.txt")):
            await crud_ingestions.update_status(
                cheshire_cat.agent_key, None, "uploading.txt", status="processing", owner=engine.OWNER,
            )
            await crud_ingestions.reset_phase_entries(cheshire_cat.agent_key, None, "uploading.txt", ["embedding"])
            worker_ready.set()
            await asyncio.wait_for(worker_done.wait(), 10)

    worker = asyncio.create_task(live_worker())
    transfer = None
    try:
        await asyncio.wait_for(worker_ready.wait(), 5)
        await asyncio.sleep(0.05)  # the switch comes later than the last write of the worker
        new_database_starts_empty(monkeypatch, cheshire_cat)
        transfer = asyncio.create_task(cheshire_cat.transfer_vector_points_from(previous))
        await asyncio.sleep(0.5)
        assert not transfer.done()  # it waits for the worker of the upload
        assert await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)
    finally:
        worker_done.set()
        await asyncio.wait_for(worker, 10)
        if transfer is not None:
            await asyncio.wait_for(transfer, 10)

    assert reruns == [("uploading.txt", ["embedding"])]
    assert await is_up_to_date_in_the_new_database(cheshire_cat, "uploading.txt")
    assert await is_up_to_date_in_the_new_database(cheshire_cat, "untouched.txt")  # copied


async def test_a_source_that_cannot_be_embedded_again_after_the_copy_is_left_to_the_resume(
    lizard, cheshire_cat, files, monkeypatch,
):
    await ingest(cheshire_cat)
    previous = await PreviousVectorDatabase.of(cheshire_cat.vector_memory_handler)
    real = RabbitHole.rerun_source
    failures = {"left": 1}

    async def rerun_source(self, cat, name, chat_id=None, blocking=True, invalidate=None):
        if failures["left"]:
            failures["left"] -= 1
            raise ConnectionError("redis down for a moment")
        return await real(self, cat, name, chat_id, blocking, invalidate)

    monkeypatch.setattr(RabbitHole, "rerun_source", rerun_source)
    outcomes = record_transfer_hook(monkeypatch, cheshire_cat)
    real_start = crud_ingestions.start_vector_transfer

    async def start_then_touch(agent_id):
        since = await real_start(agent_id)
        await asyncio.sleep(0.01)
        await crud_ingestions.touch_status(agent_id, None, "story.txt")  # an upload ends during the copy
        return since

    monkeypatch.setattr(crud_ingestions, "start_vector_transfer", start_then_touch)
    new_database_starts_empty(monkeypatch, cheshire_cat)

    await cheshire_cat.transfer_vector_points_from(previous)

    assert outcomes == [(True, True)]
    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert "embedding" not in row["completed_phases"]  # left to the resume
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)
    assert await is_up_to_date_in_the_new_database(cheshire_cat, "story.txt")


@pytest.mark.parametrize("failure", ["initialize", "halfway", "embedder"])
async def test_a_failed_transfer_leaves_every_source_to_be_embedded_again_by_the_resume(
    lizard, cheshire_cat, files, monkeypatch, failure,
):
    await ingest(cheshire_cat)
    plug_settings_phase(monkeypatch, cheshire_cat)
    await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt")  # the plugin phase is up to date
    runs_before = SettingsPhase.runs
    outcomes = record_transfer_hook(monkeypatch, cheshire_cat)
    previous = await PreviousVectorDatabase.of(
        cheshire_cat.vector_memory_handler, failing=str(VectorMemoryType.EPISODIC) if failure == "halfway" else None,
    )
    new_database_starts_empty(monkeypatch, cheshire_cat)
    if failure == "initialize":
        async def initialize(self, embedder_name_, embedder_size):
            raise ConnectionError("new vector db down")

        monkeypatch.setattr(type(cheshire_cat.vector_memory_handler), "initialize", initialize)
    elif failure == "embedder":
        real_embedder = CheshireCat.embedder
        calls = {"n": 0}

        async def embedder(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("settings unreadable")
            return await real_embedder(self)

        monkeypatch.setattr(CheshireCat, "embedder", embedder)
    procedures = []
    real_procedures = CheshireCat.embed_procedures

    async def embed_procedures(self, pt=None):
        procedures.append(self.agent_key)
        return await real_procedures(self, pt)

    monkeypatch.setattr(CheshireCat, "embed_procedures", embed_procedures)

    await cheshire_cat.transfer_vector_points_from(previous)

    assert outcomes == [(False, True)]
    assert not await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is None
    assert procedures == [cheshire_cat.agent_key]
    phases = (await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt"))["completed_phases"]
    assert "embedding" not in phases and "graph" not in phases  # the dependents too

    if failure == "initialize":
        new_database_starts_empty(monkeypatch, cheshire_cat)  # the new database is back
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)
    assert await is_up_to_date_in_the_new_database(cheshire_cat, "story.txt")
    assert SettingsPhase.runs == runs_before + 1


async def test_a_transfer_stopped_with_the_pod_is_taken_over_by_the_resume(lizard, cheshire_cat, files, monkeypatch):
    await ingest(cheshire_cat)
    outcomes = record_transfer_hook(monkeypatch, cheshire_cat)
    previous = await PreviousVectorDatabase.of(cheshire_cat.vector_memory_handler, gate=asyncio.Event())
    new_database_starts_empty(monkeypatch, cheshire_cat)
    transfer = asyncio.create_task(cheshire_cat.transfer_vector_points_from(previous))
    try:
        await wait_until(lambda: asyncio.sleep(0, bool(previous.reads)))
    finally:
        transfer.cancel()  # the pod stops: as if the process died
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(transfer, 10)

    assert outcomes == []
    from cat.db.database import get_async_db

    ttl = await get_async_db().ttl(crud_ingestions.maintenance_key(cheshire_cat.agent_key))
    assert 0 < ttl <= engine.lock_timeout_seconds()  # it expires by itself, soon
    transfer_record = await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key)
    assert transfer_record is not None

    await crud_ingestions.clear_maintenance(cheshire_cat.agent_key)  # the marker expired
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)  # the transfer may be alive on another pod: no
    assert not await is_up_to_date_in_the_new_database(cheshire_cat, "story.txt")

    # the heartbeat stopped: past the lock timeout the transfer is dead, and the resume takes it over
    await make_stale(cheshire_cat.agent_key)
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    assert await is_up_to_date_in_the_new_database(cheshire_cat, "story.txt")
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is None
    assert outcomes == [(False, False)]  # the plugins know it failed


async def test_a_dead_transfer_of_an_agent_without_sources_is_closed_by_the_resume(lizard, cheshire_cat, monkeypatch):
    procedures = []

    async def embed_procedures(self, pt=None):
        procedures.append(self.agent_key)

    monkeypatch.setattr(CheshireCat, "embed_procedures", embed_procedures)
    outcomes = record_transfer_hook(monkeypatch, cheshire_cat)
    await crud.store(crud_ingestions.vector_transfer_key(cheshire_cat.agent_key), {"since": 1.0, "beat": 1.0})

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    assert procedures == [cheshire_cat.agent_key]
    assert outcomes == [(False, False)]
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is None


async def test_a_long_copy_keeps_the_agent_in_maintenance(cheshire_cat, monkeypatch):
    """The marker lives a lock timeout, renewed by the heartbeat of the transfer, also when cleared by another actor
    (e.g. the end of a change of the embedder) while the copy runs."""
    monkeypatch.setenv("CAT_INGESTION_HEARTBEAT_SECONDS", "1")
    monkeypatch.setenv("CAT_INGESTION_LOCK_TIMEOUT_SECONDS", "2")
    previous = await PreviousVectorDatabase.of(cheshire_cat.vector_memory_handler, gate=asyncio.Event())
    transfer = asyncio.create_task(cheshire_cat.transfer_vector_points_from(previous))
    try:
        await wait_until(lambda: asyncio.sleep(0, bool(previous.reads)))
        beat = (await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key))["beat"]
        await crud_ingestions.clear_maintenance(cheshire_cat.agent_key)
        await asyncio.sleep(3)  # longer than the TTL of the marker
        assert await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)
        assert (await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key))["beat"] > beat
    finally:
        previous.gate.set()
        await asyncio.wait_for(transfer, 10)
    assert not await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is None


async def test_a_failed_source_embedded_before_the_failed_copy_gets_its_points_again(
    lizard, cheshire_cat, files, monkeypatch,
):
    """A source failed in a phase after the embedding (e.g. its graph): a failed copy leaves its embedding to do, and the
    resume runs it once, with its dependents, though the source is in error."""
    plug_settings_phase(monkeypatch, cheshire_cat)
    SettingsPhase.outcomes = ["error"]
    await ingest(cheshire_cat)
    assert (await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt"))["status"] == "error"
    previous = await PreviousVectorDatabase.of(cheshire_cat.vector_memory_handler, failing=str(VectorMemoryType.EPISODIC))
    new_database_starts_empty(monkeypatch, cheshire_cat)

    await cheshire_cat.transfer_vector_points_from(previous)
    SettingsPhase.outcomes, runs = ["error"], SettingsPhase.runs
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert row["status"] == "error" and SettingsPhase.runs == runs + 1  # its graph ran once more, and failed
    assert len(await points_of(cheshire_cat, "story.txt")) == row["completed_phases"]["embedding"]["processed"] > 0
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)
    assert SettingsPhase.runs == runs + 1  # not again: the failure is recorded with the current tag


async def test_a_new_phase_runs_once_on_a_source_failed_elsewhere(lizard, cheshire_cat, files, monkeypatch):
    """A phase never tried, whose dependencies completed, is work to do also on a source in error."""
    plug_settings_phase(monkeypatch, cheshire_cat)
    SettingsPhase.outcomes = ["error"]
    await ingest(cheshire_cat)
    await crud_ingestions.reset_phase_entries(cheshire_cat.agent_key, None, "story.txt", ["graph"])  # never tried
    runs = SettingsPhase.runs

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)

    assert SettingsPhase.runs == runs + 1


async def test_procedures_that_cannot_be_embedded_keep_the_transfer_for_the_resume(
    lizard, cheshire_cat, files, monkeypatch,
):
    await ingest(cheshire_cat)
    outcomes = record_transfer_hook(monkeypatch, cheshire_cat)
    previous = await PreviousVectorDatabase.of(cheshire_cat.vector_memory_handler, failing=str(VectorMemoryType.EPISODIC))
    new_database_starts_empty(monkeypatch, cheshire_cat)
    procedures = {"fail": 2, "done": 0}
    real_procedures = CheshireCat.embed_procedures

    async def embed_procedures(self, pt=None):
        if procedures["fail"]:
            procedures["fail"] -= 1
            raise ConnectionError("new vector db down")
        procedures["done"] += 1
        return await real_procedures(self, pt)

    monkeypatch.setattr(CheshireCat, "embed_procedures", embed_procedures)

    await cheshire_cat.transfer_vector_points_from(previous)

    assert outcomes == [(False, True)]
    assert not await crud_ingestions.is_in_maintenance(cheshire_cat.agent_key)
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is not None  # left to the resume

    await recovery.resume_agent(lizard, cheshire_cat.agent_key)  # its heartbeat is fresh: alive for the resume
    assert outcomes == [(False, True)]
    await make_stale(cheshire_cat.agent_key)
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)  # taken over: the procedures fail again
    assert outcomes == [(False, True), (False, False)]
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is not None
    assert await is_up_to_date_in_the_new_database(cheshire_cat, "story.txt")  # the sources went on meanwhile

    await make_stale(cheshire_cat.agent_key)
    await recovery.resume_agent(lizard, cheshire_cat.agent_key)
    assert procedures["done"] == 1
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is None


async def test_a_take_over_claims_the_transfer_for_its_whole_duration(lizard, cheshire_cat, files, monkeypatch):
    """While a pod takes a dead transfer over (it may wait for a live worker), its heartbeat keeps the record alive: a
    second pod, whose resume lock was freed meanwhile, leaves the agent alone."""
    monkeypatch.setenv("CAT_INGESTION_HEARTBEAT_SECONDS", "1")
    monkeypatch.setenv("CAT_INGESTION_LOCK_TIMEOUT_SECONDS", "2")
    await ingest(cheshire_cat)
    await crud.store(crud_ingestions.vector_transfer_key(cheshire_cat.agent_key), {"since": 1.0, "beat": 1.0})
    held, release = asyncio.Event(), asyncio.Event()
    takeovers = []
    real_invalidate = RabbitHole.invalidate_source

    async def slow_invalidate(self, cat, name, chat_id=None, phase_ids=("embedding",)):
        takeovers.append(name)
        held.set()
        await asyncio.wait_for(release.wait(), 10)  # a live worker holds the source
        return await real_invalidate(self, cat, name, chat_id, phase_ids)

    monkeypatch.setattr(RabbitHole, "invalidate_source", slow_invalidate)
    first = asyncio.create_task(recovery.resume_agent(lizard, cheshire_cat.agent_key))
    try:
        await asyncio.wait_for(held.wait(), 5)
        await asyncio.sleep(3)  # longer than the lock timeout: the resume lock of the first pod expired
        from cat.db.database import get_async_db

        locks = [k async for k in get_async_db().scan_iter("lock:ingestion-resume*")]
        if locks:
            await get_async_db().delete(*locks)
        assert not locks or await get_async_db().exists(*locks) == 0
        await recovery.resume_agent(lizard, cheshire_cat.agent_key)  # a second pod
        assert takeovers == ["story.txt"]
    finally:
        release.set()
        await asyncio.wait_for(first, 10)
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is None


async def test_a_failed_heartbeat_is_retried_at_the_next_one(cheshire_cat, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_HEARTBEAT_SECONDS", "1")
    monkeypatch.setenv("CAT_INGESTION_LOCK_TIMEOUT_SECONDS", "2")
    beats = {"failed": 0, "done": 0}
    real_beat = crud_ingestions.beat_vector_transfer

    async def flaky_beat(agent_id, since):
        if not beats["failed"]:
            beats["failed"] += 1
            raise ConnectionError("redis down for a moment")
        beats["done"] += 1
        return await real_beat(agent_id, since)

    monkeypatch.setattr(crud_ingestions, "beat_vector_transfer", flaky_beat)
    previous = await PreviousVectorDatabase.of(cheshire_cat.vector_memory_handler, gate=asyncio.Event())
    transfer = asyncio.create_task(cheshire_cat.transfer_vector_points_from(previous))
    try:
        await wait_until(lambda: asyncio.sleep(0, beats["done"] > 0), seconds=5)
        assert beats["failed"] == 1
    finally:
        previous.gate.set()
        await asyncio.wait_for(transfer, 10)
    assert await crud_ingestions.get_vector_transfer(cheshire_cat.agent_key) is None
