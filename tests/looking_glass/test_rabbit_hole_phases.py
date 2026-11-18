"""The ingestion of a source through its phases: the built-in ones, the ones of the plugins, the diary and the final
state, the deletion phase by phase, the lock of the source and its heartbeat."""
import asyncio
from io import BytesIO

import pytest
from redis.exceptions import LockError

from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds import settings as crud_settings
from cat.db.database import get_async_db
from cat.rabbit_hole import RabbitHole, chunks, engine
from cat.rabbit_hole.phases import (
    IngestionPhase,
    IngestionSource,
    PhaseGraphError,
    PhaseResult,
    compute_tags,
    execution_order,
    service_settings,
)
from cat.services.memory.models import VectorMemoryType

TEXT = ("The rabbit hole went straight on like a tunnel for some way, and then dipped suddenly down. " * 30).encode()


def Phase(phase_id, depends_on=("embedding",), outcomes=("done",), settings_value=None, applies=True, budget=None):
    """A plugin phase class whose outcomes are scripted: one result (or exception) per call, the last one repeated.
    The plugins pass classes; the counters live on the class, since the RabbitHole instantiates it for every run."""

    class ScriptedPhase(IngestionPhase):
        id = phase_id
        not_ready_budget_seconds = budget
        calls = 0
        deleted: list = []

        async def settings(self, cat):
            return type(self).settings_value

        async def applies_to(self, source, cat):
            return (True, None) if applies else (False, "not for this source")

        async def process(self, source, cat):
            cls = type(self)
            outcome = list(outcomes)[min(cls.calls, len(outcomes) - 1)]
            cls.calls += 1
            if isinstance(outcome, Exception):
                raise outcome
            if outcome is None:
                return None
            if outcome == "not_ready":
                return PhaseResult(status="not_ready", retry_after=0)
            warnings = [{"code": "w"}] if outcome == "done" else None
            return PhaseResult(status=outcome, reason=f"{phase_id} {outcome}", warnings=warnings)

        async def delete(self, source, cat):
            type(self).deleted.append(source.name)

    ScriptedPhase.depends_on = tuple(depends_on)
    ScriptedPhase.settings_value = settings_value
    ScriptedPhase.deleted = []
    return ScriptedPhase


@pytest.fixture
def local_files(cheshire_cat, monkeypatch):
    """The agent keeps its files (the default file manager of the tests keeps none)."""
    monkeypatch.setattr(cheshire_cat, "file_manager", LocalFileManager())
    return cheshire_cat.file_manager


def plug(cheshire_cat, monkeypatch, *phases):
    """Let the plugins of the agent add the phases (on every plugin manager: a chat resolves its own CheshireCat)."""
    manager_class = type(cheshire_cat.plugin_manager)
    original = manager_class.execute_hook

    async def execute_hook(self, name, *args, caller):
        if name == "rabbithole_ingestion_phases":
            return list(args[0]) + list(phases)
        return await original(self, name, *args, caller=caller)

    monkeypatch.setattr(manager_class, "execute_hook", execute_hook)


async def ingest(cheshire_cat, name="story.txt", data=TEXT, chat=None):
    return await cheshire_cat.rabbit_hole.ingest_file(
        cat=chat or cheshire_cat, file=BytesIO(data), metadata={"tag": "t"}, filename=name, content_type="text/plain",
    )


async def prepared(cheshire_cat):
    """The rabbit hole of the instance, set up for the agent (as every operation on a source does)."""
    rabbit_hole = __import__("copy").copy(cheshire_cat.rabbit_hole)
    await rabbit_hole.setup(cheshire_cat)
    return rabbit_hole


async def points_of(cheshire_cat, name, collection=VectorMemoryType.DECLARATIVE):
    points, _ = await cheshire_cat.vector_memory_handler.get_all_tenant_points(
        str(collection), metadata={"source": name}, with_vectors=False,
    )
    return points


# --- the graph of the phases ------------------------------------------------------------------------------------

def test_phases_run_after_the_phases_they_depend_on():
    a, b, c = Phase("a", ())(), Phase("b", ("a",))(), Phase("c", ("b", "a"))()
    assert [p.id for p in execution_order([c, b, a])] == ["a", "b", "c"]


def test_a_dependency_on_an_unknown_phase_is_refused():
    with pytest.raises(PhaseGraphError, match="unknown"):
        execution_order([Phase("a", ("ghost",))()])


def test_a_cycle_is_refused():
    with pytest.raises(PhaseGraphError, match="cycle"):
        execution_order([Phase("a", ("b",))(), Phase("b", ("a",))()])


async def test_a_change_of_settings_changes_the_tags_of_the_phase_and_of_its_dependents_only():
    a, b, c = Phase("a", (), settings_value=1)(), Phase("b", ("a",))(), Phase("c", ())()
    before = await compute_tags(execution_order([a, b, c]), None)
    assert before == await compute_tags(execution_order([a, b, c]), None)

    type(a).settings_value = 2
    after = await compute_tags(execution_order([a, b, c]), None)

    assert after["a"] != before["a"] and after["b"] != before["b"] and after["c"] == before["c"]


async def test_the_settings_of_a_service_leave_out_the_credentials(cheshire_cat):
    assert await service_settings("nobody", "chunker") == {"name": None}

    from cat.db import models
    await crud_settings.create_setting("a1", models.Setting(
        name="OpenAIEmbedderConfig", value={"model": "m", "api_key": "secret", "client_secret": "s"}, category="embedder",
    ))
    assert await service_settings("a1", "embedder") == {"name": "OpenAIEmbedderConfig", "value": {"model": "m"}}


# --- an ingestion through the built-in phases -------------------------------------------------------------------

async def test_an_ingestion_completes_both_phases_and_stores_the_points(cheshire_cat, local_files):
    row = await ingest(cheshire_cat)

    assert row["status"] == "completed" and row["phase"] is None and row["error"] is None and row["owner"] is None
    assert set(row["completed_phases"]) == {"parsing_chunking", "embedding"}
    assert {e["state"] for e in row["completed_phases"].values()} == {"complete"}
    points = await points_of(cheshire_cat, "story.txt")
    assert len(points) == row["completed_phases"]["embedding"]["processed"] > 0
    assert all(p.payload["metadata"]["tag"] == "t" for p in points)
    # the chunks are kept, outside the listing of the files
    docs, images = chunks.read_chunks(local_files, cheshire_cat.agent_key, None, "story.txt")
    assert len(docs) == len(points) and images == []
    assert [f.name for f in local_files.list_files(cheshire_cat.agent_key)] == ["story.txt"]


async def test_an_ingestion_works_with_a_file_manager_keeping_no_files(cheshire_cat):
    # the default file manager of the tests keeps nothing: the phases of the run pass the chunks in memory
    row = await ingest(cheshire_cat)
    assert row["status"] == "completed"
    assert len(await points_of(cheshire_cat, "story.txt")) > 0


async def test_a_new_upload_replaces_the_points_of_the_source(cheshire_cat, local_files):
    await ingest(cheshire_cat)
    first = {p.id for p in await points_of(cheshire_cat, "story.txt")}

    row = await ingest(cheshire_cat, data=TEXT[: len(TEXT) // 3])

    second = {p.id for p in await points_of(cheshire_cat, "story.txt")}
    assert row["status"] == "completed"
    assert second and not (first & second)
    assert len(second) == row["completed_phases"]["embedding"]["processed"]


async def test_nothing_runs_again_when_nothing_changed(cheshire_cat, local_files, monkeypatch):
    extra = Phase("graph")
    plug(cheshire_cat, monkeypatch, extra)
    await ingest(cheshire_cat)
    source = IngestionSource(cheshire_cat.agent_key, None, "story.txt")

    async def boom(*args, **kwargs):
        raise AssertionError("a fresh phase ran again")

    monkeypatch.setattr(RabbitHole, "run_parsing_chunking", boom)
    monkeypatch.setattr(RabbitHole, "run_embedding", boom)
    async with engine.source_lock(source):
        row = await engine.run_phases(await prepared(cheshire_cat), cheshire_cat, source)

    assert row["status"] == "completed" and extra.calls == 1


async def test_a_change_of_a_phase_settings_runs_it_and_its_dependents_again(cheshire_cat, local_files, monkeypatch):
    first, second = Phase("graph", settings_value=1), Phase("summary", ("graph",))
    plug(cheshire_cat, monkeypatch, first, second)
    await ingest(cheshire_cat)
    source = IngestionSource(cheshire_cat.agent_key, None, "story.txt")

    first.settings_value = 2
    async with engine.source_lock(source):
        row = await engine.run_phases(cheshire_cat.rabbit_hole, cheshire_cat, source)

    assert (first.calls, second.calls) == (2, 2)
    assert row["status"] == "completed"


async def test_a_change_of_the_embedder_reuses_the_stored_chunks(cheshire_cat, local_files, monkeypatch):
    await ingest(cheshire_cat)
    source = IngestionSource(cheshire_cat.agent_key, None, "story.txt")
    parsed = []
    real = RabbitHole.run_parsing_chunking

    async def counting(self, src, cat):
        parsed.append(src.name)
        return await real(self, src, cat)

    real_settings = service_settings

    async def other_embedder(key_id, category):
        return {"name": "another-embedder"} if category == "embedder" else await real_settings(key_id, category)

    monkeypatch.setattr(RabbitHole, "run_parsing_chunking", counting)
    monkeypatch.setattr("cat.rabbit_hole.phases.service_settings", other_embedder)
    async with engine.source_lock(source):
        row = await engine.run_phases(await prepared(cheshire_cat), cheshire_cat, source)

    assert parsed == [] and row["status"] == "completed"  # the chunker settings are "None" in both cases
    assert len(await points_of(cheshire_cat, "story.txt")) == row["completed_phases"]["embedding"]["processed"]


async def test_a_source_in_a_chat_goes_to_the_episodic_memory(cheshire_cat, stray, local_files):
    row = await ingest(cheshire_cat, chat=stray)

    assert row["chat_id"] == stray.id and row["status"] == "completed"
    episodic = await points_of(cheshire_cat, "story.txt", VectorMemoryType.EPISODIC)
    assert episodic and all(p.payload["metadata"]["chat_id"] == stray.id for p in episodic)
    assert await points_of(cheshire_cat, "story.txt") == []


# --- outcomes of the phases of the plugins ----------------------------------------------------------------------

@pytest.mark.parametrize("outcomes, state, status", [
    (["done"], "complete", "completed"),
    (["not_ready", "not_ready", "done"], "complete", "completed"),
    (["skipped"], "skipped", "completed"),
    (["error"], "error", "error"),
    ([RuntimeError("llm down")], "error", "error"),
    ([None], "error", "error"),
])
async def test_the_outcome_of_a_plugin_phase(cheshire_cat, local_files, monkeypatch, outcomes, state, status):
    extra, dependent = Phase("graph", outcomes=outcomes), Phase("summary", ("graph",))
    plug(cheshire_cat, monkeypatch, extra, dependent)

    row = await ingest(cheshire_cat)

    diary = row["completed_phases"]
    assert diary["graph"]["state"] == state
    assert row["status"] == status
    expected_dependent = {"complete": "complete", "skipped": "skipped", "error": "error"}[state]
    assert diary["summary"]["state"] == expected_dependent
    assert dependent.calls == (1 if state == "complete" else 0)
    if status == "error":
        assert row["error"] and "summary" not in (row["error"] or "")
    if state == "complete":
        assert diary["graph"].get("warnings") == [{"code": "w"}]


async def test_a_phase_not_ready_beyond_its_budget_fails(cheshire_cat, local_files, monkeypatch):
    plug(cheshire_cat, monkeypatch, Phase("graph", outcomes=["not_ready"], budget=0))

    row = await ingest(cheshire_cat)

    assert row["status"] == "error" and "Not ready" in row["completed_phases"]["graph"]["reason"]


async def test_the_budget_of_not_ready_comes_from_the_environment(cheshire_cat, local_files, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_NOT_READY_MAX_SECONDS", "0")
    plug(cheshire_cat, monkeypatch, Phase("graph", outcomes=["not_ready"]))

    row = await ingest(cheshire_cat)

    assert row["completed_phases"]["graph"]["state"] == "error"


async def test_a_phase_that_does_not_apply_is_skipped_without_running(cheshire_cat, local_files, monkeypatch):
    extra = Phase("graph", applies=False)
    plug(cheshire_cat, monkeypatch, extra)

    row = await ingest(cheshire_cat)

    assert extra.calls == 0
    assert row["completed_phases"]["graph"] | {"finished_at": 0} == {
        "state": "skipped", "tag": row["completed_phases"]["graph"]["tag"], "reason": "not for this source",
        "processed": None, "expected": None, "finished_at": 0,
    }


async def test_invalid_plugin_phases_are_left_out(cheshire_cat, local_files, monkeypatch):
    class Broken(IngestionPhase):
        id = "broken"

        def __init__(self, required):
            pass

    plug(cheshire_cat, monkeypatch, "not a phase", Phase("embedding", ())(), Phase("embedding", ()), Broken,
         Phase("graph-rag", ()), Phase("", ()))

    row = await ingest(cheshire_cat)

    assert row["status"] == "completed" and set(row["completed_phases"]) == {"parsing_chunking", "embedding"}


async def test_plugin_phase_classes_pass_through_the_real_hooks_unchanged(cheshire_cat):
    graph = Phase("graph")
    passed = await cheshire_cat.plugin_manager.execute_hook("rabbithole_ingestion_phases", [graph], caller=cheshire_cat)
    assert passed == [graph] and passed[0] is graph


async def test_plugin_phases_forming_a_cycle_fail_the_source(cheshire_cat, local_files, monkeypatch):
    plug(cheshire_cat, monkeypatch, Phase("a", ("b",)), Phase("b", ("a",)))

    row = await ingest(cheshire_cat)

    assert row["status"] == "error" and "cycle" in row["error"]


async def test_a_failed_phase_runs_again_only_after_its_diary_is_reset(cheshire_cat, local_files, monkeypatch):
    extra = Phase("graph", outcomes=["error", "done"])
    plug(cheshire_cat, monkeypatch, extra)
    await ingest(cheshire_cat)
    source = IngestionSource(cheshire_cat.agent_key, None, "story.txt")

    async with engine.source_lock(source):
        assert (await engine.run_phases(cheshire_cat.rabbit_hole, cheshire_cat, source))["status"] == "error"
    await crud_ingestions.reset_phase_entries(cheshire_cat.agent_key, None, "story.txt", ["graph"])
    async with engine.source_lock(source):
        row = await engine.run_phases(cheshire_cat.rabbit_hole, cheshire_cat, source)

    assert extra.calls == 2 and row["status"] == "completed"


async def test_running_the_phases_of_a_deleted_row_does_nothing(cheshire_cat):
    source = IngestionSource(cheshire_cat.agent_key, None, "ghost.txt")
    assert await engine.run_phases(cheshire_cat.rabbit_hole, cheshire_cat, source) is None


# --- failures before the phases ----------------------------------------------------------------------------------

async def test_an_empty_file_fails_the_source_and_notifies_the_plugins_once(cheshire_cat, monkeypatch):
    stored_hook = []
    original = cheshire_cat.plugin_manager.execute_hook

    async def execute_hook(name, *args, caller):
        if name == "after_rabbithole_stored_documents":
            stored_hook.append(args)
        return await original(name, *args, caller=caller)

    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", execute_hook)

    row = await ingest(cheshire_cat, data=b"")

    assert row["status"] == "error" and "Something went wrong" in row["error"]
    assert stored_hook == [("story.txt", [])]


async def test_a_missing_file_name_fails_before_any_state(cheshire_cat):
    assert await cheshire_cat.rabbit_hole.ingest_file(cat=cheshire_cat, file=BytesIO(b"x"), metadata={}) is None
    assert await crud_ingestions.list_statuses(cheshire_cat.agent_key) == []


async def test_a_failed_phase_notifies_the_chat(cheshire_cat, stray, local_files, monkeypatch):
    errors = []

    async def send_error(message):
        errors.append(message)

    monkeypatch.setattr(type(stray.notifier), "send_error", lambda self, message: send_error(message))
    plug(cheshire_cat, monkeypatch, Phase("graph", outcomes=["error"]))

    await ingest(cheshire_cat, chat=stray)

    assert errors == ["Error processing story.txt: graph error"]


async def test_the_deletion_of_the_agent_stops_the_ingestion(cheshire_cat, local_files, monkeypatch):
    async def agent_deleted(self, source, cat):
        await get_async_db().delete(crud_settings.format_key(cheshire_cat.agent_key))
        return PhaseResult(status="done")

    monkeypatch.setattr(RabbitHole, "run_parsing_chunking", agent_deleted)

    assert await ingest(cheshire_cat) is None
    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert row["completed_phases"] == {}  # nothing written after the deletion


async def test_the_deletion_of_the_row_stops_the_ingestion_and_cleans_up(cheshire_cat, local_files, monkeypatch):
    # e.g. the chat of the source is deleted while its source is processed
    real = RabbitHole.run_parsing_chunking

    async def chat_deleted(self, source, cat):
        result = await real(self, source, cat)
        await crud_ingestions.delete_status(source.agent_id, source.chat_id, source.name)
        return result

    monkeypatch.setattr(RabbitHole, "run_parsing_chunking", chat_deleted)

    assert await ingest(cheshire_cat) is None
    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") is None
    assert await points_of(cheshire_cat, "story.txt") == []
    with pytest.raises(FileNotFoundError):
        chunks.read_chunks(local_files, cheshire_cat.agent_key, None, "story.txt")


async def test_a_worker_that_lost_the_lock_stops_without_touching_the_source(cheshire_cat, local_files, monkeypatch):
    from redis.exceptions import LockNotOwnedError
    from redis.asyncio.lock import Lock

    monkeypatch.setenv("CAT_INGESTION_HEARTBEAT_SECONDS", "1")

    async def lost(self, *args, **kwargs):
        raise LockNotOwnedError("the lock expired: another worker may own it")

    async def slow(self, source, cat):
        await asyncio.sleep(1.5)  # a heartbeat finds the lock lost meanwhile
        return PhaseResult(status="done")

    embedded = []

    async def embedding(self, source, cat):
        embedded.append(source.name)
        return PhaseResult(status="done")

    monkeypatch.setattr(Lock, "extend", lost)
    monkeypatch.setattr(RabbitHole, "run_parsing_chunking", slow)
    monkeypatch.setattr(RabbitHole, "run_embedding", embedding)

    row = await ingest(cheshire_cat)

    assert embedded == []  # the next phase is left to the owner
    assert row["status"] == "processing" and row["completed_phases"] == {}


async def test_a_chat_deleted_during_the_download_leaves_nothing_behind(cheshire_cat, stray, local_files, monkeypatch):
    real = RabbitHole._resolve_source_bytes

    async def download_then_chat_deleted(self, file, filename, content_type):
        resolved = await real(self, file, filename, content_type)
        await crud_ingestions.destroy_chat(cheshire_cat.agent_key, stray.id)
        return resolved

    monkeypatch.setattr(RabbitHole, "_resolve_source_bytes", download_then_chat_deleted)

    assert await ingest(cheshire_cat, chat=stray) is None
    assert chunks.read_source_bytes(local_files, cheshire_cat.agent_key, stray.id, "story.txt") is None
    assert not local_files.file_exists("story.txt", f"{cheshire_cat.agent_key}/{stray.id}")


async def test_a_chat_deleted_while_keeping_the_bytes_removes_them(cheshire_cat, stray, local_files, monkeypatch):
    real = RabbitHole._keep_source_bytes

    async def keep_then_chat_deleted(self, source, file_bytes, content_type, store_file):
        await real(self, source, file_bytes, content_type, store_file)
        await crud_ingestions.destroy_chat(cheshire_cat.agent_key, stray.id)

    monkeypatch.setattr(RabbitHole, "_keep_source_bytes", keep_then_chat_deleted)

    assert await ingest(cheshire_cat, chat=stray) is None
    assert chunks.read_source_bytes(local_files, cheshire_cat.agent_key, stray.id, "story.txt") is None


async def test_a_file_already_stored_gets_no_copy(cheshire_cat, local_files):
    local_files.write_file(TEXT, "story.txt", cheshire_cat.agent_key)

    row = await cheshire_cat.rabbit_hole.ingest_file(
        cat=cheshire_cat, file=BytesIO(TEXT), metadata={}, filename="story.txt", store_file=False,
        content_type="text/plain",
    )

    assert row["status"] == "completed"
    assert not local_files.file_exists(".rabbithole/" + chunks.source_id("story.txt") + ".source", cheshire_cat.agent_key)


async def test_metadata_not_in_json_survive_the_chunks_file(cheshire_cat, local_files, monkeypatch):
    import datetime
    original = type(cheshire_cat.plugin_manager).execute_hook

    async def execute_hook(self, name, *args, caller):
        result = await original(self, name, *args, caller=caller)
        if name == "before_rabbithole_stores_documents":
            for doc in result:
                doc.metadata["parsed_at"] = datetime.datetime(2026, 1, 1)
        return result

    monkeypatch.setattr(type(cheshire_cat.plugin_manager), "execute_hook", execute_hook)

    row = await ingest(cheshire_cat)

    assert row["status"] == "completed"
    docs, _ = chunks.read_chunks(local_files, cheshire_cat.agent_key, None, "story.txt")
    assert docs[0].metadata["parsed_at"] == "2026-01-01T00:00:00"


# --- deletion -----------------------------------------------------------------------------------------------------

async def test_deleting_a_source_removes_every_phase_output_in_reverse_order(cheshire_cat, local_files, monkeypatch):
    order = []
    extra = Phase("graph")
    plug(cheshire_cat, monkeypatch, extra)
    await ingest(cheshire_cat)
    for name in ("delete_parsing_chunking", "delete_embedding"):
        real = getattr(RabbitHole, name)

        def tracking(real=real, name=name):
            async def wrapper(self, source, cat):
                order.append(name)
                await real(self, source, cat)
            return wrapper
        monkeypatch.setattr(RabbitHole, name, tracking())

    await cheshire_cat.rabbit_hole.delete_source(cheshire_cat, "story.txt")

    assert extra.deleted == ["story.txt"]
    assert order == ["delete_embedding", "delete_parsing_chunking"]
    assert await points_of(cheshire_cat, "story.txt") == []
    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") is None
    with pytest.raises(FileNotFoundError):
        chunks.read_chunks(local_files, cheshire_cat.agent_key, None, "story.txt")


async def test_an_interrupted_deletion_resumes_from_the_phases_left(cheshire_cat, local_files, monkeypatch):
    extra = Phase("graph")
    plug(cheshire_cat, monkeypatch, extra)
    await ingest(cheshire_cat)
    real = RabbitHole.delete_embedding

    async def failing(self, source, cat):
        raise ConnectionError("vector db down")

    monkeypatch.setattr(RabbitHole, "delete_embedding", failing)
    with pytest.raises(ConnectionError):
        await cheshire_cat.rabbit_hole.delete_source(cheshire_cat, "story.txt")
    row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")
    assert row["status"] == "deleting" and row["deleted_phases"] == ["graph"]

    monkeypatch.setattr(RabbitHole, "delete_embedding", real)
    await cheshire_cat.rabbit_hole.delete_source(cheshire_cat, "story.txt")

    assert extra.deleted == ["story.txt"]  # not deleted twice
    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") is None
    assert await points_of(cheshire_cat, "story.txt") == []


async def test_deleting_a_source_without_state_removes_its_points(cheshire_cat):
    await (await prepared(cheshire_cat)).store_documents(
        docs=[__import__("langchain_core.documents", fromlist=["Document"]).Document(page_content="legacy")],
        source="legacy.txt",
    )
    await cheshire_cat.rabbit_hole.delete_source(cheshire_cat, "legacy.txt")
    assert await points_of(cheshire_cat, "legacy.txt") == []


async def test_deleting_with_broken_plugin_phases_still_removes_the_built_in_outputs(cheshire_cat, local_files, monkeypatch):
    await ingest(cheshire_cat)
    plug(cheshire_cat, monkeypatch, Phase("a", ("b",)), Phase("b", ("a",)))

    await cheshire_cat.rabbit_hole.delete_source(cheshire_cat, "story.txt")

    assert await points_of(cheshire_cat, "story.txt") == []
    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") is None


async def test_the_file_manager_routes_delete_through_the_phases(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    deleted = []

    async def delete_source(self, cat, name, chat_id=None, wait=None, before=None):
        deleted.append((name, chat_id))
        return await before()

    monkeypatch.setattr(RabbitHole, "delete_source", delete_source)
    cheshire_cat.file_manager.write_file(b"x", "a.txt", cheshire_cat.agent_key)

    await secure_client.delete("/file_manager/files/a.txt", headers=secure_client_headers)

    assert deleted == [("a.txt", None)]


async def test_deleting_points_of_a_whole_source_removes_the_source(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    deleted = []

    async def delete_source(self, cat, name, chat_id=None, wait=None, before=None):
        deleted.append((name, chat_id))
        return await before()

    monkeypatch.setattr(RabbitHole, "delete_source", delete_source)
    requests = [
        ("declarative", {"source": "a.txt"}),  # a source of the knowledge base
        ("episodic", {"source": "b.txt", "chat_id": "c1"}),  # a source of a chat
        ("declarative", {"source": "c.txt", "chat_id": "c1"}),  # not a whole source: only points
        ("episodic", {"source": "d.txt"}),  # every chat: not one source
        ("declarative", {"source": "e.txt", "page": 2}),
        ("declarative", {}),
    ]
    for collection, metadata in requests:
        await secure_client.request(
            "DELETE", f"/memory/collections/{collection}/points", headers=secure_client_headers, json=metadata,
        )

    assert deleted == [("a.txt", None), ("b.txt", "c1")]


# --- the lock of the source and its heartbeat ---------------------------------------------------------------------

async def test_ingestions_of_the_same_source_run_one_at_a_time(cheshire_cat, local_files, monkeypatch):
    running, peak = 0, 0
    real = RabbitHole.run_parsing_chunking

    async def slow(self, source, cat):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        return await real(self, source, cat)

    monkeypatch.setattr(RabbitHole, "run_parsing_chunking", slow)
    monkeypatch.setattr("cat.rabbit_hole.core._ingestion_semaphore", None)
    monkeypatch.setenv("CAT_INGESTION_MAX_CONCURRENCY", "0")

    rows = await asyncio.gather(ingest(cheshire_cat), ingest(cheshire_cat), ingest(cheshire_cat))

    assert peak == 1 and [r["status"] for r in rows] == ["completed"] * 3


async def test_the_heartbeat_keeps_the_lock_and_the_row_alive(cheshire_cat, local_files, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_HEARTBEAT_SECONDS", "1")
    monkeypatch.setenv("CAT_INGESTION_LOCK_TIMEOUT_SECONDS", "2")
    source = IngestionSource(cheshire_cat.agent_key, None, "story.txt")
    seen = {}

    async def slow(self, src, cat):
        before = (await crud_ingestions.get_status(src.agent_id, None, src.name))["updated_at"]
        await asyncio.sleep(3.2)  # longer than the lock timeout: only the heartbeat keeps it
        seen["updated"] = (await crud_ingestions.get_status(src.agent_id, None, src.name))["updated_at"] > before + 1
        try:
            async with engine.source_lock(source, blocking=False):
                seen["stolen"] = True
        except LockError:
            seen["stolen"] = False
        return PhaseResult(status="done")

    monkeypatch.setattr(RabbitHole, "run_parsing_chunking", slow)
    monkeypatch.setattr(RabbitHole, "run_embedding", slow)

    await ingest(cheshire_cat)

    assert seen == {"updated": True, "stolen": False}


async def test_a_failed_heartbeat_is_retried(cheshire_cat, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_HEARTBEAT_SECONDS", "1")
    beats = []

    async def touch(*args):
        beats.append(args)
        if len(beats) == 1:
            raise ConnectionError("redis blip")
        return True

    monkeypatch.setattr(crud_ingestions, "touch_status", touch)
    async with engine.source_lock(IngestionSource(cheshire_cat.agent_key, None, "x.txt")):
        await asyncio.sleep(2.6)

    assert len(beats) >= 2


# --- deletions while the source is ingested, "delete all", files not stored -------------------------------------

async def hold_the_source(cheshire_cat, name, chat_id=None):
    """A worker holding the lock of the source (an ingestion running), until the returned event is set."""
    release, held = asyncio.Event(), asyncio.Event()

    async def worker():
        async with engine.source_lock(IngestionSource(cheshire_cat.agent_key, chat_id, name)):
            held.set()
            await release.wait()

    task = asyncio.create_task(worker())
    await held.wait()
    return release, task


async def test_deleting_a_file_being_ingested_answers_409_and_touches_nothing(
    secure_client, secure_client_headers, cheshire_cat, local_files, monkeypatch,
):
    monkeypatch.setattr("cat.routes.file_manager.DELETE_WAIT_SECONDS", 0.2)
    await ingest(cheshire_cat)
    release, task = await hold_the_source(cheshire_cat, "story.txt")

    response = await secure_client.delete("/file_manager/files/story.txt", headers=secure_client_headers)
    release.set()
    await task

    assert response.status_code == 409
    assert local_files.file_exists("story.txt", cheshire_cat.agent_key)
    assert len(await points_of(cheshire_cat, "story.txt")) > 0


async def test_deleting_the_points_of_a_source_being_ingested_answers_409(
    secure_client, secure_client_headers, cheshire_cat, local_files, monkeypatch,
):
    monkeypatch.setattr("cat.core_plugins.memory.endpoints.points.DELETE_WAIT_SECONDS", 0.2)
    await ingest(cheshire_cat)
    release, task = await hold_the_source(cheshire_cat, "story.txt")

    response = await secure_client.request(
        "DELETE", "/memory/collections/declarative/points", headers=secure_client_headers, json={"source": "story.txt"},
    )
    release.set()
    await task

    assert response.status_code == 409
    assert len(await points_of(cheshire_cat, "story.txt")) > 0


async def test_deleting_every_file_removes_every_source_of_the_scope_but_the_busy_ones(
    secure_client, secure_client_headers, cheshire_cat, monkeypatch,
):
    # the routes use the file manager configured for the agent
    await secure_client.put("/file_manager/settings/LocalFileManagerConfig", headers=secure_client_headers, json={})
    cheshire_cat.file_manager = await cheshire_cat.service_provider.get_file_manager(
        cheshire_cat.agent_key, cheshire_cat.plugin_manager,
    )
    local_files = cheshire_cat.file_manager
    monkeypatch.setattr("cat.routes.file_manager.DELETE_WAIT_SECONDS", 0.2)
    await ingest(cheshire_cat, name="a.txt")
    await ingest(cheshire_cat, name="busy.txt")
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "https://x.org/page", type_="url")
    local_files.write_file(b"legacy", "legacy.txt", cheshire_cat.agent_key)
    release, task = await hold_the_source(cheshire_cat, "busy.txt")

    response = await secure_client.delete("/file_manager/files", headers=secure_client_headers)
    release.set()
    await task

    assert response.status_code == 409 and "busy.txt" in response.json()["detail"]
    assert [r["source"] for r in await crud_ingestions.list_statuses(cheshire_cat.agent_key)] == ["busy.txt"]
    assert await points_of(cheshire_cat, "a.txt") == []
    assert [f.name for f in local_files.list_files(cheshire_cat.agent_key)] == ["busy.txt"]

    response = await secure_client.delete("/file_manager/files", headers=secure_client_headers)

    assert response.status_code == 200
    assert await crud_ingestions.list_statuses(cheshire_cat.agent_key) == []


async def test_a_file_not_stored_is_not_kept_and_cannot_be_parsed_again(cheshire_cat, local_files):
    row = await cheshire_cat.rabbit_hole.ingest_file(
        cat=cheshire_cat, file=BytesIO(TEXT), metadata={}, filename="story.txt", store_file=False,
        content_type="text/plain",
    )
    assert row["status"] == "completed"
    assert chunks.read_source_bytes(local_files, cheshire_cat.agent_key, None, "story.txt") is None

    await crud_ingestions.reset_phase_entries(cheshire_cat.agent_key, None, "story.txt", ["parsing_chunking"])
    source = IngestionSource(cheshire_cat.agent_key, None, "story.txt")
    async with engine.source_lock(source):
        row = await engine.run_phases(await prepared(cheshire_cat), cheshire_cat, source)

    assert row["status"] == "error" and "was not kept" in row["error"]


# --- faults around the dispatcher and the deletion routes ------------------------------------------------------

async def test_a_redis_blip_while_renewing_the_lock_is_retried(cheshire_cat, monkeypatch):
    from redis.asyncio.lock import Lock

    monkeypatch.setenv("CAT_INGESTION_HEARTBEAT_SECONDS", "1")
    real_extend = Lock.extend
    calls = []

    async def extend(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("redis blip")
        return await real_extend(self, *args, **kwargs)

    monkeypatch.setattr(Lock, "extend", extend)
    async with engine.source_lock(IngestionSource(cheshire_cat.agent_key, None, "x.txt")):
        await asyncio.sleep(2.6)

    assert len(calls) >= 2


async def test_the_row_gone_between_two_phases_stops_and_cleans_up(cheshire_cat, local_files, monkeypatch):
    real_record = crud_ingestions.record_phase

    async def record_then_chat_deleted(agent_id, chat_id, source, phase_id, entry):
        recorded = await real_record(agent_id, chat_id, source, phase_id, entry)
        if phase_id == "parsing_chunking":
            await crud_ingestions.delete_status(agent_id, chat_id, source)
        return recorded

    monkeypatch.setattr(crud_ingestions, "record_phase", record_then_chat_deleted)

    assert await ingest(cheshire_cat) is None
    with pytest.raises(FileNotFoundError):
        chunks.read_chunks(local_files, cheshire_cat.agent_key, None, "story.txt")


async def test_an_interrupted_deletion_is_never_resumed_as_an_ingestion(cheshire_cat, local_files):
    await ingest(cheshire_cat)
    await crud_ingestions.update_status(cheshire_cat.agent_key, None, "story.txt", status="deleting")
    source = IngestionSource(cheshire_cat.agent_key, None, "story.txt")

    async with engine.source_lock(source):
        row = await engine.run_phases(await prepared(cheshire_cat), cheshire_cat, source)

    assert row["status"] == "deleting"


async def test_a_failing_cleanup_goes_on_with_the_other_phases(cheshire_cat, local_files, monkeypatch):
    class Fragile(IngestionPhase):
        id = "fragile"
        depends_on = ("embedding",)

        async def process(self, source, cat):
            await crud_ingestions.delete_status(source.agent_id, source.chat_id, source.name)  # the chat goes
            return PhaseResult(status="done")

        async def delete(self, source, cat):
            raise ConnectionError("graph store down")

    plug(cheshire_cat, monkeypatch, Fragile)

    assert await ingest(cheshire_cat) is None
    assert await points_of(cheshire_cat, "story.txt") == []  # the built-in phases were cleaned up anyway


def test_image_metadata_not_in_json_is_left_out_of_the_chunks(tmp_path):
    files = LocalFileManager()
    chunks.write_chunks(files, "a1", None, "doc.pdf", [], [{"image_bytes": b"i", "image_mime_type": "image/png",
                                                             "pil": object()}])
    _, images = chunks.read_chunks(files, "a1", None, "doc.pdf")
    assert set(images[0]) == {"image_mime_type", "image_bytes", "image_base64"}


async def test_the_deletion_routes_report_a_failure(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    async def broken(self, *args, **kwargs):
        raise RuntimeError("vector db down")

    monkeypatch.setattr(RabbitHole, "delete_source", broken)

    single = await secure_client.delete("/file_manager/files/a.txt", headers=secure_client_headers)
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "a.txt", type_="file")
    every = await secure_client.delete("/file_manager/files", headers=secure_client_headers)
    memory = await secure_client.request(
        "DELETE", "/memory/collections/declarative/points", headers=secure_client_headers, json={"source": "a.txt"},
    )

    assert [r.status_code for r in (single, every, memory)] == [400, 400, 400]


@pytest.mark.parametrize("name, stored_as", [("README", "README"), ("my report.v2.txt", "my_report_v2.txt")])
async def test_the_name_of_an_uploaded_file(cheshire_cat, local_files, name, stored_as):
    row = await ingest(cheshire_cat, name=name)
    assert row["source"] == stored_as and row["status"] == "completed"


async def test_a_file_of_an_invalid_type_is_refused(cheshire_cat):
    assert await cheshire_cat.rabbit_hole.ingest_file(cat=cheshire_cat, file=b"raw", metadata={}, filename="x.txt") is None
    assert await crud_ingestions.list_statuses(cheshire_cat.agent_key) == []


async def test_a_url_canceled_before_its_phases_leaves_no_copy(cheshire_cat, stray, local_files, monkeypatch):
    async def download(self, file, filename, content_type):
        await crud_ingestions.update_status(cheshire_cat.agent_key, stray.id, file, status="downloaded")
        return file, TEXT, "text/plain", True

    real_keep = RabbitHole._keep_source_bytes

    async def keep_then_chat_deleted(self, source, file_bytes, content_type, store_file):
        await real_keep(self, source, file_bytes, content_type, store_file)
        await crud_ingestions.destroy_chat(cheshire_cat.agent_key, stray.id)

    monkeypatch.setattr(RabbitHole, "_resolve_source_bytes", download)
    monkeypatch.setattr(RabbitHole, "_keep_source_bytes", keep_then_chat_deleted)

    assert await cheshire_cat.rabbit_hole.ingest_file(cat=stray, file="https://x.org/page", metadata={}) is None
    assert chunks.read_source_bytes(local_files, cheshire_cat.agent_key, stray.id, "https://x.org/page") is None


async def test_a_notification_failing_does_not_stop_the_failure_handling(cheshire_cat, stray, monkeypatch):
    async def broken(self, message):
        raise ConnectionError("websocket gone")

    monkeypatch.setattr(type(stray.notifier), "send_error", broken)

    row = await ingest(cheshire_cat, data=b"", chat=stray)

    assert row["status"] == "error"


async def run_embedding_alone(cheshire_cat, docs, images, name="story.txt"):
    """The embedding phase of a source whose chunks are stored (as after a restart, or a change of embedder)."""
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, name, type_="file")
    if docs is not None:
        chunks.write_chunks(cheshire_cat.file_manager, cheshire_cat.agent_key, None, name, docs, images)
    rabbit_hole = await prepared(cheshire_cat)
    return await rabbit_hole.run_embedding(IngestionSource(cheshire_cat.agent_key, None, name), cheshire_cat)


async def test_the_embedding_without_stored_chunks_fails(cheshire_cat, local_files):
    result = await run_embedding_alone(cheshire_cat, None, [])
    assert result.status == "error" and "No chunks stored" in result.reason


async def test_the_embedding_of_blank_chunks_fails(cheshire_cat, local_files):
    from langchain_core.documents import Document
    result = await run_embedding_alone(cheshire_cat, [Document(page_content="  \n ")], [])
    assert result.status == "error" and "Nothing to embed" in result.reason


async def test_the_embedding_reads_back_the_images_and_the_source_bytes(cheshire_cat, local_files, monkeypatch):
    from langchain_core.documents import Document
    seen = {}
    original = type(cheshire_cat.plugin_manager).execute_hook

    async def execute_hook(self, name, *args, caller):
        if name == "rabbithole_stores_image_points":
            seen["images"], seen["source_bytes"] = args[1], args[3]
        return await original(self, name, *args, caller=caller)

    monkeypatch.setattr(type(cheshire_cat.plugin_manager), "execute_hook", execute_hook)
    local_files.write_file(b"the source", "story.txt", cheshire_cat.agent_key)

    result = await run_embedding_alone(
        cheshire_cat, [Document(page_content="text")], [{"image_bytes": b"img", "image_mime_type": "image/png"}],
    )

    assert result.status == "done"
    assert seen["images"][0]["image_bytes"] == b"img" and seen["source_bytes"] == b"the source"


async def test_a_fresh_source_without_stored_points_is_processed_again(cheshire_cat, local_files):
    row = await ingest(cheshire_cat)
    assert row["status"] == "completed"
    assert await points_of(cheshire_cat, "story.txt")

    await cheshire_cat.vector_memory_handler.delete_tenant_points(
        str(VectorMemoryType.DECLARATIVE), {"source": "story.txt"},
    )
    assert await points_of(cheshire_cat, "story.txt") == []

    row = await cheshire_cat.rabbit_hole.rerun_source(cheshire_cat, "story.txt")

    assert row["status"] == "completed"
    assert await points_of(cheshire_cat, "story.txt")
