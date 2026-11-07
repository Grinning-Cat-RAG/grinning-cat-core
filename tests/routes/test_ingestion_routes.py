"""The /ingestion endpoints and the maintenance API of the RabbitHole (reingest_source, rerun_agent)."""
from io import BytesIO

import pytest

from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
from cat.db.cruds import conversations as crud_conversations
from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds import users as crud_users
from cat.looking_glass import CheshireCat
from cat.rabbit_hole import RabbitHole, SourceNotKept, chunks, engine
from cat.rabbit_hole.phases import IngestionPhase, IngestionSource, PhaseResult

TEXT = ("Begin at the beginning, the King said gravely, and go on till you come to the end. " * 30).encode()


@pytest.fixture
def files(monkeypatch):
    """Every CheshireCat of the test keeps its files (the default file manager of the tests keeps none)."""
    local = LocalFileManager()
    monkeypatch.setattr(CheshireCat, "file_manager", property(lambda self: local, lambda self, value: None))
    return local


def plug(monkeypatch, cheshire_cat, *phases):
    manager_class = type(cheshire_cat.plugin_manager)
    original = manager_class.execute_hook

    async def execute_hook(self, name, *args, caller):
        if name == "rabbithole_ingestion_phases":
            return list(args[0]) + list(phases)
        return await original(self, name, *args, caller=caller)

    monkeypatch.setattr(manager_class, "execute_hook", execute_hook)


class GraphPhase(IngestionPhase):
    id = "graph"
    label = "Building the graph"
    depends_on = ("embedding",)
    outcomes: list = []

    async def process(self, source, cat):
        return PhaseResult(status=GraphPhase.outcomes.pop(0) if GraphPhase.outcomes else "done")

    async def delete(self, source, cat):
        pass


async def ingest(cat, name="story.txt", data=TEXT, store_file=True):
    return await cat.rabbit_hole.ingest_file(
        cat=cat, file=BytesIO(data), metadata={"tag": "t"}, filename=name, content_type="text/plain",
        store_file=store_file,
    )


# --- GET /ingestion/status -------------------------------------------------------------------------------------

async def test_the_states_of_the_knowledge_base_with_their_labels(secure_client, secure_client_headers, cheshire_cat, files):
    await ingest(cheshire_cat, "a.txt")
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "b.txt", type_="file")
    await crud_ingestions.update_status(cheshire_cat.agent_key, None, "b.txt", status="processing", phase="embedding")
    await crud_ingestions.create_status(cheshire_cat.agent_key, "chat-1", "chat.txt", type_="file")

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)

    assert response.status_code == 200
    rows = {r["source"]: r for r in response.json()}
    assert sorted(rows) == ["a.txt", "b.txt"]  # not the chat
    assert (rows["a.txt"]["status"], rows["a.txt"]["status_label"], rows["a.txt"]["phase_label"]) == (
        "completed", "Completed", None,
    )
    assert (rows["b.txt"]["phase"], rows["b.txt"]["phase_label"]) == ("embedding", "Computing embeddings")
    assert rows["a.txt"]["metadata"] == {"tag": "t"} and set(rows["a.txt"]["completed_phases"]) == {
        "parsing_chunking", "embedding",
    }
    assert "owner" not in rows["a.txt"]


async def test_the_states_of_the_chat_of_the_request(secure_client, secure_client_headers, cheshire_cat):
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "kb.txt", type_="file")
    await crud_ingestions.create_status(cheshire_cat.agent_key, "chat-1", "chat.txt", type_="file")

    response = await secure_client.get("/ingestion/status", headers=secure_client_headers | {"X-Chat-ID": "chat-1"})

    assert [r["source"] for r in response.json()] == ["chat.txt"]


async def test_no_states(secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.get("/ingestion/status", headers=secure_client_headers)
    assert (response.status_code, response.json()) == (200, [])


# --- GET /ingestion/phases -------------------------------------------------------------------------------------

async def test_the_phases_in_execution_order_and_the_states(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    plug(monkeypatch, cheshire_cat, GraphPhase)

    response = await secure_client.get("/ingestion/phases", headers=secure_client_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["phases"] == [
        {"id": "parsing_chunking", "label": "Parsing and chunking", "order": 1, "depends_on": []},
        {"id": "embedding", "label": "Computing embeddings", "order": 2, "depends_on": ["parsing_chunking"]},
        {"id": "graph", "label": "Building the graph", "order": 3, "depends_on": ["embedding"]},
    ]
    assert [s["id"] for s in body["states"]] == [
        "uploaded", "downloading", "downloaded", "processing", "completed", "error", "deleting",
    ]


async def test_a_phase_without_label_shows_its_id(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    class Unlabeled(GraphPhase):
        id = "concept_relations"
        label = ""

    plug(monkeypatch, cheshire_cat, Unlabeled)

    phases = (await secure_client.get("/ingestion/phases", headers=secure_client_headers)).json()["phases"]

    assert phases[-1]["label"] == "Concept relations"


async def test_plugin_phases_forming_a_cycle_are_reported(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    class A(GraphPhase):
        id = "a"
        depends_on = ("b",)

    class B(GraphPhase):
        id = "b"
        depends_on = ("a",)

    plug(monkeypatch, cheshire_cat, A, B)

    for path in ("/ingestion/phases", "/ingestion/status"):
        response = await secure_client.get(path, headers=secure_client_headers)
        assert response.status_code == 400 and "cycle" in response.json()["detail"]


# --- DELETE /ingestion/status ----------------------------------------------------------------------------------

async def test_a_failed_ingestion_is_dismissed_with_its_kept_files_only(
    secure_client, secure_client_headers, cheshire_cat, files,
):
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "https://x.org/page", type_="url")
    await crud_ingestions.update_status(cheshire_cat.agent_key, None, "https://x.org/page", status="error", error="boom")
    chunks.write_source_bytes(files, cheshire_cat.agent_key, None, "https://x.org/page", b"html")

    response = await secure_client.delete(
        "/ingestion/status", params={"source": "https://x.org/page"}, headers=secure_client_headers,
    )

    assert (response.status_code, response.json()) == (200, {"deleted": True})
    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "https://x.org/page") is None
    assert chunks.read_source_bytes(files, cheshire_cat.agent_key, None, "https://x.org/page") is None


async def test_a_completed_ingestion_cannot_be_dismissed(secure_client, secure_client_headers, cheshire_cat, files):
    await ingest(cheshire_cat)

    response = await secure_client.delete("/ingestion/status", params={"source": "story.txt"}, headers=secure_client_headers)

    assert response.status_code == 409 and "delete the file" in response.json()["detail"]
    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") is not None


async def test_dismissing_an_unknown_source_is_not_found(secure_client, secure_client_headers, cheshire_cat):
    response = await secure_client.delete("/ingestion/status", params={"source": "x.txt"}, headers=secure_client_headers)
    assert response.status_code == 404


async def test_dismissing_a_source_being_ingested_is_refused(
    secure_client, secure_client_headers, cheshire_cat, monkeypatch,
):
    monkeypatch.setattr("cat.routes.ingestion.DELETE_WAIT_SECONDS", 0.2)
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "x.txt", type_="file")
    await crud_ingestions.update_status(cheshire_cat.agent_key, None, "x.txt", status="error")

    async with engine.source_lock(IngestionSource(cheshire_cat.agent_key, None, "x.txt")):
        response = await secure_client.delete("/ingestion/status", params={"source": "x.txt"}, headers=secure_client_headers)

    assert response.status_code == 409 and "being ingested" in response.json()["detail"]


async def test_a_row_gone_while_waiting_is_not_found(secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "x.txt", type_="file")
    real_lock = engine.source_lock

    def lock_then_gone(source, blocking=True, wait=None):
        async def gone():
            await crud_ingestions.delete_status(source.agent_id, source.chat_id, source.name)  # e.g. the file deleted
        context = real_lock(source, blocking, wait)

        class Wrapped:
            async def __aenter__(self):
                lock = await context.__aenter__()
                await gone()
                return lock

            async def __aexit__(self, *exc):
                return await context.__aexit__(*exc)
        return Wrapped()

    monkeypatch.setattr(engine, "source_lock", lock_then_gone)

    response = await secure_client.delete("/ingestion/status", params={"source": "x.txt"}, headers=secure_client_headers)

    assert response.status_code == 404


# --- maintenance: reingest_source, rerun_agent -----------------------------------------------------------------

async def test_reingesting_a_source_runs_every_phase_again(cheshire_cat, files, monkeypatch):
    parsed = []
    real = RabbitHole._parse_to_docs

    async def counting(self, source, file_bytes, content_type):
        parsed.append(source)
        return await real(self, source, file_bytes, content_type)

    await ingest(cheshire_cat)
    monkeypatch.setattr(RabbitHole, "_parse_to_docs", counting)

    row = await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, "story.txt")

    assert parsed == ["story.txt"] and row["status"] == "completed"


async def test_reingesting_a_source_without_state_gives_it_one(cheshire_cat, files):
    files.write_file(TEXT, "legacy.txt", cheshire_cat.agent_key)

    row = await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, "legacy.txt")

    assert row["status"] == "completed" and row["type"] == "file"


async def test_reingesting_a_file_not_kept_touches_nothing(cheshire_cat, files):
    await ingest(cheshire_cat, store_file=False)
    before = await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt")

    with pytest.raises(SourceNotKept):
        await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, "story.txt")

    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") == before


async def test_reingesting_a_url_downloads_it_again_when_not_kept(cheshire_cat, files, monkeypatch):
    async def download(self, file, filename, content_type):
        return file, TEXT, "text/plain", True

    monkeypatch.setattr(RabbitHole, "_resolve_source_bytes", download)

    row = await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, "https://x.org/page")

    assert row["status"] == "completed" and row["type"] == "url"


async def test_reingesting_a_source_being_deleted_completes_the_deletion(cheshire_cat, files):
    await ingest(cheshire_cat)
    await crud_ingestions.update_status(cheshire_cat.agent_key, None, "story.txt", status="deleting")

    assert await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, "story.txt") is None
    assert await crud_ingestions.get_status(cheshire_cat.agent_key, None, "story.txt") is None


async def test_reingesting_a_chat_source_keeps_it_in_its_chat(cheshire_cat, files):
    user = await crud_users.create_user(cheshire_cat.agent_key, {"username": "alice", "password": "alice-password"})
    await crud_conversations.claim_conversation(cheshire_cat.agent_key, "chat-1", user["id"])
    stray = await cheshire_cat._find_stray_cat("chat-1")
    await ingest(stray)

    row = await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, "story.txt", "chat-1")

    assert row["status"] == "completed" and row["chat_id"] == "chat-1"


async def test_reingesting_a_source_whose_agent_is_deleted_meanwhile(cheshire_cat, files, monkeypatch):
    await ingest(cheshire_cat)

    async def agent_gone(agent_id):
        return False

    monkeypatch.setattr(engine, "agent_exists", agent_gone)

    assert await cheshire_cat.rabbit_hole.reingest_source(cheshire_cat, "story.txt") is None


async def test_rerunning_an_agent_retries_the_failed_phases_on_request(cheshire_cat, files, monkeypatch):
    plug(monkeypatch, cheshire_cat, GraphPhase)
    GraphPhase.outcomes = ["error"]
    await ingest(cheshire_cat, "a.txt")  # the graph fails
    await ingest(cheshire_cat, "b.txt")
    await crud_ingestions.create_status(cheshire_cat.agent_key, None, "gone.txt", type_="file")
    await crud_ingestions.update_status(cheshire_cat.agent_key, None, "gone.txt", status="deleting")

    plain = await cheshire_cat.rabbit_hole.rerun_agent(cheshire_cat)
    retried = await cheshire_cat.rabbit_hole.rerun_agent(cheshire_cat, retry_errors=True)

    assert plain == {"completed": 1, "error": 1, "skipped": 1}  # a failure stays, the deletion completes
    assert retried == {"completed": 2, "error": 0, "skipped": 0}
