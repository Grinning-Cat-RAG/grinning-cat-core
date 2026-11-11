"""The registry of the ingestion states (``cat.db.cruds.ingestions``), on the test Redis database.

One row per source, in the knowledge base of the agent (``chat_id=None``) or in a chat.
"""
import asyncio
import hashlib
from unittest import mock

import pytest

from cat.db import crud, models
from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds import plugins as crud_plugins
from cat.db.cruds import settings as crud_settings
from cat.db.cruds.ingestions import IngestionStatus
from cat.db.database import get_async_db

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, settings, strategies as st  # noqa: E402
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test  # noqa: E402

KB = None  # the knowledge base of the agent


async def _create(agent="a1", chat_id=KB, source="doc.pdf", type_="file"):
    return await crud_ingestions.create_status(agent, chat_id, source, type_=type_)


def _sources(rows):
    return sorted(r["source"] for r in rows)


def test_key_layout():
    sha = hashlib.sha256("doc.pdf".encode()).hexdigest()
    assert crud_ingestions.format_key("a1", KB, "doc.pdf") == f"agents:a1:ingestion:agent:{sha}"
    assert crud_ingestions.format_key("a1", "c1", "doc.pdf") == f"agents:a1:ingestion:chat:c1:{sha}"


async def test_a_created_row_is_read_back_with_an_empty_diary():
    created = await _create()
    stored = await crud_ingestions.get_status("a1", KB, "doc.pdf")

    assert stored == created
    assert stored["source"] == "doc.pdf"
    assert stored["chat_id"] is None
    assert stored["type"] == "file"
    assert stored["status"] == IngestionStatus.UPLOADED.value
    assert stored["completed_phases"] == {}
    assert stored["created_at"] == stored["updated_at"]


async def test_the_times_survive_the_round_trip_through_redis(monkeypatch):
    # RedisJSON does not give back every digit of this float
    monkeypatch.setattr(crud_ingestions.time, "time", lambda: 1794068713.2781823)

    created = await _create()

    assert await crud_ingestions.get_status("a1", KB, "doc.pdf") == created


async def test_only_files_and_urls_are_sources():
    with pytest.raises(ValueError):
        await _create(type_="folder")
    assert await crud_ingestions.get_status("a1", KB, "doc.pdf") is None


async def test_creating_again_starts_a_new_lifecycle():
    await _create()
    await crud_ingestions.record_phase("a1", KB, "doc.pdf", "embedding", {"state": "complete"})

    await _create()

    assert (await crud_ingestions.get_status("a1", KB, "doc.pdf"))["completed_phases"] == {}


async def test_a_missing_row_is_none():
    assert await crud_ingestions.get_status("a1", KB, "missing.pdf") is None


async def test_update_changes_the_fields_and_the_update_time():
    created = await _create()
    await asyncio.sleep(0.01)

    assert await crud_ingestions.update_status(
        "a1", KB, "doc.pdf", status=IngestionStatus.PROCESSING, phase="embedding",
    )
    stored = await crud_ingestions.get_status("a1", KB, "doc.pdf")

    assert stored["status"] == "processing"
    assert stored["phase"] == "embedding"
    assert stored["created_at"] == created["created_at"]
    assert stored["updated_at"] > created["updated_at"]


async def test_update_of_fields_other_than_the_status_keeps_the_status():
    await _create()
    assert await crud_ingestions.update_status("a1", KB, "doc.pdf", error="boom", owner={"pod": "p1"})
    stored = await crud_ingestions.get_status("a1", KB, "doc.pdf")
    assert (stored["status"], stored["error"], stored["owner"]) == ("uploaded", "boom", {"pod": "p1"})


async def test_touch_refreshes_only_the_update_time():
    created = await _create()
    await asyncio.sleep(0.01)

    assert await crud_ingestions.touch_status("a1", KB, "doc.pdf")
    stored = await crud_ingestions.get_status("a1", KB, "doc.pdf")

    assert stored["updated_at"] > created["updated_at"]
    assert {k: v for k, v in stored.items() if k != "updated_at"} == {
        k: v for k, v in created.items() if k != "updated_at"
    }
    assert not await crud_ingestions.touch_status("a1", KB, "missing.pdf")


async def test_update_never_creates_a_row():
    assert not await crud_ingestions.update_status("a1", KB, "doc.pdf", status=IngestionStatus.ERROR)
    assert await crud_ingestions.get_status("a1", KB, "doc.pdf") is None


async def test_update_refuses_unknown_fields_and_statuses():
    await _create()
    with pytest.raises(ValueError):
        await crud_ingestions.update_status("a1", KB, "doc.pdf", completed_phases={})
    with pytest.raises(ValueError):
        await crud_ingestions.update_status("a1", KB, "doc.pdf", status="complete")
    assert (await crud_ingestions.get_status("a1", KB, "doc.pdf"))["status"] == "uploaded"


async def test_statuses_may_be_given_as_text():
    await _create()
    assert await crud_ingestions.update_status("a1", KB, "doc.pdf", status="error")
    assert (await crud_ingestions.get_status("a1", KB, "doc.pdf"))["status"] == "error"


async def test_the_diary_records_and_resets_single_phases():
    await _create()
    entry = {"state": "complete", "tag": "t1"}

    assert await crud_ingestions.record_phase("a1", KB, "doc.pdf", "parsing_chunking", entry)
    assert await crud_ingestions.record_phase("a1", KB, "doc.pdf", "embedding", entry)
    assert await crud_ingestions.reset_phase_entries("a1", KB, "doc.pdf", ["embedding", "never_ran"])

    stored = await crud_ingestions.get_status("a1", KB, "doc.pdf")
    assert stored["completed_phases"] == {"parsing_chunking": entry}


async def test_the_diary_of_a_missing_row_is_not_written():
    assert not await crud_ingestions.record_phase("a1", KB, "doc.pdf", "embedding", {"state": "complete"})
    assert not await crud_ingestions.reset_phase_entries("a1", KB, "doc.pdf", ["embedding"])
    assert await crud_ingestions.get_status("a1", KB, "doc.pdf") is None


@pytest.mark.parametrize("phase_id", ["", "Embedding", "graph.llm", "a b", 'x"]', "9phase", None])
async def test_phase_ids_are_plain_identifiers(phase_id):
    await _create()
    with pytest.raises(ValueError):
        await crud_ingestions.record_phase("a1", KB, "doc.pdf", phase_id, {"state": "complete"})
    with pytest.raises(ValueError):
        await crud_ingestions.reset_phase_entries("a1", KB, "doc.pdf", [phase_id])


async def test_listing_the_knowledge_base_a_chat_or_everything():
    await _create(source="a.pdf")
    await _create(chat_id="c1", source="b.pdf")
    await _create(chat_id="c1:2", source="c.pdf")
    await _create(agent="a2", source="other.pdf")

    assert _sources(await crud_ingestions.list_statuses("a1")) == ["a.pdf", "b.pdf", "c.pdf"]
    assert _sources(await crud_ingestions.list_statuses("a1", KB)) == ["a.pdf"]
    assert _sources(await crud_ingestions.list_statuses("a1", "c1")) == ["b.pdf"]
    assert await crud_ingestions.list_statuses("a3") == []


async def test_a_chat_named_agent_is_not_the_knowledge_base():
    await _create(source="kb.pdf")
    await _create(chat_id="agent", source="chat.pdf")

    assert _sources(await crud_ingestions.list_statuses("a1", KB)) == ["kb.pdf"]
    assert _sources(await crud_ingestions.list_statuses("a1", "agent")) == ["chat.pdf"]

    await crud_ingestions.destroy_chat("a1", "agent")
    assert _sources(await crud_ingestions.list_statuses("a1")) == ["kb.pdf"]


async def test_glob_characters_in_the_ids_are_literal():
    await _create(agent="a*", source="star.pdf")
    await _create(agent="a1", source="plain.pdf")
    await _create(agent="a1", chat_id="c?", source="q.pdf")
    await _create(agent="a1", chat_id="cx", source="x.pdf")

    assert _sources(await crud_ingestions.list_statuses("a*")) == ["star.pdf"]
    assert _sources(await crud_ingestions.list_statuses("a1", "c?")) == ["q.pdf"]

    await crud_ingestions.destroy_chat("a1", "c?")
    await crud_ingestions.destroy_all("a*")
    assert _sources(await crud_ingestions.list_statuses("a1")) == ["plain.pdf", "x.pdf"]


async def test_keys_that_are_not_rows_are_not_listed():
    await _create(source="a.pdf")
    db = get_async_db()
    for key in ("agents:a1:ingestion:delete", "agents:a1:ingestion:agent:aaa", "agents:a1:ingestion:chat::" + "0" * 64):
        await db.set(key, "x")

    assert _sources(await crud_ingestions.list_statuses("a1")) == ["a.pdf"]


async def test_resetting_phases_across_an_agent_counts_the_rows_changed():
    for source in ("a.pdf", "b.pdf"):
        await _create(source=source)
    await _create(chat_id="c1", source="c.pdf")
    await crud_ingestions.record_phase("a1", KB, "a.pdf", "graph_llm", {"state": "complete"})
    await crud_ingestions.record_phase("a1", KB, "b.pdf", "embedding", {"state": "complete"})
    await crud_ingestions.record_phase("a1", "c1", "c.pdf", "graph_llm", {"state": "error"})

    assert await crud_ingestions.reset_phase_entries_for_agent("a1", ["graph_llm"]) == 2

    rows = {r["source"]: r["completed_phases"] for r in await crud_ingestions.list_statuses("a1")}
    assert rows == {"a.pdf": {}, "b.pdf": {"embedding": {"state": "complete"}}, "c.pdf": {}}


async def test_deleting_a_row():
    await _create()
    await crud_ingestions.delete_status("a1", KB, "doc.pdf")
    await crud_ingestions.delete_status("a1", KB, "doc.pdf")  # idempotent
    assert await crud_ingestions.get_status("a1", KB, "doc.pdf") is None


async def test_destroying_an_agent_removes_its_whole_namespace_only():
    await _create(source="a.pdf")
    await _create(chat_id="c1", source="b.pdf")
    await crud_ingestions.set_maintenance("a1", ttl=60)
    await crud_ingestions.start_vector_transfer("a1")
    await get_async_db().set("agents:a1:ingestion:delete", "left by an older version")
    await _create(agent="a2", source="other.pdf")
    await crud_plugins.set_setting("a1", "cat_multimodal_ingestion", {"k": 1})

    await crud_ingestions.destroy_all("a1")

    assert [k async for k in get_async_db().scan_iter("agents:a1:ingestion*")] == []
    assert _sources(await crud_ingestions.list_statuses("a2")) == ["other.pdf"]
    assert await crud_plugins.get_setting("a1", "cat_multimodal_ingestion") == {"k": 1}


async def test_destroying_a_chat_keeps_the_other_scopes():
    await _create(source="a.pdf")
    await _create(chat_id="c1", source="b.pdf")
    await _create(chat_id="c1:2", source="c.pdf")

    assert await crud_ingestions.destroy_chat("a1", "c1") == 1

    assert _sources(await crud_ingestions.list_statuses("a1")) == ["a.pdf", "c.pdf"]


async def test_maintenance_marker_expires_and_is_cleared():
    assert not await crud_ingestions.is_in_maintenance("a1")

    await crud_ingestions.set_maintenance("a1", ttl=60)
    assert await crud_ingestions.is_in_maintenance("a1")
    assert 0 < await get_async_db().ttl(crud_ingestions.maintenance_key("a1")) <= 60
    assert await crud_ingestions.list_statuses("a1") == []  # not a row

    await crud_ingestions.clear_maintenance("a1")
    assert not await crud_ingestions.is_in_maintenance("a1")


@pytest.mark.parametrize("ttl", [0, -5])
async def test_a_maintenance_marker_must_expire(ttl):
    with pytest.raises(ValueError):
        await crud_ingestions.set_maintenance("a1", ttl=ttl)
    assert not await crud_ingestions.is_in_maintenance("a1")


# --- cloning an agent ------------------------------------------------------------------------------------------

async def test_cloning_an_agent_skips_its_ingestion_rows_and_keeps_every_other_key_once():
    await crud_settings.create_setting("src", models.Setting(name="x", value={}, category="llm"))
    await crud_plugins.set_setting("src", "cat_multimodal_ingestion", {"k": 1})
    await crud_plugins.set_setting("src", "conversation_history", {"k": 2})
    await _create(agent="src", source="a.pdf")
    await _create(agent="src", chat_id="c1", source="b.pdf")
    await crud_ingestions.set_maintenance("src", ttl=60)

    cloned = await crud_settings.clone_agent("src", "dst", ["conversations", "ingestion"])

    assert cloned == 3
    assert await crud_plugins.get_setting("dst", "cat_multimodal_ingestion") == {"k": 1}
    assert await crud_plugins.get_setting("dst", "conversation_history") == {"k": 2}
    assert await crud_ingestions.list_statuses("dst") == []
    assert not await crud_ingestions.is_in_maintenance("dst")


async def test_cloning_an_agent_without_skipped_keys_clones_every_key():
    await crud_plugins.set_setting("src", "p1", {"k": 1})
    await crud_plugins.set_setting("src", "p2", {"k": 2})

    assert await crud_settings.clone_agent("src", "dst") == 2
    assert await crud_plugins.get_setting("dst", "p2") == {"k": 2}


@pytest.mark.parametrize("source_agent", ["agent", "a", "age*", "s"])
async def test_cloning_an_agent_whose_id_is_part_of_the_key_names(source_agent):
    await crud_plugins.set_setting(source_agent, "p1", {"k": 1})
    await crud_plugins.set_setting("age-other", "p1", {"k": "other"})

    assert await crud_settings.clone_agent(source_agent, "dst") == 1

    assert await crud_plugins.get_setting("dst", "p1") == {"k": 1}
    assert sorted([k async for k in get_async_db().scan_iter("*")]) == sorted([
        f"agents:{source_agent}:plugins:p1", "agents:age-other:plugins:p1", "agents:dst:plugins:p1",
    ])


async def test_cloning_copies_the_finished_rows_of_the_knowledge_base_only():
    for source, status in (("done.pdf", "completed"), ("failed.pdf", "error"), ("running.pdf", "processing")):
        await _create(agent="src", source=source)
        await crud_ingestions.update_status("src", KB, source, status=status)
    await crud_ingestions.record_phase("src", KB, "done.pdf", "embedding", {"state": "complete"})
    await _create(agent="src", chat_id="c1", source="chat.pdf")
    await crud_ingestions.update_status("src", "c1", "chat.pdf", status="completed")

    assert await crud_ingestions.clone_knowledge_base("src", "dst") == 2

    rows = {r["source"]: r for r in await crud_ingestions.list_statuses("dst")}
    assert sorted(rows) == ["done.pdf", "failed.pdf"]
    assert rows["done.pdf"] == await crud_ingestions.get_status("src", KB, "done.pdf")


async def test_cloning_a_cheshire_cat_brings_its_finished_knowledge_base_rows(lizard, cheshire_cat):
    await _create(agent=cheshire_cat.agent_key, source="done.pdf")
    await crud_ingestions.update_status(cheshire_cat.agent_key, KB, "done.pdf", status="completed")

    clone = await lizard.clone_cheshire_cat(cheshire_cat, "clone-agent")

    assert _sources(await crud_ingestions.list_statuses(clone.agent_key)) == ["done.pdf"]


# --- the agent and the chats take their rows with them ---------------------------------------------------------

async def test_destroying_the_agent_removes_its_ingestion_state(cheshire_cat, monkeypatch):
    # The seeded rows are non-terminal ("uploaded"), so the teardown would wait
    # the default 180s quiesce. Bound it (and the poll) to keep the test fast.
    monkeypatch.setenv("CAT_INGESTION_DELETE_QUIESCE_SECONDS", "1")
    monkeypatch.setenv("CAT_INGESTION_DELETE_POLL_SECONDS", "1")

    agent = cheshire_cat.agent_key
    await _create(agent=agent, source="a.pdf")
    await _create(agent=agent, chat_id="c1", source="b.pdf")
    await crud_ingestions.set_maintenance(agent, ttl=60)

    await cheshire_cat.destroy()

    assert await crud_ingestions.list_statuses(agent) == []
    assert not await crud_ingestions.is_in_maintenance(agent)


async def test_destroying_a_conversation_removes_only_its_ingestion_state(cheshire_cat):
    agent = cheshire_cat.agent_key
    await _create(agent=agent, source="a.pdf")
    await _create(agent=agent, chat_id="c1", source="b.pdf")
    await _create(agent=agent, chat_id="c2", source="c.pdf")

    await cheshire_cat.destroy_conversation("c1", "user-1")

    assert _sources(await crud_ingestions.list_statuses(agent)) == ["a.pdf", "c.pdf"]


# --- stateful: the registry is the model, also when Redis fails in the middle -----------------------------------

AGENTS = ("a1", "a*")
CHATS = (KB, "c1", "c1:2", "c?", "agent")
SOURCES = ("doc.pdf", "https://x.org/a?b", "ü.txt")
PHASES = ("parsing_chunking", "embedding", "graph_llm")
LOOP = None


class LostReplyRedis:
    """The Redis client whose writes, while ``fail`` is set, are applied but whose reply is lost (the network drops
    after the command): the caller sees ``ConnectionError`` without knowing whether the write happened."""

    def __init__(self, db):
        self.db = db
        self.fail = False

    def __getattr__(self, name):
        attribute = getattr(self.db, name)
        if not self.fail or name not in ("pipeline", "delete"):
            return attribute
        if name == "delete":
            async def delete_then_lose_the_reply(*args, **kwargs):
                await attribute(*args, **kwargs)
                raise ConnectionError("reply lost")
            return delete_then_lose_the_reply

        def pipeline_losing_the_reply(*args, **kwargs):
            pipeline = attribute(*args, **kwargs)
            execute = pipeline.execute

            async def execute_then_lose_the_reply(*a, **kw):
                await execute(*a, **kw)
                raise ConnectionError("reply lost")

            pipeline.execute = execute_then_lose_the_reply
            return pipeline
        return pipeline_losing_the_reply


def build_machine(db: LostReplyRedis):
    class RegistryMachine(RuleBasedStateMachine):
        """The model is exact after a write whose reply arrived; after a lost reply, the key is reconciled with what
        Redis holds, which must be either the state before or the state after the write."""

        def __init__(self):
            super().__init__()
            LOOP.run_until_complete(self._wipe())
            self.model = {}  # (agent, chat_id, source) -> {"status": str, "phases": set}
            self.maintenance = set()

        async def _wipe(self):
            keys = [k async for k in db.db.scan_iter("agents:*")]
            if keys:
                await db.db.delete(*keys)

        def call(self, coroutine_factory, fail):
            db.fail = fail
            try:
                return LOOP.run_until_complete(coroutine_factory()), True
            except ConnectionError:
                return None, False
            finally:
                db.fail = False

        def stored(self, key):
            row = LOOP.run_until_complete(crud_ingestions.get_status(*key))
            return None if row is None else {"status": row["status"], "phases": set(row["completed_phases"])}

        def reconcile(self, key, before, after):
            now = self.stored(key)
            assert now in (before, after), (key, before, after, now)
            if now is None:
                self.model.pop(key, None)
            else:
                self.model[key] = now

        @rule(agent=st.sampled_from(AGENTS), chat=st.sampled_from(CHATS), source=st.sampled_from(SOURCES),
              fail=st.booleans())
        def create(self, agent, chat, source, fail):
            key = (agent, chat, source)
            before = self.model.get(key)
            _, ok = self.call(lambda: crud_ingestions.create_status(agent, chat, source, type_="file"), fail)
            after = {"status": "uploaded", "phases": set()}
            if ok:
                self.model[key] = after
            else:
                self.reconcile(key, before, after)

        @rule(agent=st.sampled_from(AGENTS), chat=st.sampled_from(CHATS), source=st.sampled_from(SOURCES),
              status=st.sampled_from(list(IngestionStatus)), fail=st.booleans())
        def update(self, agent, chat, source, status, fail):
            key = (agent, chat, source)
            before = self.model.get(key)
            updated, ok = self.call(lambda: crud_ingestions.update_status(agent, chat, source, status=status), fail)
            after = None if before is None else {**before, "status": status.value}
            if ok:
                assert updated == (before is not None)
                if updated:
                    self.model[key] = after
            else:
                self.reconcile(key, before, after)

        @rule(agent=st.sampled_from(AGENTS), chat=st.sampled_from(CHATS), source=st.sampled_from(SOURCES),
              phase=st.sampled_from(PHASES), fail=st.booleans())
        def record(self, agent, chat, source, phase, fail):
            key = (agent, chat, source)
            before = self.model.get(key)
            recorded, ok = self.call(
                lambda: crud_ingestions.record_phase(agent, chat, source, phase, {"state": "complete"}), fail,
            )
            after = None if before is None else {**before, "phases": before["phases"] | {phase}}
            if ok:
                assert recorded == (before is not None)
                if recorded:
                    self.model[key] = after
            else:
                self.reconcile(key, before, after)

        @rule(agent=st.sampled_from(AGENTS), phases=st.sets(st.sampled_from(PHASES), min_size=1), fail=st.booleans())
        def reset_for_agent(self, agent, phases, fail):
            touched = [k for k in self.model if k[0] == agent and self.model[k]["phases"] & phases]
            count, ok = self.call(lambda: crud_ingestions.reset_phase_entries_for_agent(agent, sorted(phases)), fail)
            if ok:
                assert count == len(touched)
            for key in touched:  # each row is reset by its own write: all, some (a lost reply stops the loop) or none
                before = self.model[key]
                after = {**before, "phases": before["phases"] - phases}
                if ok:
                    self.model[key] = after
                else:
                    self.reconcile(key, before, after)

        @rule(agent=st.sampled_from(AGENTS), chat=st.sampled_from(CHATS), source=st.sampled_from(SOURCES),
              fail=st.booleans())
        def delete(self, agent, chat, source, fail):
            key = (agent, chat, source)
            before = self.model.get(key)
            _, ok = self.call(lambda: crud_ingestions.delete_status(agent, chat, source), fail)
            if ok:
                self.model.pop(key, None)
            else:
                self.reconcile(key, before, None)

        @rule(agent=st.sampled_from(AGENTS), chat=st.sampled_from(CHATS[1:]), fail=st.booleans())
        def destroy_chat(self, agent, chat, fail):
            gone = [k for k in self.model if k[0] == agent and k[1] == chat]
            _, ok = self.call(lambda: crud_ingestions.destroy_chat(agent, chat), fail)
            for key in gone:
                if ok:
                    self.model.pop(key)
                else:
                    self.reconcile(key, self.model[key], None)

        @rule(agent=st.sampled_from(AGENTS), fail=st.booleans())
        def destroy_agent(self, agent, fail):
            gone = [k for k in self.model if k[0] == agent]
            _, ok = self.call(lambda: crud_ingestions.destroy_all(agent), fail)
            for key in gone:
                if ok:
                    self.model.pop(key)
                else:
                    self.reconcile(key, self.model[key], None)
            if ok:
                self.maintenance.discard(agent)
            elif not LOOP.run_until_complete(crud_ingestions.is_in_maintenance(agent)):
                self.maintenance.discard(agent)

        @rule(agent=st.sampled_from(AGENTS), on=st.booleans(), fail=st.booleans())
        def maintenance(self, agent, on, fail):
            if on:
                _, ok = self.call(lambda: crud_ingestions.set_maintenance(agent, ttl=600), fail)
            else:
                _, ok = self.call(lambda: crud_ingestions.clear_maintenance(agent), fail)
            if ok or LOOP.run_until_complete(crud_ingestions.is_in_maintenance(agent)) == on:
                (self.maintenance.add if on else self.maintenance.discard)(agent)

        @invariant()
        def the_registry_is_the_model(self):
            async def read():
                listed = {}
                for agent in AGENTS:
                    for row in await crud_ingestions.list_statuses(agent):
                        listed[(agent, row["chat_id"], row["source"])] = {
                            "status": row["status"], "phases": set(row["completed_phases"]),
                        }
                    for chat in CHATS:
                        in_scope = {r["source"] for r in await crud_ingestions.list_statuses(agent, chat)}
                        assert in_scope == {k[2] for k in self.model if k[0] == agent and k[1] == chat}
                    assert await crud_ingestions.is_in_maintenance(agent) == (agent in self.maintenance)
                return listed

            assert LOOP.run_until_complete(read()) == self.model

    RegistryMachine.TestCase.settings = settings(
        max_examples=40, stateful_step_count=20, deadline=None, database=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    return RegistryMachine


def test_registry_state_machine():
    import redis.asyncio as aioredis
    from cat.db.database import get_redis_kwargs

    global LOOP
    LOOP = asyncio.new_event_loop()
    try:
        db = LostReplyRedis(aioredis.Redis(**get_redis_kwargs()))
        machine = build_machine(db)
        with mock.patch.object(crud_ingestions, "get_async_db", lambda: db), \
                mock.patch.object(crud, "get_async_db", lambda: db):
            run_state_machine_as_test(machine, settings=machine.TestCase.settings)
        LOOP.run_until_complete(db.db.aclose())
    finally:
        LOOP.close()
        LOOP = None


async def test_the_record_of_a_vector_transfer_lives_until_it_is_ended():
    assert await crud_ingestions.get_vector_transfer("a1") is None

    since = await crud_ingestions.start_vector_transfer("a1")
    assert await crud_ingestions.get_vector_transfer("a1") == {"since": since, "beat": since}
    assert await get_async_db().ttl(crud_ingestions.vector_transfer_key("a1")) == -1  # no TTL: it outlives a crash
    assert await crud_ingestions.list_statuses("a1") == []  # not a row

    await crud_ingestions.beat_vector_transfer("a1", since)
    record = await crud_ingestions.get_vector_transfer("a1")
    assert record["since"] == since and record["beat"] >= since

    await crud_ingestions.end_vector_transfer("a1")
    assert await crud_ingestions.get_vector_transfer("a1") is None
    await crud_ingestions.beat_vector_transfer("a1", since)  # a late heartbeat does not bring it back
    assert await crud_ingestions.get_vector_transfer("a1") is None
