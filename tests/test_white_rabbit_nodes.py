"""The White Rabbit on more nodes sharing the same job store: a node going away leaves the jobs of the instance."""
import asyncio

from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from hypothesis import HealthCheck, example, given, settings, strategies as st

from cat.core_plugins.white_rabbit.white_rabbit import WhiteRabbit
from cat.db.database import get_sync_db


async def periodic_work():
    """A job of the instance (module level: the store pickles it by its import path)."""


def a_node() -> WhiteRabbit:
    """A node of the Cat: its own scheduler on the shared store (the class, not the process singleton)."""
    return WhiteRabbit.__wrapped__()


def crash(node: WhiteRabbit):
    """The process of the node dies: its scheduler stops, nothing else runs."""
    node.scheduler.shutdown(wait=False)
    node._is_running = False


def stored_jobs() -> set:
    return {key.decode() if isinstance(key, bytes) else key for key in get_sync_db().hkeys("white_rabbit:jobs")}


def clear_store():
    get_sync_db().delete("white_rabbit:jobs", "white_rabbit:run_times")


def realign(node: WhiteRabbit, job_id: str, kind: str):
    """A plugin aligning its job at startup: keeping the one in the store, or scheduling it again."""
    if kind == "keep" and node.get_job(job_id) is not None:
        return
    if kind == "cron":
        node.schedule_cron_job(periodic_work, job_id=job_id, hour=3)
    else:
        node.schedule_interval_job(periodic_work, job_id=job_id, hours=1)


async def test_a_node_stopped_after_the_new_one_started_leaves_the_jobs():
    clear_store()
    old = a_node()
    realign(old, "re_embed_tools", "replace")
    realign(old, "mailbox:agent-1", "keep")
    realign(old, "autochunk", "cron")

    new = a_node()  # start-first: the new node aligns the jobs, then the old one goes away
    realign(new, "re_embed_tools", "replace")
    realign(new, "mailbox:agent-1", "keep")
    realign(new, "autochunk", "keep")
    old.shutdown()

    assert stored_jobs() == {"re_embed_tools", "mailbox:agent-1", "autochunk"}
    new.shutdown()
    assert stored_jobs() == {"re_embed_tools", "mailbox:agent-1", "autochunk"}  # the instance keeps them
    clear_store()


async def test_a_pending_one_shot_job_survives_the_node_that_scheduled_it():
    clear_store()
    node = a_node()
    node.schedule_job(periodic_work, job_id="reminder", hours=1)
    node.shutdown()

    assert stored_jobs() == {"reminder"}
    clear_store()


async def test_the_runs_missed_while_every_node_was_down_are_merged_into_one():
    clear_store()
    node = a_node()
    node.schedule_interval_job(periodic_work, job_id="every-second", seconds=1)
    node.shutdown()
    await asyncio.sleep(3.5)  # every node down: three runs missed

    events = []
    back = a_node()
    back.scheduler.add_listener(lambda e: events.append(e.code), EVENT_JOB_MISSED | EVENT_JOB_EXECUTED)
    await asyncio.sleep(0.3)  # its first pass

    assert stored_jobs() == {"every-second"}
    assert len(events) == 1  # one late run (run, or missed and logged once), not one per missed run
    back.shutdown()
    clear_store()


async def test_an_explicit_removal_removes_the_job_for_every_node():
    clear_store()
    one, other = a_node(), a_node()
    realign(one, "mailbox:agent-1", "replace")

    assert other.remove_job("mailbox:agent-1")
    assert stored_jobs() == set()
    one.shutdown()
    other.shutdown()


# --- stateful: nodes starting, stopping and crashing on the same store -----------------------------------------------

JOBS = ("re_embed_tools", "mailbox:agent-1", "autochunk")
OPERATIONS = st.one_of(
    st.tuples(st.just("start"), st.sampled_from(("keep", "replace"))),  # a node starts and aligns the jobs
    st.tuples(st.just("stop"), st.integers(0, 3)),  # a node stops cleanly (deploy, scale down)
    st.tuples(st.just("crash"), st.integers(0, 3)),  # a node dies
    st.tuples(st.just("remove"), st.sampled_from(JOBS)),  # a job removed on purpose (plugin turned off)
)


@settings(max_examples=25, deadline=None, suppress_health_check=list(HealthCheck))
@example(operations=[("start", "replace"), ("start", "replace"), ("stop", 0)])
@example(operations=[("start", "keep"), ("start", "keep"), ("stop", 0)])
@given(operations=st.lists(OPERATIONS, max_size=8))
def test_the_jobs_of_the_instance_survive_its_nodes(operations):
    asyncio.run(_the_jobs_of_the_instance_survive_its_nodes(operations))


async def _the_jobs_of_the_instance_survive_its_nodes(operations):
    clear_store()
    nodes = []
    removed = set()  # the jobs removed on purpose: no node brings them back until a plugin schedules them again
    started = False
    try:
        for kind, arg in operations:
            if kind == "start":
                node = a_node()
                for job_id in JOBS:
                    if job_id not in removed:
                        realign(node, job_id, arg if job_id != "autochunk" else ("keep" if arg == "keep" else "cron"))
                nodes.append(node)
                started = True
            elif kind in ("stop", "crash") and nodes:
                node = nodes.pop(arg % len(nodes))
                node.shutdown() if kind == "stop" else crash(node)
            elif kind == "remove" and nodes:
                nodes[0].remove_job(arg)
                removed.add(arg)
            # the rule: every job aligned by a node, and not removed on purpose, is in the store, live nodes or not
            assert stored_jobs() == (set(JOBS) - removed if started else set())
    finally:
        for node in nodes:
            node.shutdown()
        clear_store()


async def test_turning_the_re_embedding_off_removes_its_job_at_the_restart(monkeypatch):
    from types import SimpleNamespace

    from cat.core_plugins.white_rabbit import main

    clear_store()
    before = a_node()
    before.schedule_interval_job(main.re_embed_tools, job_id=main.scheduled_job_id, days=1)
    before.shutdown()

    async def load_settings():
        return {"embed_procedures_every_n_days": 0}  # the admin turned it off, then restarted

    plugin = SimpleNamespace(load_settings=load_settings)
    lizard = SimpleNamespace(plugin_manager=SimpleNamespace(get_plugin=lambda: plugin), white_rabbit=None)
    monkeypatch.setattr(main, "WhiteRabbit", a_node)

    await main.after_lizard_bootstrap.function(lizard)

    assert stored_jobs() == set()
    lizard.white_rabbit.shutdown()
