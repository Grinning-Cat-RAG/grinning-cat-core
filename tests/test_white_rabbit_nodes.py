"""The White Rabbit on more nodes sharing the same job store: a node going away leaves the jobs of the instance."""
import asyncio
import itertools
import threading
import time
from unittest.mock import patch

from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from hypothesis import HealthCheck, example, given, settings, strategies as st

from cat.core_plugins.white_rabbit.white_rabbit import WhiteRabbit
from cat.db.database import get_sync_db


async def periodic_work():
    """A job of the instance (module level: the store pickles it by its import path)."""


def a_node() -> WhiteRabbit:
    """A node of the Cat: its own scheduler and Redis client on the shared store (the class, not the process
    singleton)."""
    from redis import Redis

    from cat.core_plugins.white_rabbit import white_rabbit

    with patch.object(white_rabbit, "get_sync_db", lambda: Redis(**get_sync_db().get_connection_kwargs())):
        return WhiteRabbit.__wrapped__()


def crash(node: WhiteRabbit):
    """The process of the node dies: its scheduler stops, nothing else runs."""
    node.scheduler.shutdown(wait=False)
    node._is_running = False


NODE_NUMBERS = itertools.count()


def with_latency(node: WhiteRabbit):
    """The round trips to Redis of a node in production, its own on every node: the nodes waking at the same due time
    all read the due jobs before any of them writes, then write one after the other."""
    latency = 0.05 + 0.05 * (next(NODE_NUMBERS) % 6)
    store = node.scheduler._jobstores["default"]
    read = store.get_due_jobs

    def get_due_jobs(now):
        due = read(now)
        if due:
            time.sleep(latency)
        return due

    store.get_due_jobs = get_due_jobs


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


# --- one run per scheduled time, whatever the nodes ------------------------------------------------------------------

class ThreadNode:
    """A node in its own thread and event loop: nodes are processes in production, they read the store in parallel."""

    def __init__(self, runs: list):
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()

        async def create():  # the scheduler starts inside its running loop
            self.node = a_node()
            with_latency(self.node)
            self.node.scheduler.add_listener(
                lambda e: runs.append((e.job_id, e.scheduled_run_time)), EVENT_JOB_EXECUTED,
            )

        def serve():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(create())
            ready.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        ready.wait(5)

    def call(self, fn, *args, **kwargs):
        async def run():
            return fn(*args, **kwargs)

        return asyncio.run_coroutine_threadsafe(run(), self.loop).result(10)

    def stop(self, crashed=False):
        if not crashed:
            self.call(lambda node: node.shutdown(), self.node)
        self.loop.call_soon_threadsafe(self.loop.stop)  # a crash: the process dies, nothing runs anymore
        self.thread.join(5)

    def dies_after_booking(self):
        """The node dies right after booking its next run, before running it."""
        executor = self.node.scheduler._executors["default"]

        def die(job, run_times):
            self.loop.stop()

        self.call(setattr, executor, "_do_submit_job", die)


async def test_two_nodes_run_each_scheduled_time_once():
    clear_store()
    runs = []
    one, other = ThreadNode(runs), ThreadNode(runs)
    one.call(one.node.schedule_interval_job, periodic_work, job_id="every-second", seconds=1)
    other.call(other.node.scheduler.wakeup)

    await asyncio.sleep(3.5)

    assert len(runs) >= 3
    assert len(runs) == len(set(runs))  # no scheduled time run twice
    assert other.call(other.node.get_job, "every-second").status == "scheduled"  # nobody left it "running"
    one.stop()
    other.stop()
    clear_store()


NODE_OPERATIONS = st.one_of(
    st.just(("start", None)),
    st.tuples(st.just("stop"), st.integers(0, 3)),
    st.tuples(st.just("crash"), st.integers(0, 3)),
    st.tuples(st.just("wait"), st.sampled_from((0.4, 1.1))),
    st.tuples(st.just("dies_after_booking"), st.integers(0, 3)),
    st.just(("reminder", None)),  # a one-shot job, due after the other nodes woke up at least once
)


@settings(max_examples=10, deadline=None, suppress_health_check=list(HealthCheck))
@example(operations=[("start", None), ("start", None), ("wait", 1.1), ("wait", 1.1)])
@example(operations=[("start", None), ("reminder", None), ("wait", 1.1)])
@example(operations=[("start", None), ("dies_after_booking", 1), ("wait", 1.1)])
@given(operations=st.lists(NODE_OPERATIONS, max_size=6))
def test_each_scheduled_time_runs_once_while_nodes_come_and_go(operations):
    asyncio.run(_each_scheduled_time_runs_once(operations))


async def _each_scheduled_time_runs_once(operations):
    clear_store()
    runs = []
    first = ThreadNode(runs)
    first.call(first.node.schedule_interval_job, periodic_work, job_id="every-second", seconds=1)
    nodes = [first]
    try:
        reminders = 0
        for kind, arg in operations:
            if kind == "start":
                nodes.append(ThreadNode(runs))
            elif kind in ("stop", "crash", "dies_after_booking") and len(nodes) > 1:  # one node stays: the instance is up
                node = nodes.pop(arg % len(nodes))
                if kind == "dies_after_booking":
                    node.dies_after_booking()
                    await asyncio.sleep(1.1)  # its next run: booked, never run
                    node.stop(crashed=True)
                else:
                    node.stop(crashed=kind == "crash")
            elif kind == "reminder":
                reminders += 1
                nodes[0].call(nodes[0].node.schedule_job, periodic_work, job_id=f"reminder-{reminders}", seconds=2.5)
            elif kind == "wait":
                await asyncio.sleep(arg)
            assert len(runs) == len(set(runs))  # the rule: no scheduled time runs twice
        await asyncio.sleep(1.1)
        before = len([r for r in runs if r[0] == "every-second"])
        await asyncio.sleep(2.2)
        # every live node keeps running the jobs (none stopped by a one-shot job removed by another node)
        assert len([r for r in runs if r[0] == "every-second"]) > before
        assert len(runs) == len(set(runs))
        assert {r[0] for r in runs if r[0].startswith("reminder")} == {f"reminder-{i}" for i in range(1, reminders + 1)}
        for node in nodes:  # no live node lost its timer (it would never run a job again)
            assert node.call(lambda scheduler: scheduler._timeout is not None, node.node.scheduler)
    finally:
        for node in nodes:
            node.stop()
        clear_store()


async def test_a_scheduler_that_cannot_start_leaves_a_stopped_node(monkeypatch):
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    def unreachable(self, *args, **kwargs):
        raise ConnectionError("the job store is unreachable")

    monkeypatch.setattr(AsyncIOScheduler, "start", unreachable)

    node = a_node()

    assert node._is_running is False
    node.shutdown()  # nothing to stop
