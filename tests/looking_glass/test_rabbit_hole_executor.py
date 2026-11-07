"""The dedicated ingestion lane of ``cat.rabbit_hole`` (``run_in_ingestion_executor``).

``CAT_INGESTION_WORKERS > 0`` runs the heavy ingestion callables (parse, embed) on a process-wide pool of
de-prioritized threads; ``<= 0`` falls back to the default executor.
"""
import os
import threading
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import cat.rabbit_hole as rabbit_hole_module
from cat.rabbit_hole import _set_worker_niceness, run_in_ingestion_executor


@pytest.fixture
def fresh_pool(monkeypatch):
    """A fresh (not yet created) module pool; the one created by the test is shut down afterwards."""
    monkeypatch.setattr(rabbit_hole_module, "_pool", None)
    yield
    if rabbit_hole_module._pool is not None:
        rabbit_hole_module._pool.shutdown(wait=True)


def _describe(*args, **kwargs):
    return threading.current_thread().name, args, kwargs


async def test_positional_and_keyword_arguments_reach_the_callable(fresh_pool, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "0")

    _, args, kwargs = await run_in_ingestion_executor(_describe, 1, "two", three=3)

    assert args == (1, "two")
    assert kwargs == {"three": 3}


async def test_disabled_lane_uses_the_default_executor(fresh_pool, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "0")

    thread_name, _, _ = await run_in_ingestion_executor(_describe)

    assert not thread_name.startswith("cat-ingestion")
    assert rabbit_hole_module._pool is None


async def test_enabled_lane_runs_on_the_dedicated_pool_and_keeps_it(fresh_pool, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "1")
    monkeypatch.setenv("CAT_INGESTION_NICENESS", "0")

    thread_name, _, kwargs = await run_in_ingestion_executor(_describe, key="value")
    pool = rabbit_hole_module._pool
    await run_in_ingestion_executor(_describe)

    assert thread_name.startswith("cat-ingestion")
    assert kwargs == {"key": "value"}
    assert rabbit_hole_module._pool is pool  # created once, then cached


async def test_exceptions_of_the_callable_reach_the_caller(fresh_pool, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "1")

    def boom():
        raise ValueError("parse failed")

    with pytest.raises(ValueError, match="parse failed"):
        await run_in_ingestion_executor(boom)


def _niceness_after(niceness: int) -> int:
    """Apply ``_set_worker_niceness`` in a throwaway thread (niceness is per-thread on Linux) and read it back."""
    result = {}

    def worker():
        _set_worker_niceness(niceness)
        result["niceness"] = os.getpriority(os.PRIO_PROCESS, 0)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return result["niceness"]


def test_positive_niceness_lowers_the_worker_priority():
    assert _niceness_after(19) == 19


def test_non_positive_niceness_leaves_the_worker_priority_unchanged():
    assert _niceness_after(0) == os.getpriority(os.PRIO_PROCESS, 0)


async def test_procedures_are_embedded_on_the_ingestion_lane(cheshire_cat, monkeypatch):
    import cat.looking_glass.cheshire_cat as cheshire_cat_module

    lanes = []
    original = run_in_ingestion_executor

    async def recording_lane(func, *args, **kwargs):
        lanes.append(func)
        return await original(func, *args, **kwargs)

    class FakeProcedure:
        type = None

        async def to_document_recall(self):
            return [SimpleNamespace(document=Document(page_content="say hello to the user"))]

    monkeypatch.setattr(cheshire_cat_module, "run_in_ingestion_executor", recording_lane, raising=False)
    monkeypatch.setattr(type(cheshire_cat.plugin_manager), "procedures", property(lambda self: [FakeProcedure()]))

    await cheshire_cat.embed_procedures()

    embedder = await cheshire_cat.embedder()
    assert len(lanes) == 1
    assert lanes[0].__name__ == embedder.embed_documents.__name__
