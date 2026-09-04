"""Tests for the ingestion executor lane, now living in efficient_ingestion.

The dedicated low-concurrency ingestion pool (moved from the core
``cat/services/ingestion_executor.py``) and the ``run_in_ingestion_executor``
seam hook: the base no-op returns None (default-executor fallback), the plugin
hook (priority 1) provides the pool, and the hook is reachable from an agent's
plugin manager.
"""

import asyncio
import os

from cat.core_plugins.base_plugin.hooks import rabbithole as rabbithole_hooks
from cat.core_plugins.efficient_ingestion import ingestion_executor


# Sample the effective nice of a worker thread by running a callback inside the pool.
def _sample_worker_niceness() -> int:
    pool = ingestion_executor._get_ingestion_pool()
    assert pool is not None
    return pool.submit(os.getpriority, os.PRIO_PROCESS, 0).result()


async def test_pool_workers_are_de_prioritized_by_default(monkeypatch):
    """Worker threads get a positive ``nice`` so they yield CPU under pressure.

    With the default ``CAT_INGESTION_NICENESS``, ``_get_ingestion_pool`` builds a
    ``ThreadPoolExecutor`` whose threads run at a lower scheduler priority than the
    main thread.
    """
    # force a fresh pool so any env override below is read
    monkeypatch.setattr(ingestion_executor, "_pool", None)
    monkeypatch.delenv("CAT_INGESTION_NICENESS", raising=False)
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "1")

    pool = ingestion_executor._get_ingestion_pool()
    assert pool is not None

    main_nice = os.getpriority(os.PRIO_PROCESS, 0)
    worker_nice = await asyncio.to_thread(_sample_worker_niceness)
    assert worker_nice > main_nice


async def test_pool_niceness_disabled_when_zero(monkeypatch):
    """``CAT_INGESTION_NICENESS=0`` disables de-prioritization (no-op initializer).

    The initializer is still installed but self-guards on ``niceness <= 0``, so
    worker threads inherit the main thread's niceness.
    """
    monkeypatch.setattr(ingestion_executor, "_pool", None)
    monkeypatch.setenv("CAT_INGESTION_NICENESS", "0")
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "1")

    pool = ingestion_executor._get_ingestion_pool()
    assert pool is not None

    main_nice = os.getpriority(os.PRIO_PROCESS, 0)
    worker_nice = await asyncio.to_thread(_sample_worker_niceness)
    assert worker_nice == main_nice


def test_get_ingestion_pool_returns_none_when_workers_non_positive(monkeypatch):
    """``CAT_INGESTION_WORKERS <= 0`` disables the lane (default executor fallback)."""
    monkeypatch.setattr(ingestion_executor, "_pool", None)
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "0")
    assert ingestion_executor._get_ingestion_pool() is None


async def test_base_noop_returns_none():
    """The base no-op hook returns None: no dedicated lane -> caller falls back.

    The core ``RabbitHole``/``CheshireCat`` seam treats ``None`` as "run on the
    default executor" (upstream parity).
    """
    result = await rabbithole_hooks.run_in_ingestion_executor.function(None, lambda: "never", None)
    assert result is None


async def test_plugin_hook_runs_task_on_the_lane():
    """The efficient_ingestion hook (priority 1) executes the callable and
    returns its result (the dedicated pool or the default-executor fallback)."""
    result = await ingestion_executor._run_in_ingestion_executor_hook.function(None, lambda: 42, None)
    assert result == 42


async def test_seam_reachable_from_agent_manager(cheshire_cat):
    """The hook is registered in an agent's plugin manager and the chain returns
    the executed result (plugin provides the lane, base no-op passes through)."""
    result = await cheshire_cat.plugin_manager.execute_hook(
        "run_in_ingestion_executor", None, lambda: "lane-ok", caller=cheshire_cat,
    )
    assert result == "lane-ok"