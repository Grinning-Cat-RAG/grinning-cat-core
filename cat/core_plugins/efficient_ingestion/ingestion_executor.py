"""Dedicated low-concurrency executor lane for heavy ingestion work.

Ingestion (chunking, embedding, storing) is CPU/IO heavy and, when run on the
shared default ``ThreadPoolExecutor``, can saturate the pool that chat-time
embedding/chunking also relies on, degrading interactive latency.

This module (moved from the core ``cat/services/ingestion_executor.py``) provides
a process-wide, lazily-created ``ThreadPoolExecutor`` whose worker count is
bounded by ``CAT_INGESTION_WORKERS``. It complements the
``CAT_INGESTION_MAX_CONCURRENCY`` semaphore in ``cat.rabbit_hole``: the
semaphore bounds how many ingestion tasks run at once, while this pool keeps
those tasks off the shared default executor.

Each worker thread is additionally de-prioritized (``nice``) via
``CAT_INGESTION_NICENESS`` (default 5, ``<= 0`` disables): under CPU pressure
the OS scheduler gives the rest of MyCAT (chat/recall) a higher share than the
ingestion workers, so a heavy ingestion yields CPU to interactive work instead
of competing on equal footing.

The lane is exposed to the core through the ``run_in_ingestion_executor`` hook
(priority 1, overriding the base no-op): callers dispatch the callable there and
fall back to the default executor when no plugin provides a dedicated lane. This
module also keeps a direct ``run_in_ingestion_executor`` function for intra-plugin
use (the re-embed engine).

Importing this module has zero side effects: the pool is only created on first
use via ``_get_ingestion_pool()``.
"""

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from cat import hook
from cat.env import get_env_int

T = TypeVar("T")

#: Default ``nice`` used for the ingestion worker threads (0 disables).
_INGESTION_NICENESS_ENV = "CAT_INGESTION_NICENESS"
_INGESTION_NICENESS_DEFAULT = 5

_pool: ThreadPoolExecutor | None = None


def _set_worker_niceness(niceness: int) -> None:
    """Lower the priority of the current worker thread.

    Sets an absolute nicer value for the *current thread* on Linux
    (``PRIO_PROCESS`` with ``who=0`` is implemented per-thread in the kernel
    ``task_struct``). Raising niceness (de-prioritizing) is always permitted;
    only lowering (negative values) would require privileges.
    """
    if niceness > 0:
        os.setpriority(os.PRIO_PROCESS, 0, niceness)


def _get_ingestion_pool() -> ThreadPoolExecutor | None:
    """Return the process-wide ingestion pool, creating it on first use.

    The worker count is read at runtime from ``CAT_INGESTION_WORKERS`` and the
    pool is then cached for the process lifetime. ``None`` or ``<= 0`` means
    the lane is disabled: no pool is created and callers fall back to the
    default executor (previous behavior is preserved).
    """
    global _pool
    if _pool is None:
        max_workers = get_env_int("CAT_INGESTION_WORKERS")
        if max_workers is None or max_workers <= 0:
            return None
        niceness = get_env_int(_INGESTION_NICENESS_ENV)
        if niceness is None:
            niceness = _INGESTION_NICENESS_DEFAULT
        # ``initializer`` runs in each worker thread; ``_set_worker_niceness``
        # is a no-op when ``niceness <= 0`` (de-prioritization disabled).
        _pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cat-ingestion",
            initializer=_set_worker_niceness,
            initargs=(niceness,),
        )
    return _pool


async def run_in_ingestion_executor(
    func: Callable[..., T], *args, **kwargs
) -> T:
    """Run ``func(*args, **kwargs)`` on the dedicated ingestion pool.

    Intra-plugin use (the re-embed engine). If the ingestion lane is disabled
    (``CAT_INGESTION_WORKERS`` unset or ``<= 0``), the call is dispatched to
    the default executor, preserving the prior behavior.
    """
    loop = asyncio.get_running_loop()
    pool = _get_ingestion_pool()
    if pool is None:
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return await loop.run_in_executor(pool, func, *args, **kwargs)


@hook("run_in_ingestion_executor", priority=1)
async def _run_in_ingestion_executor_hook(result: Any, func, cat) -> Any:
    """Provide the dedicated ingestion lane to the core seam.

    Overrides the base no-op (priority 0): runs the callable on the dedicated
    pool when enabled, otherwise on the default executor.
    """
    return await run_in_ingestion_executor(func)