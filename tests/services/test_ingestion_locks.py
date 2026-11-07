"""Contract tests for the core ingestion locks.

The standard single-pass flow needs no coordination, so ``NoopLockManager`` is a
genuine no-op built on ``contextlib.nullcontext``: both locks are usable as
context managers and do not touch Redis.

Import-safe: the locks module must not touch Redis at import time.
"""

from cat.services.ingestion.locks import IngestionLockManager, NoopLockManager


def test_noop_lock_manager_implements_the_contract():
    assert issubclass(NoopLockManager, IngestionLockManager)


def test_noop_lock_manager_returns_usable_context_managers():
    manager = NoopLockManager()

    with manager.source_lock("agent", "doc.txt") as lock:
        assert lock is None

    with manager.tenant_write_lock("agent") as lock:
        assert lock is None
