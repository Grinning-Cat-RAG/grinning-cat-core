"""Contract tests for the core ingestion recovery.

The standard single-pass flow has no resumable per-phase state, so ``NoopRecovery``
is a genuine no-op: both operations report zero work.

Import-safe: the recovery module must not touch Redis at import time.
"""

from cat.services.ingestion.recovery import IngestionRecovery, NoopRecovery


def test_noop_recovery_implements_the_contract():
    assert issubclass(NoopRecovery, IngestionRecovery)


async def test_noop_recovery_returns_zero_for_both_operations():
    recovery = NoopRecovery()

    assert await recovery.resume_pending(None) == 0
    assert await recovery.reconcile(None) == 0
