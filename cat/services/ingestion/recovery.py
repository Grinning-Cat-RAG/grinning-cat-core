"""Core ingestion recovery.

The standard single-pass flow has no per-phase work to resume: an interrupted
ingestion is simply re-run from the source. The recovery seam is therefore a
genuine no-op for the core — it reports zero work instead of faking state. A
plugin that keeps a per-phase diary replaces it with its own resume/reconcile
implementation.

Import-safe: importing this module performs no Redis, filesystem or network
access.
"""

from abc import ABC, abstractmethod


class IngestionRecovery(ABC):
    """Contract for resuming and reconciling ingestion work.

    ``resume_pending`` resumes work interrupted before completion and returns
    how many sources were resumed. ``reconcile`` reconciles stored ingestion
    state against the canonical sources and returns how many rows were handled.
    """

    @abstractmethod
    async def resume_pending(self, lizard) -> int:
        """Resume pending ingestion work; return how many sources were resumed."""

    @abstractmethod
    async def reconcile(self, lizard) -> int:
        """Reconcile stored ingestion state; return how many rows were handled."""


class NoopRecovery(IngestionRecovery):
    """The standard flow has nothing to resume: a genuine no-op, zero work."""

    async def resume_pending(self, lizard) -> int:
        """Return zero: the single pass has no resumable per-phase state."""
        return 0

    async def reconcile(self, lizard) -> int:
        """Return zero: the single pass stores nothing to reconcile."""
        return 0
