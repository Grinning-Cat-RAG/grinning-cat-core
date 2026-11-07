"""Core ingestion locks.

The standard single-pass flow needs no cross-worker coordination: one worker
runs the single pass. The lock seam is therefore a genuine no-op for the core,
built on :func:`contextlib.nullcontext`, not on fake state. A plugin with
concurrent phases replaces it with real distributed locks.

Import-safe: importing this module performs no Redis, filesystem or network
access.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any


class IngestionLockManager(ABC):
    """Contract for the ingestion locks.

    ``source_lock`` serializes work on one source; ``tenant_write_lock``
    serializes writes for one tenant. Both return a context manager.
    """

    @abstractmethod
    def source_lock(self, agent_id: str, source: str) -> AbstractContextManager[Any]:
        """Return the context manager for a per-source lock."""

    @abstractmethod
    def tenant_write_lock(self, agent_id: str) -> AbstractContextManager[Any]:
        """Return the context manager for a per-tenant write lock."""


class NoopLockManager(IngestionLockManager):
    """No locks: the standard flow runs single-threaded."""

    def source_lock(self, agent_id: str, source: str) -> AbstractContextManager[Any]:
        """Return a no-op context manager for the source lock."""
        return nullcontext()

    def tenant_write_lock(self, agent_id: str) -> AbstractContextManager[Any]:
        """Return a no-op context manager for the tenant write lock."""
        return nullcontext()
