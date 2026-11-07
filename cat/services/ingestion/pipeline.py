"""Core ingestion pipeline.

The pipeline is the abstract unit that owns the ingestion of one source: it
declares the phases, runs them and deletes the source's artifacts. The core
ships the single-pass, upstream-parity implementation, which runs one
:class:`CoreIngestionPhase`; a plugin may replace it with a multi-phase
pipeline.

``CoreSinglePassPipeline.delete`` is the ONE deletion contract for the standard
flow: it reproduces the file-manager route's exact sequence (storage removal,
``before_file_manager_file_delete`` hook, points deletion,
``after_file_manager_file_deleted`` hook) and removes the status row LAST.

Import-safe: importing this module performs no Redis, filesystem or network
access. ``RabbitHole`` and the route helpers are imported lazily at call time to
keep the import graph small and avoid cycles.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

from .models import PhaseResult
from .phase import IngestionPhase
from .state import CoreIngestionStateStore, IngestionStateStore


async def _ingest_file(
    cat,
    source,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    store_file: bool = True,
    content_type: str | None = None,
) -> None:
    """Delegate to ``RabbitHole.ingest_file`` with the engine's exact arguments."""
    # local import: avoids a circular import at module load and keeps the core
    # ingestion package import-safe for the plugin manager.
    from cat.rabbit_hole import RabbitHole

    rabbit_hole = getattr(cat, "rabbit_hole", None) or RabbitHole()
    await rabbit_hole.ingest_file(
        cat=cat,
        file=source,
        filename=filename,
        metadata=metadata or {},
        store_file=store_file,
        content_type=content_type,
    )


class IngestionPipeline(ABC):
    """Contract for one ingestion pipeline.

    ``phases`` declares the ordered work units, ``execute`` runs the pipeline
    for a source and ``delete`` removes every artifact of a source. ``delete``
    is the single deletion contract shared by the file-manager route and the
    agent-teardown cascade, so file deletion and agent deletion cannot diverge.
    """

    @abstractmethod
    def phases(self) -> list[IngestionPhase]:
        """Return this pipeline's phases, in execution order."""

    @abstractmethod
    async def execute(self, source: str, cat) -> None:
        """Run the ingestion pipeline for ``source``."""

    @abstractmethod
    async def delete(self, source: str, cat, scope: str | None = None) -> bool:
        """Delete every artifact of ``source``.

        Args:
            source: The source name (file name or URL).
            cat: The CheshireCat/StrayCat owning the source.
            scope: ``"agent"`` (or ``None``) for the agent KB, or a conversation
                ``chat_id`` for an episodic source.

        Returns:
            Whether the stored file was removed.
        """


class CoreIngestionPhase(IngestionPhase):
    """The single standard phase: the upstream-parity ingestion.

    ``process`` delegates to ``RabbitHole.ingest_file`` with the same arguments
    as ``CoreIngestionEngine.ingest_file``. ``delete`` removes the source's
    points in the agent's declarative memory; it is a real deletion, never a
    silent no-op.
    """

    id = "ingestion"
    label = "Ingestion"
    settings_category = "ingestion"

    async def applies_to(self, cat, deps_state) -> tuple[bool, str | None]:
        """The single pass always applies: there is no conditional phase."""
        return True, None

    async def process(self, source: str, cat) -> PhaseResult:
        """Run the standard ingestion for ``source``."""
        await _ingest_file(cat, source)
        return PhaseResult(status="done")

    async def delete(self, source: str, cat) -> None:
        """Remove the source's points from the agent's declarative memory."""
        from cat.services.memory.models import VectorMemoryType

        await cat.vector_memory_handler.delete_tenant_points(
            str(VectorMemoryType.DECLARATIVE), {"source": source}
        )


class CoreSinglePassPipeline(IngestionPipeline):
    """The standard single-pass pipeline: one phase, the upstream flow.

    ``execute`` delegates to ``RabbitHole.ingest_file`` exactly like
    ``CoreIngestionEngine.ingest_file``. ``delete`` is behaviourally identical
    to the file-manager route's ``delete_file`` and removes the status row LAST
    through the core state store.
    """

    def __init__(self, store: IngestionStateStore | None = None):
        self._store: IngestionStateStore = store or CoreIngestionStateStore()

    def phases(self) -> list[IngestionPhase]:
        """Return the one standard ingestion phase."""
        return [CoreIngestionPhase()]

    async def execute(self, source: str, cat) -> None:
        """Run the standard single-pass ingestion for ``source``."""
        await _ingest_file(cat, source)

    async def delete(self, source: str, cat, scope: str | None = None) -> bool:
        """Delete ``source`` exactly as the file-manager route does.

        Order (the route's contract):
        1. ``file_manager.remove_file`` on the sanitized path;
        2. ``before_file_manager_file_delete`` hook (points still present);
        3. ``vector_memory_handler.delete_tenant_points``;
        4. ``after_file_manager_file_deleted`` hook;
        5. status row removal LAST.
        """
        # local imports: keep the module import-safe and avoid a route/ingestion
        # import cycle.
        from cat.exceptions import CustomValidationException
        from cat.routes.routes_utils import sanitize_source_name
        from cat.services.memory.models import VectorMemoryType

        scope_value = "agent" if scope is None else scope
        path = (
            cat.agent_key
            if scope_value == "agent"
            else os.path.join(cat.agent_key, scope_value)
        )
        collection_id = (
            VectorMemoryType.DECLARATIVE
            if scope_value == "agent"
            else VectorMemoryType.EPISODIC
        )
        metadata = {"source": source}
        if scope_value != "agent":
            metadata["chat_id"] = scope_value

        sanitized_source = sanitize_source_name(source, path=path)

        try:
            removed = cat.file_manager.remove_file(os.path.join(path, sanitized_source))

            chat_or_agent = path.split(os.sep, 1)[1] if os.sep in path else "agent"
            await cat.plugin_manager.execute_hook(
                "before_file_manager_file_delete",
                sanitized_source,
                str(chat_or_agent),
                caller=cat,
            )

            await cat.vector_memory_handler.delete_tenant_points(str(collection_id), metadata)

            await cat.plugin_manager.execute_hook(
                "after_file_manager_file_deleted",
                sanitized_source,
                str(chat_or_agent),
                caller=cat,
            )

            # The status row goes LAST: a crash before this leaves a resumable
            # row instead of a row claiming artifacts that still exist.
            await self._store.clear(cat.agent_key, scope_value, source)

            return bool(removed)
        except Exception as e:  # noqa: BLE001 - parity with the route's error handling
            raise CustomValidationException(f"Failed to delete memory points: {e}")
