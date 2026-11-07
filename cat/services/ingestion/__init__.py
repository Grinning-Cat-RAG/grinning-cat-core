"""Core-owned ingestion contract.

The public surface of ``cat.services.ingestion``: the phase model, the abstract
phase and the status vocabulary. Implementations (standard in the core, the
efficient one in the plugin) subclass :class:`IngestionPhase`.

Import-safe: importing this package performs no Redis, filesystem or network
access.
"""

from cat.services.ingestion.models import (
    PHASE_STATE_COMPLETE,
    PHASE_STATE_ERROR,
    PHASE_STATE_PENDING,
    PHASE_STATE_PROCESSING,
    PHASE_STATE_RUNNING,
    PHASE_STATE_SKIPPED,
    PHASE_STATES,
    IngestionPhaseStatus,
    IngestionStatus,
    PhaseResult,
    PhaseSpec,
)
from cat.services.ingestion.phase import IngestionPhase, own_tag, relevant_settings

__all__ = [
    "PHASE_STATE_PENDING",
    "PHASE_STATE_PROCESSING",
    "PHASE_STATE_COMPLETE",
    "PHASE_STATE_ERROR",
    "PHASE_STATE_SKIPPED",
    "PHASE_STATE_RUNNING",
    "PHASE_STATES",
    "IngestionStatus",
    "IngestionPhaseStatus",
    "PhaseSpec",
    "PhaseResult",
    "IngestionPhase",
    "own_tag",
    "relevant_settings",
]
