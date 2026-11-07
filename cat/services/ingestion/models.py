"""Core-owned ingestion contract models.

The phase DAG, the per-phase result and the ingestion status vocabulary are the
contract between the core and any ingestion implementation. They live here so
the core can describe and read an ingestion run without importing a plugin.

Moved unchanged in behaviour from the efficient-ingestion plugin's
``phases.py`` / ``registry.py``: field names, defaults and string values are the
wire shapes the diary and the tags depend on.

Import-safe: no Redis, filesystem or network access at import time.
"""

from dataclasses import dataclass
from typing import Any, Literal

from cat.utils import Enum

#: Per-(source, phase) diary state machine. Every diary entry carries an
#: explicit ``state``: ``processing`` is written BEFORE the phase body starts,
#: ``complete`` when the body returned ``done``, ``error`` with a reason when it
#: failed, and ``skipped`` with a reason when ``applies_to`` says the phase does
#: not apply. An ABSENT entry means ``pending``. An entry written by an older
#: release has no ``state`` and is read as ``complete`` (backward compatible)
#: and normalized on the next write.
PHASE_STATE_PENDING = "pending"
PHASE_STATE_PROCESSING = "processing"
PHASE_STATE_COMPLETE = "complete"
PHASE_STATE_ERROR = "error"
PHASE_STATE_SKIPPED = "skipped"

#: The explicit states stored in a diary entry (``pending`` is the absence of
#: an entry, so it is never written).
PHASE_STATES = (
    PHASE_STATE_PROCESSING,
    PHASE_STATE_COMPLETE,
    PHASE_STATE_ERROR,
    PHASE_STATE_SKIPPED,
)

#: Deprecated alias: the old "running" state is now ``processing``.
PHASE_STATE_RUNNING = PHASE_STATE_PROCESSING


class IngestionStatus(Enum):
    """Lifecycle states of a source ingestion.

    Files: ``uploaded -> processing -> completed | error``.
    URLs add ``downloading -> downloaded`` between ``uploaded`` and ``processing``.

    ``deleting`` is the file-level deletion state: written BEFORE any phase
    ``delete()`` runs, and the row is removed only after every phase delete
    succeeded. A crash therefore leaves the row ``deleting`` (resumable) instead
    of claiming a state the artifacts no longer match. It is never claimable by
    the resume sweep (it is not an in-flight ingestion status).

    ``reindexing`` is the source-level re-ingest state: written BEFORE the
    artifact wipe, kept while the artifact deletion runs, and replaced by
    ``processing`` before the phases start. The status row is NEVER removed by
    a re-ingest, so a crash leaves ``reindexing`` (recovery re-runs the reset)
    instead of a missing row. Like ``deleting`` it is explicit and is never
    overridden by the derived diary status; the dispatch loop skips it.
    """

    UPLOADED = "uploaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    DELETING = "deleting"
    REINDEXING = "reindexing"


class IngestionPhaseStatus(Enum):
    """Per-phase diary state of one source ingestion.

    The core-owned status model for ``completed_phases`` entries: ``pending``
    is the absence of an entry, the other four are the explicit states a diary
    entry stores. The string values are the wire shape and must not change.
    """

    PENDING = PHASE_STATE_PENDING
    PROCESSING = PHASE_STATE_PROCESSING
    COMPLETE = PHASE_STATE_COMPLETE
    ERROR = PHASE_STATE_ERROR
    SKIPPED = PHASE_STATE_SKIPPED


@dataclass(frozen=True)
class PhaseSpec:
    """Declaration of one ingestion phase and its invalidation inputs.

    Attributes:
        id: The phase id (also the diary key, e.g. ``"embedding"``).
        settings_category: The settings category whose ``updated_at`` is the
            opaque fast-path token (``"chunker"``, ``"embedder"``), or None
            for phases with no settings entry (marker-only invalidation).
        depends_on: Upstream phase ids whose recorded markers this phase
            consumed; a change in any upstream marker makes this phase stale.
        settings_check: Optional narrowing of the settings that feed
            :meth:`IngestionPhase.tag`: a key collection keeps only those
            keys, a (possibly async) callable returns the value that
            participates. ``None`` means the whole settings value.
    """

    id: str
    settings_category: str | None
    depends_on: tuple[str, ...]
    settings_check: Any = None


@dataclass(frozen=True)
class PhaseResult:
    """Outcome of running one phase.

    ``processed`` / ``expected`` are TELEMETRY ONLY: they may be logged or
    surfaced to an operator, but they are never a completion criterion. A
    phase is complete because ``status == "done"``, not because a count
    matches.

    ``warnings`` is the OPTIONAL degradation channel for a phase that RAN
    correctly but produced nothing VALID TO BE EMPTY (or less than it should).
    Each warning is ``{"code": <str>, "message": <str>}``; the documented codes
    are ``empty-extraction``, ``partial-extraction`` and ``degraded``. A
    warning is NOT a state: it is attached to a ``done`` result, so the phase
    stays complete, nothing retries and no dependent is blocked.

    An empty product is not universally benign: the built-in
    ``parsing_chunking`` (a non-empty source yielding no chunk) and
    ``embedding`` (points to embed yielding no stored point) report ``error``
    instead, because their product is the searchable content itself. The
    catalog and graph phases keep the warning channel. "Could not run"
    (timeout, transport error, LLM unavailable, no input) is still
    ``not_ready`` then ``error``. The phase owns this distinction; the machine
    applies no size threshold.
    """

    status: Literal["done", "not_ready", "error", "skipped"]
    reason: str | None = None
    processed: int | None = None
    expected: int | None = None
    warnings: list[dict] | None = None
