"""Core-owned ingestion state store.

The standard ingestion engine records its lifecycle in the SAME Redis-JSON
document shape the efficient-ingestion plugin writes, so the UI, the agent
teardown quiesce (``cat.looking_glass.agent_deletion``) and any other reader
see ONE contract regardless of which implementation ran.

Key namespace (identical to the plugin)::

    agents:{agent_id}:ingestion:{scope}:{sha256(source)}

where ``scope`` is ``"agent"`` for the agent KB or a conversation ``chat_id``.
The document mirrors the plugin's ``set_status``::

    source, scope, chat_id, type, status, error, error_at,
    created_at, updated_at, phase (optional),
    completed_phases (diary, one ``ingestion`` entry),
    warnings, warning_count, has_warnings

The diary entry is ``{"state": ..., "tag": ..., "reason": ...}`` (plus
``warnings`` only when a phase produced them). The standard engine runs a
single phase, so it writes exactly one entry, keyed ``"ingestion"``.

All storage goes through ``cat.db.crud``: no raw Redis client is used here.
Import-safe: importing this module performs no Redis, filesystem or network
access.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Any

from cat.db import crud
from cat.db.database import DEFAULT_AGENTS_KEY
from cat.db.models import generate_timestamp

from .models import (
    PHASE_STATE_COMPLETE,
    PHASE_STATE_ERROR,
    PHASE_STATE_PROCESSING,
    IngestionStatus,
)

#: The single diary phase id the standard engine records. The plugin uses its
#: real work-phase ids (``parsing_chunking`` / ``embedding``); the core has one
#: coarse phase, named after the subsystem so the diary stays self-describing.
PHASE_ID = "ingestion"

#: Diary state derived from the source lifecycle. Any non-terminal,
#: non-error status (``uploaded`` / ``downloading`` / ``processing`` / ...)
#: maps to ``processing``: the phase is in flight or about to start. The
#: source ``status`` remains the authoritative lifecycle value.
_PHASE_STATE_BY_STATUS = {
    IngestionStatus.COMPLETED.value: PHASE_STATE_COMPLETE,
    IngestionStatus.ERROR.value: PHASE_STATE_ERROR,
}


def status_key(agent_id: str, scope: str, source: str) -> str:
    """Redis key for a source's ingestion state (identical to EffING).

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` for the agent KB, or a conversation ``chat_id``.
        source: The ingested file name or URL.

    Returns:
        ``agents:{agent_id}:ingestion:{scope}:{sha256(source)}``
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:ingestion:{scope}:{digest}"


def _unwrap_doc(doc: Any) -> dict[str, Any] | None:
    """Unwrap the RedisJSON ``$`` array wrapper and return a dict or None."""
    while isinstance(doc, list) and len(doc) == 1 and isinstance(doc[0], (dict, list)):
        doc = doc[0]
    return doc if isinstance(doc, dict) else None


def _normalize_warnings(warnings: Any) -> list[dict[str, Any]]:
    """Return the well-formed ``{"code", "message"}`` warnings of ``warnings``.

    Mirrors the plugin: a warning is display/telemetry on a successful phase,
    never a state. Malformed entries are dropped rather than persisted.
    """
    if not isinstance(warnings, (list, tuple)):
        return []
    normalized: list[dict[str, Any]] = []
    for warning in warnings:
        if not isinstance(warning, dict):
            continue
        code = warning.get("code")
        message = warning.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            continue
        normalized.append({"code": code, "message": message})
    return normalized


def _aggregate_warnings(diary: Any) -> list[dict[str, Any]]:
    """Aggregate warnings of the ``complete`` diary entries, with the phase id.

    Identical semantics to the plugin: only a phase that completed
    participates; the source ``status`` is never derived from a warning.
    """
    if not isinstance(diary, dict):
        return []
    aggregated: list[dict[str, Any]] = []
    for phase_id, entry in diary.items():
        if not isinstance(entry, dict) or entry.get("state") != PHASE_STATE_COMPLETE:
            continue
        for warning in _normalize_warnings(entry.get("warnings")):
            aggregated.append({"phase": phase_id, **warning})
    return aggregated


def _phase_state(status_value: str) -> str:
    """Diary state for a source lifecycle value (never ``pending``)."""
    return _PHASE_STATE_BY_STATUS.get(status_value, PHASE_STATE_PROCESSING)


class IngestionStateStore(ABC):
    """Contract for reading and writing a source's ingestion state.

    Implementations own the storage; the core (and the UI / quiesce reader)
    depend only on this interface and the document shape it stores.
    """

    @abstractmethod
    async def get(self, agent_id: str, scope: str, source: str) -> dict[str, Any] | None:
        """Return the stored status document for a source, or None."""
        pass

    @abstractmethod
    async def set(self, agent_id: str, scope: str, source: str, **fields) -> dict[str, Any]:
        """Write (or update) the status document and return what was stored."""
        pass

    @abstractmethod
    async def clear(self, agent_id: str, scope: str, source: str) -> None:
        """Remove the source's status document."""
        pass


class CoreIngestionStateStore(IngestionStateStore):
    """Standard store: one ``ingestion`` phase entry, single-pass lifecycle.

    The source lifecycle is ``processing -> completed | error``. Every write
    refreshes ``updated_at`` and preserves ``created_at``; ``error_at`` is set
    only for an ``error`` status. ``set`` accepts the same knobs the plugin's
    ``set_status`` accepts (``type_``/``type``, ``status``, ``error``,
    ``chat_id``, ``phase``, ``clear_phase``, ``warnings``, ``extra``); unknown
    keyword fields are merged into the document for forward compatibility.
    """

    async def get(self, agent_id: str, scope: str, source: str) -> dict[str, Any] | None:
        return _unwrap_doc(await crud.read(status_key(agent_id, scope, source)))

    async def set(self, agent_id: str, scope: str, source: str, **fields) -> dict[str, Any]:
        status = fields.pop("status", IngestionStatus.PROCESSING)
        status_value = status.value if isinstance(status, IngestionStatus) else str(status)

        type_ = fields.pop("type_", None)
        if type_ is None:
            type_ = fields.pop("type", "file")
        else:
            fields.pop("type", None)

        error = fields.pop("error", None)
        chat_id = fields.pop("chat_id", None)
        phase = fields.pop("phase", None)
        clear_phase = bool(fields.pop("clear_phase", False))
        warnings = fields.pop("warnings", None)
        extra = fields.pop("extra", None)
        # Anything left is merged as extra metadata (mirrors ``set_status``).
        if fields:
            extra = {**(extra or {}), **fields}

        key = status_key(agent_id, scope, source)
        now = generate_timestamp()
        existing = await self.get(agent_id, scope, source)
        created_at = existing.get("created_at", now) if existing else now

        doc = dict(existing or {})
        doc.update(
            {
                "source": source,
                "scope": scope,
                "chat_id": chat_id,
                "type": type_,
                "status": status_value,
                "error": error,
                "error_at": now if status_value == IngestionStatus.ERROR.value else None,
                "created_at": created_at,
                "updated_at": now,
            }
        )
        if phase is not None:
            doc["phase"] = phase
        if clear_phase:
            doc.pop("phase", None)
        if extra:
            doc.update(extra)

        entry = {
            "state": _phase_state(status_value),
            "tag": "",
            "reason": error if status_value == IngestionStatus.ERROR.value else None,
        }
        normalized_warnings = _normalize_warnings(warnings)
        if normalized_warnings:
            entry["warnings"] = normalized_warnings
        doc["completed_phases"] = {PHASE_ID: entry}

        aggregated = _aggregate_warnings(doc["completed_phases"])
        doc["warnings"] = aggregated
        doc["warning_count"] = len(aggregated)
        doc["has_warnings"] = bool(aggregated)

        await crud.store(key, doc)
        return doc

    async def clear(self, agent_id: str, scope: str, source: str) -> None:
        await crud.delete(status_key(agent_id, scope, source))
