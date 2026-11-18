"""The registry of the ingestion states: one row per source of an agent, in the agent knowledge base or in a chat.

A row lives at ``agents:<agent_id>:ingestion:agent:<sha256(source)>`` for the knowledge base of the agent, and at
``agents:<agent_id>:ingestion:chat:<chat_id>:<sha256(source)>`` for a source uploaded in a chat (``chat_id=None``
means the knowledge base in every function). The row is a JSON document:

- ``source``, ``chat_id``, ``type`` (``"file"`` or ``"url"``), ``content_type``, ``metadata`` (the custom metadata of
  the upload, stored with every chunk);
- ``status`` (see ``IngestionStatus``), ``phase`` (the phase running, if any), ``error`` (the reason of a failure);
- ``owner`` (the worker processing the source), ``deleted_phases`` (the phases already removed by a deletion);
- ``completed_phases``: the diary, ``{phase_id: entry}``, written when a phase ends;
- ``created_at`` and ``updated_at`` (seconds since the epoch, to the millisecond).

The writes of a row are not guarded against a concurrent re-creation of the same row: the RabbitHole writes a row only
while holding the lock of its source. The maintenance marker (``agents:<agent_id>:ingestion_maintenance``) suspends the
automatic resume of an agent while a maintenance script works on it, while the embedder changes and while the vector
database of the agent changes. It has no owner: whoever clears it first clears it for everybody. The record of a vector
transfer (``agents:<agent_id>:ingestion_vector_transfer``: ``since``, ``beat``) lives from the start of the copy to its
end, with no TTL: the automatic resume takes over a transfer whose heartbeat stopped. Public API: maintenance scripts
read and reset the states through this module.
"""
import hashlib
import re
import time
from enum import Enum
from typing import Any, Dict, Iterable, List

from redis.exceptions import ResponseError

from cat.db import crud
from cat.db.database import DEFAULT_AGENTS_KEY, DEFAULT_INGESTION_KEY, get_async_db

#: ``list_statuses(agent_id, chat_id=ALL_SCOPES)``: the rows of the knowledge base and of every chat.
ALL_SCOPES = object()

SOURCE_TYPES = ("file", "url")

_UPDATABLE_FIELDS = {"status", "phase", "error", "owner", "deleted_phases", "content_type"}
#: The ids of the phases: lowercase letters, digits and underscores, starting with a letter.
PHASE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_SOURCE_ID = re.compile(r"^[0-9a-f]{64}$")
_AGENT_SEGMENT = "agent"
_CHAT_SEGMENT = "chat:"


class IngestionStatus(str, Enum):
    UPLOADED = "uploaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    DELETING = "deleting"
    REINDEXING = "reindexing"


#: Display labels of the states (English defaults: a client localizes them by the state id)
STATUS_LABELS = {
    IngestionStatus.UPLOADED.value: "Uploaded",
    IngestionStatus.DOWNLOADING.value: "Downloading",
    IngestionStatus.DOWNLOADED.value: "Downloaded",
    IngestionStatus.PROCESSING.value: "Processing",
    IngestionStatus.COMPLETED.value: "Completed",
    IngestionStatus.ERROR.value: "Error",
    IngestionStatus.DELETING.value: "Deleting",
    IngestionStatus.REINDEXING.value: "Reindexing",
}


def _now() -> float:
    """The current time, rounded to the millisecond: RedisJSON does not round-trip every digit of a float."""
    return round(time.time(), 3)


def source_id(source: str) -> str:
    """The id of a source in the keys: the SHA-256 of its name (file name or URL)."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _prefix(agent_id: str) -> str:
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:{DEFAULT_INGESTION_KEY}:"


def _scope_segment(chat_id: str | None) -> str:
    return _AGENT_SEGMENT if chat_id is None else f"{_CHAT_SEGMENT}{chat_id}"


def format_key(agent_id: str, chat_id: str | None, source: str) -> str:
    """The key of the row of ``source`` in the knowledge base (``chat_id=None``) or in a chat of the agent."""
    return f"{_prefix(agent_id)}{_scope_segment(chat_id)}:{source_id(source)}"


def maintenance_key(agent_id: str) -> str:
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:{DEFAULT_INGESTION_KEY}_maintenance"


def vector_transfer_key(agent_id: str) -> str:
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:{DEFAULT_INGESTION_KEY}_vector_transfer"


def _glob_escaped(text: str) -> str:
    """The text as a literal in a Redis pattern."""
    return re.sub(r"([*?\[\]\\])", r"\\\1", text)


def _check_phase_ids(phase_ids: Iterable[str]) -> List[str]:
    phase_ids = list(phase_ids)
    for phase_id in phase_ids:
        if not isinstance(phase_id, str) or not PHASE_ID.match(phase_id):
            raise ValueError(f"Invalid phase id {phase_id!r}: lowercase letters, digits and underscores only")
    return phase_ids


async def _row_keys(agent_id: str, chat_id: Any = ALL_SCOPES) -> List[str]:
    """The keys of the rows of the agent: in the knowledge base (None), in a chat, or everywhere (``ALL_SCOPES``).

    A chat id may contain ``:``: the key is split from the right, so a chat is never taken for another.
    """
    prefix = _prefix(agent_id)
    wanted = None if chat_id is ALL_SCOPES else _scope_segment(chat_id)
    pattern = f"{_glob_escaped(prefix)}{_glob_escaped(wanted) + ':' if wanted is not None else ''}*"
    keys = []
    async for key in get_async_db().scan_iter(pattern):
        segment, _, sid = key[len(prefix):].rpartition(":")
        is_row = _SOURCE_ID.match(sid) and (
            segment == _AGENT_SEGMENT or (segment.startswith(_CHAT_SEGMENT) and len(segment) > len(_CHAT_SEGMENT))
        )
        if is_row and (wanted is None or segment == wanted):
            keys.append(key)
    return keys


def _unwrap(value: Any) -> Dict | None:
    if isinstance(value, list):
        return value[0] if value else None
    return value


async def create_status(
    agent_id: str,
    chat_id: str | None,
    source: str,
    *,
    type_: str,
    content_type: str | None = None,
    metadata: Dict | None = None,
) -> Dict:
    """Start the lifecycle of a source: the row is written from scratch, ``uploaded``, with an empty diary."""
    if type_ not in SOURCE_TYPES:
        raise ValueError(f"Invalid source type {type_!r}: one of {SOURCE_TYPES}")
    now = _now()
    row = {
        "source": source,
        "chat_id": chat_id,
        "type": type_,
        "content_type": content_type,
        "metadata": metadata or {},
        "status": IngestionStatus.UPLOADED.value,
        "phase": None,
        "error": None,
        "owner": None,
        "deleted_phases": [],
        "completed_phases": {},
        "created_at": now,
        "updated_at": now,
    }
    await crud.store(format_key(agent_id, chat_id, source), row)
    return row


async def get_status(agent_id: str, chat_id: str | None, source: str) -> Dict | None:
    return _unwrap(await crud.read(format_key(agent_id, chat_id, source)))


async def _write_paths(key: str, values: Dict[str, Any], deletions: Iterable[str] = ()) -> bool:
    """Write the paths of an existing row in one transaction, with its update time. False if the row does not exist:
    RedisJSON refuses to set a path of a missing key, so the row is never created."""
    pipeline = get_async_db().pipeline(transaction=True)
    for path in deletions:
        pipeline.json().delete(key, path)
    for path, value in values.items():
        pipeline.json().set(key, path, value)
    pipeline.json().set(key, "$.updated_at", _now())
    try:
        await pipeline.execute()
    except ResponseError:
        return False
    return True


async def update_status(agent_id: str, chat_id: str | None, source: str, **fields: Any) -> bool:
    """Change some fields of the row (and its update time). False if the row does not exist: it is never created.

    Fields: ``status`` (an ``IngestionStatus``), ``phase``, ``error``, ``owner``, ``deleted_phases``, ``content_type``.
    """
    unknown = set(fields) - _UPDATABLE_FIELDS
    if unknown:
        raise ValueError(f"Fields not updatable: {sorted(unknown)}")
    if "status" in fields:
        fields["status"] = IngestionStatus(fields["status"]).value
    values = {f"$.{name}": value for name, value in fields.items()}
    return await _write_paths(format_key(agent_id, chat_id, source), values)


async def touch_status(agent_id: str, chat_id: str | None, source: str) -> bool:
    """Refresh the update time of the row (the heartbeat of a worker). False if the row does not exist."""
    return await _write_paths(format_key(agent_id, chat_id, source), {})


async def record_phase(agent_id: str, chat_id: str | None, source: str, phase_id: str, entry: Dict) -> bool:
    """Write the diary entry of a phase. False if the row does not exist."""
    [phase_id] = _check_phase_ids([phase_id])
    return await _write_paths(format_key(agent_id, chat_id, source), {f"$.completed_phases.{phase_id}": entry})


async def reset_phase_entries(agent_id: str, chat_id: str | None, source: str, phase_ids: Iterable[str]) -> bool:
    """Remove the diary entries of the phases (they run again). False if the row does not exist."""
    paths = [f"$.completed_phases.{phase_id}" for phase_id in _check_phase_ids(phase_ids)]
    return await _write_paths(format_key(agent_id, chat_id, source), {}, deletions=paths)


async def list_statuses(agent_id: str, chat_id: Any = ALL_SCOPES) -> List[Dict]:
    """The rows of the agent: of its knowledge base (``chat_id=None``), of a chat, or of both (``ALL_SCOPES``)."""
    keys = await _row_keys(agent_id, chat_id)
    if not keys:
        return []
    values = await get_async_db().json().mget(keys, "$")
    return [row for row in (_unwrap(v) for v in values) if row]


async def reset_phase_entries_for_agent(agent_id: str, phase_ids: Iterable[str]) -> int:
    """Remove the diary entries of the phases from every row of the agent. Returns the number of rows changed."""
    phase_ids = _check_phase_ids(phase_ids)
    changed = 0
    for row in await list_statuses(agent_id):
        recorded = [phase_id for phase_id in phase_ids if phase_id in (row.get("completed_phases") or {})]
        if recorded and await reset_phase_entries(agent_id, row["chat_id"], row["source"], recorded):
            changed += 1
    return changed


async def clone_knowledge_base(source_agent_id: str, target_agent_id: str) -> int:
    """Copy to another agent the rows of the knowledge base that finished (``completed`` or ``error``): the rows of an
    agent being cloned, whose points and files are cloned too. The chats are not cloned. Returns the rows copied."""
    finished = {IngestionStatus.COMPLETED.value, IngestionStatus.ERROR.value}
    rows = [row for row in await list_statuses(source_agent_id, None) if row.get("status") in finished]
    for row in rows:
        await crud.store(format_key(target_agent_id, None, row["source"]), row)
    return len(rows)


async def delete_status(agent_id: str, chat_id: str | None, source: str) -> None:
    await get_async_db().delete(format_key(agent_id, chat_id, source))


async def _delete_keys(keys: List[str]) -> int:
    if keys:
        await get_async_db().delete(*keys)
    return len(keys)


async def destroy_chat(agent_id: str, chat_id: str) -> int:
    """Remove the rows of a chat of the agent. Returns the number of rows removed."""
    return await _delete_keys(await _row_keys(agent_id, chat_id))


async def destroy_all(agent_id: str) -> None:
    """Remove everything of the agent under the ingestion namespace (rows or not), its maintenance marker and the record
    of its vector transfer."""
    await _delete_keys([k async for k in get_async_db().scan_iter(f"{_glob_escaped(_prefix(agent_id))}*")])
    await clear_maintenance(agent_id)
    await end_vector_transfer(agent_id)


async def set_maintenance(agent_id: str, ttl: int) -> None:
    """Suspend the automatic resume of the agent for ``ttl`` seconds (> 0), or until ``clear_maintenance``."""
    if ttl <= 0:
        raise ValueError("The maintenance marker needs a positive TTL: it would never expire")
    await crud.store(maintenance_key(agent_id), {"since": _now()}, expire=ttl)


async def clear_maintenance(agent_id: str) -> None:
    await get_async_db().delete(maintenance_key(agent_id))


async def is_in_maintenance(agent_id: str) -> bool:
    return bool(await get_async_db().exists(maintenance_key(agent_id)))


async def start_vector_transfer(agent_id: str) -> float:
    """Record that the points of the agent are being copied to a new vector database; returns the start (``since``).

    The record has no TTL: if the transfer dies, its heartbeat (``beat``) stops, and the automatic resume takes it over.
    """
    since = _now()
    await crud.store(vector_transfer_key(agent_id), {"since": since, "beat": since})
    return since


async def beat_vector_transfer(agent_id: str, since: float) -> None:
    """The heartbeat of a running transfer (nothing if its record is gone)."""
    await crud.store(vector_transfer_key(agent_id), {"since": since, "beat": _now()}, xx=True)


async def get_vector_transfer(agent_id: str) -> Dict | None:
    """The record of the vector transfer of the agent (``since``, ``beat``), or None."""
    value = await crud.read(vector_transfer_key(agent_id))
    if isinstance(value, list):
        value = value[0] if value else None
    return value or None


async def end_vector_transfer(agent_id: str) -> None:
    await get_async_db().delete(vector_transfer_key(agent_id))
