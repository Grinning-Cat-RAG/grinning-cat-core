"""Regression tests for the ingestion no-op loop (resume <-> phase machine).

A source whose phase diary is already complete but whose status was left
``processing`` (e.g. a run interrupted between the last diary write and the
terminal write) used to be re-handed by every resume sweep forever:
``reembed_sources`` took the "up to date" ``continue`` without ever writing a
terminal status. These tests lock the fix:

- a fully-processed ``processing`` row is terminalized to ``completed`` in ONE
  pass and is not handed again on the next sweep;
- a row with a genuinely stale phase still runs the stale phase;
- an ``error`` row is never resurrected (nor re-handed when its diary is fresh).

Redis keys live under ``agents:effingloop_<uuid>:`` and are destroyed in
``finally``. The host ``tests/conftest.py`` forces ``CAT_REDIS_DB=1``.
"""
import uuid
from types import SimpleNamespace

from cat.core_plugins.cat_efficient_ingestion import reembed, resume
from cat.core_plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    get_status,
    set_status,
    status_key,
)
from cat.db import crud
from cat.looking_glass.models import StoredSourceWithMetadata
from cat.services.memory.models import VectorMemoryType

_DIARY = {
    "parsing_chunking": {"marker": "c", "settings_version": None, "deps": {}},
    "embedding": {
        "marker": "e",
        "settings_version": None,
        "deps": {"parsing_chunking": "c"},
    },
}


def _agent_id() -> str:
    return f"effingloop_{uuid.uuid4().hex}"


async def _seed_master_key(agent_id: str) -> None:
    await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])


async def _drop(agent_id: str) -> None:
    await crud.destroy(f"agents:{agent_id}:*")


async def _seed(agent: str, source: str, status: IngestionStatus) -> None:
    await set_status(
        agent, "agent", source, type_="file", status=status, completed_phases=_DIARY,
    )


def _stored_chunk_point(source_name: str = "doc.pdf"):
    """One genuinely stored chunk point (what the store-grounded guard inspects)."""
    return SimpleNamespace(
        id="stored-chunk-0",
        payload={
            "page_content": "stored chunk",
            "metadata": {"source": source_name},
        },
        vector={},
    )


class _FileManager:
    """Minimal file manager so a forced parse can never hit an AttributeError."""

    def read_file(self, *args, **kwargs):
        return b"%PDF-1.4 stub"

    def write_file(self, *args, **kwargs):
        return True

    def remove_file(self, *args, **kwargs):
        return None

    def file_exists(self, *args, **kwargs):
        return True

    def list_files(self, *args, **kwargs):
        return ["doc.pdf"]


def _ccat(agent: str, points=None):
    # ``points=None`` seeds ONE stored chunk so the store-grounded freshness
    # guard (``_source_has_stored_points``) sees a populated store on the
    # diary-fresh path. An explicit list — including ``[]`` — is used verbatim,
    # so the guard test can simulate a wiped store.
    chunks = [_stored_chunk_point()] if points is None else points

    async def _get_points(*args, **kwargs):
        return list(chunks), None

    async def _write_points(*args, **kwargs):
        return [], None

    async def embedder():
        return SimpleNamespace(name="test-embedder")

    async def file_handlers():
        return {}

    async def save_file(*args, **kwargs):
        return True

    return SimpleNamespace(
        _id=agent,
        agent_key=agent,
        vector_memory_handler=SimpleNamespace(
            get_all_tenant_points=_get_points,
            add_points_to_tenant=_write_points,
            delete_tenant_points=_write_points,
        ),
        file_manager=_FileManager(),
        file_handlers=file_handlers,
        save_file=save_file,
        embedder=embedder,
        chunker=SimpleNamespace(name="test-chunker"),
        rabbit_hole=SimpleNamespace(),
    )


def _source(name: str = "doc.pdf"):
    return StoredSourceWithMetadata(name=name, path=name, content=None, metadata={})


async def test_up_to_date_processing_row_terminalizes_once(monkeypatch):
    """All phases fresh + status processing -> completed in one pass, then no-op."""
    agent = _agent_id()
    await _seed_master_key(agent)
    source_name = "doc.pdf"
    await _seed(agent, source_name, IngestionStatus.PROCESSING)

    async def fake_probe(*args, **kwargs):
        return []

    monkeypatch.setattr(reembed, "_probe_pending", fake_probe)
    ccat = _ccat(agent)
    try:
        await reembed.reembed_sources(
            ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
            stale_after=0.0, caller_cat=ccat,
        )
        doc = await get_status(agent, "agent", source_name)
        assert doc is not None
        assert doc["status"] == IngestionStatus.COMPLETED.value
        assert "phase" not in doc
        first_updated = doc["updated_at"]

        # Second pass: terminal row, no stale phase -> no write at all.
        await reembed.reembed_sources(
            ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
            stale_after=0.0, caller_cat=ccat,
        )
        doc = await get_status(agent, "agent", source_name)
        assert doc is not None
        assert doc["status"] == IngestionStatus.COMPLETED.value
        assert doc["updated_at"] == first_updated, "terminal row was rewritten (write storm)"
    finally:
        await _drop(agent)


async def test_terminal_source_is_not_handed_again_by_resume(monkeypatch):
    """A completed row is not a resume candidate, so it is never handed again."""
    agent = _agent_id()
    await _seed_master_key(agent)
    source_name = "doc.pdf"
    await _seed(agent, source_name, IngestionStatus.COMPLETED)

    handed = []

    async def fake_reembed(*args, **kwargs):
        handed.append(args)

    monkeypatch.setattr(resume, "reembed_sources", fake_reembed)

    class _Lizard:
        async def get_cheshire_cat(self, agent_id):
            return _ccat(agent_id)

    try:
        await resume._resume_agent(_Lizard(), agent)
        await resume._resume_agent(_Lizard(), agent)
        assert handed == [], "a terminal completed row was re-handed by the resume sweep"
    finally:
        await _drop(agent)


async def test_stale_phase_still_runs(monkeypatch):
    """A row with a genuinely stale phase is still claimed and processed."""
    agent = _agent_id()
    await _seed_master_key(agent)
    source_name = "doc.pdf"
    await _seed(agent, source_name, IngestionStatus.PROCESSING)

    probes = iter([[{"phase": "embedding"}], []])

    async def fake_probe(*args, **kwargs):
        return next(probes, [])

    embedded = []

    async def fake_embed(*args, **kwargs):
        embedded.append(args)

    monkeypatch.setattr(reembed, "_probe_pending", fake_probe)
    monkeypatch.setattr(reembed, "_embed_phase", fake_embed)

    point = SimpleNamespace(payload={"metadata": {"source": source_name}})
    ccat = _ccat(agent, points=[point])
    try:
        await reembed.reembed_sources(
            ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
            stale_after=0.0, caller_cat=ccat,
        )
        assert embedded, "a stale phase was skipped"
        doc = await get_status(agent, "agent", source_name)
        assert doc is not None
        assert doc["status"] == IngestionStatus.COMPLETED.value
    finally:
        await _drop(agent)


async def test_error_row_is_not_resurrected_by_up_to_date_branch(monkeypatch):
    """An up-to-date ERROR row is never rewritten to completed."""
    agent = _agent_id()
    await _seed_master_key(agent)
    source_name = "doc.pdf"
    await _seed(agent, source_name, IngestionStatus.ERROR)
    seeded = await get_status(agent, "agent", source_name)
    assert seeded is not None
    before = seeded["updated_at"]

    async def fake_probe(*args, **kwargs):
        return []

    monkeypatch.setattr(reembed, "_probe_pending", fake_probe)
    ccat = _ccat(agent)
    try:
        await reembed.reembed_sources(
            ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
            stale_after=0.0, caller_cat=ccat,
        )
        doc = await get_status(agent, "agent", source_name)
        assert doc is not None
        assert doc["status"] == IngestionStatus.ERROR.value
        assert doc["updated_at"] == before
    finally:
        await _drop(agent)


async def test_fresh_error_row_is_not_handed_by_resume(monkeypatch):
    """An ERROR row whose diary is fresh is terminal: never re-handed nor retried."""
    agent = _agent_id()
    await _seed_master_key(agent)
    source_name = "doc.pdf"
    await _seed(agent, source_name, IngestionStatus.ERROR)

    async def fresh_probe(ccat, entry):
        return []

    handed = []

    async def fake_reembed(*args, **kwargs):
        handed.append(args)

    monkeypatch.setattr(resume, "_pending_phases", fresh_probe)
    monkeypatch.setattr(resume, "reembed_sources", fake_reembed)

    class _Lizard:
        async def get_cheshire_cat(self, agent_id):
            return _ccat(agent_id)

    try:
        await resume._resume_agent(_Lizard(), agent)
        doc = await get_status(agent, "agent", source_name)
        assert doc is not None
        assert doc["status"] == IngestionStatus.ERROR.value
        assert handed == [], "a fresh error row was resurrected/re-handed"
    finally:
        await _drop(agent)


async def test_fresh_diary_with_empty_store_forces_reparse(monkeypatch):
    """A fresh diary is NOT trusted when the store holds zero chunk points.

    The store-grounded freshness guard exists for exactly this case: a wipe
    that leaves the phase diary intact would otherwise make every phase look
    fresh, the embedder would never run again, and later phases would fail
    forever. The guard must force ``parsing_chunking`` (and the matching
    ``embedding`` in the same pass) even though the probe reports nothing stale.
    """

    agent = _agent_id()
    await _seed_master_key(agent)
    source_name = "doc.pdf"
    await _seed(agent, source_name, IngestionStatus.PROCESSING)

    async def fresh_probe(*args, **kwargs):
        return []

    parsed = []
    embedded = []

    async def fake_parse(*args, **kwargs):
        parsed.append(args[3])
        return args[3], [_stored_chunk_point(args[3])], 1, 1

    async def fake_embed(*args, **kwargs):
        embedded.append(args[2])

    monkeypatch.setattr(reembed, "_probe_pending", fresh_probe)
    monkeypatch.setattr(reembed, "_run_parsing_phase", fake_parse)
    monkeypatch.setattr(reembed, "_embed_phase", fake_embed)

    ccat = _ccat(agent, points=[])
    try:
        await reembed.reembed_sources(
            ccat, VectorMemoryType.DECLARATIVE, [_source(source_name)],
            stale_after=0.0, caller_cat=ccat,
        )
        assert parsed, "fresh diary + EMPTY store did not force parsing_chunking"
        assert embedded, "the forced re-parse was not followed by the forced re-embed"
        doc = await get_status(agent, "agent", source_name)
        assert doc is not None
        assert doc["status"] == IngestionStatus.COMPLETED.value
    finally:
        await _drop(agent)
