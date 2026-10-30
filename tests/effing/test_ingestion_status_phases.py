"""Tests for the phase-aware ingestion-status registry.

Covers ``set_phase``, the merge semantics of ``set_status`` (phase /
embedder_name / chunker_name are only updated when provided), the clock-free
``completed_phases`` diary (``record_phase_completed`` /
``get_completed_phases``), and the ``claim_completed`` mode of
``claim_source_for_resume`` (engine re-embed of completed rows). Uses the
autouse Redis db=1 fixture.
"""
import uuid

import pytest

from cat.db import crud
from cat.core_plugins.cat_efficient_ingestion.registry import (
    PHASE_EMBEDDING,
    PHASE_PARSING_CHUNKING,
    IngestionStatus,
    backfill_completed_phases,
    claim_source_for_resume,
    get_completed_phases,
    get_status,
    record_phase_completed,
    set_phase,
    set_status,
)


#: Unique agent id: every seeded key lives under ``agents:effingtest_<uuid>:``.
AGENT = f"effingtest_{uuid.uuid4().hex}"


@pytest.fixture(autouse=True)
async def _ensure_agent_master_key():
    """The ``ingestion_canceled`` guard makes ``set_status`` a no-op when the
    agent master key ``agents:<agent>:agent`` is absent. These tests call
    ``set_status`` directly, so create the master key first (the production
    contract: ``set_status`` is only called for existing agents)."""
    await crud.store(f"agents:{AGENT}:agent", [{"name": "x", "value": 1}])
    yield
    await crud.destroy(f"agents:{AGENT}:*")


async def test_set_phase_records_processing_and_embedder():
    captured = await set_phase(
        AGENT, "agent", "doc.pdf",
        PHASE_EMBEDDING, embedder_name="all-MiniLM-L6-v2",
    )
    assert captured["status"] == IngestionStatus.PROCESSING.value
    assert captured["phase"] == PHASE_EMBEDDING
    assert captured["embedder_name"] == "all-MiniLM-L6-v2"

    # reload from Redis
    doc = await get_status(AGENT, "agent", "doc.pdf")
    assert doc["phase"] == PHASE_EMBEDDING
    assert doc["embedder_name"] == "all-MiniLM-L6-v2"
    assert doc["status"] == IngestionStatus.PROCESSING.value


async def test_set_phase_parsing_chunking_records_chunker():
    captured = await set_phase(
        AGENT, "agent", "doc.pdf",
        PHASE_PARSING_CHUNKING, chunker_name="RecursiveTextChunker",
    )
    assert captured["status"] == IngestionStatus.PROCESSING.value
    assert captured["phase"] == PHASE_PARSING_CHUNKING
    assert captured["chunker_name"] == "RecursiveTextChunker"


async def test_set_status_merge_keeps_phase_and_embedder():
    # engine records the phase + embedder
    await set_phase(AGENT, "agent", "doc.pdf", PHASE_EMBEDDING, embedder_name="emb-v2")

    # a lifecycle hook writes COMPLETED without phase/embedder -> must NOT clobber
    await set_status(
        AGENT, "agent", "doc.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
    )
    doc = await get_status(AGENT, "agent", "doc.pdf")
    assert doc["status"] == IngestionStatus.COMPLETED.value
    # merge semantics: embedder_name survives, phase survives (unless cleared)
    assert doc["embedder_name"] == "emb-v2"
    assert doc["phase"] == PHASE_EMBEDDING


async def test_set_status_clear_phase_removes_it():
    await set_phase(AGENT, "agent", "doc.pdf", PHASE_EMBEDDING, embedder_name="emb-v2")
    # terminal write with clear_phase drops the phase but keeps embedder_name
    await set_status(
        AGENT, "agent", "doc.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
        clear_phase=True,
    )
    doc = await get_status(AGENT, "agent", "doc.pdf")
    assert doc["status"] == IngestionStatus.COMPLETED.value
    assert "phase" not in doc
    assert doc["embedder_name"] == "emb-v2"


async def test_claim_completed_allowed_only_when_requested():
    await set_status(
        AGENT, "agent", "doc.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
        embedder_name="old-emb",
    )
    # default (resume) does NOT claim completed rows
    claimed = await claim_source_for_resume(
        AGENT, "agent", "doc.pdf", stale_after=0.0, owner="resume",
    )
    assert claimed is None

    # engine claims completed rows explicitly, transitions to processing
    claimed = await claim_source_for_resume(
        AGENT, "agent", "doc.pdf",
        stale_after=0.0, owner="engine", claim_completed=True,
    )
    assert claimed is not None
    assert claimed["status"] == IngestionStatus.PROCESSING.value
    assert claimed["embedder_name"] == "old-emb"  # preserved by merge

    # a second worker cannot claim the now-processing row while it is fresh
    claimed_again = await claim_source_for_resume(
        AGENT, "agent", "doc.pdf",
        stale_after=3600.0, owner="engine2", claim_completed=True,
    )
    assert claimed_again is None


async def test_claim_completed_does_not_bypass_in_flight_guard():
    # a fresh processing row (not completed) is never re-claimed even with claim_completed
    await set_phase(AGENT, "agent", "doc.pdf", PHASE_EMBEDDING, embedder_name="emb-v2")
    claimed = await claim_source_for_resume(
        AGENT, "agent", "doc.pdf", stale_after=3600.0, owner="engine", claim_completed=True,
    )
    assert claimed is None


# ---------------------------------------------------------------------------
# Clock-free completed_phases diary (record_phase_completed / get_completed_phases)
# ---------------------------------------------------------------------------


async def test_completed_phases_two_phases_persist():
    # record parsing_chunking then embedding, with distinct markers + deps
    await record_phase_completed(
        AGENT, "agent", "diary.pdf", PHASE_PARSING_CHUNKING,
        marker="chunker-v1", settings_version="t1",
    )
    await record_phase_completed(
        AGENT, "agent", "diary.pdf", PHASE_EMBEDDING,
        marker="emb-v1", settings_version="t2",
        deps={PHASE_PARSING_CHUNKING: "chunker-v1"},
    )

    doc = await get_status(AGENT, "agent", "diary.pdf")
    diary = get_completed_phases(doc)
    assert set(diary) == {PHASE_PARSING_CHUNKING, PHASE_EMBEDDING}
    assert diary[PHASE_PARSING_CHUNKING] == {
        "marker": "chunker-v1", "settings_version": "t1", "deps": {},
    }
    assert diary[PHASE_EMBEDDING] == {
        "marker": "emb-v1", "settings_version": "t2",
        "deps": {PHASE_PARSING_CHUNKING: "chunker-v1"},
    }


async def test_completed_phases_rerecord_replaces_only_own_entry():
    await record_phase_completed(
        AGENT, "agent", "diary.pdf", PHASE_PARSING_CHUNKING,
        marker="chunker-v1", settings_version="t1",
    )
    await record_phase_completed(
        AGENT, "agent", "diary.pdf", PHASE_EMBEDDING,
        marker="emb-v1", settings_version="t2",
    )
    # re-record ONLY embedding with a new marker -> its entry is replaced,
    # parsing_chunking preserved, diary still has exactly two entries
    await record_phase_completed(
        AGENT, "agent", "diary.pdf", PHASE_EMBEDDING,
        marker="emb-v2", settings_version="t3",
    )

    doc = await get_status(AGENT, "agent", "diary.pdf")
    diary = get_completed_phases(doc)
    assert set(diary) == {PHASE_PARSING_CHUNKING, PHASE_EMBEDDING}
    assert diary[PHASE_EMBEDDING]["marker"] == "emb-v2"
    assert diary[PHASE_EMBEDDING]["settings_version"] == "t3"
    assert diary[PHASE_PARSING_CHUNKING]["marker"] == "chunker-v1"
    assert diary[PHASE_PARSING_CHUNKING]["settings_version"] == "t1"


async def test_completed_phases_preserved_by_lifecycle_write():
    # a lifecycle write WITHOUT completed_phases must not wipe the diary
    await record_phase_completed(
        AGENT, "agent", "diary.pdf", PHASE_PARSING_CHUNKING,
        marker="chunker-v1", settings_version="t1",
    )
    await set_status(
        AGENT, "agent", "diary.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
    )

    doc = await get_status(AGENT, "agent", "diary.pdf")
    assert doc["status"] == IngestionStatus.COMPLETED.value
    diary = get_completed_phases(doc)
    assert diary == {
        PHASE_PARSING_CHUNKING: {
            "marker": "chunker-v1", "settings_version": "t1", "deps": {},
        }
    }


async def test_completed_phases_survive_clear_phase():
    # clear_phase only drops `phase`; the diary is a separate field and stays
    await record_phase_completed(
        AGENT, "agent", "diary.pdf", PHASE_EMBEDDING,
        marker="emb-v1", settings_version="t2",
    )
    await set_status(
        AGENT, "agent", "diary.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
        clear_phase=True,
    )

    doc = await get_status(AGENT, "agent", "diary.pdf")
    assert "phase" not in doc
    diary = get_completed_phases(doc)
    assert diary == {
        PHASE_EMBEDDING: {"marker": "emb-v1", "settings_version": "t2", "deps": {}},
    }


async def test_completed_phases_deps_none_defaults_empty():
    await record_phase_completed(
        AGENT, "agent", "diary.pdf", PHASE_EMBEDDING, marker="emb-v1",
    )
    doc = await get_status(AGENT, "agent", "diary.pdf")
    assert get_completed_phases(doc)[PHASE_EMBEDDING]["deps"] == {}


async def test_completed_phases_invalid_phase_is_noop():
    # missing/invalid phase -> no crash, no entry
    captured = await record_phase_completed(AGENT, "agent", "diary.pdf", None, marker="x")
    assert captured == {}
    assert await get_status(AGENT, "agent", "diary.pdf") is None

    await record_phase_completed(AGENT, "agent", "diary.pdf", "", marker="x")
    assert await get_status(AGENT, "agent", "diary.pdf") is None


async def test_set_phase_forwards_completed_phases():
    # set_phase accepts and forwards the diary updates
    await set_phase(
        AGENT, "agent", "diary.pdf", PHASE_EMBEDDING,
        embedder_name="emb-v2",
        completed_phases={
            PHASE_PARSING_CHUNKING: {
                "marker": "chunker-v1", "settings_version": "t1", "deps": {},
            }
        },
    )
    doc = await get_status(AGENT, "agent", "diary.pdf")
    assert doc["phase"] == PHASE_EMBEDDING
    assert get_completed_phases(doc)[PHASE_PARSING_CHUNKING]["marker"] == "chunker-v1"


# ---------------------------------------------------------------------------
# Legacy backfill (backfill_completed_phases)
# ---------------------------------------------------------------------------


async def test_backfill_completed_phases_sentinel_for_legacy_completed():
    # a completed row written BEFORE the diary existed carries no diary
    await set_status(
        AGENT, "agent", "legacy.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
    )
    doc = await get_status(AGENT, "agent", "legacy.pdf")
    assert doc["status"] == IngestionStatus.COMPLETED.value
    assert get_completed_phases(doc) == {}

    phases = [PHASE_PARSING_CHUNKING, PHASE_EMBEDDING]
    diary = await backfill_completed_phases(doc, phases)
    # sentinel: every phase present, markers/versions unknown (None), deps {}
    assert diary == {
        phase: {"marker": None, "settings_version": None, "deps": {}}
        for phase in phases
    }


async def test_backfill_completed_phases_non_completed_returns_empty():
    # a processing row (in flight) has no diary and gets NO backfill
    await set_phase(AGENT, "agent", "pending.pdf", PHASE_EMBEDDING, embedder_name="emb-v1")
    doc = await get_status(AGENT, "agent", "pending.pdf")
    assert doc["status"] == IngestionStatus.PROCESSING.value
    assert await backfill_completed_phases(
        doc, [PHASE_PARSING_CHUNKING, PHASE_EMBEDDING]
    ) == {}

    # an uploaded row likewise
    await set_status(
        AGENT, "agent", "fresh.pdf",
        type_="file", status=IngestionStatus.UPLOADED,
    )
    doc = await get_status(AGENT, "agent", "fresh.pdf")
    assert await backfill_completed_phases(
        doc, [PHASE_PARSING_CHUNKING, PHASE_EMBEDDING]
    ) == {}


async def test_backfill_completed_phases_keeps_existing_diary():
    # a completed row that ALREADY has a diary must NOT be overwritten by the
    # sentinel backfill (stale-state protection)
    await record_phase_completed(
        AGENT, "agent", "diary.pdf", PHASE_PARSING_CHUNKING,
        marker="chunker-v1", settings_version="t1",
    )
    await set_status(
        AGENT, "agent", "diary.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
    )
    doc = await get_status(AGENT, "agent", "diary.pdf")
    assert doc["status"] == IngestionStatus.COMPLETED.value
    diary = await backfill_completed_phases(
        doc, [PHASE_PARSING_CHUNKING, PHASE_EMBEDDING]
    )
    # unchanged: only the recorded phase, no sentinel for the missing one
    assert diary == {
        PHASE_PARSING_CHUNKING: {
            "marker": "chunker-v1", "settings_version": "t1", "deps": {},
        }
    }


async def test_backfill_completed_phases_none_doc_is_safe():
    # adversarial input: doc=None -> {} without raising
    assert await backfill_completed_phases(None, [PHASE_PARSING_CHUNKING, PHASE_EMBEDDING]) == {}


async def test_backfill_completed_phases_malformed_doc_is_safe():
    # adversarial input: doc without a status -> {} without raising
    assert await backfill_completed_phases({}, [PHASE_PARSING_CHUNKING, PHASE_EMBEDDING]) == {}
    assert await backfill_completed_phases({"status": "completed"}, None) == {}
