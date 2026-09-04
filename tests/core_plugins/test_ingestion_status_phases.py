"""Tests for the phase-aware ingestion_status registry.

Covers ``set_phase``, the merge semantics of ``set_status`` (phase /
embedder_name / chunker_name are only updated when provided), and the
``claim_completed`` mode of ``claim_source_for_resume`` (engine re-embed of
completed rows). Uses the autouse Redis db=1 fixture.
"""
from cat.core_plugins.ingestion_status.registry import (
    PHASE_EMBEDDING,
    PHASE_PARSING_CHUNKING,
    IngestionStatus,
    claim_source_for_resume,
    get_status,
    set_phase,
    set_status,
)


async def test_set_phase_records_processing_and_embedder():
    captured = await set_phase(
        "agent_1", "agent", "doc.pdf",
        PHASE_EMBEDDING, embedder_name="all-MiniLM-L6-v2",
    )
    assert captured["status"] == IngestionStatus.PROCESSING.value
    assert captured["phase"] == PHASE_EMBEDDING
    assert captured["embedder_name"] == "all-MiniLM-L6-v2"

    # reload from Redis
    doc = await get_status("agent_1", "agent", "doc.pdf")
    assert doc["phase"] == PHASE_EMBEDDING
    assert doc["embedder_name"] == "all-MiniLM-L6-v2"
    assert doc["status"] == IngestionStatus.PROCESSING.value


async def test_set_phase_parsing_chunking_records_chunker():
    captured = await set_phase(
        "agent_1", "agent", "doc.pdf",
        PHASE_PARSING_CHUNKING, chunker_name="RecursiveTextChunker",
    )
    assert captured["status"] == IngestionStatus.PROCESSING.value
    assert captured["phase"] == PHASE_PARSING_CHUNKING
    assert captured["chunker_name"] == "RecursiveTextChunker"


async def test_set_status_merge_keeps_phase_and_embedder():
    # engine records the phase + embedder
    await set_phase("agent_1", "agent", "doc.pdf", PHASE_EMBEDDING, embedder_name="emb-v2")

    # a lifecycle hook writes COMPLETED without phase/embedder -> must NOT clobber
    await set_status(
        "agent_1", "agent", "doc.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
    )
    doc = await get_status("agent_1", "agent", "doc.pdf")
    assert doc["status"] == IngestionStatus.COMPLETED.value
    # merge semantics: embedder_name survives, phase survives (unless cleared)
    assert doc["embedder_name"] == "emb-v2"
    assert doc["phase"] == PHASE_EMBEDDING


async def test_set_status_clear_phase_removes_it():
    await set_phase("agent_1", "agent", "doc.pdf", PHASE_EMBEDDING, embedder_name="emb-v2")
    # terminal write with clear_phase drops the phase but keeps embedder_name
    await set_status(
        "agent_1", "agent", "doc.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
        clear_phase=True,
    )
    doc = await get_status("agent_1", "agent", "doc.pdf")
    assert doc["status"] == IngestionStatus.COMPLETED.value
    assert "phase" not in doc
    assert doc["embedder_name"] == "emb-v2"


async def test_claim_completed_allowed_only_when_requested():
    await set_status(
        "agent_1", "agent", "doc.pdf",
        type_="file", status=IngestionStatus.COMPLETED,
        embedder_name="old-emb",
    )
    # default (resume) does NOT claim completed rows
    claimed = await claim_source_for_resume(
        "agent_1", "agent", "doc.pdf", stale_after=0.0, owner="resume",
    )
    assert claimed is None

    # engine claims completed rows explicitly, transitions to processing
    claimed = await claim_source_for_resume(
        "agent_1", "agent", "doc.pdf",
        stale_after=0.0, owner="engine", claim_completed=True,
    )
    assert claimed is not None
    assert claimed["status"] == IngestionStatus.PROCESSING.value
    assert claimed["embedder_name"] == "old-emb"  # preserved by merge

    # a second worker cannot claim the now-processing row while it is fresh
    claimed_again = await claim_source_for_resume(
        "agent_1", "agent", "doc.pdf",
        stale_after=3600.0, owner="engine2", claim_completed=True,
    )
    assert claimed_again is None


async def test_claim_completed_does_not_bypass_in_flight_guard():
    # a fresh processing row (not completed) is never re-claimed even with claim_completed
    await set_phase("agent_1", "agent", "doc.pdf", PHASE_EMBEDDING, embedder_name="emb-v2")
    claimed = await claim_source_for_resume(
        "agent_1", "agent", "doc.pdf", stale_after=3600.0, owner="engine", claim_completed=True,
    )
    assert claimed is None
