"""Tests for the ingestion-engine seam (the core routes call the engine, not
``rabbit_hole.ingest_file``).

With no plugin the core resolves ``CoreIngestionEngine`` whose ``ingest_file``
wraps the original ``rabbit_hole.ingest_file`` (upstream parity). The
``efficient_ingestion`` engine runs the two-phase phase machine instead.
"""

from io import BytesIO

from cat.services.factory.ingestion import CoreIngestionEngine, resolve_ingestion_engine


async def test_core_engine_wraps_rabbit_hole_ingest_file(cheshire_cat, monkeypatch):
    """CoreIngestionEngine.ingest_file delegates to the rabbit_hole ingestion."""
    engine = CoreIngestionEngine()

    called = {}

    async def fake_ingest_file(cat=None, file=None, filename=None, metadata=None,
                               store_file=True, content_type=None):
        called.update({"cat": cat, "file": file, "filename": filename,
                       "store_file": store_file})

    monkeypatch.setattr(cheshire_cat.rabbit_hole, "ingest_file", fake_ingest_file)

    await engine.ingest_file(
        cat=cheshire_cat, file=BytesIO(b"hello"), filename="a.txt",
    )

    assert called["filename"] == "a.txt"
    assert called["store_file"] is True


async def test_engine_seam_resolves_and_calls_ingest(lizard):
    """resolve_ingestion_engine returns an engine exposing ingest_file + run."""
    engine = await resolve_ingestion_engine(lizard)
    assert engine is not None
    assert hasattr(engine, "ingest_file")
    assert hasattr(engine, "run")
