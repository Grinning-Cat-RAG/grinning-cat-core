"""Characterization tests for ``CoreIngestionEngine`` (the upstream-parity seam).

These pin the CURRENT behaviour of the engine before it is refactored into the
per-part ingestion abstractions: ``ingest_file`` must keep delegating to
``RabbitHole.ingest_file`` with exactly the same arguments, and ``run`` must
keep returning ``True``/``False`` around ``lizard.embed_all_in_cheshire_cats``.
They assert the outward call contract, never the engine internals.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cat.services.factory.ingestion import CoreIngestionEngine


async def test_ingest_file_delegates_to_rabbit_hole_with_explicit_arguments():
    engine = CoreIngestionEngine()
    cat = MagicMock()
    cat.rabbit_hole.ingest_file = AsyncMock()
    file = MagicMock()
    metadata = {"source": "doc.txt"}

    await engine.ingest_file(
        cat,
        file,
        filename="doc.txt",
        metadata=metadata,
        store_file=False,
        content_type="text/plain",
    )

    cat.rabbit_hole.ingest_file.assert_awaited_once_with(
        cat=cat,
        file=file,
        filename="doc.txt",
        metadata=metadata,
        store_file=False,
        content_type="text/plain",
    )


async def test_ingest_file_defaults_metadata_to_empty_dict():
    engine = CoreIngestionEngine()
    cat = MagicMock()
    cat.rabbit_hole.ingest_file = AsyncMock()
    file = MagicMock()

    await engine.ingest_file(cat, file)

    cat.rabbit_hole.ingest_file.assert_awaited_once_with(
        cat=cat,
        file=file,
        filename=None,
        metadata={},
        store_file=True,
        content_type=None,
    )


async def test_ingest_file_builds_a_rabbit_hole_when_cat_has_none(monkeypatch):
    engine = CoreIngestionEngine()
    file = MagicMock()
    fake_rabbit_hole = MagicMock()
    fake_rabbit_hole.ingest_file = AsyncMock()
    monkeypatch.setattr("cat.rabbit_hole.RabbitHole", lambda: fake_rabbit_hole)
    cat = SimpleNamespace()

    await engine.ingest_file(cat, file, filename="doc.txt")

    fake_rabbit_hole.ingest_file.assert_awaited_once_with(
        cat=cat,
        file=file,
        filename="doc.txt",
        metadata={},
        store_file=True,
        content_type=None,
    )


async def test_run_returns_true_when_embedding_succeeds():
    engine = CoreIngestionEngine()
    lizard = MagicMock()
    lizard.embed_all_in_cheshire_cats = AsyncMock()

    result = await engine.run(lizard)

    assert result is True
    lizard.embed_all_in_cheshire_cats.assert_awaited_once_with()


async def test_run_logs_and_returns_false_when_embedding_raises(monkeypatch):
    engine = CoreIngestionEngine()
    lizard = MagicMock()
    lizard.embed_all_in_cheshire_cats = AsyncMock(side_effect=RuntimeError("boom"))
    log_error = MagicMock()
    monkeypatch.setattr("cat.services.factory.ingestion.log.error", log_error)

    result = await engine.run(lizard)

    assert result is False
    log_error.assert_called_once()
