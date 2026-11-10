"""The opt-in token-budget split of the embedding phase: pure unit contract plus the ingestion integration."""
import inspect

from langchain_core.documents import Document

from cat.env import get_env_int
from cat.rabbit_hole import RabbitHole
from cat.rabbit_hole.split import DEFAULT_SPLIT_MAX_TOKENS, embedder_token_budget, split_oversized

LONG_TEXT = "Alice was beginning to get very tired of sitting by her sister on the bank. " * 20


def _docs() -> list[Document]:
    return [
        Document(page_content="short one", metadata={"source": "a", "start_index": 0, "chunk_index": 0}),
        Document(page_content=LONG_TEXT, metadata={"source": "b", "start_index": 11, "chunk_index": 1, "custom": 7}),
        Document(page_content="short two", metadata={"source": "c", "start_index": 11 + len(LONG_TEXT), "chunk_index": 2}),
    ]


def test_split_oversized_is_deterministic():
    first = split_oversized(_docs(), max_tokens=50)
    second = split_oversized(_docs(), max_tokens=50)

    assert first == second


def test_split_oversized_is_idempotent():
    once = split_oversized(_docs(), max_tokens=50)

    assert split_oversized(once, max_tokens=50) == once


def test_split_oversized_keeps_indices_contiguous_and_loses_no_text():
    split = split_oversized([_docs()[1]], max_tokens=50)

    assert len(split) > 1
    assert [document.metadata["chunk_index"] for document in split] == list(range(len(split)))
    assert "".join(document.page_content for document in split) == LONG_TEXT

    offsets = [document.metadata["start_index"] for document in split]
    assert offsets == [11 + sum(len(document.page_content) for document in split[:index])
                       for index in range(len(split))]


def test_split_oversized_copies_metadata():
    split = split_oversized([_docs()[1]], max_tokens=50)

    for document in split:
        assert document.metadata["source"] == "b"
        assert document.metadata["custom"] == 7
        assert "start_index" in document.metadata


def test_split_oversized_leaves_small_documents_and_disabled_budget_untouched():
    documents = _docs()

    untouched = split_oversized(documents, max_tokens=0)

    assert [document.page_content for document in untouched] == [document.page_content for document in documents]
    assert [document.metadata["chunk_index"] for document in untouched] == [0, 1, 2]


def _budget_with(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CAT_INGESTION_SPLIT_MAX_TOKENS", raising=False)
    else:
        monkeypatch.setenv("CAT_INGESTION_SPLIT_MAX_TOKENS", str(value))
    return embedder_token_budget()


def test_embedder_token_budget_uses_default_and_env(monkeypatch):
    assert list(inspect.signature(embedder_token_budget).parameters) == []
    assert get_env_int("CAT_INGESTION_SPLIT_MAX_TOKENS") == DEFAULT_SPLIT_MAX_TOKENS
    assert _budget_with(monkeypatch, None) == DEFAULT_SPLIT_MAX_TOKENS
    assert _budget_with(monkeypatch, 64) == 64


async def test_flag_off_keeps_the_chunk_count_identical(cheshire_cat, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_SPLIT_OVERSIZED", "false")
    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    points = await rabbit_hole.store_documents(
        docs=[Document(page_content=LONG_TEXT, metadata={"start_index": 0, "chunk_index": 0})],
        source="split-off.txt",
    )

    assert len(points) == 1
    assert points[0].payload["metadata"]["chunk_index"] == 0


async def test_flag_on_splits_an_oversized_document_with_contiguous_indices(cheshire_cat, monkeypatch):
    monkeypatch.setenv("CAT_INGESTION_SPLIT_OVERSIZED", "true")
    monkeypatch.setenv("CAT_INGESTION_SPLIT_MAX_TOKENS", "50")
    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    points = await rabbit_hole.store_documents(
        docs=[Document(page_content=LONG_TEXT, metadata={"start_index": 0, "chunk_index": 0})],
        source="split-on.txt",
    )

    assert len(points) > 1
    metadata = [point.payload["metadata"] for point in points]
    assert [entry["chunk_index"] for entry in metadata] == list(range(len(points)))
    offsets = [entry["start_index"] for entry in metadata]
    assert offsets == sorted(offsets)
    assert "".join(point.payload["page_content"] for point in points) == LONG_TEXT
