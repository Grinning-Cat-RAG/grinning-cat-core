"""Token-budget splitting of documents, run in the ``embedding`` phase.

The tokenizer depends on the active embedder, so the split lives in the embedding phase (not in parsing): a change of
embedder must be able to re-split the stored chunks. The feature is opt-in through ``CAT_INGESTION_SPLIT_OVERSIZED``;
when it is disabled the documents pass through untouched (upstream parity).
"""
from typing import Callable, List

from langchain_core.documents import Document

from cat.env import get_env_int

DEFAULT_SPLIT_MAX_TOKENS = 512


def default_tokenizer(text: str) -> int:
    """Conservative character-based estimate used when the embedder exposes no tokenizer."""
    return max(1, len(text) // 3)


def _split_text(text: str, max_tokens: int, tokenizer: Callable[[str], int]) -> List[tuple[int, str]]:
    """Split ``text`` into ``(start_offset, piece)`` pairs, each within ``max_tokens`` and never dropping text."""
    pieces: List[tuple[int, str]] = []
    start = 0
    total = len(text)
    while start < total:
        if tokenizer(text[start:]) <= max_tokens:
            pieces.append((start, text[start:]))
            break

        # Largest prefix (in characters) that stays within budget. Binary search assumes a monotonic tokenizer, which
        # holds for the character-based estimator and for real tokenizers; the +1 lower bound guarantees progress.
        low, high = start + 1, total
        best = start + 1
        while low <= high:
            middle = (low + high) // 2
            if tokenizer(text[start:middle]) <= max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1

        pieces.append((start, text[start:best]))
        start = best

    if not pieces:
        pieces.append((0, ""))
    return pieces


def split_oversized(
    documents: List[Document],
    max_tokens: int,
    tokenizer: Callable[[str], int] | None = None,
) -> List[Document]:
    """Split every document exceeding ``max_tokens`` into contiguous, ordered sub-documents.

    Pure, deterministic and idempotent: the returned documents are new ``Document`` instances with copied metadata and
    ``chunk_index`` renumbered from 0 across the whole list; ``start_index`` is preserved and advanced by the character
    offset of each piece inside its source document. No text is dropped. ``max_tokens <= 0`` disables the split.
    """
    if max_tokens is None or max_tokens <= 0:
        return list(documents)

    count = tokenizer or default_tokenizer
    result: List[Document] = []

    for document in documents:
        text = document.page_content
        if count(text) <= max_tokens:
            result.append(Document(page_content=text, metadata=dict(document.metadata)))
            continue

        base_start = document.metadata.get("start_index")
        if not isinstance(base_start, int) or base_start < 0:
            base_start = 0

        for offset, piece in _split_text(text, max_tokens, count):
            metadata = dict(document.metadata)
            metadata["start_index"] = base_start + offset
            result.append(Document(page_content=piece, metadata=metadata))

    for index, document in enumerate(result):
        document.metadata["chunk_index"] = index

    return result


def embedder_token_budget() -> int:
    """The token budget for the embedding split, configured by ``CAT_INGESTION_SPLIT_MAX_TOKENS``.

    ``CAT_INGESTION_SPLIT_MAX_TOKENS`` when set and positive, otherwise :data:`DEFAULT_SPLIT_MAX_TOKENS`. The
    tokenizer itself is the active embedder's ``_estimate_tokens``, passed by the caller.
    """
    configured = get_env_int("CAT_INGESTION_SPLIT_MAX_TOKENS")
    if configured is None or configured <= 0:
        return DEFAULT_SPLIT_MAX_TOKENS
    return configured
