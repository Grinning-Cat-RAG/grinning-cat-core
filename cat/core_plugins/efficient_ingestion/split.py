"""Token-budget chunk splitter (owned by the efficient_ingestion plugin).

Finalize a document list so no chunk exceeds the active embedder's
``max_input_tokens`` ceiling. This logic historically lived in the core
``RabbitHole._split_oversized`` (a MyCAT-only addition, absent from upstream);
it was moved here so the core returns to upstream parity and the plugin owns
the sizing behaviour.

Pure functions: ``split_oversized`` is a fold over the document list (reads
in, returns a brand-new list, never mutates while scanning). Oversized chunks
are split into budget-compliant sub-chunks that replace them at their own
index, preserving relative order; each sub-chunk becomes its own stored point
with the original metadata carried forward.

The core invokes this through the ``finalize_oversized_chunks`` no-op hook
(in ``base_plugin``) and, on the ``store_documents`` path, through the existing
``before_rabbithole_stores_documents`` hook. Importing this module has zero
side effects.
"""

from collections.abc import Callable
from typing import List

from langchain_core.documents import Document

from cat.log import log


def doc_tokens(doc: Document, embedder) -> int:
    """Conservative token count for a document chunk.

    Prefers the active embedder's own ``_estimate_tokens`` when available so
    the count matches the model that will embed the chunk; otherwise falls
    back to a ~3 chars/token heuristic (never an undercount).
    """
    if embedder is not None and hasattr(embedder, "_estimate_tokens"):
        return embedder._estimate_tokens(doc.page_content)
    return max(1, len(doc.page_content) // 3)


def split_to_budget(doc: Document, embedder) -> List[Document]:
    """Split one oversized document into budget-compliant sub-chunks.

    Word-based, linear split around the token budget, carrying the original
    metadata (source/payload) forward to every sub-chunk so nothing stored
    in the vector DB is lost.

    Sub-chunks are sized by MEASURING candidates with the embedder's own
    ``_estimate_tokens`` (real tokenizer when available); a coarse
    estimate-then-refine pass keeps it linear even for very long documents.
    """
    max_tokens = getattr(embedder, "max_input_tokens", None)
    if max_tokens is None or max_tokens <= 0:
        return [doc]

    words = doc.page_content.split()
    if not words:
        return [doc]

    def _doc_tokens(text: str) -> int:
        if embedder is not None and hasattr(embedder, "_estimate_tokens"):
            return max(1, embedder._estimate_tokens(text))
        return max(1, len(text) // 3)

    # avg tokens per word for THIS document (sampled on 200 words), used to
    # guess chunk boundaries in the coarse pass
    sample = words[:200]
    sample_tokens = max(1, _doc_tokens(" ".join(sample)))
    approx_tpw = sample_tokens / max(1, len(sample))
    per_chunk_words = max(1, int(max_tokens / approx_tpw))

    sub_docs: List[Document] = []
    start = 0
    n = len(words)
    while start < n:
        part = words[start:start + per_chunk_words]
        if _doc_tokens(" ".join(part)) <= max_tokens:
            take = len(part)
        else:
            # refine: binary-search the largest prefix within budget
            lo, hi = 0, len(part)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _doc_tokens(" ".join(part[:mid])) <= max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            take = max(1, lo)  # pathological single word: still emit it
        sub_docs.append(Document(
            page_content=" ".join(words[start:start + take]),
            metadata=doc.metadata,
        ))
        start += take
    return sub_docs


def split_oversized(
    docs: List[Document],
    embedder,
    log_fn: Callable[[str], None] | None = None,
) -> List[Document]:
    """Finalize a document list so no chunk exceeds the embedder's input limit.

    This is a *pure fold* over ``docs``: it reads the input list and returns a
    brand-new list, never mutating ``docs`` while scanning it. Any oversized
    chunk is split into budget-compliant sub-chunks that replace it at its own
    index, so the relative order of all chunks is preserved and each sub-chunk
    becomes its own stored point (with its own payload) in the vector database.

    Args:
        docs: The chunked documents produced by the configured chunker.
        embedder: The active embedder, whose ``max_input_tokens`` ceiling is
            enforced. ``None`` or a missing limit disables the split.
        log_fn: Optional callable for the oversize debug message (injected so
            callers can route it to their logger); defaults to ``log.debug``.

    Returns:
        A new list of documents where every chunk is at or under the limit.
    """
    max_tokens = getattr(embedder, "max_input_tokens", None)
    if max_tokens is None or max_tokens <= 0:
        return docs
    if log_fn is None:
        log_fn = log.debug

    sized: List[Document] = []
    for doc in docs:
        token_count = doc_tokens(doc, embedder)
        if token_count <= max_tokens:
            sized.append(doc)
            continue
        sub_chunks = split_to_budget(doc, embedder)
        source = doc.metadata.get("source") if isinstance(doc.metadata, dict) else None
        log_fn(
            f"OVERSIZED_SPLIT src={source} estimated_tokens={token_count} "
            f"max_input_tokens={max_tokens} split_into={len(sub_chunks)}"
        )
        sized.extend(sub_chunks)
    return sized