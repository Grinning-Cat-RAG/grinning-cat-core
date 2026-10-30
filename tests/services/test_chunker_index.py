"""The chunker must stamp an ordering index on every produced chunk.

``chunk_index`` is the 0-based ordinal of a chunk within one split call (one
call == one logical source), and ``start_index`` is the monotonic character
offset produced by LangChain's splitter (it survives overlap).
"""
from langchain_core.documents import Document

from cat.services.factory.chunker import RecursiveTextChunker
from cat.services.memory.models import VectorMemoryType
from tests.utils import send_file


def _chunker(chunk_size: int = 10, chunk_overlap: int = 2) -> RecursiveTextChunker:
    return RecursiveTextChunker(encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


async def test_chunks_get_increasing_start_index_and_sequential_chunk_index():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    chunks = await _chunker().split_documents([Document(page_content=text)])

    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    start_indices = [c.metadata["start_index"] for c in chunks]
    assert all(later > earlier for earlier, later in zip(start_indices, start_indices[1:]))
    assert start_indices[0] == 0


async def test_a_pre_existing_chunk_index_is_preserved():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    chunks = await _chunker().split_documents([Document(page_content=text, metadata={"chunk_index": 42})])

    assert len(chunks) > 1
    assert all(c.metadata["chunk_index"] == 42 for c in chunks)


async def test_a_single_chunk_document_gets_index_zero():
    chunks = await _chunker(chunk_size=256).split_documents([Document(page_content="meow")])

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["start_index"] == 0


async def test_classic_ingestion_stores_both_fields_on_the_points(secure_client, secure_client_headers, cheshire_cat):
    # End-to-end through the classic ingestion path: the chunker-produced ordering
    # must survive all the way to the stored vector points.
    file_name = "sample.txt"
    response, _ = await send_file(file_name, "text/plain", secure_client, secure_client_headers)
    assert response.status_code == 200

    points, _ = await cheshire_cat.vector_memory_handler.get_all_tenant_points(
        str(VectorMemoryType.DECLARATIVE),
        metadata={"source": file_name},
        with_vectors=False,
    )
    assert points, "the classic ingestion produced no stored points"

    for point in points:
        metadata = point.payload["metadata"]
        assert isinstance(metadata["chunk_index"], int)
        assert isinstance(metadata["start_index"], int)
        assert metadata["start_index"] >= 0
