"""Short-chunk merging (`RabbitHole._merge_short_chunks`).

Invariant: a merge never joins chunks with a different ``slide`` (presentations): every stored
chunk keeps one exact slide. Chunks without ``slide`` merge as before.
"""
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.documents import Document

from cat.rabbit_hole import RabbitHole

CHUNKER = SimpleNamespace(splitter=SimpleNamespace(chunk_size=1000, chunk_overlap=100))  # min chunk: 50 chars


def _merge(docs):
    return RabbitHole._merge_short_chunks(RabbitHole.__new__(RabbitHole), docs, CHUNKER)


def _doc(text, **metadata):
    return Document(page_content=text, metadata=metadata)


def test_short_chunk_of_another_slide_is_not_merged():
    docs = [_doc("titolo slide uno " * 5, slide=1), _doc("breve", slide=2)]
    assert [d.metadata["slide"] for d in _merge(docs)] == [1, 2]


def test_short_chunk_of_the_same_slide_is_merged():
    docs = [_doc("titolo slide uno " * 5, slide=1), _doc("breve", slide=1)]
    [merged] = _merge(docs)
    assert merged.metadata["slide"] == 1
    assert merged.page_content.endswith("breve")


def test_chunks_without_slide_merge_as_before():
    docs = [_doc("pagina " * 20, page_number=1), _doc("breve", page_number=2)]
    [merged] = _merge(docs)
    assert merged.metadata["page_number"] == 1


def test_slide_and_no_slide_are_not_merged():
    docs = [_doc("titolo slide uno " * 5, slide=1), _doc("breve")]
    assert len(_merge(docs)) == 2


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(st.one_of(st.none(), st.integers(1, 4)), st.integers(1, 120)), max_size=25))
def test_merging_never_crosses_slides_and_loses_nothing(spec):
    docs = [_doc(f"<{i}>" + "x" * size, **({} if slide is None else {"slide": slide})) for i, (slide, size) in enumerate(spec)]
    merged = _merge(docs)

    # nothing lost, nothing reordered
    seen = [part for d in merged for part in d.page_content.replace("\n\n", "").split("<")[1:]]
    assert [int(p.split(">")[0]) for p in seen] == list(range(len(docs)))
    # every merged chunk holds parts of one slide only, and says so
    for d in merged:
        ids = [int(p.split(">")[0]) for p in d.page_content.replace("\n\n", "").split("<")[1:]]
        slides = {docs[i].metadata.get("slide") for i in ids}
        assert len(slides) == 1
        assert d.metadata.get("slide") == slides.pop()
