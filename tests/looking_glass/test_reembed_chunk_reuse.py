"""Tests for the chunk-reuse re-embed path in ``efficient_ingestion.reembed.reembed_sources``.

The embedder-change re-embed (``embed_all_in_cheshire_cats``) used to re-parse every
source from disk/URL via ``rabbit_hole.ingest_file``. This suite pins the new
chunk-reuse flow: existing stored chunks are reused and only the vectors are
recomputed, through the ``BaseVectorDatabaseHandler`` interface only (no Qdrant
classes). A fake handler implementing the interface stands in for the vector DB.
"""
import hashlib
import os

from langchain_core.documents import Document

from cat.core_plugins.efficient_ingestion.reembed import reembed_sources
from cat.db import crud
from cat.looking_glass.models import StoredSourceWithMetadata
from cat.rabbit_hole import RabbitHole
from cat.services.memory.models import Record, VectorMemoryType
from tests.utils import agent_id


class FakeEmbedder:
    """Minimal embedder exposing the attributes the reuse path reads."""

    name = "FakeEmbedder"
    size = 4
    max_input_tokens = 1000

    def embed_documents(self, texts):
        return [[0.1] * self.size for _ in texts]


class FailingEmbedder(FakeEmbedder):
    """Embedder that raises when asked to embed a specific text."""

    def __init__(self, boom_text):
        self.boom_text = boom_text

    def embed_documents(self, texts):
        for t in texts:
            if self.boom_text in t:
                raise RuntimeError(f"cannot embed {self.boom_text!r}")
        return super().embed_documents(texts)


class FakeMultimodalEmbedder(FakeEmbedder):
    """Multimodal embedder exposing ``embed_images`` and recording its calls."""

    name = "FakeMultimodalEmbedder"

    def __init__(self):
        self.embed_documents_calls = []
        self.embed_images_calls = []

    def embed_documents(self, texts):
        self.embed_documents_calls.append(list(texts))
        return [[0.1] * self.size for _ in texts]

    def embed_images(self, images):
        self.embed_images_calls.append(list(images))
        return [[0.5] * self.size for _ in images]


class RecordingTextEmbedder(FakeEmbedder):
    """Plain text-only embedder (no image API) that records its ``embed_documents``
    inputs, so a test can assert image payloads never reach the text embedder."""

    name = "RecordingTextEmbedder"

    def __init__(self):
        self.embed_documents_calls = []

    def embed_documents(self, texts):
        self.embed_documents_calls.append(list(texts))
        return [[0.1] * self.size for _ in texts]


class FakeFileManager:
    """Stub file manager resolving ``read_file`` from a {root_dir: {name: bytes}} map."""

    def __init__(self, files):
        self.files = files  # {root_dir: {name: bytes | None}}

    def read_file(self, remote_filename, remote_root_dir=None):
        return self.files.get(remote_root_dir, {}).get(remote_filename)


class FakeVectorHandler:
    """In-memory ``BaseVectorDatabaseHandler`` interface implementation.

    Only the interface methods used by the chunk-reuse path are implemented.
    """

    def __init__(self, points=None):
        self.points = list(points or [])
        self.added = []
        self.deleted = []

    async def get_all_tenant_points(
        self, collection_name, limit=None, offset=None, metadata=None, with_vectors=True
    ):
        return list(self.points), None

    async def add_points_to_tenant(self, collection_name, points):
        # persist the write like a real vector DB, so scroll/similarity_recall
        # below observe the same post-delete state the reuse path produced
        self.added.append((collection_name, points))
        self.points.extend(points)

    async def delete_tenant_points(self, collection_name, metadata=None):
        self.deleted.append((collection_name, metadata))
        if metadata and "source" in metadata:
            source = metadata["source"]
            self.points = [
                p
                for p in self.points
                if (p.payload or {}).get("metadata", {}).get("source") != source
            ]

    def scroll(self, collection_name=None):
        """Model a full Qdrant scroll listing: ALL points, including vector-less ones."""
        return list(self.points)

    def similarity_recall(self):
        """Model an ANN ``query_points`` recall against the stored points.

        Real Qdrant stores a payload-only point (``vector={}``) but does NOT return
        it from a similarity search; a vector-less point participates in scroll but
        never in ANN recall. This helper mirrors that: points with an absent or
        empty vector are excluded.
        """
        return [
            p
            for p in self.points
            if p.vector is not None and p.vector != {}
        ]


def _record(source, page_content, when=123.0, file_hash="abc", chat_id=None):
    """Build a stored ``Record`` with the payload shape a normal ingest produces."""
    metadata = {"source": source, "when": when, "hash": file_hash}
    if chat_id:
        metadata["chat_id"] = chat_id
    return Record(
        id=hashlib.sha256(f"{source}:{page_content}".encode()).hexdigest(),
        payload={
            "id": "some-id",
            "page_content": page_content,
            "metadata": metadata,
            "tenant_id": agent_id,
        },
        vector=[0.1, 0.2, 0.3, 0.4],
    )


def _image_record(source, image_file, when=123.0, file_hash="abc", chat_id=None):
    """Build a stored image ``Record`` matching the payload ``store_documents``
    writes for a multimodal image point."""
    metadata = {
        "source": source,
        "when": when,
        "hash": file_hash,
        "image": True,
        "image_file": image_file,
    }
    if chat_id:
        metadata["chat_id"] = chat_id
    return Record(
        id=hashlib.sha256(f"{source}:{image_file}".encode()).hexdigest(),
        payload={
            "id": "some-img-id",
            "page_content": f"[Image] {source}",
            "metadata": metadata,
            "tenant_id": agent_id,
        },
        vector=[0.1, 0.2, 0.3, 0.4],
    )


def _source(name, chat_id=None):
    metadata = {"source": name}
    if chat_id:
        metadata["chat_id"] = chat_id
    return StoredSourceWithMetadata(name=name, path=name, content=None, metadata=metadata)


def _status_key(source, scope="agent"):
    digest = hashlib.sha256(source.encode()).hexdigest()
    return f"agents:{agent_id}:ingestion:{scope}:{digest}"


async def _install_embedder(cheshire_cat, monkeypatch, handler, embedder):
    """Wire the fake handler + embedder onto the cat and return the ingest spy."""
    monkeypatch.setattr(cheshire_cat, "vector_memory_handler", handler)

    async def fake_embedder():
        return embedder

    monkeypatch.setattr(cheshire_cat, "embedder", fake_embedder)

    ingest_calls = []

    async def spy_ingest_file(cat, file, filename, metadata=None, store_file=False, content_type=None):
        ingest_calls.append(filename)

    monkeypatch.setattr(cheshire_cat.rabbit_hole, "ingest_file", spy_ingest_file)
    return ingest_calls


async def test_reuse_does_not_call_ingest_file(cheshire_cat, monkeypatch):
    """(a) With points present, ingest_file is NOT called; chunks are re-embedded
    and re-stored with the same payload shape."""
    fake = FakeVectorHandler(points=[
        _record("test.txt", "chunk one"),
        _record("test.txt", "chunk two"),
    ])
    ingest_calls = await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("test.txt")],
    )

    # no re-parse from disk/URL
    assert ingest_calls == []

    # exactly one add_points_to_tenant call with the two re-embedded chunks
    assert len(fake.added) == 1
    collection, points = fake.added[0]
    assert collection == str(VectorMemoryType.DECLARATIVE)
    assert len(points) == 2

    # payload shape matches a normal ingest: page_content + metadata{source,when,hash}
    for point in points:
        assert point.payload["page_content"] in ("chunk one", "chunk two")
        meta = point.payload["metadata"]
        assert meta["source"] == "test.txt"
        assert "when" in meta
        assert "hash" in meta
        # vectors were recomputed with the new embedder's dimension
        assert len(point.vector) == FakeEmbedder.size

    # the source's old points are deleted (metadata-filtered) before the
    # re-embedded points are added, so they replace rather than duplicate
    assert fake.deleted == [(str(VectorMemoryType.DECLARATIVE), {"source": "test.txt"})]

    # the delete happens BEFORE the add, so the re-embedded points replace the old ones
    assert fake.deleted and fake.added


async def test_reuse_invokes_split_oversized(cheshire_cat, monkeypatch):
    """(b) The token-budget splitter (now in the plugin) is invoked when the new
    embedder limit < chunk size."""
    import cat.core_plugins.efficient_ingestion.reembed as reembed_mod

    big_chunk = "x" * 5000
    fake = FakeVectorHandler(points=[_record("big.txt", big_chunk)])
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    calls = []

    def recording_split(docs, embedder):
        calls.append((list(docs), embedder))
        return docs

    monkeypatch.setattr(reembed_mod, "split_oversized", recording_split)

    await reembed_sources(cheshire_cat,
        VectorMemoryType.DECLARATIVE, [_source("big.txt")],
    )

    assert len(calls) == 1
    docs, embedder = calls[0]
    assert len(docs) == 1
    assert docs[0].page_content == big_chunk
    assert embedder is not None


async def test_empty_lookup_falls_back_to_parsing(cheshire_cat, monkeypatch):
    """(c) Empty lookups (no reusable chunks) fall back to the parsing_chunking phase.

    The phase machine parses the file and stores the chunks with EMPTY vectors,
    then the embedding phase recomputes the vectors and stores them.
    """
    from io import BytesIO

    import cat.core_plugins.efficient_ingestion.reembed as reembed_mod

    fake = FakeVectorHandler(points=[])

    async def fake_parse_and_chunk(ccat, rabbit_hole, source, file_bytes, content_type, cat):
        return [Document(page_content="parsed chunk one"), Document(page_content="parsed chunk two")], []

    monkeypatch.setattr(reembed_mod, "_parse_and_chunk", fake_parse_and_chunk)
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    src = _source("test.txt")
    src.content = BytesIO(b"file content")
    await reembed_sources(cheshire_cat, VectorMemoryType.DECLARATIVE, [src])

    # not chunk-reuse: two phases ran -> two adds (empty-vector store + embedding store)
    assert len(fake.added) == 2
    # first add is the parsing store (empty vectors), then embedding replaces
    first_phase_points = fake.added[0][1]
    assert all(p.vector == {} for p in first_phase_points)
    # the final (embedding) store carries real vectors
    final_points = fake.added[-1][1]
    assert len(final_points) == 2
    assert all(len(p.vector) == FakeEmbedder.size for p in final_points)


async def test_status_processing_then_completed(cheshire_cat, monkeypatch):
    """(d) Per-source status is written processing -> completed."""
    fake = FakeVectorHandler(points=[_record("test.txt", "chunk one")])
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    writes = []

    async def recording_store(key, value, path="$", nx=False, xx=False, expire=None):
        writes.append((key, value))
        return value

    monkeypatch.setattr(crud, "store", recording_store)

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("test.txt")],
    )

    key = _status_key("test.txt")
    statuses = [v["status"] for k, v in writes if k == key]
    assert statuses == ["processing", "completed"]

    # the completed payload carries the full schema
    completed = [v for k, v in writes if k == key and v["status"] == "completed"][0]
    assert completed["source"] == "test.txt"
    assert completed["scope"] == "agent"
    assert completed["chat_id"] is None
    assert completed["type"] == "file"
    assert completed["error"] is None
    assert "created_at" in completed
    assert "updated_at" in completed


async def test_one_source_failing_marks_error_others_complete(cheshire_cat, monkeypatch):
    """(e) A failing source is marked error; other sources still complete."""
    fake = FakeVectorHandler(points=[
        _record("good.txt", "good chunk"),
        _record("boom.txt", "boom chunk"),
    ])
    await _install_embedder(cheshire_cat, monkeypatch, fake, FailingEmbedder("boom chunk"))

    writes = []

    async def fake_store(key, value, path=None, nx=False, xx=False, expire=None):
        writes.append((key, value))
        return value

    monkeypatch.setattr(crud, "store", fake_store)

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("good.txt"), _source("boom.txt")],
    )

    good_statuses = [v["status"] for k, v in writes if k == _status_key("good.txt")]
    boom_statuses = [v["status"] for k, v in writes if k == _status_key("boom.txt")]

    assert good_statuses == ["processing", "completed"]
    assert boom_statuses == ["processing", "error"]

    boom_error = [v for k, v in writes if k == _status_key("boom.txt") and v["status"] == "error"][0]
    assert "cannot embed" in boom_error["error"]
    assert boom_error["error_at"] is not None


async def test_multimodal_reembeds_image_points_via_embed_images(cheshire_cat, monkeypatch):
    """(f) With a multimodal embedder, image points are re-embedded through
    ``embed_images`` (recovering bytes from the file manager), keep their original
    ``[Image] {source}`` payload + ``image=True``/``image_file`` and carry a real
    vector; text chunks keep using the ``embed_documents`` path unchanged."""
    fake = FakeVectorHandler(points=[
        _record("doc.pdf", "chunk one"),
        _image_record("doc.pdf", "doc_img_0.png", chat_id="chat-1"),
    ])
    embedder = FakeMultimodalEmbedder()
    await _install_embedder(cheshire_cat, monkeypatch, fake, embedder)

    # the image file lives under the agent_key/chat_id layout used by save_file
    root_dir = os.path.join(agent_id, "chat-1")
    monkeypatch.setattr(
        cheshire_cat, "file_manager", FakeFileManager({root_dir: {"doc_img_0.png": b"image-bytes"}}),
    )

    # a real stray cat would exist for an episodic source; simulate it so the
    # chunk-reuse path is reached instead of the cleanup-continue branch
    async def fake_find_stray_cat(_chat_id):
        return object()

    monkeypatch.setattr(cheshire_cat, "_find_stray_cat", fake_find_stray_cat)

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("doc.pdf", chat_id="chat-1")],
    )

    # images recovered from the file manager and embedded in ONE embed_images call
    assert embedder.embed_images_calls == [[b"image-bytes"]]

    # the image payload never went through embed_documents
    for texts in embedder.embed_documents_calls:
        assert all("[Image]" not in t for t in texts)

    assert len(fake.added) == 1
    collection, points = fake.added[0]
    assert collection == str(VectorMemoryType.DECLARATIVE)
    assert len(points) == 2

    image_points = [p for p in points if (p.payload.get("metadata") or {}).get("image")]
    text_points = [p for p in points if not (p.payload.get("metadata") or {}).get("image")]

    # text chunk re-embedded exactly as before
    assert len(text_points) == 1
    assert text_points[0].payload["page_content"] == "chunk one"
    assert len(text_points[0].vector) == FakeEmbedder.size

    # image point keeps the original payload shape with a real (non-empty) vector
    assert len(image_points) == 1
    img = image_points[0]
    assert img.payload["page_content"] == "[Image] doc.pdf"
    meta = img.payload["metadata"]
    assert meta["image"] is True
    assert meta["image_file"] == "doc_img_0.png"
    assert meta["source"] == "doc.pdf"
    assert img.vector and len(img.vector) == FakeEmbedder.size


async def test_multimodal_missing_file_keeps_image_point_payload_only(cheshire_cat, monkeypatch):
    """(g) H2 fallback: when a multimodal embedder's ``read_file`` returns None, the
    image point is kept payload-only with ``vector={}``, ``embed_images`` is not
    called for it, and the source completes without error."""
    fake = FakeVectorHandler(points=[
        _image_record("doc.pdf", "doc_img_0.png"),
    ])
    embedder = FakeMultimodalEmbedder()
    await _install_embedder(cheshire_cat, monkeypatch, fake, embedder)

    root_dir = agent_id
    monkeypatch.setattr(
        cheshire_cat, "file_manager", FakeFileManager({root_dir: {"doc_img_0.png": None}}),
    )

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("doc.pdf")],
    )

    # no recoverable bytes -> embed_images is never called
    assert embedder.embed_images_calls == []

    # the image point is re-added payload-only (no real vector), source completes
    assert len(fake.added) == 1
    _, points = fake.added[0]
    assert len(points) == 1
    img = points[0]
    assert img.payload["page_content"] == "[Image] doc.pdf"
    assert (img.payload.get("metadata") or {}).get("image") is True
    assert img.vector == {}


async def test_nonmultimodal_keeps_image_points_payload_only(cheshire_cat, monkeypatch):
    """(h) With a plain text (non-multimodal) embedder, image points are re-added
    PAYLOAD-ONLY with ``vector={}``: the full original payload survives
    (``[Image] {source}`` page_content, ``image=True``, ``image_file``, source,
    when, hash, chat_id). ``embed_documents`` never sees image content, the image
    file is not deleted, and text chunks still embed normally."""
    fake = FakeVectorHandler(points=[
        _record("doc.pdf", "chunk one"),
        _image_record("doc.pdf", "doc_img_0.png", chat_id="chat-1"),
    ])
    embedder = RecordingTextEmbedder()
    await _install_embedder(cheshire_cat, monkeypatch, fake, embedder)

    # the image file lives under the agent_key/chat_id layout used by save_file
    root_dir = os.path.join(agent_id, "chat-1")
    file_manager = FakeFileManager({root_dir: {"doc_img_0.png": b"image-bytes"}})
    monkeypatch.setattr(cheshire_cat, "file_manager", file_manager)

    async def fake_find_stray_cat(_chat_id):
        return object()

    monkeypatch.setattr(cheshire_cat, "_find_stray_cat", fake_find_stray_cat)

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("doc.pdf", chat_id="chat-1")],
    )

    # image payload never reached the text embedder
    for texts in embedder.embed_documents_calls:
        assert all("[Image]" not in t for t in texts)

    assert len(fake.added) == 1
    collection, points = fake.added[0]
    assert collection == str(VectorMemoryType.DECLARATIVE)
    assert len(points) == 2

    image_points = [p for p in points if (p.payload.get("metadata") or {}).get("image")]
    text_points = [p for p in points if not (p.payload.get("metadata") or {}).get("image")]

    # text chunk re-embedded exactly as before
    assert len(text_points) == 1
    assert text_points[0].payload["page_content"] == "chunk one"
    assert len(text_points[0].vector) == FakeEmbedder.size

    # image point preserved payload-only
    assert len(image_points) == 1
    img = image_points[0]
    assert img.payload["page_content"] == "[Image] doc.pdf"
    assert img.vector == {}
    meta = img.payload["metadata"]
    assert meta["image"] is True
    assert meta["image_file"] == "doc_img_0.png"
    assert meta["source"] == "doc.pdf"
    assert "when" in meta
    assert "hash" in meta
    assert meta["chat_id"] == "chat-1"

    # the image file is NOT deleted from the file manager (stays on disk)
    assert file_manager.files[root_dir]["doc_img_0.png"] == b"image-bytes"


async def test_nonmultimodal_vectorless_point_in_scroll_but_not_in_sim_recall(cheshire_cat, monkeypatch):
    """(i) Mirrors real Qdrant behavior: after the delete-then-add, the re-added
    vector-less image point (``vector={}``) IS present in a full scroll listing but
    is NOT returned by an ANN similarity recall (which skips points with no
    vector). Text points behave normally."""
    fake = FakeVectorHandler(points=[
        _record("doc.pdf", "chunk one"),
        _image_record("doc.pdf", "doc_img_0.png"),
    ])
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("doc.pdf")],
    )

    # full listing (scroll): BOTH the text chunk and the vector-less image survive
    scrolled = fake.scroll()
    assert len(scrolled) == 2
    image_scrolled = [
        p for p in scrolled if (p.payload.get("metadata") or {}).get("image")
    ]
    assert len(image_scrolled) == 1
    assert image_scrolled[0].payload["page_content"] == "[Image] doc.pdf"
    assert image_scrolled[0].vector == {}

    # ANN similarity recall drops the vector-less point, keeps the text point
    recalled = fake.similarity_recall()
    assert len(recalled) == 1
    assert not (recalled[0].payload.get("metadata") or {}).get("image")
    assert recalled[0].payload["page_content"] == "chunk one"


async def test_multimodal_whole_image_source_reembeds_source_bytes(cheshire_cat, monkeypatch):
    """(j) Whole-image source: a direct image upload stores ``image_file`` = the
    source itself (rabbit_hole.py:468), with no derived crop file. With a
    multimodal embedder, the re-embed recovers the whole image bytes via
    ``read_file(image_file=source, root_dir=agent_key)`` and re-embeds them like a
    normal image point: original ``[Image] {source}`` payload preserved + a real
    vector."""
    source = "photo.jpg"
    fake = FakeVectorHandler(points=[_image_record(source, source)])
    embedder = FakeMultimodalEmbedder()
    await _install_embedder(cheshire_cat, monkeypatch, fake, embedder)

    # whole-image bytes live directly under the agent_key root (image_file = source)
    monkeypatch.setattr(
        cheshire_cat,
        "file_manager",
        FakeFileManager({agent_id: {"photo.jpg": b"whole-image-bytes"}}),
    )

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source(source)],
    )

    # recovered via read_file(image_file=source, root_dir=agent_key) and embedded
    # in a single embed_images call
    assert embedder.embed_images_calls == [[b"whole-image-bytes"]]

    assert len(fake.added) == 1
    _, points = fake.added[0]
    assert len(points) == 1
    img = points[0]
    assert img.payload["page_content"] == "[Image] photo.jpg"
    meta = img.payload["metadata"]
    assert meta["image"] is True
    assert meta["image_file"] == "photo.jpg"
    assert meta["source"] == "photo.jpg"
    assert img.vector and len(img.vector) == FakeEmbedder.size


async def test_nonmultimodal_whole_image_source_kept_payload_only(cheshire_cat, monkeypatch):
    """(k) Whole-image source with a non-multimodal embedder: the image point is
    re-added PAYLOAD-ONLY with ``vector={}``. No bytes are read (``read_file`` is
    never called); the original payload (``[Image]`` page_content, ``image=True``,
    ``image_file == source``) survives."""
    source = "photo.jpg"
    fake = FakeVectorHandler(points=[_image_record(source, source)])
    embedder = RecordingTextEmbedder()
    await _install_embedder(cheshire_cat, monkeypatch, fake, embedder)

    file_manager = FakeFileManager({agent_id: {"photo.jpg": b"whole-image-bytes"}})
    # a read_file spy to prove the non-multimodal path never recovers bytes
    read_files = []

    class SpyingFileManager(FakeFileManager):
        def read_file(self, remote_filename, remote_root_dir=None):
            read_files.append((remote_filename, remote_root_dir))
            return super().read_file(remote_filename, remote_root_dir)

    monkeypatch.setattr(cheshire_cat, "file_manager", SpyingFileManager(file_manager.files))

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("photo.jpg")],
    )

    # non-multimodal image handling does not read the file at all
    assert read_files == []
    assert embedder.embed_documents_calls == []

    assert len(fake.added) == 1
    _, points = fake.added[0]
    assert len(points) == 1
    img = points[0]
    assert img.vector == {}
    assert img.payload["page_content"] == "[Image] photo.jpg"
    meta = img.payload["metadata"]
    assert meta["image"] is True
    assert meta["image_file"] == "photo.jpg"
    assert meta["source"] == "photo.jpg"


async def test_multimodal_reembeds_multiple_images_in_one_batch(cheshire_cat, monkeypatch):
    """(l) A source with several image points is re-embedded in ONE ``embed_images``
    batch (single call with all N bytes), not N separate calls — matching the
    ``store_documents`` batching at rabbit_hole.py:470-471."""
    fake = FakeVectorHandler(points=[
        _image_record("doc.pdf", "doc_img_0.png"),
        _image_record("doc.pdf", "doc_img_1.png"),
        _image_record("doc.pdf", "doc_img_2.png"),
    ])
    embedder = FakeMultimodalEmbedder()
    await _install_embedder(cheshire_cat, monkeypatch, fake, embedder)

    root_dir = agent_id
    monkeypatch.setattr(
        cheshire_cat,
        "file_manager",
        FakeFileManager({root_dir: {
            "doc_img_0.png": b"bytes-0",
            "doc_img_1.png": b"bytes-1",
            "doc_img_2.png": b"bytes-2",
        }}),
    )

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("doc.pdf")],
    )

    # exactly ONE embed_images call with all three byte payloads, in seed order
    assert embedder.embed_images_calls == [[b"bytes-0", b"bytes-1", b"bytes-2"]]

    assert len(fake.added) == 1
    _, points = fake.added[0]
    assert len(points) == 3
    image_points = [p for p in points if (p.payload.get("metadata") or {}).get("image")]
    assert len(image_points) == 3
    files = sorted(p.payload["metadata"]["image_file"] for p in image_points)
    assert files == ["doc_img_0.png", "doc_img_1.png", "doc_img_2.png"]
    for p in image_points:
        assert p.vector and len(p.vector) == FakeEmbedder.size


class ControlledMultimodalEmbedder(FakeMultimodalEmbedder):
    """Multimodal embedder that can fail or drop vectors from ``embed_images``,
    to exercise the atomicity (compute-before-delete) and zip-truncation paths."""

    def __init__(self, fail=False, drop_n=0):
        super().__init__()
        self.fail = fail
        self.drop_n = drop_n

    def embed_images(self, images):
        self.embed_images_calls.append(list(images))
        if self.fail:
            raise RuntimeError("cannot embed images")
        vectors = [[0.5] * self.size for _ in images]
        if self.drop_n:
            vectors = vectors[: max(0, len(vectors) - self.drop_n)]
        return vectors


async def test_multimodal_mixed_recoverable_and_missing_images(cheshire_cat, monkeypatch):
    """(m) Partial recovery: a source with MIXED image points (one recoverable, one
    missing) in the multimodal case. The recoverable one gets a real vector via
    ``embed_images``; the missing one is kept payload-only with ``vector={}``; the
    source completes without error."""
    fake = FakeVectorHandler(points=[
        _image_record("doc.pdf", "present.png"),
        _image_record("doc.pdf", "gone.png"),
    ])
    embedder = FakeMultimodalEmbedder()
    await _install_embedder(cheshire_cat, monkeypatch, fake, embedder)

    root_dir = agent_id
    monkeypatch.setattr(
        cheshire_cat,
        "file_manager",
        FakeFileManager({root_dir: {"present.png": b"present-bytes", "gone.png": None}}),
    )

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("doc.pdf")],
    )

    # only the recoverable image is embedded, and in a single call
    assert embedder.embed_images_calls == [[b"present-bytes"]]

    assert len(fake.added) == 1
    _, points = fake.added[0]
    assert len(points) == 2
    by_file = {
        (p.payload.get("metadata") or {}).get("image_file"): p
        for p in points
    }
    # recoverable image got a real vector
    present = by_file["present.png"]
    assert present.payload["page_content"] == "[Image] doc.pdf"
    assert present.vector and len(present.vector) == FakeEmbedder.size
    # missing image kept payload-only
    gone = by_file["gone.png"]
    assert gone.vector == {}
    assert gone.payload["page_content"] == "[Image] doc.pdf"
    assert (gone.payload.get("metadata") or {}).get("image") is True


async def test_reuse_embed_documents_failure_preserves_old_points(cheshire_cat, monkeypatch):
    """(n) R5/M2 atomicity (F2 #2): if ``embed_documents`` raises, the source's OLD
    points must survive — ``delete_tenant_points`` is only reached after all
    vectors are computed (compute-before-delete), so an embed failure must leave
    the delete uncalled and the original points intact. Status marked error."""
    original = [
        _record("boom.txt", "chunk one"),
        _record("boom.txt", "chunk two"),
    ]
    fake = FakeVectorHandler(points=list(original))
    await _install_embedder(cheshire_cat, monkeypatch, fake, FailingEmbedder("chunk two"))

    writes = []

    async def fake_store(key, value, path=None, nx=False, xx=False, expire=None):
        writes.append((key, value))
        return value

    monkeypatch.setattr(crud, "store", fake_store)

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("boom.txt")],
    )

    # processing -> error, never completed
    statuses = [v["status"] for k, v in writes if k == _status_key("boom.txt")]
    assert statuses == ["processing", "error"]

    # compute-before-delete: embed blew up before any delete ran
    assert fake.deleted == []

    # old points still present in the handler, not wiped by a partial delete
    assert len(fake.scroll()) == len(original)
    assert {p.id for p in fake.scroll()} == {p.id for p in original}


async def test_multimodal_embed_images_failure_preserves_old_points(cheshire_cat, monkeypatch):
    """(o) R5/M2 atomicity, image path: if ``embed_images`` raises mid-batch, the
    source's OLD points survive — delete_tenant_points is only reached after ALL
    vectors are computed, so the raise must propagate to an error status with the
    delete never called."""
    original = [
        _image_record("fotos.txt", "foto_0.png"),
        _image_record("fotos.txt", "foto_1.png"),
    ]
    fake = FakeVectorHandler(points=list(original))
    embedder = ControlledMultimodalEmbedder(fail=True)
    await _install_embedder(cheshire_cat, monkeypatch, fake, embedder)

    root_dir = agent_id
    monkeypatch.setattr(
        cheshire_cat,
        "file_manager",
        FakeFileManager({root_dir: {"foto_0.png": b"b0", "foto_1.png": b"b1"}}),
    )

    writes = []

    async def fake_store(key, value, path=None, nx=False, xx=False, expire=None):
        writes.append((key, value))
        return value

    monkeypatch.setattr(crud, "store", fake_store)

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("fotos.txt")],
    )

    # embed_images was attempted with both recoverable bytes in one batch
    assert embedder.embed_images_calls == [[b"b0", b"b1"]]

    statuses = [v["status"] for k, v in writes if k == _status_key("fotos.txt")]
    assert statuses == ["processing", "error"]
    err = next(
        v for k, v in writes if k == _status_key("fotos.txt") and v["status"] == "error"
    )
    assert "cannot embed images" in err["error"]

    # delete never ran; the old image points survive untouched
    assert fake.deleted == []
    assert len(fake.scroll()) == len(original)
    assert {p.id for p in fake.scroll()} == {p.id for p in original}


async def test_multimodal_embed_images_truncation_is_detected(cheshire_cat, monkeypatch):
    """(p) R5 #4 (F2 #4): if ``embed_images`` returns FEWER vectors than the images
    it was given, the tail must NOT be silently dropped via ``zip`` truncation —
    that would mark the source completed while a point the source still needs is
    deleted and never re-added. The correct behavior is to raise (treated as a
    failure), preserving the old points through the compute-before-delete
    invariant."""
    original = [
        _image_record("galeria.txt", "img_0.png"),
        _image_record("galeria.txt", "img_1.png"),
    ]
    fake = FakeVectorHandler(points=list(original))
    # returns only ONE vector for the TWO recoverable images (zip truncation tripwire)
    embedder = ControlledMultimodalEmbedder(drop_n=1)
    await _install_embedder(cheshire_cat, monkeypatch, fake, embedder)

    root_dir = agent_id
    monkeypatch.setattr(
        cheshire_cat,
        "file_manager",
        FakeFileManager({root_dir: {"img_0.png": b"b0", "img_1.png": b"b1"}}),
    )

    writes = []

    async def fake_store(key, value, path=None, nx=False, xx=False, expire=None):
        writes.append((key, value))
        return value

    monkeypatch.setattr(crud, "store", fake_store)

    await reembed_sources(cheshire_cat, 
        VectorMemoryType.DECLARATIVE, [_source("galeria.txt")],
    )

    # the truncating embed_images got both recoverable images in one batch
    assert embedder.embed_images_calls == [[b"b0", b"b1"]]

    # a short vector list is a failure, not a silent drop: status error, no delete
    statuses = [v["status"] for k, v in writes if k == _status_key("galeria.txt")]
    assert statuses == ["processing", "error"]

    # compute-before-delete: the mismatch aborts before delete_tenant_points runs
    assert fake.deleted == []
    assert len(fake.scroll()) == len(original)
    assert {p.id for p in fake.scroll()} == {p.id for p in original}

# ---------- phase machine: embedder/chunker-aware decision ----------


async def test_completed_matching_embedder_skips_source(cheshire_cat, monkeypatch):
    """A completed row whose embedder AND chunker match the current ones is skipped
    (no re-embed, no ingest)."""
    from cat.core_plugins.ingestion_status.registry import IngestionStatus, set_status
    from cat.utils import get_nlp_object_name

    fake = FakeVectorHandler(points=[_record("test.txt", "chunk one")])
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())
    # current names recorded at phase start would be "FakeEmbedder"/<active chunker>
    embedder_name = "FakeEmbedder"
    chunker = getattr(cheshire_cat, "chunker", None)
    chunker_name = (
        str(chunker.name) if chunker is not None and getattr(chunker, "name", None) else
        get_nlp_object_name(chunker, "default_chunker")
    )
    await set_status(
        agent_id, "agent", "test.txt",
        type_="file", status=IngestionStatus.COMPLETED,
        embedder_name=embedder_name, chunker_name=chunker_name,
    )

    writes = []
    async def fake_store(key, value, path=None, nx=False, xx=False, expire=None):
        writes.append((key, value))
        return value
    monkeypatch.setattr(crud, "store", fake_store)

    await reembed_sources(cheshire_cat, VectorMemoryType.DECLARATIVE, [_source("test.txt")])

    # no add points (skipped) — but the claim may rewrite the row; assert no re-embed
    assert fake.added == []
    # the row is NOT re-written to processing (it stays completed)
    statuses = [v["status"] for k, v in writes if k == _status_key("test.txt")]
    assert statuses == []


async def test_completed_embedder_mismatch_reembeds(cheshire_cat, monkeypatch):
    """A completed row whose embedder differs from the current one is re-embedded
    (embedding phase, chunk-reuse)."""
    from cat.core_plugins.ingestion_status.registry import IngestionStatus, set_status
    from cat.utils import get_nlp_object_name

    fake = FakeVectorHandler(points=[_record("test.txt", "chunk one")])
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())
    chunker = getattr(cheshire_cat, "chunker", None)
    chunker_name = (
        str(chunker.name) if chunker is not None and getattr(chunker, "name", None) else
        get_nlp_object_name(chunker, "default_chunker")
    )
    # old embedder mismatch -> re-embed
    await set_status(
        agent_id, "agent", "test.txt",
        type_="file", status=IngestionStatus.COMPLETED,
        embedder_name="VeryOldEmbedder", chunker_name=chunker_name,
    )

    await reembed_sources(cheshire_cat, VectorMemoryType.DECLARATIVE, [_source("test.txt")])

    # chunk-reuse happened: exactly one add, ingest_file not called
    assert len(fake.added) == 1
    assert fake.deleted == [(str(VectorMemoryType.DECLARATIVE), {"source": "test.txt"})]


async def test_completed_chunker_mismatch_full_reingest(cheshire_cat, monkeypatch):
    """A completed row whose chunker differs from the current one is fully re-ingested
    (parsing_chunking phase), NOT chunk-reused."""
    from io import BytesIO

    import cat.core_plugins.efficient_ingestion.reembed as reembed_mod
    from cat.core_plugins.ingestion_status.registry import IngestionStatus, set_status

    fake = FakeVectorHandler(points=[_record("test.txt", "chunk one")])

    async def fake_parse_and_chunk(ccat, rabbit_hole, source, file_bytes, content_type, cat):
        return [Document(page_content="fresh parsed chunk")], []

    monkeypatch.setattr(reembed_mod, "_parse_and_chunk", fake_parse_and_chunk)
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())
    # chunker mismatch (points exist but chunker changed) -> full re-ingest
    await set_status(
        agent_id, "agent", "test.txt",
        type_="file", status=IngestionStatus.COMPLETED,
        embedder_name="FakeEmbedder", chunker_name="OldChunker",
    )

    src = _source("test.txt")
    src.content = BytesIO(b"file content")
    await reembed_sources(cheshire_cat, VectorMemoryType.DECLARATIVE, [src])

    # full re-ingest via the phase machine (parsing -> embedding), not chunk-reuse:
    # the old points were deleted and the source re-parsed from scratch
    assert fake.deleted  # the old points were cleared
    # final (embedding) store carries the fresh parse content and real vectors
    final_points = fake.added[-1][1]
    assert len(final_points) == 1
    assert final_points[0].payload["page_content"] == "fresh parsed chunk"
    assert len(final_points[0].vector) == FakeEmbedder.size


async def test_inflight_embedding_phase_resumes_chunk_reuse(cheshire_cat, monkeypatch):
    """[restart] A processing row recorded at ``embedding`` (crash/restart mid-pass)
    resumes with chunk-reuse, NOT a full re-ingest: the chunks are valid, only the
    vectors need recomputing."""
    from cat.core_plugins.ingestion_status.registry import (
        PHASE_EMBEDDING,
        IngestionStatus,
        set_status,
    )
    from cat.utils import get_nlp_object_name

    fake = FakeVectorHandler(points=[_record("test.txt", "chunk one")])
    ingest_calls = await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())
    chunker = getattr(cheshire_cat, "chunker", None)
    chunker_name = (
        str(chunker.name) if chunker is not None and getattr(chunker, "name", None) else
        get_nlp_object_name(chunker, "default_chunker")
    )
    # a previous pass crashed while embedding: processing + phase=embedding
    await set_status(
        agent_id, "agent", "test.txt",
        type_="file", status=IngestionStatus.PROCESSING,
        phase=PHASE_EMBEDDING, embedder_name="FakeEmbedder", chunker_name=chunker_name,
    )

    writes = []
    async def recording_store(key, value, path="$", nx=False, xx=False, expire=None):
        writes.append((key, value))
        return value
    monkeypatch.setattr(crud, "store", recording_store)

    await reembed_sources(cheshire_cat, VectorMemoryType.DECLARATIVE, [_source("test.txt")])

    # resumes from embedding: chunk-reuse (no re-parse), points re-stored
    assert ingest_calls == []
    assert len(fake.added) == 1
    assert fake.deleted == [(str(VectorMemoryType.DECLARATIVE), {"source": "test.txt"})]

    # the row transitions (claim/phase writes are processing) and ends completed
    key = _status_key("test.txt")
    statuses = [
        (v["status"].value if hasattr(v["status"], "value") else v["status"])
        for k, v in writes if k == key
    ]
    assert statuses[-1] == "completed"
    assert "processing" in statuses


async def test_inflight_parsing_phase_resumes_full_reingest(cheshire_cat, monkeypatch):
    """[restart] A processing row recorded at ``parsing_chunking`` resumes with a
    full re-ingest (the text was mid-parse/chunk when the pass died)."""
    from io import BytesIO

    import cat.core_plugins.efficient_ingestion.reembed as reembed_mod
    from cat.core_plugins.ingestion_status.registry import (
        PHASE_PARSING_CHUNKING,
        IngestionStatus,
        set_status,
    )
    from cat.utils import get_nlp_object_name

    fake = FakeVectorHandler(points=[_record("test.txt", "chunk one")])

    async def fake_parse_and_chunk(ccat, rabbit_hole, source, file_bytes, content_type, cat):
        return [Document(page_content="resumed parsed chunk")], []

    monkeypatch.setattr(reembed_mod, "_parse_and_chunk", fake_parse_and_chunk)
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())
    chunker = getattr(cheshire_cat, "chunker", None)
    chunker_name = (
        str(chunker.name) if chunker is not None and getattr(chunker, "name", None) else
        get_nlp_object_name(chunker, "default_chunker")
    )
    await set_status(
        agent_id, "agent", "test.txt",
        type_="file", status=IngestionStatus.PROCESSING,
        phase=PHASE_PARSING_CHUNKING, chunker_name=chunker_name,
    )

    src = _source("test.txt")
    src.content = BytesIO(b"file content")
    await reembed_sources(cheshire_cat, VectorMemoryType.DECLARATIVE, [src])

    # full re-ingest (parsing -> embedding) via the phase machine, not chunk-reuse
    final_points = fake.added[-1][1]
    assert len(final_points) == 1
    assert final_points[0].payload["page_content"] == "resumed parsed chunk"
    assert len(final_points[0].vector) == FakeEmbedder.size


async def test_inflight_embedding_phase_chunker_mismatch_full_reingest(cheshire_cat, monkeypatch):
    """[chunker change] A processing row recorded at ``embedding`` whose chunker
    differs from the current one must NOT chunk-reuse: the stored chunks were
    produced by the OLD chunker, so it falls back to a full re-ingest
    (parsing_chunking) even though the recorded phase is embedding."""
    from io import BytesIO

    import cat.core_plugins.efficient_ingestion.reembed as reembed_mod
    from cat.core_plugins.ingestion_status.registry import (
        PHASE_EMBEDDING,
        IngestionStatus,
        set_status,
    )

    fake = FakeVectorHandler(points=[_record("test.txt", "chunk one")])

    async def fake_parse_and_chunk(ccat, rabbit_hole, source, file_bytes, content_type, cat):
        return [Document(page_content="fresh after chunker change")], []

    monkeypatch.setattr(reembed_mod, "_parse_and_chunk", fake_parse_and_chunk)
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())
    # row was re-embedding (embedding phase) when the chunker changed
    await set_status(
        agent_id, "agent", "test.txt",
        type_="file", status=IngestionStatus.PROCESSING,
        phase=PHASE_EMBEDDING, embedder_name="FakeEmbedder", chunker_name="OldChunker",
    )

    src = _source("test.txt")
    src.content = BytesIO(b"file content")
    await reembed_sources(cheshire_cat, VectorMemoryType.DECLARATIVE, [src])

    # full re-ingest (parsing -> embedding): chunks invalidated by the chunker change
    final_points = fake.added[-1][1]
    assert len(final_points) == 1
    assert final_points[0].payload["page_content"] == "fresh after chunker change"
    assert len(final_points[0].vector) == FakeEmbedder.size


async def test_url_without_web_points_recovered_from_status(cheshire_cat, monkeypatch):
    """[restart] A URL in-flight at ``parsing_chunking`` whose points were deleted
    (crash right after the clean-sweep) is recovered from the STATUS doc, not from
    the chunk points: the machine re-downloads the URL and completes."""
    import time as _time

    import cat.core_plugins.efficient_ingestion.reembed as reembed_mod
    from cat.core_plugins.ingestion_status.registry import (
        PHASE_PARSING_CHUNKING,
        IngestionStatus,
        get_status,
    )

    url = "https://example.com/doc.pdf"
    # NO web points at all: the URL lives ONLY in the status doc (a mid-parse
    # crash after the clean-sweep deleted the chunks)
    fake = FakeVectorHandler(points=[])

    downloads = []

    async def fake_resolve_source_bytes(self, file, filename, content_type):
        downloads.append((file, filename))
        return None, b"%PDF-1.4 fake content", "application/pdf", False

    monkeypatch.setattr(RabbitHole, "_resolve_source_bytes", fake_resolve_source_bytes)

    async def fake_parse_and_chunk(ccat, rabbit_hole, source, file_bytes, content_type, cat):
        return [Document(page_content="url parsed chunk")], []

    monkeypatch.setattr(reembed_mod, "_parse_and_chunk", fake_parse_and_chunk)
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    # stale processing row at parsing_chunking (container was down)
    old = _time.time() - 1000
    await crud.store(_status_key(url), {
        "source": url, "scope": "agent", "chat_id": None, "type": "url",
        "status": IngestionStatus.PROCESSING.value, "phase": PHASE_PARSING_CHUNKING,
        "error": None, "error_at": None, "created_at": old, "updated_at": old,
    })

    source = StoredSourceWithMetadata(name=url, path=url, content=None, metadata={})
    await reembed_sources(cheshire_cat, VectorMemoryType.DECLARATIVE, [source])

    # the URL was re-downloaded (recovered from the status entry, NOT from chunks)
    assert downloads == [(url, url)]
    # and the source completed
    doc = await get_status(agent_id, "agent", url)
    assert doc is not None
    assert doc["status"] == "completed"


async def test_url_status_written_before_clean_sweep(cheshire_cat, monkeypatch):
    """[order] The status doc holding the URL is written (set_phase) BEFORE the
    clean-sweep deletes the chunk points, so the URL value is never lost to the
    artifact deletion — the recovery derives the URL from the status, not from
    the points."""
    import time as _time

    import cat.core_plugins.efficient_ingestion.reembed as reembed_mod
    from cat.core_plugins.ingestion_status.registry import (
        IngestionStatus,
    )

    url = "https://example.com/order.pdf"
    # the URL currently lives in the web points (completed row whose chunker
    # changed -> forced parsing_chunking)
    fake = FakeVectorHandler(points=[_record(url, "old chunk")])

    event_order = []

    # intercept the status writes (through the registry crud.store)
    real_store = crud.store

    async def recording_store(key, value, path="$", nx=False, xx=False, expire=None):
        if isinstance(key, str) and key.startswith(f"agents:{agent_id}:ingestion:"):
            event_order.append(("status", value.get("source")))
        return await real_store(key, value, path=path, nx=nx, xx=xx, expire=expire)

    monkeypatch.setattr(crud, "store", recording_store)

    # intercept the point deletions
    real_delete = fake.delete_tenant_points

    async def recording_delete(collection_name, metadata=None):
        if isinstance(metadata, dict) and metadata.get("source") == url:
            event_order.append(("delete_points", url))
        return await real_delete(collection_name, metadata)

    fake.delete_tenant_points = recording_delete

    downloads = []

    async def fake_resolve_source_bytes(self, file, filename, content_type):
        downloads.append((file, filename))
        return None, b"%PDF-1.4 order content", "application/pdf", False

    monkeypatch.setattr(RabbitHole, "_resolve_source_bytes", fake_resolve_source_bytes)

    async def fake_parse_and_chunk(ccat, rabbit_hole, source, file_bytes, content_type, cat):
        return [Document(page_content="order parsed chunk")], []

    monkeypatch.setattr(reembed_mod, "_parse_and_chunk", fake_parse_and_chunk)
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    # completed + chunker mismatch -> parsing_chunking
    old = _time.time() - 1000
    await crud.store(_status_key(url), {
        "source": url, "scope": "agent", "chat_id": None, "type": "url",
        "status": IngestionStatus.COMPLETED.value,
        "embedder_name": "FakeEmbedder", "chunker_name": "OldChunker",
        "error": None, "error_at": None, "created_at": old, "updated_at": old,
    })

    await reembed_sources(cheshire_cat, VectorMemoryType.DECLARATIVE, [_url_source(url)])

    # the status write (URL persisted to Redis) precedes the artifact deletion
    status_idx = [i for i, e in enumerate(event_order) if e[0] == "status"]
    delete_idx = [i for i, e in enumerate(event_order) if e[0] == "delete_points"]
    assert status_idx, "no status write happened"
    assert delete_idx, "no point delete happened"
    assert status_idx[0] < delete_idx[0], "status must be written BEFORE the clean-sweep"


def _url_source(url):
    return StoredSourceWithMetadata(name=url, path=url, content=None, metadata={})
