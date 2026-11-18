import asyncio
import base64
import types
import uuid
from io import BytesIO

from langchain_core.documents import Document

from cat.core_plugins.base_plugin.parsers import MimeTypeBasedParser
from cat.db.cruds import ingestions as crud_ingestions
from cat.rabbit_hole import RabbitHole
from cat.services.factory.embedder import MultimodalEmbeddings
from cat.services.memory.models import PointStruct, VectorMemoryType
from tests.utils import agent_id


class FakeMultimodalEmbedder(MultimodalEmbeddings):
    """Minimal multimodal embedder used to exercise the image-ingestion branch."""

    def embed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 4

    def embed_image(self, image):
        return [0.1] * 4

    def embed_images(self, images):
        return [[0.1] * 4 for _ in images]


def _image_payload(data: bytes, mime: str = "image/jpeg") -> dict:
    return {
        "image_base64": base64.b64encode(data).decode(),
        "image_bytes": data,
        "image_mime_type": mime,
    }


async def test_store_documents_delegates_image_points_to_plugin_hook(cheshire_cat, monkeypatch):
    """Image points are built by a plugin through ``rabbithole_stores_image_points``.

    The core only collects what the hook returns and appends it to the same
    collection: with no plugin installed the no-op returns ``[]`` (upstream
    parity, no image points).
    """
    stored: dict = {}
    hook_calls: list = []

    async def fake_add_points(collection_name, points):
        stored["collection"] = collection_name
        stored["points"] = points

    fake_image_point = PointStruct(
        id=uuid.uuid4().hex,
        payload={"page_content": "", "metadata": {"image": True, "image_file": "test_img_0.jpg"}},
        vector=[0.1] * 4,
    )

    original_execute_hook = cheshire_cat.plugin_manager.execute_hook

    async def fake_execute_hook(hook_name, *args, caller=None):
        if hook_name == "rabbithole_stores_image_points":
            hook_calls.append({"args": args, "caller": caller})
            return [fake_image_point]
        return await original_execute_hook(hook_name, *args, caller=caller)

    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="a text chunk", metadata={})]
    images = [_image_payload(b"\x89PNG\r\n\x1a\n")]
    source_bytes = b"the raw source"

    points = await rabbit_hole.store_documents(
        docs=docs, source="test.txt", file_hash="hash", metadata={"k": "v"}, images=images,
        source_bytes=source_bytes,
    )

    # the plugin got everything it needs to build the image points
    assert len(hook_calls) == 1
    seed, hook_images, hook_source, hook_source_bytes, hook_metadata, hook_hash, hook_chat_id = hook_calls[0]["args"]
    assert seed == []
    assert hook_images == images
    assert hook_source == "test.txt"
    assert hook_source_bytes == source_bytes
    assert hook_metadata == {"k": "v"}
    assert hook_hash == "hash"
    assert hook_chat_id is None
    assert hook_calls[0]["caller"] is cheshire_cat

    # one text point + the point returned by the plugin, in the same collection
    assert stored["collection"] == str(VectorMemoryType.DECLARATIVE)
    assert len(points) == 2
    assert fake_image_point in points
    assert stored["points"] == points


async def test_store_documents_image_hook_receives_chat_id_in_conversation(cheshire_cat, monkeypatch):
    """In a conversation (stray set) the plugin is handed the chat id, and the
    points land in the episodic collection."""
    stored: dict = {}
    hook_calls: list = []

    async def fake_add_points(collection_name, points):
        stored["collection"] = collection_name

    original_execute_hook = cheshire_cat.plugin_manager.execute_hook

    async def fake_execute_hook(hook_name, *args, caller=None):
        if hook_name == "rabbithole_stores_image_points":
            hook_calls.append(args)
            return []
        return await original_execute_hook(hook_name, *args, caller=caller)

    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = types.SimpleNamespace(id="chat_abc")

    await rabbit_hole.store_documents(
        docs=[Document(page_content="a text chunk")], source="test.txt", metadata={},
        images=[_image_payload(b"IMG1")],
    )

    assert len(hook_calls) == 1
    assert hook_calls[0][-1] == "chat_abc"
    assert stored["collection"] == str(VectorMemoryType.EPISODIC)


async def test_store_documents_image_hook_not_called_without_images(cheshire_cat, monkeypatch):
    """No images collected by the parsers: the image seam is never entered."""
    hook_calls: list = []

    async def fake_add_points(collection_name, points):
        pass

    original_execute_hook = cheshire_cat.plugin_manager.execute_hook

    async def fake_execute_hook(hook_name, *args, caller=None):
        if hook_name == "rabbithole_stores_image_points":
            hook_calls.append(args)
            return []
        return await original_execute_hook(hook_name, *args, caller=caller)

    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    points = await rabbit_hole.store_documents(docs=[Document(page_content="a text chunk")], source="test.txt")

    assert hook_calls == []
    assert len(points) == 1


async def test_store_documents_text_only_ignores_images(cheshire_cat, monkeypatch):
    """With no plugin answering the image seam, images are ignored and only text is
    stored (upstream parity: the no-op hook returns ``[]``)."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="a text chunk")]

    points = await rabbit_hole.store_documents(docs=docs, source="test.txt", metadata={}, images=[_image_payload(b"IMG1")])

    # only the text point is stored, the image is dropped
    assert len(points) == 1
    assert not points[0].payload["metadata"].get("image")


async def test_store_documents_embedding_runs_in_ingestion_executor(cheshire_cat, monkeypatch):
    """Text embedding in ``store_documents`` is routed to the ingestion lane.

    The embedding call must be dispatched through ``run_in_ingestion_executor``
    (the dedicated low-concurrency pool), not run inline on the event loop.
    """
    stored: dict = {}
    calls = {"deferred_callables": []}

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    async def fake_run_in_ingestion_executor(func, *args, **kwargs):
        # Record the deferred embedding callable and return a fixed vector list.
        calls["deferred_callables"].append(func)
        return [[0.1] * 4 for _ in range(2)]

    monkeypatch.setattr("cat.rabbit_hole.core.run_in_ingestion_executor", fake_run_in_ingestion_executor)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [
        Document(page_content="a text chunk"),
        Document(page_content="another text chunk"),
    ]

    points = await rabbit_hole.store_documents(docs=docs, source="test.txt", metadata={})

    # the embedding was routed to the dedicated lane, not run inline
    assert len(calls["deferred_callables"]) == 1
    # the returned points carry the vectors produced by the deferred embedding
    assert len(points) == 2
    assert all(p.vector == [0.1] * 4 for p in points)


async def test_text_points_do_not_carry_image_base64(cheshire_cat, monkeypatch):
    """The base64 image payload must not be stored in the TEXT points' metadata.

    Documents produced by a multimodal parser carry ``image_base64`` in their
    metadata; the default chunker clones that metadata onto every text chunk.
    The image content must be removed before the text points are stored, so it
    is neither persisted in the vector DB nor forwarded to the LLM on recall.
    """
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        return source

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    img_data = b"\x89PNG\r\n\x1a\n"
    docs = [
        Document(
            page_content="a text chunk that surrounds an extracted image",
            metadata={
                "image_base64": base64.b64encode(img_data).decode(),
                "image_mime_type": "image/jpeg",
                "other_metadata": "keep me",
            },
        )
    ]

    points = await rabbit_hole.store_documents(
        docs=docs, source="test.txt", metadata={}, images=[_image_payload(img_data)],
    )

    text_points = [p for p in points if not p.payload["metadata"].get("image")]
    assert len(text_points) == 1
    text_metadata = text_points[0].payload["metadata"]
    assert "image_base64" not in text_metadata
    # the rest of the metadata survives untouched
    assert text_metadata["other_metadata"] == "keep me"


async def test_parse_to_docs_strips_nested_image_payload_after_the_hook(cheshire_cat, monkeypatch):
    """The transient image payload must be stripped at every depth, after the collection hook.

    A multimodal parser can nest its per-image metadata under ``source_chunks``
    (and deeper): the recursive walker removes every ``image_base64`` key while
    leaving ``source_chunks`` and ``coordinates`` untouched, and the images the
    hook collected are still returned to the embedding phase.
    """
    docs = [
        Document(
            page_content="chunk",
            metadata={
                "image_base64": "root",
                "coordinates": {"page": 2},
                "source_chunks": [
                    {
                        "text": "a",
                        "metadata": {
                            "image_base64": "nested",
                            "coordinates": {"page": 1, "bbox": [1, 2, 3, 4]},
                            "source_chunks": [{"metadata": {"image_base64": "deep"}}],
                        },
                    },
                ],
            },
        )
    ]
    images = [{"image_base64": "AAA", "image_bytes": b"x", "image_mime_type": "image/png"}]

    async def fake_run_in_ingestion_executor(func, *args, **kwargs):
        return docs

    def fake_parse(self, blob):
        return docs

    async def fake_split(self, docs):
        return docs

    original_execute_hook = cheshire_cat.plugin_manager.execute_hook

    async def fake_execute_hook(hook_name, *args, caller=None):
        if hook_name == "rabbithole_collects_document_images":
            return list(images)
        return await original_execute_hook(hook_name, *args, caller=caller)

    monkeypatch.setattr("cat.rabbit_hole.core.run_in_ingestion_executor", fake_run_in_ingestion_executor)
    monkeypatch.setattr(MimeTypeBasedParser, "parse", fake_parse)
    monkeypatch.setattr(RabbitHole, "_split_text", fake_split)
    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    parsed, returned = await rabbit_hole._parse_to_docs(
        source="test.txt", file_bytes=b"hello world", content_type="text/plain"
    )

    assert returned == images
    metadata = parsed[0].metadata
    assert "image_base64" not in metadata
    assert metadata["coordinates"] == {"page": 2}
    chunk = metadata["source_chunks"][0]["metadata"]
    assert "image_base64" not in chunk
    assert chunk["coordinates"] == {"page": 1, "bbox": [1, 2, 3, 4]}
    assert "image_base64" not in chunk["source_chunks"][0]["metadata"]


def test_agent_id_is_test_agent(cheshire_cat):
    # guard that tests run against the expected agent key
    assert cheshire_cat.agent_key == agent_id


async def test_parse_to_docs_runs_off_event_loop(cheshire_cat, monkeypatch):
    """The synchronous parser must be deferred via ``run_in_ingestion_executor``.

    ``MimeTypeBasedParser.parse`` is CPU/IO-bound (e.g. PyMuPDF on a large PDF)
    and must not run inline on the asyncio event loop. This test records every
    ``run_in_ingestion_executor`` call made by ``_parse_to_docs`` and asserts the
    parse happens inside the deferred callable, not inline.
    """
    calls = {"deferred_callables": [], "parse_calls": 0}

    async def fake_run_in_ingestion_executor(func, *args, **kwargs):
        # Record the deferred callable but do NOT run it: the test asserts the
        # parse is deferred, then invokes the callable itself to prove it.
        calls["deferred_callables"].append(func)
        return []

    def fake_parse(self, blob):
        calls["parse_calls"] += 1
        return [Document(page_content="parsed content")]

    monkeypatch.setattr("cat.rabbit_hole.core.run_in_ingestion_executor", fake_run_in_ingestion_executor)
    monkeypatch.setattr(MimeTypeBasedParser, "parse", fake_parse)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    # keep the test focused on the parse step: skip chunking and multimodal detection
    async def fake_split(self, docs):
        return docs

    monkeypatch.setattr(RabbitHole, "_split_text", fake_split)

    await rabbit_hole._parse_to_docs(source="test.txt", file_bytes=b"hello world", content_type="text/plain")

    # the parse must NOT have run inline on the event loop
    assert calls["parse_calls"] == 0
    # it must have been deferred through run_in_ingestion_executor, exactly once
    assert len(calls["deferred_callables"]) == 1

    # invoking the deferred callable performs the actual parse
    parsed = calls["deferred_callables"][0]()
    assert calls["parse_calls"] == 1
    assert parsed[0].page_content == "parsed content"


def _mock_ingestion_pipeline(monkeypatch, cheshire_cat, state):
    """Make the parse of ``ingest_file`` slow and observable.

    ``_parse_to_docs`` is replaced by a slow coroutine that tracks a shared
    active counter, so concurrent calls can be observed overlapping. The file
    save, the hooks and the notifications are stubbed out (the hooks answer
    nothing: the sources end in error after the parse, which is not observed).
    """
    async def slow_parse_to_docs(self, source, file_bytes, content_type=None):
        state["active"] += 1
        state["entered"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.05)  # keep the body busy so overlap is observable
        state["active"] -= 1
        return [Document(page_content="hello")], []

    async def fake_send_notification(self, message):
        pass

    async def fake_save_file(self, file_bytes, content_type, source, chat_id=None):
        pass

    async def fake_execute_hook(self, *args, **kwargs):
        return None

    monkeypatch.setattr(RabbitHole, "_parse_to_docs", slow_parse_to_docs)
    monkeypatch.setattr(RabbitHole, "_send_notification_message", fake_send_notification)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)
    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)

    rabbit_hole_a = RabbitHole()
    rabbit_hole_a.cat = cheshire_cat
    rabbit_hole_a.stray = None
    rabbit_hole_b = RabbitHole()
    rabbit_hole_b.cat = cheshire_cat
    rabbit_hole_b.stray = None
    return rabbit_hole_a, rabbit_hole_b


async def test_ingest_file_concurrency_bounded_by_max_concurrency(cheshire_cat, monkeypatch):
    """Concurrent ``ingest_file`` calls are serialized to ``CAT_INGESTION_MAX_CONCURRENCY``.

    The semaphore is process-wide (module-level, shared across RabbitHole
    instances) and its size is read lazily at runtime. With the env var set to
    ``1``, two concurrent ``ingest_file`` calls must never run their heavy body
    at the same time: the second blocks on the semaphore until the first
    releases it.
    """
    state = {"active": 0, "max_active": 0, "entered": 0}
    rabbit_hole_a, rabbit_hole_b = _mock_ingestion_pipeline(monkeypatch, cheshire_cat, state)

    # force a fresh module-level semaphore so the env value below is read
    # (raising=False: on pre-change code the attribute does not exist yet and
    # the test must still run to prove the concurrency is unbounded)
    monkeypatch.setattr("cat.rabbit_hole.core._ingestion_semaphore", None, raising=False)
    monkeypatch.setenv("CAT_INGESTION_MAX_CONCURRENCY", "1")

    await asyncio.gather(
        rabbit_hole_a.ingest_file(cat=cheshire_cat, file=BytesIO(b"a"), metadata={}, filename="a.txt"),
        rabbit_hole_b.ingest_file(cat=cheshire_cat, file=BytesIO(b"b"), metadata={}, filename="b.txt"),
    )

    assert state["entered"] == 2
    # bounded: the two bodies never overlap, even though they were launched together
    assert state["max_active"] == 1


async def test_ingest_file_unlimited_when_max_concurrency_non_positive(cheshire_cat, monkeypatch):
    """``CAT_INGESTION_MAX_CONCURRENCY <= 0`` means no semaphore: calls overlap.

    The unlimited path must NOT acquire the semaphore: with the env var set to
    ``0``, two concurrent ``ingest_file`` calls run their heavy body at the same
    time (previous behavior is preserved).
    """
    state = {"active": 0, "max_active": 0, "entered": 0}
    rabbit_hole_a, rabbit_hole_b = _mock_ingestion_pipeline(monkeypatch, cheshire_cat, state)

    monkeypatch.setattr("cat.rabbit_hole.core._ingestion_semaphore", None, raising=False)
    monkeypatch.setenv("CAT_INGESTION_MAX_CONCURRENCY", "0")

    await asyncio.gather(
        rabbit_hole_a.ingest_file(cat=cheshire_cat, file=BytesIO(b"a"), metadata={}, filename="a.txt"),
        rabbit_hole_b.ingest_file(cat=cheshire_cat, file=BytesIO(b"b"), metadata={}, filename="b.txt"),
    )

    assert state["entered"] == 2
    # unlimited: both bodies ran concurrently (no semaphore acquired)
    assert state["max_active"] == 2


async def test_ingest_file_saves_file_before_parsing(cheshire_cat, monkeypatch):
    """The uploaded file bytes are persisted BEFORE the parser runs.

    ``ingest_file`` must call ``save_file`` before ``_parse_to_docs`` so a
    container restart during a long parse does not lose the uploaded file.
    """
    order: list = []
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        order.append("save_file")
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_parse_to_docs(self, source, file_bytes, content_type=None):
        order.append("parse")
        return [Document(page_content="x")], []

    async def fake_execute_hook(self, *args, **kwargs):
        return None

    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)
    monkeypatch.setattr(RabbitHole, "_parse_to_docs", fake_parse_to_docs)
    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    await rabbit_hole.ingest_file(
        cat=cheshire_cat, file=BytesIO(b"hello world"), metadata={}, filename="hello.txt", content_type="text/plain",
    )

    # the file is persisted before the parser runs
    assert order == ["save_file", "parse"]
    # agent-scope ingest: the file is saved with the resolved bytes, content type,
    # source and a None chat_id
    assert saved_files == [(b"hello world", "text/plain", "hello.txt", None)]


async def test_ingest_file_sets_processing_before_parse(cheshire_cat, monkeypatch):
    """The source is ``processing``, with its phase, while the parser runs: after the file is persisted."""
    order: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        order.append("save_file")

    async def fake_parse_to_docs(self, source, file_bytes, content_type=None):
        row = await crud_ingestions.get_status(cheshire_cat.agent_key, None, source)
        order.append(("parse", row["status"], row["phase"]))
        return [Document(page_content="x")], []

    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)
    monkeypatch.setattr(RabbitHole, "_parse_to_docs", fake_parse_to_docs)

    await cheshire_cat.rabbit_hole.ingest_file(
        cat=cheshire_cat, file=BytesIO(b"hello world"), metadata={}, filename="hello.txt", content_type="text/plain",
    )

    assert order == ["save_file", ("parse", "processing", "parsing_chunking")]
