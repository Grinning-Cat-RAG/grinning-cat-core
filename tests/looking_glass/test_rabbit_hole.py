import asyncio
import base64
import types
from io import BytesIO

from langchain_core.documents import Document

from cat.core_plugins.base_plugin.parsers import MimeTypeBasedParser
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.rabbit_hole import RabbitHole
from cat.services.factory.embedder import MultimodalEmbeddings
from cat.services.memory.models import VectorMemoryType
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


class PartialFailMultimodalEmbedder(FakeMultimodalEmbedder):
    """Multimodal embedder that returns ``None`` for a failed/skipped image embed.

    Mirrors the MyPLUS vLLM embedder contract after the "skip image and continue"
    decision: a failed image yields a ``None`` placeholder in the aligned
    ``embed_images`` result list (and ``embed_image`` returns ``None``).
    """

    def embed_image(self, image):
        return None

    def embed_images(self, images):
        return [None if i == 0 else [0.1] * 4 for i in range(len(images))]



def _image_payload(data: bytes, mime: str = "image/jpeg") -> dict:
    return {
        "image_base64": base64.b64encode(data).decode(),
        "image_bytes": data,
        "image_mime_type": mime,
    }


async def test_store_documents_multimodal_embeds_and_stores_images(cheshire_cat, monkeypatch):
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["collection"] = collection_name
        stored["points"] = points

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    # Force multimodal detection and swap the embedder + the vector memory storage.
    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    # Record save_file calls instead of writing the image files to the real storage.
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="a text chunk", metadata={})]
    images = [_image_payload(b"\x89PNG\r\n\x1a\n")]

    points = await rabbit_hole.store_documents(
        docs=docs, source="test.txt", file_hash="hash", metadata={}, images=images,
    )

    assert stored["collection"] == str(VectorMemoryType.DECLARATIVE)
    # one text point + one image point
    assert len(points) == 2

    text_points = [p for p in points if not p.payload["metadata"].get("image")]
    image_points = [p for p in points if p.payload["metadata"].get("image")]

    assert len(text_points) == 1
    assert len(image_points) == 1

    image_metadata = image_points[0].payload["metadata"]
    assert image_metadata["image"] is True
    assert image_metadata["source"] == "test.txt"
    assert "image_base64" not in image_metadata
    assert "image_mime_type" not in image_metadata
    image_file = image_metadata["image_file"]
    assert image_file.startswith("test_img_0_")
    assert image_file.endswith(".jpg")

    # the image was saved as a file, not embedded in the point metadata
    assert saved_files == [(b"\x89PNG\r\n\x1a\n", "image/jpeg", image_file, None)]


async def test_store_documents_multimodal_uses_agent_embedder(cheshire_cat, monkeypatch):
    """The image points must come from the agent's own embedder (embed_images)."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    calls: dict = {"images": []}

    class RecordingEmbedder(FakeMultimodalEmbedder):
        def embed_images(self, images):
            calls["images"] = list(images)
            return super().embed_images(images)

    async def fake_embedder():
        return RecordingEmbedder()

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    # Record save_file calls instead of writing the image files to the real storage.
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="a text chunk")]

    await rabbit_hole.store_documents(docs=docs, source="test.txt", metadata={}, images=[_image_payload(b"IMG1")])

    # the raw image bytes are what gets embedded
    assert calls["images"] == [b"IMG1"]

    # the image was saved as a file in the agent storage
    assert len(saved_files) == 1
    assert saved_files[0][2].startswith("test_img_")


async def test_store_documents_multimodal_in_conversation_adds_chat_id(cheshire_cat, monkeypatch):
    """In a conversation (stray set), image points carry chat_id in their metadata
    and the image file is saved under the conversation id."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["collection"] = collection_name
        stored["points"] = points

    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = types.SimpleNamespace(id="chat_abc")

    docs = [Document(page_content="a text chunk")]

    await rabbit_hole.store_documents(docs=docs, source="test.txt", metadata={}, images=[_image_payload(b"IMG1")])

    # image points land in the episodic collection and carry chat_id
    assert stored["collection"] == str(VectorMemoryType.EPISODIC)
    image_points = [p for p in stored["points"] if p.payload["metadata"].get("image")]
    assert len(image_points) == 1
    assert image_points[0].payload["metadata"]["chat_id"] == "chat_abc"
    # the saved image file is scoped to the conversation
    assert len(saved_files) == 1
    assert saved_files[0][3] == "chat_abc"


async def test_store_documents_multimodal_image_source_does_not_duplicate(cheshire_cat, monkeypatch):
    """Uploading an image file embeds the whole file (no derived files/points).

    The hi_res parser can split an image into sub-crops: those must be ignored and
    the source file itself embedded as a single image point (image_file = source).
    """
    stored: dict = {}
    saved_files: list = []

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="")]
    source_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16  # fake jpeg payload
    # parser crops for the image source (must be ignored)
    crops = [_image_payload(b"crop1", mime="image/jpeg"), _image_payload(b"crop2", mime="image/jpeg")]

    points = await rabbit_hole.store_documents(
        docs=docs, source="photo.jpeg", metadata={}, images=crops, source_bytes=source_bytes,
    )

    image_points = [p for p in points if p.payload["metadata"].get("image")]
    # exactly ONE image point, no derived files
    assert len(image_points) == 1
    assert image_points[0].payload["metadata"]["image_file"] == "photo.jpeg"
    assert image_points[0].payload["metadata"]["source"] == "photo.jpeg"
    assert saved_files == []


async def test_store_documents_skips_none_image_vectors(cheshire_cat, monkeypatch):
    """A failed/skipped image embed (``None`` vector) must be dropped entirely:
    its file is NOT saved and no point is stored for it. ``add_points_to_tenant``
    must never receive a ``None``-vector point (Qdrant would reject it or store
    payload-only garbage)."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["collection"] = collection_name
        stored["points"] = points

    async def fake_embedder():
        return PartialFailMultimodalEmbedder()

    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="a text chunk", metadata={})]
    # first image fails to embed (None), second succeeds
    images = [_image_payload(b"IMG_FAIL"), _image_payload(b"IMG_OK")]

    points = await rabbit_hole.store_documents(
        docs=docs, source="test.txt", file_hash="hash", metadata={}, images=images,
    )

    # exactly ONE image point: the failed image is dropped
    image_points = [p for p in points if p.payload["metadata"].get("image")]
    assert len(image_points) == 1
    assert image_points[0].payload["metadata"]["image_file"].startswith("test_img_1_")

    # the failed image's file was NOT saved (only the successful one)
    assert len(saved_files) == 1
    assert saved_files[0][2].startswith("test_img_1_")

    # add_points_to_tenant received no None-vector point
    assert all(p.vector is not None for p in stored["points"])


async def test_store_documents_whole_image_none_embed_skipped(cheshire_cat, monkeypatch):
    """Whole-image source: if ``embed_images`` returns ``[None]`` (failed embed),
    no image point is stored and no file is saved."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_embedder():
        return PartialFailMultimodalEmbedder()

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="")]
    source_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16  # fake jpeg payload

    points = await rabbit_hole.store_documents(
        docs=docs, source="photo.jpeg", metadata={}, images=[_image_payload(b"crop")], source_bytes=source_bytes,
    )

    # the failed whole-image embed produces no image point and no saved file
    image_points = [p for p in points if p.payload["metadata"].get("image")]
    assert image_points == []
    assert saved_files == []
    assert all(p.vector is not None for p in stored["points"])


async def test_store_documents_text_only_ignores_images(cheshire_cat, monkeypatch):
    """When the embedder is not multimodal, images are ignored and only text is stored."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    # detection reports a text-only embedder
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

    async def fake_run_in_ingestion_executor(self, func, *args):
        # Record the deferred embedding callable and return a fixed vector list.
        calls["deferred_callables"].append(func)
        return [[0.1] * 4 for _ in range(2)]

    # detection reports a text-only embedder so only the text-embedding lane runs
    monkeypatch.setattr(RabbitHole, "_run_in_ingestion_executor", fake_run_in_ingestion_executor)
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


def test_agent_id_is_test_agent(cheshire_cat):
    # guard that tests run against the expected agent key
    assert cheshire_cat.agent_key == agent_id


async def test_is_multimodal_embedder_uses_lizard_context(cheshire_cat, monkeypatch):
    """The embedder factory must run in the lizard (system) plugin-manager context.

    The ``factory_allowed_embedders`` hooks are declared with a ``lizard`` parameter
    (core base_plugin and PLUS alike): ``MadHatter.context_execute_hook`` passes the
    caller under that keyword only when the executing manager belongs to
    BillTheLizard. Using an agent plugin manager would pass ``cat`` and make the
    hooks raise ``TypeError: unexpected keyword argument 'cat'``.

    Moved to the multimodal_ingestion plugin (``is_multimodal_embedder_active``).
    """
    from cat.core_plugins.multimodal_ingestion import ingestion as mmi
    captured = {}

    class FakeServiceFactory:
        def __init__(self, agent_key, hook_manager, **kwargs):
            captured["agent_key"] = agent_key
            captured["plugin_manager_agent_key"] = hook_manager.agent_key
            captured["kwargs"] = kwargs

        async def get_config_class_from_adapter(self, obj):
            return None

    monkeypatch.setattr("cat.services.service_factory.ServiceFactory", FakeServiceFactory)

    assert await mmi.is_multimodal_embedder_active(cheshire_cat) is False

    assert captured["agent_key"] == DEFAULT_SYSTEM_KEY
    assert captured["plugin_manager_agent_key"] == DEFAULT_SYSTEM_KEY


async def test_file_to_docs_parse_runs_off_event_loop(cheshire_cat, monkeypatch):
    """The synchronous parser must be deferred via ``run_in_ingestion_executor``.

    ``MimeTypeBasedParser.parse`` is CPU/IO-bound (e.g. PyMuPDF on a large PDF)
    and must not run inline on the asyncio event loop. This test records every
    ``run_in_ingestion_executor`` call made by ``_file_to_docs`` and asserts the
    parse happens inside the deferred callable, not inline.
    """
    calls = {"deferred_callables": [], "parse_calls": 0}

    async def fake_run_in_ingestion_executor(self, func, *args):
        # Record the deferred callable but do NOT run it: the test asserts the
        # parse is deferred, then invokes the callable itself to prove it.
        calls["deferred_callables"].append(func)
        return []

    def fake_parse(self, blob):
        calls["parse_calls"] += 1
        return [Document(page_content="parsed content")]

    monkeypatch.setattr(RabbitHole, "_run_in_ingestion_executor", fake_run_in_ingestion_executor)
    monkeypatch.setattr(MimeTypeBasedParser, "parse", fake_parse)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    # keep the test focused on the parse step: skip chunking and multimodal detection
    async def fake_split(self, docs):
        return docs

    monkeypatch.setattr(RabbitHole, "_split_text", fake_split)

    await rabbit_hole._file_to_docs(
        file=BytesIO(b"hello world"), filename="test.txt", content_type="text/plain",
    )

    # the parse must NOT have run inline on the event loop
    assert calls["parse_calls"] == 0
    # it must have been deferred through run_in_ingestion_executor (exactly once:
    # the BytesIO path skips the file-read to_thread at the top of _file_to_docs)
    assert len(calls["deferred_callables"]) == 1

    # invoking the deferred callable performs the actual parse
    parsed = calls["deferred_callables"][0]()
    assert calls["parse_calls"] == 1
    assert parsed[0].page_content == "parsed content"


def _mock_ingestion_pipeline(monkeypatch, cheshire_cat, state):
    """Monkeypatch the heavy ``ingest_file`` body so only ``_parse_to_docs`` runs.

    ``_parse_to_docs`` is replaced by a slow coroutine that tracks a shared
    active counter, so concurrent calls can be observed overlapping. Everything
    downstream (store, file save, hooks, notifications) is stubbed out.
    """
    async def slow_parse_to_docs(self, source, file_bytes, content_type=None):
        state["active"] += 1
        state["entered"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.05)  # keep the body busy so overlap is observable
        state["active"] -= 1
        return [Document(page_content="hello")], []

    async def fake_store_documents(self, docs, source, file_hash=None, metadata=None, images=None, source_bytes=None):
        return []

    async def fake_send_notification(self, message):
        pass

    async def fake_save_file(self, file_bytes, content_type, source, chat_id=None):
        pass

    async def fake_execute_hook(self, *args, **kwargs):
        return None

    monkeypatch.setattr(RabbitHole, "_parse_to_docs", slow_parse_to_docs)
    monkeypatch.setattr(RabbitHole, "store_documents", fake_store_documents)
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
    monkeypatch.setattr("cat.rabbit_hole._ingestion_semaphore", None, raising=False)
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

    monkeypatch.setattr("cat.rabbit_hole._ingestion_semaphore", None, raising=False)
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
    container restart during a long parse does not lose the uploaded file
    (the resume mechanism reads it back from disk).
    """
    order: list = []
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        order.append("save_file")
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_parse_to_docs(self, source, file_bytes, content_type=None):
        order.append("parse")
        return [Document(page_content="x")], []

    async def fake_store_documents(self, docs, source, file_hash=None, metadata=None, images=None, source_bytes=None):
        return []

    async def fake_execute_hook(self, *args, **kwargs):
        return None

    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)
    monkeypatch.setattr(RabbitHole, "_parse_to_docs", fake_parse_to_docs)
    monkeypatch.setattr(RabbitHole, "store_documents", fake_store_documents)
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
    """The ``processing`` status is recorded BEFORE the parser runs.

    ``ingest_file`` must fire ``rabbithole_ingestion_processing`` after the
    file is persisted (``save_file``) and before ``_parse_to_docs``, so a long
    parse is visible as ``processing`` (and the heartbeat keeps it fresh).
    """
    order: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        order.append("save_file")

    async def fake_parse_to_docs(self, source, file_bytes, content_type=None):
        order.append("parse")
        return [Document(page_content="x")], []

    async def fake_store_documents(self, docs, source, file_hash=None, metadata=None, images=None, source_bytes=None):
        return []

    async def fake_execute_hook(hook_name, *args, **kwargs):
        if hook_name == "rabbithole_ingestion_processing":
            order.append("processing")

    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)
    monkeypatch.setattr(RabbitHole, "_parse_to_docs", fake_parse_to_docs)
    monkeypatch.setattr(RabbitHole, "store_documents", fake_store_documents)
    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    await rabbit_hole.ingest_file(
        cat=cheshire_cat, file=BytesIO(b"hello world"), metadata={}, filename="hello.txt", content_type="text/plain",
    )

    # processing fires after the file is persisted but before the parser runs
    assert order == ["save_file", "processing", "parse"]


async def test_ingest_file_survives_restart_via_resume(cheshire_cat, monkeypatch):
    """F3: a container restart during a large-file parse does not lose the file.

    ``ingest_file`` persists the bytes to the file manager BEFORE the (slow)
    parse runs. A simulated restart + resume of the stale ``uploaded`` entry
    then re-reads the file from disk and completes it — never hitting the
    "file missing" path.
    """
    import time
    from unittest.mock import AsyncMock

    from cat.core_plugins.ingestion_status import plugin as ingestion_plugin
    from cat.core_plugins.efficient_ingestion import resume as ingestion_resume
    from cat.core_plugins.ingestion_status.registry import get_status, status_key
    from cat.db import crud

    agent_key = cheshire_cat.agent_key
    source = "hello.txt"
    file_bytes = b"hello world"

    # ---- phase 1: in-flight large-file ingestion ----------------------------
    # The fixture's file manager is a DummyFileManager (not writable), so the
    # durable-persist guarantee is proven by ORDER: save_file must run BEFORE
    # the (slow) parse. The resume phase below then proves the persisted bytes
    # are read back from disk and the ingestion completes.
    order: list = []
    saved: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        order.append("save_file")
        saved.append((file_bytes, content_type, source, chat_id))

    async def slow_parse_to_docs(self, source, file_bytes, content_type=None):
        order.append("parse")
        await asyncio.sleep(0.05)  # simulate a long parse
        return [Document(page_content="x")], []

    async def fake_store_documents(self, docs, source, file_hash=None, metadata=None, images=None, source_bytes=None):
        return []

    async def fake_execute_hook(self, *args, **kwargs):
        return None

    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)
    monkeypatch.setattr(RabbitHole, "_parse_to_docs", slow_parse_to_docs)
    monkeypatch.setattr(RabbitHole, "store_documents", fake_store_documents)
    monkeypatch.setattr(cheshire_cat.plugin_manager, "execute_hook", fake_execute_hook)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    await rabbit_hole.ingest_file(
        cat=cheshire_cat, file=BytesIO(file_bytes), metadata={}, filename=source, content_type="text/plain",
    )

    # the file bytes were persisted BEFORE the slow parse completed
    assert order == ["save_file", "parse"]
    assert saved == [(file_bytes, "text/plain", source, None)]

    # ---- phase 2: simulated restart + resume --------------------------------
    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", source), {
        "source": source,
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "uploaded",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(cheshire_cat.lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    # the file is present on disk (persisted in phase 1)
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda s, p: file_bytes)

    # phase 2 goes through the ONE phase machine: spy on it (it re-reads the
    # persisted file and completes the source)
    import cat.core_plugins.efficient_ingestion.resume as efficient_resume_mod
    from cat.core_plugins.ingestion_status.registry import set_status as _set_status_c

    machine_calls = []

    async def fake_machine(ccat, collection_name, stored_sources, stale_after=None):
        for s in stored_sources:
            machine_calls.append((str(collection_name), s.name))
            await _set_status_c(ccat.agent_key, "agent", s.name, type_="file", status="completed", chat_id=None)

    monkeypatch.setattr(efficient_resume_mod, "reembed_sources", fake_machine)

    await ingestion_resume._resume_agent(cheshire_cat.lizard, agent_key)

    # the resume handed the source to the phase machine exactly once (no "file
    # missing" path) and it completed
    assert machine_calls == [("declarative", source)]
    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "completed"
