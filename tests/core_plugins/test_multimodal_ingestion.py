"""Tests for the ``multimodal_ingestion`` core plugin.

Covers the image lifecycle now owned by the plugin:
- ingestion: ``collect_document_images`` / ``strip_image_payload`` /
  ``is_multimodal_embedder_active`` / ``build_image_points`` (the logic that
  used to live in ``RabbitHole``) plus the core wiring through
  ``store_documents`` -> ``rabbithole_stores_image_points`` hook.
- recall: ``build_recalled_images`` (the logic that used to live in
  ``StrayCat._attach_recalled_images``) plus the ``before_agentic_workflow``
  hook.

Uses ``tests/conftest.py`` fixtures: Redis db=1 (isolated).
"""

import base64

from langchain_core.documents import Document

from cat.core_plugins.multimodal_ingestion import ingestion, recall
from cat.core_plugins.multimodal_ingestion import plugin as multimodal_plugin
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.looking_glass.models import AgenticWorkflowTask
from cat.rabbit_hole import RabbitHole
from cat.services.factory.embedder import MultimodalEmbeddings
from cat.services.memory.models import VectorMemoryType


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
    ``embed_images`` result list.
    """

    def embed_image(self, image):
        return None

    def embed_images(self, images):
        return [None if i == 0 else [0.1] * 4 for i in range(len(images))]


class FakeTextEmbedder:
    """Plain text-only embedder: exposes ``embed_documents`` but no image API."""

    name = "FakeTextEmbedder"
    size = 4

    def embed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 4


def _image_payload(data: bytes, mime: str = "image/jpeg") -> dict:
    return {
        "image_base64": base64.b64encode(data).decode(),
        "image_bytes": data,
        "image_mime_type": mime,
    }


# ---------- collection (used to be RabbitHole._collect_document_images) ----------


def test_collect_document_images():
    docs = [
        Document(page_content="plain text, no images"),
        Document(page_content="image element", metadata={"image_base64": "aGVsbG8=", "image_mime_type": "image/png"}),
    ]

    images = ingestion.collect_document_images(docs)

    assert len(images) == 1
    assert images[0]["image_bytes"] == b"hello"
    assert images[0]["image_mime_type"] == "image/png"


def test_collect_document_images_defaults_mime():
    docs = [Document(page_content="image", metadata={"image_base64": "aGVsbG8="})]
    images = ingestion.collect_document_images(docs)

    assert images[0]["image_mime_type"] == "image/jpeg"


def test_strip_image_payload_removes_base64():
    docs = [
        Document(
            page_content="text",
            metadata={"image_base64": "aGVsbG8=", "image_mime_type": "image/png", "keep": "me"},
        )
    ]
    ingestion.strip_image_payload(docs)
    assert "image_base64" not in docs[0].metadata
    assert docs[0].metadata["keep"] == "me"


# ---------- multimodal detection (used to be RabbitHole._is_multimodal_embedder) ----------


async def test_is_multimodal_embedder_uses_lizard_context(cheshire_cat, monkeypatch):
    """The embedder factory must run in the lizard (system) plugin-manager context.

    The ``factory_allowed_embedders`` hooks are declared with a ``lizard``
    parameter; ``MadHatter.context_execute_hook`` passes the caller under that
    keyword only when the executing manager belongs to BillTheLizard.
    """
    captured = {}

    class FakeServiceFactory:
        def __init__(self, agent_key, hook_manager, **kwargs):
            captured["agent_key"] = agent_key
            captured["plugin_manager_agent_key"] = hook_manager.agent_key
            captured["kwargs"] = kwargs

        async def get_config_class_from_adapter(self, obj):
            return None

    monkeypatch.setattr("cat.services.service_factory.ServiceFactory", FakeServiceFactory)

    assert await ingestion.is_multimodal_embedder_active(cheshire_cat) is False

    assert captured["agent_key"] == DEFAULT_SYSTEM_KEY
    assert captured["plugin_manager_agent_key"] == DEFAULT_SYSTEM_KEY


# ---------- image-point building (used to be the store_documents image block) ----------


async def test_build_image_points_embeds_and_stores_images(cheshire_cat, monkeypatch):
    async def fake_embedder():
        return FakeMultimodalEmbedder()

    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    image_points = await ingestion.build_image_points(
        cheshire_cat, [_image_payload(b"\x89PNG\r\n\x1a\n")],
        source="test.txt", source_bytes=None, metadata={}, file_hash="hash", chat_id=None,
    )

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


async def test_build_image_points_uses_agent_embedder(cheshire_cat, monkeypatch):
    """The image points must come from the agent's own embedder (embed_images)."""
    calls: dict = {"images": []}

    class RecordingEmbedder(FakeMultimodalEmbedder):
        def embed_images(self, images):
            calls["images"] = list(images)
            return super().embed_images(images)

    async def fake_embedder():
        return RecordingEmbedder()

    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    await ingestion.build_image_points(
        cheshire_cat, [_image_payload(b"IMG1")],
        source="test.txt", source_bytes=None, metadata={}, file_hash=None, chat_id=None,
    )

    # the raw image bytes are what gets embedded
    assert calls["images"] == [b"IMG1"]

    # the image was saved as a file in the agent storage
    assert len(saved_files) == 1
    assert saved_files[0][2].startswith("test_img_")


async def test_build_image_points_in_conversation_adds_chat_id(cheshire_cat, monkeypatch):
    """Image points built for a conversation carry chat_id in their metadata
    and the image file is saved under the conversation id."""
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    image_points = await ingestion.build_image_points(
        cheshire_cat, [_image_payload(b"IMG1")],
        source="test.txt", source_bytes=None, metadata={}, file_hash=None, chat_id="chat_abc",
    )

    assert len(image_points) == 1
    assert image_points[0].payload["metadata"]["chat_id"] == "chat_abc"
    # the saved image file is scoped to the conversation
    assert saved_files == [(b"IMG1", "image/jpeg", image_points[0].payload["metadata"]["image_file"], "chat_abc")]


async def test_build_image_points_image_source_does_not_duplicate(cheshire_cat, monkeypatch):
    """Uploading an image file embeds the whole file (no derived files/points).

    The hi_res parser can split an image into sub-crops: those must be ignored and
    the source file itself embedded as a single image point (image_file = source).
    """
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    source_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16  # fake jpeg payload
    # parser crops for the image source (must be ignored)
    crops = [_image_payload(b"crop1"), _image_payload(b"crop2")]

    image_points = await ingestion.build_image_points(
        cheshire_cat, crops,
        source="photo.jpeg", source_bytes=source_bytes, metadata={}, file_hash=None, chat_id=None,
    )

    # exactly ONE image point, no derived files
    assert len(image_points) == 1
    assert image_points[0].payload["metadata"]["image_file"] == "photo.jpeg"
    assert image_points[0].payload["metadata"]["source"] == "photo.jpeg"
    assert saved_files == []


async def test_build_image_points_skips_none_image_vectors(cheshire_cat, monkeypatch):
    """A failed/skipped image embed (``None`` vector) must be dropped entirely:
    its file is NOT saved and no point is returned for it."""
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_embedder():
        return PartialFailMultimodalEmbedder()

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    # first image fails to embed (None), second succeeds
    image_points = await ingestion.build_image_points(
        cheshire_cat, [_image_payload(b"IMG_FAIL"), _image_payload(b"IMG_OK")],
        source="test.txt", source_bytes=None, metadata={}, file_hash="hash", chat_id=None,
    )

    # exactly ONE image point: the failed image is dropped
    assert len(image_points) == 1
    assert image_points[0].payload["metadata"]["image_file"].startswith("test_img_1_")

    # the failed image's file was NOT saved (only the successful one)
    assert len(saved_files) == 1
    assert saved_files[0][2].startswith("test_img_1_")


async def test_build_image_points_whole_image_none_embed_skipped(cheshire_cat, monkeypatch):
    """Whole-image source: if ``embed_images`` returns ``[None]`` (failed embed),
    no image point is returned and no file is saved."""
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_embedder():
        return PartialFailMultimodalEmbedder()

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    source_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16  # fake jpeg payload
    image_points = await ingestion.build_image_points(
        cheshire_cat, [_image_payload(b"crop")],
        source="photo.jpeg", source_bytes=source_bytes, metadata={}, file_hash=None, chat_id=None,
    )

    assert image_points == []
    assert saved_files == []


# ---------- core wiring: store_documents -> rabbithole_stores_image_points ----------


async def test_store_documents_wiring_builds_image_points_via_plugin(cheshire_cat, monkeypatch):
    """``store_documents`` fires the ``rabbithole_stores_image_points`` hook; the
    plugin (auto-loaded in the test env) builds the image points."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["collection"] = collection_name
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

    docs = [Document(page_content="a text chunk")]

    points = await rabbit_hole.store_documents(
        docs=docs, source="test.txt", file_hash="hash", metadata={}, images=[_image_payload(b"IMG1")],
    )

    assert stored["collection"] == str(VectorMemoryType.DECLARATIVE)
    # one text point + one image point produced by the plugin hook
    assert len(points) == 2
    image_points = [p for p in points if p.payload["metadata"].get("image")]
    assert len(image_points) == 1
    assert image_points[0].payload["metadata"]["image_file"].startswith("test_img_0_")


async def test_stores_image_points_hook_returns_built_points(cheshire_cat, monkeypatch):
    """The plugin hook itself returns the points built by build_image_points."""
    async def fake_embedder():
        return FakeMultimodalEmbedder()

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        return source

    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    points = await multimodal_plugin.rabbithole_stores_image_points.function(
        [], [_image_payload(b"IMG1")], "test.txt", None, {}, "hash", None, cheshire_cat,
    )
    assert len(points) == 1
    assert points[0].payload["metadata"]["image"] is True


# ---------- deletion (used to be RabbitHole/_delete_source_image_files in the core) ----------


def _image_point(payload_metadata: dict):
    import types

    return types.SimpleNamespace(payload={"metadata": payload_metadata})


async def test_before_file_manager_file_delete_agent_scope(cheshire_cat, monkeypatch):
    """The plugin hook cascade-removes the image files of a deleted source.

    Agent scope: declarative collection, storage path = agent_key.
    """
    removed_files = []
    queries = []

    class FakeHandler:
        async def get_all_tenant_points(self, collection_name, limit, offset, metadata):
            queries.append((collection_name, limit, offset, metadata))
            points = [
                _image_point({"source": "test.txt", "image": True, "image_file": "test_img_0_abcd.jpg"}),
                _image_point({"source": "test.txt"}),  # no image_file -> no removal
            ]
            return points, None

    class FakeFileManager:
        def remove_file(self, file_path):
            removed_files.append(file_path)
            return True

    monkeypatch.setattr(cheshire_cat, "vector_memory_handler", FakeHandler())
    monkeypatch.setattr(cheshire_cat, "file_manager", FakeFileManager())

    await multimodal_plugin.before_file_manager_file_delete.function("test.txt", "agent", cheshire_cat)

    assert queries == [("declarative", 100, None, {"source": "test.txt", "image": True})]
    assert removed_files == ["agent_test/test_img_0_abcd.jpg"]


async def test_before_file_manager_file_delete_chat_scope(cheshire_cat, monkeypatch):
    """Chat scope: episodic collection, storage path = agent_key/chat_id."""
    removed_files = []
    queries = []

    class FakeHandler:
        async def get_all_tenant_points(self, collection_name, limit, offset, metadata):
            queries.append((collection_name, limit, offset, metadata))
            return [_image_point({"source": "doc.pdf", "image": True, "image_file": "doc_img_0.png"})], None

    class FakeFileManager:
        def remove_file(self, file_path):
            removed_files.append(file_path)
            return True

    monkeypatch.setattr(cheshire_cat, "vector_memory_handler", FakeHandler())
    monkeypatch.setattr(cheshire_cat, "file_manager", FakeFileManager())

    await multimodal_plugin.before_file_manager_file_delete.function("doc.pdf", "chat_abc", cheshire_cat)

    assert queries == [("episodic", 100, None, {"source": "doc.pdf", "image": True})]
    assert removed_files == ["agent_test/chat_abc/doc_img_0.png"]


# ---------- recall (used to be StrayCat._attach_recalled_images) ----------

# A real 1x1 PNG so base64 decode and Pillow ``Image.open`` both work.
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
PNG_BYTES = base64.b64decode(PNG_B64)


def _image_recall(image_file="x.png", mime="image/png", chat_id=None, recall_id="1"):
    """Build a ``DocumentRecall`` whose document metadata marks it as an image."""
    from cat.services.memory.models import DocumentRecall

    metadata = {"image": True, "image_file": image_file, "image_mime_type": mime}
    if chat_id:
        metadata["chat_id"] = chat_id
    return DocumentRecall(
        document=Document(page_content="[Image] source", metadata=metadata),
        vector=[0.1] * 4,
        id=recall_id,
    )


def _make_stray(stray_no_memory, read_result=PNG_BYTES, vision=True, recalls=None):
    """Wire a StrayCat with controlled file bytes, a controlled vision hook and
    a multimodal embedder (so the hook-level tests exercise the full path)."""
    stray = stray_no_memory
    stray.working_memory.context_memories = recalls if recalls is not None else [_image_recall()]
    stray.file_manager.read_file = lambda name, root_dir=None: read_result

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    stray.embedder = fake_embedder

    async def fake_execute_hook(name, *args, **kwargs):
        if name == "llm_vision_capable":
            return vision
        return args[0] if args else None

    stray.plugin_manager.execute_hook = fake_execute_hook
    return stray


async def test_recall_happy_path_attaches_data_uri(stray_no_memory):
    stray = _make_stray(stray_no_memory)
    read_calls = []
    stray.file_manager.read_file = lambda name, root_dir=None: (read_calls.append((name, root_dir)) or PNG_BYTES)

    images = await recall.build_recalled_images(stray, FakeMultimodalEmbedder())

    assert len(images) == 1
    part = images[0]
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # the data URI must round-trip back to the real PNG bytes
    decoded = base64.b64decode(url.split("base64,")[1])
    assert decoded == PNG_BYTES
    # root_dir follows the agent_key[/chat_id] layout
    assert read_calls == [("x.png", stray.agent_key)]


async def test_recall_non_multimodal_embedder_returns_empty(stray_no_memory):
    stray = _make_stray(stray_no_memory)
    images = await recall.build_recalled_images(stray, FakeTextEmbedder())
    assert images == []


async def test_recall_missing_file_returns_empty(stray_no_memory):
    stray = _make_stray(stray_no_memory, read_result=None)
    images = await recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert images == []


async def test_recall_vision_hook_false_returns_empty(stray_no_memory):
    stray = _make_stray(stray_no_memory, vision=False)
    images = await recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert images == []


async def test_recall_no_image_recalls_returns_empty(stray_no_memory):
    stray = _make_stray(stray_no_memory, recalls=[])
    images = await recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert images == []


async def test_recall_count_cap_enforced(stray_no_memory):
    recalls = [_image_recall(f"x{i}.png", recall_id=f"doc{i}") for i in range(recall.MAX_IMAGES_PER_TURN + 3)]
    stray = _make_stray(stray_no_memory, recalls=recalls)
    images = await recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert len(images) == recall.MAX_IMAGES_PER_TURN


async def test_recall_total_bytes_cap_enforced(stray_no_memory):
    # each image is bigger than half the total budget, so the second must be dropped
    big = b"x" * (recall.MAX_IMAGE_TOTAL_BYTES // 2 + 1)
    recalls = [_image_recall(f"big{i}.png", recall_id=f"doc{i}") for i in range(3)]
    stray = _make_stray(stray_no_memory, read_result=big, recalls=recalls)
    images = await recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert len(images) == 1


async def test_recall_chat_id_metadata_uses_chat_root_dir(stray_no_memory):
    stray = _make_stray(stray_no_memory, recalls=[_image_recall(chat_id="chat-1")])
    read_calls = []
    stray.file_manager.read_file = lambda name, root_dir=None: (read_calls.append((name, root_dir)) or PNG_BYTES)

    images = await recall.build_recalled_images(stray, FakeMultimodalEmbedder())

    assert len(images) == 1
    assert read_calls == [("x.png", f"{stray.agent_key}/chat-1")]


async def test_before_agentic_workflow_sets_task_images(stray_no_memory):
    """The plugin hook attaches the recalled images to the agentic task."""
    stray = _make_stray(stray_no_memory)
    task = AgenticWorkflowTask(user_prompt="describe this")

    task = await multimodal_plugin.before_agentic_workflow.function(task, stray)

    assert len(task.images) == 1
    assert task.images[0]["type"] == "image_url"


async def test_before_agentic_workflow_noop_for_non_chat_caller(cheshire_cat):
    """The hook leaves the task untouched when the caller has no working memory."""
    task = AgenticWorkflowTask(user_prompt="hello")

    task = await multimodal_plugin.before_agentic_workflow.function(task, cheshire_cat)

    assert task.images == []
