"""Tests for the multimodal recall logic (now in the ``multimodal_ingestion`` plugin)
and image rendering in ``CoreAgenticWorkflow``.

Multimodal-images-to-llm plan: pin the behavior of the seam that
recovers recalled multimodal memory images (``metadata.image is True``) from the
file manager, builds ``data:`` URI content parts, applies the per-turn caps, and
hands them to the agentic workflow as full LangChain ``image_url`` content parts.

The suite covers: the happy path (multimodal embedder + image recall + real PNG
bytes), the empty-result gates (non-multimodal embedder, vision hook False,
missing file, no image recalls), the per-turn caps (count and total bytes), and
the rendering path (the ``image_url`` part survives into the messages actually
passed to the LLM).
"""
import base64
from io import BytesIO

from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image

from cat.core_plugins.multimodal_ingestion import recall as mm_recall
from cat.looking_glass.models import AgenticWorkflowOutput, AgenticWorkflowTask
from cat.services.factory.agentic_workflow import CoreAgenticWorkflow
from cat.services.factory.embedder import MultimodalEmbeddings
from cat.services.memory.models import DocumentRecall

# A real 1x1 PNG so base64 decode and Pillow ``Image.open`` both work.
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
PNG_BYTES = base64.b64decode(PNG_B64)


class FakeMultimodalEmbedder(MultimodalEmbeddings):
    """Concrete ``MultimodalEmbeddings`` subclass (implements the abstract API)."""

    def embed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 4

    def embed_image(self, image):
        return [0.1] * 4

    def embed_images(self, images):
        return [[0.1] * 4 for _ in images]


class FakeTextEmbedder:
    """Plain text-only embedder: exposes ``embed_documents`` but no image API."""

    name = "FakeTextEmbedder"
    size = 4

    def embed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 4


def _image_recall(image_file="x.png", mime="image/png", chat_id=None, recall_id="1"):
    """Build a ``DocumentRecall`` whose document metadata marks it as an image."""
    metadata = {"image": True, "image_file": image_file, "image_mime_type": mime}
    if chat_id:
        metadata["chat_id"] = chat_id
    return DocumentRecall(
        document=LangChainDocument(page_content="[Image] source", metadata=metadata),
        vector=[0.1] * 4,
        id=recall_id,
    )


def _make_stray(stray_no_memory, read_result: bytes | None = PNG_BYTES, vision=True, recalls=None):
    """Wire a StrayCat with controlled file bytes and a controlled vision hook."""
    stray = stray_no_memory
    stray.working_memory.context_memories = recalls if recalls is not None else [_image_recall()]
    stray.file_manager.read_file = lambda name, root_dir=None: read_result

    async def fake_execute_hook(name, *args, **kwargs):
        if name == "llm_vision_capable":
            return vision
        return args[0] if args else None

    stray.plugin_manager.execute_hook = fake_execute_hook
    return stray


async def test_happy_path_attaches_data_uri(stray_no_memory):
    stray = _make_stray(stray_no_memory)
    read_calls = []
    stray.file_manager.read_file = lambda name, root_dir=None: (read_calls.append((name, root_dir)) or PNG_BYTES)

    images = await mm_recall.build_recalled_images(stray, FakeMultimodalEmbedder())

    assert len(images) == 1
    part = images[0]
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # the data URI must round-trip back to the real PNG bytes
    decoded = base64.b64decode(url.split("base64,")[1])
    assert decoded == PNG_BYTES
    assert Image.open(BytesIO(decoded)).format == "PNG"
    # root_dir follows the agent_key[/chat_id] layout
    assert read_calls == [("x.png", stray.agent_key)]


async def test_non_multimodal_embedder_returns_empty(stray_no_memory):
    stray = _make_stray(stray_no_memory)
    images = await mm_recall.build_recalled_images(stray, FakeTextEmbedder())
    assert images == []


async def test_missing_file_returns_empty(stray_no_memory):
    stray = _make_stray(stray_no_memory, read_result=None)
    images = await mm_recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert images == []


async def test_vision_hook_false_returns_empty(stray_no_memory):
    stray = _make_stray(stray_no_memory, vision=False)
    images = await mm_recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert images == []


async def test_no_image_recalls_returns_empty(stray_no_memory):
    stray = _make_stray(stray_no_memory, recalls=[])
    images = await mm_recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert images == []


async def test_count_cap_enforced(stray_no_memory):
    recalls = [_image_recall(f"x{i}.png", recall_id=f"doc{i}") for i in range(mm_recall.MAX_IMAGES_PER_TURN + 3)]
    stray = _make_stray(stray_no_memory, recalls=recalls)
    images = await mm_recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert len(images) == mm_recall.MAX_IMAGES_PER_TURN


async def test_total_bytes_cap_enforced(stray_no_memory):
    # each image is bigger than half the total budget, so the second must be dropped
    big = b"x" * (mm_recall.MAX_IMAGE_TOTAL_BYTES // 2 + 1)
    recalls = [_image_recall(f"big{i}.png", recall_id=f"doc{i}") for i in range(3)]
    stray = _make_stray(stray_no_memory, read_result=big, recalls=recalls)
    images = await mm_recall.build_recalled_images(stray, FakeMultimodalEmbedder())
    assert len(images) == 1


async def test_chat_id_metadata_uses_chat_root_dir(stray_no_memory):
    stray = _make_stray(stray_no_memory, recalls=[_image_recall(chat_id="chat-1")])
    read_calls = []
    stray.file_manager.read_file = lambda name, root_dir=None: (read_calls.append((name, root_dir)) or PNG_BYTES)

    images = await mm_recall.build_recalled_images(stray, FakeMultimodalEmbedder())

    assert len(images) == 1
    assert read_calls == [("x.png", f"{stray.agent_key}/chat-1")]


class CapturingWorkflow(CoreAgenticWorkflow):
    """CoreAgenticWorkflow that captures the prompt instead of invoking an LLM."""

    def __init__(self):
        super().__init__()
        self.captured_prompt: ChatPromptTemplate | None = None

    async def _run_no_tool_binding(self, prompt):
        self.captured_prompt = prompt
        return AgenticWorkflowOutput(output="ok")


class FakeLLM:
    """Stand-in LLM: no ``bind_tools``, so the no-tool-binding path is taken."""


async def test_rendering_preserves_image_url():
    wf = CapturingWorkflow()
    task = AgenticWorkflowTask(
        user_prompt="describe this",
        images=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
    )

    await wf.run(task=task, llm=FakeLLM())

    assert wf.captured_prompt is not None
    rendered = wf.captured_prompt.format_messages()
    found = any(
        isinstance(getattr(m, "content", None), list)
        and any(
            p.get("type") == "image_url"
            and p.get("image_url", {}).get("url", "").startswith("data:image/png;base64,")
            for p in m.content
        )
        for m in rendered
    )
    assert found