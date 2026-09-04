"""Multimodal recall helper (attach recalled images to the LLM prompt).

Owns the image-recall behaviour that MyCAT historically kept in the core
``StrayCat._attach_recalled_images``: recovering the recalled multimodal memory
images and building full LangChain content parts
(``{"type": "image_url", "image_url": {"url": "<data-uri>"}}``) so a
vision-capable LLM can see them. The core now only fires the
``before_agentic_workflow`` no-op hook; the plugin implements it here.

Importing this module has zero side effects.
"""

import base64
import os
from io import BytesIO

from PIL import Image

from cat.log import log
from cat.services.factory.embedder import is_multimodal_embedder

#: Caps for attaching recalled multimodal images to the LLM prompt per turn.
#: Consumed by ``build_recalled_images`` to bound how many recalled images get
#: attached to the LLM prompt in a single turn.
MAX_IMAGES_PER_TURN = 5
MAX_IMAGE_TOTAL_BYTES = 5 * 1024 * 1024  # 5 MiB total across all images in one turn


async def build_recalled_images(cat, embedder) -> list[dict]:
    """Recover recalled multimodal memory images and build LLM content parts.

    Returns a list of full LangChain content parts
    (``{"type": "image_url", "image_url": {"url": "<data-uri>"}}``) for the
    images recalled into the working memory this turn, so a vision-capable
    LLM can see them. Returns ``[]`` when the embedder is not multimodal,
    the LLM is not vision-capable, there are no image recalls, or anything
    fails — the ``[Image]`` text placeholder already in the context keeps
    the turn working text-only.
    """
    try:
        if not is_multimodal_embedder(embedder):
            return []

        capable = await cat.plugin_manager.execute_hook("llm_vision_capable", True, caller=cat)
        if not capable:
            return []

        images: list[dict] = []
        total_bytes = 0
        for m in cat.working_memory.context_memories:
            if len(images) >= MAX_IMAGES_PER_TURN:
                break

            metadata = m.document.metadata
            if metadata.get("image") is not True:
                continue

            image_file = metadata.get("image_file")
            if not image_file:
                continue

            # Same agent_key[/chat_id] layout as CheshireCat.save_file and the
            # image-file handling of this plugin's ingestion module.
            root_dir = cat.agent_key
            if chat_id := metadata.get("chat_id"):
                root_dir = os.path.join(root_dir, str(chat_id))

            image_bytes = cat.file_manager.read_file(image_file, root_dir)
            if image_bytes is None:
                continue

            if total_bytes + len(image_bytes) > MAX_IMAGE_TOTAL_BYTES:
                break

            mime = metadata.get("image_mime_type")
            if not mime:
                try:
                    fmt = Image.open(BytesIO(image_bytes)).format
                    mime = Image.MIME[fmt] if fmt else "image/png"
                except Exception:  # noqa: BLE001 - best-effort mime sniffing, never breaks the turn
                    mime = "image/png"

            data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
            images.append({"type": "image_url", "image_url": {"url": data_uri}})
            total_bytes += len(image_bytes)

        return images
    except Exception as e:  # noqa: BLE001 - image recall is best-effort, never breaks the turn
        log.warning(f"Agent id: {cat.agent_key}. Could not attach recalled images to the LLM prompt. Error: {e}")
        return []
