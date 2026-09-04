"""Multimodal ingestion helpers (image extraction and image-point building).

These functions own the image-handling behaviour that MyCAT historically kept
in the core ``RabbitHole``: collecting the images produced by multimodal
parsers before chunking, and building the ``PointStruct`` image points
(embed via ``embed_images``, save as files via ``save_file``, metadata with
``image=True`` / ``image_file``). The core now only fires the two no-op hooks
(``rabbithole_collects_document_images`` / ``rabbithole_stores_image_points``);
this module implements them.

Importing this module has zero side effects.
"""

import asyncio
import base64
import hashlib
import mimetypes
import os
import re
import time
import uuid

from langchain_core.documents import Document

from cat.services.factory.embedder import is_multimodal_embedder
from cat.services.memory.models import PointStruct


async def is_multimodal_embedder_active(cat) -> bool:
    """Check whether the active embedder supports multimodality.

    Mirrors the detection the core RabbitHole used to run: the active embedder
    instance is resolved to its settings class and ``is_multimodal()`` tells
    whether the images extracted by multimodal parsers should be embedded and
    stored in memory.

    The embedder factory must be resolved with the LIZARD's plugin manager
    (system context): the ``factory_allowed_embedders`` hooks are declared with
    a ``lizard`` parameter and ``MadHatter.context_execute_hook`` passes the
    caller under the keyword ``lizard`` only when the executing manager belongs
    to BillTheLizard. ``cat`` here is the CheshireCat (or a StrayCat exposing
    ``.lizard`` via BotMixin).
    """
    lizard = getattr(cat, "lizard", None)
    if lizard is None:
        return False

    embedder = await lizard.embedder()

    from cat.services.service_factory import ServiceFactory

    sp = ServiceFactory(
        agent_key=lizard.agent_key,
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_embedders",
        setting_category="embedder",
        schema_name="languageEmbedderName",
    )
    embedder_config = await sp.get_config_class_from_adapter(embedder)
    if embedder_config is not None:
        return bool(embedder_config.is_multimodal())

    # Custom/instance embedders not registered in the factory (e.g. test
    # fakes, ad-hoc instances): fall back to an instance-level check so the
    # multimodal path still works for them.
    return is_multimodal_embedder(embedder)


def collect_document_images(docs: list[Document]) -> list[dict]:
    """Collect the images extracted by multimodal parsers from the parsed documents.

    Multimodal parsers (e.g. the PLUS ``UnstructuredParser`` configured with
    ``extract_image_block_to_payload=True``) attach each extracted image to the
    parsed ``Document`` metadata as a base64-encoded payload in ``image_base64``,
    with the mime type in ``image_mime_type``. Walks the parsed documents and
    returns one entry per image, carrying the raw bytes ready to be embedded and
    the base64 payload metadata.

    Returns:
        A list of dicts with keys ``image_base64``, ``image_bytes`` and
        ``image_mime_type`` (defaulting to ``image/jpeg``).
    """
    images: list[dict] = []
    for doc in docs:
        image_base64 = doc.metadata.get("image_base64")
        if not image_base64:
            continue
        images.append({
            "image_base64": image_base64,
            "image_bytes": base64.b64decode(image_base64),
            "image_mime_type": doc.metadata.get("image_mime_type", "image/jpeg"),
        })
    return images


def strip_image_payload(docs: list[Document]) -> None:
    """Drop the transient parser ``image_base64`` payload from the documents.

    Called by the plugin after the images have been collected: the payload must
    never reach the text chunk metadata or the vector DB (it would bloat the
    payloads and be forwarded to the LLM on recall).
    """
    for doc in docs:
        doc.metadata.pop("image_base64", None)


def image_file_name(source: str, index: int, mime_type: str, image_bytes: bytes) -> str:
    """Deterministic, unique file name for an extracted image.

    The stem comes from the source file name so the association between an
    image and the document it was extracted from is recoverable by name.
    """
    stem = os.path.splitext(os.path.basename(source))[0]
    stem = re.sub(r"[^a-zA-Z0-9._-]", "_", stem).strip("._") or "image"
    ext = mimetypes.guess_extension(mime_type or "") or ".png"
    digest = hashlib.sha256(image_bytes).hexdigest()[:8]
    return f"{stem}_img_{index}_{digest}{ext}"


async def delete_source_image_files(cat, collection_id: str, path: str, source_name: str) -> None:
    """Remove every stored image file extracted from ``source_name``.

    Image points carry ``metadata.image == True`` and the saved file name in
    ``metadata.image_file``; query them by source before the points are deleted
    and remove the corresponding files from the agent storage. Works with any
    BaseVectorDatabaseHandler implementation (Qdrant or the Neo4j GraphRAG one).
    """
    offset = None
    while True:
        points, offset = await cat.vector_memory_handler.get_all_tenant_points(
            collection_name=collection_id, limit=100, offset=offset,
            metadata={"source": source_name, "image": True},
        )
        for point in points:
            image_file = (point.payload or {}).get("metadata", {}).get("image_file")
            if image_file:
                cat.file_manager.remove_file(os.path.join(path, image_file))
        if offset is None:
            break


async def _run_in_ingestion_executor(cat, func, *args):
    """Run a heavy ingestion callable via the dedicated ingestion lane.

    Dispatches through the ``run_in_ingestion_executor`` hook (the
    efficient_ingestion plugin provides the dedicated pool); falls back to the
    default executor when no plugin provides it.
    """
    task = func if not args else (lambda: func(*args))
    result = await cat.plugin_manager.execute_hook(
        "run_in_ingestion_executor", None, task, caller=cat,
    )
    if result is None:
        result = await asyncio.to_thread(task)
    return result


async def build_image_points(
    cat,
    images: list[dict],
    source: str,
    source_bytes: bytes | None,
    metadata: dict | None,
    file_hash: str | None,
    chat_id: str | None,
) -> list[PointStruct]:
    """Embed the collected images and build their ``PointStruct`` list.

    Called by the core ``RabbitHole.store_documents`` through the
    ``rabbithole_stores_image_points`` hook; the core appends the returned
    points to the same collection as the text chunks.

    Args:
        cat: The CheshireCat (agent-scoped) that owns the embedder and storage.
        images: Collected images (``image_base64`` / ``image_bytes`` /
            ``image_mime_type``).
        source: The source name (file name or URL).
        source_bytes: Raw bytes of the ingested source file, used when the
            source itself is an image.
        metadata: Optional extra metadata carried on every image point.
        file_hash: Hash of the source, stored on the point metadata.
        chat_id: Conversation id when the ingestion is chat-scoped.

    Returns:
        One ``PointStruct`` per successfully embedded image; failed/skipped
        image embeds (``None`` vector placeholders) are dropped entirely.
    """
    embedder = await cat.lizard.embedder()
    is_image_source = (mimetypes.guess_type(source)[0] or "").startswith("image/")

    if is_image_source:
        # Uploaded image files: the parser (hi_res) can split the file into
        # sub-crops. Embed the source file itself as a single whole-image point
        # (image_file = the source, no derived file) and ignore crops.
        whole_image = source_bytes if source_bytes is not None else (images[0]["image_bytes"] if images else None)
        embeds = await _run_in_ingestion_executor(cat, lambda: embedder.embed_images([whole_image])) if whole_image is not None else []
        files_and_vectors = [(source, embeds[0])] if embeds and embeds[0] is not None else []
    else:
        image_vectors = await _run_in_ingestion_executor(
            cat, lambda: embedder.embed_images([img["image_bytes"] for img in images])
        )
        files_and_vectors = []
        for idx, (img, vector) in enumerate(zip(images, image_vectors)):
            if vector is None:
                # A failed/skipped image embed (None placeholder) is dropped
                # entirely: neither saved as a file nor stored as a point.
                continue
            img_file = image_file_name(source, idx, img["image_mime_type"], img["image_bytes"])
            await cat.save_file(img["image_bytes"], img["image_mime_type"], img_file, chat_id)
            files_and_vectors.append((img_file, vector))

    return [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=vector,
            payload={
                "page_content": f"[Image] {source}",
                "metadata": {
                    **(metadata or {}),
                    "source": source,
                    "when": time.time(),
                    "hash": file_hash,
                    "image": True,
                    "image_file": image_file,
                    **({"chat_id": chat_id} if chat_id else {}),
                },
            },
        )
        for image_file, vector in files_and_vectors
    ]
