"""Multimodal Ingestion — owns the image lifecycle of the Cat.

This plugin centralizes every image concern that MyCAT historically kept in the
core ``RabbitHole``, ``StrayCat`` and ``routes/file_manager``:

- ingestion: extracts the images produced by multimodal parsers before chunking
  (``rabbithole_collects_document_images``) and builds the stored image points
  (``rabbithole_stores_image_points``) — embed via ``embed_images``, save as
  files via ``save_file``, points with ``image=True`` / ``image_file`` metadata.
- deletion: cascade-removes the extracted-image files when a source file is
  deleted (``before_file_manager_file_delete``), before its memory points go.
- recall: attaches the recalled multimodal images to the agentic workflow task
  (``before_agentic_workflow``) as full ``image_url`` content parts, so a
  vision-capable LLM can see them.

With this plugin disabled the core returns to upstream parity: no image
collection, no image points, no image-file cascade, no image recall.
"""

import os

from cat import hook
from cat.core_plugins.multimodal_ingestion import ingestion, recall
from cat.looking_glass.models import AgenticWorkflowTask
from cat.services.memory.models import PointStruct, VectorMemoryType


@hook(priority=0)
async def before_file_manager_file_delete(filename: str, scope: str, cat) -> None:
    """Cascade-remove the extracted-image files of a deleted source.

    Fired by the DELETE /file_manager single-file route right after the source
    file is removed from storage and BEFORE its memory points are deleted (the
    point metadata still records the image file names). Derives the collection
    (declarative/episodic) and the storage path from the deletion scope
    ("agent" or a chat id), then removes the image files of the source.
    """
    chat_id = None if scope == "agent" else scope
    collection_id = str(VectorMemoryType.DECLARATIVE if chat_id is None else VectorMemoryType.EPISODIC)
    path = cat.agent_key
    if chat_id:
        path = os.path.join(path, chat_id)
    await ingestion.delete_source_image_files(cat, collection_id, path, filename)


@hook(priority=0)
async def rabbithole_collects_document_images(images: list[dict], docs, cat) -> list[dict]:
    """Extract the images produced by multimodal parsers from the parsed docs.

    Runs BEFORE chunking (chunkers may drop the ``image_base64`` metadata).
    Collects one entry per extracted image and strips the transient payload
    from the docs; returns [] when the active embedder is not multimodal.
    """
    if not docs:
        return images
    if not await ingestion.is_multimodal_embedder_active(cat):
        return images
    extracted = ingestion.collect_document_images(docs)
    if extracted:
        ingestion.strip_image_payload(docs)
    return extracted


@hook(priority=0)
async def rabbithole_stores_image_points(
    image_points: list[PointStruct],
    images: list[dict],
    source: str,
    source_bytes: bytes | None,
    metadata: dict | None,
    file_hash: str | None,
    chat_id: str | None,
    cat,
) -> list[PointStruct]:
    """Build the image points stored alongside the source's text chunks.

    Embeds the images via ``embed_images``, saves them as files and returns the
    ``PointStruct`` list; the core appends them to the same collection.
    """
    return await ingestion.build_image_points(
        cat, images, source, source_bytes, metadata, file_hash, chat_id,
    )


@hook(priority=0)
async def before_agentic_workflow(task: AgenticWorkflowTask, cat) -> AgenticWorkflowTask:
    """Attach the recalled multimodal images to the agentic workflow task.

    Fired by ``StrayCat.__call__`` right before ``agentic_workflow.run``: when
    the active embedder is multimodal and the LLM is vision-capable, the
    recalled image points become full ``image_url`` content parts on
    ``task.images``. Duck-typed guard: only chat turns (the caller has a
    working memory) are considered.
    """
    if hasattr(cat, "working_memory"):
        embedder = await cat.embedder()
        task.images = await recall.build_recalled_images(cat, embedder)
    return task
