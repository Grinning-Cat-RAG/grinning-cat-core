"""Efficient re-embed engine (owned by the efficient_ingestion plugin).

Implements ``EfficientIngestionEngine`` — the replaceable, more efficient
implementation of the core ``BaseIngestionEngine``. It is the ONE phase machine
for the whole ingestion lifecycle: fresh uploads, recovery, re-embed and
re-ingest all go through the same two-phase flow, driven by the
``ingestion_status`` doc:

  - ``parsing_chunking``: clean-sweep of every artifact produced by this phase
    AND the following ones (text chunk points, image points, saved image files),
    then parse + chunk + ALWAYS extract the images (via the multimodal_ingestion
    plugin) and store the text chunks AND image points with EMPTY vectors
    (``vector={}``). No embedding is computed here.
  - ``embedding``: recompute the embeddings of the stored text chunks and image
    points (``embed_documents`` / ``embed_images``), replace them, then mark the
    source ``completed``.

Recovering/resuming in a phase deletes its artifacts (and the ones of the later
phases) and restarts from the beginning of that phase.

Status is written through ``ingestion_status.registry``. Import-safe: nothing
runs at import time.
"""

import asyncio
import hashlib
import io
import os
import time
import uuid

from langchain_core.documents import Document
from langchain_core.documents.base import Blob

from cat.core_plugins.base_plugin.parsers import MimeTypeBasedParser
from cat.core_plugins.efficient_ingestion.ingestion_executor import (
    run_in_ingestion_executor,
)
from cat.core_plugins.efficient_ingestion.split import split_oversized
from cat.core_plugins.ingestion_status.registry import (
    PHASE_EMBEDDING,
    PHASE_PARSING_CHUNKING,
    IngestionStatus,
    claim_source_for_resume,
    get_status,
    set_phase,
    set_status,
)
from cat.core_plugins.multimodal_ingestion.ingestion import (
    collect_document_images,
    image_file_name,
    strip_image_payload,
)
from cat.db.cruds import settings as crud_settings
from cat.env import get_env_int
from cat.log import log
from cat.looking_glass.models import StoredSourceWithMetadata
from cat.services.factory.embedder import is_multimodal_embedder
from cat.services.factory.ingestion import BaseIngestionEngine
from cat.services.memory.models import PointStruct, VectorMemoryType
from cat.utils import get_nlp_object_name, guess_file_type, is_url


async def _set_status(ccat, source: str, status: IngestionStatus, error: str | None = None, chat_id: str | None = None) -> None:
    """Best-effort status write via the plugin's own registry."""
    scope = str(chat_id) if chat_id else "agent"
    try:
        await set_status(
            ccat.agent_key,
            scope,
            source,
            type_="url" if is_url(source) else "file",
            status=status,
            chat_id=chat_id,
            error=error,
        )
    except Exception as e:  # noqa: BLE001 - status must never break the re-embed pass
        log.error(f"Agent id: {ccat._id}. Failed to write ingestion status for {source}: {e}")


def _claim_stale_after() -> float:
    """Staleness gate for the per-source claim during the re-embed pass.

    Reuses the resume threshold so a source being actively re-embedded by
    another worker is not re-claimed. ``<=0`` via env disables the gate.
    """
    value = get_env_int("CAT_INGESTION_RESUME_STALE_SECONDS")
    return float(value) if value and value > 0 else 0.0


async def _cleanup_orphan_images(ccat, collection_name, source_name, chat_id) -> None:
    """Remove image points + saved image files of a source before a full re-ingest.

    A full re-ingest (parsing_chunking) regenerates the chunks AND re-extracts
    the images from scratch. The image points and their saved ``image_file``
    from the PREVIOUS (possibly incomplete) parse are therefore stale: delete
    the points in the target first and the files from the agent storage, so the
    re-ingest starts clean and no orphan image file lingers on disk.
    """
    try:
        points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
            str(collection_name), with_vectors=False,
            metadata={"source": source_name, "image": True},
        )
    except Exception:  # noqa: BLE001 - cleanup must never break the pass
        return
    image_files = [
        (p.payload or {}).get("metadata", {}).get("image_file")
        for p in points
        if (p.payload or {}).get("metadata", {}).get("image_file")
    ]
    if not image_files:
        return
    root_dir = ccat.agent_key
    if chat_id:
        root_dir = os.path.join(root_dir, str(chat_id))
    for image_file in image_files:
        try:
            ccat.file_manager.remove_file(os.path.join(root_dir, image_file))
        except Exception:  # noqa: BLE001,S110 - best-effort, cleanup must never break the pass
            pass
    await ccat.vector_memory_handler.delete_tenant_points(
        str(collection_name), metadata={"source": source_name, "image": True}
    )


async def _resolve_callers(ccat, chat_id):
    """Return the RabbitHole/plugin-manager caller and the embedding cat for a source.

    Agent-scoped sources use the CheshireCat itself (scope ``"agent"``); a
    chat-scoped source needs its StrayCat for the correct ``scope``/hook caller.
    Returns ``(cat, scope, chat_id)`` where ``cat`` is what receives the
    ``after_rabbithole_stored_documents`` hook caller, or ``(None, None, None)``
    when the chat does not (or no longer) exist.
    """
    if not chat_id:
        return ccat, "agent", None
    stray_cat = await ccat._find_stray_cat(str(chat_id))
    if stray_cat is None:
        return None, str(chat_id), chat_id
    return stray_cat, str(chat_id), chat_id


async def _clear_source_artifacts(ccat, collection_name, source_name, chat_id) -> None:
    """Remove EVERY artifact of a source: text chunk points, image points and
    saved image files. Used when a phase is restarted (clean-sweep of the phase
    and all the later ones)."""
    # image files + image points first (the point metadata carries image_file)
    await _cleanup_orphan_images(ccat, collection_name, source_name, chat_id)
    # then any remaining point (text chunks and any leftover image point)
    await ccat.vector_memory_handler.delete_tenant_points(
        str(collection_name), metadata={"source": source_name}
    )


async def _parse_and_chunk(
    ccat, rabbit_hole, source, file_bytes, content_type, cat
):
    """Parse the source bytes into chunked LangChain docs, ALWAYS extracting images.

    Mirrors the core ``_parse_to_docs`` but collects the extracted images
    UNCONDITIONALLY (the multimodal_ingestion plugin's pure helper reads the
    ``image_base64`` payload the parser attached), strips the transient payload
    from the docs, and returns ``(docs, images)``.
    """
    source_name = source.name
    fh = await ccat.file_handlers()
    super_docs = await run_in_ingestion_executor(
        lambda: MimeTypeBasedParser(handlers=fh).parse(
            Blob(data=file_bytes, mimetype=content_type).from_data(
                data=file_bytes, mime_type=content_type, path=source_name
            )
        )
    )
    for doc in super_docs:
        if isinstance(doc.metadata, dict):
            doc.metadata.setdefault("source", source_name)

    # ALWAYS extract the images (regardless of the active embedder): the
    # embedding phase decides later whether to embed them as images or keep
    # them payload-only.
    images = collect_document_images(super_docs)
    strip_image_payload(super_docs)

    docs = await rabbit_hole._split_text(super_docs)
    return docs, images


async def _store_empty_vectors(
    ccat, collection_name, source_name, chat_id, docs, images,
    file_hash=None, metadata=None,
):
    """Store the parsed text chunks and image points with EMPTY vectors.

    Phase ``parsing_chunking`` output: the points carry the full payload (text
    page_content + image file references in metadata) but ``vector={}`` — no
    embedding is computed here. The image files are persisted via ``save_file``
    so the embedding phase can re-read them.

    Returns the list of stored ``PointStruct``.
    """
    points: list[PointStruct] = []

    for doc in docs:
        # enrich the metadata as the core store_documents does: source, when,
        # hash (deduped compute) and chat_id for chat-scoped sources.
        doc.metadata = (
            doc.metadata
            | (metadata or {})
            | {"source": source_name, "when": time.time(), "hash": file_hash}
            | ({"chat_id": chat_id} if chat_id else {})
        )
        points.append(
            PointStruct(id=uuid.uuid4().hex, payload=doc.model_dump(), vector={})
        )

    if images:
        for idx, img in enumerate(images):
            image_bytes = img["image_bytes"]
            mime_type = img["image_mime_type"]
            img_file = image_file_name(source_name, idx, mime_type, image_bytes)
            await ccat.save_file(image_bytes, mime_type, img_file, chat_id)
            points.append(
                PointStruct(
                    id=uuid.uuid4().hex,
                    payload={
                        "page_content": f"[Image] {source_name}",
                        "metadata": {
                            **(metadata or {}),
                            "source": source_name,
                            "when": time.time(),
                            "image": True,
                            "image_file": img_file,
                            **({"chat_id": chat_id} if chat_id else {}),
                        },
                    },
                    vector={},
                )
            )

    await ccat.vector_memory_handler.add_points_to_tenant(
        collection_name=str(collection_name), points=points
    )
    return points


async def _embed_phase(ccat, collection_name, source_name, source_points, embedder, chat_id, cat, rabbit_hole):
    """Recompute the embeddings of the stored points and mark the source completed.

    Phase ``embedding``: rebuild LangChain docs from the stored text points and
    recompute their vectors via ``embed_documents``; for image points, embed via
    ``embed_images`` when the active embedder is multimodal, otherwise keep them
    payload-only (``vector={}``). The old points are deleted and replaced. Fires
    ``after_rabbithole_stored_documents`` so analytics / ingestion_status are
    informed, then returns the stored points.

    Returns the list of replaced points (with vectors), or None when the
    embedding failed (the source is left for a later pass).
    """
    source_points_text = [p for p in source_points if not (p.payload or {}).get("metadata", {}).get("image")]
    source_points_image = [p for p in source_points if (p.payload or {}).get("metadata", {}).get("image")]

    points: list[PointStruct] = []

    # --- text points: recompute the vector from the stored page_content ---
    if source_points_text:
        docs = [
            Document(
                page_content=(p.payload or {}).get("page_content", ""),
                metadata=dict((p.payload or {}).get("metadata", {})),
            )
            for p in source_points_text
        ]
        # Re-chunk only chunks exceeding the current embedder's input limit.
        docs = split_oversized(docs, embedder)
        vectors = await run_in_ingestion_executor(
            embedder.embed_documents, [d.page_content for d in docs]
        )
        points.extend(
            PointStruct(id=uuid.uuid4().hex, payload=d.model_dump(), vector=vector)
            for d, vector in zip(docs, vectors)
        )

    # --- image points: embed via embed_images when multimodal, else payload-only ---
    if source_points_image:
        if is_multimodal_embedder(embedder):
            recoverable = []
            for p in source_points_image:
                meta = (p.payload or {}).get("metadata", {})
                image_file = meta.get("image_file")
                root_dir = ccat.agent_key
                if metadata_chat := meta.get("chat_id"):
                    root_dir = os.path.join(root_dir, str(metadata_chat))
                image_bytes = ccat.file_manager.read_file(image_file, root_dir) if image_file else None
                if image_bytes is None:
                    # H2 fallback: the image file is gone; keep the point
                    # payload-only (no vector) instead of failing the source.
                    points.append(
                        PointStruct(
                            id=uuid.uuid4().hex,
                            payload={"page_content": f"[Image] {source_name}", "metadata": dict(meta)},
                            vector={},
                        )
                    )
                else:
                    recoverable.append((p, image_bytes))
            if recoverable:
                image_vectors = await run_in_ingestion_executor(
                    embedder.embed_images, [b for _, b in recoverable]
                )
                if len(image_vectors) != len(recoverable):
                    raise ValueError(
                        f"embed_images returned {len(image_vectors)} vectors "
                        f"for {len(recoverable)} images"
                    )
                for (p, _b), vector in zip(recoverable, image_vectors):
                    meta = dict((p.payload or {}).get("metadata", {}))
                    points.append(
                        PointStruct(
                            id=uuid.uuid4().hex,
                            payload={"page_content": f"[Image] {source_name}", "metadata": meta},
                            vector=vector,
                        )
                    )
        else:
            # Non-multimodal embedder: preserve the image points payload-only.
            for p in source_points_image:
                meta = dict((p.payload or {}).get("metadata", {}))
                points.append(
                    PointStruct(
                        id=uuid.uuid4().hex,
                        payload={"page_content": f"[Image] {source_name}", "metadata": meta},
                        vector={},
                    )
                )

    # All vectors were computed BEFORE the delete so a failure leaves the old
    # points intact (compute-before-delete).
    await ccat.vector_memory_handler.delete_tenant_points(
        str(collection_name), metadata={"source": source_name}
    )
    await ccat.vector_memory_handler.add_points_to_tenant(
        collection_name=str(collection_name), points=points
    )

    # Token accounting + completion signal.
    await ccat.plugin_manager.execute_hook(
        "after_rabbithole_stored_documents", source_name, points, caller=cat,
    )
    return points


async def reembed_sources(
    ccat,
    collection_name: VectorMemoryType,
    stored_sources: list[StoredSourceWithMetadata],
    stale_after: float | None = None,
    caller_cat=None,
) -> None:
    """
    Run the ONE two-phase ingestion machine for a set of stored sources.

    ``caller_cat``: optional pre-resolved caller (the StrayCat/CheshireCat that
    issued the ingestion). When provided for chat-scoped sources it is used as
    the hook caller instead of re-resolving it via ``_resolve_callers`` (which
    is needed by the background re-embed/resume passes but would fail for a
    brand-new chat upload).

    Phase decision (per source), from the ``ingestion_status`` doc:
      - ``completed`` + embedder == active + chunker == active  -> skip
      - ``completed`` + chunker mismatch                         -> ``parsing_chunking``
      - ``completed`` + embedder mismatch (chunks valid)         -> ``embedding``
      - in-flight / stale row (``uploaded``/``processing``/``error``/``downloading``/``downloaded``):
          resumes from the phase recorded in ``doc["phase"]`` (conservative).
      - no status doc / old row -> ``embedding`` if reusable points exist, else ``parsing_chunking``.

    For each source that must run ``parsing_chunking``, the source's artifacts
    (text/image points and saved image files) are FIRST deleted, then the file
    is parsed and the chunks + images stored with EMPTY vectors; finally the
    ``embedding`` phase recomputes the vectors and marks the source completed.
    On a crash between the two phases, the doc records ``embedding`` and the
    next pass resumes from there (the empty-vector points are still present).
    """
    log.info(f"Agent id: {ccat._id}. Embedding stored files to the vector memory")

    existing_points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
        str(collection_name), with_vectors=False
    )

    rabbit_hole = ccat.rabbit_hole
    embedder = await ccat.embedder()
    active_embedder_name = getattr(embedder, "name", None) or get_nlp_object_name(embedder, "default_embedder")
    chunker = getattr(ccat, "chunker", None)
    active_chunker_name = (
        str(chunker.name) if chunker is not None and getattr(chunker, "name", None) else
        get_nlp_object_name(chunker, "default_chunker")
    )
    owner = f"reembed-{os.getpid()}"
    counter = 0

    for source in stored_sources:
        source_name = source.name
        chat_id = source.metadata.get("chat_id")

        if caller_cat is not None:
            # ingestion flow: the caller (StrayCat/CheshireCat) is already
            # resolved; use it directly and derive the scope from it
            cat = caller_cat
            scope = str(cat.id) if hasattr(cat, "id") and getattr(cat, "id", None) else "agent"
            chat_id = chat_id if chat_id else (getattr(cat, "id", None) if hasattr(cat, "id") else None)
        else:
            cat, scope, chat_id = await _resolve_callers(ccat, chat_id)
            if cat is None:
                log.warning(
                    f"Stray cat with id {chat_id} not found. Skipping file {source.path}/{source.name}"
                )
                continue

        # ---- decide the start phase from the status doc ----
        doc = await get_status(ccat.agent_key, scope, source_name)
        doc_status = (doc or {}).get("status")
        doc_embedder = (doc or {}).get("embedder_name")
        doc_chunker = (doc or {}).get("chunker_name")
        doc_phase = (doc or {}).get("phase")

        if (
            doc_status == IngestionStatus.COMPLETED.value
            and doc_embedder == active_embedder_name
            and doc_chunker == active_chunker_name
        ):
            log.debug(
                f"Agent id: {ccat._id}. Source {source_name}: already completed with the "
                f"active embedder/chunker ({active_embedder_name!r}/{active_chunker_name!r}), skipping"
            )
            continue

        if doc_status == IngestionStatus.COMPLETED.value:
            if doc_chunker != active_chunker_name:
                start_phase = PHASE_PARSING_CHUNKING
            else:
                start_phase = PHASE_EMBEDDING
        elif doc_status in (
            IngestionStatus.UPLOADED.value,
            IngestionStatus.PROCESSING.value,
            IngestionStatus.ERROR.value,
            IngestionStatus.DOWNLOADING.value,
            IngestionStatus.DOWNLOADED.value,
        ):
            # in-flight / stale: resume from the recorded phase (conservative).
            # A chunker change invalidates even an embedding-phase row: the stored
            # chunks were produced by the OLD chunker, so a chunk-reuse would keep
            # stale chunks -> full re-ingest (parsing_chunking) instead.
            if doc_chunker != active_chunker_name:
                start_phase = PHASE_PARSING_CHUNKING
            else:
                start_phase = doc_phase if doc_phase in (PHASE_EMBEDDING, PHASE_PARSING_CHUNKING) else PHASE_PARSING_CHUNKING
        else:
            # no doc or unknown: embedding if reusable points exist, else full re-ingest
            has_points = any(
                (p.payload or {}).get("metadata", {}).get("source") == source_name
                for p in existing_points
            )
            start_phase = PHASE_EMBEDDING if has_points else PHASE_PARSING_CHUNKING

        # ---- log the phase transition (debug) ----
        log.debug(
            f"Agent id: {ccat._id}. Source {source_name}: ingestion phase "
            f"{doc_phase or '(none)'} -> {start_phase} (status {doc_status or '(none)'} -> "
            f"{IngestionStatus.PROCESSING.value}, embedder {doc_embedder!r} -> {active_embedder_name!r}, "
            f"chunker {doc_chunker!r} -> {active_chunker_name!r})"
        )

        # ---- claim the per-source work (skip if another worker holds it) ----
        if doc is not None:
            claimed = await claim_source_for_resume(
                ccat.agent_key,
                scope,
                source_name,
                stale_after=_claim_stale_after() if stale_after is None else stale_after,
                owner=owner,
                claim_completed=(doc_status == IngestionStatus.COMPLETED.value),
            )
            if claimed is None:
                # another worker is already (re)processing this source
                continue

        try:
            # ---- EXECUTE the phase(s), possibly chaining parsing -> embedding ----
            current_phase = start_phase

            if current_phase == PHASE_EMBEDDING:
                # chunks (text) already stored for this source: chunk-reuse path.
                # For episodic sources, the chat must still exist.
                if chat_id and not (await ccat._find_stray_cat(str(chat_id))):
                    log.warning(
                        f"Stray cat with id {chat_id} not found. Cleaning up {source.path}/{source.name}"
                    )
                    await ccat.vector_memory_handler.delete_tenant_points(
                        str(collection_name), metadata={"source": source_name}
                    )
                    await _set_status(ccat, source_name, IngestionStatus.COMPLETED, chat_id=chat_id)
                    continue

                await set_phase(
                    ccat.agent_key, scope, source_name,
                    PHASE_EMBEDDING,
                    embedder_name=active_embedder_name,
                    type_="url" if is_url(source_name) else "file",
                    chat_id=chat_id,
                )
                source_points = [
                    p for p in existing_points
                    if (p.payload or {}).get("metadata", {}).get("source") == source_name
                ]
                if source_points:
                    await _embed_phase(
                        ccat, collection_name, source_name, source_points,
                        embedder, chat_id, cat, rabbit_hole,
                    )
                    counter += 1
                    continue
                # no stored points -> fall through to parsing_chunking
                current_phase = PHASE_PARSING_CHUNKING

            if current_phase == PHASE_PARSING_CHUNKING:
                # ---- parsing_chunking phase ----
                await set_phase(
                    ccat.agent_key, scope, source_name,
                    PHASE_PARSING_CHUNKING,
                    chunker_name=active_chunker_name,
                    type_="url" if is_url(source_name) else "file",
                    chat_id=chat_id,
                )

                # 1. clean-sweep: remove every artifact of this phase and the
                #    following ones (text chunk points, image points, image files).
                await _clear_source_artifacts(ccat, collection_name, source_name, chat_id)

                # ensure the rabbit_hole context is wired on the ccat (URL
                # download + parsing helpers read ``self.cat``)
                if rabbit_hole.cat is None:
                    await rabbit_hole.setup(ccat)

                # 2. resolve the source bytes: file already on disk (via the
                #    resume/upload), or a URL to (re)download.
                if source.content is not None:
                    file_io = source.content
                    file_bytes = file_io.read()
                    content_type, _ = guess_file_type(file_io)
                elif is_url(source_name):
                    # URL: re-download via the core source resolver.
                    source_name_resolved, file_bytes, content_type, _ = await rabbit_hole._resolve_source_bytes(
                        source_name, source_name, None
                    )
                    if file_bytes is None:
                        raise Exception(f"Something went wrong with the source '{source_name}'")
                    if source_name_resolved:
                        source_name = source_name_resolved
                else:
                    # re-read from disk (the persisted file)
                    path = ccat.agent_key
                    if chat_id:
                        path = os.path.join(path, str(chat_id))
                    file_bytes = ccat.file_manager.read_file(source_name, path)
                    if file_bytes is None:
                        raise Exception(f"File '{source_name}' not found on disk; cannot re-ingest.")
                    content_type = None

                # 3. parse + chunk + ALWAYS extract images.
                docs, images = await _parse_and_chunk(
                    ccat, rabbit_hole, source, file_bytes, content_type, cat
                )
                if not docs:
                    raise Exception(f"No valid chunks found in the file '{source_name}'.")

                # 4. store text chunks + image points with EMPTY vectors.
                sha256 = hashlib.sha256()
                sha256.update(file_bytes or b"")
                file_hash = sha256.hexdigest()
                stored = await _store_empty_vectors(
                    ccat, collection_name, source_name, chat_id,
                    docs, images, file_hash=file_hash, metadata=source.metadata or {},
                )

                # 5. transition to the embedding phase and recompute the vectors.
                await set_phase(
                    ccat.agent_key, scope, source_name,
                    PHASE_EMBEDDING,
                    embedder_name=active_embedder_name,
                    type_="url" if is_url(source_name) else "file",
                    chat_id=chat_id,
                )
                await _embed_phase(
                    ccat, collection_name, source_name, stored,
                    embedder, chat_id, cat, rabbit_hole,
                )
                counter += 1

        except Exception as e:  # noqa: BLE001 - a failing source must not abort the pass
            log.error(f"Agent id: {ccat._id}. Error re-embedding source {source_name}: {e}")
            await _set_status(ccat, source_name, IngestionStatus.ERROR, error=str(e), chat_id=chat_id)

    log.info(f"Agent id: {ccat._id}. Embedded {counter} files to the vector memory")


class EfficientIngestionEngine(BaseIngestionEngine):
    """Efficient re-embed engine (one two-phase phase machine).

    Pluggable implementation of the base ingestion engine, registered by the
    plugin through the ``factory_allowed_ingestions`` hook as
    ``EfficientIngestionConfiguration``. Runs the same phase machine for fresh
    ingestion (via ``ingest_file``) and for the re-embed pass (via ``run``),
    writes the ingestion status through ``ingestion_status.registry`` and honors
    a configurable concurrency cap (settings category ``ingestion``).
    """

    def __init__(self, ingestion_max_concurrency: int = 5, reembed_max_concurrency: int | None = None, **kwargs):
        super().__init__(**kwargs)
        # legacy alias: pre-rename configs saved `reembed_max_concurrency`
        if (
            reembed_max_concurrency is not None
            and reembed_max_concurrency > 0
            and ingestion_max_concurrency == 5
        ):
            ingestion_max_concurrency = reembed_max_concurrency
        self.ingestion_max_concurrency = max(1, int(ingestion_max_concurrency))

    async def ingest_file(
        self,
        cat,
        file,
        filename: str | None = None,
        metadata: dict | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ) -> None:
        """Ingest a single file through the two-phase phase machine.

        Similar lifecycle to the core flow: fires the ingestion/processing hooks,
        persists the file (when ``store_file``), then hands the source to the
        phase machine (parsing_chunking -> embedding -> completed).
        """
        source_name = filename or (file if isinstance(file, str) else None)
        if not source_name:
            raise ValueError("No filename provided.")

        # normalize the caller: the routes pass a StrayCat (chat scope) or a
        # CheshireCat (agent scope). The phase machine works on the CheshireCat
        # (ccat) and derives the chat scope from the source metadata.
        if hasattr(cat, "agent_key") and not hasattr(cat, "_id"):
            # StrayCat: resolve its CheshireCat and keep the chat id
            ccat = await cat.lizard.get_cheshire_cat(cat.agent_key)
            chat_id = cat.id
        else:
            ccat = cat
            chat_id = None
        if ccat is None:
            raise ValueError(f"Agent '{getattr(cat, 'agent_key', None)}' not found; cannot ingest.")

        scope = str(chat_id) if chat_id else "agent"
        collection_name = VectorMemoryType.EPISODIC if chat_id else VectorMemoryType.DECLARATIVE

        # Materialize the file bytes once (the BytesIO from the route is consumed
        # by reads); both persistence and the phase machine use a fresh copy.
        if is_url(source_name):
            file_data = None
            content = None
        elif isinstance(file, bytes):
            file_data = file
            content = io.BytesIO(file)
        elif hasattr(file, "read"):
            file_data = file.read()
            content = io.BytesIO(file_data)
        else:
            file_data = None
            content = file

        # lifecycle: start + persist (durable across restarts) + processing
        await ccat.plugin_manager.execute_hook(
            "rabbithole_ingestion_start", source_name, metadata or {},
            is_url(source_name), caller=cat,
        )
        if store_file and file_data is not None:
            await ccat.save_file(file_data, content_type, source_name, chat_id)
        await ccat.plugin_manager.execute_hook(
            "rabbithole_ingestion_processing", source_name, caller=cat,
        )
        heartbeat_interval = get_env_int("CAT_INGESTION_HEARTBEAT_SECONDS") or 30
        await ccat.plugin_manager.execute_hook(
            "rabbithole_processing_heartbeat_start",
            source_name, scope, heartbeat_interval, caller=cat,
        )

        try:
            source_obj = StoredSourceWithMetadata(
                name=source_name, path=source_name, content=content,
                metadata={**(metadata or {}), **({"chat_id": chat_id} if chat_id else {})},
            )
            await reembed_sources(
                ccat, collection_name, [source_obj], stale_after=0.0,
                caller_cat=cat,
            )
        finally:
            await ccat.plugin_manager.execute_hook(
                "rabbithole_processing_heartbeat_stop",
                source_name, scope, caller=cat,
            )

    async def run(self, lizard) -> bool:
        """Resolve the current embedder and run the re-embed pass."""
        success = False
        try:
            embedder = await lizard.embedder()
            embedder_name = embedder.name
            embedder_size = embedder.size

            ccat_ids = await crud_settings.get_agents_main_keys()
            stored_files_by_ccat = []
            # first, get all the stored files from all the Cheshire Cats with the
            # metadata stored within the vector memory; nothing is removed from
            # the latter to avoid any race condition
            for ccat_id in ccat_ids:
                if (ccat := await lizard.get_cheshire_cat(ccat_id)) is None:
                    continue
                stored_files_by_ccat.append({
                    "ccat": ccat,
                    "stored_sources": await ccat.get_stored_sources_with_metadata(),
                })

            # re-initialize all the vector databases in a serialized way, outside
            # threads to avoid race conditions
            for entry in stored_files_by_ccat:
                await entry["ccat"].vector_memory_handler.initialize(embedder_name, embedder_size)

            # then re-embed every stored file/procedure, limiting concurrent
            # embeddings to avoid overwhelming resources (tunable via the plugin
            # settings, category ingestion)
            semaphore = asyncio.Semaphore(self.ingestion_max_concurrency)

            async def embed_with_limit(entry_):
                async with semaphore:
                    tasks = [
                        reembed_sources(entry_["ccat"], collection_name, sources)
                        for collection_name, sources in entry_["stored_sources"].items()
                        if sources
                    ] + [entry_["ccat"].embed_procedures()]
                    await asyncio.gather(*tasks)

            await asyncio.gather(*[embed_with_limit(entry) for entry in stored_files_by_ccat])

            success = True
        except Exception as e:  # noqa: BLE001 - surfaced on the hook, never raised
            log.error(f"Error embedding all stored files: {e}")

        await lizard.plugin_manager.execute_hook(
            "after_all_cheshire_cats_embedded", success, caller=lizard,
        )
        return success
