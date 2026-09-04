import asyncio
import hashlib
import json
import mimetypes
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Dict, List, Tuple

from httpx import AsyncClient
from langchain_core.documents.base import Blob, Document

from cat.core_plugins.base_plugin.parsers import MimeTypeBasedParser
from cat.env import get_env_int
from cat.log import log
from cat.services.factory.chunker import BaseChunker
from cat.services.memory.models import PointStruct, VectorMemoryType
from cat.utils import is_url as fnc_is_url

# Process-wide semaphore bounding concurrent ``ingest_file`` executions.
# Created lazily on first use; the size is read at runtime from
# ``CAT_INGESTION_MAX_CONCURRENCY`` (``None`` or ``<= 0`` means unlimited).
_ingestion_semaphore: asyncio.Semaphore | None = None


def _get_ingestion_semaphore() -> asyncio.Semaphore | None:
    """Return the process-wide ingestion semaphore, creating it on first use.

    The concurrency limit is read at runtime from ``CAT_INGESTION_MAX_CONCURRENCY``
    and the semaphore is then cached for the process lifetime. ``None`` or
    ``<= 0`` means unlimited: no semaphore is created and callers must NOT
    acquire (previous behavior is preserved).
    """
    global _ingestion_semaphore
    if _ingestion_semaphore is None:
        max_concurrency = get_env_int("CAT_INGESTION_MAX_CONCURRENCY")
        if max_concurrency is None or max_concurrency <= 0:
            return None
        _ingestion_semaphore = asyncio.Semaphore(max_concurrency)
    return _ingestion_semaphore


@asynccontextmanager
async def _ingestion_guard(semaphore: asyncio.Semaphore | None):
    """Async context manager bounding a body to ``semaphore``; no-op when unlimited.

    Using ``async with`` guarantees the semaphore is released on every exit path
    (success, exception, and the enclosing ``finally``). When ``semaphore`` is
    ``None`` the body runs unconstrained (no acquire).
    """
    if semaphore is None:
        yield
    else:
        async with semaphore:
            yield


class RabbitHole:
    def __init__(self):
        self.cat = None
        self.stray = None
        self.embedder = None

    async def setup(self, _cat: "BotMixin"):  # type: ignore[name-defined]
        from cat.looking_glass import CheshireCat, StrayCat

        if isinstance(_cat, CheshireCat):
            self.cat = _cat
            self.stray = None
            return

        if isinstance(_cat, StrayCat):
            self.stray = _cat
            self.cat = await _cat.lizard.get_cheshire_cat(_cat.agent_key)
            return

        raise ValueError("RabbitHole can only be setup with CheshireCat or StrayCat instances.")

    """Manages content ingestion. I'm late... I'm late!"""

    async def ingest_memory(self, cat: "CheshireCat", file: BytesIO, filename: str):  # type: ignore[name-defined]
        """Upload memories to the declarative memory from a JSON file.

        Args:
            cat (CheshireCat): Cheshire Cat instance.
            file (BytesIO): JSON file containing vector and content memories.
            filename (str): Filename of the uploaded file.

        Notes
        -----
        This method allows uploading a JSON file containing vector and content memories directly to the declarative
        memory.
        When doing this, please, make sure the embedder used to export the memories is the same as the one used
        when uploading.
        The method also performs a check on the dimensionality of the embeddings (i.e. length of each vector).
        """
        try:
            await self.setup(cat)
            lizard = self.cat.lizard

            # fire the hook with the source before the memories are stored
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_ingestion_start", filename, {}, False, caller=self.cat,
            )

            # Load fyle byte in a dict
            memories = json.loads(file.read().decode("utf-8"))

            # Check the embedder used for the uploaded memories is the same the Cat is using now
            upload_embedder = memories["embedder"]
            embedder = await lizard.embedder()
            cat_embedder = str(embedder.__class__.__name__)
            if upload_embedder != cat_embedder:
                raise Exception(f"Embedder mismatch for file '{filename}': file embedder {upload_embedder} is different from {cat_embedder}")

            # Get Declarative memories in file
            declarative_memories = memories["collections"][str(VectorMemoryType.DECLARATIVE)]
            if not declarative_memories:
                raise Exception(f"No Declarative memories found in the uploaded file '{filename}'.")

            # Store data to upload the memories in batch
            points = [PointStruct(
                id=m["id"],
                payload={"page_content": m["page_content"], "metadata": m["metadata"]},
                vector=m["vector"],
            ) for m in declarative_memories]

            log.info(f"Agent id: {self.cat.agent_key}. Preparing to load {len(points)} vector memories")

            # Check embedding size is correct
            embedder = await lizard.embedder()
            embedder_size = embedder.size
            len_mismatch = [len(p.vector) == embedder_size for p in points]  # type: ignore[union-attr]

            if not any(len_mismatch):
                raise Exception(f"Embedding size mismatch for file '{filename}': vectors length should be {embedder_size}")

            # Upsert memories in batch mode
            await cat.vector_memory_handler.add_points_to_tenant(
                collection_name=str(VectorMemoryType.DECLARATIVE), points=points,
            )
        except Exception as e:
            log.error(f"Error uploading memories from file '{filename}': {e}")
            # fire the error hook alongside the existing log
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_ingestion_error", filename, str(e), caller=self.cat,
            )

    async def ingest_file(
        self,
        cat: "BotMixin",  # type: ignore[name-defined]
        file: str | BytesIO,
        metadata: Dict,
        filename: str | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ):
        """
        Load a file in the Cat's declarative memory.

        The method splits and converts the file in Langchain `Document`. Then, it stores the `Document` in the Cat's
        memory.

        Args:
            cat (CheshireCat | StrayCat): Cheshire Cat or Stray Cat instance.
            file (str | BytesIO): The file can be a path passed as a string or a `BytesIO` object if the document is ingested using the `rabbithole` endpoint.
            metadata (Dict): Metadata to be stored with each chunk.
            filename (str): The filename of the file to be ingested, if coming from the `/rabbithole/` endpoint.
            store_file (bool): Whether to store the file in the Cat's file storage.
            content_type (str): The content type of the file. If not provided, it will be guessed based on the file extension.

        See Also:
            before_rabbithole_stores_documents
        """
        source = ""
        points = []

        try:
            await self.setup(cat)

            filename = filename or (file if isinstance(file, str) else None)
            if not filename:
                raise ValueError("No filename provided.")

            # fire the hook with the source (the filename; for URLs the filename IS the URL)
            # before the file is downloaded/parsed, so plugins can observe the full lifecycle
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_ingestion_start", filename, metadata, filename.startswith("http"),
                caller=self.stray or self.cat,
            )

            # bound the heavy ingestion body (parse -> embed -> store -> notify)
            # to CAT_INGESTION_MAX_CONCURRENCY process-wide; None/<=0 = unlimited.
            # setup() and the ingestion_start hook above run unconstrained so
            # status hooks are always visible.
            async with _ingestion_guard(_get_ingestion_semaphore()):
                # resolve the source bytes BEFORE parsing so the file can be
                # persisted to the file manager first: a container restart during
                # a long parse no longer loses the uploaded file (the resume
                # mechanism reads it back from disk).
                source, file_bytes, content_type, is_url = await self._resolve_source_bytes(
                    file=file, filename=filename, content_type=content_type
                )
                if not file_bytes:
                    raise ValueError(f"Something went wrong with the source '{source}'")

                # store the file bytes in the agent's persistent storage as early as
                # possible, so a container restart can resume reading the bytes from
                # disk even if processing/embedding is interrupted. LocalFileManager
                # overwrites an existing destination, so this is idempotent.
                if store_file and not is_url:
                    chat_id = self.stray.id if self.stray else None
                    await self.cat.save_file(file_bytes, content_type, source, chat_id)

                # fire the processing hook once the worker has picked the file up
                # (persisted to disk), before the potentially long parse/embed body
                await self.cat.plugin_manager.execute_hook(
                    "rabbithole_ingestion_processing", source, caller=self.stray or self.cat,
                )

                # keep the PROCESSING row fresh while the body runs, so a long
                # parse is not re-claimed as stale by another worker. The
                # heartbeat task itself is owned by the ingestion_status plugin:
                # the core only fires the start hook.
                scope = self.stray.id if self.stray else "agent"
                heartbeat_interval = get_env_int("CAT_INGESTION_HEARTBEAT_SECONDS") or 30
                await self.cat.plugin_manager.execute_hook(
                    "rabbithole_processing_heartbeat_start",
                    source, scope, heartbeat_interval,
                    caller=self.stray or self.cat,
                )

                # split a file into a list of docs
                docs, images = await self._parse_to_docs(source, file_bytes, content_type)

                if not docs:
                    raise Exception(f"No valid chunks found in the file '{filename}'.")

                # store in memory
                sha256 = hashlib.sha256()
                sha256.update(file_bytes)
                points = await self.store_documents(
                    docs=docs, source=source, file_hash=sha256.hexdigest(), metadata=metadata, images=images,
                    source_bytes=file_bytes,
                )

                # notify client
                images_info = f" and {len(images)} images" if images else ""
                await self._send_notification_message(
                    f"Finished reading {source}, I made {len(docs)} thoughts{images_info} on it."
                )

                log.info(f"Agent id: {self.cat.agent_key}. Successfully ingested file: {filename}")
        except Exception as e:
            log.error(f"Error ingesting file {filename}: {e}")
            # Don't raise in background tasks - just log the error
            if self.stray:
                try:
                    await self.stray.notifier.send_error(f"Error processing {filename}: {str(e)}")
                except Exception as notify_error:
                    log.error(f"Failed to send error notification: {notify_error}")
            # fire the error hook alongside the existing log/notify
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_ingestion_error", source or filename, str(e), caller=self.stray or self.cat,
            )
        finally:
            # stop the plugin-owned heartbeat so no orphan task keeps bumping
            # the row (the ingestion_status plugin cancels its task here)
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_processing_heartbeat_stop",
                source or filename,
                self.stray.id if self.stray else "agent",
                caller=self.stray or self.cat,
            )
            # hook the points after they are stored in the vector memory
            await self.cat.plugin_manager.execute_hook(
                "after_rabbithole_stored_documents", source, points, caller=self.stray or self.cat,
            )

    async def _file_to_docs(
        self, file: str | BytesIO, filename: str, content_type: str | None = None
    ) -> Tuple[str, bytes, str | None, List[Document], List[Dict], bool]:
        """
        Load and convert files to Langchain `Document`.

        This method takes a file either from a Python script, from the `/rabbithole/` or `/rabbithole/web` endpoints.
        Hence, it loads it in memory and splits it in chunks.

        Args:
            file (str | BytesIO): The file can be either a string path if loaded programmatically, a `BytesIO` if coming from the `/rabbithole/` endpoint, or a URL if coming from the `/rabbithole/web` endpoint.
            filename (str): The filename of the file to be ingested.
            content_type (str): The content type of the file. If not provided, it will be guessed based on the file extension.

        Returns:
            (source, file_bytes, content_type, docs, images, is_url): Tuple[str, bytes, str | None, List[Document], List[Dict], bool].
                The file name, the file content in bytes, the content type, the list of chunked Langchain `Document`,
                the list of images extracted by multimodal parsers (empty for text-only embedders) and
                a boolean indicating if the file was loaded from a URL.
        """
        source, file_bytes, content_type, is_url = await self._resolve_source_bytes(
            file=file, filename=filename, content_type=content_type
        )
        if not file_bytes:
            raise ValueError(f"Something went wrong with the source '{source}'")
        docs, images = await self._parse_to_docs(source, file_bytes, content_type)
        return source, file_bytes, content_type, docs, images, is_url  # type: ignore[return-value]

    async def _resolve_source_bytes(
        self, file: str | BytesIO, filename: str, content_type: str | None
    ) -> Tuple[str | None, bytes | None, str | None, bool]:
        """Resolve the source bytes of an incoming file WITHOUT parsing it.

        This method takes a file either from a Python script, from the `/rabbithole/` or `/rabbithole/web` endpoints,
        and resolves its source name, raw bytes and content type. It does NOT parse or chunk the content: that is
        the responsibility of ``_parse_to_docs``. Resolving the bytes first allows ``ingest_file`` to persist the
        file to the file manager before the (potentially long) parse step, so a container restart during ingestion
        does not lose the uploaded file.

        Args:
            file (str | BytesIO): The file can be either a string path if loaded programmatically, a `BytesIO` if coming from the `/rabbithole/` endpoint, or a URL if coming from the `/rabbithole/web` endpoint.
            filename (str): The filename of the file to be ingested.
            content_type (str): The content type of the file. If not provided, it will be guessed based on the file extension.

        Returns:
            (source, file_bytes, content_type, is_url): Tuple[str | None, bytes | None, str | None, bool].
                The file name, the file content in bytes, the content type and a boolean indicating if the file
                was loaded from a URL. On a failed URL download both ``source`` and ``file_bytes`` are ``None``.
        """
        if not isinstance(file, BytesIO) and not isinstance(file, str):
            raise ValueError(f"{type(file)} is not a valid type.")

        def sanitize_filename(file_name: str) -> str:
            if "." not in file_name:
                return file_name
            # Split on the LAST dot only (if any)
            base, ext = file_name.rsplit(".", 1)
            ext = "." + ext
            # Replace any sequence of dots or spaces in the base name only
            base = re.sub(r"[.\s]+", "_", base)
            return base + ext

        if isinstance(file, BytesIO):
            # Get the source of UploadFile, file bytes and whether it's a URL
            return sanitize_filename(filename), file.read(), content_type, False
        if fnc_is_url(file):
            try:
                # notify plugins that the URL download is about to start
                await self.cat.plugin_manager.execute_hook(
                    "rabbithole_url_downloading", file, filename, caller=self.stray or self.cat,
                )
                # Make a request with a fake browser name - use async httpx
                async with AsyncClient() as client:
                    response = await client.get(file, headers={"User-Agent": "Magic Browser"})
                    response.raise_for_status()
                    # Define mime type and source of url
                    # Add fallback for empty/None content_type
                    ct = response.headers.get(
                        "Content-Type", "text/html" if file.startswith("http") else "text/plain"
                    ).split(";")[0]
                    # Get binary content of url
                    content = response.content
                # notify plugins that the URL download completed successfully
                await self.cat.plugin_manager.execute_hook(
                    "rabbithole_url_download_completed", file, filename, caller=self.stray or self.cat,
                )
                return file, content, ct, True
            except Exception as e:
                log.error(f"Agent id: {self.cat.agent_key}. Error: {e}")
                return None, None, content_type, True
        # Get file bytes - use async file reading
        fb = await asyncio.to_thread(lambda: open(file, "rb").read())  # type: ignore[union-attr]
        return sanitize_filename(os.path.basename(file)), fb, mimetypes.guess_type(file)[0], False  # type: ignore[return-value]

    async def _run_in_ingestion_executor(self, func, *args):
        """Run a heavy ingestion callable via the dedicated ingestion lane.

        Dispatches through the ``run_in_ingestion_executor`` hook: when a plugin
        (efficient_ingestion) provides a dedicated low-concurrency pool the
        callable runs there; otherwise falls back to the default executor
        (upstream parity via ``asyncio.to_thread``).
        """
        task = func if not args else (lambda: func(*args))
        result = await self.cat.plugin_manager.execute_hook(
            "run_in_ingestion_executor", None, task, caller=self.cat,
        )
        if result is None:
            result = await asyncio.to_thread(task)
        return result

    async def _parse_to_docs(
        self, source: str, file_bytes: bytes, content_type: str | None
    ) -> Tuple[List[Document], List[Dict]]:
        """Parse resolved source bytes into Langchain `Document` chunks.

        This method takes the source name and raw bytes resolved by ``_resolve_source_bytes``, parses the content
        with the MIME-type based parser, propagates the source to every parsed document, collects the images
        extracted by multimodal parsers (if a multimodal embedder is active) and splits the documents in chunks.

        Args:
            source (str): The source name of the file to be ingested.
            file_bytes (bytes): The raw bytes of the file to be ingested.
            content_type (str | None): The content type of the file.

        Returns:
            Tuple[List[Document], List[Dict]]: The list of chunked Langchain `Document` and the list of images
                extracted by multimodal parsers (empty for text-only embedders).
        """
        fh = await self.cat.file_handlers()
        log.debug(f"Attempting to parse source: {source}. Detected MIME type: {content_type}. Available handlers: {list(fh.keys())}")

        # Load the bytes in the Blob schema and parse the content. Parser based on the mime type.
        # The parser is CPU/IO-bound (e.g. PyMuPDF on a large PDF): run it off the event loop.
        await self._send_notification_message("I'm parsing the content. Big content could require some minutes...")
        super_docs = await self._run_in_ingestion_executor(
            lambda: MimeTypeBasedParser(handlers=fh).parse(
                Blob(data=file_bytes, mimetype=content_type).from_data(data=file_bytes, mime_type=content_type, path=source)
            )
        )

        # Propagate the source to every parsed document BEFORE chunking, so that
        # hooks such as `before_rabbithole_splits_documents` can rely on it
        # (metadata['source'] is otherwise only added later, in store_documents).
        # setdefault preserves a more specific source set by the parser itself.
        for doc in super_docs:
            if isinstance(doc.metadata, dict):
                doc.metadata.setdefault("source", source)

        # Collect the images extracted by multimodal parsers (if a multimodal embedder is
        # active) BEFORE chunking: the chunkers may drop or alter the metadata that carries
        # the image payload (e.g. PLUS SemanticChunker discards metadata, _merge_short_chunks
        # keeps only the first chunk's metadata). The multimodal_ingestion plugin owns the
        # extraction AND strips the transient image_base64 from the docs; with no plugin the
        # no-op returns [] (upstream parity: no image handling).
        images = await self.cat.plugin_manager.execute_hook(
            "rabbithole_collects_document_images", [], super_docs, caller=self.stray or self.cat,
        ) or []

        # Split
        await self._send_notification_message("Parsing completed. Now let's go with reading process...")
        docs = await self._split_text(docs=super_docs)
        return docs, images

    async def store_documents(
        self,
        docs: List[Document],
        source: str,
        file_hash: str | None = None,
        metadata: Dict | None = None,
        images: List[Dict] | None = None,
        source_bytes: bytes | None = None,
    ) -> List[PointStruct]:
        """Add documents to the Cat's declarative memory.

        This method loops a list of Langchain `Document` and adds some metadata. Namely, the source filename and the
        timestamp of insertion. Once done, the method notifies the client via Websocket connection.

        If a multimodal embedder is active and the multimodal parsers extracted some images, this method also embeds
        the images via ``embed_images``, saves them as files in the agent/chat storage via ``save_file`` and stores
        them in the same collection, keeping the file name in the point metadata (``image_file``, no base64 payload).

        Args:
            docs (List[Document]): List of Langchain `Document` to be inserted in the Cat's declarative memory.
            source (str): Source name to be added as a metadata. It can be a file name or an URL.
            file_hash (str | None): Optional hash of the source to be added as a metadata.
            metadata (Dict | None): Optional metadata to be stored with each chunk.
            images (List[Dict] | None): Optional images extracted by multimodal parsers. Each entry has
                ``image_base64``, ``image_bytes`` and ``image_mime_type`` keys. The images are embedded and saved
                as files via ``save_file``; the point metadata only keeps the file name in ``image_file``.
            source_bytes (bytes | None): Optional raw bytes of the ingested source file. Used when the source
                itself is an image: the file is embedded as a single whole-image point instead of the parser
                sub-crops, and no derived file is created.

        Returns:
            stored_points (List[PointStruct]): List of points stored in the Cat's declarative memory
                (text chunks and, if any, image points).

        See Also:
            before_rabbithole_stores_documents
            after_rabbithole_stored_documents

        Notes
        -------
        At this point, it is possible to customize the Cat's behavior using the `before_rabbithole_stores_documents`
        hook to edit the memories before they are inserted in the vector database.
        The hook `after_rabbithole_stored_documents` could be used to track the end of the process, indeed.
        """
        log.info(f"Agent id: {self.cat.agent_key}. Preparing to memorize {len(docs)} vectors for {source}.")

        embedder = await self.cat.lizard.embedder()
        plugin_manager = self.cat.plugin_manager

        # add custom metadata (sent via endpoint) and default metadata (source and when and eventual chat_id)
        for doc in docs:
            # Drop the transient parser image payload, if any: images are embedded and
            # saved as files separately, so their content must never reach the vector
            # DB metadata (and from there the LLM context on recall). This also covers
            # direct callers that pass documents still carrying image_base64.
            doc.metadata.pop("image_base64", None)
            doc.metadata = (
                    doc.metadata
                    | metadata
                    | {"source": source, "when": time.time(), "hash": file_hash}
                    | ({"chat_id": self.stray.id} if self.stray else {})
            )

# hook the docs before they are stored in the vector memory
        docs = await plugin_manager.execute_hook("before_rabbithole_stores_documents", docs, caller=self.stray or self.cat)

        # The token-budget split (formerly _split_oversized) is owned by the
        # efficient_ingestion plugin via the before/finalize hooks: the docs
        # may be re-sized so the embed below is 1:1 with the stored points.
        # The default no-op preserves upstream behavior (no sizing).

        # hook the points before they are stored in the vector memory
        valid_documents = list(filter(lambda doc_: doc_.page_content.strip(), docs))
        storing_vectors = await self._run_in_ingestion_executor(
            lambda: embedder.embed_documents([doc_.page_content for doc_ in valid_documents])
        )
        points = [PointStruct(
            id=uuid.uuid4().hex,
            payload=doc.model_dump(),
            vector=vector,
        ) for doc, vector in zip(valid_documents, storing_vectors)]

        # If the multimodal_ingestion plugin collected images for this source, let it
        # build the image points (embed_images + save_file + PointStruct metadata): the
        # core only appends them to the same collection. With no plugin the no-op returns
        # [] (upstream parity: no image points).
        if images:
            chat_id = self.stray.id if self.stray else None
            image_points = await plugin_manager.execute_hook(
                "rabbithole_stores_image_points",
                [], images, source, source_bytes, metadata, file_hash, chat_id,
                caller=self.cat,
            )
            points.extend(image_points or [])

        collection_name = str(VectorMemoryType.DECLARATIVE if not self.stray else VectorMemoryType.EPISODIC)
        await self.cat.vector_memory_handler.add_points_to_tenant(collection_name=collection_name, points=points)

        return points

    async def _split_text(self, docs: List[Document]):
        """Split LangChain documents in chunks.

        This method splits the incoming documents in chunks. Other two hooks are available to edit the
        documents before and after the split step.

        Args:
            docs (List[Document]): Content of the loaded file.

        Returns:
            docs (List[Document]): List of split Langchain `Document`.

        See Also:
            before_rabbithole_splits_documents

        Notes
        -----
        The default behavior splits the content and executes the hooks, before the splitting.
        `before_rabbithole_splits_documents` hook returns the original input without any modification.
        """
        plugin_manager = self.cat.plugin_manager

        # do something on the docs before they are split
        docs = await plugin_manager.execute_hook("before_rabbithole_splits_documents", docs, caller=self.stray or self.cat)

        # split docs
        docs = await self.cat.chunker.split_documents(docs)

        # join each short chunk with previous one, instead of deleting them
        try:
            docs = self._merge_short_chunks(docs, self.cat.chunker)
        except Exception as e:
            # Log error but don't fail the entire process
            log.warning(f"Error merging short chunks: {e}. Proceeding with original chunks.")

        # Finalize the document list so no chunk exceeds the active embedder's
        # max_input_tokens. This is a build-phase step, fully separate from the
        # embedding loop in store_documents: it produces the correct final list
        # (splitting oversized chunks into in-place sub-chunks) so the later
        # 1:1 embed/store pairing is never broken by re-chunking mid-loop. The
        # actual split is owned by a plugin (efficient_ingestion) through the
        # finalize_oversized_chunks hook; the default no-op preserves the
        # upstream behavior (no token-budget split).
        try:
            docs = await self.cat.plugin_manager.execute_hook(
                "finalize_oversized_chunks", docs, caller=self.cat,
            )
        except Exception as e:
            # Log error but don't fail the entire process
            log.warning(f"Failed to finalize oversized chunks: {e}. Proceeding with original chunks.")

        return docs

    def _merge_short_chunks(self, docs: List[Document], chunker: BaseChunker) -> List[Document]:
        """Safely merge short chunks with adjacent ones.

        Args:
            docs: List of documents to process
            chunker: The chunker instance for configuration

        Returns:
            List of documents with short chunks merged
        """
        def should_merge_chunk() -> bool:
            """Determine if a chunk should be merged."""
            return (
                    min_chunk_size > len(current_content) > 0 and  # Don't merge empty content
                    len(merged_docs) > 0  # Need previous chunk to merge with
            )

        def can_safely_merge(prev_doc: Document) -> bool:
            """Check if two documents can be safely merged."""
            potential_size = len(prev_doc.page_content) + len(current_doc.page_content) + 2
            return potential_size <= max_merge_size

        if not docs:
            return docs

        # Get configuration with safe defaults
        chunk_size = getattr(chunker.splitter, "chunk_size", getattr(chunker.splitter, "max_chunk_size", 1000))
        chunk_overlap = getattr(chunker.splitter, "chunk_overlap", 100)

        # Conservative thresholds
        min_chunk_size = max(50, chunk_size // 20)  # At least 50 chars
        max_merge_size = chunk_size + chunk_overlap  # Respect splitter's intended size

        merged_docs: list = []  # type: ignore[var-annotated]
        i = 0

        while i < len(docs):
            current_doc = docs[i]
            current_content = current_doc.page_content.strip()

            # Check if this chunk should be merged
            if should_merge_chunk() and can_safely_merge(merged_docs[-1]):
                try:
                    merged_docs[-1] = self._create_merged_document(merged_docs[-1], current_doc)
                except Exception:
                    # If merge fails, keep both documents separate
                    merged_docs.append(current_doc)
            else:
                merged_docs.append(current_doc)

            i += 1

        return merged_docs

    def _create_merged_document(self, prev_doc: Document, current_doc: Document) -> Document:
        """Create a new merged document safely."""
        # Merge content with clear separator
        merged_content = prev_doc.page_content.rstrip() + "\n\n" + current_doc.page_content.lstrip()

        # Merge metadata - since source is the same, we can safely combine
        merged_metadata = prev_doc.metadata.copy()

        # Add all metadata from current doc, handling conflicts intelligently
        for key, value in current_doc.metadata.items():
            if key in merged_metadata and merged_metadata[key] != value:
                # For numeric values (like page numbers), take the range or sum
                if isinstance(merged_metadata[key], (int, float)) and isinstance(value, (int, float)):
                    if key in ["page", "page_number", "chunk_index"]:
                        # For page/chunk numbers, keep the starting one
                        pass  # Keep the previous value
                    else:
                        # For other numeric values, might want to sum or take max
                        merged_metadata[key] = max(merged_metadata[key], value)
                else:
                    # For other conflicts, keep the first value
                    pass
            else:
                merged_metadata[key] = value

        # Add merge tracking
        merge_count = merged_metadata.get("_merge_count", 1) + 1
        merged_metadata["_merge_count"] = merge_count
        merged_metadata["_is_merged"] = True

        return Document(page_content=merged_content, metadata=merged_metadata)

    async def _send_notification_message(self, message: str):
        if self.stray and self.stray.notifier.has_ws_connection():
            await self.stray.notifier.send_notification(message)
