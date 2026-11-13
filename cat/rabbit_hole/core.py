import copy
import asyncio
import functools
import hashlib
import json
import mimetypes
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Tuple

from httpx import AsyncClient
from langchain_core.documents.base import Blob, Document

from cat.db.cruds import ingestions as crud_ingestions
from cat.db.cruds.ingestions import IngestionStatus
from cat.rabbit_hole import chunks, engine
from cat.rabbit_hole.phases import PHASE_EMBEDDING, IngestionSource, PhaseGraphError, PhaseResult
from cat.rabbit_hole.split import embedder_token_budget, split_oversized
from cat.services.factory.embedder import embedding_size
from cat.core_plugins.base_plugin.parsers import MimeTypeBasedParser
from cat.env import get_env_bool, get_env_int
from cat.log import log
from cat.services.factory.chunker import BaseChunker
from cat.services.memory.models import PointStruct, VectorMemoryType
from cat.utils import guess_file_type, is_url as fnc_is_url

# Process-wide semaphore bounding the parsing and embedding phases running at the same time.
# Created lazily on first use; the size is read at runtime from
# ``CAT_INGESTION_MAX_CONCURRENCY`` (``None`` or ``<= 0`` means unlimited).
_ingestion_semaphore: asyncio.Semaphore | None = None

# Process-wide pool of de-prioritized threads running the heavy ingestion callables (parse, embed).
# Created lazily on first use; the size is read at runtime from ``CAT_INGESTION_WORKERS`` and the
# niceness of its threads from ``CAT_INGESTION_NICENESS``.
_pool: ThreadPoolExecutor | None = None


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


def _set_worker_niceness(niceness: int) -> None:
    """Lower the priority of the current worker thread.

    Sets an absolute nicer value for the *current thread* on Linux
    (``PRIO_PROCESS`` with ``who=0`` is implemented per-thread in the kernel
    ``task_struct``). Raising niceness (de-prioritizing) is always permitted;
    only lowering (negative values) would require privileges.
    """
    if niceness > 0:
        os.setpriority(os.PRIO_PROCESS, 0, niceness)


def _get_ingestion_pool() -> ThreadPoolExecutor | None:
    """Return the process-wide ingestion pool, creating it on first use.

    The worker count is read at runtime from ``CAT_INGESTION_WORKERS`` (default 2) and the
    pool is then cached for the process lifetime. ``<= 0`` means the lane is disabled: no
    pool is created and callers fall back to the default executor.
    """
    global _pool
    if _pool is None:
        max_workers = get_env_int("CAT_INGESTION_WORKERS")
        if max_workers is None or max_workers <= 0:
            return None
        niceness = get_env_int("CAT_INGESTION_NICENESS") or 0
        # ``initializer`` runs in each worker thread; ``_set_worker_niceness``
        # is a no-op when ``niceness <= 0`` (de-prioritization disabled).
        _pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cat-ingestion",
            initializer=_set_worker_niceness,
            initargs=(niceness,),
        )
    return _pool


async def run_in_ingestion_executor(func, *args, **kwargs):
    """Run ``func(*args, **kwargs)`` on the dedicated ingestion pool, off the event loop.

    Used by the RabbitHole for the heavy ingestion steps (parse, embed) and available to the plugins doing heavy
    ingestion work of their own (e.g. embedding the images of a source). If the ingestion lane is disabled
    (``CAT_INGESTION_WORKERS <= 0``), the call is dispatched to the default executor.
    """
    return await asyncio.get_running_loop().run_in_executor(
        _get_ingestion_pool(), functools.partial(func, *args, **kwargs)
    )


def _strip_image_payload(docs) -> None:
    """Remove every ``image_base64`` key from each document metadata, at any depth."""
    def walk(value) -> None:
        if isinstance(value, dict):
            value.pop("image_base64", None)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for doc in docs:
        if isinstance(getattr(doc, "metadata", None), dict):
            walk(doc.metadata)


#: Seconds an HTTP deletion waits for an ingestion of the same source to end, before answering 409 (retry later)
DELETE_WAIT_SECONDS = 5


#: The metadata of a chunk set by the phases themselves (not the custom metadata of the upload)
_PHASE_METADATA = ("source", "when", "hash", "chat_id")
_MISSING = object()


class SourceBusy(Exception):
    """The source is being processed (e.g. ingested) by a worker: retry later."""


class SourceNotKept(Exception):
    """The bytes of the source were not kept (a file ingested without storing it): it cannot be processed again."""


def _sanitize_filename(file_name: str) -> str:
    """The name of an uploaded file as a source: dots and spaces of the base name become underscores."""
    if "." not in file_name:
        return file_name
    # Split on the LAST dot only (if any)
    base, ext = file_name.rsplit(".", 1)
    # Replace any sequence of dots or spaces in the base name only
    return re.sub(r"[.\s]+", "_", base) + "." + ext


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
        """Upload memories to the declarative memory from a JSON file (see ``_ingest_memory``).

        The rabbit hole of the instance serves every agent at the same time: every ingestion runs on a copy of its own,
        which keeps the cat (and the chat) of that ingestion only. A memory file is not a source: it has no ingestion
        state.
        """
        return await copy.copy(self)._ingest_memory(cat, file, filename)

    def _resolve_caller(self, on_cat: bool):
        """The caller of the plugin hooks: the agent (``on_cat``), or the chat of the ingestion when there is one."""
        if on_cat:
            return self.cat
        return self.stray or self.cat

    def _collection(self, chat_id: str | None) -> str:
        return str(VectorMemoryType.DECLARATIVE if chat_id is None else VectorMemoryType.EPISODIC)

    @staticmethod
    def _source_filter(source: IngestionSource) -> Dict:
        return {"source": source.name} | ({"chat_id": source.chat_id} if source.chat_id is not None else {})

    async def _ingestion_url_downloading(self, url: str):
        """The download of ``url`` is about to start: the state of the upload says so (a download inside a phase, e.g.
        a resume, leaves the state to the phase)."""
        if (source := getattr(self, "_uploading", None)) is not None:
            await crud_ingestions.update_status(source.agent_id, source.chat_id, url, status=IngestionStatus.DOWNLOADING)

    async def _ingestion_url_downloaded(self, url: str):
        """The download of ``url`` completed successfully (see ``_ingestion_url_downloading``)."""
        if (source := getattr(self, "_uploading", None)) is not None:
            await crud_ingestions.update_status(source.agent_id, source.chat_id, url, status=IngestionStatus.DOWNLOADED)

    async def _ingest_memory(self, cat: "CheshireCat", file: BytesIO, filename: str):  # type: ignore[name-defined]
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
            embedder_size = await embedding_size(embedder)
            len_mismatch = [len(p.vector) == embedder_size for p in points]  # type: ignore[union-attr]

            if not any(len_mismatch):
                raise Exception(f"Embedding size mismatch for file '{filename}': vectors length should be {embedder_size}")

            # Upsert memories in batch mode
            await cat.vector_memory_handler.add_points_to_tenant(
                collection_name=str(VectorMemoryType.DECLARATIVE), points=points,
            )
        except Exception as e:
            log.error(f"Error uploading memories from file '{filename}': {e}")

    async def ingest_file(
        self,
        cat: "BotMixin",  # type: ignore[name-defined]
        file: str | BytesIO,
        metadata: Dict,
        filename: str | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ) -> Dict | None:
        """Load a file in the Cat's memory (see ``_ingest_file``).

        The rabbit hole of the instance serves every agent at the same time: every ingestion runs on a copy of its own,
        which keeps the cat (and the chat) of that ingestion only.
        """
        return await copy.copy(self)._ingest_file(cat, file, metadata, filename, store_file, content_type)

    async def _ingest_file(
        self,
        cat: "BotMixin",  # type: ignore[name-defined]
        file: str | BytesIO,
        metadata: Dict,
        filename: str | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ) -> Dict | None:
        """
        Load a file in the Cat's memory: in the declarative memory, or in the episodic memory of the chat when ``cat`` is
        a StrayCat.

        The source gets a row in the registry of the ingestion states (``cat.db.cruds.ingestions``) and goes through
        its phases (``cat.rabbit_hole.phases``), holding the lock of the source: ``uploaded`` (``downloading`` and
        ``downloaded`` for a URL), then ``processing``, then ``completed`` or ``error``. A new upload of the same source
        starts its lifecycle again. The call returns when the source is done; errors are not raised (the ingestion runs
        in background tasks): they are logged, notified to the chat and written in the row.

        Args:
            cat (CheshireCat | StrayCat): Cheshire Cat or Stray Cat instance.
            file (str | BytesIO): The file can be a path passed as a string or a `BytesIO` object if the document is ingested using the `rabbithole` endpoint.
            metadata (Dict): Metadata to be stored with each chunk.
            filename (str): The filename of the file to be ingested, if coming from the `/rabbithole/` endpoint.
            store_file (bool): Whether to store the file in the Cat's file storage.
            content_type (str): The content type of the file. If not provided, it will be guessed based on the file extension.

        Returns:
            The final row of the source (None if it could not even be named).

        See Also:
            before_rabbithole_stores_documents
        """
        row = None
        source = None
        try:
            await self.setup(cat)

            filename = filename or (file if isinstance(file, str) else None)
            if not filename:
                raise ValueError("No filename provided.")
            if not isinstance(file, (str, BytesIO)):
                raise ValueError(f"{type(file)} is not a valid type.")
            is_url = isinstance(file, str) and fnc_is_url(file)
            name = file if is_url else _sanitize_filename(filename if isinstance(file, BytesIO) else os.path.basename(file))
            source = IngestionSource(
                self.cat.agent_key, self.stray.id if self.stray else None, name, "url" if is_url else "file",
            )

            async with engine.source_lock(source):
                # a new upload starts the lifecycle of the source again (empty diary: every phase runs)
                await crud_ingestions.create_status(
                    source.agent_id, source.chat_id, name, type_=source.type, metadata=metadata or {},
                    content_type=content_type,
                )
                try:
                    self._uploading = source
                    _, file_bytes, content_type, _ = await self._resolve_source_bytes(
                        file=file, filename=filename, content_type=content_type
                    )
                    self._uploading = None
                    await self._check_still_wanted(source)  # e.g. the chat was deleted during the download
                    if not file_bytes:
                        raise ValueError(f"Something went wrong with the source '{name}'")
                    await self._keep_source_bytes(source, file_bytes, content_type, store_file)
                    if not await self._is_still_wanted(source):
                        await self._drop_kept_bytes(source, store_file)
                        raise engine.IngestionCanceled(name)
                    self._pending = (name, file_bytes)  # the first parse reads the bytes from memory
                    row = await engine.run_phases(self, self.cat, source)
                except (engine.IngestionCanceled, engine.LockLost):
                    raise
                except Exception as e:
                    await crud_ingestions.update_status(
                        source.agent_id, source.chat_id, name,
                        status=IngestionStatus.ERROR, phase=None, owner=None, error=str(e),
                    )
                    raise
        except engine.IngestionCanceled:
            log.warning(f"Ingestion of {filename} stopped: the source was deleted meanwhile")
            return None
        except engine.LockLost:
            # another worker may own the source now: it notifies the outcome
            log.warning(f"Ingestion of {filename} stopped: the lock of the source was lost")
            return await crud_ingestions.get_status(source.agent_id, source.chat_id, source.name)
        except Exception as e:
            log.error(f"Error ingesting file {filename}: {e}")
            await self._notify_error(filename, str(e))
            await self._stored_points_hook(source.name if source else filename, [])
            return await crud_ingestions.get_status(source.agent_id, source.chat_id, source.name) if source else None
        finally:
            self._pending = None
            self._pending_chunks = None
            self._uploading = None

        if row and row["status"] == IngestionStatus.COMPLETED.value:
            embedding = row["completed_phases"].get(PHASE_EMBEDDING) or {}
            await self._send_notification_message(
                f"Finished reading {source.name}, I made {embedding.get('processed') or 0} thoughts on it."
            )
            log.info(f"Agent id: {self.cat.agent_key}. Successfully ingested file: {filename}")
        elif row:
            await self._notify_error(filename, row.get("error") or "ingestion failed")
            if (row["completed_phases"].get(PHASE_EMBEDDING) or {}).get("state") != engine.STATE_COMPLETE:
                # the embedding notifies the stored points; a failure before it notifies none, once
                await self._stored_points_hook(source.name, [])
        return row

    async def delete_source(
        self,
        cat: "CheshireCat",  # type: ignore[name-defined]
        name: str,
        chat_id: str | None = None,
        wait: float | None = None,
        before: Callable[[], Awaitable[Any]] | None = None,
    ) -> Any:
        """Remove a source from the memory of the agent (``chat_id=None``: its knowledge base) or of a chat: what every
        ingestion phase produced for it (its points among them), the files the RabbitHole kept for it and its state.

        It waits for a running ingestion of the source to end: as long as needed, or ``wait`` seconds, then raises
        ``SourceBusy`` without touching anything. ``before`` runs holding the lock, before the phases are removed
        (e.g. a route removing the uploaded file, which is not removed here); its result is returned."""
        rabbit_hole = copy.copy(self)
        await rabbit_hole.setup(cat)
        source = IngestionSource(rabbit_hole.cat.agent_key, chat_id, name, "url" if fnc_is_url(name) else "file")
        try:
            async with engine.source_lock(source, wait=wait):
                result = await before() if before is not None else None
                await engine.delete_source(rabbit_hole, rabbit_hole.cat, source)
                return result
        except engine.LockError as e:
            raise SourceBusy(name) from e

    async def rerun_source(
        self,
        cat: "CheshireCat",  # type: ignore[name-defined]
        name: str,
        chat_id: str | None = None,
        blocking: bool = True,
        invalidate: Iterable[str] | None = None,
    ) -> Dict | None:
        """Bring a source of the agent (``chat_id=None``: its knowledge base) or of a chat up to date: run its stale
        phases (an ingestion interrupted by a crash, a change of the embedder, of the chunker or of the settings of a
        phase), or complete an interrupted deletion. Nothing is written when nothing changed. Public API: the automatic
        resume, the change of the embedder and the maintenance scripts use it.

        ``invalidate``: phases to run again even if their diary says they are up to date (e.g. the embedding, once the
        collections were recreated); their dependents run again too (their entries are reset with them: a worker dying
        in between leaves them to do). A diary that names a phase no longer installed (a plugin uninstalled) is pruned
        and every phase of the source runs again, so the source is re-ingested from scratch. It waits for a worker
        holding the source, unless ``blocking=False``. Returns the row of the source, or None when the source is gone
        (deleted, canceled) or held by a live worker (``blocking=False``)."""
        rabbit_hole = copy.copy(self)
        await rabbit_hole.setup(cat)
        ccat = rabbit_hole.cat
        if chat_id is not None:
            rabbit_hole.stray = await ccat._find_stray_cat(chat_id)  # the owner of the chat
        source = IngestionSource(ccat.agent_key, chat_id, name, "url" if fnc_is_url(name) else "file")
        try:
            async with engine.source_lock(source, blocking=blocking):
                row = await crud_ingestions.get_status(source.agent_id, chat_id, name)
                if row is not None and row["status"] == IngestionStatus.DELETING.value:
                    await engine.delete_source(rabbit_hole, ccat, source)
                    return None
                if row is not None:
                    removed = await self._removed_phases(rabbit_hole, ccat, row)
                    if removed:
                        # their classes are gone, so their delete cannot run and their points linger: the source is
                        # re-ingested from scratch (its new points replace what the phases still installed produce)
                        log.warning(
                            f"Agent id: {source.agent_id}. The source {name} was ingested with phases no longer "
                            f"installed: {', '.join(removed)}. It is re-ingested from scratch."
                        )
                        invalidate = sorted(await self._current_phase_ids(rabbit_hole, ccat))
                    if invalidate or removed:
                        reset_ids = (
                            await self._with_dependents(rabbit_hole, ccat, invalidate) if invalidate else []
                        )
                        await crud_ingestions.reset_phase_entries(
                            source.agent_id, chat_id, name, sorted({*reset_ids, *removed}),
                        )
                return await engine.run_phases(rabbit_hole, ccat, source)
        except (engine.LockError, engine.IngestionCanceled, engine.LockLost):
            return None

    @staticmethod
    async def _current_phase_ids(rabbit_hole, ccat) -> List[str]:
        """The ids of the phases the agent has now; none when the graph of phases is invalid (``run_phases`` records
        the error)."""
        try:
            return [phase.id for phase in await engine.collect_phases(rabbit_hole, ccat)]
        except PhaseGraphError:
            return []

    async def _removed_phases(self, rabbit_hole, ccat, row: Dict) -> List[str]:
        """The diary ids of the row that are no longer in the set of phases (a plugin installed or uninstalled)."""
        current = set(await self._current_phase_ids(rabbit_hole, ccat))
        return sorted(set(row.get("completed_phases") or {}) - current)

    async def invalidate_source(
        self,
        cat: "CheshireCat",  # type: ignore[name-defined]
        name: str,
        chat_id: str | None = None,
        phase_ids: Iterable[str] = (PHASE_EMBEDDING,),
    ) -> bool:
        """Leave phases of a source (and every phase depending on them) to run again, without running them: their diary
        entries are removed, so the automatic resume runs them (also on a source in error). It waits for a worker
        holding the source (its result would be taken for up to date). Returns False when the source has no row. Public
        API, e.g. for the end of a failed change of the vector database."""
        rabbit_hole = copy.copy(self)
        await rabbit_hole.setup(cat)
        ccat = rabbit_hole.cat
        source = IngestionSource(ccat.agent_key, chat_id, name, "url" if fnc_is_url(name) else "file")
        async with engine.source_lock(source):
            return await crud_ingestions.reset_phase_entries(
                source.agent_id, chat_id, name, await self._with_dependents(rabbit_hole, ccat, phase_ids),
            )

    @staticmethod
    async def _with_dependents(rabbit_hole, ccat, phase_ids: Iterable[str]) -> List[str]:
        """The phases and every phase depending on them, directly or not. The dependents of a phase that ran run again
        because the engine remembers it in memory: a worker dying before them would leave entries the resume takes for
        up to date. An invalid graph of phases adds none (``run_phases`` records the error)."""
        wanted = list(phase_ids)
        try:
            phases = await engine.collect_phases(rabbit_hole, ccat)  # in execution order: a dependency comes first
        except PhaseGraphError:
            return wanted
        for phase in phases:
            if phase.id not in wanted and any(dep in wanted for dep in phase.depends_on):
                wanted.append(phase.id)
        return wanted

    async def reingest_source(
        self, cat: "CheshireCat", name: str, chat_id: str | None = None,  # type: ignore[name-defined]
    ) -> Dict | None:
        """Process a source again from scratch: every phase runs again (each one replaces what it produced before), from
        the bytes kept for it (a URL may be downloaded again). A source ingested before the states existed gets its
        state. Public API, for the maintenance scripts.

        It waits for a worker holding the source. ``SourceNotKept`` when the bytes of the file were not kept: nothing is
        touched. Returns the row of the source, or None when it is gone (or was being deleted: the deletion completes)."""
        rabbit_hole = copy.copy(self)
        await rabbit_hole.setup(cat)
        ccat = rabbit_hole.cat
        if chat_id is not None:
            rabbit_hole.stray = await ccat._find_stray_cat(chat_id)  # the owner of the chat
        source = IngestionSource(ccat.agent_key, chat_id, name, "url" if fnc_is_url(name) else "file")
        try:
            async with engine.source_lock(source):
                kept = await asyncio.to_thread(chunks.read_source_bytes, ccat.file_manager, ccat.agent_key, chat_id, name)
                if kept is None and source.type == "file":
                    raise SourceNotKept(name)
                row = await crud_ingestions.get_status(ccat.agent_key, chat_id, name)
                if row is not None and row["status"] == IngestionStatus.DELETING.value:
                    await engine.delete_source(rabbit_hole, ccat, source)
                    return None
                if row is None:
                    await rabbit_hole._adopt(source, require_points=False)
                else:
                    phases = await engine.collect_phases(rabbit_hole, ccat)
                    recorded = set(row.get("completed_phases") or {}) | {phase.id for phase in phases}
                    await crud_ingestions.reset_phase_entries(ccat.agent_key, chat_id, name, sorted(recorded))
                return await engine.run_phases(rabbit_hole, ccat, source, reindexing=True)
        except (engine.IngestionCanceled, engine.LockLost):
            return None

    async def adopt_source(self, cat: "CheshireCat", name: str, chat_id: str | None = None) -> bool:  # type: ignore[name-defined]
        """Give a state to a source ingested before the states existed (its points, and its file for a file, but no
        row), holding its lock: from then on it follows the lifecycle of any source (a deletion removes it, the resume
        and the reruns process it). Nothing is done for a source with a state, or deleted meanwhile. The change of the
        embedder uses it, before the collections are recreated. Returns whether the source was adopted."""
        rabbit_hole = copy.copy(self)
        await rabbit_hole.setup(cat)
        source = IngestionSource(rabbit_hole.cat.agent_key, chat_id, name, "url" if fnc_is_url(name) else "file")
        async with engine.source_lock(source):
            return await rabbit_hole._adopt(source, require_points=True)

    async def _adopt(self, source: IngestionSource, require_points: bool) -> bool:
        """The lock of the source held: create the row (``uploaded``) of a source without one. Its metadata are the custom
        metadata of its upload, set the same on every chunk: the ones its points share (not the ones of a chunk, e.g. a
        page, nor the ones the phases set again). Its content type comes from its kept bytes, or from its name when the
        bytes only say text (CSV, JSON, Markdown, code). Not adopted: a source with a row, a file not kept, a source
        without points when ``require_points`` (deleted meanwhile)."""
        ccat = self.cat
        if await crud_ingestions.get_status(source.agent_id, source.chat_id, source.name) is not None:
            return False
        kept = await asyncio.to_thread(
            chunks.read_source_bytes, ccat.file_manager, source.agent_id, source.chat_id, source.name,
        )
        if kept is None and source.type == "file":
            return False
        points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
            self._collection(source.chat_id), metadata=self._source_filter(source), with_vectors=False,
        )
        if not points and require_points:
            return False
        shared: Dict = {}
        for index, point in enumerate(points):
            metadata = (point.payload or {}).get("metadata") or {}
            shared = dict(metadata) if index == 0 else {k: v for k, v in shared.items() if metadata.get(k, _MISSING) == v}
        content_type = guess_file_type(BytesIO(kept))[0] if kept is not None else None
        if content_type in (None, "text/plain"):
            content_type = mimetypes.guess_type(source.name)[0] or content_type
        await crud_ingestions.create_status(
            source.agent_id, source.chat_id, source.name, type_=source.type,
            metadata={k: v for k, v in shared.items() if k not in _PHASE_METADATA}, content_type=content_type,
        )
        return True

    async def rerun_agent(self, cat: "CheshireCat", retry_errors: bool = False) -> Dict[str, int]:  # type: ignore[name-defined]
        """Bring every source of the agent (its knowledge base and its chats) up to date (see ``rerun_source``); with
        ``retry_errors``, the failed phases run again too. Public API, for the maintenance scripts: they put the agent
        in maintenance meanwhile, so that the automatic resume leaves it alone.

        Returns how many sources ended ``completed``, ``error``, or ``skipped`` (gone, or deleted meanwhile)."""
        counts = {"completed": 0, "error": 0, "skipped": 0}
        for row in await crud_ingestions.list_statuses(cat.agent_key):
            failed = [
                phase_id for phase_id, entry in (row.get("completed_phases") or {}).items()
                if entry.get("state") == engine.STATE_ERROR
            ] if retry_errors else None
            result = await self.rerun_source(cat, row["source"], row["chat_id"], invalidate=failed)
            status = (result or {}).get("status")
            counts[status if status in ("completed", "error") else "skipped"] += 1
        return counts

    async def _is_still_wanted(self, source: IngestionSource) -> bool:
        """Whether the source is still to be ingested: its agent and its row exist (a deleted chat removes its rows)."""
        return await engine.agent_exists(source.agent_id) and await crud_ingestions.get_status(
            source.agent_id, source.chat_id, source.name,
        ) is not None

    async def _check_still_wanted(self, source: IngestionSource):
        if not await self._is_still_wanted(source):
            raise engine.IngestionCanceled(source.name)

    async def _keep_source_bytes(self, source: IngestionSource, file_bytes: bytes, content_type: str | None, store_file: bool):
        """Keep the bytes of the source before processing it, so that a restart resumes it: a file among the uploaded
        files (``store_file``, or a file already there), a URL in the files of the RabbitHole. A file the caller does
        not store is not kept: this run reads it from memory, a later one needing its bytes fails."""
        await crud_ingestions.update_status(source.agent_id, source.chat_id, source.name, content_type=content_type)
        file_manager = self.cat.file_manager
        folder = chunks.owner_folder(source.agent_id, source.chat_id)
        if source.type == "file":
            if store_file:
                await self.cat.save_file(file_bytes, content_type, source.name, source.chat_id)
            if not store_file or await asyncio.to_thread(file_manager.file_exists, source.name, folder):
                return
        await asyncio.to_thread(
            chunks.write_source_bytes, file_manager, source.agent_id, source.chat_id, source.name, file_bytes,
        )

    async def _drop_kept_bytes(self, source: IngestionSource, store_file: bool):
        """Remove the bytes kept for a source whose ingestion was canceled before its phases."""
        file_manager = self.cat.file_manager
        await asyncio.to_thread(chunks.delete_all, file_manager, source.agent_id, source.chat_id, source.name)
        if store_file and source.type == "file":
            await asyncio.to_thread(
                file_manager.remove_file, os.path.join(chunks.owner_folder(source.agent_id, source.chat_id), source.name),
            )

    async def _notify_error(self, filename: str, error: str):
        # Don't raise in background tasks - just tell the chat
        if self.stray:
            try:
                await self.stray.notifier.send_error(f"Error processing {filename}: {error}")
            except Exception as notify_error:
                log.error(f"Failed to send error notification: {notify_error}")

    async def _stored_points_hook(self, source: str, points: List[PointStruct]):
        await self.cat.plugin_manager.execute_hook(
            "after_rabbithole_stored_documents", source, points, caller=self._resolve_caller(False),
        )

    async def run_parsing_chunking(self, source: IngestionSource, cat) -> PhaseResult:
        """The ``parsing_chunking`` phase: parse the source, collect its images, split it in chunks and store them.

        The heavy body is bounded to CAT_INGESTION_MAX_CONCURRENCY process-wide (with the embedding), whatever runs the
        phase: an upload, a resume, a change of settings."""
        async with _ingestion_guard(_get_ingestion_semaphore()):
            return await self._parsing_chunking(source)

    async def _parsing_chunking(self, source: IngestionSource) -> PhaseResult:
        file_manager = self.cat.file_manager
        pending = getattr(self, "_pending", None)
        if pending and pending[0] == source.name:
            file_bytes = pending[1]
        else:
            file_bytes = await asyncio.to_thread(
                chunks.read_source_bytes, file_manager, source.agent_id, source.chat_id, source.name,
            )
        if file_bytes is None and source.type == "url":
            # e.g. the worker crashed while downloading it: download it again, and keep it
            _, file_bytes, content_type, _ = await self._resolve_source_bytes(source.name, source.name, None)
            if not file_bytes:
                return PhaseResult(status="error", reason=f"Cannot download the source {source.name}")
            await crud_ingestions.update_status(source.agent_id, source.chat_id, source.name, content_type=content_type)
            await asyncio.to_thread(
                chunks.write_source_bytes, file_manager, source.agent_id, source.chat_id, source.name, file_bytes,
            )
        if file_bytes is None:
            return PhaseResult(
                status="error", reason=f"The source {source.name} was not kept: upload it again to process it",
            )

        row = await crud_ingestions.get_status(source.agent_id, source.chat_id, source.name) or {}
        content_type = row.get("content_type") or mimetypes.guess_type(source.name)[0]
        docs, images = await self._parse_to_docs(source.name, file_bytes, content_type)
        if not docs:
            return PhaseResult(status="error", reason=f"No valid chunks found in the file '{source.name}'.")

        docs = await self._prepare_documents(
            docs, source.name, hashlib.sha256(file_bytes).hexdigest(), row.get("metadata") or {}, source.chat_id,
        )
        await asyncio.to_thread(chunks.delete_all, file_manager, source.agent_id, source.chat_id, source.name, True)
        await asyncio.to_thread(
            chunks.write_chunks, file_manager, source.agent_id, source.chat_id, source.name, docs, images,
        )
        # the embedding of this run reads them from memory: a file manager may not keep files (e.g. the dummy one)
        self._pending_chunks = (source.name, docs, images, file_bytes)
        return PhaseResult(status="done", processed=len(docs))

    async def delete_parsing_chunking(self, source: IngestionSource, cat) -> None:
        await asyncio.to_thread(chunks.delete_all, self.cat.file_manager, source.agent_id, source.chat_id, source.name)

    async def run_embedding(self, source: IngestionSource, cat) -> PhaseResult:
        """The ``embedding`` phase: embed the stored chunks and replace the points of the source with the new ones
        (bounded to CAT_INGESTION_MAX_CONCURRENCY, as the parsing)."""
        async with _ingestion_guard(_get_ingestion_semaphore()):
            return await self._embedding(source)

    async def _embedding(self, source: IngestionSource) -> PhaseResult:
        file_manager = self.cat.file_manager
        pending = getattr(self, "_pending_chunks", None)
        if pending and pending[0] == source.name:
            _, docs, images, source_bytes = pending
        else:
            try:
                docs, images = await asyncio.to_thread(
                    chunks.read_chunks, file_manager, source.agent_id, source.chat_id, source.name,
                )
            except FileNotFoundError as e:
                return PhaseResult(status="error", reason=str(e))
            source_bytes = None
            if images:
                source_bytes = await asyncio.to_thread(
                    chunks.read_source_bytes, file_manager, source.agent_id, source.chat_id, source.name,
                )
        row = await crud_ingestions.get_status(source.agent_id, source.chat_id, source.name) or {}
        file_hash = next((d.metadata.get("hash") for d in docs if d.metadata.get("hash")), None)
        points = await self._embed_documents(
            docs, source.name, row.get("metadata") or {}, file_hash, images, source_bytes, source.chat_id,
        )
        if not points:
            return PhaseResult(status="error", reason=f"Nothing to embed in the source '{source.name}'.")

        # replace the points of the source: the new ones first, then the old ones go (no moment without the source)
        collection = self._collection(source.chat_id)
        handler = self.cat.vector_memory_handler
        old, _ = await handler.get_all_tenant_points(collection, metadata=self._source_filter(source), with_vectors=False)
        await handler.add_points_to_tenant(collection_name=collection, points=points)
        new_ids = {p.id for p in points}
        old_ids = [p.id for p in old or [] if p.id not in new_ids]
        if old_ids:
            await handler.delete_tenant_points_by_ids(collection, old_ids)

        await self._stored_points_hook(source.name, points)
        return PhaseResult(status="done", processed=len(points))

    async def delete_embedding(self, source: IngestionSource, cat) -> None:
        await self.cat.vector_memory_handler.delete_tenant_points(
            self._collection(source.chat_id), self._source_filter(source),
        )

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

        if isinstance(file, BytesIO):
            # Get the source of UploadFile, file bytes and whether it's a URL
            return _sanitize_filename(filename), file.read(), content_type, False
        if fnc_is_url(file):
            try:
                await self._ingestion_url_downloading(file)
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
                await self._ingestion_url_downloaded(file)
                return file, content, ct, True
            except Exception as e:
                log.error(f"Agent id: {self.cat.agent_key}. Error: {e}")
                return None, None, content_type, True
        # Get file bytes - use async file reading
        fb = await asyncio.to_thread(lambda: open(file, "rb").read())  # type: ignore[union-attr]
        return _sanitize_filename(os.path.basename(file)), fb, mimetypes.guess_type(file)[0], False  # type: ignore[return-value]

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
        super_docs = await run_in_ingestion_executor(
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
        # keeps only the first chunk's metadata). A multimodal plugin owns the extraction AND
        # strips the transient image_base64 from the docs; with no plugin the no-op returns []
        # (upstream parity: no image handling).
        images = await self.cat.plugin_manager.execute_hook(
            "rabbithole_collects_document_images", [], super_docs, caller=self._resolve_caller(False),
        ) or []

        # The hook dispatcher deep-copies its arguments, so a plugin stripping the transient
        # image payload only mutates its own copy. Strip it from the core's own documents
        # here (right after the hook, before chunking) so it never reaches the chunker, the
        # vector DB metadata or the LLM context on recall. The collected images are kept.
        _strip_image_payload(super_docs)

        # Split
        await self._send_notification_message("Parsing completed. Now let's go with reading process...")
        docs = await self._split_text(docs=super_docs)
        return docs, images

    async def _prepare_documents(
        self, docs: List[Document], source: str, file_hash: str | None, metadata: Dict, chat_id: str | None,
    ) -> List[Document]:
        """Add the metadata to the chunks (the custom ones, then source, when, hash and the chat) and pass them to the
        ``before_rabbithole_stores_documents`` hook, once per ingestion of the source. ``when`` is the same for every
        chunk of the ingestion."""
        when = time.time()
        for doc in docs:
            # Drop the transient parser image payload, if any: images are embedded and
            # saved as files separately, so their content must never reach the vector
            # DB metadata (and from there the LLM context on recall). This also covers
            # direct callers that pass documents still carrying image_base64.
            _strip_image_payload([doc])
            doc.metadata = (
                    doc.metadata
                    | metadata
                    | {"source": source, "when": when, "hash": file_hash}
                    | ({"chat_id": chat_id} if chat_id is not None else {})
            )

        # hook the docs before they are stored in the vector memory
        return await self.cat.plugin_manager.execute_hook(
            "before_rabbithole_stores_documents", docs, caller=self._resolve_caller(False),
        )

    async def _embed_documents(
        self,
        docs: List[Document],
        source: str,
        metadata: Dict,
        file_hash: str | None,
        images: List[Dict] | None,
        source_bytes: bytes | None,
        chat_id: str | None,
    ) -> List[PointStruct]:
        """The points of the chunks (and of the images, built by the plugins), embedded with the active embedder."""
        plugin_manager = self.cat.plugin_manager

        # Finalize the document list so no chunk exceeds the active embedder's max_input_tokens, right before the
        # embedding (it depends on the embedder): the actual split is owned by a plugin through the
        # finalize_oversized_chunks hook; the default no-op preserves the upstream behavior (no token-budget split).
        try:
            docs = await plugin_manager.execute_hook("finalize_oversized_chunks", docs, caller=self._resolve_caller(True))
        except Exception as e:
            # Log error but don't fail the entire process
            log.warning(f"Failed to finalize oversized chunks: {e}. Proceeding with original chunks.")

        log.info(f"Agent id: {self.cat.agent_key}. Preparing to memorize {len(docs)} vectors for {source}.")
        embedder = await self.cat.lizard.embedder()

        # Opt-in token-budget split, after the plugin hook above and right before embedding: the tokenizer depends on
        # the embedder. Idempotent, so a re-embedding (e.g. after a change of embedder) is harmless.
        if get_env_bool("CAT_INGESTION_SPLIT_OVERSIZED"):
            budget = embedder_token_budget()
            docs = split_oversized(docs, budget, getattr(embedder, "_estimate_tokens", None))

        # embed the non-empty docs, 1:1 with the stored points
        valid_documents = list(filter(lambda doc_: doc_.page_content.strip(), docs))
        storing_vectors = await run_in_ingestion_executor(
            lambda: embedder.embed_documents([doc_.page_content for doc_ in valid_documents])
        ) if valid_documents else []
        points = [PointStruct(
            id=uuid.uuid4().hex,
            payload=doc.model_dump(),
            vector=vector,
        ) for doc, vector in zip(valid_documents, storing_vectors)]

        # If a multimodal plugin collected images for this source, let it build the image
        # points (embed_images + save_file + PointStruct metadata): the core only appends
        # them to the same collection. With no plugin the no-op returns [] (upstream parity:
        # no image points).
        if images:
            image_points = await plugin_manager.execute_hook(
                "rabbithole_stores_image_points",
                [], images, source, source_bytes, metadata, file_hash, chat_id,
                caller=self._resolve_caller(True),
            )
            points.extend(image_points or [])
        return points

    async def store_documents(
        self,
        docs: List[Document],
        source: str,
        file_hash: str | None = None,
        metadata: Dict | None = None,
        images: List[Dict] | None = None,
        source_bytes: bytes | None = None,
    ) -> List[PointStruct]:
        """Add documents to the Cat's declarative memory (or to the episodic memory of the chat), in one step.

        The ingestion of a source goes through the phases (see ``ingest_file``); this method is for the callers that
        hold their documents already. It adds the metadata (the custom ones, the source, the timestamp of insertion,
        the hash and the chat), runs ``before_rabbithole_stores_documents``, embeds the documents and, if a multimodal
        plugin collected images, lets it build their points; then it adds every point to the collection.

        Args:
            docs (List[Document]): List of Langchain `Document` to be inserted in the Cat's declarative memory.
            source (str): Source name to be added as a metadata. It can be a file name or an URL.
            file_hash (str | None): Optional hash of the source to be added as a metadata.
            metadata (Dict | None): Optional metadata to be stored with each chunk.
            images (List[Dict] | None): Optional images extracted by multimodal parsers. Each entry has
                ``image_base64``, ``image_bytes`` and ``image_mime_type`` keys.
            source_bytes (bytes | None): Optional raw bytes of the ingested source file. Used when the source
                itself is an image: the file is embedded as a single whole-image point instead of the parser
                sub-crops, and no derived file is created.

        Returns:
            stored_points (List[PointStruct]): List of points stored (text chunks and, if any, image points).

        See Also:
            before_rabbithole_stores_documents
        """
        metadata = metadata or {}
        chat_id = self.stray.id if self.stray else None
        docs = await self._prepare_documents(docs, source, file_hash, metadata, chat_id)
        points = await self._embed_documents(docs, source, metadata, file_hash, images, source_bytes, chat_id)
        await self.cat.vector_memory_handler.add_points_to_tenant(
            collection_name=self._collection(chat_id), points=points,
        )
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
        docs = await plugin_manager.execute_hook(
            "before_rabbithole_splits_documents", docs, caller=self._resolve_caller(False),
        )

        # split docs
        docs = await self.cat.chunker.split_documents(docs)

        # join each short chunk with previous one, instead of deleting them
        try:
            docs = self._merge_short_chunks(docs, self.cat.chunker)
        except Exception as e:
            # Log error but don't fail the entire process
            log.warning(f"Error merging short chunks: {e}. Proceeding with original chunks.")

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
            """Check if two documents can be safely merged.

            Never across slides: a presentation chunk must keep one exact `slide` (chunks without
            `slide`, on either side, are compared as None, so plain documents merge as before).
            """
            if prev_doc.metadata.get("slide") != current_doc.metadata.get("slide"):
                return False
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
