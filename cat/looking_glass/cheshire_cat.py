import asyncio
import mimetypes
import os
import tempfile
import uuid
from io import BytesIO
from typing import List, Set

from cat.auth.permissions import AuthUserInfo
from cat.db.cruds import (
    conversations as crud_conversations,
    plugins as crud_plugins,
    settings as crud_settings,
    users as crud_users,
)
from cat.log import log
from cat.looking_glass.mad_hatter.mad_hatter import MadHatter
from cat.looking_glass.mad_hatter.procedures import CatProcedureType
from cat.looking_glass.models import StoredSourceWithMetadata
from cat.looking_glass.stray_cat import StrayCat
from cat.mixins import BotMixin, NonCopyableMixin
from cat.services.factory.file_manager import BaseFileManager
from cat.services.factory.vector_db import BaseVectorDatabaseHandler
from cat.services.memory.models import PointStruct, VectorMemoryType
from cat.utils import guess_file_type, is_url


class CheshireCat(BotMixin, NonCopyableMixin):
    """
    The Cheshire Cat.

    This is the main class that manages the whole AI application.
    It contains references to all the main modules and is responsible for the bootstrapping of the application.

    In most cases you will not need to interact with this class directly, but rather with class `StrayCat` which will be
    available in your plugin's hooks, tools, forms end endpoints.
    """
    def __init__(self, agent_id: str):
        """
        Cat initialization. At init time, the Cat executes the bootstrap.

        Args:
            agent_id: The agent identifier

        Notes
        -----
        Bootstrapping is the process of loading the plugins, the LLM, the memories.
        """
        self._id = agent_id

        # instantiate plugin manager (loads all plugins' hooks and tools)
        self.plugin_manager = MadHatter(self.agent_key)

    @classmethod
    async def create(cls, agent_id: str) -> "CheshireCat":
        """Factory method to create a CheshireCat instance."""
        cat = cls(agent_id)

        await cat.plugin_manager.discover_plugins()

        # allows plugins to do something before cat components are loaded
        await cat.plugin_manager.execute_hook("before_cat_bootstrap", caller=cat)

        await cat.service_provider.bootstrap_services(cat.agent_key, cat.plugin_manager)

        cat.agentic_workflow = await cat.service_provider.get_agentic_workflow(cat.agent_key, cat.plugin_manager)
        cat.chunker = await cat.service_provider.get_chunker(cat.agent_key, cat.plugin_manager)
        cat.context_retriever = await cat.service_provider.get_context_retriever(cat.agent_key, cat.plugin_manager)
        cat.custom_auth_handler = await cat.service_provider.get_custom_auth_handler(cat.agent_key, cat.plugin_manager)
        cat.file_manager = await cat.service_provider.get_file_manager(cat.agent_key, cat.plugin_manager)
        cat.large_language_model = await cat.service_provider.get_large_language_model(cat.agent_key, cat.plugin_manager)
        cat.vector_memory_handler = await cat.service_provider.get_vector_memory_handler(
            cat.agent_key, cat.plugin_manager,
        )

        # allows plugins to do something after the cat bootstrap is complete
        await cat.plugin_manager.execute_hook("after_cat_bootstrap", caller=cat)

        return cat

    def __eq__(self, other: "CheshireCat") -> bool:
        """Check if two cats are equal."""
        if not isinstance(other, CheshireCat):
            return False
        return self._id == other.agent_key

    def __hash__(self) -> int:
        return hash(self._id)

    def __repr__(self) -> str:
        return f"CheshireCat(agent_id={self._id})"

    async def destroy_memory(self):
        """Destroy all data from the cat's memory."""
        log.info(f"Agent id: {self._id}. Destroying all data from the cat's memory")

        # destroy all memories
        for collection_name in await self.vector_memory_handler.get_collection_names():
            await self.vector_memory_handler.delete_tenant_points(collection_name)

    async def destroy(self):
        """Destroy all data from the cat."""
        log.info(f"Agent id: {self._id}. Destroying all data from the cat")

        # destroy all memories
        await self.destroy_memory()

        # remove the folder from storage
        self.file_manager.remove_folder(self._id)

        await crud_settings.destroy_all(self._id)
        await crud_conversations.destroy_all(self._id)
        await crud_plugins.destroy_all(self._id)
        await crud_users.destroy_all(self._id)

        # notify plugins the agent has been destroyed (before shutdown tears down
        # the plugin manager), so plugin-owned namespaces can be cleaned up —
        # e.g. ingestion_status drops its agents:<id>:ingestion:* registry rows
        if self.plugin_manager is not None:
            await self.plugin_manager.execute_hook(
                "after_cheshire_cat_destroy", self._id, caller=self,
            )

        await self.shutdown()

    async def get_stored_sources_with_metadata(self) -> dict[VectorMemoryType, list[StoredSourceWithMetadata]]:
        """Get all stored files with their metadata."""
        results = {
            VectorMemoryType.DECLARATIVE: set(),
            VectorMemoryType.EPISODIC: set(),
        }
        for collection_name in results:
            points, _ = await self.vector_memory_handler.get_all_tenant_points(str(collection_name), with_vectors=False)
            for point in points:
                metadata = point.payload.get("metadata", {})  # type: ignore[union-attr]
                filename = metadata.get("source")
                if not filename:
                    continue

                if is_url(filename):
                    results[collection_name].add(
                        StoredSourceWithMetadata(name=filename, content=None, metadata=metadata, path=filename)
                    )
                    continue

                file_path = self.agent_key
                if chat_id := metadata.get("chat_id"):
                    file_path = os.path.join(file_path, chat_id)

                file_content = self.file_manager.read_file(filename, file_path)
                if not file_content:
                    continue

                results[collection_name].add(
                    StoredSourceWithMetadata(
                        name=filename, content=BytesIO(file_content), metadata=metadata, path=file_path,
                    )
                )

        return {k: list(v) for k, v in results.items()}

    async def _run_in_ingestion_executor(self, func, *args):
        """Run a heavy ingestion callable via the dedicated ingestion lane.

        Dispatches through the ``run_in_ingestion_executor`` hook (the
        efficient_ingestion plugin provides the dedicated pool); falls back to
        the default executor when no plugin provides it (upstream parity).
        """
        task = func if not args else (lambda: func(*args))
        result = await self.plugin_manager.execute_hook(
            "run_in_ingestion_executor", None, task, caller=self,
        )
        if result is None:
            result = await asyncio.to_thread(task)
        return result


    async def embed_procedures(self, pt: CatProcedureType | None = None):
        # Collect all texts up-front so we can embed them in one batch call
        # instead of N individual embed_query calls.
        documents = [
            t.document
            for p in self.plugin_manager.procedures
            for t in await p.to_document_recall()
            if pt is None or p.type == pt
        ]
        if not documents:
            return

        # Single batched embed call — much cheaper than N × embed_query, and offloaded
        # to a thread so the event loop is not blocked by the (synchronous) embedder.
        embedder = await self.embedder()
        vectors = await self._run_in_ingestion_executor(
            embedder.embed_documents, [document.page_content for document in documents]
        )

        # Guard: the embedder must produce vectors whose dimension matches the one used to
        # (re)create the vector collections (embedder.size from embed_query). A silent factory
        # fallback to a different embedder (e.g. DumbEmbedder) emits a different dimension and
        # Qdrant rejects it with an opaque "Vector dimension error" — failing loudly here makes
        # the real cause obvious instead of surfacing as an unhelpful Qdrant upsert error.
        expected_dim = embedder.size
        wrong_dims = {len(v) for v in vectors if len(v) != expected_dim}
        if wrong_dims:
            raise ValueError(
                f"Embedder `{embedder.name}` produced vectors of dimension {sorted(wrong_dims)} "
                f"but collection `{VectorMemoryType.PROCEDURAL!s}` expects {expected_dim}. "
                f"This usually means the configured embedder failed to instantiate and the factory "
                f"silently fell back to a different one. Check the ServiceFactory logs for the real error."
            )

        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                payload=d.model_dump(),
                vector=vector,
            )
            for d, vector in zip(documents, vectors)
        ]

        log.info(f"Agent id: {self._id}. Embedding procedures in vector memory")
        collection_name = str(VectorMemoryType.PROCEDURAL)

        # first, clear all existing procedural embeddings
        await self.vector_memory_handler.delete_tenant_points(collection_name)

        await self.vector_memory_handler.add_points_to_tenant(collection_name=collection_name, points=points)
        log.info(f"Agent id: {self._id}. Embedded {len(points)} triggers in {collection_name} vector memory")

    async def embed_stored_sources(
        self, collection_name: VectorMemoryType, stored_sources: List[StoredSourceWithMetadata]
    ):
        """
        Embeds stored sources into a vector memory collection.

        This method retrieves and processes a list of stored sources with their associated metadata and incorporates
        them into a vector memory collection. During this process, any pre-existing embeddings in the collection are
        cleared, and files are systematically ingested. The method handles potential irregularities such as missing
        content or stray references and logs appropriate messages.

        Args:
            collection_name (VectorMemoryType): The name of the collection where the stored sources
                will be embedded in vector memory.
            stored_sources (List[StoredSourceWithMetadata]): A list of sources, each containing content
                and metadata, to be embedded into vector memory.

        Raises:
            This method does not explicitly raise any exceptions but relies on the calling context to
            handle exceptions raised by dependent operations such as file ingestion.
        """
        log.info(f"Agent id: {self._id}. Embedding stored files to the vector memory")

        # first, clear all existing declarative and episodic embeddings
        await self.vector_memory_handler.delete_tenant_points(str(collection_name))

        rabbit_hole = self.rabbit_hole
        counter = 0
        for source in stored_sources:
            content_type = None
            if source.content:
                content_type, _ = guess_file_type(source.content)

            cat = self
            if chat_id := source.metadata.get("chat_id"):
                if not (stray_cat := await self._find_stray_cat(str(chat_id))):
                    log.warning(f"Stray cat with id {chat_id} not found. Skipping file {source.path}/{source.name}")
                    continue

                cat = stray_cat

            await rabbit_hole.ingest_file(
                cat=cat,
                file=source.content or source.name,
                filename=source.name,
                metadata=source.metadata or {},
                store_file=False,
                content_type=content_type,
            )
            counter += 1

        log.info(f"Agent id: {self._id}. Embedded {counter} files to the vector memory")

    async def save_file(self, file_bytes: bytes, content_type: str, source: str, chat_id: str | None = None):
        """
        Save file to the remote storage handled by the CheshireCat's file manager.

        Args:
            file_bytes (bytes): The file bytes to be saved.
            content_type (str): The content type of the file.
            source (str): The source of the file, i.e., the name used to store the file in the file manager.
            chat_id (str | None): The chat id of the stray cat, if any.
        """
        # save a file in a temporary folder
        extension = mimetypes.guess_extension(content_type)
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(file_bytes)
            file_path = temp_file.name

        # upload a file to CheshireCat's file manager
        try:
            remote_root_dir = self.agent_key
            if chat_id:
                remote_root_dir = os.path.join(remote_root_dir, chat_id)

            self.file_manager.upload_file(file_path, remote_root_dir, source)
        except Exception as e:
            log.error(f"Error while uploading file {file_path}: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    async def toggle_plugin(self, plugin_id: str):
        await self.plugin_manager.toggle_plugin(plugin_id)

        # destroy all procedural embeddings and re-embed them
        await self.vector_memory_handler.delete_tenant_points(str(VectorMemoryType.PROCEDURAL))
        await self.embed_procedures()

        await self.plugin_manager.execute_hook("after_plugin_toggling_on_agent", plugin_id, caller=self)

    async def _find_stray_cat(self, chat_id: str) -> StrayCat | None:
        """Finds a stray cat by chat id.

        Args:
            chat_id (str): The chat id of the stray cat.

        Returns:
            StrayCat | None: The stray cat if found, None otherwise.
        """
        # look for an existing conversation with the id = chat_id
        user_id = await crud_conversations.get_user_id_from_conversation_keys(self.agent_key, chat_id)
        if not user_id:
            return None

        user = await crud_users.get_user(self.agent_key, user_id)
        if not user:
            return None

        user_info = AuthUserInfo(
            id=user["id"],
            name=user["username"],
            permissions=user.get("permissions"),
            extra={k: v for k, v in user.items() if k not in AuthUserInfo.known_keys()},
        )
        return await StrayCat.from_cat(cat=self, user_data=user_info, stray_id=chat_id)

    def has_custom_endpoint(self, path: str, methods: Set[str] | List[str] | None = None):
        """
        Check if an endpoint with the given path and methods exists in the active plugins.

        Args:
            path (str): The path of the endpoint to check.
            methods (set[str] | List[str] | None): The HTTP methods of the endpoint to check. If None, checks all methods.

        Returns:
            bool: True if the endpoint exists, False otherwise.
        """
        for plugin in self.plugin_manager.plugins.values():
            # Check if the plugin has an endpoint with the given path and methods
            for ep in plugin.endpoints:
                if ep.real_path == path and (methods is None or set(ep.methods) == set(methods)):
                    return True

        return False

    def plugin_exists(self, plugin_id: str):
        return plugin_id in self.plugin_manager.plugins

    async def clone_from(self, ccat: "CheshireCat"):
        embedder = await self.embedder()
        await self.vector_memory_handler.initialize(embedder.name, embedder.size)

        log.info(f"Cloning vector memory from agent {ccat.agent_key} to agent {self.agent_key}")
        collection_name = str(VectorMemoryType.DECLARATIVE)

        points, _ = await ccat.vector_memory_handler.get_all_tenant_points(collection_name, with_vectors=True)
        if points:
            await self.vector_memory_handler.add_points_to_tenant(
                collection_name=collection_name,
                points=[
                    PointStruct(**{**p.model_dump(exclude={"shard_key", "order_value"}), "id": uuid.uuid4().hex})
                    for p in points
                ],
            )
        await self.embed_procedures()

        # clone the files from the ccat to the provided agent
        log.info(f"Cloning files from agent {ccat.agent_key} to agent {self.agent_key}")
        ccat.file_manager.clone_folder(ccat.agent_key, self.agent_key)

    async def transfer_files_from(self, previous_file_manager: BaseFileManager):
        try:
            self.file_manager.transfer(previous_file_manager, self.agent_key)
            success = True
        except Exception as e:
            log.error(f"Error while transferring files from previous file manager: {e}")
            success = False

        await self.plugin_manager.execute_hook("after_file_manager_transfer_on_agent", success, caller=self)

    async def transfer_vector_points_from(self, previous_vector_memory_handler: BaseVectorDatabaseHandler):
        embedder = await self.embedder()
        try:
            await self.vector_memory_handler.initialize(embedder.name, embedder.size)
            for collection_name in await previous_vector_memory_handler.get_collection_names():
                points, _ = await previous_vector_memory_handler.get_all_tenant_points(collection_name, with_vectors=True)
                if points:
                    await self.vector_memory_handler.add_points_to_tenant(
                        collection_name=collection_name,
                        points=[PointStruct(**p.model_dump(exclude={"shard_key", "order_value"})) for p in points],
                    )
            success = True
        except Exception as e:
            log.error(f"Error while transferring vector points from previous vector memory handler: {e}")
            success = False

        await self.plugin_manager.execute_hook("after_vector_memory_transfer_on_agent", success, caller=self)

    @property
    def agent_key(self) -> str:
        """
        The unique identifier of the cat.

        Returns:
            agent_id (str): The unique identifier of the cat.
        """
        return self._id
