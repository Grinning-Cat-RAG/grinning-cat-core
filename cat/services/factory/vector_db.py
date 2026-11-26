import asyncio
import inspect
import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Iterable, Dict, Tuple, Type
from urllib.parse import urlparse
from langchain_core.documents import Document as LangChainDocument
import aiofiles
import httpx
from pydantic import ConfigDict
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    CreateAliasOperation,
    CreateAlias,
    OptimizersConfigDiff,
    PayloadSchemaType,
    Filter,
    HasIdCondition,
    FieldCondition,
    MatchValue,
    MatchText,
    SearchParams,
    QuantizationSearchParams,
    PointStruct as QdrantPointStruct,
    SparseVectorParams,
    FusionQuery,
    Fusion,
    Prefetch,
)

from cat.env import get_env
from cat.log import log
from cat.services.factory.models import BaseFactoryConfigModel
from cat.services.memory.models import (
    Document,
    DocumentRecall,
    PointStruct,
    Record,
    ScoredPoint,
    UpdateResult,
    VectorMemoryType,
)


class BaseVectorDatabaseHandler(ABC):
    """
    Base class for vector database handlers.
    """
    _agent_id: str = None  # type: ignore[assignment]

    def __init__(self, save_memory_snapshots: bool = False):
        self.save_memory_snapshots = save_memory_snapshots
        self._collection_names = [str(v) for v in VectorMemoryType]

        self._client = None  # Placeholder for the database client

    def __eq__(self, other):
        if not isinstance(other, BaseVectorDatabaseHandler):
            return False

        return self.__class__.__name__ == other.__class__.__name__ and self._eq(other)

    def __del__(self):
        if not inspect.iscoroutinefunction(self.close):
            self.close()
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (e.g. an AnyIO worker thread): the explicit lifecycle owns the cleanup.
            log.debug(
                f"Skipping async close of the vector database handler for the agent `{self.agent_id}`: "
                "no running event loop"
            )
            return

        try:
            loop.create_task(self.close())
        except Exception:
            log.debug(f"Error while scheduling close of the vector database handler for the agent `{self.agent_id}`")

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value: str):
        if not value:
            raise ValueError("Agent ID cannot be empty")
        self._agent_id = value

    @abstractmethod
    def tenant_field_condition(self) -> Any:
        """
        Returns the field condition for tenant filtering.
        """
        pass

    @abstractmethod
    def _eq(self, other: "BaseVectorDatabaseHandler") -> bool:
        """
        Compares the current object with another object for equality.

        This method is an abstract method and must be implemented by subclasses.
        The equality comparison should be defined based on the specific criteria or attributes relevant to the subclass
        implementation.

        Args:
            other: An object to compare against the current instance.

        Returns:
            bool: The result of the equality comparison.
        """
        pass

    @abstractmethod
    async def initialize(self, embedder_name: str, embedder_size: int):
        """
        Initializes the vector database with the specified embedder name and size.
        Args:
            embedder_name: str, the name of the embedder to use.
            embedder_size: int, the size of the vector embeddings.
        """
        pass

    @abstractmethod
    def is_db_remote(self) -> bool:
        """
        Returns whether the vector database is remote or local.

        Returns:
            bool: True if the database is remote, False if local.
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Closes the connection to the vector database.
        This method should be overridden by subclasses if they maintain a connection.
        Implementations must be idempotent: calling ``close`` more than once is a no-op.
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str, timeout: int | None = None):
        """
        Delete a collection from the vector database.

        Args:
            collection_name: Name of the collection to delete
            timeout: Optional timeout for the operation
        """
        pass

    @abstractmethod
    async def save_dump(self, collection_name: str, folder="dormouse/"):
        """
        Save a dump of the specified collection to a folder.

        Args:
            collection_name: Name of the collection to dump
            folder: Folder where the dump will be saved
        """
        pass

    @abstractmethod
    async def retrieve_tenant_points(self, collection_name:str, points: List) -> List[Record]:
        """
        Retrieve points from the collection by their ids

        Args:
            collection_name: the name of the collection to retrieve points from
            points: the ids of the points to retrieve

        Returns:
            the list of points
        """
        pass

    @abstractmethod
    async def add_point_to_tenant(
        self,
        collection_name: str,
        content: str,
        vector: Iterable,
        metadata: Dict = None,  # type: ignore[assignment]
        id_point: str | None = None,
        **kwargs,
    ) -> PointStruct | None:
        """Add a point (and its metadata) to the vectorstore.

        Args:
            collection_name: Name of the collection to add the point to.
            content: original text.
            vector: Embedding vector.
            metadata: Optional metadata dictionary associated with the text.
            id_point: Optional id to associate with the point. Id has to be an uuid-like string.

        Returns:
            PointStruct: The stored point.
        """
        pass

    @abstractmethod
    async def add_points_to_tenant(
        self, collection_name: str, points: List[PointStruct]
    ) -> UpdateResult:
        """
        Upsert memories in batch mode

        Args:
            collection_name: the name of the collection to upsert points into
            points: the list of points to upsert

        Returns:
            The response of the upsert operation
        """
        pass

    @abstractmethod
    async def delete_tenant_points(self, collection_name: str, metadata: Dict | None = None) -> UpdateResult:
        """
        Delete points from the collection, by eventual metadata filter if provided.

        Args:
            collection_name: Name of the collection to delete points from.
            metadata: Optional metadata filter to select points for deletion.

        Returns:
            UpdateResult: The result of the delete operation.
        """
        pass

    @abstractmethod
    async def delete_tenant_points_by_ids(self, collection_name: str, points_ids: List) -> UpdateResult:
        """
        Delete points from the collection by their ids.

        Args:
            collection_name: Name of the collection to delete points from.
            points_ids: List of point ids to delete.

        Returns:
            UpdateResult: The result of the delete operation.
        """
        pass

    @abstractmethod
    async def recall_tenant_memory_from_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        metadata: Dict | None = None,
        k: int | None = 5,
        threshold: float | None = None,
    ) -> List[DocumentRecall]:
        """
        Retrieve memories from the collection based on an embedding vector. The memories are sorted by similarity to the
        embedding vector. The metadata filter is applied to the memories before retrieving them. The number of memories
        to retrieve is limited by the k parameter. The threshold parameter is used to filter out memories with a score
        below the threshold. The memories are returned as a list of tuples, where each tuple contains a Document, the
        similarity score, and the embedding vector of the memory. The Document contains the page content and metadata of
        the memory. The similarity score is a float between 0 and 1, where 1 is the highest similarity. The embedding
        vector is a list of floats. The list of tuples is sorted by similarity score in descending order. If the k
        parameter is None, all memories are retrieved. If the threshold parameter is None, no memories are filtered out.

        Args:
            collection_name: Name of the collection to search in.
            embedding: Embedding vector.
            metadata: Dictionary containing metadata filter.
            k: Number of memories to retrieve.
            threshold: Similarity threshold.

        Returns:
            List: List of DocumentRecall.
        """
        pass

    @abstractmethod
    async def recall_tenant_memory(self, collection_name: str) -> List[DocumentRecall]:
        """
        Retrieve the entire memories. It is similar to `recall_memories_from_embedding`, but without the embedding
        vector. Like `get_all_points`, it retrieves all the memories in the collection. The memories are returned in the
        same format as `recall_memories_from_embedding`.

        Args:
            collection_name: Name of the collection to retrieve memories from.

        Returns:
            List: List of DocumentRecall, like `recall_memories_from_embedding`, but with the nulled 2nd element
            (the score).

        See Also:
            VectorMemoryCollection.recall_memories_from_embedding
            VectorMemoryCollection.get_all_points
        """

    @abstractmethod
    async def get_all_tenant_points(
        self,
        collection_name: str,
        limit: int | None = None,
        offset: str | None = None,
        metadata: Dict | None = None,
        with_vectors: bool = True,
    ) -> Tuple[List[Record], int | str | None]:
        """
        Retrieve all the points in the collection with an optional offset and limit.

        Args:
            collection_name: The name of the collection to retrieve points from.
            limit: The maximum number of points to retrieve.
            offset: The offset from which to start retrieving points.
            metadata: Optional metadata filter to apply to the points.
            with_vectors: If True, returns the points with their vectors. If False, returns the points without their vectors.

        Returns:
            Tuple: A tuple containing the list of points and the next offset.
        """
        pass

    @abstractmethod
    async def get_all_tenant_points_from_web(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], int | str | None]:
        """
        Retrieve all the points in the collection with an optional offset and limit, specifically for web access.

        Args:
            collection_name: The name of the collection to retrieve points from.
            limit: The maximum number of points to retrieve.
            offset: The offset from which to start retrieving points.

        Returns:
            Tuple: A tuple containing the list of points and the next offset.
        """
        pass

    @abstractmethod
    async def get_all_tenant_points_from_files(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], int | str | None]:
        """
        Retrieve all the points in the collection with an optional offset and limit, specifically for file access.

        Args:
            collection_name: The name of the collection to retrieve points from.
            limit: The maximum number of points to retrieve.
            offset: The offset from which to start retrieving points.

        Returns:
            Tuple: A tuple containing the list of points and the next offset.
        """
        pass

    @abstractmethod
    async def get_tenant_vectors_count(self, collection_name: str) -> int:
        """
        Get the count of vectors in the specified collection.

        Args:
            collection_name: The name of the collection to count vectors in.

        Returns:
            int: The number of vectors in the collection.
        """
        pass

    @abstractmethod
    async def create_collection(self, embedder_name: str, embedder_size: int, collection_name: str):
        """
        Create a new collection in the vector database.

        Args:
            embedder_name: Name of the embedder to use for the collection
            embedder_size: Size of the embedding vectors
            collection_name: Name of the collection to create
        """
        pass

    @abstractmethod
    async def create_hybrid_collection(
        self, collection_name: str, dense_vector_config_name: str, sparse_vector_config_name: str
    ):
        """
        Creates a hybrid collection in the database with specified configurations.

        Args:
            collection_name (str): The name of the collection to be created.
            dense_vector_config_name (str): The name of the dense vector configuration to be used for the collection.
            sparse_vector_config_name (str): The name of the sparse vector configuration to be used for the collection.
        """
        pass

    @abstractmethod
    async def get_collection_names(self) -> List[str]:
        """
        Get the list of collection names in the vector database.

        Returns:
            List[str]: List of collection names
        """
        pass

    @abstractmethod
    async def check_collection_existence(self, collection_name: str) -> bool:
        """
        Checks if a collection exists in the database.

        This method verifies the existence of a specified collection within the database. It returns a boolean
        indicating whether the collection is found or not.

        Args:
            collection_name (str): The name of the collection to check for existence.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        pass

    @abstractmethod
    async def search_in_tenant(
        self,
        collection_name: str,
        query_vector: List[float],
        query_filter: Any = None,
        with_payload: bool = True,
        with_vectors: bool = True,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> List[ScoredPoint]:
        """
        Performs a search operation in the database and retrieves results based on the provided query parameters.

        Args:
            collection_name (str): Name of the collection to search in.
            query_vector (List[float]): Vector representation of the query for similarity matching.
            query_filter (Any, optional): Filter criteria to narrow down search results. Defaults to None.
            with_payload (bool, optional): Determines whether to include payload data in the results. Defaults to True.
            with_vectors (bool, optional): Determines whether to include vectors in the results. Defaults to True.
            limit (int, optional): Maximum number of results to retrieve. Defaults to 10.
            score_threshold (float | None, optional): Minimum score threshold for the results to be considered relevant. Defaults to None.

        Returns:
            List[ScoredPoint]: A list containing the search results, with each result being an instance of `ScoredPoint`.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def search_prefetched_in_tenant(
        self,
        collection_name: str,
        query: str,
        query_vector: List[float],
        query_filter: Any,
        k: int,
        k_prefetched: int,
        threshold: float,
    ) -> List[ScoredPoint]:
        """
        Search the prefetched items in the given collection based on query and filtering criteria.

        This is an asynchronous method used to perform a search within a specified collection, applying a query and
        additional filtering mechanisms. It retrieves a ranked list of items that match the provided criteria,
        considering both the main query and prefetching parameters.

        Args:
            collection_name: The name of the collection in which the search will be executed.
            query: The query string or identifier used to retrieve relevant items.
            query_vector: A list of floats representing the vectorized form of the query for similarity search.
            query_filter: Additional filtering criteria to narrow down the search results.
            k: The number of top-ranked items to retrieve from the search results.
            k_prefetched: The number of items to be prefetched before applying the ranking.
            threshold: A score threshold above which a point is considered relevant.

        Returns:
            A list of `ScoredPoint` objects representing the ranked results matching the query and other criteria.
        """

    @abstractmethod
    def build_condition(self, key: str, value: Any) -> List[FieldCondition]:
        """
        Builds a list of conditions based on the provided key and value.

        This method should be implemented in subclasses to provide the specific logic for building conditions. A
        key-value pair is used as input to produce a list of  `FieldCondition`, which represents conditions in a
        structured format.

        Args:
            key: A string representing the field name or identifier.
            value: A value of any type that represents the criteria or data for the condition.

        Returns:
            List[FieldCondition]: A list of `FieldCondition` objects, each representing  a specific condition derived from the key and value.
        """
        pass

    @abstractmethod
    def filter_from_dict(self, filter_dict: Dict) -> Any:
        """
        Filters data based on the given dictionary of filter conditions.

        This method should be implemented by subclasses to provide specific filtering logic using the provided filter
        dictionary. The `filter_dict` defines the criteria or conditions used to filter data objects.

        Args:
            filter_dict (Dict): A dictionary containing filter criteria.

        Returns:
            The filter to be applied to the data.
        """

    @property
    @abstractmethod
    def client(self):
        """
        Get the client instance for the vector database.

        Returns:
            The client instance
        """
        pass


class QdrantHandler(BaseVectorDatabaseHandler):
    def __init__(
        self,
        host: str,
        port: int,
        api_key: str | None = None,
        client_timeout: int | None = 100,
        save_memory_snapshots: bool = False,
    ):
        if host is None:
            raise ValueError("CAT_QDRANT_HOST environment variable is not set.")

        super().__init__(save_memory_snapshots)
        self.port = port
        self.api_key = api_key or None

        try:
            parsed_url = urlparse(host)
            self.is_https = parsed_url.scheme == "https"
            self.host = parsed_url.netloc + parsed_url.path
        except:
            self.is_https = False
            self.host = host

        qdrant_client_timeout = int(client_timeout) if client_timeout is not None else None

        try:
            self._client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                https=self.is_https,
                api_key=self.api_key,
                prefer_grpc=True,
                force_disable_check_same_thread=True,
                timeout=qdrant_client_timeout,
            )
        except Exception as e:
            log.error(f"Qdrant does not respond to {self.host}:{port}")
            raise e

    def _eq(self, other: "QdrantHandler") -> bool:
        return self.client.init_options == other.client.init_options

    @property
    def client(self):
        return self._client

    async def initialize(self, embedder_name: str, embedder_size: int):
        """
        Initializes the vector database with the specified embedder name and size.
        Args:
            embedder_name: str, the name of the embedder to use.
            embedder_size: int, the size of the vector embeddings.
        """
        for collection_name in self._collection_names:
            is_collection_existing = await self.check_collection_existence(collection_name)
            has_same_size = (
                await self._check_embedding_size(embedder_name, embedder_size, collection_name)
            ) if is_collection_existing else False
            if is_collection_existing and has_same_size:
                continue

            # dump collection on disk before deleting
            if self.save_memory_snapshots:
                await self.save_dump(collection_name)

            if is_collection_existing:
                await self.delete_collection(collection_name=collection_name)
            await self.create_collection(embedder_name, embedder_size, collection_name)

    def tenant_field_condition(self) -> FieldCondition:
        return FieldCondition(key="tenant_id", match=MatchValue(value=self.agent_id))

    # adapted from https://github.com/langchain-ai/langchain/blob/bfc12a4a7644cfc4d832cc4023086a7a5374f46a/libs/langchain/langchain/vectorstores/qdrant.py#L1941
    # see also https://github.com/langchain-ai/langchain/blob/bfc12a4a7644cfc4d832cc4023086a7a5374f46a/libs/langchain/langchain/vectorstores/qdrant.py#L1965
    def build_condition(self, key: str, value: Any) -> List[FieldCondition]:
        out = []

        if isinstance(value, dict):
            for k, v in value.items():
                out.extend(self.build_condition(f"{key}.{k}", v))
            return out

        if isinstance(value, list):
            for v in value:
                out.extend(self.build_condition(f"{key}[]" if isinstance(v, dict) else f"{key}", v))
            return out

        out.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))
        return out

    def _build_metadata_conditions(self, metadata: Dict | None = None) -> List[FieldCondition]:
        conditions = [self.tenant_field_condition()]
        if metadata:
            conditions.extend([
                condition for key, value in metadata.items() for condition in self.build_condition(key, value)
            ])
        return conditions

    async def check_collection_existence(self, collection_name: str) -> bool:
        collection_names = await self.get_collection_names()
        if any(c == collection_name for c in collection_names):
            # collection exists. Do nothing
            log.debug(f"Collection `{collection_name}` for the agent `{self.agent_id}` already present in vector store")
            return True

        return False

    @staticmethod
    def _get_local_alias(embedder_name: str, collection_name: str) -> str:
        return f"{embedder_name}_{collection_name}"

    async def _check_embedding_size(self, embedder_name: str, embedder_size: int, collection_name: str) -> bool:
        # having the same size does not necessarily imply being the same embedder
        # having vectors with the same size but from different embedder in the same vector space is wrong
        collection_vector_size = (await self._client.get_collection(collection_name=collection_name)).config.params.vectors.size  # type: ignore[attr-defined]
        same_size = collection_vector_size == embedder_size
        local_alias = self._get_local_alias(embedder_name, collection_name)

        existing_aliases = (await self._client.get_collection_aliases(collection_name=collection_name)).aliases  # type: ignore[attr-defined]

        if same_size and existing_aliases and local_alias == existing_aliases[0].alias_name:
            log.debug(f"Collection `{collection_name}` for the agent `{self.agent_id}` has the same embedder")
            return True

        log.warning(f"Collection `{collection_name}` for the agent `{self.agent_id}` has different embedder")
        return False

    async def create_collection(self, embedder_name: str, embedder_size: int, collection_name: str):
        log.warning(f"Creating collection `{collection_name}` for the agent `{self.agent_id}`...")

        try:
            await self._client.create_collection(  # type: ignore[attr-defined]
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedder_size, distance=Distance.COSINE),
                # hybrid mode: original vector on Disk, quantized vector in RAM
                optimizers_config=OptimizersConfigDiff(memmap_threshold=20000, indexing_threshold=20000),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8, quantile=0.95, always_ram=True
                    )
                ),
                # shard_number=3,
            )
        except Exception as e:
            log.error(
                f"Error creating collection `{collection_name}` for the agent `{self.agent_id}`. Try setting a higher timeout value: {e}"
            )
            raise

        alias_name = self._get_local_alias(embedder_name, collection_name)
        log.warning(f"Creating alias `{alias_name}` for collection `{collection_name}` and the agent `{self.agent_id}`...")
        try:
            await self._client.update_collection_aliases(  # type: ignore[attr-defined]
                change_aliases_operations=[
                    CreateAliasOperation(
                        create_alias=CreateAlias(
                            collection_name=collection_name,
                            alias_name=alias_name,
                        )
                    )
                ]
            )
        except Exception as e:
            log.error(f"Error creating collection alias `{alias_name}` for collection `{collection_name}` and the agent `{self.agent_id}`: {e}")
            await self._client.delete_collection(collection_name)  # type: ignore[attr-defined]
            log.error(f"Collection `{collection_name}` for the agent `{self.agent_id}` deleted")
            raise e

        # if the client is remote, create an index on the tenant_id field
        if self.is_db_remote():
            log.warning(f"Creating payload index for collection `{collection_name}` and the agent `{self.agent_id}`...")
            try:
                await self._client.create_payload_index(  # type: ignore[attr-defined]
                    collection_name=collection_name,
                    field_name="tenant_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                log.error(f"Error when creating a schema index: {e}")

    async def create_hybrid_collection(
        self, collection_name: str, dense_vector_config_name: str, sparse_vector_config_name: str
    ):
        if await self._client.collection_exists(collection_name):  # type: ignore[attr-defined]
            return

        embedding_size = (
            await self._client.get_collection(collection_name=str(VectorMemoryType.DECLARATIVE))  # type: ignore[attr-defined]
        ).config.params.vectors.size

        try:
            await self._client.create_collection(  # type: ignore[attr-defined]
                collection_name=collection_name,
                vectors_config={
                    dense_vector_config_name: VectorParams(
                        size=embedding_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={sparse_vector_config_name: SparseVectorParams()},
            )
        except Exception as e:
            log.error(
                f"Error creating collection `{collection_name}` for the agent `{self.agent_id}`. Try setting a higher timeout value: {e}"
            )
            raise

        # if the client is remote, create an index on the tenant_id field
        if self.is_db_remote():
            log.warning(f"Creating payload index for collection `{collection_name}` and the agent `{self.agent_id}`...")
            try:
                await self._client.create_payload_index(  # type: ignore[attr-defined]
                    collection_name=collection_name,
                    field_name="tenant_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                log.error(f"Error when creating a schema index: {e}")

    async def close(self):
        if self._client is None:
            return
        client, self._client = self._client, None
        if not client._client.closed:
            await client.close()

    async def delete_collection(self, collection_name: str, timeout: int | None = None):
        """
        Delete a collection from the vector database.

        Args:
            collection_name: Name of the collection to delete
            timeout: Optional timeout for the operation
        """
        await self._client.delete_collection(collection_name=collection_name, timeout=timeout)  # type: ignore[attr-defined]
        log.warning(f"Collection `{collection_name}` for the agent `{self.agent_id}` deleted")

    # dump collection on disk before deleting
    async def save_dump(self, collection_name: str, folder="dormouse/"):
        # only do snapshotting if using remote Qdrant
        if not self.is_db_remote():
            return

        host = self._client._client._host  # type: ignore[attr-defined]
        port = self._client._client._port  # type: ignore[attr-defined]

        os.makedirs(folder, exist_ok=True)

        snapshot_info = await self._client.create_snapshot(collection_name=collection_name)  # type: ignore[attr-defined]
        snapshot_url_in = (
            "http://"
            + str(host)
            + ":"
            + str(port)
            + "/collections/"
            + collection_name
            + "/snapshots/"
            + snapshot_info.name
        )
        snapshot_url_out = os.path.join(folder, snapshot_info.name)
        # rename snapshots for an easier restore in the future
        alias = (await self._client.get_collection_aliases(collection_name=collection_name)).aliases[0].alias_name  # type: ignore[attr-defined]

        async with httpx.AsyncClient() as client:
            response = await client.get(snapshot_url_in)
            async with aiofiles.open(snapshot_url_out, "wb") as f:
                await f.write(response.content)  # Write the content asynchronously

        new_name = os.path.join(folder, alias.replace("/", "-") + ".snapshot")
        os.rename(snapshot_url_out, new_name)

        for s in (await self._client.list_snapshots(collection_name=collection_name)):  # type: ignore[attr-defined]
            await self._client.delete_snapshot(collection_name=collection_name, snapshot_name=s.name)  # type: ignore[attr-defined]
        log.warning(f"Dump `{new_name}` for the agent `{self.agent_id}` completed")

    async def retrieve_tenant_points(self, collection_name:str, points: List) -> List[Record]:
        """
        Retrieve points from the collection by their ids

        Args:
            collection_name: the name of the collection to retrieve points from
            points: the ids of the points to retrieve

        Returns:
            the list of points
        """
        points_found, _ = await self._client.scroll(  # type: ignore[attr-defined]
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=self.agent_id)), HasIdCondition(has_id=points)]
            ),
            limit=len(points),
            with_payload=True,
            with_vectors=True,
        )

        return [Record(**point.model_dump()) for point in points_found]

    async def add_point_to_tenant(
        self,
        collection_name: str,
        content: str,
        vector: Iterable,
        metadata: Dict = None,  # type: ignore[assignment]
        id_point: str | None = None,
        **kwargs,
    ) -> PointStruct | None:
        """Add a point (and its metadata) to the vectorstore.

        Args:
            collection_name: Name of the collection to add the point to.
            content: original text.
            vector: Embedding vector.
            metadata: Optional metadata dictionary associated with the text.
            id_point: Optional id to associate with the point. Id has to be an uuid-like string.

        Returns:
            PointStruct: The stored point.
        """
        id_point = id_point or uuid.uuid4().hex

        point = QdrantPointStruct(
            id=id_point,
            payload={
                "id": id_point,
                "page_content": content,
                "metadata": metadata,
                "tenant_id": self.agent_id,
            },
            vector=vector,  # type: ignore
        )

        update_status = await self._client.upsert(collection_name=collection_name, points=[point], **kwargs)  # type: ignore[attr-defined]

        if update_status.status == "completed":
            # returning stored point
            return PointStruct(**point.model_dump())

        return None

    # add points in collection
    async def add_points_to_tenant(
        self, collection_name: str, points: List[PointStruct]
    ) -> UpdateResult:
        for point in points:
            point.payload["tenant_id"] = self.agent_id  # type: ignore[index]

        res = await self._client.upsert(  # type: ignore[attr-defined]
            collection_name=collection_name, points=[QdrantPointStruct(**p.model_dump()) for p in points],
        )

        return UpdateResult(status=res.status, operation_id=res.operation_id)

    async def delete_tenant_points(self, collection_name: str, metadata: Dict | None = None) -> UpdateResult:
        conditions = self._build_metadata_conditions(metadata=metadata)

        res = await self._client.delete(collection_name=collection_name, points_selector=Filter(must=conditions))  # type: ignore[attr-defined]
        return UpdateResult(
            status=res.status,
            operation_id=res.operation_id,
        )

    # delete point in collection
    async def delete_tenant_points_by_ids(self, collection_name: str, points_ids: List) -> UpdateResult:
        res = await self._client.delete(collection_name=collection_name, points_selector=points_ids)  # type: ignore[attr-defined]
        return UpdateResult(
            status=res.status,
            operation_id=res.operation_id,
        )

    def _to_document_recall(self, m: Record | ScoredPoint) -> DocumentRecall:
        """
        Convert a Qdrant point to a DocumentRecall object

        Args:
            m (Record | ScoredPoint): The Qdrant point

        Returns:
            DocumentRecall: The converted DocumentRecall object
        """
        page_content = m.payload.get("page_content", "") if m.payload else ""
        if isinstance(page_content, dict):
            page_content = json.dumps(page_content)

        metadata = m.payload.get("metadata", {}) if m.payload else {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        document = DocumentRecall(
            document=LangChainDocument(
                page_content=page_content,
                metadata=metadata,
                id=m.id,
            ),
            vector=m.vector,
            id=m.id,
        )

        if isinstance(m, ScoredPoint):
            document.score = m.score

        return document

    # retrieve similar memories from embedding
    async def recall_tenant_memory_from_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        metadata: Dict | None = None,
        k: int | None = 5,
        threshold: float | None = None,
    ) -> List[DocumentRecall]:
        conditions = self._build_metadata_conditions(metadata=metadata)

        # retrieve memories
        query_response = await self._client.query_points(  # type: ignore[attr-defined]
            collection_name=collection_name,
            query=embedding,  # type: ignore
            query_filter=Filter(must=conditions),
            with_payload=True,
            with_vectors=True,
            limit=k,
            score_threshold=threshold,
            search_params=SearchParams(
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,  # Available as of v1.3.0
                )
            ),
        )

        # convert Qdrant points to a structure containing langchain.Document and its information
        retrieved_points = [ScoredPoint(**point.model_dump()) for point in query_response.points]
        return [self._to_document_recall(m) for m in retrieved_points]

    async def recall_tenant_memory(self, collection_name: str) -> List[DocumentRecall]:
        all_points, _ = await self.get_all_tenant_points(collection_name)
        memories = [self._to_document_recall(p) for p in all_points]

        return memories

    async def _get_all_points(
        self,
        collection_name: str,
        scroll_filter: Filter,
        limit: int | None = None,
        offset: str | None = None,
        with_vectors: bool = True,
    ) -> Tuple[List[Record], int | str | None]:
        if limit is not None:
            # retrieving the points
            points_batch, next_offset = await self._client.scroll(  # type: ignore[attr-defined]
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                with_vectors=with_vectors,
                offset=offset,  # Start from the given offset, or the beginning if None.
                limit=limit  # Limit the number of points retrieved to the specified limit.
            )
            return [Record(**point.model_dump()) for point in points_batch], next_offset

        # retrieve all points without limit
        memory_points = []
        limit = 10000
        while True:
            # Get a batch of points
            points_batch, next_offset = await self._get_all_points(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_vectors=with_vectors,
            )

            # Add filtered points to our collection
            memory_points.extend(points_batch)

            # Check if we have more pages
            if next_offset is None:
                # No more pages
                break

            # Set offset for the next iteration
            offset = next_offset  # type: ignore[assignment]

        return memory_points, None

    # retrieve all the points in the collection
    async def get_all_tenant_points(
        self,
        collection_name: str,
        limit: int | None = None,
        offset: str | None = None,
        metadata: Dict | None = None,
        with_vectors: bool = True,
    ) -> Tuple[List[Record], int | str | None]:
        conditions = self._build_metadata_conditions(metadata)
        return await self._get_all_points(
            collection_name=collection_name,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            offset=offset,
            with_vectors=with_vectors,
        )

    async def get_all_tenant_points_from_web(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], int | str | None]:
        conditions = [
            self.tenant_field_condition(),
            FieldCondition(
                key="metadata.source",
                match=MatchText(text="http")  # Regex for "starts with http"
            )
        ]

        return await self._get_all_points(
            collection_name=collection_name,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            offset=offset,
            with_vectors=False,
        )

    async def get_all_tenant_points_from_files(
        self, collection_name: str, limit: int | None = None, offset: str | None = None
    ) -> Tuple[List[Record], int | str | None]:
        filter_condition = Filter(
            must_not=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchText(text="http")  # Regex for "starts with http"
                )
            ],
            must=[
                self.tenant_field_condition(),
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value="^http")  # Regex for "starts with http"
                )
            ]
        )

        return await self._get_all_points(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=limit,
            offset=offset,
            with_vectors=False
        )

    async def get_tenant_vectors_count(self, collection_name: str) -> int:
        return (await self._client.count(  # type: ignore[attr-defined]
            collection_name=collection_name,
            count_filter=Filter(must=[self.tenant_field_condition()]),
        )).count

    def is_db_remote(self) -> bool:
        return True

    async def get_collection_names(self) -> List[str]:
        collections_response = await self._client.get_collections()  # type: ignore[attr-defined]
        return [c.name for c in collections_response.collections]

    async def search_in_tenant(
        self,
        collection_name: str,
        query_vector: List[float],
        query_filter: Any = None,
        with_payload: bool = True,
        with_vectors: bool = True,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> List[ScoredPoint]:
        response = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,  # type: ignore
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
            limit=limit,
            score_threshold=score_threshold,
            search_params=SearchParams(
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,  # Available as of v1.3.0
                )
            ),
        )

        return response.points

    async def search_prefetched_in_tenant(
        self,
        collection_name: str,
        query: str,
        query_vector: List[float],
        query_filter: Any,
        k: int,
        k_prefetched: int,
        threshold: float,
    ) -> List[ScoredPoint]:
        response = await self._client.query_points(
            collection_name=collection_name,
            query=FusionQuery(fusion=Fusion.RRF),  # type: ignore
            query_filter=query_filter,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=k_prefetched,
                ),
                Prefetch(
                    query=Document(text=query, model="Qdrant/bm25"),
                    using="sparse",
                    limit=k_prefetched,
                ),
            ],
            with_payload=True,
            with_vectors=True,
            limit=k,
            score_threshold=threshold,
        )

        return response.points

    def filter_from_dict(self, filter_dict: Dict) -> Filter | None:
        if not filter_dict or len(filter_dict) < 1:
            return None

        return Filter(
            should=[
                condition
                for key, value in filter_dict.items()
                for condition in self.build_condition(key, value)
            ]
        )


class VectorDatabaseSettings(BaseFactoryConfigModel, ABC):
    save_memory_snapshots: bool = False

    # This is related to pydantic, because "model_*" attributes are protected.
    # We deactivate the protection because langchain relies on several "model_*" named attributes
    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def base_class(cls) -> Type[BaseVectorDatabaseHandler]:
        return BaseVectorDatabaseHandler

    @classmethod
    @abstractmethod
    def pyclass(cls) -> Type[BaseVectorDatabaseHandler]:
        pass


class QdrantConfig(VectorDatabaseSettings):
    host: str = get_env("CAT_QDRANT_HOST")
    port: int = 6333
    api_key: str | None = get_env("CAT_QDRANT_API_KEY") or None
    client_timeout: int | None = 100

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Remote Qdrant Vector Database",
            "description": "Configuration for Remote Qdrant Vector Database",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[QdrantHandler]:
        return QdrantHandler
