import asyncio
import os
from typing import Dict, List, Any
from fastapi import Query, Depends
from pydantic import BaseModel

from cat import (
    AuthorizedInfo,
    AuthPermission,
    AuthResource,
    check_permissions,
    endpoint,
)
from cat.core_plugins.memory.routes_utils import (
    MemoryPointBase,
    MemoryPoint,
    upsert_memory_point,
    verify_memory_point_existence,
)
from cat.exceptions import CustomValidationException
from cat.routes.routes_utils import create_dict_parser
from cat.services.memory.models import DocumentRecall, UpdateResult, Record, VectorMemoryType


class RecallResponseQuery(BaseModel):
    text: str
    vector: List[float]


class RecallResponseVectors(BaseModel):
    embedder: str
    collections: Dict[str, List[Dict[str, Any]]]


class RecallResponse(BaseModel):
    query: RecallResponseQuery
    vectors: RecallResponseVectors


class GetPointsInCollectionResponse(BaseModel):
    points: List[Record]
    next_offset: int | str | None


class DeleteMemoryPointResponse(BaseModel):
    deleted: str


class DeleteMemoryPointsByMetadataResponse(BaseModel):
    deleted: UpdateResult


@endpoint.get(
    "/recall", response_model=RecallResponse, tags=["Vector Memory - Points"], prefix="/memory"
)
async def recall_memory_points_from_text(
    text: str = Query(description="Find memories similar to this text."),
    k: int = Query(default=100, description="How many memories to return."),
    metadata: Dict[str, Any] = Depends(create_dict_parser(
        "metadata",
        description="Flat dictionary where each key-value pair represents a filter."
                    "The memory points returned will match the specified metadata criteria."
    )),
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> RecallResponse:
    """
    Search k memories similar to given text with specified metadata criteria.

    Example
    ----------
    ```
    collection = "declarative"
    content = "MIAO!"
    metadata = {"custom_key": "custom_value"}
    req_json = {
        "content": content,
        "metadata": metadata,
    }
    # create a point
    res = requests.post(
        f"http://localhost:1865/memory/collections/{collection}/points", json=req_json
    )

    # recall with metadata
    req_json = {
        "text": "CAT",
        "metadata":{"custom_key":"custom_value"}
    }
    res = requests.post(
        f"http://localhost:1865/memory/recall", json=req_json
    )
    json = res.json()
    print(json)
    ```
    """
    def build_memory_dict(document_recall: DocumentRecall) -> Dict[str, Any]:
        memory_dict = dict(document_recall.document)
        memory_dict.pop("lc_kwargs", None)  # langchain stuff, not needed
        memory_dict["id"] = document_recall.id
        memory_dict["score"] = float(document_recall.score) if document_recall.score else None
        memory_dict["vector"] = document_recall.vector
        return memory_dict

    lizard = info.lizard

    # Embed the query to plot it in the Memory page
    embedder = await lizard.embedder()
    query_embedding = await asyncio.to_thread(embedder.embed_query, text)
    collection_name = str(VectorMemoryType.DECLARATIVE if not info.stray_cat else VectorMemoryType.EPISODIC)
    metadata = {k: v for k, v in metadata.items() if k != "source"}
    if info.stray_cat:
        metadata["chat_id"] = info.stray_cat.id

    dm = await info.cheshire_cat.vector_memory_handler.recall_tenant_memory_from_embedding(
        collection_name, query_embedding, k=k, metadata=metadata,
    )

    return RecallResponse(
        query=RecallResponseQuery(text=text, vector=query_embedding),
        vectors=RecallResponseVectors(
            embedder=embedder.name,
            collections={
                collection_name: [build_memory_dict(document_recall) for document_recall in dm]
            }
        )
    )


@endpoint.post(
    "/collections/{collection_id}/points", response_model=MemoryPoint, tags=["Vector Memory - Points"], prefix="/memory"
)
async def create_memory_point(
    collection_id: str,
    point: MemoryPointBase,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.WRITE),
) -> MemoryPoint:
    """Create a point in memory"""
    try:
        return await upsert_memory_point(collection_id, point, info)
    except Exception as e:
        raise CustomValidationException(f"Failed to create memory point: {e}")


@endpoint.put(
    "/collections/{collection_id}/points/{point_id}",
    response_model=MemoryPoint,
    tags=["Vector Memory - Points"],
    prefix="/memory",
)
async def edit_memory_point(
    collection_id: str,
    point_id: str,
    point: MemoryPointBase,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.WRITE),
) -> MemoryPoint:
    """Edit a point in memory

    Example
    ----------
    ```

    collection = "declarative"
    content = "MIAO!"
    metadata = {"custom_key": "custom_value"}
    req_json = {
        "content": content,
        "metadata": metadata,
    }
    # create a point
    res = requests.post(
        f"http://localhost:1865/memory/collections/{collection}/points", json=req_json
    )
    json = res.json()
    #get the id
    point_id = json["id"]
    # new point values
    content = "NEW MIAO!"
    metadata = {"custom_key": "new_custom_value"}
    req_json = {
        "content": content,
        "metadata": metadata,
    }
    # edit the point
    res = requests.put(
        f"http://localhost:1865/memory/collections/{collection}/points/{point_id}", json=req_json
    )
    json = res.json()
    print(json)
    ```
    """
    try:
        await verify_memory_point_existence(info.cheshire_cat, collection_id, point_id)

        return await upsert_memory_point(collection_id, point, info, point_id)
    except Exception as e:
        raise CustomValidationException(f"Failed to edit memory point: {e}")


@endpoint.delete(
    "/collections/{collection_id}/points/{point_id}",
    response_model=DeleteMemoryPointResponse,
    tags=["Vector Memory - Points"],
    prefix="/memory",
)
async def delete_memory_point(
    collection_id: str,
    point_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> DeleteMemoryPointResponse:
    """Delete a specific point in memory"""
    try:
        await verify_memory_point_existence(info.cheshire_cat, collection_id, point_id)

        # delete point
        await info.cheshire_cat.vector_memory_handler.delete_tenant_points_by_ids(collection_id, [point_id])

        return DeleteMemoryPointResponse(deleted=point_id)
    except Exception as e:
        raise CustomValidationException(f"Failed to delete memory point: {e}")


@endpoint.delete(
    "/collections/{collection_id}/points",
    response_model=DeleteMemoryPointsByMetadataResponse,
    tags=["Vector Memory - Points"],
    prefix="/memory",
)
async def delete_memory_points_by_metadata(
    collection_id: str,
    metadata: Dict = None,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> DeleteMemoryPointsByMetadataResponse:
    """Delete points in memory by filter"""
    try:
        ccat = info.cheshire_cat
        metadata = metadata or {}

        # delete points
        ret = await ccat.vector_memory_handler.delete_tenant_points(collection_id, metadata)

        if source := metadata.get("source"):
            # delete the file with path `metadata["source"]` from the file storage
            ccat.file_manager.remove_file(os.path.join(ccat.agent_key, source))

            # delete the file with path `metadata["source"]/metadata["chat_id"]` from the file storage
            chat_id = metadata.get("chat_id")
            if chat_id:
                ccat.file_manager.remove_file(os.path.join(ccat.agent_key, chat_id, source))

        return DeleteMemoryPointsByMetadataResponse(deleted=ret)
    except Exception as e:
        raise CustomValidationException(f"Failed to delete memory points: {e}")


# GET all the points from a single collection
@endpoint.get(
    "/collections/{collection_id}/points",
    response_model=GetPointsInCollectionResponse,
    tags=["Vector Memory - Points"],
    prefix="/memory",
)
async def get_points_in_collection(
    collection_id: str,
    limit: int = Query(
        default=100,
        description="How many points to return"
    ),
    offset: str = Query(
        default=None,
        description="If provided (or not empty string) - skip points with ids less than given `offset`"
    ),
    metadata: Dict[str, Any] = Depends(create_dict_parser(
        "metadata",
        description="Flat dictionary where each key-value pair represents a filter."
                    "The memory points returned will match the specified metadata criteria."
    )),
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> GetPointsInCollectionResponse:
    """Retrieve all the points from a single collection

    Example
    ----------
    ```
    collection = "declarative"
    res = requests.get(
        f"http://localhost:1865/memory/collections/{collection}/points",
    )
    json = res.json()
    points = json["points"]

    for point in points:
        payload = point["payload"]
        vector = point["vector"]
        print(payload)
        print(vector)
    ```

    Example using offset
    ----------
    ```
    # get all the points with limit 10
    limit = 10
    next_offset = ""
    collection = "declarative"

    while True:
        res = requests.get(
            f"http://localhost:1865/memory/collections/{collection}/points?limit={limit}&offset={next_offset}",
        )
        json = res.json()
        points = json["points"]
        next_offset = json["next_offset"]

        for point in points:
            payload = point["payload"]
            vector = point["vector"]
            print(payload)
            print(vector)

        if next_offset is None:
            break
    ```
    """
    try:
        # if offset is an empty string set to null
        if offset == "":
            offset = None

        points, next_offset = await info.cheshire_cat.vector_memory_handler.get_all_tenant_points(
            collection_name=collection_id, limit=limit, offset=offset, metadata=metadata
        )

        return GetPointsInCollectionResponse(points=points, next_offset=next_offset)
    except Exception as e:
        raise CustomValidationException(f"Failed to get points from collection: {e}")
