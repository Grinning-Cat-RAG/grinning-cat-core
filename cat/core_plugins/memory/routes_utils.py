import asyncio
import time
from typing import Dict, List
from pydantic import BaseModel, Field

from cat import AuthorizedInfo, CheshireCat
from cat.exceptions import CustomNotFoundException


class MemoryPointBase(BaseModel):
    content: str
    metadata: Dict = Field(default_factory=dict)


class MemoryPoint(MemoryPointBase):
    id: str
    vector: List[float]


async def verify_memory_point_existence(cheshire_cat: CheshireCat, collection_id: str, point_id: str) -> None:
    # check if point exists
    points = await cheshire_cat.vector_memory_handler.retrieve_tenant_points(collection_id, [point_id])
    if not points:
        raise CustomNotFoundException("Point does not exist.")


async def upsert_memory_point(
    collection_id: str, point: MemoryPointBase, info: AuthorizedInfo, point_id: str = None
) -> MemoryPoint:
    # embed content
    embedder = await info.lizard.embedder()
    embedding = await asyncio.to_thread(embedder.embed_query, point.content)

    # ensure source is set
    if not point.metadata.get("source"):
        point.metadata["source"] = info.user.id  # this will do also for declarative memory

    # ensure when is set
    if not point.metadata.get("when"):
        point.metadata["when"] = time.time()  # if when is not in the metadata set the current time

    # create point
    qdrant_point = await info.cheshire_cat.vector_memory_handler.add_point_to_tenant(
        collection_name=collection_id,
        content=point.content,
        vector=embedding,
        metadata=point.metadata,
        id_point=point_id,
    )

    return MemoryPoint(
        metadata=qdrant_point.payload["metadata"],  # type: ignore[attr-defined]
        content=qdrant_point.payload["page_content"],  # type: ignore[attr-defined]
        vector=qdrant_point.vector,  # type: ignore[attr-defined]
        id=qdrant_point.id  # type: ignore[attr-defined]
    )
