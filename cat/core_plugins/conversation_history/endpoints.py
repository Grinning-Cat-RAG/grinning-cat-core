import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field, model_validator

from cat import (
    AuthorizedInfo,
    AuthPermission,
    AuthResource,
    ConversationMessage,
    check_permissions,
    endpoint,
    log,
)
from cat.db.cruds import conversations as crud_conversations
from cat.exceptions import CustomValidationException, CustomNotFoundException
from cat.services.memory.models import VectorMemoryType


class DeleteConversationHistoryResponse(BaseModel):
    deleted: bool


class GetConversationHistoryResponse(BaseModel):
    history: List[ConversationMessage]


class PutConversationAttributes(BaseModel):
    name: str | None = None
    metadata: Dict[str, Any] | None = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metadata(self):
        if self.metadata is None:
            self.metadata = {}
        if self.name is None and not self.metadata:
            raise ValueError("Either name or metadata must be provided")
        return self


class PutConversationResponse(BaseModel):
    changed: bool


class GetConversationsResponse(BaseModel):
    chat_id: str
    name: str
    num_messages: int
    metadata: Dict[str, Any]
    created_at: float | None
    updated_at: float | None


# DELETE conversation
@endpoint.delete(
    "/{chat_id}",
    response_model=DeleteConversationHistoryResponse,
    tags=["Conversation"],
    prefix="/conversations",
)
async def delete_conversation(
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> DeleteConversationHistoryResponse:
    """Delete the specified user's conversation"""
    stray_cat = info.stray_cat
    if not stray_cat:
        log.warning("Trying to change conversation name but no StrayCat found in AuthorizedInfo")
        return DeleteConversationHistoryResponse(deleted=False)

    cat = info.cheshire_cat
    try:
        # delete the files related to the conversation from the storage
        cat.file_manager.remove_folder(os.path.join(cat.agent_key, stray_cat.id))

        # delete the elements of the conversation from the vector memory
        await cat.vector_memory_handler.delete_tenant_points(
            str(VectorMemoryType.EPISODIC), {"chat_id": stray_cat.id},
        )

        # Delete conversation from the database
        await crud_conversations.delete_conversation(stray_cat.agent_key, stray_cat.user.id, stray_cat.id)  # type: ignore[union-attr]

        return DeleteConversationHistoryResponse(deleted=True)
    except Exception as e:
        raise CustomValidationException(f"Failed to delete conversation {stray_cat.id}: {e}")


# GET conversation history from working memory
@endpoint.get(
    "/{chat_id}/history",
    response_model=GetConversationHistoryResponse,
    tags=["Conversation"],
    prefix="/conversations",
)
async def get_conversation_history(
    chat_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> GetConversationHistoryResponse:
    """Get the specified user's conversation history from working memory"""
    stray_cat = await info.cheshire_cat._find_stray_cat(chat_id)
    if stray_cat is None:
        raise CustomNotFoundException(f"Conversation '{chat_id}' not found")
    return GetConversationHistoryResponse(history=stray_cat.working_memory.history)


# GET conversation attributes
@endpoint.get(
    "/{chat_id}",
    response_model=GetConversationsResponse,
    tags=["Conversation"],
    prefix="/conversations",
)
async def get_conversation(
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> GetConversationsResponse:
    """Get the specified user's conversation from working memory"""
    attributes = await crud_conversations.get_conversation_attributes(
        info.cheshire_cat.agent_key, info.user.id, info.stray_cat.id,
    )
    if attributes is None:
        raise CustomNotFoundException("Conversation not found")

    return GetConversationsResponse(**attributes)


# PUT conversation name or metadata change
@endpoint.put(
    "/{chat_id}",
    response_model=PutConversationResponse,
    tags=["Conversation"],
    prefix="/conversations",
)
async def change_attribute_conversation(
    payload: PutConversationAttributes,
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.WRITE),
) -> PutConversationResponse:
    """Insert the specified conversation item into the working memory"""
    if not info.stray_cat:
        log.warning("Trying to change conversation name but no StrayCat found in AuthorizedInfo")
        return PutConversationResponse(changed=False)

    cat = info.stray_cat

    await crud_conversations.set_attributes(
        cat.agent_key, cat.user.id, cat.id, name=payload.name, metadata=payload.metadata,
    )
    return PutConversationResponse(changed=True)


# GET all conversations of the picked user and the picked agent
@endpoint.get(
    "/",
    response_model=List[GetConversationsResponse],
    tags=["Conversation"],
    prefix="/conversations",
)
async def get_conversations(
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> List[GetConversationsResponse]:
    """Get the specified user's conversation history from working memory"""
    agent_id = info.cheshire_cat.agent_key
    user_id = info.user.id
    attributes_list = await crud_conversations.get_conversations_attributes(agent_id, user_id)

    return [GetConversationsResponse(**attributes_item) for attributes_item in attributes_list]


# GET all conversations of the picked agent
@endpoint.get(
    "/",
    response_model=Dict[str, List[GetConversationsResponse]],
    tags=["Conversation"],
    prefix="/agents/conversations",
)
async def get_conversations_by_agent(
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> Dict[str, List[GetConversationsResponse]]:
    """Get the specified user's conversation history from working memory"""
    agent_id = info.cheshire_cat.agent_key
    attributes_list = await crud_conversations.get_conversations_attributes(agent_id, "*", with_user=True)

    grouped = {}
    for attributes in attributes_list:
        user_id = attributes.pop("user_id")
        grouped.setdefault(user_id, []).append(GetConversationsResponse(**attributes))

    return grouped
