from typing import List, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict, field_validator

from cat.auth.auth_utils import hash_password
from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, get_base_permissions, check_permissions
from cat.db.cruds import conversations as crud_conversations
from cat.db.cruds import users as crud_users
from cat.exceptions import CustomNotFoundException, CustomValidationException
from cat.routes.routes_utils import validate_permissions as fnc_validate_permissions
from cat.utils import sanitize_permissions

router = APIRouter(tags=["Users"], prefix="/users")


class UserBase(BaseModel):
    username: str = Field(min_length=4)
    permissions: Dict[str, List[str]] = Field(default_factory=get_base_permissions)
    metadata: Dict[str, Any] | None = Field(default=None)

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v):
        return fnc_validate_permissions(v, AuthResource)


class UserCreate(UserBase):
    password: str = Field(min_length=5)
    # no additional fields allowed
    model_config = ConfigDict(extra="forbid")

    def __init__(self, **data):
        super().__init__(**data)
        self.password = hash_password(self.password)


class UserUpdate(UserBase):
    password: str = Field(default=None, min_length=5)
    model_config = ConfigDict(extra="forbid")

    def __init__(self, **data):
        super().__init__(**data)
        if self.password:
            self.password = hash_password(self.password)


class UserResponse(UserBase):
    id: str
    created_at: float
    updated_at: float


@router.post("/", response_model=UserResponse)
async def create_user(
    new_user: UserCreate,
    info: AuthorizedInfo = check_permissions(AuthResource.USERS, AuthPermission.WRITE),
) -> UserResponse:
    agent_id = info.cheshire_cat.agent_key if info.cheshire_cat else info.lizard.agent_key
    new_user.permissions = sanitize_permissions(new_user.permissions, agent_id)  # type: ignore[arg-type]
    if len(new_user.permissions) == 0:
        raise CustomValidationException("User must have at least one permission")

    created_user = await crud_users.create_user(agent_id, new_user.model_dump())  # type: ignore[arg-type]
    if not created_user:
        raise CustomValidationException("Cannot duplicate user")

    return UserResponse(**created_user)


@router.get("/", response_model=List[UserResponse])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    info: AuthorizedInfo = check_permissions(AuthResource.USERS, AuthPermission.READ),
) -> List[UserResponse]:
    agent_id = info.cheshire_cat.agent_key if info.cheshire_cat else info.lizard.agent_key
    users_db = await crud_users.get_users(agent_id, with_timestamps=True, limit=limit, offset=skip)  # type: ignore[arg-type]

    return [UserResponse(**u) for u in users_db.values()]


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(
    user_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.USERS, AuthPermission.READ),
) -> UserResponse:
    agent_id = info.cheshire_cat.agent_key if info.cheshire_cat else info.lizard.agent_key
    user_db = await crud_users.get_user(agent_id, user_id, full=True)  # type: ignore[arg-type]

    if not user_db:
        raise CustomNotFoundException("User not found")
    return UserResponse(**user_db)


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user: UserUpdate,
    info: AuthorizedInfo = check_permissions(AuthResource.USERS, AuthPermission.WRITE),
) -> UserResponse:
    agent_id = info.cheshire_cat.agent_key if info.cheshire_cat else info.lizard.agent_key
    user.permissions = sanitize_permissions(user.permissions, agent_id)  # type: ignore[arg-type]

    stored_user = await crud_users.get_user(agent_id, user_id, full=True)  # type: ignore[arg-type]
    if not stored_user:
        raise CustomNotFoundException("User not found")

    if user.password:
        user.password = hash_password(user.password)

    if len(user.permissions) == 0:
        user.permissions = stored_user["permissions"]

    updated_info = {**stored_user, **user.model_dump(exclude_unset=True)}

    await crud_users.update_user(agent_id, user_id, updated_info)  # type: ignore[arg-type]
    return UserResponse(**updated_info)


@router.delete("/{user_id}", response_model=UserResponse)
async def delete_user(
    user_id: str,
    info: AuthorizedInfo = check_permissions(AuthResource.USERS, AuthPermission.DELETE),
) -> UserResponse:
    agent_id = info.cheshire_cat.agent_key if info.cheshire_cat else info.lizard.agent_key
    if not await crud_users.get_user(agent_id, user_id):  # type: ignore[arg-type]
        raise CustomNotFoundException("User not found")

    # the conversations of the user go with it: files, episodic memories, history and owners (found by the owners,
    # which never expire, unlike the history). Before the user: if this fails, the deletion can be repeated
    if info.cheshire_cat is not None:
        for chat_id in await crud_conversations.get_owned_chats(agent_id, user_id):  # type: ignore[arg-type]
            await info.cheshire_cat.destroy_conversation(chat_id, user_id)
    await crud_conversations.delete_conversations(agent_id, user_id)  # type: ignore[arg-type]

    deleted_user = await crud_users.delete_user(agent_id, user_id)  # type: ignore[arg-type]
    if not deleted_user:
        raise CustomNotFoundException("User not found")
    return UserResponse(**deleted_user)
