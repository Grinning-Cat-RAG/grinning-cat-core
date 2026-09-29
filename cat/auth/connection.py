from abc import ABC, abstractmethod
from typing import Self

from fastapi import Request, WebSocket, WebSocketException
from fastapi.requests import HTTPConnection
from pydantic import BaseModel, ConfigDict, SkipValidation, model_validator

from cat.auth.auth_utils import (
    extract_agent_id_from_request,
    extract_chat_id_from_request,
)
from cat.auth.permissions import AuthPermission, AuthResource, AuthUserInfo
from cat.db.cruds import conversations as crud_conversations
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.exceptions import (
    CustomForbiddenException,
    CustomNotFoundException,
    CustomUnauthorizedException,
    ManagementModeException,
    UnknownAgentException,
)
from cat.looking_glass import BillTheLizard, CheshireCat, StrayCat


class AuthorizedInfo(BaseModel):
    agent_id: str | None
    lizard: SkipValidation[BillTheLizard]
    cheshire_cat: CheshireCat | None = None
    user: AuthUserInfo
    stray_cat: StrayCat | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def check_cheshire_cat(self) -> Self:
        if self.agent_id == DEFAULT_SYSTEM_KEY:
            return self

        if self.agent_id is not None and self.cheshire_cat is None:
            raise ValueError("CheshireCat cannot be None for non-system agents.")

        return self


class ConnectionAuth(ABC):
    def __init__(self, resource: AuthResource, permission: AuthPermission, is_chat: bool = False):
        self.resource = resource
        self.permission = permission
        self.is_chat = is_chat

    async def __call__(self, connection: HTTPConnection) -> AuthorizedInfo:
        lizard: BillTheLizard = connection.app.state.lizard

        agent_id = extract_agent_id_from_request(connection)
        try:
            ccat = await lizard.get_cheshire_cat(agent_id) if agent_id else None
        except UnknownAgentException:
            # do not disclose which agents exist: same answer as for invalid credentials
            self._not_authorized(connection)

        stray_cat = None

        # the path of the request, matched with the paths of the endpoints (e.g. /items/{id}), and its method (none for
        # a websocket)
        url_path = connection.url.path
        method = connection.scope.get("method")
        is_custom_endpoint = lizard.is_custom_endpoint(url_path, method=method)
        is_triggered_by_cat = ccat is not None
        has_cat_custom_endpoint = ccat.has_custom_endpoint(url_path, method=method) if ccat is not None else False

        # if the request comes from a custom endpoint, and it is not available in the picked CheshireCat, block it and
        # return a 404-HTTP error
        if is_custom_endpoint and is_triggered_by_cat and not has_cat_custom_endpoint:
            raise CustomNotFoundException("Not Found")

        # always try core auth first (less costly, in general)
        user = None
        is_system_user = False
        if not self.is_chat:
            # is that an admin able to manage agents?
            user = await lizard.core_auth_handler.authorize(
                connection,
                self.resource,
                self.permission,
                lizard.agent_key,
            )
            is_system_user = user is not None

        # fallback to agent-specific auth if needed and available
        if not user and ccat is not None:
            user = await ccat.custom_auth_handler.authorize(
                connection,
                self.resource,
                self.permission,
                ccat.agent_key,
            )

        # if no user was obtained, raise an exception
        if not user:
            self._not_authorized(connection)

        # if user has no permissions, raise forbidden exception
        if user and user.permissions is None:
            self._not_allowed(connection)

        # management gate: let plugins decide whether this request is allowed
        if "auth_request" in lizard.plugin_manager.hooks:
            try:
                auth_res = await lizard.plugin_manager.execute_hook(
                    "auth_request", user, agent_id, connection, caller=lizard
                )
                if isinstance(auth_res, str) and auth_res:
                    if connection.scope.get("type") == "websocket":
                        raise WebSocketException(code=1008, reason=auth_res)
                    raise CustomForbiddenException(auth_res)
            except ManagementModeException as e:
                # mgmt_message plugin raises a dedicated exception; translate to
                # the protocol-specific gate error (same behaviour as the str return)
                if connection.scope.get("type") == "websocket":
                    raise WebSocketException(code=1008, reason=str(e)) from e
                raise

        if ccat is not None and (chat_id := extract_chat_id_from_request(connection)):
            # the chat id is chosen by the client: a conversation belongs to the first user of its id, and the files,
            # the episodic memories and the deletion of the conversation are identified by it. The system users
            # (administrators of every agent) can access any conversation, without owning it
            if not is_system_user and await crud_conversations.claim_conversation(ccat.agent_key, chat_id, user.id) != user.id:  # type: ignore[union-attr]
                self._not_authorized(connection)
            stray_cat = await StrayCat.from_cat(user_data=user, cat=ccat, stray_id=chat_id)  # type: ignore[arg-type]

        return AuthorizedInfo(lizard=lizard, cheshire_cat=ccat, user=user, stray_cat=stray_cat, agent_id=agent_id)  # type: ignore[arg-type]

    @abstractmethod
    def _not_authorized(self, connection: HTTPConnection, **kwargs):
        pass

    @abstractmethod
    def _not_allowed(self, connection: HTTPConnection, **kwargs):
        pass


class HTTPAuth(ConnectionAuth):
    def _not_allowed(self, connection: Request, **kwargs):
        raise CustomForbiddenException("Forbidden")

    def _not_authorized(self, connection: Request, **kwargs):
        raise CustomUnauthorizedException("Unauthorized")


class WebSocketAuth(ConnectionAuth):
    def _not_allowed(self, connection: WebSocket, **kwargs):
        raise WebSocketException(code=1004, reason="Invalid Credentials")

    def _not_authorized(self, connection: WebSocket, **kwargs):
        raise WebSocketException(code=1008, reason="Unauthorized")
