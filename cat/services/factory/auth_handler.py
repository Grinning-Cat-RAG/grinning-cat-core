from abc import ABC, abstractmethod
from typing import Type, Literal
from fastapi.requests import HTTPConnection
from pydantic import ConfigDict

from cat.auth.auth_utils import is_jwt, extract_user_info_on_api_key, extract_token_from_request, secure_compare
from cat.auth.tokens import decode_access_token, get_agent_bindings
from cat.auth.permissions import AuthResource, AuthPermission, AuthUserInfo
from cat.db.cruds import users as crud_users
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.env import get_env
from cat.log import log
from cat.services.factory.models import BaseFactoryConfigModel


class BaseAuthHandler(ABC):
    """
    Base class to build custom Auth systems that will live alongside core auth.
    Methods `authorize_user_from_credential`
    MUST be implemented by subclasses.
    """
    # when there is no JWT, user id is passed via `user_id: xxx` header or via websocket path
    # with JWT, the user id is in the token ad has priority
    async def authorize(
        self,
        request: HTTPConnection,
        auth_resource: AuthResource,
        auth_permission: AuthPermission,
        key_id: str,
    ) -> AuthUserInfo | None:
        """
        Authorize a user based on the request and the given resource and permission. This method will extract the token
        from the request and call the appropriate authorization method based on the protocol used. If the token is a JWT,
        it will call `authorize_user_from_jwt`, otherwise it will call `authorize_user_from_key`. If the user is
        authorized, it will return an AuthUserInfo object, otherwise it will return None.

        Args:
            request: the Starlette request to authorize the user on
            auth_resource: the resource to authorize the user on
            auth_permission: the permission to authorize the user on
            key_id: the key ID of the agent to authorize the user with

        Returns:
            An AuthUserInfo object if the user is authorized, None otherwise.
        """
        # get protocol from Starlette request
        protocol = request.scope.get("type")
        if protocol not in ["http", "websocket"]:
            log.error(f"Unsupported protocol: {protocol}")
            return None

        # extract token from request
        token = extract_token_from_request(request)
        if not token:
            return None

        if is_jwt(token):
            # JSON Web Token auth
            return await self.authorize_user_from_jwt(token, auth_resource, auth_permission, key_id)

        # API_KEY auth
        return await self.authorize_user_from_key(request, protocol, token, auth_resource, auth_permission, key_id)

    @abstractmethod
    async def authorize_user_from_jwt(
        self,
        token: str,
        auth_resource: AuthResource,
        auth_permission: AuthPermission,
        key_id: str,
    ) -> AuthUserInfo | None:
        """
        Authorize a user from a JWT token. This method is used to authorize users when they are using a JWT token.

        Args:
            token: the JWT token to authorize the user from
            auth_resource: the resource to authorize the user on
            auth_permission: the permission to authorize the user on
            key_id: the key ID of the agent to authorize the user with

        Returns:
            An AuthUserInfo object if the user is authorized, None otherwise.
        """
        # will raise: NotImplementedError
        pass

    @abstractmethod
    async def authorize_user_from_key(
        self,
        request: HTTPConnection,
        protocol: Literal["http", "websocket"],
        api_key: str,
        auth_resource: AuthResource,
        auth_permission: AuthPermission,
        agent_key: str,
    ) -> AuthUserInfo | None:
        """
        Authorize a user from an API key. This method is used to authorize users when they are not using a JWT token.
        Args:
            request: the Starlette request to authorize the user on
            protocol: the protocol used to authorize the user (either "http" or "websocket")
            api_key: the API key to authorize the user
            auth_resource: the resource to authorize the user on
            auth_permission: the permission to authorize the user on
            agent_key: the key ID of the agent to authorize the user with

        Returns:
            An AuthUserInfo object if the user is authorized, None otherwise.
        """
        # will raise: NotImplementedError
        pass


# Core auth handler, verify token on local idp
class CoreAuthHandler(BaseAuthHandler):
    async def authorize_user_from_jwt(
        self,
        token: str,
        auth_resource: AuthResource,
        auth_permission: AuthPermission,
        key_id: str,
    ) -> AuthUserInfo | None:
        # verify signature, pinned algorithm, exp/nbf/iat, issuer, audience and token type
        payload = decode_access_token(token)
        if payload is None:
            return None
        username = payload["sub"]

        # the token is valid ONLY for the (agent, user_id) pairs whose password was verified at login.
        # Looking the user up by username alone would let a user authenticated on agent A impersonate a different
        # user that happens to have the same username on agent B.
        bindings = get_agent_bindings(payload)
        user = None
        if bound_user_id := bindings.get(key_id):
            # get user from DB: it must still exist and keep the same username
            user = await crud_users.get_user(key_id, bound_user_id)
        elif key_id != DEFAULT_SYSTEM_KEY and (system_user_id := bindings.get(DEFAULT_SYSTEM_KEY)):
            # a verified *system* user (super-admin, who can manage every agent anyway) keeps the previous
            # behaviour: they act as the same-username user of the requested agent
            system_user = await crud_users.get_user(DEFAULT_SYSTEM_KEY, system_user_id)
            if system_user and system_user.get("username") == username:
                user = await crud_users.get_user_by_username(key_id, username)

        if not user or user.get("username") != username:
            # do not pass
            return None

        ar = str(auth_resource)
        ap = str(auth_permission)

        if ar not in user["permissions"].keys() or ap not in user["permissions"][ar]:
            # do not pass
            return AuthUserInfo(
                id=user["id"],
                name=username,
                extra=user,
            )

        return AuthUserInfo(
            id=user["id"],
            name=username,
            permissions=user["permissions"],
            extra=user,
        )

    async def authorize_user_from_key(
        self,
        request: HTTPConnection,
        protocol: Literal["http", "websocket"],
        api_key: str,
        auth_resource: AuthResource,
        auth_permission: AuthPermission,
        agent_key: str,
    ) -> AuthUserInfo | None:
        if not (current_api_key := get_env("CAT_API_KEY")):
            return None
        if not secure_compare(api_key, current_api_key):
            return None

        request_user_id = request.headers.get("X-User-ID") if protocol == "http" else request.query_params.get("user_id")
        if not (user_info := await extract_user_info_on_api_key(agent_key, request_user_id)):
            return None

        # No match -> deny access
        if not user_info.permissions:
            return AuthUserInfo(
                id=user_info.user_id,
                name=user_info.username,
            )

        return AuthUserInfo(
            id=user_info.user_id,
            name=user_info.username,
            permissions=user_info.permissions
        )


class AuthHandlerConfig(BaseFactoryConfigModel, ABC):
    @classmethod
    def base_class(cls) -> Type[BaseAuthHandler]:
        return BaseAuthHandler

    @classmethod
    @abstractmethod
    def pyclass(cls) -> Type[BaseAuthHandler]:
        pass


class CoreAuthConfig(AuthHandlerConfig):
    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Standalone Core Auth Handler",
            "description": "Delegate auth to Cat core, without any additional auth systems. "
            "Do not change this if you don't know what you are doing!",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CoreAuthHandler]:
        return CoreAuthHandler
