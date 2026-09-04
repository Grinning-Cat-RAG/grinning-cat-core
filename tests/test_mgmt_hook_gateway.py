from types import SimpleNamespace

import pytest
from fastapi import WebSocketException

from cat.auth.connection import AuthorizedInfo, HTTPAuth, WebSocketAuth
from cat.auth.permissions import (
    AuthPermission,
    AuthResource,
    AuthUserInfo,
    get_base_permissions,
)
from cat.exceptions import CustomForbiddenException


class _RaiseIfCalled:
    """Sentinel: execute_hook must never be invoked when the hook is absent."""


class FakePluginManager:
    """Minimal stand-in for MadHatter's plugin manager."""

    def __init__(self, hooks=None, execute_hook_result=None):
        self.hooks = hooks or {}
        self._execute_hook_result = execute_hook_result

    async def execute_hook(self, hook_name, *args, **kwargs):
        if self._execute_hook_result is _RaiseIfCalled:
            raise AssertionError("execute_hook must not be called when auth_request is absent")
        return self._execute_hook_result


class FakeCoreAuthHandler:
    def __init__(self, user):
        self._user = user

    async def authorize(self, connection, resource, permission, agent_key):
        return self._user


class FakeLizard:
    """Minimal stand-in for BillTheLizard."""

    def __init__(self, plugin_manager, user):
        self.plugin_manager = plugin_manager
        self.agent_key = "system"
        self.core_auth_handler = FakeCoreAuthHandler(user)

    async def get_cheshire_cat(self, agent_id):
        return None

    def is_custom_endpoint(self, url_path):
        return False


class FakeConnection:
    def __init__(self, scope_type="http"):
        self.scope = {"type": scope_type}
        self.url = SimpleNamespace(path="/test")
        self.path_params = {}
        self.query_params = {}
        self.headers = {}
        self.app = SimpleNamespace(state=SimpleNamespace(lizard=None))


def _make_user():
    return AuthUserInfo(id="user", name="User", permissions=get_base_permissions())


def _make_connection(lizard, scope_type="http"):
    connection = FakeConnection(scope_type=scope_type)
    connection.app.state.lizard = lizard
    return connection


def _make_lizard(execute_hook_result, user=None):
    user = user or _make_user()
    plugin_manager = FakePluginManager(
        hooks={"auth_request": [object()]},
        execute_hook_result=execute_hook_result,
    )
    return FakeLizard(plugin_manager, user)


# (a) no auth_request hook present -> request passes, hook never executed
async def test_no_auth_request_hook_request_passes():
    user = _make_user()
    lizard = FakeLizard(FakePluginManager(hooks={}, execute_hook_result=_RaiseIfCalled), user)
    connection = _make_connection(lizard)

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    result = await auth(connection)

    assert isinstance(result, AuthorizedInfo)
    assert result.user is user


# (b) auth_request returns a denial string on HTTP -> CustomForbiddenException
async def test_auth_request_denial_http():
    message = "Sistema in manutenzione"
    lizard = _make_lizard(execute_hook_result=message)
    connection = _make_connection(lizard, scope_type="http")

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    with pytest.raises(CustomForbiddenException) as exc_info:
        await auth(connection)

    assert exc_info.value.args[0] == message


# (c) auth_request returns a denial string on WebSocket -> WebSocketException 1008
async def test_auth_request_denial_websocket():
    message = "Sistema in manutenzione"
    lizard = _make_lizard(execute_hook_result=message)
    connection = _make_connection(lizard, scope_type="websocket")

    auth = WebSocketAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    with pytest.raises(WebSocketException) as exc_info:
        await auth(connection)

    assert exc_info.value.code == 1008
    assert exc_info.value.reason == message


# (d) auth_request returns None -> request passes normally
async def test_auth_request_none_passes():
    user = _make_user()
    lizard = _make_lizard(execute_hook_result=None, user=user)
    connection = _make_connection(lizard)

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    result = await auth(connection)

    assert isinstance(result, AuthorizedInfo)
    assert result.user is user


# non-str return (e.g. the deepcopied user tea_cup) -> request passes normally
async def test_auth_request_non_str_passes():
    user = _make_user()
    lizard = _make_lizard(execute_hook_result=user, user=user)
    connection = _make_connection(lizard)

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    result = await auth(connection)

    assert isinstance(result, AuthorizedInfo)
    assert result.user is user