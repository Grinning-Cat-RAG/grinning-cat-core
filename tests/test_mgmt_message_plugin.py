"""Integration tests for the ``mgmt_message`` core plugin.

Coverage:
(a) storage round-trip inside the global ``system:agent`` settings list
    (entry ``mgmt_message`` with ``category="mgmt_message"`` — same mechanism
    as the system-level embedder configuration, NOT ``system:plugins:*``),
(b) the management gate: the real ``auth_request`` hook denies unprivileged
    principals when ``management_active`` is true by raising
    ``ManagementModeException`` (a ``CustomForbiddenException``), translated
    by ``ConnectionAuth`` into ``CustomForbiddenException`` (HTTP) /
    ``WebSocketException(code=1008)`` (WS),
(c) ``GET /plugins/settings/mgmt_message`` returns the 4 settings in normal
    mode (the RITA read path),
(d) ``management_active=false`` is a no-op for any principal.

Uses the ``tests/conftest.py`` fixtures: Redis db=1 (isolated), agent
``"agent_test"``, mocked Qdrant, synchronous background tasks. No live
LLM/embedder required.
"""

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
from cat.core_plugins.mgmt_message.plugin import auth_request
from cat.core_plugins.mgmt_message.settings import (
    _MGMT_SETTING_CATEGORY,
    _MGMT_SETTING_NAME,
)
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_AGENT_KEY, DEFAULT_SYSTEM_KEY, get_sync_db
from cat.db.models import Setting
from cat.exceptions import CustomForbiddenException, ManagementModeException
from tests.utils import get_client_admin_headers

# the plugin id (folder name) and the global entry where its settings live
PLUGIN_ID = "mgmt_message"
MGMT_SYSTEM_AGENT_KEY = f"{DEFAULT_SYSTEM_KEY}:{DEFAULT_AGENT_KEY}"


def _make_user():
    """A normal chat user: no SYSTEM permission."""
    return AuthUserInfo(id="user", name="User", permissions=get_base_permissions())


def _make_admin_user():
    return AuthUserInfo(
        id="admin",
        name="Admin",
        permissions={str(AuthResource.SYSTEM): [str(AuthPermission.WRITE)]},
    )


async def _store(payload: dict):
    await crud_settings.upsert_setting_by_name(
        DEFAULT_SYSTEM_KEY,
        Setting(name=_MGMT_SETTING_NAME, value=payload, category=_MGMT_SETTING_CATEGORY),
    )


async def _cleanup():
    db = get_sync_db()
    db.json().delete(MGMT_SYSTEM_AGENT_KEY, f'$[?(@.name=="{_MGMT_SETTING_NAME}")]')


# ---------------------------------------------------------------------------
# (a) storage round-trip in the global system:agent settings list
# ---------------------------------------------------------------------------

async def test_storage_round_trip_in_system_agent():
    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": True,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }

    await _store(payload)

    # stored inside system:agent under the mgmt_message entry
    db = get_sync_db()
    found = db.json().get(MGMT_SYSTEM_AGENT_KEY, f'$[?(@.name=="{_MGMT_SETTING_NAME}")]')
    assert isinstance(found, list) and found
    entry = found[0]
    assert isinstance(entry, dict)
    assert entry["value"] == payload
    assert entry["category"] == _MGMT_SETTING_CATEGORY

    # real async read path used by the hook
    loaded = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, _MGMT_SETTING_NAME)
    assert loaded["value"] == payload

    # no system:plugins:* key is written
    assert db.keys("system:plugins:*") == []
    # and no per-agent keys either
    agent_keys = db.keys("agents:*")
    assert agent_keys == []

    await _cleanup()


async def test_legacy_key_migrated_on_load():
    # seed the legacy system:plugins:mgmt_message key (old mechanism)
    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": True,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }
    db = get_sync_db()
    db.json().set(f"{DEFAULT_SYSTEM_KEY}:plugins:{PLUGIN_ID}", "$", payload)

    from cat.core_plugins.mgmt_message.settings import load_settings

    loaded = await load_settings.function(PLUGIN_ID, DEFAULT_SYSTEM_KEY)
    assert loaded == payload

    # migrated into system:agent, legacy key removed
    assert db.keys("system:plugins:*") == []
    found = db.json().get(MGMT_SYSTEM_AGENT_KEY, f'$[?(@.name=="{_MGMT_SETTING_NAME}")]')
    assert isinstance(found, list) and found
    assert found[0]["value"] == payload

    await _cleanup()


# ---------------------------------------------------------------------------
# (b) management gate: the real auth_request hook
# ---------------------------------------------------------------------------

async def test_auth_request_denies_unprivileged_when_active():
    message = "Sistema in manutenzione"
    await _store({"management_message": message, "management_active": True})

    with pytest.raises(ManagementModeException) as exc_info:
        await auth_request.function(_make_user(), "system", None)

    assert exc_info.value.args[0] == message
    assert isinstance(exc_info.value, CustomForbiddenException)  # still a 403
    await _cleanup()


async def test_auth_request_allows_system_principal_when_active():
    await _store({"management_message": "Sistema in manutenzione", "management_active": True})

    result = await auth_request.function(_make_admin_user(), "system", None)

    assert result is None
    await _cleanup()


# ---------------------------------------------------------------------------
# (b) management gate E2E: real hook wired through ConnectionAuth
# ---------------------------------------------------------------------------

class _RealHookPluginManager:
    """Plugin-manager stand-in that executes the real ``auth_request`` hook."""

    def __init__(self, hooks):
        self.hooks = hooks

    async def execute_hook(self, hook_name, *args, **kwargs):
        tea_cup = args[0]
        for hook in self.hooks[hook_name]:
            result = await hook.function(tea_cup, *args[1:], **kwargs)
            if result is not None:
                tea_cup = result
        return tea_cup


class _FakeCoreAuthHandler:
    def __init__(self, user):
        self._user = user

    async def authorize(self, connection, resource, permission, agent_key):
        return self._user


class _FakeLizard:
    """Minimal stand-in for BillTheLizard (same shape as test_mgmt_hook_gateway)."""

    def __init__(self, plugin_manager, user):
        self.plugin_manager = plugin_manager
        self.agent_key = DEFAULT_SYSTEM_KEY
        self.core_auth_handler = _FakeCoreAuthHandler(user)

    async def get_cheshire_cat(self, agent_id):
        return None

    def is_custom_endpoint(self, url_path):
        return False


class _FakeConnection:
    def __init__(self, scope_type="http"):
        self.scope = {"type": scope_type}
        self.url = SimpleNamespace(path="/test")
        self.path_params = {}
        self.query_params = {}
        self.headers = {}
        self.app = SimpleNamespace(state=SimpleNamespace(lizard=None))


def _make_connection(lizard, scope_type="http"):
    connection = _FakeConnection(scope_type=scope_type)
    connection.app.state.lizard = lizard
    return connection


def _make_lizard_with_real_hook(user):
    return _FakeLizard(_RealHookPluginManager({"auth_request": [auth_request]}), user)


async def test_http_gateway_denial_with_real_hook(monkeypatch):
    message = "Sistema in manutenzione"

    async def fake_get_setting_by_name(key_id, name):
        return {"value": {"management_active": True, "management_message": message}}

    monkeypatch.setattr(crud_settings, "get_setting_by_name", fake_get_setting_by_name)

    lizard = _make_lizard_with_real_hook(_make_user())
    connection = _make_connection(lizard, scope_type="http")

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    with pytest.raises(CustomForbiddenException) as exc_info:
        await auth(connection)

    assert exc_info.value.args[0] == message


async def test_websocket_gateway_denial_with_real_hook(monkeypatch):
    message = "Sistema in manutenzione"

    async def fake_get_setting_by_name(key_id, name):
        return {"value": {"management_active": True, "management_message": message}}

    monkeypatch.setattr(crud_settings, "get_setting_by_name", fake_get_setting_by_name)

    lizard = _make_lizard_with_real_hook(_make_user())
    connection = _make_connection(lizard, scope_type="websocket")

    auth = WebSocketAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    with pytest.raises(WebSocketException) as exc_info:
        await auth(connection)

    assert exc_info.value.code == 1008
    assert exc_info.value.reason == message


async def test_http_gateway_allows_system_principal_with_real_hook(monkeypatch):
    async def fake_get_setting_by_name(key_id, name):
        return {"value": {"management_active": True, "management_message": "Sistema in manutenzione"}}

    monkeypatch.setattr(crud_settings, "get_setting_by_name", fake_get_setting_by_name)

    admin = _make_admin_user()
    lizard = _make_lizard_with_real_hook(admin)
    connection = _make_connection(lizard, scope_type="http")

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    result = await auth(connection)

    assert isinstance(result, AuthorizedInfo)
    assert result.user is admin


# ---------------------------------------------------------------------------
# (c) normal-mode / RITA read: GET /mgmt_message/settings (SYSTEM READ, moved
# from the old core /plugins/system/settings route into the plugin)
# ---------------------------------------------------------------------------

async def test_get_plugin_settings_normal_mode(secure_client, secure_client_headers, cheshire_cat):
    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": False,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }
    await _store(payload)

    # admin (system agent) reads the global settings — this is the RITA read path
    response = await secure_client.get("/mgmt_message/settings", headers=secure_client_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "mgmt_message"
    assert body["value"] == payload
    assert set(body["value"].keys()) == {
        "management_message",
        "management_active",
        "global_message",
        "show_global_msg",
    }
    # the schema exposes the same 4 fields
    assert set(body["scheme"]["properties"].keys()) == {
        "management_message",
        "management_active",
        "global_message",
        "show_global_msg",
    }

    await _cleanup()


# the plugin's own settings route (same pattern as the embedder upsert)
# persists the plugin settings into system:agent — the MyADMIN Management mode
# save path
async def test_put_mgmt_message_settings(client, secure_client, secure_client_headers, cheshire_cat):
    # activate the plugin so its settings are loaded via the plugin overrides
    await secure_client.put("/plugins/toggle/mgmt_message", headers=secure_client_headers)

    payload = {
        "management_message": "Nuovo messaggio",
        "management_active": True,
        "global_message": "Nuovo avviso",
        "show_global_msg": True,
    }
    # system-level write: authenticate as the admin (SYSTEM WRITE)
    admin_headers = await get_client_admin_headers(client)
    response = await secure_client.put("/mgmt_message/settings", headers=admin_headers, json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "mgmt_message"
    assert body["value"] == payload

    # persisted inside system:agent under the mgmt_message entry
    db = get_sync_db()
    found = db.json().get(MGMT_SYSTEM_AGENT_KEY, f'$[?(@.name=="{_MGMT_SETTING_NAME}")]')
    assert isinstance(found, list) and found
    assert found[0]["value"] == payload
    assert found[0]["category"] == _MGMT_SETTING_CATEGORY

    # the value is served from system:agent
    loaded = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, _MGMT_SETTING_NAME)
    assert loaded["value"] == payload

    await _cleanup()


# ---------------------------------------------------------------------------
# (c2) the plugin settings routes must NOT be public: read and write require
# SYSTEM permission, exactly like the old core /plugins/system/settings routes
# ---------------------------------------------------------------------------

async def test_settings_endpoints_require_system_permission(client, secure_client, secure_client_headers, cheshire_cat):
    # activate the plugin so its custom endpoints are registered
    await secure_client.put("/plugins/toggle/mgmt_message", headers=secure_client_headers)

    unauthenticated_put = await client.put("/mgmt_message/settings", json={"management_active": True})
    assert unauthenticated_put.status_code == 401

    unauthenticated_get = await client.get("/mgmt_message/settings")
    assert unauthenticated_get.status_code == 401

    await _cleanup()


# ---------------------------------------------------------------------------
# (d) management_active=false -> no-op
# ---------------------------------------------------------------------------

async def test_auth_request_noop_when_inactive(monkeypatch):
    async def fake_get_setting_by_name(key_id, name):
        return {"value": {"management_active": False, "management_message": "Sistema in manutenzione"}}

    monkeypatch.setattr(crud_settings, "get_setting_by_name", fake_get_setting_by_name)

    result = await auth_request.function(_make_user(), "local", None)

    assert result is None


async def test_auth_request_noop_when_no_setting(monkeypatch):
    async def fake_get_setting_by_name(key_id, name):
        return None

    monkeypatch.setattr(crud_settings, "get_setting_by_name", fake_get_setting_by_name)

    result = await auth_request.function(_make_user(), "local", None)

    assert result is None


# ---------------------------------------------------------------------------
# (e) public global_message endpoint
# ---------------------------------------------------------------------------

async def test_public_global_message_endpoint_no_auth(client, secure_client, secure_client_headers, cheshire_cat):
    # activate the plugin so its custom endpoint is registered
    await secure_client.put("/plugins/toggle/mgmt_message", headers=secure_client_headers)

    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": False,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }
    await _store(payload)

    # unauthenticated client (no headers) must still reach the endpoint
    response = await client.get("/mgmt_message/global_message")

    assert response.status_code == 200
    assert response.json() == payload

    await _cleanup()