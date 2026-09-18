"""Integration tests for the ``mgmt_message`` core plugin.

Coverage:
(a) storage round-trip on the system agent's plugin key
    (``system:plugins:mgmt_message``, the standard plugin storage every other
    system plugin uses — NOT the ``system:agent`` settings list),
(a2) the plugin belongs to the system agent alone: no agent carries it among
    its active plugins, and no ``agents:<id>:plugins:mgmt_message`` key is
    written,
(b) the management gate: the real ``auth_request`` hook denies unprivileged
    principals when ``management_active`` is true by raising
    ``ManagementModeException`` (a ``CustomForbiddenException``), translated
    by ``ConnectionAuth`` into ``CustomForbiddenException`` (HTTP) /
    ``WebSocketException(code=1008)`` (WS),
(c) the read paths return the 4 settings in normal mode: the core's
    ``GET /plugins/system/settings/mgmt_message`` (the RITA read path) and the
    plugin's own ``GET /mgmt_message/settings``, which also serves the schema,
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
from cat.core_plugins.mgmt_message.settings import MGMT_SETTING_NAME, PluginSettings
from cat.db.cruds import plugins as crud_plugins
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_AGENT_KEY, DEFAULT_SYSTEM_KEY, get_sync_db
from cat.db.models import Setting
from cat.exceptions import CustomForbiddenException, ManagementModeException
from tests.utils import get_client_admin_headers

# the plugin id (folder name) and the key its settings live on
PLUGIN_ID = MGMT_SETTING_NAME
MGMT_PLUGIN_KEY = crud_plugins.format_key(DEFAULT_SYSTEM_KEY, PLUGIN_ID)
# the settings list this plugin used to write into, and must not touch any more
SYSTEM_AGENT_KEY = f"{DEFAULT_SYSTEM_KEY}:{DEFAULT_AGENT_KEY}"


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
    await crud_plugins.set_setting(DEFAULT_SYSTEM_KEY, PLUGIN_ID, payload)


async def _cleanup():
    db = get_sync_db()
    db.delete(MGMT_PLUGIN_KEY)
    # nothing must be left behind in the settings list either
    db.json().delete(SYSTEM_AGENT_KEY, f'$[?(@.name=="{PLUGIN_ID}")]')


# ---------------------------------------------------------------------------
# (a) storage: the plugin uses the standard load/save on the system agent
# ---------------------------------------------------------------------------

async def test_plugin_load_and_save_use_the_system_plugin_key(lizard):
    """The plugin used to override load/save to persist into the ``system:agent``
    settings list, which left the standard ``system:plugins:mgmt_message`` key
    holding stale defaults — two sources of truth for the same settings. It now
    stores them the standard way, like every other system plugin."""
    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": True,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }
    plugin = lizard.plugin_manager.plugins[PLUGIN_ID]

    await plugin.save_settings(payload, DEFAULT_SYSTEM_KEY)

    db = get_sync_db()
    stored = db.json().get(MGMT_PLUGIN_KEY)
    if isinstance(stored, list):
        stored = stored[0]
    assert stored == payload
    # and nothing is left in the settings list the plugin used to write into
    assert db.json().get(SYSTEM_AGENT_KEY, f'$[?(@.name=="{PLUGIN_ID}")]') in (None, [])

    assert await plugin.load_settings(DEFAULT_SYSTEM_KEY) == payload
    assert await crud_plugins.get_setting(DEFAULT_SYSTEM_KEY, PLUGIN_ID) == payload

    await _cleanup()


# ---------------------------------------------------------------------------
# (a2) the plugin belongs to the system agent alone
# ---------------------------------------------------------------------------

async def test_agent_plugin_manager_does_not_carry_the_plugin(lizard, cheshire_cat):
    """The settings are global and the hooks run on BillTheLizard's manager, so
    an agent must not list the plugin among its own — which is also what kept
    writing an ``agents:<id>:plugins:mgmt_message`` key holding stale defaults."""
    assert PLUGIN_ID in await lizard.plugin_manager.load_active_plugins_ids_from_db()

    agent_plugin_manager = cheshire_cat.plugin_manager
    assert PLUGIN_ID not in await agent_plugin_manager.load_active_plugins_ids_from_db()
    assert PLUGIN_ID not in agent_plugin_manager.active_plugins
    assert PLUGIN_ID not in agent_plugin_manager.plugins

    # ...and no per-agent settings key is written for it
    db = get_sync_db()
    assert db.json().get(crud_plugins.format_key(cheshire_cat.agent_key, PLUGIN_ID)) is None
    # an agent-scoped plugin still gets one, so the exclusion is not a blanket one
    assert db.json().get(crud_plugins.format_key(cheshire_cat.agent_key, "memory")) is not None


async def test_stored_active_plugins_of_an_agent_drop_the_plugin(lizard, cheshire_cat):
    """An agent created before this rule keeps the plugin in its stored list:
    loading the list must drop it there too, not only for new agents."""
    agent_plugin_manager = cheshire_cat.plugin_manager
    stored = await agent_plugin_manager.load_active_plugins_ids_from_db()
    await crud_settings.upsert_setting_by_name(
        cheshire_cat.agent_key, Setting(name="active_plugins", value=stored + [PLUGIN_ID])
    )

    assert PLUGIN_ID not in await agent_plugin_manager.load_active_plugins_ids_from_db()


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

    async def fake_get_setting(key_id, plugin_id):
        return {"management_active": True, "management_message": message}

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    lizard = _make_lizard_with_real_hook(_make_user())
    connection = _make_connection(lizard, scope_type="http")

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    with pytest.raises(CustomForbiddenException) as exc_info:
        await auth(connection)

    assert exc_info.value.args[0] == message


async def test_websocket_gateway_denial_with_real_hook(monkeypatch):
    message = "Sistema in manutenzione"

    async def fake_get_setting(key_id, plugin_id):
        return {"management_active": True, "management_message": message}

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    lizard = _make_lizard_with_real_hook(_make_user())
    connection = _make_connection(lizard, scope_type="websocket")

    auth = WebSocketAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    with pytest.raises(WebSocketException) as exc_info:
        await auth(connection)

    assert exc_info.value.code == 1008
    assert exc_info.value.reason == message


async def test_http_gateway_allows_system_principal_with_real_hook(monkeypatch):
    async def fake_get_setting(key_id, plugin_id):
        return {"management_active": True, "management_message": "Sistema in manutenzione"}

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    admin = _make_admin_user()
    lizard = _make_lizard_with_real_hook(admin)
    connection = _make_connection(lizard, scope_type="http")

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    result = await auth(connection)

    assert isinstance(result, AuthorizedInfo)
    assert result.user is admin


# ---------------------------------------------------------------------------
# (c) normal-mode reads: the core's GET /plugins/system/settings/mgmt_message
# (SYSTEM READ, the RITA path) and the plugin's own GET /mgmt_message/settings
# ---------------------------------------------------------------------------

async def test_get_plugin_settings_normal_mode(secure_client, secure_client_headers, cheshire_cat):
    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": False,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }
    await _store(payload)

    # admin (system agent) reads the global settings — this is the RITA read
    # path, served by the core's system plugin-settings route.
    response = await secure_client.get(
        "/plugins/system/settings/mgmt_message", headers=secure_client_headers
    )

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


async def test_get_plugin_settings_serves_model_defaults_when_not_stored(
    secure_client, secure_client_headers, cheshire_cat
):
    """Nothing stored: the standard read path falls back to the settings model."""
    await _cleanup()

    response = await secure_client.get(
        "/plugins/system/settings/mgmt_message", headers=secure_client_headers
    )

    assert response.status_code == 200
    assert response.json()["value"] == PluginSettings().model_dump()


async def test_get_mgmt_message_settings_route(client, secure_client, cheshire_cat):
    """The plugin's own read route: same content as the core system-level one,
    plus the settings schema."""
    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": False,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }
    await _store(payload)

    admin_headers = await get_client_admin_headers(client)
    response = await secure_client.get("/mgmt_message/settings", headers=admin_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == PLUGIN_ID
    assert body["value"] == payload
    assert body["scheme"] == PluginSettings.model_json_schema()

    await _cleanup()


async def test_get_mgmt_message_settings_route_serves_defaults_when_not_stored(
    client, secure_client, cheshire_cat
):
    await _cleanup()

    admin_headers = await get_client_admin_headers(client)
    response = await secure_client.get("/mgmt_message/settings", headers=admin_headers)

    assert response.status_code == 200
    assert response.json()["value"] == PluginSettings().model_dump()


# the plugin's own settings route persists the plugin settings on the system
# agent's plugin key — the MyADMIN Management mode save path
async def test_put_mgmt_message_settings(client, secure_client, secure_client_headers, cheshire_cat):
    # mgmt_message is a system plugin: always active, no toggle needed
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

    # persisted on system:plugins:mgmt_message, not in the settings list
    db = get_sync_db()
    found = db.json().get(MGMT_PLUGIN_KEY)
    if isinstance(found, list):
        found = found[0]
    assert found == payload
    assert db.json().get(SYSTEM_AGENT_KEY, f'$[?(@.name=="{PLUGIN_ID}")]') in (None, [])

    loaded = await crud_plugins.get_setting(DEFAULT_SYSTEM_KEY, PLUGIN_ID)
    assert loaded == payload

    await _cleanup()


# ---------------------------------------------------------------------------
# (c2) the settings routes must NOT be public: the plugin's own read and write
# routes and the core's system read route all require SYSTEM permission
# ---------------------------------------------------------------------------

async def test_settings_endpoints_require_system_permission(client, secure_client, secure_client_headers, cheshire_cat):
    # mgmt_message is a system plugin: always active, its endpoints are registered
    unauthenticated_put = await client.put("/mgmt_message/settings", json={"management_active": True})
    assert unauthenticated_put.status_code == 401

    # the plugin's own read route is authenticated too, unlike /global_message
    unauthenticated_get = await client.get("/mgmt_message/settings")
    assert unauthenticated_get.status_code == 401

    # the settings read is served by the system plugin route, which is protected too
    unauthenticated_get = await client.get("/plugins/system/settings/mgmt_message")
    assert unauthenticated_get.status_code == 401

    await _cleanup()


# ---------------------------------------------------------------------------
# (d) management_active=false -> no-op
# ---------------------------------------------------------------------------

# The gate allows by default, so asserting only "it allowed" would pass even if
# the hook read a store the fake does not patch: record the read and assert the
# gate looked the settings up where they actually live.

async def test_auth_request_noop_when_inactive(monkeypatch):
    reads = []

    async def fake_get_setting(key_id, plugin_id):
        reads.append((key_id, plugin_id))
        return {"management_active": False, "management_message": "Sistema in manutenzione"}

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    result = await auth_request.function(_make_user(), "local", None)

    assert result is None
    assert reads == [(DEFAULT_SYSTEM_KEY, PLUGIN_ID)]


async def test_auth_request_noop_when_no_setting(monkeypatch):
    reads = []

    async def fake_get_setting(key_id, plugin_id):
        reads.append((key_id, plugin_id))
        return None

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    result = await auth_request.function(_make_user(), "local", None)

    assert result is None
    assert reads == [(DEFAULT_SYSTEM_KEY, PLUGIN_ID)]


# ---------------------------------------------------------------------------
# (e) public global_message endpoint
# ---------------------------------------------------------------------------

async def test_public_global_message_endpoint_no_auth(client, secure_client, secure_client_headers, cheshire_cat):
    # mgmt_message is a system plugin: always active, its endpoints are registered
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