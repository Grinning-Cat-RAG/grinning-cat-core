"""Tests for the base admin permission reconciliation at bootstrap.

The base admin is created once, on a fresh install, with the full permission set
of that moment. When a new ``AuthResource`` is introduced later (e.g.
``ANALYTICS``), the pre-existing admin must gain it at boot; other users must
never be touched, and a fresh install must keep creating a full-permission admin.
"""
from copy import deepcopy

from cat.auth.auth_utils import DEFAULT_ADMIN_USERNAME, hash_password
from cat.auth.permissions import get_full_permissions
from cat.looking_glass import bill_the_lizard as btl
from cat.utils import sanitize_permissions

SYSTEM = "system"
FULL = sanitize_permissions(get_full_permissions(), SYSTEM)


class _FakeUsers:
    """Minimal in-memory stand-in for ``cat.db.cruds.users``."""

    def __init__(self, users=None):
        self.users = users or {}
        self.updates = []

    async def get_user_by_username(self, agent_id, username, with_password=False):
        user = self.users.get(username)
        return deepcopy(user) if user else None

    async def update_user(self, agent_id, user_id, updated_info):
        self.updates.append((agent_id, user_id, deepcopy(updated_info)))
        for username, user in self.users.items():
            if user.get("id") == user_id:
                self.users[username] = deepcopy(updated_info)
        return deepcopy(updated_info)


def _patch_users(monkeypatch, fake):
    monkeypatch.setattr(btl.crud_users, "get_user_by_username", fake.get_user_by_username)
    monkeypatch.setattr(btl.crud_users, "update_user", fake.update_user)


def _admin(permissions):
    return {
        "id": "admin-id",
        "username": DEFAULT_ADMIN_USERNAME,
        "password": hash_password("secret"),
        "permissions": permissions,
        "created_at": 1234.0,
        "updated_at": 5678.0,
    }


async def test_existing_admin_gains_missing_analytics(monkeypatch):
    permissions = deepcopy(FULL)
    permissions.pop("ANALYTICS")
    original = _admin(permissions)
    fake = _FakeUsers({DEFAULT_ADMIN_USERNAME: original})
    _patch_users(monkeypatch, fake)

    added = await btl._reconcile_base_admin_permissions()

    assert "ANALYTICS" in added
    stored = fake.users[DEFAULT_ADMIN_USERNAME]
    assert stored["permissions"]["ANALYTICS"] == FULL["ANALYTICS"]
    assert stored["permissions"]["MEMORY"] == FULL["MEMORY"]
    assert stored["username"] == DEFAULT_ADMIN_USERNAME
    assert stored["password"] == original["password"]
    assert stored["created_at"] == 1234.0
    assert len(fake.updates) == 1
    assert fake.updates[0][1] == "admin-id"


async def test_non_admin_user_is_not_modified(monkeypatch):
    restricted = {"MEMORY": ["READ"], "CHAT": ["READ"]}
    fake = _FakeUsers({"student": {
        "id": "student-id",
        "username": "student",
        "password": hash_password("pw"),
        "permissions": deepcopy(restricted),
        "created_at": 1.0,
        "updated_at": 2.0,
    }})
    _patch_users(monkeypatch, fake)

    added = await btl._reconcile_base_admin_permissions()

    assert added == {}
    assert fake.updates == []
    assert fake.users["student"]["permissions"] == restricted


async def test_already_full_admin_is_a_noop(monkeypatch):
    fake = _FakeUsers({DEFAULT_ADMIN_USERNAME: _admin(deepcopy(FULL))})
    _patch_users(monkeypatch, fake)

    added = await btl._reconcile_base_admin_permissions()

    assert added == {}
    assert fake.updates == []


class _FakePluginManager:
    endpoints = {}

    async def discover_plugins(self):
        pass

    async def execute_hook(self, *args, **kwargs):
        pass

    def get_core_plugins_ids(self):
        return set()

    def get_system_only_plugin_ids(self):
        return set()


class _FakeServiceProvider:
    async def bootstrap_services(self, *args, **kwargs):
        pass


class _FakeWebSocketManager:
    async def start(self):
        pass


async def test_fresh_install_creates_admin_with_full_permissions(monkeypatch):
    from cat.looking_glass.bill_the_lizard import BillTheLizard

    created = {}
    fake = _FakeUsers({})

    async def fake_get_users(agent_id, limit=None):
        return {}

    async def fake_create_user(agent_id, new_user):
        created.update(new_user)
        fake.users[new_user["username"]] = {**new_user, "id": "new-admin-id"}
        return new_user

    monkeypatch.setattr(btl.crud_users, "get_users", fake_get_users)
    monkeypatch.setattr(btl.crud_users, "create_user", fake_create_user)
    _patch_users(monkeypatch, fake)
    monkeypatch.setattr(btl, "WebSocketManager", _FakeWebSocketManager)
    monkeypatch.setattr(btl, "RabbitHole", lambda: object())
    monkeypatch.setattr(btl, "CoreAuthHandler", lambda: object())

    lizard = object.__new__(BillTheLizard.__wrapped__)
    lizard.plugin_manager = _FakePluginManager()  # type: ignore[assignment]
    lizard._fastapi_app = None
    lizard._pending_endpoints = []
    lizard._service_provider = _FakeServiceProvider()  # type: ignore[assignment]
    lizard._delete_completion_task = None
    lizard._schedule_delete_completion = lambda: None

    await lizard.bootstrap()

    assert created["username"] == DEFAULT_ADMIN_USERNAME
    assert created["permissions"] == FULL
    assert created["password"] != "admin"
