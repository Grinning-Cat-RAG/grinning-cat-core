"""The ingestion profile-switch transition.

These tests pin the transition contract with fake profiles: every source of
every agent is deleted through the OLD profile, the old state rows are cleared,
then the NEW engine runs. A failure during deletion restores the previous
setting so the old configuration stays in force, and a same-profile change is
never a transition.
"""

import os
import shutil
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
from cat.db import crud
from cat.services.factory.ingestion import CoreIngestionConfiguration
from cat.services.ingestion.locks import NoopLockManager
from cat.services.ingestion.models import IngestionStatus
from cat.services.ingestion.pipeline import IngestionPipeline
from cat.services.ingestion.profile import IngestionProfile
from cat.services.ingestion.recovery import NoopRecovery
from cat.services.ingestion.state import (
    CoreIngestionStateStore,
    IngestionStateStore,
    status_key,
)
from cat.services.ingestion.switch import (
    profile_changed,
    transition_ingestion_profile,
)
from cat.services.ingestion.tags import SettingsVersionTags


class FakePipeline(IngestionPipeline):
    def __init__(self, fail_on: str | None = None):
        self.deleted: list[tuple[str, Any, str | None]] = []
        self.fail_on = fail_on

    def phases(self) -> list:
        return []

    async def execute(self, source: str, cat) -> None:
        pass

    async def delete(self, source, cat, scope=None) -> bool:
        if self.fail_on == source:
            raise RuntimeError(f"cannot delete {source}")
        self.deleted.append((source, cat, scope))
        return True


class FakeStateStore(IngestionStateStore):
    def __init__(self):
        self.cleared: list[tuple[str, str, str]] = []

    async def get(self, agent_id: str, scope: str, source: str) -> dict[str, Any] | None:
        return None

    async def set(self, agent_id: str, scope: str, source: str, **fields) -> dict[str, Any]:
        return {}

    async def clear(self, agent_id, scope, source) -> None:
        self.cleared.append((agent_id, scope, source))


class FakeEngine:
    def __init__(self):
        self.runs: list = []

    async def run(self, lizard) -> bool:
        self.runs.append(lizard)
        return True


class FakeFileManager:
    def __init__(self, names: list[str]):
        self._names = names

    def list_files(self, remote_root_dir) -> list:
        return [SimpleNamespace(name=name) for name in self._names]


class FakeLizard:
    def __init__(self, cats: dict[str, Any], engine: FakeEngine):
        self._cats = cats
        self._engine = engine

    async def get_cheshire_cat(self, agent_id):
        return self._cats.get(agent_id)

    async def ingestion(self):
        return self._engine


def make_profile(pipeline: IngestionPipeline | None = None) -> IngestionProfile:
    return IngestionProfile(
        pipeline=pipeline or FakePipeline(),
        state_store=FakeStateStore(),
        recovery=NoopRecovery(),
        locks=NoopLockManager(),
        tags=SettingsVersionTags(),
    )


def make_cat(agent_id: str, names: list[str]):
    return SimpleNamespace(agent_key=agent_id, file_manager=FakeFileManager(names))


async def test_transition_deletes_every_source_clears_state_and_runs_new_engine(monkeypatch):
    a1 = make_cat("a1", ["one.txt", "two.txt"])
    a2 = make_cat("a2", ["three.txt"])
    old_profile = make_profile()
    new_profile = make_profile()
    engine = FakeEngine()
    lizard = FakeLizard({"a1": a1, "a2": a2}, engine)

    async def get_agents_main_keys():
        return ["a1", "a2", "system"]

    monkeypatch.setattr(
        "cat.services.ingestion.switch.crud_settings.get_agents_main_keys",
        get_agents_main_keys,
    )

    result = await transition_ingestion_profile(
        lizard, old_profile, new_profile, {"name": "OldConfig", "value": {}}
    )

    assert result is True
    assert isinstance(old_profile.pipeline, FakePipeline)
    assert old_profile.pipeline.deleted == [
        ("one.txt", a1, "agent"),
        ("two.txt", a1, "agent"),
        ("three.txt", a2, "agent"),
    ]
    assert isinstance(old_profile.state_store, FakeStateStore)
    assert old_profile.state_store.cleared == [
        ("a1", "agent", "one.txt"),
        ("a1", "agent", "two.txt"),
        ("a2", "agent", "three.txt"),
    ]
    assert engine.runs == [lizard]


async def test_delete_failure_restores_previous_setting_and_does_not_run(monkeypatch):
    cat = make_cat("a1", ["one.txt"])
    old_profile = make_profile(pipeline=FakePipeline(fail_on="one.txt"))
    new_profile = make_profile()
    engine = FakeEngine()
    lizard = FakeLizard({"a1": cat}, engine)
    restore = AsyncMock()

    async def get_agents_main_keys():
        return ["a1"]

    monkeypatch.setattr(
        "cat.services.ingestion.switch.crud_settings.get_agents_main_keys",
        get_agents_main_keys,
    )
    monkeypatch.setattr(
        "cat.services.ingestion.switch.crud_settings.upsert_setting_by_category",
        restore,
    )

    previous = {"name": "OldConfig", "category": "ingestion", "value": {"marker": "old"}}
    result = await transition_ingestion_profile(lizard, old_profile, new_profile, previous)

    assert result is False
    restore.assert_awaited_once()
    assert restore.await_args is not None
    key_id, restored = restore.await_args.args
    assert key_id == "system"
    assert restored.name == "OldConfig"
    assert restored.category == "ingestion"
    assert restored.value == {"marker": "old"}
    assert engine.runs == []


def test_same_profile_is_not_a_change():
    assert profile_changed(make_profile(), make_profile()) is False


def test_different_implementation_is_a_change():
    class OtherPipeline(FakePipeline):
        pass

    old_profile = make_profile()
    new_profile = make_profile(pipeline=OtherPipeline())

    assert profile_changed(old_profile, new_profile) is True


class FakePluginManager:
    async def execute_hook(self, *args, **kwargs) -> None:
        return None


class FakeVectorMemoryHandler:
    def __init__(self):
        self.deleted: list[tuple[str, dict[str, Any]]] = []

    async def delete_tenant_points(self, collection: str, metadata: dict[str, Any]) -> None:
        self.deleted.append((collection, metadata))


def make_real_artifact_cat(agent_id: str, file_manager: LocalFileManager):
    """A cat backed by a REAL LocalFileManager; only memory/hooks are faked."""
    return SimpleNamespace(
        agent_key=agent_id,
        id="chat",
        file_manager=file_manager,
        plugin_manager=FakePluginManager(),
        vector_memory_handler=FakeVectorMemoryHandler(),
    )


async def test_core_profile_switch_removes_real_status_row_and_file(monkeypatch, tmp_path):
    """A real status row and a real file created by the OLD (core) profile are gone."""
    agent_id = f"test-switch-{uuid.uuid4().hex}"
    source = "orphan.txt"
    store = CoreIngestionStateStore()
    file_manager = LocalFileManager()
    file_manager._root_dir = str(tmp_path)
    key = status_key(agent_id, "agent", source)
    file_path = os.path.join(str(tmp_path), agent_id, source)

    try:
        # 1. a REAL status row through the core state store
        await store.set(agent_id, "agent", source, status=IngestionStatus.COMPLETED)
        assert await store.get(agent_id, "agent", source) is not None

        # 2. a REAL file through the agent's REAL file manager
        assert file_manager.write_file(b"old profile artifact", source, agent_id) is True
        assert os.path.isfile(file_path)

        cat = make_real_artifact_cat(agent_id, file_manager)
        old_profile = CoreIngestionConfiguration.profile()
        new_profile = make_profile()
        engine = FakeEngine()
        lizard = FakeLizard({agent_id: cat}, engine)

        async def get_agents_main_keys():
            return [agent_id]

        monkeypatch.setattr(
            "cat.services.ingestion.switch.crud_settings.get_agents_main_keys",
            get_agents_main_keys,
        )

        result = await transition_ingestion_profile(
            lizard,
            old_profile,
            new_profile,
            {"name": "Old", "category": "ingestion", "value": {}},
        )

        assert result is True
        # the transition deleted what the OLD profile created
        assert await store.get(agent_id, "agent", source) is None
        assert await crud.read(key) is None
        assert not os.path.exists(file_path)
        assert engine.runs == [lizard]
    finally:
        await crud.delete(key)
        shutil.rmtree(os.path.join(str(tmp_path), agent_id), ignore_errors=True)


class MarkerDeletingPipeline(FakePipeline):
    """An OLD plugin pipeline that removes a marker artifact on delete."""

    def __init__(self, marker_path: str):
        super().__init__()
        self.marker_path = marker_path

    async def delete(self, source, cat, scope=None) -> bool:
        self.deleted.append((source, cat, scope))
        if os.path.exists(self.marker_path):
            os.remove(self.marker_path)
        return True


async def test_old_plugin_profile_delete_runs_when_switching_to_core(monkeypatch, tmp_path):
    """Switching away from a plugin profile runs ITS delete: the marker disappears."""
    agent_id = f"test-switch-{uuid.uuid4().hex}"
    source = "plugin-source.txt"
    marker_path = os.path.join(str(tmp_path), "plugin-marker.bin")
    file_manager = LocalFileManager()
    file_manager._root_dir = str(tmp_path)

    try:
        with open(marker_path, "wb") as marker:
            marker.write(b"created by the old plugin profile")
        # a stored source so the transition has something to enumerate/delete
        assert file_manager.write_file(b"plugin source", source, agent_id) is True
        assert os.path.isfile(marker_path)

        cat = make_real_artifact_cat(agent_id, file_manager)
        old_profile = make_profile(pipeline=MarkerDeletingPipeline(marker_path))
        new_profile = CoreIngestionConfiguration.profile()
        engine = FakeEngine()
        lizard = FakeLizard({agent_id: cat}, engine)

        async def get_agents_main_keys():
            return [agent_id]

        monkeypatch.setattr(
            "cat.services.ingestion.switch.crud_settings.get_agents_main_keys",
            get_agents_main_keys,
        )

        result = await transition_ingestion_profile(
            lizard,
            old_profile,
            new_profile,
            {"name": "Old", "category": "ingestion", "value": {}},
        )

        assert result is True
        assert isinstance(old_profile.pipeline, MarkerDeletingPipeline)
        assert old_profile.pipeline.deleted == [(source, cat, "agent")]
        assert not os.path.exists(marker_path)
    finally:
        shutil.rmtree(os.path.join(str(tmp_path), agent_id), ignore_errors=True)
        if os.path.exists(marker_path):
            os.remove(marker_path)
