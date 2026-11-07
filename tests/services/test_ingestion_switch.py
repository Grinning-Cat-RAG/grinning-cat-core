"""The ingestion profile-switch transition.

These tests pin the transition contract with fake profiles: every source of
every agent is deleted through the OLD profile, the old state rows are cleared,
then the NEW engine runs. A failure during deletion restores the previous
setting so the old configuration stays in force, and a same-profile change is
never a transition.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from cat.services.ingestion.locks import NoopLockManager
from cat.services.ingestion.pipeline import IngestionPipeline
from cat.services.ingestion.profile import IngestionProfile
from cat.services.ingestion.recovery import NoopRecovery
from cat.services.ingestion.state import IngestionStateStore
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
