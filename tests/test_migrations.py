"""The Redis migrations refuse to upgrade from a revision they do not know (e.g. applied by a newer version, before a
rollback): computed from it, the chain would restart from the base and re-apply every migration over migrated data."""
import pytest

from migrations.env import MigrationEnvironment


class FakeJson:
    def __init__(self, store: dict):
        self.store = store

    def get(self, key):
        return self.store.get(key)

    def set(self, key, path, value):
        self.store[key] = value


class FakeRedis:
    """Just what the migration environment uses, in memory: no Redis needed."""

    def __init__(self):
        self.store: dict = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)

    def json(self):
        return FakeJson(self.store)


def write_revisions(folder, *chain):
    """Revision modules r1 <- r2 <- ...; each upgrade counts its runs in the store."""
    folder.mkdir()
    down = None
    for rev in chain:
        (folder / f"{rev}.py").write_text(
            f'"""{rev}"""\n'
            f"revision = {rev!r}\n"
            f"down_revision = {down!r}\n"
            "def upgrade(context):\n"
            f"    context.redis.store[{rev!r} + ':runs'] = context.redis.store.get({rev!r} + ':runs', 0) + 1\n"
        )
        down = rev
    return folder


def runs(db: FakeRedis, rev: str) -> int:
    return db.store.get(f"{rev}:runs", 0)


def test_upgrade_applies_the_pending_migrations_once(tmp_path):
    db = FakeRedis()
    env = MigrationEnvironment(db, str(write_revisions(tmp_path / "versions", "r1", "r2")))

    env.upgrade()
    env.upgrade()

    assert (runs(db, "r1"), runs(db, "r2")) == (1, 1)
    assert env.get_current_head() == "r2"


def test_upgrade_from_a_known_revision_applies_only_the_next_ones(tmp_path):
    db = FakeRedis()
    env = MigrationEnvironment(db, str(write_revisions(tmp_path / "versions", "r1", "r2")))
    env.set_current_head("r1")

    env.upgrade()

    assert (runs(db, "r1"), runs(db, "r2")) == (0, 1)


def test_upgrade_refuses_a_revision_it_does_not_know(tmp_path):
    db = FakeRedis()
    # a newer version applied r3, then the application was rolled back to r1 <- r2
    env = MigrationEnvironment(db, str(write_revisions(tmp_path / "versions", "r1", "r2")))
    env.set_current_head("r3")

    with pytest.raises(RuntimeError, match="revision r3, unknown"):
        env.upgrade()

    assert (runs(db, "r1"), runs(db, "r2")) == (0, 0)  # nothing re-applied
    assert env.get_current_head() == "r3"
