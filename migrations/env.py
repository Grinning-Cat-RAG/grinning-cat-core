import sys
import importlib.util
from typing import Any, Dict, List, Callable
from datetime import datetime, timezone
from pathlib import Path
import redis

sys.path.insert(0, str(Path(__file__).parent.parent))


class MigrationContext:
    """Context passed to migration functions"""
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def get_json(self, key: str) -> Any | None:
        """Get and parse JSON from Redis"""
        data = self.redis.json().get(key)
        return data if data else None

    def set_json(self, key: str, value: Any, path: str | None = "$") -> None:
        """Set JSON value in Redis"""
        self.redis.json().set(key, path, value)

    def delete(self, key: str, path: str | None = "$") -> None:
        """Delete a key from Redis"""
        self.redis.json().delete(key, path)

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self.redis.exists(key) > 0


class MigrationRevision:
    """Represents a migration revision"""

    def __init__(
        self,
        revision: str,
        down_revision: str | None,
        description: str | None = None,
        upgrade_func: Callable | None = None,
        downgrade_func: Callable | None = None
    ):
        self.revision = revision
        self.down_revision = down_revision
        self.description = description or ""
        self.upgrade_func = upgrade_func
        self.downgrade_func = downgrade_func
        self.created_at = datetime.now()

    def upgrade(self, context: MigrationContext) -> None:
        """Execute upgrade"""
        if self.upgrade_func:
            self.upgrade_func(context)

    def downgrade(self, context: MigrationContext) -> None:
        """Execute downgrade"""
        if self.downgrade_func:
            self.downgrade_func(context)


class MigrationEnvironment:
    """Migration environment configuration"""
    MIGRATIONS_KEY = "schema:migrations"
    CURRENT_HEAD_KEY = "schema:current_head"

    def __init__(
        self,
        redis_client: redis.Redis,
        migrations_dir: str = "migrations/versions"
    ):
        self.redis = redis_client
        self.migrations_dir = Path(migrations_dir)
        self.context = MigrationContext(redis_client)
        self.revisions: Dict[str, MigrationRevision] = {}
        self._load_revisions()

    def _load_revisions(self) -> None:
        """Load all migration revisions from the versions directory"""
        if not self.migrations_dir.exists():
            return

        for file_path in sorted(self.migrations_dir.glob("*.py")):
            if file_path.name.startswith("__"):
                continue

            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "revision"):
                    revision = MigrationRevision(
                        revision=module.revision,
                        down_revision=getattr(module, "down_revision", None),
                        description=getattr(module, "__doc__", "").strip(),
                        upgrade_func=getattr(module, "upgrade", None),
                        downgrade_func=getattr(module, "downgrade", None)
                    )
                    self.revisions[revision.revision] = revision

    def get_current_head(self) -> str | None:
        """Get current head revision"""
        head = self.redis.get(self.CURRENT_HEAD_KEY)
        return head.decode() if isinstance(head, bytes) else head

    def set_current_head(self, revision: str | None) -> None:
        """Set current head revision"""
        if revision:
            self.redis.set(self.CURRENT_HEAD_KEY, revision)
            return

        self.redis.delete(self.CURRENT_HEAD_KEY)

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history"""
        data = self.redis.json().get(self.MIGRATIONS_KEY)
        return data if data else []  # type: ignore[return-value]

    def add_to_history(self, revision: str, action: str) -> None:
        """Add migration to history"""
        now = datetime.now(timezone.utc)

        history = self.get_migration_history()
        history.append({
            "revision": revision,
            "action": action,
            "applied_at": now.isoformat(),
            "timestamp": now.timestamp(),
        })
        self.redis.json().set(self.MIGRATIONS_KEY, "$", history)

    def get_revision_chain(
        self,
        start: str | None = None,
        end: str | None = None
    ) -> List[MigrationRevision]:
        """Get chain of revisions between start and end"""
        if end is None:
            # Get all revisions from start to latest
            end = self._get_head_revision()

        if start is None:
            # Get all revisions from base to end
            chain = []
            current = end
            while current:
                if current in self.revisions:
                    rev = self.revisions[current]  # type: ignore[union-attr]
                    chain.insert(0, rev)
                    current = rev.down_revision
                else:
                    break
            return chain

        # Get revisions between start and end
        chain = []
        current = end
        while current and current != start:
            if current in self.revisions:
                rev = self.revisions[current]  # type: ignore[union-attr]
                chain.insert(0, rev)
                current = rev.down_revision
            else:
                break
        return chain

    def _get_head_revision(self) -> str | None:
        """Get the latest head revision (the one with no children)"""
        children = {rev.down_revision for rev in self.revisions.values() if rev.down_revision}
        for revision_id in self.revisions:
            if revision_id not in children:
                return revision_id
        return None

    def upgrade(self, target: str = "head") -> None:
        """Upgrade to target revision"""
        current = self.get_current_head()

        # a revision this code does not know (e.g. applied by a newer image, before a rollback): the chain cannot be
        # computed from it, and would restart from the base, re-applying every migration over migrated data
        if current and current not in self.revisions:
            raise RuntimeError(
                f"The database is at revision {current}, unknown to these migrations: downgrade it first with the "
                f"version that knows it (manage_migrations.py downgrade <revision>), or upgrade the application"
            )

        if target == "head":
            target = self._get_head_revision()  # type: ignore[union-attr]

        if not target:
            print("No migrations to apply")
            return

        if current == target:
            print(f"Already at revision {target}")
            return

        # Get revision chain
        chain = self.get_revision_chain(current, target)

        if not chain:
            print("No migrations to apply")
            return

        print(f"Upgrading from {current or 'base'} to {target}")
        print(f"Will apply {len(chain)} migration(s)\n")

        for revision in chain:
            print(f"→ Applying {revision.revision}: {revision.description}")
            try:
                revision.upgrade(self.context)
                self.set_current_head(revision.revision)
                self.add_to_history(revision.revision, "upgrade")
                print("  ✓ Success\n")
            except Exception as e:
                print(f"  ✗ Failed: {e}\n")
                raise

        print(f"Upgrade complete. Current head: {target}")

    def downgrade(self, target: str = "-1") -> None:
        """Downgrade to target revision"""
        current = self.get_current_head()

        if not current:
            print("Already at base")
            return

        # Handle relative downgrades
        if target.startswith("-"):
            steps = int(target)
            chain = []
            temp_current = current
            for _ in range(abs(steps)):
                if temp_current and temp_current in self.revisions:
                    rev = self.revisions[temp_current]
                    chain.append(rev)
                    temp_current = rev.down_revision
                else:
                    break
            target = temp_current  # type: ignore[union-attr]
        else:
            # Get all revisions from target to current
            chain = []
            temp = current
            while temp and temp != target:
                if temp in self.revisions:
                    rev = self.revisions[temp]
                    chain.append(rev)
                    temp = rev.down_revision
                else:
                    break

        if not chain:
            print("No migrations to downgrade")
            return

        print(f"Downgrading from {current} to {target or 'base'}")
        print(f"Will revert {len(chain)} migration(s)\n")

        for revision in chain:
            print(f"← Reverting {revision.revision}: {revision.description}")
            try:
                revision.downgrade(self.context)
                self.set_current_head(revision.down_revision)
                self.add_to_history(revision.revision, "downgrade")
                print("  ✓ Success\n")
            except Exception as e:
                print(f"  ✗ Failed: {e}\n")
                raise

        print(f"Downgrade complete. Current head: {target or 'base'}")

    def current(self) -> None:
        """Show the current revision"""
        head = self.get_current_head()
        if head and head in self.revisions:
            rev = self.revisions[head]
            print(f"Current revision: {head}")
            print(f"Description: {rev.description}")

            return

        print("Current revision: base (no migrations applied)")

    def history(self, verbose: bool = False) -> None:
        """Show migration history"""
        history = self.get_migration_history()

        if not history:
            print("No migration history")
            return

        print("\nMigration History")
        print("=" * 70)

        for entry in reversed(history):
            action_symbol = "↑" if entry["action"] == "upgrade" else "↓"
            print(f"{action_symbol} {entry['revision']}")
            print(f"  Action: {entry['action']}")
            print(f"  Applied: {entry['timestamp']}")

            if verbose and entry['revision'] in self.revisions:
                rev = self.revisions[entry['revision']]
                print(f"  Description: {rev.description}")
            print()

    def heads(self) -> None:
        """Show head revisions"""
        head = self._get_head_revision()
        if head:
            print(f"Head revision: {head}")
            if head in self.revisions:
                print(f"Description: {self.revisions[head].description}")
            return

        print("No head revision found")

    def show(self, revision: str) -> None:
        """Show details of a specific revision"""
        if revision not in self.revisions:
            print(f"Revision {revision} not found")
            return

        rev = self.revisions[revision]
        print(f"Revision: {rev.revision}")
        print(f"Down revision: {rev.down_revision or 'base'}")
        print(f"Description: {rev.description}")


# CLI-style commands
class MigrationCommands:
    """Command-line style interface for migrations"""

    def __init__(self, redis_client: redis.Redis, migrations_dir: str = "migrations/versions"):
        self.env = MigrationEnvironment(redis_client, migrations_dir)

    def upgrade(self, revision: str = "head") -> None:
        """Upgrade to a later version"""
        self.env.upgrade(revision)

    def downgrade(self, revision: str = "-1") -> None:
        """Revert to a previous version"""
        self.env.downgrade(revision)

    def current(self) -> None:
        """Display the current revision"""
        self.env.current()

    def history(self, verbose: bool = False) -> None:
        """List changeset scripts in chronological order"""
        self.env.history(verbose)

    def heads(self) -> None:
        """Show current available heads"""
        self.env.heads()

    def show(self, revision: str) -> None:
        """Show the revision details"""
        self.env.show(revision)
