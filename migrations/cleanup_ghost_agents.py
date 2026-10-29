#!/usr/bin/env python3
"""Remove ghost agents — orphaned peripheral Redis state without a master key.

A ghost is an agent that still has peripheral keys (``agents:<id>:settings:*``,
``:conversations:*``, ``:users:*``, ``:plugins:*``, ``:ingestion:*``) but no
master key ``agents:<id>:agent``. It is what a delete-while-ingest race used to
leave behind, and the boot-completion path does not touch it because a ghost has
no delete marker (``agents:<id>:ingestion:delete``).

This is a DESTRUCTIVE, manually-run maintenance script — it is NOT part of app
startup and NOT part of the alembic chain.

Safety contract
---------------
* Dry-run is the default: with no flags it prints the plan and exits ``2``
  without writing anything.
* ``--apply`` is required for any mutation and exits ``0`` on full success.
* Exit codes: ``0`` ok, ``1`` partial failure, ``2`` aborted / nothing done.
* Each agent is processed in isolation: one failure never aborts the others.
* Agents that still own their master key or a delete marker are never touched.
* Namespaces whose id contains ``:`` are reported as ``ambiguous_id`` and are
  never classified or deleted, in dry-run and in ``--apply``.
* Only ``cat.db.crud`` is used (``scan_keys``, ``destroy``, ``read``): no raw
  Redis client, no ``get_async_db``.

Usage
-----
    python migrations/cleanup_ghost_agents.py            # dry run, exit 2
    python migrations/cleanup_ghost_agents.py --apply    # delete, exit 0/1
    python migrations/cleanup_ghost_agents.py --json     # machine-readable

Import-safe: importing this module performs no Redis/filesystem work.
"""
import argparse
import asyncio
import json
import sys
from typing import Any

from cat.db import crud
from cat.db.database import DEFAULT_AGENT_KEY, DEFAULT_AGENTS_KEY, DEFAULT_SYSTEM_KEY

EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_ABORTED = 2

#: Key that marks a living agent. An agent holding it is never a ghost.
MASTER_KEY = DEFAULT_AGENT_KEY

#: Suffix of the persistent delete-progress marker owned by the boot path.
DELETE_MARKER_SUFFIX = "ingestion:delete"

#: Key segments that can legally follow ``agents:<id>:`` for a colon-free id.
#: Any other second segment means the actual id may continue past the first
#: segment, so the namespace cannot be parsed safely.
KNOWN_FAMILIES = frozenset({"agent", "settings", "conversations", "users", "plugins", "ingestion"})


def _escape_glob(agent_id: str) -> str:
    """Escape Redis glob metacharacters so an agent id cannot widen its pattern.

    ``agent_id`` is user-controlled (``POST /agents/create``): an id like
    ``a*`` must not turn the deletion pattern into a wildcard that wipes
    unrelated agents.
    """
    escaped = agent_id
    for char in ("\\", "*", "?", "[", "]"):
        escaped = escaped.replace(char, "\\" + char)
    return escaped


def master_key(agent_id: str) -> str:
    """Redis key whose presence proves the agent still exists."""
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:{MASTER_KEY}"


def delete_marker_key(agent_id: str) -> str:
    """Redis key of the delete-progress marker for an agent."""
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:{DELETE_MARKER_SUFFIX}"


def delete_pattern(agent_id: str) -> str:
    """Glob matching every key of one agent, with the id glob-escaped."""
    return f"{DEFAULT_AGENTS_KEY}:{_escape_glob(agent_id)}:*"


def is_protected_id(agent_id: str) -> bool:
    """Whether the id belongs to the system namespace (never touched)."""
    return not agent_id or agent_id == DEFAULT_SYSTEM_KEY or agent_id.startswith(f"{DEFAULT_SYSTEM_KEY}:")


def _ambiguous_namespace(segments: list[str]) -> str:
    """Name the namespace a malformed key belongs to.

    For an id containing ``:`` the family is the segment right before the
    trailing payload (``agents:a:b:agent`` -> id ``a:b``), so the id is every
    segment before the last known family. When no family is present, fall back
    to the first two segments. The exact name only affects reporting: the
    namespace is never classified or deleted.
    """
    for index in range(len(segments) - 1, 0, -1):
        if segments[index] in KNOWN_FAMILIES:
            return ":".join(segments[:index])
    return ":".join(segments[:2])


def group_keys_by_agent(keys: list[str]) -> tuple[dict[str, list[str]], set[str]]:
    """Group scanned ``agents:<id>:*`` keys into safe and ambiguous namespaces.

    The id is taken from the first segment, and the second segment must be a
    known family. An agent id containing ``:`` produces a key whose second
    segment is part of the id (e.g. live ``a:b`` -> ``agents:a:b:agent``), so
    that segment can never be a known family. Such a namespace cannot be
    classified safely: skipping it is the only safe action while the id
    charset is unvalidated (that decision is tracked separately). Returns the
    grouped safe namespaces and the set of ambiguous namespace names.
    """
    grouped: dict[str, list[str]] = {}
    ambiguous: set[str] = set()

    for key in keys:
        prefix = f"{DEFAULT_AGENTS_KEY}:"
        if not key.startswith(prefix):
            continue
        segments = key[len(prefix):].split(":")
        agent_id = segments[0]
        if is_protected_id(agent_id):
            continue
        if len(segments) < 2 or segments[1] not in KNOWN_FAMILIES:
            ambiguous.add(_ambiguous_namespace(segments))
            continue
        grouped.setdefault(agent_id, []).append(key)

    return {agent_id: sorted(agent_keys) for agent_id, agent_keys in sorted(grouped.items())}, ambiguous


async def classify_agent(agent_id: str, keys: list[str]) -> str:
    """Return ``ghost``, ``has_master_key`` or ``has_delete_marker``.

    The protection checks read the actual keys (via ``crud.read``) rather than
    trusting the scan snapshot: an agent that still owns its master key or a
    marker must never be reaped, even if the snapshot is stale.
    """
    if await crud.read(master_key(agent_id)) is not None:
        return "has_master_key"
    if await crud.read(delete_marker_key(agent_id)) is not None:
        return "has_delete_marker"

    peripheral = [k for k in keys if k != master_key(agent_id)]
    return "ghost" if peripheral else "healthy"


async def plan(apply: bool) -> dict[str, Any]:
    """Build the full report, deleting only when ``apply`` is true."""
    report: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "actions": [],
        "skipped": [],
        "failed": [],
        "exit_code": EXIT_OK,
    }

    keys = await crud.scan_keys(f"{DEFAULT_AGENTS_KEY}:*")
    grouped, ambiguous = group_keys_by_agent(keys)

    # Ambiguous namespaces (ids containing ``:``) are never classified and never
    # deleted, in dry-run and in --apply alike.
    for namespace in sorted(ambiguous):
        report["skipped"].append({"agent_id": namespace, "reason": "ambiguous_id"})

    for agent_id, agent_keys in grouped.items():
        try:
            classification = await classify_agent(agent_id, agent_keys)
        except Exception as exc:  # noqa: BLE001 - one bad agent must not abort the run
            report["failed"].append({"agent_id": agent_id, "error": str(exc)})
            continue

        if classification != "ghost":
            report["skipped"].append({"agent_id": agent_id, "reason": classification})
            continue

        action = {"agent_id": agent_id, "keys": agent_keys, "deleted": None}
        if apply:
            exclude = {master_key(agent_id), delete_marker_key(agent_id)}
            try:
                action["deleted"] = await crud.destroy(delete_pattern(agent_id), exclude=exclude)
            except Exception as exc:  # noqa: BLE001 - per-agent isolation
                report["failed"].append({"agent_id": agent_id, "error": str(exc)})
                continue

        report["actions"].append(action)

    if report["failed"]:
        report["exit_code"] = EXIT_PARTIAL_FAILURE
    elif not apply:
        report["exit_code"] = EXIT_ABORTED

    return report


def _print_json(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))


def _print_human(report: dict[str, Any], apply: bool) -> None:
    if not apply:
        print("Ghost agent cleanup — DRY RUN (nothing is written).")

    for action in report["actions"]:
        if apply:
            print(f"Removed ghost '{action['agent_id']}' ({action['deleted']} keys):")
        else:
            print(f"Ghost '{action['agent_id']}' would be removed ({len(action['keys'])} keys):")
        for key in action["keys"]:
            print(f"  - {key}")

    for skipped in report["skipped"]:
        print(f"Skipped '{skipped['agent_id']}': {skipped['reason']}")

    for failure in report["failed"]:
        print(f"FAILED '{failure['agent_id']}': {failure['error']}", file=sys.stderr)

    ghosts = len(report["actions"])
    if not apply:
        print(f"Dry run: {ghosts} ghost(s) would be removed. Re-run with --apply to delete. Nothing was written.")
    else:
        print(f"Done: {ghosts} ghost(s) removed, {len(report['failed'])} failed, {len(report['skipped'])} skipped.")
        if report["failed"]:
            print("Partial failure: some agents could not be processed.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code; never calls ``sys.exit``."""
    parser = argparse.ArgumentParser(
        description="Remove ghost agents (peripheral Redis state without a master key).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete ghost keys (default: dry run, exit 2, writes nothing)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable report",
    )
    args = parser.parse_args(argv)

    try:
        report = asyncio.run(plan(args.apply))
    except Exception as exc:  # noqa: BLE001 - connection/scan failures abort safely
        print(f"Aborted: {exc}", file=sys.stderr)
        return EXIT_ABORTED

    if args.json:
        _print_json(report)
    else:
        _print_human(report, args.apply)

    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
