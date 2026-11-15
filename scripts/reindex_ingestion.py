#!/usr/bin/env python3
"""Reindex the ingestion of an agent: re-run every phase from scratch, keeping the source files and the status rows.

Works on agents with status rows AND on old agents that have only files in the filesystem and old chunks in the
vector store: the sources are the UNION of the status rows, the vector-store sources and the stored files (deduped
by ``(name, chat_id)``, derived image artifacts skipped). A rowless source is adopted by the core
(``RabbitHole.reingest_source`` -> ``_adopt(require_points=False)``) and every phase runs again.

URL sources: ``chunks.read_source_bytes`` reads the kept ``.source`` copy; when it exists the URL is NOT downloaded
again. When it is missing, ``reingest_source`` does not raise ``SourceNotKept`` for URLs and the parsing phase
re-downloads the URL (the content is the current one, not the one indexed before).

Usage:
    python3 scripts/reindex_ingestion.py --agent <id|substring> [--agent ...] [--source <name>] [--from-phase <id>] [--yes]
    python3 scripts/reindex_ingestion.py --agent <id|substring> [--agent ...] --errored [--yes]
    python3 scripts/reindex_ingestion.py --all [--yes]
    python3 scripts/reindex_ingestion.py --all --errored [--yes]

``--agent`` accepts an id or a substring of it (case-insensitive): every matching agent is selected.

``--errored``: only the sources with at least one phase in ``error`` state; for each, the artifacts of the earliest
errored phase (and of its dependents) are purged, their diary entries reset, and the phases run again. The phases
before it keep their artifacts.

Dry-run is the default: it prints the plan and exits 2. Pass --yes to execute.
Exit codes: 0 ok, 1 partial failure, 2 aborted/dry-run.
"""
import argparse
import asyncio
import os
import re
import sys

from cat import BillTheLizard
from cat.db.cruds import ingestions as crud
from cat.db.cruds import settings as crud_settings
from cat.rabbit_hole import RabbitHole, SourceNotKept, engine

_DERIVED_IMAGE = re.compile(r"_img_\d+_[0-9a-f]+\.\w+$", re.IGNORECASE)
_PROTECTED = ("system",)


def _is_protected(agent_id: str) -> bool:
    return agent_id in _PROTECTED or agent_id.startswith("__")


async def _rabbit_hole(ccat):
    """The RabbitHole bound to the agent, created when the process has none (maintenance CLI)."""
    rabbit_hole = getattr(ccat, "rabbit_hole", None)
    if rabbit_hole is not None:
        return rabbit_hole
    rabbit_hole = RabbitHole()
    await rabbit_hole.setup(ccat)
    return rabbit_hole


async def _enumerate(ccat, agent_id: str):
    """The sources of the agent: status rows ∪ vector store ∪ stored files, deduped by ``(name, chat_id)``."""
    seen = {}

    for row in await crud.list_statuses(agent_id):
        name = row.get("source")
        if name:
            seen.setdefault((name, row.get("chat_id")), "row")

    try:
        stored = await ccat.get_stored_sources_with_metadata()
    except Exception as e:  # noqa: BLE001 - the enumeration must not abort the run
        print(f"  warning: vector-store enumeration failed: {e}", file=sys.stderr)
        stored = {}
    for sources in (stored or {}).values():
        for source in sources or []:
            name = getattr(source, "name", None)
            metadata = getattr(source, "metadata", None) or {}
            if name:
                seen.setdefault((name, metadata.get("chat_id")), "vector")

    try:
        for file in ccat.file_manager.list_files(agent_id) or []:
            name = getattr(file, "name", None)
            if name:
                seen.setdefault((name, None), "file")
    except Exception as e:  # noqa: BLE001
        print(f"  warning: file enumeration failed: {e}", file=sys.stderr)

    chats = {chat for (_, chat) in seen if chat}
    for chat in chats:
        try:
            for file in ccat.file_manager.list_files(os.path.join(agent_id, chat)) or []:
                name = getattr(file, "name", None)
                if name:
                    seen.setdefault((name, chat), "file")
        except Exception as e:  # noqa: BLE001
            print(f"  warning: file enumeration for chat {chat} failed: {e}", file=sys.stderr)

    return [
        (name, chat, origin)
        for (name, chat), origin in seen.items()
        if not _DERIVED_IMAGE.search(name)
    ]


async def reindex_agent(
    lizard, agent_id: str, source: str | None, dry_run: bool, from_phase: str | None = None, errored: bool = False,
) -> int:
    ccat = await lizard.get_cheshire_cat(agent_id)
    if ccat is None:
        print(f"agent {agent_id} not found", file=sys.stderr)
        return 1

    sources = await _enumerate(ccat, agent_id)
    if source is not None:
        sources = [s for s in sources if s[0] == source]
    if not sources:
        print(f"agent {agent_id}: no sources to reindex", file=sys.stderr)
        return 1

    rabbit_hole = await _rabbit_hole(ccat)
    try:
        phases = await engine.collect_phases(rabbit_hole, ccat)
    except Exception as e:  # noqa: BLE001 - the CLI reports the invalid phase graph
        print(f"agent {agent_id}: cannot collect the phases: {e}", file=sys.stderr)
        return 1
    valid = {phase.id for phase in phases}
    if from_phase is not None and from_phase not in valid:
        print(f"unknown phase {from_phase!r}; valid: {sorted(valid)}", file=sys.stderr)
        return 1
    order = {phase.id: index for index, phase in enumerate(phases)}

    failures = 0
    for name, chat_id, origin in sources:
        restart = from_phase
        if errored:
            row = await crud.get_status(agent_id, chat_id, name)
            failed = [
                phase_id for phase_id, entry in ((row or {}).get("completed_phases") or {}).items()
                if entry.get("state") == engine.STATE_ERROR
            ]
            if not failed:
                continue
            # The earliest errored phase: purging it (and its dependents) is what makes the later errored phases
            # run again too; the phases before it keep their artifacts.
            restart = min(failed, key=lambda phase_id: order.get(phase_id, len(order)))
            print(
                f"{'[dry-run] ' if dry_run else ''}reindex {name!r} "
                f"(chat={chat_id!r}, origin={origin}, errored={sorted(failed)}, from={restart})"
            )
        else:
            print(
                f"{'[dry-run] ' if dry_run else ''}reindex {name!r} "
                f"(chat={chat_id!r}, origin={origin})"
            )
        if dry_run:
            continue
        try:
            result = await rabbit_hole.reingest_source(ccat, name, chat_id, from_phase=restart)
            print(f"  -> {result.get('status') if result else 'gone'}")
        except SourceNotKept as e:
            print(f"  -> skipped: {e}", file=sys.stderr)
            failures += 1
        except Exception as e:  # noqa: BLE001 - one source must not abort the run
            print(f"  -> error: {e}", file=sys.stderr)
            failures += 1
    return 0 if failures == 0 else 1


async def main(args) -> int:
    known = [a["agent_id"] for a in await crud_settings.get_agents()]
    if args.all:
        agents = [agent_id for agent_id in known if not _is_protected(agent_id)]
    else:
        # Each --agent value is an id OR a substring of it (case-insensitive): every matching agent is selected.
        agents = []
        for wanted in args.agent or []:
            matches = [agent_id for agent_id in known if wanted.lower() in agent_id.lower()]
            if not matches:
                print(f"no agent matches {wanted!r}", file=sys.stderr)
                return 1
            agents.extend(matches)
        agents = list(dict.fromkeys(agents))
    if not agents:
        print("no agents selected", file=sys.stderr)
        return 1

    lizard = BillTheLizard()
    await lizard.bootstrap()

    failures = 0
    for agent_id in agents:
        print(f"== agent {agent_id}")
        failures += await reindex_agent(
            lizard, agent_id, args.source, not args.yes, args.from_phase, args.errored,
        )
    return 0 if failures == 0 else 1


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="append", help="agent id or a substring of it, case-insensitive (repeatable)")
    parser.add_argument("--all", action="store_true", help="every agent except system/__*__")
    parser.add_argument("--source", help="only this source")
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--from-phase",
        help="re-run from this phase and its dependents "
             "(parsing_chunking, embedding, catalog_cards, graph_spacy_extraction, graph_llm_extraction)",
    )
    phase_group.add_argument(
        "--errored",
        action="store_true",
        help="only the sources with a phase in error: purge the earliest errored phase and its dependents, "
             "reset their entries and run them again",
    )
    parser.add_argument("--yes", action="store_true", help="execute (default: dry-run)")
    args = parser.parse_args()
    if not args.all and not args.agent:
        parser.error("--agent or --all is required")
    rc = asyncio.run(main(args))
    sys.exit(2 if not args.yes else rc)


if __name__ == "__main__":
    main_cli()
