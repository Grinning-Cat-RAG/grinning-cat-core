"""Ghost-agent cleanup CLI: dry-run safety, deletion, skip reasons, isolation.

Every test seeds its own uniquely-named agents and removes them in ``finally``.
The suite runs against the dedicated test Redis db forced by ``tests/conftest.py``.
"""
import asyncio
import json
from uuid import uuid4

import migrations.cleanup_ghost_agents as cleanup_ghost_agents
from cat.db import crud
from cat.db.database import Database


def _reset_client() -> None:
    """Drop the cached async client so the next ``asyncio.run`` gets its own.

    ``Database`` is a process singleton bound to the event loop that created the
    client; each ``asyncio.run`` below uses a fresh loop.
    """
    Database().reset_async()


def _run(coro):
    _reset_client()
    return asyncio.run(coro)


def _run_main(argv):
    _reset_client()
    return cleanup_ghost_agents.main(argv)


async def _seed(keys):
    for key, value in keys.items():
        await crud.store(key, value)


async def _read(key):
    return await crud.read(key)


def _cleanup(keys) -> None:
    async def _delete_all():
        for key in keys:
            try:
                await crud.delete(key)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    _run(_delete_all())


def _agent_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def test_dry_run_is_aborted_and_writes_nothing():
    ghost = _agent_id("ghost-dry")
    ghost_keys = [
        f"agents:{ghost}:conversations:alice:chat",
        f"agents:{ghost}:settings:llm",
    ]
    keys: dict[str, dict] = {key: {"value": 1} for key in ghost_keys}

    try:
        _run(_seed(keys))

        assert _run_main([]) == 2

        for key in ghost_keys:
            assert _run(_read(key)) is not None
    finally:
        _cleanup(keys)


def test_apply_removes_ghost_and_keeps_healthy_agent():
    ghost = _agent_id("ghost-apply")
    healthy = _agent_id("healthy")
    marker_only = _agent_id("marker")

    ghost_keys = [
        f"agents:{ghost}:conversations:alice:chat",
        f"agents:{ghost}:users:alice",
    ]
    healthy_master = f"agents:{healthy}:agent"
    healthy_peripheral = f"agents:{healthy}:conversations:alice:chat"
    marker_key = f"agents:{marker_only}:ingestion:delete"
    keys: dict[str, dict] = {key: {"value": 1} for key in ghost_keys}
    keys.update({healthy_master: {"name": "metadata"}, healthy_peripheral: {"c": 1}, marker_key: {"delete_marker": True}})

    try:
        _run(_seed(keys))

        assert _run_main(["--apply"]) == 0

        for key in ghost_keys:
            assert _run(_read(key)) is None
        assert _run(_read(healthy_master)) is not None
        assert _run(_read(healthy_peripheral)) is not None
        assert _run(_read(marker_key)) is not None
    finally:
        _cleanup(keys)


def test_apply_reports_skipped_agents(capsys):
    healthy = _agent_id("healthy-json")
    marker_only = _agent_id("marker-json")
    healthy_master = f"agents:{healthy}:agent"
    healthy_peripheral = f"agents:{healthy}:plugins:base"
    marker_key = f"agents:{marker_only}:ingestion:delete"
    keys: dict[str, dict] = {healthy_master: {"name": "metadata"}, healthy_peripheral: {"p": 1}, marker_key: {"delete_marker": True}}

    try:
        _run(_seed(keys))

        assert _run_main(["--json"]) == 2

        report = json.loads(capsys.readouterr().out)
        reasons = {entry["agent_id"]: entry["reason"] for entry in report["skipped"]}
        assert reasons[healthy] == "has_master_key"
        assert reasons[marker_only] == "has_delete_marker"
    finally:
        _cleanup(keys)


def test_prefix_ghost_cannot_delete_ambiguous_live_agent(capsys):
    # Dangerous shape: ghost ``a`` has ONLY ``settings`` (no master key, no
    # marker), while live ``a:b`` owns a master key and a conversation. The glob
    # ``agents:a:*`` matches ``agents:a:b:*`` too, so on the pre-fix logic the
    # CLI reaps ghost ``a`` and destroys live ``a:b`` — this test fails there.
    keys: dict[str, dict] = {
        "agents:a:settings:x": {"value": 1},
        "agents:a:b:agent": {"name": "metadata-a-b"},
        "agents:a:b:conversations:u1": {"chat": 1},
        "agents:g:users:u1": {"user": 1},
    }

    try:
        _run(_seed(keys))

        # Dry run: exit 2 and not a single key touched.
        assert _run_main([]) == 2
        capsys.readouterr()
        for key in keys:
            assert _run(_read(key)) is not None

        # Machine-readable dry run names the ambiguous live namespace.
        assert _run_main(["--json"]) == 2
        report = json.loads(capsys.readouterr().out)
        reasons = {entry["agent_id"]: entry["reason"] for entry in report["skipped"]}
        assert reasons.get("a:b") == "ambiguous_id"
        assert reasons.get("a") == "ambiguous_id"

        assert _run_main(["--apply"]) == 0

        # The live agent and the prefix-guarded ghost survive; only the genuine
        # ghost is removed.
        assert _run(_read("agents:a:b:agent")) is not None
        assert _run(_read("agents:a:b:conversations:u1")) is not None
        assert _run(_read("agents:a:settings:x")) is not None
        assert _run(_read("agents:g:users:u1")) is None
    finally:
        _cleanup(keys)


def test_partial_failure_returns_1(capsys, monkeypatch):
    failing_ghost = _agent_id("ghost-fail")
    succeeding_ghost = _agent_id("ghost-ok")
    healthy = _agent_id("healthy-partial")

    failing_keys = [f"agents:{failing_ghost}:conversations:alice:chat"]
    succeeding_keys = [f"agents:{succeeding_ghost}:users:alice"]
    healthy_master = f"agents:{healthy}:agent"
    keys: dict[str, dict] = {key: {"value": 1} for key in failing_keys + succeeding_keys}
    keys[healthy_master] = {"name": "metadata"}

    real_destroy = crud.destroy

    async def flaky_destroy(pattern, exclude=None):
        if failing_ghost in pattern:
            raise RuntimeError("boom")
        return await real_destroy(pattern, exclude=exclude)

    monkeypatch.setattr(cleanup_ghost_agents.crud, "destroy", flaky_destroy)

    try:
        _run(_seed(keys))

        assert _run_main(["--apply", "--json"]) == 1

        report = json.loads(capsys.readouterr().out)
        assert [entry["agent_id"] for entry in report["failed"]] == [failing_ghost]
        assert [entry["agent_id"] for entry in report["actions"]] == [succeeding_ghost]

        assert _run(_read(failing_keys[0])) is not None
        assert _run(_read(succeeding_keys[0])) is None
        assert _run(_read(healthy_master)) is not None
    finally:
        monkeypatch.setattr(cleanup_ghost_agents.crud, "destroy", real_destroy)
        _cleanup(keys)

