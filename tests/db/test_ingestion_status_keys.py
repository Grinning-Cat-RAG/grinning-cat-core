"""Tests for the narrow, agent-scoped ingestion status key listing.

``cat.db.cruds.settings.get_ingestion_status_keys`` is the read-only API the
ingestion plugins use to discover their agent's status rows. It must be scoped
to a single agent (no cross-agent leaks), escape glob metacharacters in the
agent id, and never write or delete anything.
"""
import uuid

from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_AGENTS_KEY


def _unique_agent_id(prefix: str = "test-ing") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _ingestion_key(agent_id: str, name: str) -> str:
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:ingestion:{name}"


async def _seed(keys: list[str]) -> None:
    for key in keys:
        await crud.store(key, {"status": "uploaded", "source": key})


async def _cleanup(keys: list[str]) -> None:
    for key in keys:
        await crud.delete(key)


async def test_returns_exactly_the_seeded_keys_sorted():
    agent_id = _unique_agent_id()
    keys = [_ingestion_key(agent_id, "b-source.pdf"), _ingestion_key(agent_id, "a-source.pdf")]
    try:
        await _seed(keys)

        found = await crud_settings.get_ingestion_status_keys(agent_id)

        assert found == sorted(keys)
    finally:
        await _cleanup(keys)


async def test_returns_empty_list_when_the_agent_has_none():
    agent_id = _unique_agent_id()

    assert await crud_settings.get_ingestion_status_keys(agent_id) == []


async def test_does_not_leak_across_agent_ids():
    agent_a = _unique_agent_id()
    agent_a2 = f"{agent_a}2"
    keys_a = [_ingestion_key(agent_a, "one"), _ingestion_key(agent_a, "two")]
    keys_a2 = [_ingestion_key(agent_a2, "other")]
    try:
        await _seed(keys_a + keys_a2)

        found = await crud_settings.get_ingestion_status_keys(agent_a)

        assert found == sorted(keys_a)
    finally:
        await _cleanup(keys_a + keys_a2)


async def test_escapes_glob_metacharacters_in_the_agent_id():
    suffix = uuid.uuid4().hex
    star_agent = f"a*{suffix}"
    ab_agent = f"ab{suffix}"
    star_key = _ingestion_key(star_agent, "y")
    ab_key = _ingestion_key(ab_agent, "x")
    try:
        await _seed([star_key, ab_key])

        found = await crud_settings.get_ingestion_status_keys(star_agent)

        assert found == [star_key]
        assert ab_key not in found
    finally:
        await _cleanup([star_key, ab_key])


async def test_is_read_only_and_leaves_every_seeded_key_readable():
    agent_id = _unique_agent_id()
    keys = [_ingestion_key(agent_id, "keep-a"), _ingestion_key(agent_id, "keep-b")]
    try:
        await _seed(keys)

        await crud_settings.get_ingestion_status_keys(agent_id)

        for key in keys:
            assert await crud.read(key) is not None
    finally:
        await _cleanup(keys)
