import inspect
from uuid import uuid4

import cat.db.crud as crud


def _unique_prefix() -> str:
    return f"test:crud:scan:{uuid4().hex}:"


async def _cleanup(keys: list[str]) -> None:
    for key in keys:
        await crud.delete(key)


async def test_scan_keys_returns_seeded_keys_sorted():
    prefix = _unique_prefix()
    keys = [f"{prefix}{i}" for i in range(3)]
    try:
        for i, key in enumerate(keys):
            await crud.store(key, {"i": i})

        found = await crud.scan_keys(f"{prefix}*")

        assert found == sorted(keys)
    finally:
        await _cleanup(keys)


async def test_scan_keys_returns_empty_for_no_match():
    prefix = _unique_prefix()

    found = await crud.scan_keys(f"{prefix}*")

    assert found == []


async def test_destroy_with_exclude_keeps_the_excluded_key():
    prefix = _unique_prefix()
    keys = [f"{prefix}{i}" for i in range(3)]
    try:
        for i, key in enumerate(keys):
            await crud.store(key, {"i": i})

        deleted = await crud.destroy(f"{prefix}*", exclude={keys[1]})

        assert deleted == 2
        assert await crud.read(keys[1]) == [{"i": 1}]
        assert await crud.read(keys[0]) is None
        assert await crud.read(keys[2]) is None
    finally:
        await _cleanup(keys)


async def test_destroy_without_exclude_removes_everything():
    prefix = _unique_prefix()
    keys = [f"{prefix}{i}" for i in range(3)]
    try:
        for i, key in enumerate(keys):
            await crud.store(key, {"i": i})

        deleted = await crud.destroy(f"{prefix}*")

        assert deleted == 3
        for key in keys:
            assert await crud.read(key) is None
    finally:
        await _cleanup(keys)


def test_destroy_signature_is_backwards_compatible():
    params = inspect.signature(crud.destroy).parameters

    assert "exclude" in params
    assert params["exclude"].default is None
