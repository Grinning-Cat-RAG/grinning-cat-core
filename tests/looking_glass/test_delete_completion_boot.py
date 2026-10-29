"""Tests for the startup completion of interrupted agent deletions.

``BillTheLizard.bootstrap`` schedules ``complete_pending_deletes`` as a
fire-and-forget task, behind ``CAT_DELETE_COMPLETION_ON_BOOT``. These tests are
hermetic: they exercise the flag helper and the scheduler directly, never
booting the whole application.
"""
import pytest

from cat.looking_glass.bill_the_lizard import (
    BillTheLizard,
    _delete_completion_enabled,
)


def _bare_lizard():
    """Build a BillTheLizard without running __init__ or touching Redis."""
    lizard = object.__new__(BillTheLizard.__wrapped__)
    lizard._delete_completion_task = None
    return lizard


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("", False),
    ],
)
def test_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("CAT_DELETE_COMPLETION_ON_BOOT", value)
    assert _delete_completion_enabled() is expected


def test_flag_defaults_to_true_when_unset(monkeypatch):
    monkeypatch.delenv("CAT_DELETE_COMPLETION_ON_BOOT", raising=False)
    assert _delete_completion_enabled() is True


def _patch_create_task(monkeypatch):
    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        return object()

    monkeypatch.setattr("cat.looking_glass.bill_the_lizard.asyncio.create_task", fake_create_task)
    return scheduled


def test_schedules_when_enabled(monkeypatch):
    monkeypatch.setenv("CAT_DELETE_COMPLETION_ON_BOOT", "true")
    scheduled = _patch_create_task(monkeypatch)
    lizard = _bare_lizard()

    lizard._schedule_delete_completion()

    assert len(scheduled) == 1
    assert scheduled[0].cr_code.co_name == "_complete_pending_deletes_safely"
    assert lizard._delete_completion_task is not None
    scheduled[0].close()


def test_does_not_schedule_when_disabled(monkeypatch):
    monkeypatch.setenv("CAT_DELETE_COMPLETION_ON_BOOT", "false")
    scheduled = _patch_create_task(monkeypatch)
    lizard = _bare_lizard()

    lizard._schedule_delete_completion()

    assert scheduled == []
    assert lizard._delete_completion_task is None


async def test_wrapper_logs_failure_without_raising(monkeypatch):
    async def boom():
        raise RuntimeError("teardown failed")

    monkeypatch.setattr(
        "cat.looking_glass.bill_the_lizard.complete_pending_deletes", boom
    )
    errors = []
    monkeypatch.setattr(
        "cat.looking_glass.bill_the_lizard.log.error", lambda message: errors.append(message)
    )
    lizard = _bare_lizard()

    await lizard._complete_pending_deletes_safely()

    assert errors == ["Could not complete pending agent deletions at boot: teardown failed"]
