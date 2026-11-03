"""Regression tests for the Efficient Ingestion startup ordering.

The revalidation sweep must be *deferred* to the lifecycle hook (never run at
import/construction time, and never synchronously from the hook), and it must
degrade cleanly when the calling agent's plugin manager has not registered the
``ingestion_phase_pending`` hook — for example an agent whose stored
``active_plugins`` predates this plugin. Before the fix the bare
``execute_hook`` raised ``Hook ingestion_phase_pending not present in any
plugin`` in a burst at boot.
"""

import asyncio
from unittest.mock import AsyncMock

from cat.core_plugins.cat_efficient_ingestion import resume


async def test_first_revalidation_is_deferred_to_the_lifecycle_hook(monkeypatch):
    """``after_lizard_bootstrap`` only schedules the pass: nothing runs inline."""
    scheduled = []

    def fake_ensure_future(coro):
        scheduled.append(coro)
        return coro

    monkeypatch.setattr(asyncio, "ensure_future", fake_ensure_future)
    monkeypatch.setenv("CAT_INGESTION_RESUME_INTERVAL_SECONDS", "0")

    # returning normally proves the hook itself never revalidates inline
    await resume.after_lizard_bootstrap.function(None)

    assert len(scheduled) == 1
    assert scheduled[0].cr_code.co_name == "_startup_pass"
    # the coroutine is only scheduled, not executed: the first revalidation
    # cannot run before the plugin manager has registered the hooks
    assert scheduled[0].cr_frame is not None
    scheduled[0].close()


async def test_revalidation_without_registered_phase_hook_does_not_raise(monkeypatch):
    """A missing phase hook degrades to "no pending phases", never an exception."""

    class _Manager:
        hooks = {}

        async def execute_hook(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("the bare dispatcher must not be called for a missing hook")

    class _Cat:
        agent_key = "agent_1"
        plugin_manager = _Manager()

    monkeypatch.setattr(resume, "ingestion_canceled", AsyncMock(return_value=False))
    monkeypatch.setattr(
        resume,
        "list_statuses",
        AsyncMock(return_value=[{
            "source": "doc.pdf",
            "scope": "agent",
            "type": "file",
            "status": "completed",
        }]),
    )
    monkeypatch.setattr(
        resume, "backfill_completed_phases",
        AsyncMock(return_value={"parsing_chunking": {"marker": "m"}}),
    )
    reembed = AsyncMock()
    monkeypatch.setattr(resume, "reembed_sources", reembed)

    await resume._revalidate_agent(object(), "agent_1", ccat=_Cat())

    reembed.assert_not_awaited()
