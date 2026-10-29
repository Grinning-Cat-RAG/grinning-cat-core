import pytest

from cat import AgenticWorkflowOutput, CatMessage
from cat.exceptions import CustomHookVetoException
from cat.looking_glass.mad_hatter.decorators.hook import CatHook


def test_hook_discovery(plugin_manager):
    mock_plugin_hooks = plugin_manager.plugins["mock_plugin"].hooks

    assert len(mock_plugin_hooks) == 3
    for h in mock_plugin_hooks:
        assert isinstance(h, CatHook)
        assert h.plugin_id == "mock_plugin"


async def test_hook_async_priority_execution(stray):
    fake_message = CatMessage(text="Priorities:")
    agent_output = AgenticWorkflowOutput()

    out = await stray.plugin_manager.execute_hook("before_cat_sends_message", fake_message, agent_output, caller=stray)
    assert out.text == "Priorities: priority 3 priority 2"


async def test_hook_veto_propagates_and_stops_the_chain(stray):
    """A CustomHookVetoException crosses _run_hooks and short-circuits it.

    The veto is a deliberate refusal, not an error: it must reach the caller
    and the remaining hooks in the chain must NOT run.
    """
    calls = []

    def veto(source, cat):
        calls.append("veto")
        raise CustomHookVetoException("refused")

    def after(source, cat):
        calls.append("after")

    hooks = [
        CatHook("veto_chain", veto, 10, plugin_id="mock_plugin"),
        CatHook("veto_chain", after, 0, plugin_id="mock_plugin"),
    ]

    with pytest.raises(CustomHookVetoException, match="refused"):
        await stray.plugin_manager._run_hooks(hooks, "doc.pdf", caller=stray)

    assert calls == ["veto"]


async def test_plain_hook_exception_is_still_swallowed(stray):
    """Ordinary hook errors keep the legacy behaviour: log and continue.

    Only the dedicated veto escapes; a RuntimeError must not abort the chain
    nor reach the caller.
    """
    calls = []

    def boom(source, cat):
        calls.append("boom")
        raise RuntimeError("ordinary failure")

    def after(source, cat):
        calls.append("after")

    hooks = [
        CatHook("boom_chain", boom, 10, plugin_id="mock_plugin"),
        CatHook("boom_chain", after, 0, plugin_id="mock_plugin"),
    ]

    result = await stray.plugin_manager._run_hooks(hooks, "doc.pdf", caller=stray)

    assert calls == ["boom", "after"]
    assert result == "doc.pdf"
