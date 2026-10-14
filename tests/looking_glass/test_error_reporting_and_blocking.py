"""
The LangChain callbacks forward no error frames (one failure fired one per nested run); an error escaping the chat
turn is reported once by StrayCat.run_websocket. The event loop serving every client is never blocked by downloads
made while preparing the agent's input.
"""
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from cat.looking_glass.callbacks import WebSocketCallbackManager
from cat.looking_glass.stray_cat import StrayCat


@tool
def broken_tool(x: str) -> str:
    """A tool that fails."""
    raise ValueError("tool exploded")


class _ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_a_failing_agent_sends_no_error_from_the_callbacks():
    # LangChain reports one failure once per nested run (tool + each chain): forwarding them sent 3 errors
    notifier = MagicMock(
        send_error=AsyncMock(), send_tool_message=AsyncMock(), send_chat_token=AsyncMock(),
        send_llm_thinking=AsyncMock(), send_thought_message=AsyncMock(), send_context=AsyncMock(),
    )
    model = _ToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "broken_tool", "args": {"x": "1"}, "id": "call-1"}]),
    ]))
    agent = create_agent(model=model, tools=[broken_tool], system_prompt="system")

    async def run():
        with pytest.raises(ValueError, match="tool exploded"):
            await agent.ainvoke(
                {"messages": [HumanMessage(content="hi")]},
                config=RunnableConfig(callbacks=[WebSocketCallbackManager(notifier)]),
            )

    asyncio.run(run())

    notifier.send_error.assert_not_awaited()


def test_an_error_escaping_the_turn_is_reported_once(monkeypatch):
    stray = object.__new__(StrayCat)
    stray.shutdown = lambda: None  # never initialised: nothing to release when it is garbage collected
    stray.notifier = MagicMock(send_error=AsyncMock(), send_chat_message=AsyncMock())
    monkeypatch.setattr(StrayCat, "agent_key", property(lambda self: "agent"))

    async def failing_call(self, user_message):
        raise ValueError("llm down")

    monkeypatch.setattr(StrayCat, "__call__", failing_call)

    asyncio.run(stray.run_websocket(MagicMock()))

    stray.notifier.send_error.assert_awaited_once()


def test_the_history_with_images_is_prepared_off_the_event_loop():
    threads = []

    class _Item:
        def langchainfy(self):
            threads.append(threading.current_thread())  # may download an image: blocking I/O
            return HumanMessage(content="old message")

    history = asyncio.run(StrayCat._langchain_history([_Item(), _Item(), _Item()], 2))

    assert [m.content for m in history] == ["old message", "old message"]
    assert len(threads) == 2 and all(t is not threading.main_thread() for t in threads)


def test_image_downloads_have_a_timeout(monkeypatch):
    import cat.utils as cat_utils

    seen = {}

    def get(url, **kwargs):
        seen.update(kwargs)
        raise cat_utils.requests.Timeout("no answer")

    monkeypatch.setattr(cat_utils.requests, "get", get)

    assert cat_utils.retrieve_image("http://images.invalid/a.png") is None
    assert seen.get("timeout")


def _stray_for_a_turn(monkeypatch, run_agent):
    """A StrayCat running its real __call__, with the external collaborators (embedder, memories, plugins) faked."""
    from types import SimpleNamespace

    from cat.looking_glass.models import AgenticWorkflowOutput  # noqa: F401 - used by the callers' fakes

    stray = object.__new__(StrayCat)
    stray.shutdown = lambda: None  # never initialised: nothing to release when it is garbage collected
    stray._agent_id = "agent"

    async def execute_hook(name, *args, caller=None):
        if name in ("fast_reply", "agent_fast_reply"):
            return None
        return args[0] if args else None

    stray._plugin_manager = SimpleNamespace(execute_hook=execute_hook)
    stray.working_memory = SimpleNamespace(model_interactions=set(), user_message=None, context_memories=[], history=[])
    embedder = SimpleNamespace(embed_query=lambda text: [0.0, 1.0])
    monkeypatch.setattr(StrayCat, "lizard", property(lambda self: SimpleNamespace(embedder=AsyncMock(return_value=embedder))))
    monkeypatch.setattr(StrayCat, "get_procedures", AsyncMock(return_value=[]))
    monkeypatch.setattr(StrayCat, "recall_context_to_working_memory", AsyncMock())
    monkeypatch.setattr(StrayCat, "_run_agent", run_agent)
    return stray


def test_a_failed_turn_carries_its_error_in_the_answer(monkeypatch):
    from cat.services.memory.messages import UserMessage

    async def failing_agent(self, system_prompt, config, procedures_task):
        procedures_task.cancel()
        raise RuntimeError("llm down")

    stray = _stray_for_a_turn(monkeypatch, failing_agent)

    answer = asyncio.run(stray(UserMessage(text="hi")))

    assert answer.error == "llm down"  # a client can tell the failure without reading the text
    assert "llm down" in answer.text


def test_a_turn_without_a_configured_llm_carries_an_error(monkeypatch):
    from cat import utils
    from cat.looking_glass.models import AgenticWorkflowOutput
    from cat.services.memory.messages import UserMessage

    async def unconfigured_agent(self, system_prompt, config, procedures_task):
        await procedures_task
        return AgenticWorkflowOutput(output=utils.default_llm_answer_prompt(), with_llm_error=True)

    stray = _stray_for_a_turn(monkeypatch, unconfigured_agent)

    answer = asyncio.run(stray(UserMessage(text="hi")))

    assert answer.error == utils.default_llm_answer_prompt()


def test_a_successful_turn_has_no_error(monkeypatch):
    from cat.looking_glass.models import AgenticWorkflowOutput
    from cat.services.memory.messages import UserMessage

    async def agent(self, system_prompt, config, procedures_task):
        await procedures_task
        return AgenticWorkflowOutput(output="the answer")

    stray = _stray_for_a_turn(monkeypatch, agent)

    answer = asyncio.run(stray(UserMessage(text="hi")))

    assert answer.text == "the answer"
    assert answer.error is None


def test_a_failure_without_a_message_is_named_by_its_type(monkeypatch):
    from cat.services.memory.messages import UserMessage

    async def failing_agent(self, system_prompt, config, procedures_task):
        procedures_task.cancel()
        raise TimeoutError()

    stray = _stray_for_a_turn(monkeypatch, failing_agent)

    assert asyncio.run(stray(UserMessage(text="hi"))).error == "TimeoutError"
