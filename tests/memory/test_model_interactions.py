import copy
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cat.core_plugins.interactions.handlers import ModelInteractionHandler
from cat.core_plugins.interactions.models import LLMModelInteraction


class Stray:
    def __init__(self):
        self.working_memory = SimpleNamespace(model_interactions=set())


def a_stray():
    return Stray()


def callbacks_of(stray):
    handler = ModelInteractionHandler("test")
    handler.inject_stray_cat(stray)
    # the core deep-copies the callbacks returned by the hooks
    return copy.deepcopy([handler])


def recorded(stray):
    # in the order of the calls: the fake models answer at once, so two calls may start at the same time
    return sorted(stray.working_memory.model_interactions, key=lambda i: (i.started_at, i.ended_at, i.reply))


async def test_every_llm_call_of_a_turn_is_recorded():
    # e.g. an agent calling the LLM several times, with the same prompt too
    stray = a_stray()
    callbacks = callbacks_of(stray)
    llm = FakeListChatModel(responses=["first", "second", "third"], cache=False)
    for prompt in ("question", "question", "other question"):
        await llm.ainvoke(prompt, config={"callbacks": callbacks})

    interactions = recorded(stray)
    # the fake model answers at once: the calls may have the same times, the order is not checked
    assert sorted((i.prompt, i.reply) for i in interactions) == [
        (["other question"], "third"), (["question"], "first"), (["question"], "second")
    ]
    assert all(i.started_at <= i.ended_at for i in interactions)
    assert all(i.input_tokens > 0 and i.output_tokens > 0 for i in interactions)


async def test_calls_of_text_llms_are_recorded():
    stray = a_stray()
    callbacks = callbacks_of(stray)
    llm = FakeListLLM(responses=["one", "two"], cache=False)
    await llm.ainvoke("first prompt", config={"callbacks": callbacks})
    await llm.ainvoke("second prompt", config={"callbacks": callbacks})

    assert sorted((i.prompt, i.reply) for i in recorded(stray)) == [(["first prompt"], "one"), (["second prompt"], "two")]


async def test_a_failed_call_is_not_recorded_and_the_next_ones_are():
    stray = a_stray()
    callbacks = callbacks_of(stray)

    class Failing(FakeListChatModel):
        async def _agenerate(self, *args, **kwargs):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        await Failing(responses=["x"], cache=False).ainvoke("question", config={"callbacks": callbacks})
    await FakeListChatModel(responses=["answer"], cache=False).ainvoke("question", config={"callbacks": callbacks})

    assert [(i.prompt, i.reply) for i in recorded(stray)] == [(["question"], "answer")]


async def test_calls_of_different_turns_are_not_mixed():
    first, second = a_stray(), a_stray()
    first_callbacks, second_callbacks = callbacks_of(first), callbacks_of(second)
    await FakeListChatModel(responses=["a"], cache=False).ainvoke("to the first", config={"callbacks": first_callbacks})
    await FakeListChatModel(responses=["b"], cache=False).ainvoke("to the second", config={"callbacks": second_callbacks})

    assert [i.reply for i in recorded(first)] == ["a"]
    assert [i.reply for i in recorded(second)] == ["b"]


def test_the_registry_does_not_keep_the_strays():
    import gc
    from cat.core_plugins.interactions import handlers

    stray = a_stray()
    callbacks_of(stray)  # a turn without LLM calls
    key = id(stray)
    del stray
    gc.collect()
    assert key not in handlers._stray_registry


async def test_replies_made_of_content_blocks_are_recorded():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    stray = a_stray()
    handler, = callbacks_of(stray)
    handler.on_chat_model_start({}, [[AIMessage(content=["plain", {"type": "text", "text": "block"}])]], run_id="r")
    message = AIMessage(content=[{"type": "text", "text": "the "}, {"type": "tool_use", "id": "t"}, "answer"])
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id="r")
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id="unknown")

    interaction, = recorded(stray)
    assert (interaction.prompt, interaction.reply) == (["plain", "block"], "the answer")


def test_every_interaction_is_distinct():
    # regression: two calls with the same prompt (even at the same time) were one interaction in the set of the turn
    one = LLMModelInteraction(source="s", prompt=["p"], reply="r", input_tokens=1, output_tokens=1, ended_at=2.0, started_at=1.0)
    same_time = LLMModelInteraction(source="s", prompt=["p"], reply="r", input_tokens=1, output_tokens=1, ended_at=2.0, started_at=1.0)
    assert len({one, same_time, one}) == 2
    assert one == one.model_copy() and one != same_time


def test_replies_that_are_not_text():
    assert ModelInteractionHandler._text(None) == ""
    assert ModelInteractionHandler._text(42) == "42"
