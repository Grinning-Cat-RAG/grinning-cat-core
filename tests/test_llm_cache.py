from langchain_core.globals import get_llm_cache


async def test_the_answers_of_the_llm_are_not_cached(client):
    # regression: a cache of the process (never emptied) answered the same prompt with the same reply, to every agent
    # (tenant) and every conversation, and made the LLM calls look paid
    assert get_llm_cache() is None
