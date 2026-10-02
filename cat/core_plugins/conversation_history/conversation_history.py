from cat import hook, UserMessage, CatMessage, AgenticWorkflowOutput
from cat.db.cruds import conversations as crud_conversations


@hook(priority=2)
async def before_cat_reads_message(user_message: UserMessage, cat) -> UserMessage:
    """
    Note: this hook runs before the cat processes the user message. `cat` is the StrayCat instance.
    It updates the conversation history with the user's message and sets the conversation name if it's the first
    message.
    """
    # update conversation history (user turn)
    await cat.working_memory.update_history(who="user", content=user_message)
    # if first message, set conversation name
    if len(cat.working_memory.history) == 1:
        # first message, set name and metadata in the conversation
        await crud_conversations.set_attributes(
            cat.agent_key, cat.user.id, cat.id, name=cat.id, metadata={}, first_time=True
        )

    return user_message


@hook(priority=1)
async def before_cat_sends_message(message: CatMessage, agent_output: AgenticWorkflowOutput, cat) -> CatMessage:
    """
    Note: this hook runs before the cat processes the agent message. `cat` is the StrayCat instance.
    It updates the conversation history with the agent's message unless there was an LLM error.
    """
    if agent_output.with_llm_error:
        await cat.working_memory.pop_last_message_if_human()
        return message

    # update conversation history (AI turn)
    await cat.working_memory.update_history(who="assistant", content=message)
    return message
