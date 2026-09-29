from cat import hook, AgenticWorkflowOutput


@hook(priority=10)
def agent_fast_reply(cat) -> AgenticWorkflowOutput | None:
    if "hello" in cat.working_memory.user_message.text:
        return AgenticWorkflowOutput(output="This is an agent fast reply")

    return None


@hook(priority=10)
def before_cat_sends_message(message, agent_output, cat):
    message.text = f"{message.text} (seen by the plugins)"
    return message
