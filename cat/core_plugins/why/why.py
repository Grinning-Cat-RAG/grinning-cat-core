from cat import hook, MessageWhy, CatMessage, AgenticWorkflowOutput


@hook(priority=2)
def before_cat_sends_message(message: CatMessage, agent_output: AgenticWorkflowOutput, cat) -> CatMessage:
    memory = [
        dict(d.document)
        | {
            "score": float(d.score) if d.score else None,
            "id": d.id,
        }
        for d in cat.working_memory.context_memories
    ]

    # why this response?
    message.why = MessageWhy(
        input=cat.working_memory.user_message.text,
        intermediate_steps=agent_output.intermediate_steps,
        memory=memory,
    )

    return message
