from typing import Dict, List, Any
from redis.exceptions import RedisError

from cat.db import crud
from cat.db.database import DEFAULT_AGENTS_KEY, DEFAULT_CONVERSATIONS_KEY, get_async_db
from cat.env import get_env_int
from cat.log import log
from cat.services.memory.messages import ConversationMessage


def _get_expiration() -> int | None:
    expiration = get_env_int("CAT_HISTORY_EXPIRATION")
    if expiration is None:
        return None

    try:
        expiration = expiration * 60
        if expiration <= 0:
            raise ValueError("CAT_HISTORY_EXPIRATION must be positive")

        return expiration
    except ValueError as e:
        message = f"Invalid CAT_HISTORY_EXPIRATION: {e}"
        log.error(message)
        raise ValueError(message)


def format_key(agent_id: str, user_id: str, chat_id: str) -> str:
    """
    Format Redis key for a conversation.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.
        chat_id: ID of the chat session.

    Returns:
        Formatted key (e.g., "agents:<agent_id>:conversations:<user_id>:<chat_id>").
    """
    return f"{DEFAULT_AGENTS_KEY}:{agent_id}:{DEFAULT_CONVERSATIONS_KEY}:{user_id}:{chat_id}"


async def get_conversation(agent_id: str, user_id: str, chat_id: str) -> Dict[str, Any] | None:
    """
    Retrieve a specific conversation from Redis.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.
        chat_id: ID of the chat session.

    Returns:
        Conversation data as a dictionary, or None if not found.
    """
    try:
        conversation = await crud.read(format_key(agent_id, user_id, chat_id))
        if not conversation:
            return None

        if isinstance(conversation, list):
            conversation = conversation[0]

        return conversation
    except RedisError as e:
        log.error(f"Failed to get conversation '{chat_id}' for '{agent_id}:{user_id}': {e}")
        raise


async def get_conversations_attributes(agent_id: str, user_id: str, with_user: bool = False) -> List[Dict[str, Any]]:
    """
    Retrieve conversations parameters from Redis.

    Uses a single SCAN + one MGET pipeline instead of N+1 individual round-trips.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.
        with_user: Whether to include the user_id in the results (default: False).

    Returns:
        List of dictionaries, each one having the format
            {
                "user_id": str,
                "chat_id": str,
                "name": str,
                "num_messages": int,
                "metadata": dict,
                "created_at": datetime,
                "updated_at": datetime,
            }

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        db = get_async_db()
        pattern = format_key(agent_id, user_id, "*")

        # One SCAN to collect all conversation keys
        keys = [k async for k in db.scan_iter(match=pattern)]
        if not keys:
            return []

        # One MGET to read every conversation in a single round-trip
        raw_results = await db.json().mget(keys, "$")

        results = []
        for key_str, raw in zip(keys, raw_results):
            if not raw:
                continue
            parts = key_str.split(":")
            if len(parts) != 5:
                continue
            _, _agent_id, _, _user_id, chat_id = parts

            conversation = raw[0] if isinstance(raw, list) else raw
            messages = conversation.get("messages", [])
            num_messages = len(messages)
            result = {
                "chat_id": chat_id,
                "name": conversation.get("name", chat_id),
                "num_messages": num_messages,
                "metadata": conversation.get("metadata", {}),
                "created_at": messages[0]["when"] if num_messages > 0 else None,
                "updated_at": messages[-1]["when"] if num_messages > 0 else None,
            }
            if with_user:
                result["user_id"] = _user_id

            results.append(result)

        return results
    except RedisError as e:
        log.error(f"Failed to get chat attributes for '{agent_id}:{user_id}': {e}")
        raise


async def get_conversation_attributes(agent_id: str, user_id: str, chat_id: str) -> Dict[str, Any] | None:
    """
    Retrieve conversation parameters from Redis.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.
        chat_id: ID of the chat session.

    Returns:
        Dictionary having the format
            {
                "chat_id": str,
                "name": str,
                "num_messages": int,
                "metadata": dict,
                "created_at": datetime,
                "updated_at": datetime,
            }
        or None if conversation not found.
    """
    try:
        conversation = await get_conversation(agent_id, user_id, chat_id)
        if not conversation:
            return None

        messages = conversation.get("messages", [])
        num_messages = len(messages)

        # Extract created_at (first message) and updated_at (last message)
        created_at = messages[0]["when"] if num_messages > 0 else None
        updated_at = messages[-1]["when"] if num_messages > 0 else None

        # Retrieve the conversation name for the specific key
        return {
            "chat_id": chat_id,
            "name": conversation.get("name", chat_id),
            "num_messages": num_messages,
            "metadata": conversation.get("metadata", {}),
            "created_at": created_at,
            "updated_at": updated_at,
        }
    except RedisError as e:
        log.error(f"Failed to get chat attributes for '{agent_id}:{user_id}': {e}")
        raise


async def get_messages(agent_id: str, user_id: str, chat_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve conversation messages from Redis.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.
        chat_id: ID of the chat session.

    Returns:
        List of conversation messages, or empty list if not found.

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        messages = await crud.read(format_key(agent_id, user_id, chat_id), path="$.messages")
        return messages if messages else []  # type: ignore[return-value]
    except RedisError as e:
        log.error(f"Failed to get conversation '{chat_id}' for '{agent_id}:{user_id}': {e}")
        raise


async def set_messages(
    agent_id: str, user_id: str, chat_id: str, messages: List[ConversationMessage]
) -> List[Dict[str, Any]]:
    """
    Store conversation messages in Redis with optional TTL.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.
        chat_id: ID of the chat session.
        messages: List of conversation messages.

    Returns:
        Stored messages.

    Raises:
        ValueError: If CAT_HISTORY_EXPIRATION is invalid.
        RedisError: If Redis connection fails.
    """
    try:
        formatted = [message.model_dump() for message in messages]
        expiration = _get_expiration()
        key = format_key(agent_id, user_id, chat_id)

        existing_data = await crud.read(key)

        if existing_data is None:
            await crud.store(key, {"name": chat_id, "messages": formatted}, expire=expiration)
            return formatted

        await crud.store(key, formatted, path="$.messages", expire=expiration)
        return formatted
    except RedisError as e:
        log.error(f"Redis error storing conversation '{chat_id}' for '{agent_id}:{user_id}': {e}")
        raise


async def set_attributes(
    agent_id: str,
    user_id: str,
    chat_id: str,
    name: str | None = None,
    metadata: Dict[str, Any] | None = None,
    first_time: bool = False,
):
    """
    Set the name and/or metadata of a conversation in Redis.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.
        chat_id: ID of the chat session.
        name: New name for the conversation, if any.
        metadata: New metadata for the conversation, if any.
        first_time: Whether this is the first time the conversation is being created.

    Raises:
        RedisError: If Redis connection fails.
    """
    if not name and not metadata:
        return

    try:
        conversation = await get_conversation(agent_id, user_id, chat_id)
        if not conversation:
            return

        if name:
            await crud.store(format_key(agent_id, user_id, chat_id), name, path="$.name")

        current_metadata = conversation.get("metadata", {})
        if metadata or first_time:
            current_metadata.update(metadata)
            await crud.store(
                format_key(agent_id, user_id, chat_id), current_metadata or {}, path="$.metadata"
            )
    except RedisError as e:
        log.error(f"Redis error setting conversation name for '{agent_id}:{user_id}:{chat_id}': {e}")
        raise


async def update_messages(
    agent_id: str, user_id: str, chat_id: str, updated_info: ConversationMessage
) -> List[Dict[str, Any]]:
    """
    Append a new message to the conversation in Redis atomically.

    Uses JSON.SET NX to initialise the structure if absent, then JSON.ARRAPPEND to append the message. Both operations
    are atomic on the Redis side, so concurrent calls from multiple replicas cannot produce lost updates.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.
        chat_id: ID of the chat session.
        updated_info: New conversation message to append.

    Returns:
        Updated conversation messages.

    Raises:
        RedisError: If Redis connection fails.
        ValueError: If serialization or TTL configuration fails.
    """
    try:
        key = format_key(agent_id, user_id, chat_id)
        serialized = crud.serialize_to_redis_json(updated_info.model_dump())
        expiration = _get_expiration()

        db = get_async_db()

        # Atomically initialise the root structure only if the key does not exist yet.
        # If two replicas race here, only one SET NX will succeed; the other is a no-op,
        # and both will then safely append via ARRAPPEND.
        await db.json().set(key, "$", {"name": chat_id, "messages": []}, nx=True)

        # ARRAPPEND is a single atomic Redis command: no read-modify-write, no lost updates.
        await db.json().arrappend(key, "$.messages", serialized)  # type: ignore[arg-type]

        if expiration:
            await db.expire(key, expiration)

        log.debug(f"Appended conversation item for {agent_id}:{user_id}")
        return await get_messages(agent_id, user_id, chat_id)
    except (RedisError, ValueError) as e:
        log.error(f"Redis error updating conversation '{chat_id}' for '{agent_id}:{user_id}': {e}")
        raise


async def delete_conversation(agent_id: str, user_id: str, chat_id: str):
    """
    Delete conversation for a specific user and agent.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.
        chat_id: ID of the chat session.

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        return await crud.delete(format_key(agent_id, user_id, chat_id))
    except RedisError as e:
        log.error(f"Redis error deleting conversation '{chat_id}' for '{agent_id}:{user_id}': {e}")
        raise


async def delete_conversations(agent_id: str, user_id: str) -> int:
    """
    Delete all conversations for a specific user and agent.

    The conversations are stored one key per chat, so the removal is pattern-based
    (``crud.destroy``): ``crud.delete`` targets a single exact key and would leave
    every conversation behind, matching nothing for a wildcard.

    Args:
        agent_id: ID of the chatbot.
        user_id: ID of the user.

    Returns:
        Number of conversations deleted.

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        return await crud.destroy(format_key(agent_id, user_id, "*"))
    except RedisError as e:
        log.error(f"Redis error deleting conversations for {agent_id}:{user_id}: {e}")
        raise


async def destroy_all(agent_id: str):
    """
    Delete all conversations for a specific agent.

    Args:
        agent_id: ID of the chatbot.

    Raises:
        RedisError: If Redis connection fails.
    """
    try:
        await crud.destroy(format_key(agent_id, "*", "*"))
    except RedisError as e:
        log.error(f"Redis error destroying conversations for {agent_id}: {e}")
        raise


async def get_user_id_from_conversation_keys(agent_id: str, chat_id: str) -> str | None:
    pattern = format_key(agent_id, "*", chat_id)

    user_ids = list({k.split(":")[3] async for k in get_async_db().scan_iter(pattern)})
    return user_ids[0] if len(user_ids) == 1 else None
