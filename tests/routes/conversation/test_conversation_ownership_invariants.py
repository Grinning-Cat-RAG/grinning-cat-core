"""Stateful, property-based test of the ownership of the conversations (requires ``hypothesis``, skipped otherwise).

Several users (and the system administrators) use a few chat ids at the same time, from several instances;
conversations are deleted and Redis fails in the middle of a claim.

Invariants:

- ownership: a chat id belongs to at most one user, the first one who used it; every other user is refused, until the
  owner deletes the conversation;
- administrators: the system users access every conversation and never own one;
- resilience: a failure of Redis while claiming a chat id refuses the request and leaves the ownership unchanged.
"""
import asyncio
import contextvars
from unittest import mock

import pytest

from cat.db import crud
from cat.db.cruds import conversations as crud_conversations

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, settings, strategies as st  # noqa: E402
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test  # noqa: E402

USERS = ("u1", "u2", "u3")
CHATS = ("c1", "c:2", "c*")
FAIL = contextvars.ContextVar("fail", default=False)


def build_machine(agent_prefix: str):
    request = st.fixed_dictionaries({
        "user": st.sampled_from([*USERS, "admin"]),
        "chat": st.sampled_from(CHATS),
        "fail": st.sampled_from([None, None, "set", "get"]),
    })

    class OwnershipMachine(RuleBasedStateMachine):
        counter = 0

        def __init__(self):
            super().__init__()
            OwnershipMachine.counter += 1
            self.agent = f"{agent_prefix}-{OwnershipMachine.counter}"
            self.owners = {}  # the model: chat -> owner

        def access(self, user, chat):
            """What the authentication does with a request of the user on the chat id: True if it is allowed."""
            async def run():
                if user == "admin":
                    return True
                return await crud_conversations.claim_conversation(self.agent, chat, user) == user
            return run()

        @rule(requests=st.lists(request, min_size=1, max_size=6))
        def users_write_together(self, requests):
            async def one(r):
                FAIL.set(r["fail"])
                try:
                    return await self.access(r["user"], r["chat"])
                except ConnectionError:
                    return None  # refused

            async def together():
                return await asyncio.gather(*(one(r) for r in requests))

            results = LOOP.run_until_complete(together())
            winners = {}
            for r, allowed in zip(requests, results):
                if r["user"] == "admin":
                    assert allowed
                    continue
                if allowed is None:
                    continue
                if r["chat"] not in self.owners and allowed:
                    winners.setdefault(r["chat"], set()).add(r["user"])
                expected_owner = self.owners.get(r["chat"])
                if expected_owner is not None:
                    assert allowed == (r["user"] == expected_owner), (r, expected_owner)
            for chat, users in winners.items():
                assert len(users) == 1, ("two users own the same chat id", chat, users)
                self.owners[chat] = users.pop()

        @rule(chat=st.sampled_from(CHATS), user=st.sampled_from(USERS))
        def owner_deletes_the_conversation(self, chat, user):
            if self.owners.get(chat) != user:
                return  # refused by the authentication before the endpoint
            async def delete():
                # as the endpoint does (cat/core_plugins/conversation_history/endpoints.py)
                await crud_conversations.delete_conversation(self.agent, user, chat)
                await crud_conversations.release_conversation(self.agent, chat)

            LOOP.run_until_complete(delete())
            del self.owners[chat]

        @invariant()
        def owners_are_the_model(self):
            async def read():
                return {chat: await crud_conversations.get_owner(self.agent, chat) for chat in CHATS}
            stored = LOOP.run_until_complete(read())
            assert stored == {chat: self.owners.get(chat) for chat in CHATS}

    OwnershipMachine.TestCase.settings = settings(
        max_examples=40, stateful_step_count=15, deadline=None, database=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    return OwnershipMachine


class FailingRedis:
    """The Redis client, whose ``set`` or ``get`` fail in the requests (asyncio tasks) that ask for it."""

    def __init__(self, db):
        self.db = db

    def __getattr__(self, name):
        attribute = getattr(self.db, name)
        if name not in ("set", "get"):
            return attribute

        async def command(*args, **kwargs):
            await asyncio.sleep(0)  # a network round trip: the requests interleave
            if FAIL.get() == name:
                raise ConnectionError("redis down")
            return await attribute(*args, **kwargs)
        return command


LOOP = None


def test_conversation_ownership_state_machine():
    import redis.asyncio as aioredis
    from cat.db.database import get_redis_kwargs

    global LOOP
    LOOP = asyncio.new_event_loop()
    try:
        # the test Redis database (see tests/conftest.py), with a client of this event loop
        db = FailingRedis(aioredis.Redis(**get_redis_kwargs()))
        machine = build_machine(f"ownership-{id(db)}")
        with mock.patch.object(crud_conversations, "get_async_db", lambda: db), \
                mock.patch.object(crud, "get_async_db", lambda: db):
            run_state_machine_as_test(machine, settings=machine.TestCase.settings)
        LOOP.run_until_complete(db.db.aclose())
    finally:
        LOOP.close()
        LOOP = None
