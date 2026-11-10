from typing import Dict
import redis
import redis.asyncio as aioredis

from cat.env import get_env, get_env_int, get_env_bool
from cat.utils import singleton

DEFAULT_AGENTS_KEY = "agents"
DEFAULT_AGENT_KEY = "agent"
DEFAULT_CONVERSATIONS_KEY = "conversations"
DEFAULT_INGESTION_KEY = "ingestion"
DEFAULT_PLUGINS_KEY = "plugins"
DEFAULT_USERS_KEY = "users"
DEFAULT_SYSTEM_KEY = "system"


@singleton
class Database:
    def __init__(self):
        self._async_db = None
        self._sync_db = None

    @property
    def async_db(self) -> aioredis.Redis:
        if self._async_db is None:
            self._async_db = aioredis.Redis(**get_redis_kwargs())
        return self._async_db

    @property
    def sync_db(self) -> redis.Redis:
        if self._sync_db is None:
            self._sync_db = redis.Redis(**get_redis_kwargs())
        return self._sync_db

    def reset_async(self):
        self._async_db = None

    def reset_sync(self):
        self._sync_db = None


def get_async_db() -> aioredis.Redis:
    return Database().async_db


def get_sync_db() -> aioredis.Redis:
    return Database().sync_db


def get_db_connection_string() -> str:
    secure = "s" if get_env_bool("CAT_REDIS_TLS") else ""

    host = get_env("CAT_REDIS_HOST")
    port = get_env("CAT_REDIS_PORT")
    db = get_env_int("CAT_REDIS_DB")

    password = get_env("CAT_REDIS_PASSWORD")

    return (
        f"redis{secure}://{host}:{port}/{db}"
        if not password else f"redis{secure}://:{password}@{host}:{port}/{db}"
    )

def get_redis_kwargs() -> Dict:
    host = get_env("CAT_REDIS_HOST")
    if host is None:
        raise ValueError("CAT_REDIS_HOST environment variable is not set.")

    password = get_env("CAT_REDIS_PASSWORD")
    tls = get_env_bool("CAT_REDIS_TLS")

    kwargs = dict(
        host=host,
        port=get_env_int("CAT_REDIS_PORT"),
        db=get_env_int("CAT_REDIS_DB"),
        encoding="utf-8",
        decode_responses=True,
        ssl=tls,
    )
    if password:
        kwargs["password"] = password

    return kwargs