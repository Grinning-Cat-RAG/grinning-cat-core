import os
import shutil
import time
import warnings
from inspect import iscoroutinefunction
from uuid import uuid4

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from pydantic import PydanticDeprecatedSince20
from qdrant_client import AsyncQdrantClient

from cat import utils
from cat.auth import auth_utils
from cat.auth.permissions import AuthUserInfo, get_base_permissions
from cat.env import get_env, get_env_int
from cat.looking_glass import StrayCat
from cat.looking_glass.mad_hatter.plugin import Plugin
from cat.services.factory.vector_db import QdrantHandler
from cat.services.memory.messages import UserMessage
from cat.startup import create_app
from tests.utils import (
    agent_id,
    api_key,
    create_mock_plugin_zip,
    fake_timestamp,
    jwt_secret,
    mock_plugin_path,
)

pytest_plugins = ["pytest"]

# ---------------------------------------------------------------------------
# TEST-DATA SAFETY (MANDATORY — see AGENTS.md "Safety rules")
# Force the whole test suite onto a dedicated, isolated Redis DB **before** the
# Database singleton can ever be created. This must be an env variable, not
# only a mock: a `Database` already bound to the real env (default
# CAT_REDIS_DB=0 == production) would make `flushdb()` wipe real agent data.
# Incident: get_sync_db().flushdb() hit db0 in CAT. [scope:DEH-RAG-wide]
# ---------------------------------------------------------------------------
TEST_REDIS_DB = "1"
os.environ["CAT_REDIS_DB"] = TEST_REDIS_DB
# substitute classes' methods where necessary for testing purposes
def mock_classes(monkeypatch, memory_client):
    # Mock the entire __init__ method to set _client to memory client, and the close method to do nothing
    def mock_init_vector_database(self, *args, **kwargs):
        # Call the parent __init__
        super(QdrantHandler, self).__init__()
        # Set the _client to memory client
        self._client = memory_client
        self.save_memory_snapshots = kwargs.get("save_memory_snapshots", False)
    monkeypatch.setattr(QdrantHandler, "__init__", mock_init_vector_database)
    monkeypatch.setattr(QdrantHandler, "close", lambda self: None)

    def mock_get_redis_kwargs():
        return {
            "host": get_env("CAT_REDIS_HOST"),
            "port": get_env_int("CAT_REDIS_PORT"),
            "db": TEST_REDIS_DB,
            "encoding": "utf-8",
            "decode_responses": True,
        }
    monkeypatch.setattr("cat.db.database.get_redis_kwargs", mock_get_redis_kwargs)

    utils.get_plugins_path = lambda: "tests/mocks/mock_plugin_folder/"
    utils.get_file_manager_root_storage_path = lambda: "tests/data/storage"

    # do not check plugin dependencies at every restart
    async def mock_install_requirements(self):
        pass
    monkeypatch.setattr(Plugin, "_install_requirements", mock_install_requirements)

    # mock the agent_id in the request
    auth_utils.extract_agent_id_from_request = lambda: agent_id

    async def immediate_task(background_tasks, func, *args, **kwargs):
        await func(*args, **kwargs) if iscoroutinefunction(func) else func(*args, **kwargs)

    monkeypatch.setattr("cat.routes.routes_utils.run_background_task", immediate_task)


async def clean_up():
    from cat.db.database import get_sync_db

    # clean up service files and mocks
    to_be_removed = [
        "tests/mocks/mock_plugin.zip",
        "tests/mocks/mock_plugin_fast_reply.zip",
        "tests/mocks/mock_plugin_multimodal_embedder.zip",
        "tests/mocks/mock_plugin_with_dependencies.zip",
        "tests/mocks/mock_plugin/settings.json",
        "tests/mocks/mock_plugin_folder/mock_plugin",
        "tests/mocks/mock_plugin_folder/mock_plugin_fast_reply",
        "tests/mocks/mock_plugin_folder/mock_plugin_multimodal_embedder",
        "tests/mocks/mock_plugin_folder/mock_plugin_with_dependencies",
        "tests/mocks/empty_folder",
        "tests/data",
    ]
    for tbr in to_be_removed:
        if os.path.exists(tbr):
            if os.path.isdir(tbr):
                shutil.rmtree(tbr)
            else:
                os.remove(tbr)

    # flush only the dedicated test database — NEVER the production one (db=0).
    # A hard guard: if the client is bound to any non-test DB, refuse to flush
    # rather than risk destroying real data (see AGENTS.md "Safety rules").
    db = get_sync_db()

    # connection_kwargs['db'] reflects the actual db the client was initialized
    # with; falling back to env for safety. If it is not the dedicated test db,
    # abort instead of flushing whatever it points at.
    bound_db = db.connection_pool.connection_kwargs.get("db", os.environ.get("CAT_REDIS_DB"))
    if str(bound_db) != TEST_REDIS_DB:
        raise RuntimeError(
            f"Refusing to flushdb(): bound to Redis db={bound_db!r}, expected test db={TEST_REDIS_DB!r}. "
            "The Database singleton is probably pointing at production. Aborting to prevent data loss."
        )

    db.flushdb()


@pytest.fixture(scope="function")
async def memory_client():
    client = AsyncQdrantClient(":memory:")
    yield client
    collections = await client.get_collections()
    for collection in collections.collections:
        await client.delete_collection(collection.name)
    await client.close()


@pytest.fixture(autouse=True, scope="function")
async def encapsulate_each_test(request, monkeypatch, memory_client):
    utils.singleton.instances.clear()

    mock_classes(monkeypatch, memory_client)

    await clean_up()
    yield
    await clean_up()

    utils.singleton.instances.clear()


@pytest.fixture(scope="function")
async def client(encapsulate_each_test):
    app = create_app()
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac:
            ac._fastapi_test_app = app
            yield ac


# This fixture sets the CAT_API_KEY environment variable,
# making mandatory for clients to possess api key or JWT
@pytest.fixture(scope="function")
async def secure_client(client):
    current_api_key = os.getenv("CAT_API_KEY")
    current_jwt_secret = os.getenv("CAT_JWT_SECRET")

    # set ENV variables
    os.environ["CAT_API_KEY"] = api_key
    os.environ["CAT_JWT_SECRET"] = jwt_secret

    yield client

    # clean up
    if current_api_key:
        os.environ["CAT_API_KEY"] = current_api_key
    else:
        del os.environ["CAT_API_KEY"]
    if current_jwt_secret:
        os.environ["CAT_JWT_SECRET"] = current_jwt_secret
    else:
        del os.environ["CAT_JWT_SECRET"]


@pytest.fixture(scope="function")
def secure_client_headers():
    yield {"X-Agent-ID": agent_id, "Authorization": f"Bearer {api_key}"}


@pytest.fixture(scope="function")
async def lizard(client):
    yield client._fastapi_test_app.state.lizard


@pytest.fixture(scope="function")
async def cheshire_cat(lizard):
    cheshire_cat = await lizard.create_cheshire_cat(agent_id)
    yield cheshire_cat


@pytest.fixture(scope="function")
async def plugin_manager(lizard):
    plugin_manager = lizard.plugin_manager

    # install plugin
    new_plugin_zip_path = create_mock_plugin_zip(flat=True)
    await lizard.install_plugin(new_plugin_zip_path)

    yield plugin_manager

    await plugin_manager.uninstall_plugin("mock_plugin")


@pytest.fixture(scope="function")
async def agent_plugin_manager(cheshire_cat):
    plugin_manager = cheshire_cat.plugin_manager

    # install plugin
    new_plugin_zip_path = create_mock_plugin_zip(flat=True)
    plugin_id = await cheshire_cat.lizard.install_plugin(new_plugin_zip_path)

    # activate the plugin within the Cheshire Cat whose plugin manager is being used
    await cheshire_cat.toggle_plugin(plugin_id)

    yield plugin_manager


@pytest.fixture(scope="function")
async def stray_no_memory(cheshire_cat, agent_plugin_manager):
    stray_cat = await StrayCat.from_cat(
        user_data=AuthUserInfo(id=str(uuid4()), name="Alice", permissions=get_base_permissions()),
        cat=cheshire_cat,
    )
    yield stray_cat


# fixture to have available an instance of StrayCat
@pytest.fixture(scope="function")
async def stray(stray_no_memory):
    stray_no_memory.working_memory.user_message = UserMessage(text="meow")
    yield stray_no_memory


# auto-use fixture will be applied to *all* the tests
@pytest.fixture(autouse=True, scope="function")
def apply_warning_filters():
    # ignore deprecation warnings due to langchain not updating to pydantic v2
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)


#fixture for mock time.time function
@pytest.fixture(scope="function")
def patch_time_now(monkeypatch):
    def my_time():
        return fake_timestamp

    monkeypatch.setattr(time, "time", my_time)


# this fixture will give test functions a ready instantiated plugin
# (and having the `client` fixture, a clean setup every unit)
@pytest.fixture(scope="function")
async def plugin(lizard):
    p = Plugin(mock_plugin_path)
    yield p


@pytest.fixture(scope="function")
def mocked_default_llm_answer_prompt():
    def mock_default_llm_answer_prompt() -> str:
        return "Ops AI: You did not configure a Language Model. Do it in the settings!"

    fnc = utils.default_llm_answer_prompt
    utils.default_llm_answer_prompt = mock_default_llm_answer_prompt

    yield

    utils.default_llm_answer_prompt = fnc


# Define the custom marker
pytest.mark.skip_encapsulation = pytest.mark.skip_encapsulation
