"""Dual-mode, end-to-end proof of the CATALOG catalogue cards on the real app.

The same real corpus (``tests/mocks/sample.txt``) is ingested through both
ingestion engines on the real app (``create_app`` + ``LifespanManager``):

* **classic mode** — the system ``ingestion`` setting is the core engine, so the
  cat_alog ``before_rabbithole_splits_documents`` / ``before_rabbithole_stores_documents``
  hooks own the card;
* **efficient mode** — the system ``ingestion`` setting is
  ``EfficientIngestionConfiguration``, so the ``catalog_cards`` phase (declared
  by cat_alog through the EffING phase contract) owns the card and the classic
  hooks are never fired.

Real dependencies
-----------------
* Redis: the dedicated host test db enforced by ``tests/conftest.py``
  (``CAT_REDIS_DB=1``), never production;
* Qdrant: the in-memory client installed by the host ``conftest`` — the default
  vector store of the scenarios that do not need graph edges;
* Neo4j: the live instance at ``bolt://127.0.0.1:7687``. Neo4j Community
  supports a single database, so isolation is a unique tenant id
  (``catalog_dual_<uuid>``) and every node carrying that tenant is deleted in
  ``finally`` (the DB has no throwaway-database equivalent on Community). If
  Neo4j is unreachable the tests FAIL explicitly (never skip silently).

Plugins
-------
The plugins are not installed in the host. ``tests/catalog/conftest.py``
symlinks ``cat_alog``, ``cat_efficient_ingestion`` and ``cat_multimodal_ingestion``
under ``cat/core_plugins/`` (created if missing, removed in ``finally``); this
module symlinks MyGRAPH under ``cat/plugins/cat_graphrag`` for the ``NEXT``
chunk edges (removed in ``finally``). No CAT core file and no plugin behaviour
is changed here.

Test-only instrumentation (never a production change): the default embedder's
``max_input_tokens`` is lowered for the oversized-card scenario, and the
cat_alog hook functions are wrapped to count their invocations.
"""

import asyncio
import importlib
import json
import os
import time
import uuid
from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from neo4j import AsyncGraphDatabase

from cat import utils
from cat.db.cruds import conversations as crud_conversations
from cat.services.factory.embedder import Embeddings
from cat.services.memory.models import PointStruct, VectorMemoryType
from cat.startup import create_app
from tests.catalog.conftest import created_links
from tests.utils import api_key, create_new_user, jwt_secret

PLUGIN_SRC = Path("/root/git/plugins/cat-graphrag.my")
PLUGIN_ID = "cat_graphrag"
PLUGINS_DIR = Path(utils.get_base_path()) / "plugins"
PLUGIN_LINK = PLUGINS_DIR / PLUGIN_ID

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_DATABASE = "neo4j"
NEO4J_AUTH_CONTAINERS = ("neo4j-db", "minimal-neo4j-1")

SAMPLE_FILE = Path("tests/mocks/sample.txt")
SAMPLE_NAME = SAMPLE_FILE.name
SAMPLE_CONTENT_TYPE = "text/plain"
LONG_NAME = "long_catalog.txt"

CORE_ENGINE = "CoreIngestionConfiguration"
EFFICIENT_ENGINE = "EfficientIngestionConfiguration"

DECLARATIVE = str(VectorMemoryType.DECLARATIVE)
EPISODIC = str(VectorMemoryType.EPISODIC)


# ---------------------------------------------------------------------------
# Real-app fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def agent_key() -> str:
    """Unique tenant for this test; also the Neo4j cleanup prefix."""
    return f"catalog_dual_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def graph_link(monkeypatch):
    """Expose the MyGRAPH worktree under ``cat/plugins/cat_graphrag``.

    The plugin loader derives the dotted module path from the folder name;
    ``cat-graphrag.my`` contains a dot and a hyphen and cannot be imported. The
    symlink lives for the test and is removed in ``finally``.
    """
    created = False
    if PLUGIN_LINK.exists() or PLUGIN_LINK.is_symlink():
        target = os.readlink(PLUGIN_LINK) if PLUGIN_LINK.is_symlink() else str(PLUGIN_LINK)
        if Path(target).resolve() != PLUGIN_SRC.resolve():
            pytest.fail(f"{PLUGIN_LINK} already exists and points to {target}")
    else:
        PLUGIN_LINK.symlink_to(PLUGIN_SRC, target_is_directory=True)
        created = True

    # the host conftest points the plugin folder at tests/mocks: restore the
    # real one so the plugin is discovered by the real app
    monkeypatch.setattr(utils, "get_plugins_path", lambda: str(PLUGINS_DIR))
    try:
        yield PLUGIN_LINK
    finally:
        if created and PLUGIN_LINK.is_symlink() and Path(os.readlink(PLUGIN_LINK)) == PLUGIN_SRC:
            PLUGIN_LINK.unlink()
        assert not (created and (PLUGIN_LINK.exists() or PLUGIN_LINK.is_symlink()))


@pytest.fixture
async def app_client(catalog_plugin_symlinks, graph_link, encapsulate_each_test):
    """The real FastAPI app, with the host Redis/Qdrant test doubles."""
    previous_api_key = os.environ.get("CAT_API_KEY")
    previous_jwt = os.environ.get("CAT_JWT_SECRET")
    os.environ["CAT_API_KEY"] = api_key
    os.environ["CAT_JWT_SECRET"] = jwt_secret
    app = None
    try:
        app = create_app()
        async with LifespanManager(app) as manager:
            async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as ac:
                ac._fastapi_test_app = app
                yield ac
    finally:
        # no leftover server: shutdown_app deletes the lizard from the app state
        if app is not None:
            assert not hasattr(app.state, "lizard")
        if previous_api_key is None:
            os.environ.pop("CAT_API_KEY", None)
        else:
            os.environ["CAT_API_KEY"] = previous_api_key
        if previous_jwt is None:
            os.environ.pop("CAT_JWT_SECRET", None)
        else:
            os.environ["CAT_JWT_SECRET"] = previous_jwt


# ---------------------------------------------------------------------------
# Neo4j helpers (real instance; tenant-scoped cleanup in finally)
# ---------------------------------------------------------------------------
def _neo4j_password() -> str:
    password = os.environ.get("CAT_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD")
    if password:
        return password

    import subprocess

    for container in NEO4J_AUTH_CONTAINERS:
        try:
            proc = subprocess.run(
                ["docker", "inspect", container, "--format", "{{json .Config.Env}}"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:  # noqa: BLE001 - try the next container
            continue
        if proc.returncode != 0:
            continue
        try:
            env = json.loads(proc.stdout)
        except json.JSONDecodeError:
            continue
        for entry in env:
            if entry.startswith("NEO4J_AUTH="):
                _, _, password = entry.partition("=")[2].partition("/")
                if password:
                    return password

    raise RuntimeError(
        "Neo4j password not found: set CAT_NEO4J_PASSWORD/NEO4J_PASSWORD or run "
        f"one of the local containers {NEO4J_AUTH_CONTAINERS} with NEO4J_AUTH."
    )


def _neo4j_driver():
    return AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, _neo4j_password()), connection_timeout=10,
    )


async def _neo4j_counts(tenant: str) -> dict:
    driver = _neo4j_driver()
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:

            async def one(query):
                result = await session.run(query, tenant=tenant)
                record = await result.single()
                await result.consume()
                return record["c"] if record else 0

            return {
                "nodes": await one("MATCH (n {tenant_id: $tenant}) RETURN count(n) AS c"),
                "relationships": await one(
                    "MATCH (a {tenant_id: $tenant})-[r]->(b {tenant_id: $tenant}) RETURN count(r) AS c"
                ),
                "documents": await one("MATCH (d:Document {tenant_id: $tenant}) RETURN count(d) AS c"),
                "cards": await one(
                    "MATCH (d:Document {tenant_id: $tenant}) "
                    "WHERE d.metadata CONTAINS '\"is_catalogue_card\": true' RETURN count(d) AS c"
                ),
            }
    finally:
        await driver.close()


async def _wipe_tenant(tenant: str) -> dict:
    driver = _neo4j_driver()
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("MATCH (n {tenant_id: $tenant}) DETACH DELETE n", tenant=tenant)
            await result.consume()
    finally:
        await driver.close()
    return await _neo4j_counts(tenant)


async def _assert_neo4j_reachable() -> None:
    driver = _neo4j_driver()
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 AS x")
            await result.single()
            await result.consume()
    except Exception as exc:  # noqa: BLE001 - surfaced explicitly, never silently skipped
        pytest.fail(f"Neo4j is unreachable at {NEO4J_URI}: {exc!r}")
    finally:
        await driver.close()


async def _next_pairs(tenant: str, collection: str = DECLARATIVE) -> list[tuple[str, str, dict, dict]]:
    """Every ``NEXT`` edge of the tenant in ``collection``, with endpoint metadata."""
    driver = _neo4j_driver()
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(
                "MATCH (a:Document {tenant_id: $tenant})-[:BELONGS_TO]->"
                "(c:Collection {tenant_id: $tenant, name: $collection}) "
                "MATCH (a)-[r:NEXT]->(b:Document {tenant_id: $tenant}) "
                "RETURN a.id AS aid, b.id AS bid, a.metadata AS am, b.metadata AS bm",
                tenant=tenant, collection=collection,
            )
            rows = [record async for record in result]
            await result.consume()
    finally:
        await driver.close()

    def _meta(raw):
        try:
            return json.loads(raw) if raw else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    return [(r["aid"], r["bid"], _meta(r["am"]), _meta(r["bm"])) for r in rows]


async def _count_tenant_prefix(prefix: str) -> int:
    driver = _neo4j_driver()
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(
                "MATCH (n) WHERE n.tenant_id STARTS WITH $prefix RETURN count(n) AS c",
                prefix=prefix,
            )
            record = await result.single()
            await result.consume()
            return record["c"] if record else 0
    finally:
        await driver.close()


# ---------------------------------------------------------------------------
# App-level configuration helpers
# ---------------------------------------------------------------------------
def _headers(agent_key: str) -> dict:
    return {"X-Agent-ID": agent_key, "Authorization": f"Bearer {api_key}"}


def _neo4j_payload() -> dict:
    # The derived graph (SourceFile/PART_OF/NEXT) is always built; the heavy
    # entity/concept extraction is disabled so the test isolates the chunk
    # structure and stays fast (and offline).
    return {
        "neo4j_uri": NEO4J_URI,
        "neo4j_user": NEO4J_USER,
        "neo4j_password": _neo4j_password(),
        "neo4j_database": NEO4J_DATABASE,
        "enable_derived_graph": True,
        "enable_entity_extraction": False,
        "spacy_models": {"en": "en_core_web_sm"},
        "enable_knowledge_graph": False,
        "enable_concept_relations": False,
    }


async def _activate_plugin(lizard, plugin_id: str) -> None:
    if plugin_id not in lizard.plugin_manager.active_plugins:
        await lizard.plugin_manager.activate_plugin(plugin_id)


async def _set_neo4j(client, agent_key: str) -> None:
    response = await client.put(
        "/vector_database/settings/Neo4jGraphRAGConfig",
        json=_neo4j_payload(),
        headers=_headers(agent_key),
    )
    assert response.status_code == 200, response.text
    assert response.json()["name"] == "Neo4jGraphRAGConfig"


async def _set_file_manager(client, agent_key: str) -> None:
    response = await client.put(
        "/file_manager/settings/LocalFileManagerConfig", json={}, headers=_headers(agent_key),
    )
    assert response.status_code == 200, response.text


async def _set_engine(client, agent_key: str, ingestion_name: str) -> None:
    response = await client.put(
        f"/ingestion/settings/{ingestion_name}", json={}, headers=_headers(agent_key),
    )
    assert response.status_code == 200, response.text


async def _configure_agent(client, agent_key: str, ingestion_name: str, graphrag: bool = False):
    lizard = client._fastapi_test_app.state.lizard
    if graphrag:
        await _activate_plugin(lizard, PLUGIN_ID)
    cat = await lizard.create_cheshire_cat(agent_key)
    if graphrag:
        if PLUGIN_ID not in cat.plugin_manager.active_plugins:
            await cat.plugin_manager.activate_plugin(PLUGIN_ID)
        assert PLUGIN_ID in cat.plugin_manager.active_plugins, cat.plugin_manager.active_plugins
        await _set_neo4j(client, agent_key)
    await _set_engine(client, agent_key, ingestion_name)
    # the upstream default file manager is a no-op; the prompt list and the
    # oversized-card abstract need the real local storage
    await _set_file_manager(client, agent_key)
    cat = await lizard.get_cheshire_cat(agent_key)
    if graphrag:
        assert type(cat.vector_memory_handler).__name__ == "GraphRAGHandler", type(cat.vector_memory_handler)
    return cat


async def _ingest(client, agent_key: str, name: str, data: bytes, content_type: str = "text/plain", chat_id=None):
    url = "/rabbithole/" if not chat_id else f"/rabbithole/{chat_id}"
    return await client.post(url, files={"file": (name, data, content_type)}, headers=_headers(agent_key))


async def _ingest_sample(client, agent_key: str, chat_id=None):
    return await _ingest(client, agent_key, SAMPLE_NAME, SAMPLE_FILE.read_bytes(), SAMPLE_CONTENT_TYPE, chat_id)


async def _status(agent_key: str, source: str = SAMPLE_NAME, scope: str = "agent") -> dict:
    registry = importlib.import_module("cat.core_plugins.cat_efficient_ingestion.registry")
    return await registry.get_status(agent_key, scope, source) or {}


# ---------------------------------------------------------------------------
# Point/assertion helpers
# ---------------------------------------------------------------------------
async def _points(cat, collection: str, metadata: dict) -> list:
    records, _ = await cat.vector_memory_handler.get_all_tenant_points(
        collection, metadata=metadata, with_vectors=False,
    )
    return list(records or [])


async def _card_points(cat, collection: str, source: str) -> list:
    return await _points(cat, collection, {"source": source, "is_catalogue_card": True})


async def _chunk_points(cat, collection: str, source: str) -> list:
    records = await _points(cat, collection, {"source": source})
    return [
        r for r in records
        if not (r.payload or {}).get("metadata", {}).get("is_catalogue_card")
        and (r.payload or {}).get("metadata", {}).get("image") is not True
    ]


def _meta(record) -> dict:
    return (record.payload or {}).get("metadata", {}) or {}


def _order_key(record) -> tuple:
    meta = _meta(record)
    return (meta.get("start_index", 1 << 60), meta.get("chunk_index", 1 << 60))


async def _assert_next_chain(agent_key: str, chunks: list) -> list:
    """Assert the ``NEXT`` edges are exactly the start_index-ordered chain."""
    ordered = sorted(chunks, key=_order_key)
    pairs = await _next_pairs(agent_key)
    actual = {(a, b) for a, b, _, _ in pairs}
    expected = {(ordered[i].id, ordered[i + 1].id) for i in range(len(ordered) - 1)}
    assert actual == expected, (actual, expected)
    for _, _, a_meta, b_meta in pairs:
        assert b_meta["start_index"] > a_meta["start_index"], (a_meta, b_meta)
    return ordered


async def _prompt_list(cat, base: str = "You are a helpful cat.") -> str:
    hook = next(
        h for h in cat.plugin_manager.plugins["cat_alog"].hooks if h.name == "agent_prompt_suffix"
    )
    return await hook.function(base, cat)


def _cat_alog_module(cat):
    hook = next(
        h for h in cat.plugin_manager.plugins["cat_alog"].hooks if h.name == "agent_prompt_suffix"
    )
    return importlib.import_module(hook.function.__module__)


def _install_card_hook_spy(plugin_manager, monkeypatch) -> dict:
    """Count the cat_alog classic card work (test-only instrumentation).

    ``stores`` counts the hook that APPENDS the classic card; ``cache_writes``
    counts the classic summary-cache writes (the only card work the splits hook
    does). EffING's shared chunking step still invokes the splits hook, but with
    the phase machine active it must be a no-op: no summary, no cache write.
    """
    counts = {"splits": 0, "stores": 0, "cache_writes": 0}
    wanted = {
        "before_rabbithole_splits_documents": "splits",
        "before_rabbithole_stores_documents": "stores",
    }

    def _wrap(key, original):
        async def wrapper(*args, **kwargs):
            counts[key] += 1
            return await original(*args, **kwargs)

        return wrapper

    # resolve the plugin module through an untouched hook (the wrapped ones
    # would report this test module as their __module__ after monkeypatching)
    cat_alog_module = None
    for hook in plugin_manager.plugins["cat_alog"].hooks:
        if hook.name == "agent_prompt_suffix":
            cat_alog_module = importlib.import_module(hook.function.__module__)
            break

    for name, key in wanted.items():
        for hook in plugin_manager.hooks.get(name, []):
            if hook.plugin_id == "cat_alog":
                monkeypatch.setattr(hook, "function", _wrap(key, hook.function))

    assert cat_alog_module is not None, "cat_alog hooks not registered"
    original_set = cat_alog_module.CATALOGUES.set

    def _counting_set(*args, **kwargs):
        counts["cache_writes"] += 1
        return original_set(*args, **kwargs)

    monkeypatch.setattr(cat_alog_module.CATALOGUES, "set", _counting_set)
    assert counts == {"splits": 0, "stores": 0, "cache_writes": 0}
    return counts


# ---------------------------------------------------------------------------
# Scenario 1 — classic mode
# ---------------------------------------------------------------------------
async def test_classic_mode_one_card_ordered_next_and_prompt_once(app_client, agent_key, monkeypatch):
    await _assert_neo4j_reachable()
    try:
        cat = await _configure_agent(app_client, agent_key, CORE_ENGINE, graphrag=True)
        spy = _install_card_hook_spy(cat.plugin_manager, monkeypatch)

        response = await _ingest_sample(app_client, agent_key)
        assert response.status_code == 200, response.text

        cards = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
        assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
        assert _meta(cards[0])["is_catalogue_card"] is True

        chunks = await _chunk_points(cat, DECLARATIVE, SAMPLE_NAME)
        assert len(chunks) >= 2, len(chunks)
        for chunk in chunks:
            meta = _meta(chunk)
            assert isinstance(meta.get("chunk_index"), int), meta
            assert isinstance(meta.get("start_index"), int), meta

        ordered = await _assert_next_chain(agent_key, chunks)
        assert len(await _next_pairs(agent_key)) == len(chunks) - 1

        prompt = await _prompt_list(cat)
        assert prompt.count(SAMPLE_NAME) == 1, prompt

        # classic mode owns the card through the two classic hooks, exactly once
        assert spy == {"splits": 1, "stores": 1, "cache_writes": 1}, spy

        print("CLASSIC_CATALOG", {
            "cards": len(cards), "chunks": len(chunks),
            "next": len(await _next_pairs(agent_key)),
            "starts": [_meta(c)["start_index"] for c in ordered], "spy": spy,
        })
    finally:
        leftover = await _wipe_tenant(agent_key)
        assert leftover == {"nodes": 0, "relationships": 0, "documents": 0, "cards": 0}, leftover


# ---------------------------------------------------------------------------
# Scenario 2 — efficient mode
# ---------------------------------------------------------------------------
async def test_efficient_mode_phase_builds_one_card_and_never_fires_classic_hooks(app_client, agent_key, monkeypatch):
    await _assert_neo4j_reachable()
    try:
        cat = await _configure_agent(app_client, agent_key, EFFICIENT_ENGINE, graphrag=True)
        spy = _install_card_hook_spy(cat.plugin_manager, monkeypatch)

        response = await _ingest_sample(app_client, agent_key)
        assert response.status_code == 200, response.text

        status = await _status(agent_key)
        assert status.get("status") == "completed", status
        recorded = set(status.get("completed_phases") or {})
        assert {"parsing_chunking", "embedding", "catalog_cards"} <= recorded, recorded

        cards = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
        assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
        assert _meta(cards[0])["is_catalogue_card"] is True

        # the classic card work is inert: the phase owns the card. EffING's
        # shared chunking step still invokes the splits hook, but it performs no
        # card work (no summary cache write) and the card-appending hook never runs.
        assert spy["stores"] == 0 and spy["cache_writes"] == 0, spy

        chunks = await _chunk_points(cat, DECLARATIVE, SAMPLE_NAME)
        assert len(chunks) >= 2, len(chunks)
        ordered = await _assert_next_chain(agent_key, chunks)

        prompt = await _prompt_list(cat)
        assert f"**{SAMPLE_NAME}**" in prompt, prompt
        assert prompt.count(SAMPLE_NAME) == 1, prompt

        print("EFFICIENT_CATALOG", {
            "cards": len(cards), "chunks": len(chunks),
            "phases": sorted(recorded),
            "starts": [_meta(c)["start_index"] for c in ordered], "spy": spy,
        })
    finally:
        leftover = await _wipe_tenant(agent_key)
        assert leftover == {"nodes": 0, "relationships": 0, "documents": 0, "cards": 0}, leftover


# ---------------------------------------------------------------------------
# Scenario 3 — EPISODIC chat-scoped card
# ---------------------------------------------------------------------------
async def test_episodic_card_is_chat_scoped(app_client, agent_key):
    chat = f"catalog_chat_{uuid.uuid4().hex[:8]}"
    cat = await _configure_agent(app_client, agent_key, CORE_ENGINE)

    response = await _ingest_sample(app_client, agent_key, chat_id=chat)
    assert response.status_code == 200, response.text

    cards = await _card_points(cat, EPISODIC, SAMPLE_NAME)
    assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
    for card in cards:
        assert _meta(card).get("chat_id") == chat, _meta(card)
    assert await _card_points(cat, DECLARATIVE, SAMPLE_NAME) == []

    print("EPISODIC_CATALOG", {"segments": len(cards), "chat_id": chat})


# ---------------------------------------------------------------------------
# Scenario 4 — oversized card is segmented, every segment carries full_card
# ---------------------------------------------------------------------------
async def test_oversized_card_is_segmented_with_full_card(app_client, agent_key, monkeypatch):
    # test-only instrumentation: lower the embedder input window so the card
    # must be split into several segments (DumbEmbedder normally has no limit)
    monkeypatch.setattr(Embeddings, "max_input_tokens", 100)

    cat = await _configure_agent(app_client, agent_key, EFFICIENT_ENGINE)
    long_text = ("Alice was beginning to get very tired of sitting by her sister. " * 220).encode()

    response = await _ingest(app_client, agent_key, LONG_NAME, long_text)
    assert response.status_code == 200, response.text

    cards = await _card_points(cat, DECLARATIVE, LONG_NAME)
    assert len(cards) > 1, len(cards)
    full_cards = {_meta(c).get("full_card") for c in cards}
    assert len(full_cards) == 1, len(full_cards)
    full_card = full_cards.pop()
    assert isinstance(full_card, str) and "## Summary" in full_card, full_card[:200]
    for card in cards:
        assert _meta(card).get("is_catalogue_card") is True

    print("OVERSIZED_CATALOG", {"segments": len(cards), "full_card_chars": len(full_card)})


# ---------------------------------------------------------------------------
# Scenario 5 — engine switch: the card is regenerated, never duplicated
# ---------------------------------------------------------------------------
async def test_engine_switch_regenerates_card_without_duplicating(app_client, agent_key, monkeypatch):
    cat = await _configure_agent(app_client, agent_key, EFFICIENT_ENGINE)
    response = await _ingest_sample(app_client, agent_key)
    assert response.status_code == 200, response.text

    first = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
    assert len(first) == 1, [(_meta(c), c.id) for c in first]
    first_ids = {c.id for c in first}

    # switch the SYSTEM ingestion engine to the core one and re-ingest
    await _set_engine(app_client, agent_key, CORE_ENGINE)
    spy = _install_card_hook_spy(cat.plugin_manager, monkeypatch)
    response = await _ingest_sample(app_client, agent_key)
    assert response.status_code == 200, response.text
    assert spy == {"splits": 1, "stores": 1, "cache_writes": 1}, spy

    second = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
    assert len(second) == 1, [(_meta(c), c.id) for c in second]

    print("SWITCH_CATALOG", {"first_ids": first_ids, "second_ids": {c.id for c in second}})


# ---------------------------------------------------------------------------
# Scenario 6 — images extracted from a file never leak into the prompt list
# ---------------------------------------------------------------------------
async def test_derived_image_file_is_excluded_from_prompt_list(app_client, agent_key):
    cat = await _configure_agent(app_client, agent_key, CORE_ENGINE)
    response = await _ingest_sample(app_client, agent_key)
    assert response.status_code == 200, response.text

    # reproduce the multimodal-ingestion artifact shape on the real stores: the
    # extracted image is SAVED as a file (so the FileManager lists it) and
    # stored as a point with image=True and image_file != source.
    derived = "sample_img_0_deadbeef.png"
    await cat.save_file(b"\x89PNG\r\n\x1a\n", "image/png", derived, None)
    vector = await asyncio.to_thread((await cat.embedder()).embed_query, "image")
    await cat.vector_memory_handler.add_points_to_tenant(
        collection_name=DECLARATIVE,
        points=[PointStruct(
            id=uuid.uuid4().hex,
            vector=vector,
            payload={
                "page_content": f"[Image] {SAMPLE_NAME}",
                "metadata": {
                    "source": SAMPLE_NAME, "when": time.time(),
                    "image": True, "image_file": derived,
                },
            },
        )],
    )
    assert {p.name for p in cat.file_manager.list_files(agent_key)} >= {SAMPLE_NAME, derived}

    prompt = await _prompt_list(cat)
    assert prompt.count(SAMPLE_NAME) == 1, prompt
    assert derived not in prompt, prompt

    print("IMAGE_EXCLUDED", {"derived_image": derived, "listed_source_once": True})


# ---------------------------------------------------------------------------
# Scenario 6b — card id is tenant/chat scoped (B1) and survives a re-embed (B2)
# ---------------------------------------------------------------------------
async def test_same_source_in_two_agents_gets_distinct_card_ids(app_client):
    """The same file name uploaded to two agents must not collide (B1, phase)."""
    agent_a = f"catalog_dual_agents_a_{uuid.uuid4().hex[:8]}"
    agent_b = f"catalog_dual_agents_b_{uuid.uuid4().hex[:8]}"

    cat_a = await _configure_agent(app_client, agent_a, EFFICIENT_ENGINE)
    cat_b = await _configure_agent(app_client, agent_b, EFFICIENT_ENGINE)

    assert (await _ingest_sample(app_client, agent_a)).status_code == 200
    assert (await _ingest_sample(app_client, agent_b)).status_code == 200

    cards_a = await _card_points(cat_a, DECLARATIVE, SAMPLE_NAME)
    cards_b = await _card_points(cat_b, DECLARATIVE, SAMPLE_NAME)
    assert len(cards_a) == 1, [(_meta(c), c.id) for c in cards_a]
    assert len(cards_b) == 1, [(_meta(c), c.id) for c in cards_b]

    ids_a = {c.id for c in cards_a}
    ids_b = {c.id for c in cards_b}
    assert ids_a.isdisjoint(ids_b), (ids_a, ids_b)

    print("TENANT_SCOPED_CATALOG", {"agent_a": sorted(ids_a), "agent_b": sorted(ids_b)})


async def test_same_source_in_two_chats_gets_distinct_card_ids(app_client, agent_key):
    """Two chats of ONE agent get two distinct cards (B1, classic hooks)."""
    cat = await _configure_agent(app_client, agent_key, CORE_ENGINE)
    chat_1 = f"catalog_chat_{uuid.uuid4().hex[:8]}"
    chat_2 = f"catalog_chat_{uuid.uuid4().hex[:8]}"

    assert (await _ingest_sample(app_client, agent_key, chat_id=chat_1)).status_code == 200
    assert (await _ingest_sample(app_client, agent_key, chat_id=chat_2)).status_code == 200

    cards = await _card_points(cat, EPISODIC, SAMPLE_NAME)
    assert len(cards) == 2, [(_meta(c).get("chat_id"), c.id) for c in cards]
    by_chat = {_meta(c).get("chat_id"): c.id for c in cards}
    assert set(by_chat) == {chat_1, chat_2}, by_chat
    assert len(set(by_chat.values())) == 2, by_chat

    print("CHAT_SCOPED_CATALOG", by_chat)


async def test_reembed_keeps_card_id_and_does_not_duplicate(app_client, agent_key):
    """A re-embed rebuild keeps the card id, so it never becomes a duplicate (B2)."""
    from cat.core_plugins.cat_efficient_ingestion import reembed

    cat = await _configure_agent(app_client, agent_key, EFFICIENT_ENGINE)
    assert (await _ingest_sample(app_client, agent_key)).status_code == 200

    cards = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
    assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
    original_ids = {c.id for c in cards}

    # Emulate the re-embed rebuild the embedding phase performs: it reconstructs
    # every stored point of the source from its payload and re-stores it.
    points = await _points(cat, DECLARATIVE, {"source": SAMPLE_NAME})
    embedder = await cat.embedder()
    rebuilt = await reembed._embed_phase(
        cat, DECLARATIVE, SAMPLE_NAME, points, embedder, None, cat, rabbit_hole=None,
    )
    rebuilt_cards = [
        p for p in rebuilt
        if (p.payload or {}).get("metadata", {}).get("is_catalogue_card") is True
    ]
    assert len(rebuilt_cards) == 1, [(_meta(p), p.id) for p in rebuilt_cards]
    assert {p.id for p in rebuilt_cards} == original_ids, (
        "the re-embed rebuild lost the card id",
        original_ids,
        {p.id for p in rebuilt_cards},
    )

    # After an embedding change the catalog_cards phase is stale and re-runs: it
    # must UPSERT the same id, never add a second card.
    run_hook = next(
        h for h in cat.plugin_manager.plugins["cat_alog"].hooks
        if h.name == "ingestion_phase_run"
    )
    result = await run_hook.function("catalog_cards", SAMPLE_NAME, {}, cat)
    assert result == {"status": "done"}, result

    cards = await _card_points(cat, DECLARATIVE, SAMPLE_NAME)
    assert len(cards) == 1, [(_meta(c), c.id) for c in cards]
    assert {c.id for c in cards} == original_ids, (
        "re-embed + phase re-run changed the card id",
        original_ids,
        {c.id for c in cards},
    )

    print("REEMBED_CATALOG", {"cards": len(cards), "ids": sorted(original_ids)})


async def test_same_source_in_two_chats_phase_mode_is_isolated(app_client, agent_key):
    """B1: the phase keeps two chats of one filename isolated (two cards)."""
    cat = await _configure_agent(app_client, agent_key, EFFICIENT_ENGINE)
    chat_1 = f"catalog_chat_{uuid.uuid4().hex[:8]}"
    chat_2 = f"catalog_chat_{uuid.uuid4().hex[:8]}"
    text_1 = ("ALPHA CHAT ONLY " * 60).encode()
    text_2 = ("BETA CHAT ONLY " * 60).encode()

    # The phase worker resolves each chat's StrayCat by its conversation owner,
    # so the conversations must exist (the admin API key bypasses the ownership
    # claim the UI user path performs). Attach both chats to a real user.
    user = await create_new_user(
        app_client,
        username=f"catalog_user_{uuid.uuid4().hex[:8]}",
        headers=_headers(agent_key),
    )
    for chat in (chat_1, chat_2):
        assert await crud_conversations.claim_conversation(agent_key, chat, user["id"]) == user["id"]

    assert (
        await _ingest(app_client, agent_key, SAMPLE_NAME, text_1, chat_id=chat_1)
    ).status_code == 200
    assert (
        await _ingest(app_client, agent_key, SAMPLE_NAME, text_2, chat_id=chat_2)
    ).status_code == 200

    cards = await _card_points(cat, EPISODIC, SAMPLE_NAME)
    assert len(cards) == 2, [(_meta(c).get("chat_id"), c.id) for c in cards]
    by_chat = {_meta(c).get("chat_id"): c for c in cards}
    assert set(by_chat) == {chat_1, chat_2}, by_chat.keys()
    assert by_chat[chat_1].id != by_chat[chat_2].id, by_chat

    full_1 = _meta(by_chat[chat_1]).get("full_card") or ""
    full_2 = _meta(by_chat[chat_2]).get("full_card") or ""
    assert "ALPHA CHAT ONLY" in full_1 and "BETA CHAT ONLY" not in full_1, full_1[:200]
    assert "BETA CHAT ONLY" in full_2 and "ALPHA CHAT ONLY" not in full_2, full_2[:200]
    assert await _card_points(cat, DECLARATIVE, SAMPLE_NAME) == []

    print("CHAT_SCOPED_PHASE_CATALOG", {
        "segments": len(cards),
        "ids": {chat: c.id for chat, c in by_chat.items()},
    })


async def test_reembed_keeps_oversized_card_ids(app_client, agent_key, monkeypatch):
    """B2: a re-embed must not re-split the card, so count and ids stay stable."""
    from cat.core_plugins.cat_efficient_ingestion import reembed

    # test-only instrumentation: force the phase to segment the oversized card.
    monkeypatch.setattr(Embeddings, "max_input_tokens", 100)
    cat = await _configure_agent(app_client, agent_key, EFFICIENT_ENGINE)
    long_text = ("Alice was beginning to get very tired of sitting by her sister. " * 220).encode()

    assert (await _ingest(app_client, agent_key, LONG_NAME, long_text)).status_code == 200

    cards = await _card_points(cat, DECLARATIVE, LONG_NAME)
    assert len(cards) > 1, [(_meta(c), c.id) for c in cards]
    original_ids = sorted(c.id for c in cards)
    cat_alog = _cat_alog_module(cat)
    for card in cards:
        meta = _meta(card)
        assert meta.get("is_catalogue_card_segment") is True, meta
        assert isinstance(meta.get("start_index"), int), meta
        assert card.id == cat_alog._catalogue_card_point_id(
            agent_key, None, LONG_NAME, start_index=meta["start_index"]
        )

    points = await _points(cat, DECLARATIVE, {"source": LONG_NAME})
    embedder = await cat.embedder()
    # A LOWER limit after the fact: without the card-segment exemption this
    # re-split would mint a different segment count and new ids.
    monkeypatch.setattr(embedder, "max_input_tokens", 20, raising=False)
    rebuilt = await reembed._embed_phase(
        cat, DECLARATIVE, LONG_NAME, points, embedder, None, cat, rabbit_hole=None,
    )
    rebuilt_cards = [
        p for p in rebuilt
        if (p.payload or {}).get("metadata", {}).get("is_catalogue_card") is True
    ]
    assert len(rebuilt_cards) == len(cards), [p.id for p in rebuilt_cards]
    assert sorted(p.id for p in rebuilt_cards) == original_ids, (
        "the re-embed re-split the card and lost its ids",
        original_ids,
        sorted(p.id for p in rebuilt_cards),
    )

    print("REEMBED_OVERSIZED_CATALOG", {
        "segments": len(cards), "ids": original_ids,
    })


# ---------------------------------------------------------------------------
# Scenario 7 — cleanup receipt (runs last)
# ---------------------------------------------------------------------------
async def test_zz_cleanup_receipt():
    """No leftover symlinks, server, seeded keys/files/dirs or Neo4j tenant."""
    for link in created_links():
        assert not os.path.lexists(link), link
    assert not PLUGIN_LINK.exists() and not PLUGIN_LINK.is_symlink()

    # server: the app fixtures assert app.state.lizard is gone; no process is
    # left behind because every request ran in-process (ASGITransport).
    if Path("tests/data").exists():
        assert not list(Path("tests/data/storage").glob("catalog_dual_*"))
    assert not list(Path("/tmp").glob("catalog_dual_*"))

    from cat.db.database import get_sync_db

    db = get_sync_db()
    assert str(db.connection_pool.connection_kwargs.get("db")) == "1"
    assert db.dbsize() == 0, db.dbsize()

    await _assert_neo4j_reachable()
    assert await _count_tenant_prefix("catalog_dual_") == 0

    print("CLEANUP_RECEIPT", {"redis_dbsize": 0, "neo4j_catalog_tenants": 0, "seeded_links": 0})
