"""Tests for the efficient_ingestion plugin (replaceable ingestion engine).

The engine is exposed through the ServiceFactory (``ingestion`` category):
configs are stored in the global ``system:agent`` settings list, the plugin
registers ``EfficientIngestionConfiguration`` via ``factory_allowed_ingestions``
and, when selected, it is the active engine.

The selection follows the embedder pattern: the category ``ingestion`` holds a
SINGLE setting whose ``name`` is the active configuration class — there is no
separate selection entry (that would break the uniqueness invariant and cause
the "new dict appended instead of replaced" bug).

Runs against UPSTREAM CAT (``cat/services/factory/ingestion.py``): the config
inherits the upstream ``BaseIngestionConfiguration`` and the engine is resolved
through the upstream ``ServiceFactory`` API (``get_schemas`` /
``get_from_config_name`` / ``upsert_service``).

The tests that need the ``factory_allowed_ingestions`` hook to fire install the
plugin (and its MmodING dependency) into the mock plugin folder through the
standard MadHatter activation path, exactly like the ``plugin_manager`` fixture
does for the mock plugin.

Uses ``tests/conftest.py`` fixtures: Redis db=1 (isolated).
"""

import os

import pytest

from cat.core_plugins.cat_efficient_ingestion.configs import EfficientIngestionConfiguration
from cat.core_plugins.cat_efficient_ingestion.reembed import EfficientIngestionEngine
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY, get_sync_db
from cat.db.models import Setting
from cat.services.factory.ingestion import (
    BaseIngestionConfiguration,
    BaseIngestionEngine,
)
from cat.services.service_factory import ServiceFactory

EFFING_SRC = "/root/git/plugins/cat-efficient-ingestion"
MMODING_SRC = "/root/git/plugins/cat-multimodal-ingestion"
MOCK_PLUGIN_FOLDER = "tests/mocks/mock_plugin_folder"


def _cleanup():
    db = get_sync_db()
    db.json().delete("system:agent", '$[?(@.category == "ingestion")]')


def _build_factory(lizard) -> ServiceFactory:
    return ServiceFactory(
        agent_key=DEFAULT_SYSTEM_KEY,
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_ingestions",
        setting_category="ingestion",
        schema_name="ingestionName",
    )


@pytest.fixture
async def effing_plugin_installed(lizard):
    """Make sure the EffING plugin is loaded so the real
    ``factory_allowed_ingestions`` hook fires.

    On this CAT checkout the plugin may already be a core plugin (symlinked in
    ``cat/core_plugins/``, loaded at bootstrap) — then nothing to do. Otherwise
    install it (and its MmodING dependency) into the mock plugin folder through
    the standard MadHatter activation path, like the ``plugin_manager`` fixture
    does for the mock plugin.
    """
    pm = lizard.plugin_manager
    if "cat_efficient_ingestion" in pm.get_core_plugins_ids:
        yield
        return
    for plugin_id, src in (
        ("cat_multimodal_ingestion", MMODING_SRC),
        ("cat_efficient_ingestion", EFFING_SRC),
    ):
        link = os.path.join(MOCK_PLUGIN_FOLDER, plugin_id)
        if not os.path.exists(link):
            os.symlink(src, link)
        await pm.install_extracted_plugin(plugin_id)
    yield
    # deactivate EffING first: MmodING cannot be deactivated while EffING
    # (which depends on it) is still active
    for plugin_id in ("cat_efficient_ingestion", "cat_multimodal_ingestion"):
        if plugin_id in pm.active_plugins:
            await pm.deactivate_plugin(plugin_id)
        link = os.path.join(MOCK_PLUGIN_FOLDER, plugin_id)
        if os.path.islink(link):
            os.remove(link)


async def test_configuration_defaults():
    cfg = EfficientIngestionConfiguration()
    assert cfg.model_dump() == {"ingestion_max_concurrency": 5}
    assert cfg.pyclass() is EfficientIngestionEngine


async def test_configuration_inherits_upstream_base():
    """[upstream-compat] The config must inherit the upstream
    ``BaseIngestionConfiguration`` (not ``BaseFactoryConfigModel`` directly), so
    the factory resolves it as a replaceable class on upstream CAT. Instantiating
    the engine directly also proves every abstract method of
    ``BaseIngestionEngine`` (``run``, ``ingest_file``) is implemented — a missing
    one would raise TypeError here."""
    assert issubclass(EfficientIngestionConfiguration, BaseIngestionConfiguration)
    assert EfficientIngestionConfiguration.base_class() is BaseIngestionEngine
    assert EfficientIngestionConfiguration.pyclass() is EfficientIngestionEngine

    # instantiation would raise TypeError if any abstract method were missing
    cfg = EfficientIngestionConfiguration(ingestion_max_concurrency=3)
    engine = cfg.pyclass()(ingestion_max_concurrency=cfg.ingestion_max_concurrency)
    assert isinstance(engine, EfficientIngestionEngine)
    assert engine.ingestion_max_concurrency == 3


async def test_factory_allows_plugin_config_and_instantiates_engine(lizard, effing_plugin_installed):
    """The ``factory_allowed_ingestions`` hook exposes the plugin config class on
    upstream CAT, and the factory instantiates the engine from it."""
    _cleanup()
    sf = _build_factory(lizard)

    schemas = await sf.get_schemas()
    # the core default config (whatever its current name) plus the plugin's
    assert sf.default_config_class.__name__ in schemas
    assert "EfficientIngestionConfiguration" in schemas

    await sf.upsert_service("EfficientIngestionConfiguration", {"ingestion_max_concurrency": 5})
    engine = await sf.get_from_config_name("EfficientIngestionConfiguration")
    # the factory may instantiate the plugin copy loaded by the MadHatter
    # (cat.core_plugins.* when symlinked as a core plugin, or the installed
    # user-plugin copy): same code, so assert on name + behavior, not identity
    assert type(engine).__name__ == "EfficientIngestionEngine"
    assert engine.ingestion_max_concurrency == 5
    _cleanup()


async def test_upsert_stores_category_ingestion_and_value(lizard, effing_plugin_installed):
    _cleanup()
    sf = _build_factory(lizard)
    await sf.upsert_service("EfficientIngestionConfiguration", {"ingestion_max_concurrency": 3})

    entry = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, "EfficientIngestionConfiguration")
    assert entry is not None
    assert entry["category"] == "ingestion"
    assert entry["value"] == {"ingestion_max_concurrency": 3}

    engine = await sf.get_from_config_name("EfficientIngestionConfiguration")
    assert type(engine).__name__ == "EfficientIngestionEngine"
    assert engine.ingestion_max_concurrency == 3
    _cleanup()


async def test_upsert_does_not_duplicate_category_entry(lizard, effing_plugin_installed):
    """[regression] Editing the ingestion settings must replace the single
    category entry, NOT append a new dict to the ``system:agent`` list — same
    invariant as the embedder category."""
    _cleanup()
    sf = _build_factory(lizard)
    await sf.upsert_service("EfficientIngestionConfiguration", {"ingestion_max_concurrency": 3})
    await sf.upsert_service("EfficientIngestionConfiguration", {"ingestion_max_concurrency": 7})

    # exactly ONE entry with category "ingestion"
    db = get_sync_db()
    matches = db.json().get("system:agent", '$[?(@.category == "ingestion")]') or []
    assert len(matches) == 1
    assert matches[0]["value"] == {"ingestion_max_concurrency": 7}

    engine = await sf.get_from_config_name("EfficientIngestionConfiguration")
    assert engine.ingestion_max_concurrency == 7
    _cleanup()


async def test_endpoints_list_and_select(secure_client, secure_client_headers, cheshire_cat, lizard, effing_plugin_installed):
    # select the plugin engine explicitly (creates the single category entry)
    res = await secure_client.put(
        "/ingestion/settings/EfficientIngestionConfiguration",
        headers=secure_client_headers,
        json={},
    )
    assert res.status_code == 200

    listing = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    assert listing.status_code == 200
    body = listing.json()
    names = [s["name"] for s in body["settings"]]
    assert "EfficientIngestionConfiguration" in names
    assert body["selected_configuration"] == "EfficientIngestionConfiguration"

    # select the core default engine explicitly (name-agnostic: the parallel
    # upstream port may rename the core config class)
    core_name = _build_factory(lizard).default_config_class.__name__
    assert core_name in names
    res = await secure_client.put(
        f"/ingestion/settings/{core_name}",
        headers=secure_client_headers,
        json={},
    )
    assert res.status_code == 200

    # still a single category entry after the PUT (regression guard)
    db = get_sync_db()
    matches = db.json().get("system:agent", '$[?(@.category == "ingestion")]') or []
    assert len(matches) == 1

    listing2 = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    assert listing2.json()["selected_configuration"] == core_name
    _cleanup()


async def test_legacy_reembed_max_concurrency_is_migrated(lizard, effing_plugin_installed):
    """[migration] A config previously saved with ``reembed_max_concurrency``
    (pre-rename, when the engine only handled the re-embed) keeps working: the
    value is mapped onto ``ingestion_max_concurrency``."""
    _cleanup()

    # a PREVIOUSLY saved entry still in the DB uses the old field name
    await crud_settings.upsert_setting_by_category(
        DEFAULT_SYSTEM_KEY,
        Setting(
            name="EfficientIngestionConfiguration",
            value={"reembed_max_concurrency": 8},
            category="ingestion",
        ),
    )

    sf = _build_factory(lizard)
    engine = await sf.get_from_config_name("EfficientIngestionConfiguration")
    assert type(engine).__name__ == "EfficientIngestionEngine"
    assert engine.ingestion_max_concurrency == 8
    _cleanup()