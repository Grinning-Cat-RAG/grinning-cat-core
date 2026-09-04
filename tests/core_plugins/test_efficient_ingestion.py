"""Tests for the efficient_ingestion plugin (replaceable ingestion engine).

The engine is exposed through the ServiceFactory (``ingestion`` category):
configs are stored in the global ``system:agent`` settings list, the plugin
registers ``EfficientIngestionConfiguration`` via ``factory_allowed_ingestions``
and, when present, it is the preferred engine.

The selection follows the embedder pattern: the category ``ingestion`` holds a
SINGLE setting whose ``name`` is the active configuration class — there is no
separate selection entry (that would break the uniqueness invariant and cause
the "new dict appended instead of replaced" bug).

Uses ``tests/conftest.py`` fixtures: Redis db=1 (isolated).
"""

from cat.core_plugins.efficient_ingestion.configs import EfficientIngestionConfiguration
from cat.core_plugins.efficient_ingestion.reembed import EfficientIngestionEngine
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY, get_sync_db
from cat.db.models import Setting
from cat.services.factory.ingestion import (
    resolve_ingestion_engine,
    resolved_config_name,
)
from tests.utils import get_client_admin_headers


def _cleanup():
    db = get_sync_db()
    db.json().delete("system:agent", '$[?(@.category == "ingestion")]')


async def test_configuration_defaults():
    cfg = EfficientIngestionConfiguration()
    assert cfg.model_dump() == {"ingestion_max_concurrency": 5}
    assert cfg.pyclass() is EfficientIngestionEngine


async def test_resolved_default_prefers_plugin_engine(lizard):
    _cleanup()
    # efficient_ingestion is a core plugin: its config class is always allowed
    assert await resolved_config_name(lizard) == "EfficientIngestionConfiguration"

    engine = await resolve_ingestion_engine(lizard)
    assert engine.__class__.__name__ == "EfficientIngestionEngine"
    assert engine.ingestion_max_concurrency == 5


async def test_saved_config_is_the_single_selection(lizard):
    _cleanup()
    # saving the base engine config (single entry in the ingestion category)
    await crud_settings.upsert_setting_by_category(
        DEFAULT_SYSTEM_KEY,
        Setting(name="BaseIngestionConfiguration", value={}, category="ingestion"),
    )
    assert await resolved_config_name(lizard) == "BaseIngestionConfiguration"

    engine = await resolve_ingestion_engine(lizard)
    assert engine.__class__.__name__ == "CoreIngestionEngine"

    # an entry whose name matches no known class falls back to the plugin engine
    await crud_settings.upsert_setting_by_category(
        DEFAULT_SYSTEM_KEY,
        Setting(name="NotAConfig", value={}, category="ingestion"),
    )
    assert await resolved_config_name(lizard) == "EfficientIngestionConfiguration"
    _cleanup()


async def test_upsert_stores_category_ingestion_and_value(lizard):
    _cleanup()
    from cat.services.factory.ingestion import build_factory

    sf = build_factory(lizard)
    await sf.upsert_service("EfficientIngestionConfiguration", {"ingestion_max_concurrency": 3})

    entry = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, "EfficientIngestionConfiguration")
    assert entry is not None
    assert entry["category"] == "ingestion"
    assert entry["value"] == {"ingestion_max_concurrency": 3}

    engine = await resolve_ingestion_engine(lizard)
    assert engine.__class__.__name__ == "EfficientIngestionEngine"
    assert engine.ingestion_max_concurrency == 3
    _cleanup()


async def test_upsert_does_not_duplicate_category_entry(lizard):
    """[regression] Editing the ingestion settings must replace the single
    category entry, NOT append a new dict to the ``system:agent`` list — same
    invariant as the embedder category."""
    _cleanup()
    from cat.services.factory.ingestion import build_factory

    sf = build_factory(lizard)
    await sf.upsert_service("EfficientIngestionConfiguration", {"ingestion_max_concurrency": 3})
    await sf.upsert_service("EfficientIngestionConfiguration", {"ingestion_max_concurrency": 7})

    # exactly ONE entry with category "ingestion"
    db = get_sync_db()
    matches = db.json().get("system:agent", '$[?(@.category == "ingestion")]') or []
    assert len(matches) == 1
    assert matches[0]["value"] == {"ingestion_max_concurrency": 7}

    engine = await resolve_ingestion_engine(lizard)
    assert engine.ingestion_max_concurrency == 7
    _cleanup()


async def test_available_configs_include_base_and_efficient(lizard):
    _cleanup()
    from cat.services.factory.ingestion import build_factory

    sf = build_factory(lizard)
    schemas = await sf.get_schemas()
    assert "BaseIngestionConfiguration" in schemas
    assert "EfficientIngestionConfiguration" in schemas


async def test_endpoints_list_and_select(secure_client, secure_client_headers, cheshire_cat, lizard):
    _cleanup()
    admin_headers = await get_client_admin_headers(secure_client)

    listing = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    assert listing.status_code == 200
    body = listing.json()
    names = [s["name"] for s in body["settings"]]
    assert "BaseIngestionConfiguration" in names
    assert "EfficientIngestionConfiguration" in names
    assert body["selected_configuration"] == "EfficientIngestionConfiguration"

    # select the base engine explicitly
    res = await secure_client.put(
        "/ingestion/settings/BaseIngestionConfiguration",
        headers=admin_headers,
        json={},
    )
    assert res.status_code == 200

    # still a single category entry after the PUT (regression guard)
    db = get_sync_db()
    matches = db.json().get("system:agent", '$[?(@.category == "ingestion")]') or []
    assert len(matches) == 1

    listing2 = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    assert listing2.json()["selected_configuration"] == "BaseIngestionConfiguration"
    _cleanup()


async def test_legacy_reembed_max_concurrency_is_migrated(lizard):
    """[migration] A config previously saved with ``reembed_max_concurrency``
    (pre-rename, when the engine only handled the re-embed) keeps working: the
    value is mapped onto ``ingestion_max_concurrency``."""
    _cleanup()
    from cat.db.database import DEFAULT_SYSTEM_KEY as _DS

    # a PREVIOUSLY saved entry still in the DB uses the old field name
    await crud_settings.upsert_setting_by_category(
        _DS,
        Setting(
            name="EfficientIngestionConfiguration",
            value={"reembed_max_concurrency": 8},
            category="ingestion",
        ),
    )

    engine = await resolve_ingestion_engine(lizard)
    assert engine.__class__.__name__ == "EfficientIngestionEngine"
    assert engine.ingestion_max_concurrency == 8
    _cleanup()
