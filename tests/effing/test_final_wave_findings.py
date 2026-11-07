"""Final-wave route/lifecycle finding (B2) kept in core.

The chunker-change case drives the REAL ``PUT /chunking/settings`` route and
the real engine re-run on a full ``CheshireCat``/``BillTheLizard`` boot, so it
stays here as a cross-repo integration test.

The pure lease/heartbeat/NX cases were handed to the plugin's own suite:
``cat-efficient-ingestion/test_final_wave_findings.py``.

Redis keys are isolated by the host ``tests/conftest.py`` (``CAT_REDIS_DB=1``).
"""
from io import BytesIO

from cat.core_plugins.cat_efficient_ingestion import reembed
from cat.core_plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    get_status,
)
from cat.db.cruds import settings as crud_settings
from cat.db.models import Setting
from cat.services.memory.models import VectorMemoryType

_LONG_TEXT = ("The quick brown fox jumps over the lazy dog. " * 120).strip()


async def test_chunker_change_through_the_real_route_reruns_parsing(
    secure_client, secure_client_headers, cheshire_cat, lizard, monkeypatch
):
    """Change the chunker via ``PUT /chunking/settings`` and re-run the engine.

    The route writes the agent-scoped chunker; the phase marker must read that
    same scope, otherwise the rewrite is invisible and the source is skipped.
    The engine really re-parses (the real chunker splits the real text) and the
    stored point count changes.
    """
    # Do not let the route fire the engine in the background: the test drives it
    # explicitly below.
    monkeypatch.setattr("cat.routes.chunker.run_background_task", lambda *a, **k: None)

    # The default file manager is the no-op Dummy one; a real LocalFileManager is
    # needed for the engine's re-embed pass to find the stored source on disk.
    await crud_settings.upsert_setting_by_category(
        cheshire_cat.agent_key,
        Setting(name="LocalFileManagerConfig", value={}, category="file_manager"),
    )
    ccat = await lizard.get_cheshire_cat(cheshire_cat.agent_key)

    engine = reembed.EfficientIngestionEngine()
    source = "final-wave-chunker.txt"

    async def point_count() -> int:
        points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
            str(VectorMemoryType.DECLARATIVE), with_vectors=False
        )
        return sum(
            1
            for p in points
            if (p.payload or {}).get("metadata", {}).get("source") == source
        )

    async def put_chunk_size(size: int) -> None:
        res = await secure_client.put(
            "/chunking/settings/RecursiveTextChunkerSettings",
            headers=secure_client_headers,
            json={"encoding_name": "cl100k_base", "chunk_size": size, "chunk_overlap": 0},
        )
        assert res.status_code == 200, res.text
        current = await crud_settings.get_settings_by_category(ccat.agent_key, "chunker")
        assert current["value"]["chunk_size"] == size

    await put_chunk_size(32)
    # Re-resolve the cat: its services are snapshotted at creation, so the
    # chunker must be re-read after the route changed the stored config.
    ccat = await lizard.get_cheshire_cat(cheshire_cat.agent_key)
    assert ccat.chunker._chunk_size == 32
    await engine.ingest_file(ccat, BytesIO(_LONG_TEXT.encode()), source, content_type="text/plain")
    first = await point_count()
    assert first > 1, f"expected several chunks at chunk_size=32, got {first}"

    doc = await get_status(ccat.agent_key, "agent", source)
    assert doc is not None and doc.get("status") == IngestionStatus.COMPLETED.value
    first_marker = doc["completed_phases"]["parsing_chunking"]["marker"]
    assert first_marker is not None

    # Real route change -> parsing phase must be re-run by the engine.
    await put_chunk_size(128)
    assert await engine.run(lizard) is True

    second = await point_count()
    assert second < first, (
        f"a larger chunk_size must produce fewer points ({second} !< {first}); "
        "the chunker change was not detected through the real route"
    )

    doc = await get_status(ccat.agent_key, "agent", source)
    assert doc.get("status") == IngestionStatus.COMPLETED.value
    assert doc["completed_phases"]["parsing_chunking"]["marker"] != first_marker
