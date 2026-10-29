"""RED baseline for the four known EffING blockers (plan todo 1).

These tests are deliberately written against the PUBLIC seams the plan intends
to fix, not against the current implementation, so they stay RED until the
corresponding todo lands:

- **B1** status listing returns nothing: the registry enumerates the
  ``agents:<id>:ingestion:*`` namespace through ``get_agents_main_keys``,
  whose ``split(":")[1]`` yields agent ids, not full keys.
- **B2** the built-in ``ingestion_phase_settings_marker`` hook returns ``None``
  for every phase, so a rewritten embedder setting is judged "not material"
  and the ``embedding`` phase stays fresh -> no re-embed.
- **B3** a source stuck in ``downloaded`` used to be absent from the claimable
  status set, so ``claim_source_for_resume`` refused it -> permanent dead end.
  The registry now treats ``downloading``/``downloaded`` as resumable, gated by
  the existing ``updated_at`` staleness threshold.
- **B4** the completion veto used to be swallowed: the dispatcher
  (``PluginManager._run_hooks``) re-raised only ``ManagementModeException``;
  every other exception was logged and swallowed. The dedicated
  ``CustomHookVetoException`` now lands in ``cat/exceptions.py`` and is
  re-raised, so a vetoing ``before_ingestion_status_completed`` hook forces the
  source to ``error`` instead of ``completed``.

Every seeded key lives under ``agents:effingtest_<uuid>:`` and is destroyed in
``finally``. The host ``tests/conftest.py`` forces ``CAT_REDIS_DB=1``.
"""
import inspect
import time
import uuid

from cat.core_plugins.cat_efficient_ingestion import hooks
from cat.core_plugins.cat_efficient_ingestion import plugin as ingestion_plugin
from cat.core_plugins.cat_efficient_ingestion.phases import PHASES, is_phase_stale
from cat.core_plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    claim_source_for_resume,
    get_status,
    list_statuses,
    set_status,
    status_key,
)
from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.db.models import Setting


def _agent_id() -> str:
    return f"effingtest_{uuid.uuid4().hex}"


async def _drop(agent_id: str) -> None:
    await crud.destroy(f"agents:{agent_id}:*")


async def _seed_master_key(agent_id: str) -> None:
    await crud.store(f"agents:{agent_id}:agent", [{"name": "x", "value": 1}])


async def _marker(phase: str, cat) -> object:
    """Call the built-in marker hook, awaiting only if the hook is async."""
    result = hooks.ingestion_phase_settings_marker.function(phase, cat)
    return await result if inspect.isawaitable(result) else result


async def test_b1_status_listing_returns_the_seeded_rows():
    """B1: ``list_statuses`` must return the agent's own status docs."""
    agent = _agent_id()
    try:
        await _seed_master_key(agent)
        await set_status(agent, "agent", "doc.pdf", type_="file", status=IngestionStatus.COMPLETED)

        docs = await list_statuses(agent)

        assert [d["source"] for d in docs] == ["doc.pdf"], (
            "B1: list_statuses returned no rows although a status doc exists "
            "(get_agents_main_keys yields agent ids, not full keys)"
        )
    finally:
        await _drop(agent)


async def test_b2_embedder_change_changes_the_embedding_phase_marker(cheshire_cat):
    """B2: a rewritten embedder setting must produce a DIFFERENT phase marker.

    The marker is what turns a settings rewrite into a real re-embed; with a
    constant ``None`` marker the slow-path always sees "no change" and the
    ``embedding`` phase is deemed fresh.
    """
    marker_before = await _marker("embedding", cheshire_cat)

    await crud_settings.upsert_setting_by_category(
        DEFAULT_SYSTEM_KEY,
        Setting(name="EmbeddingSettings", value={"embeddingName": "embedder-b"}, category="embedder"),
    )
    marker_after = await _marker("embedding", cheshire_cat)

    assert marker_before is not None, (
        "B2: the built-in embedding marker is None -> an embedder change cannot be detected"
    )
    assert marker_after != marker_before, (
        "B2: changing the embedder left the embedding marker unchanged -> no re-embed"
    )

    # And the staleness decision built on top of it must flag the phase.
    diary = {
        "parsing_chunking": {"marker": "pc-marker", "settings_version": "sv", "deps": {}},
        "embedding": {
            "marker": marker_before,
            "settings_version": "stale-token",  # settings entry WAS rewritten
            "deps": {"parsing_chunking": "pc-marker"},
        },
    }
    assert is_phase_stale(PHASES["embedding"], diary, "current-token", marker_after) is True, (
        "B2: the embedding phase is deemed fresh after an embedder change"
    )


async def test_b2b_chunker_change_changes_the_parsing_phase_marker(cheshire_cat):
    """B2 (chunker): a rewritten chunker setting must mark parsing stale.

    Chunker and embedder are system-level services, so the marker is resolved
    from the ``DEFAULT_SYSTEM_KEY`` settings entry. Same config -> identical
    marker; a material chunker change -> different marker -> stale phase.
    """
    base = {"encoding_name": "cl100k_base", "chunk_size": 256, "chunk_overlap": 64}
    await crud_settings.upsert_setting_by_category(
        DEFAULT_SYSTEM_KEY,
        Setting(name="RecursiveTextChunkerSettings", value=dict(base), category="chunker"),
    )
    marker_before = await _marker("parsing_chunking", cheshire_cat)
    marker_same = await _marker("parsing_chunking", cheshire_cat)

    await crud_settings.upsert_setting_by_category(
        DEFAULT_SYSTEM_KEY,
        Setting(
            name="RecursiveTextChunkerSettings",
            value={**base, "chunk_size": 512},
            category="chunker",
        ),
    )
    marker_after = await _marker("parsing_chunking", cheshire_cat)

    assert marker_before is not None, (
        "B2: the built-in parsing_chunking marker is None -> a chunker change cannot be detected"
    )
    assert marker_same == marker_before, (
        "B2: the parsing_chunking marker is not deterministic for an unchanged chunker"
    )
    assert marker_after != marker_before, (
        "B2: changing the chunker left the parsing_chunking marker unchanged -> no re-parse"
    )

    diary = {
        "parsing_chunking": {
            "marker": marker_before,
            "settings_version": "stale-token",
            "deps": {},
        },
    }
    assert is_phase_stale(PHASES["parsing_chunking"], diary, "current-token", marker_after) is True, (
        "B2: parsing_chunking is deemed fresh after a chunker change"
    )


async def test_b3_downloaded_source_can_be_resumed():
    """B3: a stale ``downloaded`` source is claimable for a retry.

    The claim guard accepts ``downloading``/``downloaded`` too, so the resume
    sweep reclaims a download that crashed before processing instead of leaving
    the source a permanent dead end.
    """
    agent = _agent_id()
    source = "https://example.com/stuck.pdf"
    try:
        await _seed_master_key(agent)
        await set_status(
            agent, "agent", source, type_="url", status=IngestionStatus.DOWNLOADED
        )
        # make the row look old enough for the staleness gate
        key = status_key(agent, "agent", source)
        stored = await crud.read(key)
        if isinstance(stored, list):
            stored = stored[0]
        stored["updated_at"] = time.time() - 10_000
        await crud.store(key, stored)

        claimed = await claim_source_for_resume(
            agent, "agent", source, stale_after=0, owner="b3-test"
        )

        assert claimed is not None, (
            "B3: a stale 'downloaded' source is never claimed -> permanent dead end"
        )
        assert claimed["status"] == IngestionStatus.PROCESSING.value
    finally:
        await _drop(agent)


async def test_b3b_active_download_is_not_stolen():
    """B3 (guard): a fresh ``downloading`` row is never stolen.

    The resumable policy must not reclaim a row that is still actively
    downloading: the existing ``updated_at`` staleness gate protects it.
    """
    agent = _agent_id()
    source = "https://example.com/active.pdf"
    try:
        await _seed_master_key(agent)
        await set_status(
            agent, "agent", source, type_="url", status=IngestionStatus.DOWNLOADING
        )

        claimed = await claim_source_for_resume(
            agent, "agent", source, stale_after=300, owner="b3b-test"
        )

        assert claimed is None, (
            "B3: a fresh 'downloading' source was stolen by the resume sweep"
        )
        doc = await get_status(agent, "agent", source)
        assert doc is not None
        assert doc["status"] == IngestionStatus.DOWNLOADING.value
    finally:
        await _drop(agent)


async def test_b4_completion_veto_must_survive_the_hook_dispatcher(stray):
    """B4: a vetoing completion hook must force the source to ``error``.

    Drives the plugin's real completion handler through the real
    ``PluginManager._run_hooks`` dispatcher. The built-in
    ``before_ingestion_status_completed`` vetoes because the caller reports
    pending source tasks; the dedicated ``CustomHookVetoException`` must cross
    the dispatcher instead of being swallowed, so the source ends in ``error``
    (never ``completed``) with the veto reason recorded.
    """
    agent = _agent_id()
    pm = stray.plugin_manager
    original_pending = pm.hooks.get("ingestion_phase_pending")
    original_gate = pm.hooks.get("before_ingestion_status_completed")

    class _VetoingCat:
        """Agent-scoped caller that reports pending ingestion tasks."""

        def __init__(self, agent_key):
            self.agent_key = agent_key
            self.plugin_manager = pm

        def has_pending_source_tasks(self, source):
            return True

    cat = _VetoingCat(agent)

    try:
        await _seed_master_key(agent)
        # Real EffING hooks, dispatched by the real _run_hooks implementation.
        pm.hooks["ingestion_phase_pending"] = [hooks.ingestion_phase_pending]
        pm.hooks["before_ingestion_status_completed"] = [
            hooks.before_ingestion_status_completed
        ]

        await ingestion_plugin.rabbithole_ingestion_start.function(
            "doc.pdf", {}, False, cat
        )
        await ingestion_plugin.after_rabbithole_stored_documents.function(
            "doc.pdf", [object()], cat
        )

        doc = await get_status(agent, "agent", "doc.pdf")
        assert doc is not None
        assert doc["status"] == IngestionStatus.ERROR.value, (
            "B4: a vetoing completion hook was swallowed -> source completed "
            f"instead of erroring (status={doc.get('status')!r})"
        )
        assert doc["status"] != IngestionStatus.COMPLETED.value
        assert "Pending ingestion tasks" in doc["error"]
    finally:
        if original_pending is None:
            pm.hooks.pop("ingestion_phase_pending", None)
        else:
            pm.hooks["ingestion_phase_pending"] = original_pending
        if original_gate is None:
            pm.hooks.pop("before_ingestion_status_completed", None)
        else:
            pm.hooks["before_ingestion_status_completed"] = original_gate
        await _drop(agent)
