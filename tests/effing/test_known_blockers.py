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
- **B3** a source stuck in ``downloaded`` is not in the claimable status set,
  so ``claim_source_for_resume`` refuses it -> permanent dead end.
- **B4** BLOCKED-SEAM: the completion veto has no exception seam yet. The
  dispatcher (``PluginManager._run_hooks``) re-raises only
  ``ManagementModeException``; every other exception is logged and swallowed,
  so no honest behavioural test can be written before the dedicated
  ``CustomHookVetoException`` lands in ``cat/exceptions.py``.

Every seeded key lives under ``agents:effingtest_<uuid>:`` and is destroyed in
``finally``. The host ``tests/conftest.py`` forces ``CAT_REDIS_DB=1``.
"""
import inspect
import time
import uuid

import pytest

from cat.core_plugins.cat_efficient_ingestion import hooks
from cat.core_plugins.cat_efficient_ingestion.phases import PHASES, is_phase_stale
from cat.core_plugins.cat_efficient_ingestion.registry import (
    IngestionStatus,
    claim_source_for_resume,
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


async def test_b3_downloaded_source_can_be_resumed():
    """B3: a stale ``downloaded`` source is claimable for a retry.

    ``downloaded`` is terminal in practice today: the claim guard only accepts
    ``uploaded``/``processing``/``error``, so the resume sweep never picks it
    up and the source is a permanent dead end.
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


@pytest.mark.skip(
    reason="BLOCKED-SEAM: no CustomHookVetoException yet; _run_hooks swallows every "
    "non-ManagementModeException, so the veto cannot be observed behaviourally."
)
async def test_b4_completion_veto_must_survive_the_hook_dispatcher():
    """B4: a vetoing completion hook must force the source to ``error``.

    Cannot be written honestly until the dedicated veto exception exists and
    ``PluginManager._run_hooks`` re-raises it (plan todo 6).
    """
    raise AssertionError("blocked seam")
