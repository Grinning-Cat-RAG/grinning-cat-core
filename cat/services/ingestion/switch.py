"""Transition between two ingestion profiles.

Changing the configured ingestor is a three-step operation, owned by the profile
the OLD configuration resolved:

1. delete every artifact the old profile created, for every agent and every
   source of that agent (the same per-source ``pipeline.delete`` operation used
   by the file-manager route and the agent teardown);
2. clear the old profile's state rows;
3. re-ingest with the NEW engine (``engine.run``).

The transition is idempotent: a half-done switch converges on a re-run, because
``pipeline.delete`` is idempotent and ``engine.run`` re-ingests. It never fails
silently: on any error the previous setting is upserted back, so the old
configuration stays in force.

Import-safe: importing this module performs no Redis, filesystem or network
access.
"""

from typing import TYPE_CHECKING, Any

from cat.db import models
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.log import log
from cat.services.ingestion.profile import IngestionProfile

if TYPE_CHECKING:
    from cat.looking_glass.bill_the_lizard import BillTheLizard

#: The agent knowledge base scope: profile switches always target the agent KB,
#: never a conversation.
_AGENT_SCOPE = "agent"


def profile_signature(profile: IngestionProfile) -> tuple[type, ...]:
    """Return the class signature of every part of ``profile``.

    Two profiles are the same "profile class" only when all five replaceable
    parts resolve to the same implementation. A settings-only change leaves the
    signature unchanged, so it stays incremental and never triggers a
    transition.
    """
    return (
        type(profile.pipeline),
        type(profile.state_store),
        type(profile.recovery),
        type(profile.locks),
        type(profile.tags),
    )


def profile_changed(old_profile: IngestionProfile, new_profile: IngestionProfile) -> bool:
    """Whether switching from ``old_profile`` to ``new_profile`` needs a transition."""
    return profile_signature(old_profile) != profile_signature(new_profile)


async def _restore_previous_setting(previous_setting: dict[str, Any] | None) -> None:
    """Upsert ``previous_setting`` back as the active ingestion configuration.

    Restoring goes through the category upsert: the ingestion category is a
    single-entry setting, so replacing by category removes the new selection
    instead of appending the old one beside it.
    """
    if not previous_setting:
        log.error("Ingestion profile switch failed and there is no previous setting to restore")
        return

    try:
        await crud_settings.upsert_setting_by_category(
            DEFAULT_SYSTEM_KEY,
            models.Setting(
                name=previous_setting["name"],
                category=previous_setting.get("category", "ingestion"),
                value=previous_setting["value"],
            ),
        )
        log.warning(
            "Ingestion profile switch failed: restored the previous configuration "
            f"'{previous_setting['name']}'"
        )
    except Exception as e:  # noqa: BLE001 - the rollback is best-effort, log and keep the old state
        log.error(f"Ingestion profile switch failed and the previous configuration could not be restored: {e}")


async def transition_ingestion_profile(
    lizard: "BillTheLizard",
    old_profile: IngestionProfile,
    new_profile: IngestionProfile,
    previous_setting: dict[str, Any] | None,
) -> bool:
    """Move every agent's sources from ``old_profile`` to ``new_profile``.

    Args:
        lizard: The orchestrator, used to boot each agent and resolve the new
            engine.
        old_profile: The profile that created the current artifacts.
        new_profile: The profile just selected.
        previous_setting: The stored setting (name/value) before the switch,
            used to roll back on failure.

    Returns:
        ``True`` when the transition completed, ``False`` when it failed and
        the previous configuration was restored.
    """
    try:
        for agent_id in await crud_settings.get_agents_main_keys():
            ccat = await lizard.get_cheshire_cat(agent_id)
            if ccat is None:
                continue

            # Enumerate through the file manager the core already uses: it is
            # the single source of truth for the stored sources of an agent.
            sources = [file.name for file in ccat.file_manager.list_files(ccat.agent_key)]

            for source in sources:
                await old_profile.pipeline.delete(source, ccat, _AGENT_SCOPE)
                await old_profile.state_store.clear(agent_id, _AGENT_SCOPE, source)

        # Re-ingest with the NEW engine: the engine is resolved after the
        # upsert, so it already points at the new configuration.
        log.info(
            "Re-ingesting with the new ingestion profile "
            f"({type(new_profile.pipeline).__name__})"
        )
        engine = await lizard.ingestion()
        await engine.run(lizard)
        return True
    except Exception as e:  # noqa: BLE001 - any failure restores the old configuration
        log.error(f"Ingestion profile switch failed: {e}")
        await _restore_previous_setting(previous_setting)
        return False
