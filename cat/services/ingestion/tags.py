"""Core ingestion tags.

A phase's tag is a detached, deterministic identity of the settings version it
consumed. The core uses the settings entry's ``updated_at`` as an opaque version
token: it hashes that value and never reads a clock. No LLM call is made and no
time-based decision is taken — a settings change bumps ``updated_at`` and
therefore the tag, and an unchanged settings entry produces the same tag.

Import-safe: importing this module performs no Redis, filesystem or network
access. Settings are read only when :meth:`SettingsVersionTags.resolve` runs.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping

from cat.db.cruds import settings as crud_settings


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class IngestionTagResolver(ABC):
    """Contract for resolving a phase's settings-version tag."""

    @abstractmethod
    def own_tag(self, phase, settings) -> str:
        """Return the detached tag of one phase from its settings entry."""

    @abstractmethod
    async def resolve(self, cat, phases) -> dict[str, str]:
        """Return ``{phase_id: tag}`` for every phase."""


class SettingsVersionTags(IngestionTagResolver):
    """Tag a phase with the sha256 of its settings entry's ``updated_at``.

    The ``updated_at`` value is an opaque version token, not a correctness
    clock: the tag only changes when the stored settings entry changes, and the
    same entry always yields the same tag.
    """

    def own_tag(self, phase, settings) -> str:
        """Return ``sha256(phase.id + ":" + settings["updated_at"])``."""
        updated_at = (
            settings.get("updated_at") if isinstance(settings, Mapping) else None
        )
        return _digest(f"{phase.id}:{updated_at}")

    async def resolve(self, cat, phases) -> dict[str, str]:
        """Resolve every phase's tag from its settings category entry."""
        tags: dict[str, str] = {}
        for phase in phases:
            category = getattr(phase, "settings_category", None)
            settings = None
            if category:
                settings = await crud_settings.get_settings_by_category(
                    cat.agent_key, category
                )
            tags[phase.id] = self.own_tag(phase, settings or {})
        return tags
