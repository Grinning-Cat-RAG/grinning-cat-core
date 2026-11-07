"""The core ingestion-phase contract.

``IngestionPhase`` is the abstract unit the pipeline is built from. It is moved
verbatim from the ``efficient_ingestion`` plugin's ``phases.py`` so that the
core and any adapter share the exact method surface; implementations still live
in the plugins that ship them.

Import-safe: no Redis, no database, no filesystem and no network access at
import time.
"""

import hashlib
import inspect
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from cat.env import get_env_float
from cat.log import log

from .models import PhaseResult, PhaseSpec  # noqa: F401  (re-exported for phase implementers)


def canonical_json(value: Any) -> str:
    """Render ``value`` as canonical JSON (sorted keys, no incidental spacing)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def own_tag(kind: str, relevant_settings: Any) -> str:
    """Detached tag of one phase from its kind and relevant settings only."""
    return _digest(f"{kind}:{canonical_json(relevant_settings)}")


async def relevant_settings(settings: Any, settings_check: Any = None) -> Any:
    """Return the settings slice that participates in the tag.

    - ``settings_check`` is ``None``: the whole ``settings`` value.
    - ``settings_check`` is callable: its return value (awaited when it is a
      coroutine) — the async ``settings_check`` contract.
    - ``settings_check`` is a key (or an iterable of keys): only those keys are
      kept from a mapping ``settings``.
    """
    if settings_check is None:
        return settings
    if callable(settings_check):
        value = settings_check(settings)
        if inspect.isawaitable(value):
            value = await value
        return value
    keys = [settings_check] if isinstance(settings_check, str) else list(settings_check)
    if not isinstance(settings, Mapping):
        return {}
    return {key: settings.get(key) for key in keys if key in settings}


class IngestionPhase(ABC):
    """One ingestion phase, owning its applicability, work and cleanup.

    Every phase MUST implement all four operations. ``delete`` is deliberately
    abstract: a phase that cannot clean up its own artifacts would make the
    recovery path delete data blindly, so it cannot exist. There is no no-op
    default anywhere — a subclass without ``delete`` raises ``TypeError`` at
    instantiation time.
    """

    #: Stable phase id (the phase dict key, e.g. ``"embedding"``).
    id: str = ""
    #: Human-readable label for display only. Never a key: the machine
    #: identity is always :attr:`id`. ``None`` falls back to :attr:`id`.
    label: str | None = None
    #: Upstream phase ids this phase consumes.
    depends_on: tuple[str, ...] = ()
    #: Settings category whose material configuration feeds :meth:`tag`.
    settings_category: str | None = None
    #: Optional narrowing of the settings that feed :meth:`tag` (see
    #: :class:`PhaseSpec`). ``None`` uses the whole settings value.
    settings_check: Any = None
    #: Optional wall-clock budget (seconds) this phase may stay ``not_ready``
    #: before the machine marks it ``error``. ``None`` uses the generic
    #: ``CAT_INGESTION_NOT_READY_MAX_SECONDS`` budget. An LLM-backed phase sets
    #: this (and :attr:`not_ready_budget_env`) so an upstream model failure
    #: does not hold the source for the generic hour.
    not_ready_budget_seconds: float | None = None
    #: Optional env var overriding :attr:`not_ready_budget_seconds` when set to
    #: a non-negative number. Unset/invalid falls back to the declared value.
    not_ready_budget_env: str | None = None

    def not_ready_budget(self) -> float | None:
        """Resolve this phase's own not-ready budget, or ``None``.

        Resolution order for the PHASE's budget:

        1. :attr:`not_ready_budget_env`, when declared and set to a
           non-negative number;
        2. :attr:`not_ready_budget_seconds`, when declared and coercible to
           a non-negative finite number.

        Returns ``None`` when the phase declares no budget, which makes the
        caller fall back to the generic
        ``CAT_INGESTION_NOT_READY_MAX_SECONDS``.
        """
        if self.not_ready_budget_env:
            try:
                value = get_env_float(self.not_ready_budget_env)
            except (TypeError, ValueError):
                value = None
            # A non-finite value (``inf``/``nan``) would either never expire or
            # break the comparison, and a negative budget is meaningless: both
            # are rejected so resolution falls through to the next step.
            if value is not None and math.isfinite(value) and value >= 0:
                return float(value)
        declared = self.not_ready_budget_seconds
        if declared is None:
            return None
        try:
            # The attribute is typed ``float | None``, but it is a class
            # attribute a third-party phase may set to anything. A numeric
            # string ("300") is coerced; a genuinely wrong type is rejected
            # instead of raising out of the resolver.
            if isinstance(declared, bool):
                raise TypeError("a bool is not a numeric budget")
            value = float(declared)
        except (TypeError, ValueError):
            log.warning(
                f"Phase '{self.id}' declares an invalid "
                f"not_ready_budget_seconds {declared!r}; ignoring it and "
                "falling back to the generic budget."
            )
            return None
        # The same rules as the env path: a non-finite value would never
        # expire (or break the comparison) and a negative budget is
        # meaningless, so warn about the bad declared value and fall through.
        if not math.isfinite(value) or value < 0:
            log.warning(
                f"Phase '{self.id}' declares an invalid "
                f"not_ready_budget_seconds {declared!r}; ignoring it and "
                "falling back to the generic budget."
            )
            return None
        return value

    @property
    def name(self) -> str:
        """Stable phase name (alias of :attr:`id`)."""
        return self.id

    def display_label(self) -> str:
        """Human-readable label for this phase.

        Returns :attr:`label` when declared, else :attr:`id`. The label is
        display-only: clients localize by ``id`` and the machine never uses
        the label as a key.
        """
        return self.label or self.id

    @abstractmethod
    async def applies_to(self, cat, deps_state) -> tuple[bool, str | None]:
        """Whether this phase applies, with an optional human-readable reason."""

    @abstractmethod
    async def process(self, source: str, cat) -> PhaseResult:
        """Run the phase body for ``source``."""

    @abstractmethod
    async def delete(self, source: str, cat) -> None:
        """Delete ONLY the artifacts this phase produced for ``source``.

        Must raise when the phase has no cleanup implementation: a silent
        no-op would let recovery delete data blindly.
        """

    async def tag(self, settings: Any = None) -> str:
        """Detached identity of this phase's OWN material configuration.

        ``sha256(kind + ":" + canonical_json(relevant_settings))`` where
        ``relevant_settings`` is ``settings_check``-narrowed when declared,
        else the whole ``settings`` value. Async because a declared
        ``settings_check`` may be a coroutine. Only configuration is read —
        never an LLM call. Folding in dependency tags is
        :func:`phase_tag_resolver.resolve_tags`' job.
        """
        relevant = await relevant_settings(
            {} if settings is None else settings, self.settings_check
        )
        return own_tag(self.id, relevant)
