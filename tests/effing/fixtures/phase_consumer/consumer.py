"""Tiny third-party plugin exposing one ingestion phase (test fixture).

Mirrors the contract documented in the EffING README:

- ``PhaseSpec`` is imported from the CANONICAL path
  (``cat.core_plugins.cat_efficient_ingestion.phases``) behind ``try/except
  ImportError`` at module top level, so EffING stays optional;
- the phase hooks are defined ONLY when that import succeeded;
- the classic CAT ingestion hooks are always defined, so basic ingestion keeps
  working when EffING is absent.

The host never imports this file: it lives under ``tests/``, not in a plugin
folder.
"""
from cat import hook

try:
    from cat.core_plugins.cat_efficient_ingestion.phases import PhaseSpec
except ImportError:  # EffING not installed: degrade to classic ingestion only
    PhaseSpec = None


PHASE_ID = "consumer_phase"


@hook(priority=0)
def rabbithole_ingestion_processing(source, cat) -> None:
    """Classic CAT ingestion hook: always present, independent of EffING."""
    return None


if PhaseSpec is not None:

    @hook(priority=0)
    def ingestion_phase_specs(specs, cat) -> list:
        assert PhaseSpec is not None
        specs = specs if specs is not None else []
        specs.append(PhaseSpec(PHASE_ID, None, ("embedding",)))
        return specs

    @hook(priority=0)
    def ingestion_phase_settings_marker(phase, cat):
        return "consumer-marker" if phase == PHASE_ID else None

    @hook(priority=0)
    def ingestion_phase_run(phase, source, completed_phases, cat):
        if phase == PHASE_ID:
            return {"status": "done"}
        return None
