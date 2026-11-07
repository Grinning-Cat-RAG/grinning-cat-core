"""Contract tests for the core ingestion tags.

``SettingsVersionTags`` hashes the settings entry's ``updated_at``: the same
entry yields the same tag (deterministic) and a changed entry yields a different
tag. No LLM is called and no clock is read.

Import-safe: the tags module must not touch Redis at import time.
"""

import types
from typing import Any

from cat.services.ingestion import tags as tags_module
from cat.services.ingestion.tags import IngestionTagResolver, SettingsVersionTags


class _Phase:
    id = "ingestion"
    settings_category = "ingestion"


def test_settings_version_tags_implements_the_contract():
    assert issubclass(SettingsVersionTags, IngestionTagResolver)


def test_own_tag_is_deterministic_and_changes_with_the_settings_entry():
    resolver = SettingsVersionTags()
    phase = _Phase()

    first = resolver.own_tag(phase, {"updated_at": 1000.0})
    same = resolver.own_tag(phase, {"updated_at": 1000.0})
    changed = resolver.own_tag(phase, {"updated_at": 2000.0})

    assert first == same
    assert first != changed


async def test_resolve_reads_the_settings_entry_per_phase(monkeypatch):
    calls: list[Any] = []

    async def fake_get_settings_by_category(key_id, category):
        calls.append((key_id, category))
        return {"updated_at": 1000.0, "category": category}

    monkeypatch.setattr(
        tags_module.crud_settings,
        "get_settings_by_category",
        fake_get_settings_by_category,
    )

    resolver = SettingsVersionTags()
    cat = types.SimpleNamespace(agent_key="agent")

    resolved = await resolver.resolve(cat, [_Phase()])

    assert calls == [("agent", "ingestion")]
    assert resolved == {"ingestion": resolver.own_tag(_Phase(), {"updated_at": 1000.0})}
