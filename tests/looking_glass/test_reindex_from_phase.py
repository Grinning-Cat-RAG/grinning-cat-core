"""Reindexing a source from a chosen phase: the phases it depends on keep their artifacts and only the wanted ones are
purged, in reverse execution order. The CLI rejects an unknown phase id."""
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from cat.db.cruds import ingestions as crud_ingestions
from cat.rabbit_hole import RabbitHole, chunks, engine
from cat.rabbit_hole.phases import IngestionPhase, PhaseResult

ROOT = Path(__file__).resolve().parents[2]
ALL = [
    "parsing_chunking",
    "embedding",
    "catalog_cards",
    "graph_spacy_extraction",
    "graph_llm_extraction",
]


def phase_class(phase_id, deleted, depends_on=(), artifacts_group=None):
    class FakePhase(IngestionPhase):
        id = phase_id

        async def process(self, source, cat):
            return PhaseResult(status="done")

        async def delete(self, source, cat):
            deleted.append(phase_id)

    FakePhase.depends_on = tuple(depends_on)
    FakePhase.artifacts_group = artifacts_group
    return FakePhase


def phases(deleted):
    return [
        phase_class("parsing_chunking", deleted)(),
        phase_class("embedding", deleted, ("parsing_chunking",))(),
        phase_class("catalog_cards", deleted, ("embedding",))(),
        phase_class("graph_spacy_extraction", deleted, ("embedding",), "graph")(),
        phase_class("graph_llm_extraction", deleted, (), "graph")(),
    ]


@pytest.fixture
def reindex(monkeypatch):
    deleted, calls = [], {}
    collected = phases(deleted)
    cat = SimpleNamespace(agent_key="agent", file_manager=None)

    async def fake_setup(self, _cat):
        self.cat, self.stray = _cat, None

    monkeypatch.setattr(RabbitHole, "setup", fake_setup)

    @contextlib.asynccontextmanager
    async def fake_lock(source, *args, **kwargs):
        yield

    monkeypatch.setattr(engine, "source_lock", fake_lock)

    async def fake_collect(rabbit_hole, _cat):
        return collected

    monkeypatch.setattr(engine, "collect_phases", fake_collect)
    monkeypatch.setattr(chunks, "read_source_bytes", lambda *args, **kwargs: b"kept")

    async def fake_status(*args, **kwargs):
        return None

    monkeypatch.setattr(crud_ingestions, "get_status", fake_status)

    async def fake_adopt(self, source, require_points):
        calls["adopt"] = require_points
        return True

    monkeypatch.setattr(RabbitHole, "_adopt", fake_adopt)

    async def fake_reset(agent_id, chat_id, source, phase_ids):
        calls["wanted"] = sorted(phase_ids)
        return True

    monkeypatch.setattr(crud_ingestions, "reset_phase_entries", fake_reset)

    async def fake_run(rabbit_hole, _cat, source, *, reindexing=False):
        calls["reindexing"] = reindexing
        return {"status": "completed"}

    monkeypatch.setattr(engine, "run_phases", fake_run)

    return SimpleNamespace(cat=cat, deleted=deleted, calls=calls)


async def test_from_embedding_keeps_parsing_and_purges_reverse(reindex):
    result = await RabbitHole().reingest_source(reindex.cat, "source.pdf", from_phase="embedding")

    assert result == {"status": "completed"}
    assert reindex.calls["wanted"] == sorted(
        ["embedding", "catalog_cards", "graph_spacy_extraction", "graph_llm_extraction"]
    )
    assert "parsing_chunking" not in reindex.calls["wanted"]
    assert reindex.deleted == [
        "graph_llm_extraction",
        "graph_spacy_extraction",
        "catalog_cards",
        "embedding",
    ]
    assert reindex.calls["reindexing"] is True


async def test_from_parsing_runs_and_purges_every_phase(reindex):
    await RabbitHole().reingest_source(reindex.cat, "source.pdf", from_phase="parsing_chunking")

    assert reindex.calls["wanted"] == sorted(ALL)
    assert reindex.deleted == list(reversed(ALL))


async def test_from_catalog_purges_only_the_catalog(reindex):
    await RabbitHole().reingest_source(reindex.cat, "source.pdf", from_phase="catalog_cards")

    assert reindex.calls["wanted"] == ["catalog_cards"]
    assert reindex.deleted == ["catalog_cards"]


async def test_from_graph_spacy_includes_the_whole_graph_group(reindex):
    await RabbitHole().reingest_source(reindex.cat, "source.pdf", from_phase="graph_spacy_extraction")

    assert reindex.calls["wanted"] == ["graph_llm_extraction", "graph_spacy_extraction"]
    assert reindex.deleted == ["graph_llm_extraction", "graph_spacy_extraction"]


def load_cli():
    spec = importlib.util.spec_from_file_location("reindex_ingestion", ROOT / "scripts" / "reindex_ingestion.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_unknown_phase_is_rejected_with_the_valid_ids(monkeypatch, capsys):
    module = load_cli()
    cat = SimpleNamespace(agent_key="agent", file_manager=None)

    async def fake_requested_agent(agent_id):
        return cat

    async def fake_enumerate(_cat, agent_id):
        return [("source.pdf", None, "file")]

    async def fake_rabbit_hole(_cat):
        return SimpleNamespace()

    async def fake_collect(rabbit_hole, _cat):
        return phases([])

    monkeypatch.setattr(module.engine, "collect_phases", fake_collect)
    monkeypatch.setattr(module, "_enumerate", fake_enumerate)
    monkeypatch.setattr(module, "_rabbit_hole", fake_rabbit_hole)

    lizard = SimpleNamespace(get_cheshire_cat=fake_requested_agent)
    rc = await module.reindex_agent(lizard, "agent", None, False, from_phase="unknown")

    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown phase 'unknown'" in err
    for phase_id in ALL:
        assert phase_id in err
