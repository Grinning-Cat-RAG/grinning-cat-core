"""MyGRAPH phase markers must track what actually changes each phase output.

Loads the real plugin ``graphrag_handler`` (relative imports resolved through a
synthetic package) and pins the two material fingerprints:

- ``_spacy_fingerprint``: spaCy models + technology patterns +
  ``enable_entity_extraction`` (the toggle gates per-point extraction).
- ``_concept_fingerprint``: concept/relation definitions + prompt +
  agent-level LLM identity + ``enable_concept_relations``.

The frontend-visualisation toggle (``enable_student_knowledge_graph``) and the
display-only ``enable_knowledge_graph`` must NOT move either marker. Cosmetic
edits (key order, trailing whitespace) must not move them either.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path("/root/git/plugins/cat-graphrag.my")
PACKAGE = "_mygraph_fingerprint"


def _load_handler_module():
    module = sys.modules.get(PACKAGE)
    if module is None:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(PLUGIN_DIR)]
        sys.modules[PACKAGE] = package
    return importlib.import_module(PACKAGE + ".graphrag_handler")


def _handler(**overrides):
    handler_mod = _load_handler_module()
    kwargs = dict(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
        enable_entity_extraction=True,
        enable_concept_relations=True,
        concept_definitions="ALPHA: first\nBETA: second",
        relation_definitions="LINKS: connects",
        concept_relations_prompt="Extract the concepts.",
    )
    kwargs.update(overrides)
    handler = handler_mod.GraphRAGHandler(**kwargs)
    handler.agent_id = "agent_test"
    return handler


def _fixed_llm(handler, identity):
    async def _identity():
        return identity

    handler._resolve_agent_llm_identity = _identity
    return handler


def test_spacy_fingerprint_sensitive_to_entity_extraction_toggle():
    on = _handler(enable_entity_extraction=True)
    off = _handler(enable_entity_extraction=False)
    assert on._spacy_fingerprint() != off._spacy_fingerprint()


def test_spacy_fingerprint_cosmetic_edits_are_stable():
    a = _handler(
        spacy_models={"en": "en_core_web_sm", "it": "it_core_news_sm"},
        extra_technology_patterns=["Python", "  Docker  "],
    )
    b = _handler(
        spacy_models={"it": "it_core_news_sm", "en": "en_core_web_sm"},
        extra_technology_patterns=["Python", "Docker"],
    )
    assert a._spacy_fingerprint() == b._spacy_fingerprint()


@pytest.mark.asyncio
async def test_concept_fingerprint_moves_with_llm_identity():
    a = _fixed_llm(_handler(), "OpenRouterConfig:qwen/qwen3")
    b = _fixed_llm(_handler(), "OpenRouterConfig:llama/llama-3")
    assert await a._concept_fingerprint() != await b._concept_fingerprint()


@pytest.mark.asyncio
async def test_concept_fingerprint_moves_with_definitions_and_prompt():
    base = _fixed_llm(_handler(), "LLM:model")
    other_defs = _fixed_llm(
        _handler(concept_definitions="ALPHA: first\nGAMMA: third"), "LLM:model"
    )
    other_rel = _fixed_llm(
        _handler(relation_definitions="LINKS: connects\nCAUSES: causes"), "LLM:model"
    )
    other_prompt = _fixed_llm(
        _handler(concept_relations_prompt="Extract the concepts differently."),
        "LLM:model",
    )
    base_fp = await base._concept_fingerprint()
    assert await other_defs._concept_fingerprint() != base_fp
    assert await other_rel._concept_fingerprint() != base_fp
    assert await other_prompt._concept_fingerprint() != base_fp


@pytest.mark.asyncio
async def test_concept_fingerprint_moves_with_concept_relations_toggle():
    on = _fixed_llm(_handler(enable_concept_relations=True), "LLM:model")
    off = _fixed_llm(_handler(enable_concept_relations=False), "LLM:model")
    assert await on._concept_fingerprint() != await off._concept_fingerprint()


@pytest.mark.asyncio
async def test_concept_fingerprint_ignores_display_only_toggles():
    base = _fixed_llm(_handler(), "LLM:model")
    student = _fixed_llm(
        _handler(enable_student_knowledge_graph=True), "LLM:model"
    )
    knowledge = _fixed_llm(_handler(enable_knowledge_graph=True), "LLM:model")
    base_fp = await base._concept_fingerprint()
    assert await student._concept_fingerprint() == base_fp
    assert await knowledge._concept_fingerprint() == base_fp


@pytest.mark.asyncio
async def test_concept_fingerprint_cosmetic_edits_are_stable():
    a = _fixed_llm(
        _handler(
            concept_definitions="ALPHA: first\nBETA: second",
            relation_definitions="LINKS: connects",
            concept_relations_prompt="Extract the concepts.   \n",
        ),
        "LLM:model",
    )
    b = _fixed_llm(
        _handler(
            concept_definitions="BETA: second\nALPHA: first",
            relation_definitions="LINKS: connects",
            concept_relations_prompt="  Extract the concepts.",
        ),
        "LLM:model",
    )
    assert await a._concept_fingerprint() == await b._concept_fingerprint()


@pytest.mark.asyncio
async def test_llm_identity_is_config_name_and_model_never_secrets(monkeypatch):
    handler_mod = _load_handler_module()
    handler = handler_mod.GraphRAGHandler(
        neo4j_uri="bolt://fake",
        neo4j_user="u",
        neo4j_password="p",
    )
    handler.agent_id = "agent_test"

    async def fake_settings(agent_key, category):
        assert category == "llm"
        return {
            "name": "OpenRouterConfig",
            "value": {"model": "qwen/qwen3", "api_key": "SECRET-DO-NOT-LEAK"},
        }

    monkeypatch.setattr(
        "cat.db.cruds.settings.get_settings_by_category", fake_settings
    )
    identity = await handler._resolve_agent_llm_identity()
    assert identity == "OpenRouterConfig:qwen/qwen3"
    assert "SECRET" not in identity
