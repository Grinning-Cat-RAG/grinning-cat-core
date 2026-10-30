"""Regression: the catalogue card must be produced against the REAL host cat.

The host removed ``cat.mad_hatter`` and exposes ``cat.plugin_manager`` instead.
CATALOG used to call ``cat.mad_hatter.get_plugin().load_settings()``; the
resulting ``AttributeError`` was swallowed by ``_run_hooks``, so the summary was
never cached, no card was created and nothing was logged as an error.

This test boots the real host app/plugin manager (no ``mad_hatter`` stub, no
plugin-manager stub) with the cat_alog plugin discovered through the temporary
symlink installed by ``tests/catalog/conftest.py``. Only the LLM call is
stubbed: the summary agentic workflow returns a fixed summary.
"""

import sys
from types import SimpleNamespace

from langchain_core.documents import Document

from cat.mixins import ContextMixin


class _FakeSummaryWorkflow:
    async def run(self, task=None, llm=None):
        return SimpleNamespace(output="FAKE SUMMARY")


def _hook_function(plugin_manager, name):
    plugin = plugin_manager.plugins["cat_alog"]
    hook = next(h for h in plugin.hooks if h.name == name)
    return hook.function


async def test_legacy_hooks_produce_catalogue_card_with_real_plugin_manager(catalog_plugin_symlinks, cheshire_cat, monkeypatch):
    # Target host contract: `cat.mad_hatter` does not exist. This checkout
    # still carries a compatibility property, so remove it to make a
    # regression to `cat.mad_hatter` fail here exactly as it does on the host.
    monkeypatch.delattr(ContextMixin, "mad_hatter", raising=False)
    assert not hasattr(cheshire_cat, "mad_hatter")

    plugin_manager = cheshire_cat.plugin_manager

    # The plugin must have been resolved through the REAL host plugin manager.
    assert type(plugin_manager).__name__ == "MadHatter"
    assert "cat_alog" in plugin_manager.plugins

    splits_hook = _hook_function(plugin_manager, "before_rabbithole_splits_documents")
    stores_hook = _hook_function(plugin_manager, "before_rabbithole_stores_documents")
    cat_alog = sys.modules[splits_hook.__module__]

    # Only the LLM is stubbed: the summary comes from the fake agentic workflow.
    cheshire_cat.agentic_workflow = _FakeSummaryWorkflow()

    # Isolate the legacy before_rabbithole path; this test targets the
    # plugin-manager resolution, not EffING's phase-machine gating.
    async def _phase_path_inactive(cat=None):
        return False

    monkeypatch.setattr(cat_alog, "_phase_machine_active", _phase_path_inactive)

    docs = [Document(page_content="hello world content", metadata={"source": "real.txt"})]

    split_out = await splits_hook(docs, cheshire_cat)
    assert split_out is docs
    assert cat_alog.CATALOGUES.get(cheshire_cat.agent_key, None, "real.txt") == "FAKE SUMMARY"

    stored_out = await stores_hook(docs, cheshire_cat)
    cards = [doc for doc in stored_out if doc.metadata.get("is_catalogue_card") is True]

    assert len(cards) == 1, (
        "no catalogue card produced through the real plugin manager: "
        f"{[doc.metadata for doc in stored_out]}"
    )
    assert isinstance(cards[0], Document)
    assert "FAKE SUMMARY" in cards[0].page_content
    assert "real.txt" in cards[0].page_content
