"""``before_rabbithole_stores_documents`` on the efficient engine.

The efficient phase machine emits the hook exactly ONCE per source, at the end
of ``parsing_chunking`` and before the first (empty-vector) store. The returned
list is what gets stored, so a plugin can append or rewrite documents; the
``embedding`` phase rebuilds the points from the stored payload and must NOT
emit the hook again. EffING's own handler is also the single token-budget split
path, so ``finalize_oversized_chunks`` must not be applied on top of it.

These run against the real app harness (Redis db=1 forced by
``tests/conftest.py``); the test hook is appended to the real dispatcher, so
the ordinary-error / veto semantics are the real ones.
"""

import uuid
from io import BytesIO

from langchain_core.documents import Document

from cat.core_plugins.cat_efficient_ingestion import plugin as effing_plugin
from cat.core_plugins.cat_efficient_ingestion import reembed
from cat.core_plugins.cat_efficient_ingestion.registry import IngestionStatus, get_status
from cat.exceptions import CustomHookVetoException
from cat.looking_glass.mad_hatter.decorators.hook import CatHook
from cat.services.factory.embedder import Embeddings
from cat.services.memory.models import VectorMemoryType

HOOK = "before_rabbithole_stores_documents"
SOURCE = "hook_once.txt"
TEXT = ("Alice was beginning to get very tired of sitting by her sister. " * 220).encode()


def _agent_id() -> str:
    return f"effinghook_{uuid.uuid4().hex}"


def _install(pm, fn, priority: int) -> CatHook:
    # reuse an installed plugin id: the dispatcher asks that plugin for its
    # error message when the hook raises (a made-up id would KeyError there).
    hook = CatHook(HOOK, fn, priority, plugin_id="cat_efficient_ingestion")
    pm.hooks.setdefault(HOOK, []).append(hook)
    pm.hooks[HOOK].sort(key=lambda h: h.priority, reverse=True)
    return hook


def _remove(pm, hook: CatHook) -> None:
    try:
        pm.hooks[HOOK].remove(hook)
    except (KeyError, ValueError):
        pass


async def _cat(lizard):
    agent = _agent_id()
    await lizard.create_cheshire_cat(agent)
    return await lizard.get_cheshire_cat(agent)


async def _points(ccat, with_vectors: bool = True):
    records, _ = await ccat.vector_memory_handler.get_all_tenant_points(
        str(VectorMemoryType.DECLARATIVE), with_vectors=with_vectors,
    )
    return list(records or [])


def _text_points(points):
    return [
        p for p in points
        if not (p.payload or {}).get("metadata", {}).get("image")
        and not (p.payload or {}).get("metadata", {}).get("is_catalogue_card")
    ]


async def _ingest(ccat, name: str = SOURCE, data: bytes = TEXT):
    await reembed.EfficientIngestionEngine().ingest_file(
        ccat, BytesIO(data), name, content_type="text/plain", store_file=False,
    )


async def test_appended_document_is_stored_and_the_hook_runs_once(lizard, monkeypatch):
    """An appended document survives the empty-vector pass and is really embedded."""
    ccat = await _cat(lizard)
    calls = []
    add_calls = []
    real_add = ccat.vector_memory_handler.add_points_to_tenant

    async def recording_add(*args, **kwargs):
        points = kwargs.get("points")
        if points is None and len(args) > 1:
            points = args[1]
        add_calls.append(list(points or []))
        return await real_add(*args, **kwargs)

    monkeypatch.setattr(ccat.vector_memory_handler, "add_points_to_tenant", recording_add)

    async def append(docs, cat):
        calls.append(cat.agent_key)
        return docs + [Document(page_content="APPENDED BY HOOK", metadata={"appended": True})]

    hook = _install(ccat.plugin_manager, append, priority=50)
    try:
        await _ingest(ccat)

        status = await get_status(ccat.agent_key, "agent", SOURCE)
        assert status is not None and status["status"] == IngestionStatus.COMPLETED.value, status
        assert len(calls) == 1, f"hook must run exactly once, got {calls}"

        # First write = parsing phase (empty vectors); later = embedding rebuild.
        assert len(add_calls) >= 2, [len(points) for points in add_calls]
        empty_pass = [p for p in add_calls[0] if (p.payload or {}).get("page_content") == "APPENDED BY HOOK"]
        assert len(empty_pass) == 1, [p.payload for p in add_calls[0]]
        assert not empty_pass[0].vector, "the parsing pass must store an empty vector"

        points = _text_points(await _points(ccat))
        appended = [p for p in points if (p.payload or {}).get("page_content") == "APPENDED BY HOOK"]
        assert len(appended) == 1, [p.payload for p in points]
        assert appended[0].vector, "the appended document must carry a real embedding"
    finally:
        _remove(ccat.plugin_manager, hook)


async def test_rewritten_page_content_is_what_gets_stored(lizard):
    """A rewrite is preserved through the embedding rebuild (the hook is not re-run)."""
    ccat = await _cat(lizard)
    calls = []

    async def rewrite(docs, cat):
        calls.append(1)
        for doc in docs:
            doc.page_content = "REWRITTEN " + doc.page_content
        return docs

    hook = _install(ccat.plugin_manager, rewrite, priority=50)
    try:
        await _ingest(ccat)

        status = await get_status(ccat.agent_key, "agent", SOURCE)
        assert status is not None and status["status"] == IngestionStatus.COMPLETED.value, status
        assert len(calls) == 1, f"hook must run exactly once, got {calls}"

        points = _text_points(await _points(ccat))
        assert points, "no text points stored"
        for point in points:
            assert point.payload["page_content"].startswith("REWRITTEN "), point.payload["page_content"][:60]
            assert point.vector, "rewritten text must be embedded after the rebuild"
    finally:
        _remove(ccat.plugin_manager, hook)


async def test_token_budget_split_is_applied_exactly_once(lizard, monkeypatch):
    """EffING's own split runs once: no ``finalize_oversized_chunks`` on top."""
    monkeypatch.setattr(Embeddings, "max_input_tokens", 100)
    ccat = await _cat(lizard)
    calls = []
    real_split = effing_plugin.split_oversized

    def counting_split(docs, embedder):
        calls.append(list(docs))
        return real_split(docs, embedder)

    monkeypatch.setattr(effing_plugin, "split_oversized", counting_split)
    await _ingest(ccat)

    status = await get_status(ccat.agent_key, "agent", SOURCE)
    assert status is not None and status["status"] == IngestionStatus.COMPLETED.value, status
    # The store hook is the single split path. If the chunking step also
    # applied ``finalize_oversized_chunks`` this would be 2.
    assert len(calls) == 1, f"token-budget split applied {len(calls)} times"

    points = _text_points(await _points(ccat))
    embedder = await ccat.embedder()
    expected = real_split(calls[0], embedder)
    assert len(points) == len(expected), (len(points), len(expected))
    assert len(points) > 1, "the oversized input must be split into several chunks"


async def test_ordinary_hook_error_is_logged_and_ingestion_completes(lizard):
    """A plain exception is swallowed by the dispatcher; the source still completes."""
    ccat = await _cat(lizard)
    calls = []

    async def boom(docs, cat):
        calls.append("boom")
        raise RuntimeError("ordinary failure")

    async def after(docs, cat):
        calls.append("after")
        return docs

    boom_hook = _install(ccat.plugin_manager, boom, priority=60)
    after_hook = _install(ccat.plugin_manager, after, priority=50)
    try:
        await _ingest(ccat)

        status = await get_status(ccat.agent_key, "agent", SOURCE)
        assert status is not None and status["status"] == IngestionStatus.COMPLETED.value, status
        assert calls == ["boom", "after"], calls
        assert _text_points(await _points(ccat)), "the source must still be stored"
    finally:
        _remove(ccat.plugin_manager, boom_hook)
        _remove(ccat.plugin_manager, after_hook)


async def test_veto_hook_propagates_and_marks_the_source_error(lizard):
    """The dedicated veto is re-raised by the dispatcher, so it is not swallowed."""
    ccat = await _cat(lizard)

    async def veto(docs, cat):
        raise CustomHookVetoException("refused by the test hook")

    hook = _install(ccat.plugin_manager, veto, priority=60)
    try:
        await _ingest(ccat)

        status = await get_status(ccat.agent_key, "agent", SOURCE)
        assert status is not None, status
        assert status["status"] == IngestionStatus.ERROR.value, status
        assert "refused by the test hook" in (status.get("error") or ""), status
    finally:
        _remove(ccat.plugin_manager, hook)
