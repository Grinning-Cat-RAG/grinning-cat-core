"""Tests for the RabbitHole ingestion-status hooks.

Covers the two hooks added to the ingestion pipeline:
- ``rabbithole_ingestion_start``: fired near the START of ``ingest_file`` /
  ``ingest_memory``, once the source is known but before anything is stored.
- ``rabbithole_ingestion_error``: fired inside the existing ``except`` block of
  ``ingest_file`` / ``ingest_memory``, alongside the existing log/notify.

These tests use a fake cat with a recording ``execute_hook`` so the hook calls
are observable without depending on the real plugin manager.
"""
from langchain_core.documents import Document

from cat.rabbit_hole import RabbitHole


class RecordingPluginManager:
    """Records every execute_hook call; passes through the first arg (no-op)."""

    def __init__(self):
        self.calls = []

    async def execute_hook(self, hook_name, *args, caller=None):
        self.calls.append((hook_name, args, caller))
        return args[0] if args else None


class FakeEmbedder:
    size = 4

    def embed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]


class FakeLizard:
    async def embedder(self):
        return FakeEmbedder()


class FakeVectorMemoryHandler:
    def __init__(self):
        self.stored = []

    async def add_points_to_tenant(self, collection_name, points):
        self.stored.append((collection_name, points))


class FakeCat:
    """Minimal cat surface used by ingest_file/ingest_memory."""

    def __init__(self):
        self.agent_key = "agent_test"
        self.plugin_manager = RecordingPluginManager()
        self.lizard = FakeLizard()
        self.vector_memory_handler = FakeVectorMemoryHandler()
        self.saved_files = []

    async def save_file(self, file_bytes, content_type, source, chat_id=None):
        self.saved_files.append((file_bytes, content_type, source, chat_id))


class FakeNotifier:
    def __init__(self):
        self.errors = []

    def has_ws_connection(self):
        return False

    async def send_error(self, message):
        self.errors.append(message)


class FakeStray:
    def __init__(self):
        self.id = "chat_abc"
        self.notifier = FakeNotifier()


def _make_rabbit_hole(cat, stray=None):
    rh = RabbitHole()
    rh.cat = cat
    rh.stray = stray
    return rh


async def _fake_file_to_docs(source, is_url=False):
    async def _file_to_docs(self, file, filename, content_type=None):
        return source, b"file bytes", "text/plain", [], [], is_url

    return _file_to_docs


async def test_ingestion_start_fires_before_store_file(monkeypatch):
    """rabbithole_ingestion_start fires once with correct args, before any store."""
    cat = FakeCat()
    rh = _make_rabbit_hole(cat)

    async def fake_setup(self, _cat):
        self.cat = cat
        self.stray = None

    async def fake_file_to_docs(self, file, filename, content_type=None):
        return "test.txt", b"file bytes", "text/plain", [Document(page_content="hello")], None, False

    monkeypatch.setattr(RabbitHole, "setup", fake_setup)
    monkeypatch.setattr(RabbitHole, "_file_to_docs", fake_file_to_docs)

    metadata = {"author": "alice"}
    await rh.ingest_file(cat=cat, file=b"file bytes", metadata=metadata, filename="test.txt")

    # the start hook fired exactly once, before any store
    start_calls = [c for c in cat.plugin_manager.calls if c[0] == "rabbithole_ingestion_start"]
    assert len(start_calls) == 1
    _, args, caller = start_calls[0]
    assert args == ("test.txt", metadata, False)
    assert caller is cat

    # the start hook fired before the store hook and before any vector write
    call_names = [c[0] for c in cat.plugin_manager.calls]
    assert call_names.index("rabbithole_ingestion_start") < call_names.index("before_rabbithole_stores_documents")
    assert cat.vector_memory_handler.stored  # the file was actually stored


async def test_ingest_start_fires_with_is_url_true(monkeypatch):
    """A URL source fires rabbithole_ingestion_start with is_url=True."""
    cat = FakeCat()
    rh = _make_rabbit_hole(cat)

    async def fake_setup(self, file):
        self.cat = cat
        self.stray = None

    async def fake_file_to_docs(self, file, filename, content_type=None):
        return "http://example.com/doc", b"file bytes", "text/html", [Document(page_content="hello")], None, True

    monkeypatch.setattr(RabbitHole, "setup", fake_setup)
    monkeypatch.setattr(RabbitHole, "_file_to_docs", fake_file_to_docs)

    await rh.ingest_file(cat=cat, file="http://example.com/doc", metadata={})

    start_calls = [c for c in cat.plugin_manager.calls if c[0] == "rabbithole_ingestion_start"]
    assert len(start_calls) == 1
    _, args, caller = start_calls[0]
    assert args[0] == "http://example.com/doc"
    assert args[2] is True
    assert caller is cat


async def test_ingestion_error_fires_and_notify_still_runs(monkeypatch):
    """rabbithole_ingestion_error fires in the except path with the message,
    and the existing error notification still runs."""
    cat = FakeCat()
    stray = FakeStray()
    rh = _make_rabbit_hole(cat, stray)

    async def fake_setup(self, file):
        self.cat = cat
        self.stray = stray

    async def failing_file_to_docs(self, file, filename, content_type=None):
        raise Exception("boom during parse")

    monkeypatch.setattr(RabbitHole, "setup", fake_setup)
    monkeypatch.setattr(RabbitHole, "_file_to_docs", failing_file_to_docs)

    await rh.ingest_file(cat=cat, file=b"file bytes", metadata={}, filename="broken.txt")

    # the error hook fired with the exception message
    error_calls = [c for c in cat.plugin_manager.calls if c[0] == "rabbithole_ingestion_error"]
    assert len(error_calls) == 1
    _, args, caller = error_calls[0]
    assert args[0] == "broken.txt"  # source was empty -> filename used
    assert "boom during parse" in args[1]
    assert caller is stray

    # the existing error notification still ran
    assert stray.notifier.errors == ["Error processing broken.txt: boom during parse"]


async def test_ingestion_error_uses_resolved_source(monkeypatch):
    """If the source was already resolved, the error hook receives it (not the filename)."""
    cat = FakeCat()
    rh = _make_rabbit_hole(cat)

    async def fake_setup(self, file):
        self.cat = cat
        self.stray = None

    async def failing_file_to_docs(self, file, filename, content_type=None):
        # source resolved, but the store step fails afterwards
        return "http://example.com/doc", b"file bytes", "text/html", [Document(page_content="hello")], None, True

    async def failing_store(self, docs, source, file_hash=None, metadata=None, images=None, source_bytes=None):
        raise Exception("store failed")

    monkeypatch.setattr(RabbitHole, "setup", fake_setup)
    monkeypatch.setattr(RabbitHole, "_file_to_docs", failing_file_to_docs)
    monkeypatch.setattr(RabbitHole, "store_documents", failing_store)

    await rh.ingest_file(cat=cat, file="http://example.com/doc", metadata={})

    error_calls = [c for c in cat.plugin_manager.calls if c[0] == "rabbithole_ingestion_error"]
    assert len(error_calls) == 1
    _, args, _ = error_calls[0]
    assert args[0] == "http://example.com/doc"
    assert "store failed" in args[1]


async def test_ingest_memory_fires_start_and_error(monkeypatch):
    """ingest_memory fires both hooks with source=filename, is_url=False, caller=cat."""
    cat = FakeCat()
    rh = _make_rabbit_hole(cat)

    async def fake_setup(self, file):
        self.cat = cat
        self.stray = None

    monkeypatch.setattr(RabbitHole, "setup", fake_setup)

    # happy path: valid memory JSON
    import json
    from io import BytesIO

    memory = {
        "embedder": "FakeEmbedder",
        "collections": {
            "declarative": [
                {"id": "1", "page_content": "hello", "metadata": {}, "vector": [0.1, 0.2, 0.3, 0.4]},
            ]
        },
    }
    file = BytesIO(json.dumps(memory).encode())
    await rh.ingest_memory(cat=cat, file=file, filename="mem.json")

    start_calls = [c for c in cat.plugin_manager.calls if c[0] == "rabbithole_ingestion_start"]
    assert len(start_calls) == 1
    _, args, caller = start_calls[0]
    assert args == ("mem.json", {}, False)
    assert caller is cat

    # error path: embedder mismatch
    cat.plugin_manager.calls.clear()
    bad_memory = {
        "embedder": "OtherEmbedder",
        "collections": {
            "declarative": [
                {"id": "1", "page_content": "hello", "metadata": {}, "vector": [0.1, 0.2, 0.3, 0.4]},
            ]
        },
    }
    file = BytesIO(json.dumps(bad_memory).encode())
    await rh.ingest_memory(cat=cat, file=file, filename="mem.json")

    error_calls = [c for c in cat.plugin_manager.calls if c[0] == "rabbithole_ingestion_error"]
    assert len(error_calls) == 1
    _, args, caller = error_calls[0]
    assert args[0] == "mem.json"
    assert "Embedder mismatch" in args[1]
    assert caller is cat


async def test_raising_hook_does_not_crash_pipeline(monkeypatch):
    """A hook handler that raises must not propagate out of the pipeline.

    The real ``execute_hook`` (mad_hatter) catches per-handler exceptions
    internally and logs them; the pipeline must keep running.
    """
    cat = FakeCat()

    class RaisingPluginManager(RecordingPluginManager):
        async def execute_hook(self, hook_name, *args, caller=None):
            self.calls.append((hook_name, args, caller))
            try:
                if hook_name == "rabbithole_ingestion_start":
                    raise RuntimeError("hook handler exploded")
            except RuntimeError:
                # mimic mad_hatter: handler exceptions are swallowed + logged
                pass
            return args[0] if args else None

    cat.plugin_manager = RaisingPluginManager()
    rh = _make_rabbit_hole(cat)

    async def fake_setup(self, file):
        self.cat = cat
        self.stray = None

    async def fake_file_to_docs(self, file, filename, content_type=None):
        return "test.txt", b"file bytes", "text/plain", [Document(page_content="hello")], None, False

    monkeypatch.setattr(RabbitHole, "setup", fake_setup)
    monkeypatch.setattr(RabbitHole, "_file_to_docs", fake_file_to_docs)

    # must not raise even though the start hook handler raises
    await rh.ingest_file(cat=cat, file=b"file bytes", metadata={}, filename="test.txt")

    assert cat.vector_memory_handler.stored  # the pipeline completed