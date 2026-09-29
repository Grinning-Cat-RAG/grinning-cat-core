"""The embedder is synchronous (LangChain), and a remote one makes an HTTP call: it runs in a worker thread, never on
the event loop, which serves every request of the instance."""
import threading

from cat.services.factory.embedder import embedding_size
from cat.services.memory.messages import UserMessage


class RecordingEmbedder:
    """The embedder of the Cat, recording whether it was called on the thread of the event loop."""

    def __init__(self, real, loop_thread):
        self.real = real
        self.loop_thread = loop_thread
        self.on_the_loop = []

    def embed_query(self, text):
        self.on_the_loop.append(threading.current_thread() is self.loop_thread)
        return self.real.embed_query(text)

    @property
    def size(self):
        self.on_the_loop.append(threading.current_thread() is self.loop_thread)
        return self.real.size

    def __getattr__(self, name):
        return getattr(self.real, name)


async def recording_embedder(lizard, monkeypatch):
    recording = RecordingEmbedder(await lizard.embedder(), threading.current_thread())

    async def embedder(self):
        return recording

    monkeypatch.setattr(type(lizard), "embedder", embedder)
    return recording


async def test_the_message_is_embedded_off_the_event_loop(lizard, stray, monkeypatch):
    # regression: every user message was embedded on the event loop, blocking the other requests meanwhile
    recording = await recording_embedder(lizard, monkeypatch)
    await stray(UserMessage(text="meow"))
    assert recording.on_the_loop == [False]


async def test_the_memory_endpoints_embed_off_the_event_loop(lizard, secure_client, secure_client_headers, cheshire_cat, monkeypatch):
    recording = await recording_embedder(lizard, monkeypatch)
    response = await secure_client.get("/memory/recall", params={"text": "meow"}, headers=secure_client_headers)
    assert response.status_code == 200
    response = await secure_client.post(
        "/memory/collections/declarative/points", json={"content": "meow", "metadata": {}}, headers=secure_client_headers,
    )
    assert response.status_code == 200
    assert recording.on_the_loop == [False, False]


async def test_the_size_of_the_embedder_is_computed_off_the_event_loop(lizard, monkeypatch):
    recording = await recording_embedder(lizard, monkeypatch)
    assert await embedding_size(recording) == (await lizard.embedder()).size
    assert recording.on_the_loop[0] is False
