"""Tests for the ``is_multimodal_embedder`` detection predicate.

Reembed-multimodal-images plan: pin the predicate that decides
whether a *resolved embedder instance* can embed images. The chunk-reuse re-embed
path (``CheshireCat.embed_stored_sources``) has to know this without a
``RabbitHole`` (so ``_is_multimodal_embedder`` is unusable) and without settings
(an instance, not settings, is resolved). The predicate locks on the instance
itself: it is a ``MultimodalEmbeddings`` subclass OR it exposes ``embed_images``.

These tests deliberately do NOT import the PLUS modal embedders: the point is to
pin the predicate, not the concrete plugin classes.
"""
from cat.services.factory.embedder import MultimodalEmbeddings, is_multimodal_embedder


class FakeMultimodalEmbedder(MultimodalEmbeddings):
    """Concrete ``MultimodalEmbeddings`` subclass (implements the abstract API)."""

    def embed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 4

    def embed_image(self, image):
        return [0.1] * 4

    def embed_images(self, images):
        return [[0.1] * 4 for _ in images]


class FakeTextEmbedder:
    """Plain text-only embedder: exposes ``embed_documents`` but no image API."""

    name = "FakeTextEmbedder"
    size = 4

    def embed_documents(self, texts):
        return [[0.1] * self.size for _ in texts]


class FakeHasattrMultimodal:
    """Not a ``MultimodalEmbeddings`` subclass, but exposes ``embed_images``."""

    name = "FakeHasattrMultimodal"

    def embed_images(self, images):
        return [[0.1] for _ in images]


class FakeNeither:
    """Exposes neither the ABC nor ``embed_images``."""

    name = "FakeNeither"

    def embed_documents(self, texts):
        return [[0.1] for _ in texts]


def test_returns_true_for_multimodal_embeddings_subclass():
    assert is_multimodal_embedder(FakeMultimodalEmbedder())


def test_returns_false_for_plain_text_embedder():
    assert not is_multimodal_embedder(FakeTextEmbedder())


def test_returns_true_for_hasattr_embed_images_without_abc():
    assert is_multimodal_embedder(FakeHasattrMultimodal())


def test_returns_false_when_neither_abc_nor_embed_images():
    assert not is_multimodal_embedder(FakeNeither())