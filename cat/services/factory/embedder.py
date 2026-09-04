import re
import string
from abc import ABC, abstractmethod
from functools import cached_property
from itertools import combinations
from typing import Type, List, Optional
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from pydantic import ConfigDict
from sklearn.feature_extraction.text import CountVectorizer

from cat.services.factory.models import BaseFactoryConfigModel
from cat.utils import get_nlp_object_name


class Embeddings(LangChainEmbeddings, ABC):
    """Base class for all embedders."""

    # Max input tokens the embedder can accept per input text. ``None`` means
    # "unknown/unlimited" and is the safe default for embedders that do not
    # declare a limit. Concrete embedders override this, e.g. from the model's
    # context window or the provider's documented input cap. Callers (chunkers,
    # rabbit hole) may read it to size chunks that never exceed the embedder.
    max_input_tokens: Optional[int] = None

    def __eq__(self, other):
        if not isinstance(other, Embeddings):
            return False

        return (
            self.name == other.name
            and self.size == other.size
        )

    @property
    def name(self) -> str:
        return get_nlp_object_name(self, "default_embedder")

    @cached_property
    def size(self) -> int:
        """Embedding dimensionality — computed once per instance, then cached."""
        return len(self.embed_query("hello world"))

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate for a text input.

        Default is a conservative ~3 characters per token, so the estimate is
        never an undercount. Embedders that can use a real tokenizer (or read
        the provider's actual token count) should override this for accuracy.
        """
        return max(1, len(text) // 3)


class MultimodalEmbeddings(Embeddings, ABC):
    """Base class for all multimodal embedders."""

    @abstractmethod
    def embed_image(self, image: str | bytes) -> List[float]:
        """Embed an image and returns the embedding vector as a list of floats."""
        pass

    @abstractmethod
    def embed_images(self, images: List[str | bytes]) -> List[List[float]]:
        """Embed a list of images and returns the embedding vectors that are lists of floats."""
        pass


def is_multimodal_embedder(embedder) -> bool:
    """Whether a resolved embedder *instance* can embed images.

    Used by the chunk-reuse re-embed path, where only the instance is available
    (no ``RabbitHole``/settings). True iff the embedder is a
    ``MultimodalEmbeddings`` subclass OR exposes an ``embed_images`` attribute.
    """
    return isinstance(embedder, MultimodalEmbeddings) or hasattr(embedder, "embed_images")


class DumbEmbedder(Embeddings):
    """Default Dumb Embedder.

    This is the default embedder used for testing purposes
    and to replace official embedders when they are not available.

    Notes
    -----
    This class relies on the `CountVectorizer`[1]_ offered by Scikit-learn.
    This embedder uses a naive approach to extract features from a text and build an embedding vector.
    Namely, it looks for pairs of characters in text starting form a vocabulary with all possible pairs of
    printable characters, digits excluded.
    """
    def __init__(self):
        # Get all printable characters numbers excluded and make everything lowercase
        chars = [p.lower() for p in string.printable[10:]]

        # Make the vocabulary with all possible combinations of 2 characters
        voc = sorted(set([f"{k[0]}{k[1]}" for k in combinations(chars, 2)]))

        # Naive embedder that counts occurrences of a couple of characters in text
        self.embedder = CountVectorizer(
            vocabulary=voc, analyzer=lambda s: re.findall("..", s), binary=True
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text and returns the embedding vectors that are lists of floats."""
        return self.embedder.transform(texts).astype(float).todense().tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a string of text and returns the embedding vector as a list of floats."""
        return self.embed_documents([text])[0]


class EmbedderSettings(BaseFactoryConfigModel, ABC):
    # This is related to pydantic, because "model_*" attributes are protected.
    # We deactivate the protection because langchain relies on several "model_*" named attributes
    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def base_class(cls) -> Type[Embeddings]:
        return Embeddings

    @classmethod
    @abstractmethod
    def pyclass(cls) -> Type[Embeddings]:
        pass

    @classmethod
    def is_multimodal(cls) -> bool:
        return False


class EmbedderMultimodalSettings(EmbedderSettings, ABC):
    @classmethod
    def is_multimodal(cls) -> bool:
        return True


class EmbedderDumbConfig(EmbedderSettings):
    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Dumb Embedder",
            "description": "Configuration for default embedder. It encodes the pairs of characters",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type:
        return DumbEmbedder
