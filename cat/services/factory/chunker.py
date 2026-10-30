import asyncio
from abc import ABC, abstractmethod
from typing import Type, List, Iterable, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict

from cat.services.factory.models import BaseFactoryConfigModel


class BaseChunker(ABC):
    """
    Base class to build custom chunkers. This class is used to create custom chunkers that can be used to split text into
    smaller chunks. The chunkers are used to split text into smaller chunks that can be processed by the model.
    MUST be implemented by subclasses.
    """
    def __init__(self):
        self._splitter = None

    @abstractmethod
    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """
        Split the documents into smaller chunks.
        Implementations that perform CPU-bound work should offload it via
        ``asyncio.to_thread`` to avoid blocking the event loop.

        Args:
            documents: the documents to split

        Returns:
            The list of documents after splitting
        """
        pass

    @abstractmethod
    def _get_splitter(self) -> Any:
        pass

    @property
    def splitter(self):
        if not self._splitter:
            self._splitter = self._get_splitter()
        return self._splitter


class EmbeddedBaseChunker(BaseChunker, ABC):
    async def _ensure_embedder(self):
        from cat.looking_glass.bill_the_lizard import BillTheLizard

        if hasattr(self.splitter, "embedder") and not self.splitter.embedder:
            embedder = await BillTheLizard().embedder()
            self.splitter.embedder = embedder


class RecursiveTextChunker(BaseChunker):
    def __init__(self, encoding_name: str, chunk_size: int, chunk_overlap: int):
        super().__init__()

        self._encoding_name = encoding_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\\n\\n", "\n\n", ".\\n", ".\n", "\\n", "\n", " ", ""],
            encoding_name=self._encoding_name,
            keep_separator=True,
            strip_whitespace=True,
            allowed_special={"\n"},  # Explicitly allow the special token ‘\n’
            disallowed_special=(),  # Disallow control for other special tokens
            add_start_index=True,  # Each chunk gets metadata["start_index"] (monotonic, survives overlap)
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """
        Split the documents into chunks, numbering them for stable ordering.

        ONE CALL == ONE LOGICAL SOURCE: ``chunk_index`` is the 0-based ordinal of
        every produced chunk within this call, so it is only meaningful when the
        input documents together represent a single logical source. A pre-existing
        ``chunk_index`` is preserved untouched (``setdefault``), e.g. when a
        caller numbers chunks itself.

        Every produced chunk also carries ``start_index``: the monotonic character
        offset of the chunk in its source text, which survives overlap. The
        splitter is built with ``add_start_index=True``, but LangChain derives the
        offset assuming the overlap is measured in the same unit as the chunk
        length; with a tiktoken-length splitter the overlap is in tokens and the
        derived value can point at the wrong occurrence (or go to -1). We therefore
        re-derive the offset from the source text, searching forward from the
        previous chunk's start so overlaps and repeated text resolve correctly.
        """
        docs = list(documents)

        def _split_and_index() -> List[Document]:
            chunks: List[Document] = []
            for doc in docs:
                source = doc.page_content
                doc_chunks = self.splitter.split_documents([doc])
                cursor = 0
                for chunk in doc_chunks:
                    position = source.find(chunk.page_content, cursor)
                    if position >= 0:
                        chunk.metadata["start_index"] = position
                        cursor = position
                    elif chunk.metadata.get("start_index", -1) < 0:
                        # Never emit a negative offset: keep the best monotonic guess.
                        chunk.metadata["start_index"] = cursor
                    chunks.append(chunk)

            for i, chunk in enumerate(chunks):
                chunk.metadata.setdefault("chunk_index", i)
            return chunks

        return await asyncio.to_thread(_split_and_index)


class ChunkerSettings(BaseFactoryConfigModel, ABC):
    # This is related to pydantic, because "model_*" attributes are protected.
    # We deactivate the protection because langchain relies on several "model_*" named attributes
    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def base_class(cls) -> Type[BaseChunker]:
        return BaseChunker

    @classmethod
    @abstractmethod
    def pyclass(cls) -> Type[BaseChunker]:
        pass


class RecursiveTextChunkerSettings(ChunkerSettings):
    encoding_name: str = "cl100k_base"
    chunk_size: int = 256
    chunk_overlap: int = 64

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Recursive text splitter",
            "description": "Configuration for a recursive text splitter to be used to split text into smaller chunks",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[RecursiveTextChunker]:
        return RecursiveTextChunker
