from typing import Iterator

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document


class TextParser(BaseBlobParser):
    """Parse a blob of plain text.

    Drop-in replacement for ``langchain_community.document_loaders.parsers.txt.TextParser``
    (the ``langchain-community`` package is sunset; this keeps the RabbitHole
    text ingestion working with only ``langchain-core`` primitives).
    """

    def __init__(self, autodetect_encoding: bool = False):
        self._autodetect_encoding = autodetect_encoding

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        with blob.as_bytes_io() as f:
            data = f.read()

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

        yield Document(page_content=text, metadata={})