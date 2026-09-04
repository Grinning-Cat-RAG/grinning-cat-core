from typing import Iterator

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document


class LanguageParser(BaseBlobParser):
    """Parse a blob of source code, tagging it with its language.

    Drop-in replacement for ``langchain_community.document_loaders.parsers.language.language_parser.LanguageParser``
    (the ``langchain-community`` package is sunset). The original used
    ``tree-sitter`` to structure code; this lightweight replacement avoids the
    extra dependency: it returns the whole file as a single Document carrying
    the language in metadata. Further splitting is handled by the chunker, so
    indexing behaviour is preserved.
    """

    def __init__(self, language: str, parser_threshold: int = 0):
        self._language = language
        self._parser_threshold = parser_threshold

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        with blob.as_bytes_io() as f:
            data = f.read()

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

        yield Document(page_content=text, metadata={"language": self._language})