from typing import Iterator

from bs4 import BeautifulSoup
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document


class BS4HTMLParser(BaseBlobParser):
    """Parse a blob of HTML into plain text using BeautifulSoup.

    Drop-in replacement for ``langchain_community.document_loaders.parsers.html.bs4.BS4HTMLParser``
    (the ``langchain-community`` package is sunset). ``features`` selects the
    underlying parser (e.g. ``"lxml"`` or ``"html.parser"``), mirroring the
    original interface; ``get_text_separator`` is used to join extracted text.
    """

    def __init__(self, *, features: str = "html.parser", get_text_separator: str = ""):
        self._features = features
        self._get_text_separator = get_text_separator

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        with blob.as_bytes_io() as f:
            content = f.read()

        soup = BeautifulSoup(content, self._features)
        text = soup.get_text(self._get_text_separator).strip()

        yield Document(page_content=text, metadata={})