from typing import Iterator

import pymupdf
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document

from cat.log import log


class PyMuPDFParser(BaseBlobParser):
    """Parse a blob from a PDF using the ``pymupdf`` library.

    Drop-in replacement for ``langchain_community.document_loaders.parsers.pdf.PyMuPDFParser``
    (the ``langchain-community`` package is sunset). Supports ``mode="single"``
    (one Document for the whole file, pages joined by ``pages_delimiter``) and
    ``mode="page"`` (one Document per page, with page number in metadata).
    """

    _DEFAULT_PAGES_DELIMITER = "\n\n"

    def __init__(
        self,
        *,
        password: str | None = None,
        mode: str = "single",
        pages_delimiter: str | None = None,
        text_kwargs: dict | None = None,
    ):
        self._password = password
        self._mode = mode
        self._pages_delimiter = pages_delimiter or self._DEFAULT_PAGES_DELIMITER
        self._text_kwargs = text_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        with blob.as_bytes_io() as f:
            data = f.read()

        doc = pymupdf.open(stream=data, filetype="pdf")
        if doc.is_encrypted and self._password is not None:
            doc.authenticate(self._password)

        metadata = {"source": blob.source or ""}
        try:
            metadata |= doc.metadata or {}
        except Exception as e:
            log.warning(f"Could not read PDF metadata: {e}")

        if self._mode == "page":
            for page in doc:
                content = page.get_text(**self._text_kwargs)
                if not content.endswith("\n"):
                    content += "\n"
                yield Document(page_content=content, metadata={**metadata, "page": page.number})
        else:
            pages = [page.get_text(**self._text_kwargs).strip() for page in doc]
            yield Document(page_content=self._pages_delimiter.join(pages), metadata=metadata)