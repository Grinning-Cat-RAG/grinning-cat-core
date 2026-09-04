from typing import Iterator, Mapping, Optional

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document


class MimeTypeBasedParser(BaseBlobParser):
    """Parser that uses ``mime``-types to parse a blob.

    Drop-in replacement for ``langchain_community.document_loaders.parsers.generic.MimeTypeBasedParser``
    (the ``langchain-community`` package is sunset). It dispatches the blob to
    the parser registered for its mime-type in ``handlers``; if the mime-type is
    unknown, ``fallback_parser`` is used when provided, otherwise a
    ``ValueError`` is raised.
    """

    def __init__(
        self,
        handlers: Mapping[str, BaseBlobParser],
        *,
        fallback_parser: Optional[BaseBlobParser] = None,
    ) -> None:
        """Define a parser that uses mime-types to determine how to parse a blob.

        Args:
            handlers: A mapping from mime-types to parsers that take a blob, parse
                      it and return documents.
            fallback_parser: A fallback parser to use if the mime-type is not
                             found in the handlers. If provided, this parser will
                             be used to parse blobs with all mime-types not found
                             in the handlers. If not provided, a ValueError will be
                             raised if the mime-type is not found in the handlers.
        """
        self.handlers = handlers
        self.fallback_parser = fallback_parser

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Load documents from a blob."""
        mimetype = blob.mimetype

        if mimetype is None:
            raise ValueError(f"{blob} does not have a mimetype.")

        if mimetype in self.handlers:
            handler = self.handlers[mimetype]
            yield from handler.lazy_parse(blob)
        else:
            if self.fallback_parser is not None:
                yield from self.fallback_parser.lazy_parse(blob)
            else:
                raise ValueError(f"Unsupported mime type: {mimetype}")