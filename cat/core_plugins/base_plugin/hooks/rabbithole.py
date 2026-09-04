"""Hooks to modify the RabbitHole's documents ingestion.

Here is a collection of methods to hook into the RabbitHole execution pipeline.

These hooks allow to intercept the uploaded documents at different places before they are saved into memory.

"""
from typing import Any, Dict, List

from langchain_core.documents import Document

from cat import PointStruct, hook
from cat.core_plugins.base_plugin.parsers import (
    BS4HTMLParser,
    JSONParser,
    LanguageParser,
    PyMuPDFParser,
    TableParser,
    TextParser,
)


@hook(priority=999)
def rabbithole_instantiates_parsers(file_handlers: Dict, cat) -> Dict:
    """Hook the available parsers for ingesting files in the declarative memory.

    Allows replacing or extending existing supported mime types and related parsers to customize the file ingestion.

    Args:
        file_handlers: Dict
            Keys are the supported mime types and values are the related parsers.
        cat: CheshireCat
            Cheshire Cat instance.

    Returns:
        file_handlers: Dict
            Edited dictionary of supported mime types and related parsers.
    """
    file_handlers.update({
        "application/json": JSONParser(),
        "application/pdf": PyMuPDFParser(),
        "text/csv": TableParser(),
        "text/html": BS4HTMLParser(features="lxml"),
        "text/javascript": LanguageParser(language="js"),
        "text/markdown": TextParser(),
        "text/plain": TextParser(),
        "text/x-python": LanguageParser(language="python"),
    })
    return file_handlers


# Hook called just before rabbithole splits file content converted to LangChain Documents.
# Input is whole list of Documents
@hook(priority=0)
def before_rabbithole_splits_documents(docs: List[Document], cat) -> List[Document]:
    """Hook the `Documents` before they are split into chunks.

    Allows editing the uploaded document main Document(s) before the *RabbitHole* recursively splits it in shorter ones.
    Please note that this is a list because parsers can output one or more Document, that are afterward split.

    For instance, the hook allows to change the content or edit/add metadata.

    Args:
        docs: List[Document]
            Langchain `Document`s resulted after parsing the file uploaded in the *RabbitHole*.
        cat: CheshireCat or StrayCat
            Cheshire Cat or Stray Cat instance.

    Returns:
        docs: List[Document]
            Edited Langchain `Document`s.
    """
    return docs


# Hook called when a list of Document is going to be inserted in memory from the rabbit hole.
# Here you can edit/summarize the documents before inserting them in memory
# Should return a list of documents (each is a langchain Document)
@hook(priority=0)
def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    """Hook into the memory insertion pipeline.

    Allows modifying how the list of `Document` is inserted in the vector memory.

    For example, this hook is a good point to summarize the incoming documents and save both original and
    summarized contents.

    Args:
        docs: List[Document]
            List of Langchain `Document` to be edited.
        cat: CheshireCat or StrayCat
            Cheshire Cat or Stray Cat instance.

    Returns:
        docs: List[Document]
            List of edited Langchain documents.
    """
    return docs


@hook(priority=0)
def finalize_oversized_chunks(docs: List[Document], cat) -> List[Document]:
    """Hook to finalize the chunk list so no chunk exceeds the embedder's limit.

    Fired by the core RabbitHole after chunking/splitting (and by the
    efficient_ingestion machine): the ``efficient_ingestion`` plugin implements
    it, resolving the active embedder and splitting any oversized chunk into
    budget-compliant sub-chunks (token-budget split, formerly a MyCAT-only core
    method). The no-op default returns the docs unchanged (upstream parity: no
    token-budget split).
    """
    return docs


@hook(priority=0)
def after_rabbithole_stored_documents(source, stored_points: List[PointStruct], cat) -> None:
    """Hook the Document after is inserted in the vector memory.

    Allows editing and enhancing the list of Document after is inserted in the vector memory.

    Args:
        source: str
            Name of ingested file/url
        stored_points: List[PointStruct]
            List of PointStruct just inserted into the db.
        cat: CheshireCat
            Cheshire Cat instance.
    """
    pass


@hook(priority=0)
def rabbithole_ingestion_start(source, metadata, is_url, cat) -> None:
    """Hook fired at the START of a RabbitHole ingestion, once the source is known.

    Allows plugins to observe that an ingestion is about to begin (e.g. to track
    an ingestion-status lifecycle) before the document is parsed/split/stored.

    Args:
        source: str
            Name of the ingested file/url (or filename for memory uploads).
        metadata: Dict
            Metadata to be stored with each chunk.
        is_url: bool
            Whether the source is a URL.
        cat: CheshireCat or StrayCat
            Cheshire Cat or Stray Cat instance.
    """
    pass


@hook(priority=0)
def rabbithole_ingestion_error(source, error: str, cat) -> None:
    """Hook fired when a RabbitHole ingestion fails.

    Allows plugins to observe ingestion failures (e.g. to record an error state in
    an ingestion-status lifecycle). Fired alongside the existing log/notify.

    Args:
        source: str
            Name of the ingested file/url (or filename if the source was not resolved).
        error: str
            The exception message.
        cat: CheshireCat or StrayCat
            Cheshire Cat or Stray Cat instance.
    """
    pass


@hook(priority=0)
def rabbithole_url_downloading(url, filename, cat) -> None:
    """Hook fired right before a URL is downloaded by the RabbitHole.

    Allows plugins to observe the ``downloading`` sub-state of a URL ingestion.

    Args:
        url: str
            The URL being downloaded.
        filename: str
            The filename (for URLs this is the URL string itself).
        cat: CheshireCat or StrayCat
            Cheshire Cat or Stray Cat instance.
    """
    pass


@hook(priority=0)
def rabbithole_url_download_completed(url, filename, cat) -> None:
    """Hook fired right after a URL is downloaded successfully by the RabbitHole.

    Allows plugins to observe the ``downloaded`` sub-state of a URL ingestion.

    Args:
        url: str
            The URL that was downloaded.
        filename: str
            The filename (for URLs this is the URL string itself).
        cat: CheshireCat or StrayCat
            Cheshire Cat or Stray Cat instance.
    """
    pass


@hook(priority=0)
def rabbithole_ingestion_processing(source, cat) -> None:
    """Hook fired right before the documents of an ingestion are embedded/stored.

    Allows plugins to observe the ``processing`` state of an ingestion lifecycle.

    Args:
        source: str
            Name of the ingested file/url.
        cat: CheshireCat or StrayCat
            Cheshire Cat or Stray Cat instance.
    """
    pass


@hook(priority=0)
def rabbithole_processing_heartbeat_start(source: str, scope: str, interval: float, cat) -> None:
    """Hook to start a background heartbeat for a PROCESSING ingestion row.

    Fired by RabbitHole when a source starts being processed; the
    ``ingestion_status`` plugin uses it to spawn the task that keeps the row
    fresh during long parses. No-op default.
    """
    pass


@hook(priority=0)
def rabbithole_processing_heartbeat_stop(source: str, scope: str, cat) -> None:
    """Hook to stop the heartbeat started by ``..._heartbeat_start``.

    Fired by RabbitHole when the processing ends (success or error); the
    ``ingestion_status`` plugin cancels its heartbeat task here. No-op default.
    """
    pass


@hook(priority=0)
async def run_in_ingestion_executor(result: Any, func, cat) -> Any:
    """Hook to run a heavy ingestion callable off the event loop on a dedicated lane.

    Fired by the core (RabbitHole, CheshireCat) and by plugins that do heavy
    ingestion work (chunking, embedding). The ``efficient_ingestion`` plugin
    provides a dedicated low-concurrency, de-prioritized thread pool and runs
    ``func`` on it. The no-op default returns None: no dedicated lane is active
    and callers fall back to the default executor (upstream parity).
    """
    return None


@hook(priority=0)
def rabbithole_collects_document_images(images: List[Dict], docs: List[Document], cat) -> List[Dict]:
    """Hook to extract the images produced by multimodal parsers from the parsed docs.

    Fired by RabbitHole ``_parse_to_docs`` BEFORE chunking (chunkers may drop
    the ``image_base64`` metadata that carries the payload). The ``multimodal_ingestion``
    plugin walks the parsed documents, returns one entry per extracted image
    (``image_base64`` / ``image_bytes`` / ``image_mime_type``) and strips the
    transient payload from the docs. First arg ``images`` is the chain carrier:
    with no plugin the no-op returns it unchanged (upstream parity: no images).
    """
    return images


@hook(priority=0)
def rabbithole_stores_image_points(
    image_points: List[PointStruct],
    images: List[Dict],
    source: str,
    source_bytes: bytes | None,
    metadata: Dict | None,
    file_hash: str | None,
    chat_id: str | None,
    cat,
) -> List[PointStruct]:
    """Hook to build the image points stored alongside a source's text chunks.

    Fired by RabbitHole ``store_documents`` when the source produced images
    (from the parse-time collection). The ``multimodal_ingestion`` plugin embeds
    them via ``embed_images``, saves them as files (``save_file``) and returns
    the ``PointStruct`` list; the core appends them to the same collection.
    First arg ``image_points`` is the chain carrier: with no plugin the no-op
    returns it unchanged (upstream parity: no image points).
    """
    return image_points
