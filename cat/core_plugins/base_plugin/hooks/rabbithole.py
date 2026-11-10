"""Hooks to modify the RabbitHole's documents ingestion.

Here is a collection of methods to hook into the RabbitHole execution pipeline.

These hooks allow to intercept the uploaded documents at different places before they are saved into memory.

"""
from typing import Dict, List
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

    Fired by the RabbitHole right before embedding the chunks (it depends on the embedder): in the ``embedding``
    phase, so again at every re-embedding of the stored chunks (e.g. a change of the embedder), and in
    ``store_documents``. The no-op default returns the docs unchanged (upstream parity: no token-budget split).
    """
    return docs


@hook(priority=0)
def after_rabbithole_stored_documents(source, stored_points: List[PointStruct], cat) -> None:
    """Hook the Document after is inserted in the vector memory.

    Allows editing and enhancing the list of Document after is inserted in the vector memory.

    Args:
        source: str
            Name of ingested file/url (the requested filename/url when the source could not be resolved).
        stored_points: List[PointStruct]
            List of PointStruct just inserted into the db; empty when the ingestion failed.
        cat: CheshireCat
            Cheshire Cat instance.
    """
    pass


@hook(priority=0)
def rabbithole_ingestion_phases(phases: List, cat) -> List:
    """Hook to add phases to the ingestion of the sources of the agent.

    Each registrant appends its ``cat.rabbit_hole.phases.IngestionPhase`` subclasses (e.g. a phase extracting a graph
    from the stored chunks, depending on ``embedding``) and returns the extended list: classes, not instances (the
    hooks deep-copy what they pass along). The RabbitHole instantiates them without arguments for every run, runs them
    after the phases they depend on, records them in the diary of each source and runs them again when their settings
    (or the ones of a phase they depend on) change. The built-in phases (``parsing_chunking``, ``embedding``) win on a
    duplicate id; a class with an invalid id is left out. No-op default: no phases added.

    Args:
        phases: List of IngestionPhase subclasses
        cat: CheshireCat
    """
    return phases


@hook(priority=0)
def rabbithole_collects_document_images(images: List[Dict], docs: List[Document], cat) -> List[Dict]:
    """Hook to extract the images produced by multimodal parsers from the parsed docs.

    Fired by RabbitHole ``_parse_to_docs`` BEFORE chunking (chunkers may drop the ``image_base64`` metadata that carries
    the payload). First arg ``images`` is the chain carrier: with no plugin the no-op returns it unchanged (upstream
    parity: no images).
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

    Fired by the RabbitHole when the source produced images (from the parse-time collection): in the ``embedding``
    phase, with the images and the bytes of the source read back from the files kept for it (also at every
    re-embedding), and in ``store_documents``.
    First arg ``image_points`` is the chain carrier: with no plugin the no-op
    returns it unchanged (upstream parity: no image points).
    """
    return image_points
