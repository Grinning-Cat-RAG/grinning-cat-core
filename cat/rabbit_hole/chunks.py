"""The files the RabbitHole keeps for a source next to the uploaded files, under a ``.rabbithole`` folder.

For the source ``name`` of an agent (``<agent>/``) or of a chat (``<agent>/<chat_id>/``), named by ``sha256(name)``:

- ``<sha>.json``: the chunks produced by the ``parsing_chunking`` phase (documents with their metadata, after the
  ``before_rabbithole_stores_documents`` hook) and the images extracted by the parser: the ``embedding`` phase computes
  the vectors from them, so a change of embedder does not parse the sources again;
- ``<sha>.img<n>``: the bytes of the n-th extracted image;
- ``<sha>.source``: the bytes of a source not kept among the uploaded files (a URL, or a file ingested without storing
  it), so that it can be processed again without downloading it.

The folder lives inside the folder of the agent or of the chat, so removing them removes it too; the file listings
skip it.
"""
import json
import os
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from cat.db.cruds.ingestions import source_id
from cat.utils import is_url

FOLDER = ".rabbithole"


def owner_folder(agent_id: str, chat_id: str | None) -> str:
    """The folder of the uploaded files of the agent (``chat_id=None``) or of a chat."""
    return agent_id if chat_id is None else os.path.join(agent_id, chat_id)


def _is_json(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _name(source: str, suffix: str) -> str:
    return os.path.join(FOLDER, f"{source_id(source)}{suffix}")


def write_chunks(file_manager, agent_id: str, chat_id: str | None, source: str, docs: List[Document], images: List[Dict]):
    """Store the chunks and the extracted images of a source (each image dict carries ``image_bytes``)."""
    folder = owner_folder(agent_id, chat_id)
    entries = []
    for index, image in enumerate(images):
        if not file_manager.write_file(image["image_bytes"], _name(source, f".img{index}"), folder):
            raise IOError(f"Cannot store the image {index} of {source}")
        entries.append({k: v for k, v in image.items() if k not in ("image_bytes", "image_base64") and _is_json(v)})
    payload = json.dumps(
        {"docs": [doc.model_dump(mode="json") for doc in docs], "images": entries}, ensure_ascii=False, default=str,
    )
    if not file_manager.write_file(payload, _name(source, ".json"), folder):
        raise IOError(f"Cannot store the chunks of {source}")


def read_chunks(file_manager, agent_id: str, chat_id: str | None, source: str) -> Tuple[List[Document], List[Dict]]:
    """The chunks and the images stored for a source; ``FileNotFoundError`` when there are none."""
    import base64

    folder = owner_folder(agent_id, chat_id)
    raw = file_manager.read_file(_name(source, ".json"), folder)
    if raw is None:
        raise FileNotFoundError(f"No chunks stored for {source}")
    payload = json.loads(raw)
    images = []
    for index, entry in enumerate(payload.get("images", [])):
        data = file_manager.read_file(_name(source, f".img{index}"), folder)
        if data is None:
            raise FileNotFoundError(f"The image {index} of {source} is missing")
        images.append({**entry, "image_bytes": data, "image_base64": base64.b64encode(data).decode()})
    return [Document(**doc) for doc in payload["docs"]], images


def write_source_bytes(file_manager, agent_id: str, chat_id: str | None, source: str, data: bytes):
    if not file_manager.write_file(data, _name(source, ".source"), owner_folder(agent_id, chat_id)):
        raise IOError(f"Cannot store the bytes of {source}")


def read_source_bytes(file_manager, agent_id: str, chat_id: str | None, source: str) -> bytes | None:
    """The bytes of the source: the uploaded file if stored, else the copy kept by the RabbitHole, else None."""
    folder = owner_folder(agent_id, chat_id)
    if not is_url(source):
        data = file_manager.read_file(source, folder)
        if data is not None:
            return data
    return file_manager.read_file(_name(source, ".source"), folder)


def delete_all(file_manager, agent_id: str, chat_id: str | None, source: str, keep_source: bool = False):
    """Remove the files the RabbitHole keeps for the source, but its bytes when ``keep_source`` (idempotent)."""
    folder = owner_folder(agent_id, chat_id)
    prefix = os.path.join(folder, FOLDER, source_id(source))
    for path in _stored(file_manager, folder, source):
        file_manager.remove_file(path)
    file_manager.remove_file(f"{prefix}.json")
    if not keep_source:
        file_manager.remove_file(f"{prefix}.source")


def _stored(file_manager, folder: str, source: str) -> List[str]:
    """The image files stored for the source (``<sha>.img<n>``), from one listing of the folder of the kept files."""
    prefix = f"{source_id(source)}.img"
    root = os.path.join(file_manager._root_dir, folder, FOLDER)
    return [
        os.path.join(folder, FOLDER, file.name)
        for file in file_manager._list_files(root)
        if file.name.startswith(prefix) and file.name[len(prefix):].isdigit()
    ]
