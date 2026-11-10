"""The files the RabbitHole keeps for a source (``cat.rabbit_hole.chunks``) and their place among the agent's files."""
from typing import List

import pytest
from langchain_core.documents import Document

from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
from cat.rabbit_hole import chunks
from cat.services.factory.file_manager import DummyFileManager, FileResponse


@pytest.fixture
def files():
    return LocalFileManager()


def test_chunks_and_images_round_trip(files):
    docs = [Document(page_content="uno", metadata={"page": 1}), Document(page_content="due ü", metadata={})]
    images = [{"image_bytes": b"\x89PNG1", "image_base64": "ignored", "image_mime_type": "image/png"},
              {"image_bytes": b"\xff\xd8JPG", "image_mime_type": "image/jpeg"}]

    chunks.write_chunks(files, "a1", "c1", "doc.pdf", docs, images)
    read_docs, read_images = chunks.read_chunks(files, "a1", "c1", "doc.pdf")

    assert [(d.page_content, d.metadata) for d in read_docs] == [("uno", {"page": 1}), ("due ü", {})]
    assert [(i["image_bytes"], i["image_mime_type"]) for i in read_images] == [
        (b"\x89PNG1", "image/png"), (b"\xff\xd8JPG", "image/jpeg"),
    ]
    assert read_images[0]["image_base64"] == "iVBORzE="


def test_missing_chunks_or_images_are_reported(files):
    with pytest.raises(FileNotFoundError):
        chunks.read_chunks(files, "a1", None, "doc.pdf")

    chunks.write_chunks(files, "a1", None, "doc.pdf", [Document(page_content="x")], [{"image_bytes": b"i"}])
    files.remove_file("a1/.rabbithole/" + chunks.source_id("doc.pdf") + ".img0")
    with pytest.raises(FileNotFoundError, match="image 0"):
        chunks.read_chunks(files, "a1", None, "doc.pdf")


def test_a_file_manager_that_cannot_write_fails_loudly():
    with pytest.raises(IOError):
        chunks.write_chunks(DummyFileManagerFailing(), "a1", None, "doc.pdf", [], [])
    with pytest.raises(IOError):
        chunks.write_chunks(DummyFileManagerFailing(), "a1", None, "doc.pdf", [], [{"image_bytes": b"i"}])
    with pytest.raises(IOError):
        chunks.write_source_bytes(DummyFileManagerFailing(), "a1", None, "https://x.org", b"b")


class DummyFileManagerFailing(DummyFileManager):
    def write_file(self, file_content, remote_filename, remote_root_dir=None) -> bool:
        return False


def test_the_bytes_of_a_source_come_from_the_uploaded_file_or_from_the_kept_copy(files):
    assert chunks.read_source_bytes(files, "a1", None, "doc.pdf") is None

    chunks.write_source_bytes(files, "a1", None, "https://x.org/page", b"html")
    files.write_file(b"pdf", "doc.pdf", "a1")
    chunks.write_source_bytes(files, "a1", None, "doc.pdf", b"copy")

    assert chunks.read_source_bytes(files, "a1", None, "https://x.org/page") == b"html"
    assert chunks.read_source_bytes(files, "a1", None, "doc.pdf") == b"pdf"  # the uploaded file wins
    files.remove_file("a1/doc.pdf")
    assert chunks.read_source_bytes(files, "a1", None, "doc.pdf") == b"copy"


def test_deleting_removes_every_kept_file_or_all_but_the_bytes(files):
    chunks.write_chunks(files, "a1", None, "doc.pdf", [Document(page_content="x")], [{"image_bytes": b"1"}, {"image_bytes": b"2"}])
    chunks.write_source_bytes(files, "a1", None, "doc.pdf", b"copy")

    chunks.delete_all(files, "a1", None, "doc.pdf", keep_source=True)
    assert chunks.read_source_bytes(files, "a1", None, "doc.pdf") == b"copy"
    with pytest.raises(FileNotFoundError):
        chunks.read_chunks(files, "a1", None, "doc.pdf")
    assert not files.file_exists(".rabbithole/" + chunks.source_id("doc.pdf") + ".img1", "a1")

    chunks.delete_all(files, "a1", None, "doc.pdf")
    chunks.delete_all(files, "a1", None, "doc.pdf")  # idempotent
    assert chunks.read_source_bytes(files, "a1", None, "doc.pdf") is None


def test_the_kept_files_are_not_listed_and_hidden_files_hide_only_themselves(files):
    files.write_file(b"x", "doc.pdf", "a1")
    files.write_file(b"x", ".gitignore", "a1")
    chunks.write_chunks(files, "a1", None, "doc.pdf", [Document(page_content="x")], [])

    assert [f.name for f in files.list_files("a1")] == ["doc.pdf"]


class ListingFileManager(DummyFileManager):
    """A file manager of a storage listing every file under a prefix (as an object store does)."""

    def __init__(self, paths: List[str]):
        super().__init__()
        self.paths = paths

    def _list_files(self, remote_root_dir: str) -> List[FileResponse]:
        root = remote_root_dir.rstrip("/") + "/"
        return [
            FileResponse(path=p, name=p.rsplit("/", 1)[-1], hash="", size=0, last_modified="")
            for p in self.paths if p.startswith(root)
        ]


def test_a_listing_storage_hides_the_kept_files_and_finds_files_in_subfolders():
    manager = ListingFileManager([])
    root = manager._root_dir
    manager.paths = [f"{root}/a1/doc.pdf", f"{root}/a1/.rabbithole/abc.json", f"{root}/a1/__pycache__/x.pyc"]

    assert [f.name for f in manager.list_files("a1")] == ["doc.pdf"]
    assert manager.file_exists(".rabbithole/abc.json", "a1")
    assert manager.file_exists("doc.pdf", "a1")
    assert not manager.file_exists(".rabbithole/other.json", "a1")
    assert manager.file_exists(f"{root}/a1/doc.pdf")


def test_moving_the_files_of_an_agent_keeps_its_subfolders(files, tmp_path):
    files.write_file(b"pdf", "doc.pdf", "a1")
    files.write_file(b"chat", "c1/notes.txt", "a1")
    chunks.write_chunks(files, "a1", None, "doc.pdf", [Document(page_content="x")], [{"image_bytes": b"i"}])

    downloaded = files.download_folder(str(tmp_path), "a1")
    files.upload_folder(str(tmp_path), "a2")

    assert len(downloaded) == 4
    assert files.read_file("doc.pdf", "a2") == b"pdf"
    assert files.read_file("notes.txt", "a2/c1") == b"chat"
    docs, images = chunks.read_chunks(files, "a2", None, "doc.pdf")
    assert docs[0].page_content == "x" and images[0]["image_bytes"] == b"i"


def test_images_of_a_source_are_found_without_probing_each_one(files):
    chunks.write_chunks(files, "a1", None, "doc.pdf", [Document(page_content="x")], [{"image_bytes": b"1"}] * 3)
    chunks.write_chunks(files, "a1", None, "other.pdf", [Document(page_content="x")], [{"image_bytes": b"2"}])

    chunks.delete_all(files, "a1", None, "doc.pdf")

    assert chunks.read_chunks(files, "a1", None, "other.pdf")[1][0]["image_bytes"] == b"2"
    assert sorted(f.name for f in files._list_files(files._root_dir + "/a1/.rabbithole")) == sorted([
        chunks.source_id("other.pdf") + ".json", chunks.source_id("other.pdf") + ".img0",
    ])
