"""The local file manager reads a file only when needed: listing a folder or looking for a file never reads the content
of the files that did not change."""
import hashlib
import os
from pathlib import Path

import pytest

from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager


@pytest.fixture
def manager(tmp_path):
    fm = LocalFileManager()
    fm._root_dir = str(tmp_path)
    return fm


@pytest.fixture
def reads(monkeypatch):
    """Paths of the files opened for reading."""
    opened = []
    real_open = Path.open

    def counting_open(self, mode="r", *args, **kwargs):
        if "r" in mode:
            opened.append(os.path.basename(str(self)))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    return opened


def test_the_files_are_hashed_once(manager, reads):
    # regression: every listing read every file of the folder, to hash it
    for i in range(3):
        manager.write_file(b"x" * 1000, f"f{i}.bin", "agent")
    first = manager.list_files("agent")
    assert sorted(reads) == ["f0.bin", "f1.bin", "f2.bin"]
    assert {f.name: f.hash for f in first}["f0.bin"] == hashlib.sha256(b"x" * 1000).hexdigest()

    reads.clear()
    assert manager.list_files("agent") == first
    assert reads == []


def test_a_changed_file_is_hashed_again(manager):
    manager.write_file(b"aaaa", "f.bin", "agent")
    before, = manager.list_files("agent")
    path = os.path.join(manager._root_dir, "agent", "f.bin")
    stat = os.stat(path)
    manager.write_file(b"bbbb", "f.bin", "agent")  # same size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    after, = manager.list_files("agent")
    assert after.hash == hashlib.sha256(b"bbbb").hexdigest() != before.hash


def test_looking_for_a_file_never_reads_the_files(manager, reads):
    # regression: removing (or reading) a file hashed every file of its folder
    for i in range(3):
        manager.write_file(b"y" * 1000, f"f{i}.bin", "agent")
    reads.clear()
    assert manager.file_exists("f1.bin", "agent")
    assert manager.file_exists(os.path.join(manager._root_dir, "agent", "f1.bin"))
    assert not manager.file_exists("missing.bin", "agent")
    assert not manager.file_exists("agent", None), "a folder is not a file"
    assert manager.remove_file("agent/f1.bin")
    assert reads == []
    assert manager.read_file("f2.bin", "agent") == b"y" * 1000
    assert manager.read_file("f1.bin", "agent") is None


def test_the_cache_of_the_hashes_is_bounded(manager, monkeypatch):
    from cat.core_plugins.base_plugin.file_managers import custom

    monkeypatch.setattr(custom, "MAX_CACHED_HASHES", 1)
    monkeypatch.setattr(custom, "_HASHES", custom.OrderedDict())
    for i in range(3):
        manager.write_file(b"z", f"f{i}.bin", "agent")
    manager.list_files("agent")
    assert len(custom._HASHES) == 1


def test_a_file_removed_while_listing_is_skipped(manager, monkeypatch):
    manager.write_file(b"a", "kept.bin", "agent")
    manager.write_file(b"b", "gone.bin", "agent")
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if str(path).endswith("gone.bin"):
            raise FileNotFoundError(path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    assert [f.name for f in manager.list_files("agent")] == ["kept.bin"]
