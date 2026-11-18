"""The atomic filesystem helpers never expose a partially written path and are idempotent."""
import os

import pytest

from cat.services.atomic_fs import (
    atomic_move_file,
    atomic_replace_dir,
    sweep_trash,
    trash_then_delete,
)


def _write(path: str, content: str = "x") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_atomic_move_file_moves_the_content(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "nested" / "dst.txt"
    _write(str(src), "hello")

    atomic_move_file(str(src), str(dst))

    assert dst.read_text() == "hello"
    assert not src.exists()
    # no temporary file is left next to the destination
    assert [p.name for p in dst.parent.iterdir()] == ["dst.txt"]


def test_atomic_move_file_is_a_noop_for_the_same_path(tmp_path):
    src = tmp_path / "src.txt"
    _write(str(src), "hello")

    atomic_move_file(str(src), str(src))

    assert src.read_text() == "hello"


def test_atomic_move_file_does_not_raise_on_a_missing_source(tmp_path):
    atomic_move_file(str(tmp_path / "missing.txt"), str(tmp_path / "dst.txt"))

    assert not (tmp_path / "dst.txt").exists()


def test_atomic_move_file_leaves_the_destination_untouched_on_error(tmp_path, monkeypatch):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    _write(str(src), "new")
    _write(str(dst), "old")

    def failing_replace(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        atomic_move_file(str(src), str(dst))

    assert dst.read_text() == "old"
    assert src.read_text() == "new"
    assert [p.name for p in tmp_path.iterdir() if ".tmp." in p.name] == []


def test_atomic_replace_dir_replaces_an_existing_target(tmp_path):
    tmp_dir = tmp_path / "plugins" / ".tmp" / "abc"
    target = tmp_path / "plugins" / "my_plugin"
    trash = tmp_path / "plugins" / ".trash"
    _write(str(tmp_dir / "new.py"), "new")
    _write(str(target / "old.py"), "old")

    atomic_replace_dir(str(tmp_dir), str(target), str(trash))

    assert (target / "new.py").read_text() == "new"
    assert not (target / "old.py").exists()
    assert not tmp_dir.exists()
    # the previous version is parked in the trash, not deleted in place
    assert len(list(trash.iterdir())) == 1


def test_atomic_replace_dir_creates_a_missing_target(tmp_path):
    tmp_dir = tmp_path / "tmp_plugin"
    target = tmp_path / "plugins" / "my_plugin"
    _write(str(tmp_dir / "new.py"), "new")

    atomic_replace_dir(str(tmp_dir), str(target), str(tmp_path / "trash"))

    assert (target / "new.py").read_text() == "new"
    assert not (tmp_path / "trash").exists()


def test_atomic_replace_dir_requires_the_temporary_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        atomic_replace_dir(str(tmp_path / "missing"), str(tmp_path / "target"), str(tmp_path / "trash"))


def test_trash_then_delete_removes_a_folder_and_is_idempotent(tmp_path):
    folder = tmp_path / "storage" / "agent"
    trash = tmp_path / "storage" / ".trash"
    _write(str(folder / "f.txt"), "x")

    trash_then_delete(str(folder), str(trash))
    assert not folder.exists()
    assert list(trash.iterdir()) == []

    # a second call (another worker, or a retry) is a no-op and does not raise
    trash_then_delete(str(folder), str(trash))
    assert not folder.exists()


def test_trash_then_delete_removes_a_file(tmp_path):
    file = tmp_path / "f.txt"
    _write(str(file), "x")

    trash_then_delete(str(file), str(tmp_path / ".trash"))

    assert not file.exists()


def test_trash_then_delete_does_not_raise_on_a_missing_path(tmp_path):
    trash_then_delete(str(tmp_path / "missing"), str(tmp_path / ".trash"))

    assert not (tmp_path / ".trash").exists()


def test_sweep_trash_removes_only_the_old_entries(tmp_path):
    trash = tmp_path / ".trash"
    old_dir = trash / "old_dir"
    old_file = trash / "old_file"
    fresh = trash / "fresh"
    _write(str(old_dir / "f.txt"), "x")
    _write(str(old_file), "x")
    _write(str(fresh), "x")
    for path in (old_dir, old_file):
        os.utime(path, (0, 0))

    assert sweep_trash(str(trash), max_age_seconds=3600) == 2
    assert [p.name for p in trash.iterdir()] == ["fresh"]


def test_sweep_trash_on_a_missing_root_returns_zero(tmp_path):
    assert sweep_trash(str(tmp_path / "missing")) == 0
