"""Concurrency-safe filesystem primitives.

The dev stack mounts the plugins folder and the storage folder on a shared filesystem, and several uvicorn workers
(possibly in several replicas) operate on them at the same time. ``shutil.move`` and ``shutil.rmtree`` are neither
atomic nor idempotent: a reader can observe a half-written plugin, and two writers can interleave a delete with a
move.

The helpers here use the standard library only and are safe to call concurrently for DIFFERENT paths. Callers that
operate on the SAME path must serialize through ``cat.db.crud.distributed_lock`` (namespaces ``plugin-fs:<id>``,
``agent-files:<id>``, ``chat-files:<agent>:<chat>``, ``ingestion:<source>``).

Atomicity notes:
- ``os.replace`` is atomic only within the same filesystem. ``atomic_replace_dir`` therefore requires ``tmp_dir`` and
  ``target_dir`` to live on the same filesystem: the caller creates the temporary folder next to the target.
- ``atomic_move_file`` copies into the destination directory first, so it works across filesystems too.
"""
import logging
import os
import shutil
import time
import uuid

log = logging.getLogger(__name__)


def _trash_path(trash_root: str) -> str:
    return os.path.join(trash_root, uuid.uuid4().hex)


def _delete(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        os.remove(path)


def atomic_move_file(src: str, dst: str) -> None:
    """Move ``src`` to ``dst`` without ever exposing a partially written ``dst``.

    The content is copied to ``dst + ".tmp.<uuid>"`` in the destination directory and then renamed with
    ``os.replace``, which is atomic. Works across filesystems. No-op when ``src`` and ``dst`` are the same path or
    when ``src`` does not exist.

    Args:
        src: The file to move.
        dst: The destination path, parent folders included.
    """
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if not os.path.exists(src):
        return

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp_path = f"{dst}.tmp.{uuid.uuid4().hex}"
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    try:
        os.remove(src)
    except FileNotFoundError:
        pass


def atomic_replace_dir(tmp_dir: str, target_dir: str, trash_root: str) -> None:
    """Replace ``target_dir`` with ``tmp_dir`` atomically.

    An existing ``target_dir`` is first moved to ``trash_root/<uuid>``, then ``tmp_dir`` is renamed onto
    ``target_dir`` with ``os.replace``. A concurrent reader therefore sees the previous version, the new one, or —
    briefly, between the two renames — nothing at all. The window is two ``os.replace`` calls and never contains
    partially written content; a symlink-swap layout would remove it entirely (follow-up, not in this patch).

    ``tmp_dir`` and ``target_dir`` MUST be on the same filesystem: ``os.replace`` cannot move a directory across
    devices. The caller is responsible for creating ``tmp_dir`` next to ``target_dir``.

    Args:
        tmp_dir: The fully extracted and validated folder.
        target_dir: The final folder, replaced if it already exists.
        trash_root: Where the previous version is parked before being swept.

    Raises:
        FileNotFoundError: if ``tmp_dir`` does not exist (a caller bug, not a concurrency race).
    """
    if not os.path.isdir(tmp_dir):
        raise FileNotFoundError(f"Temporary directory not found: {tmp_dir}")

    os.makedirs(os.path.dirname(target_dir) or ".", exist_ok=True)
    if os.path.exists(target_dir):
        os.makedirs(trash_root, exist_ok=True)
        os.replace(target_dir, _trash_path(trash_root))

    os.replace(tmp_dir, target_dir)


def trash_then_delete(path: str, trash_root: str) -> None:
    """Remove ``path`` (file or directory) by first renaming it into ``trash_root``.

    The rename is atomic, so a concurrent reader either sees the whole path or nothing. The trashed entry is then
    deleted; leftovers are cleaned by :func:`sweep_trash`. Idempotent: a missing ``path`` is a no-op, and two
    concurrent callers cannot fail each other.

    Args:
        path: The file or folder to remove.
        trash_root: The folder where the entry is parked before deletion.
    """
    if not os.path.exists(path):
        return

    os.makedirs(trash_root, exist_ok=True)
    trashed = _trash_path(trash_root)
    try:
        os.replace(path, trashed)
    except FileNotFoundError:
        # another worker removed it between the check and the rename
        return
    except OSError as e:
        # e.g. a cross-device rename: fall back to an in-place removal, still idempotent
        log.warning(f"Could not move {path} to the trash ({e}), removing it in place")
        try:
            _delete(path)
        except FileNotFoundError:
            # another worker removed it meanwhile
            pass
        return

    try:
        _delete(trashed)
    except FileNotFoundError:
        # a concurrent sweep already removed the trashed entry
        pass


def sweep_trash(trash_root: str, max_age_seconds: int = 86400) -> int:
    """Delete the entries of ``trash_root`` older than ``max_age_seconds``.

    Args:
        trash_root: The folder holding the trashed entries.
        max_age_seconds: The age above which an entry is removed.

    Returns:
        The number of entries removed.
    """
    if not os.path.isdir(trash_root):
        return 0

    now = time.time()
    removed = 0
    for entry in os.scandir(trash_root):
        try:
            if now - entry.stat(follow_symlinks=False).st_mtime < max_age_seconds:
                continue
        except FileNotFoundError:
            continue

        try:
            _delete(entry.path)
            removed += 1
        except FileNotFoundError:
            continue
        except OSError as e:
            log.warning(f"Could not remove trashed entry {entry.path}: {e}")

    return removed
