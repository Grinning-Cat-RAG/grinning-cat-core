import hashlib
import os
import shutil
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from stat import S_ISREG
from typing import List, Tuple

from cat.log import log
from cat.services.factory.file_manager import BaseFileManager, FileResponse

#: SHA-256 of the files listed by this instance, by (path, size, modification time, inode): a file is read again only
#: when it changes
_HASHES: "OrderedDict[Tuple[str, int, int, int], str]" = OrderedDict()
_HASHES_LOCK = threading.Lock()
MAX_CACHED_HASHES = 100_000


def _file_hash(path: str, stat: os.stat_result, chunk_size: int = 8192) -> str:
    key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns, stat.st_ino)
    with _HASHES_LOCK:
        if (cached := _HASHES.get(key)) is not None:
            _HASHES.move_to_end(key)
            return cached

    sha256 = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    digest = sha256.hexdigest()

    with _HASHES_LOCK:
        _HASHES[key] = digest
        while len(_HASHES) > MAX_CACHED_HASHES:
            _HASHES.popitem(last=False)
    return digest


class LocalFileManager(BaseFileManager):
    def _eq(self, other: "LocalFileManager") -> bool:
        return self.__class__.__name__ == other.__class__.__name__

    def _download_file(self, file_path: str) -> bytes | None:
        try:
            if not os.path.exists(file_path):
                return None

            with open(file_path, "rb") as f:
                return f.read()
        except Exception as e:
            log.error(f"Error while downloading file {file_path} from storage: {e}")
            return None

    def _upload_file(self, file_path: str, destination_path: str) -> str:
        if file_path != destination_path:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            # move the file from file_path to destination_path
            shutil.move(file_path, destination_path)
        return destination_path

    def _download_file_to_local(self, file_path: str, local_path: str) -> str:
        if file_path != local_path:
            # move the file from origin_path to local_path
            shutil.move(file_path, local_path)
        return local_path

    def _remove_file(self, file_path: str) -> bool:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                log.error(f"Error while removing file {file_path} from storage: {e}")
                return False
        return True

    def _remove_folder(self, remote_root_dir: str) -> bool:
        if os.path.exists(remote_root_dir) and os.path.isdir(remote_root_dir):
            try:
                shutil.rmtree(remote_root_dir)
            except Exception as e:
                log.error(f"Error while removing storage: {e}")
                return False
        return True

    def _list_files(self, remote_root_dir: str) -> List[FileResponse]:
        if not os.path.exists(remote_root_dir):
            return []

        # List only the files in the remote_root_dir (no subfolders)
        files = []
        for file in os.listdir(remote_root_dir):
            path = os.path.join(remote_root_dir, file)
            try:
                stat = os.stat(path)
            except FileNotFoundError:  # removed meanwhile
                continue
            if not S_ISREG(stat.st_mode):
                continue
            files.append(FileResponse(
                path=path,
                name=file,
                hash=_file_hash(path, stat),
                size=int(stat.st_size),
                last_modified=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
            ))
        return files

    def file_exists(self, filename: str, remote_root_dir: str | None = None) -> bool:
        # the same paths as the base class, without listing (and hashing) the whole folder
        if remote_root_dir is None:
            remote_root_dir = os.path.dirname(filename)
            filename = os.path.basename(filename)

        if self._root_dir not in remote_root_dir:
            remote_root_dir = os.path.join(self._root_dir, remote_root_dir)

        return os.path.isfile(os.path.join(remote_root_dir, filename))

    def _clone_folder(self, remote_root_dir_from: str, remote_root_dir_to: str) -> List[str]:
        cloned_files = []
        for root, _, files in os.walk(remote_root_dir_from):
            for file in files:
                relative_path = os.path.relpath(root, remote_root_dir_from)
                destination_dir = os.path.join(remote_root_dir_to, relative_path)
                os.makedirs(destination_dir, exist_ok=True)
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_dir, file)
                shutil.copy2(source_file, destination_file)
                cloned_files.append(destination_file)
        return cloned_files

    def _read_file(self, file_path: str) -> bytes:
        with open(file_path, "rb") as f:
            return f.read()

    def _write_file(self, file_content: str | bytes, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        mode, encoding = ("wb", None) if isinstance(file_content, bytes) else ("w", "utf-8")
        with open(file_path, mode, encoding=encoding) as f:
            f.write(file_content)
