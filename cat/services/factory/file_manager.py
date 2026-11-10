import os
import tempfile
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Type, List
from pydantic import ConfigDict, BaseModel

from cat import utils
from cat.log import log
from cat.services.factory.models import BaseFactoryConfigModel


class FileResponse(BaseModel):
    path: str
    name: str
    hash: str
    size: int
    last_modified: str


class BaseFileManager(ABC):
    """
    Base class for file storage managers. It defines the interface that all storage managers must implement. It is used
    to upload files and folders to a storage service.
    """
    def __init__(self):
        # ".rabbithole": the files kept by the RabbitHole for the sources (chunks, images, bytes of the URLs)
        self._excluded_dirs = ["__pycache__", ".rabbithole"]
        self._excluded_files = [".gitignore", ".DS_Store", ".gitkeep", ".git", ".dockerignore"]
        self._root_dir = utils.get_file_manager_root_storage_path()

    def __eq__(self, other):
        if not isinstance(other, BaseFileManager):
            return False

        return self.__class__.__name__ == other.__class__.__name__ and self._eq(other)

    @abstractmethod
    def _eq(self, other: "BaseFileManager") -> bool:
        """
        Compares the current object with another object for equality.

        This method is an abstract method and must be implemented by subclasses.
        The equality comparison should be defined based on the specific criteria or attributes relevant to the subclass
        implementation.

        Args:
            other: An object to compare against the current instance.

        Returns:
            bool: The result of the equality comparison.
        """
        pass

    def upload_file(
        self, file_path: str, remote_root_dir: str, remote_filename: str | None = None
    ) -> str | None:
        """
        Upload a single file on the storage, within the directory specified by `remote_root_dir`.

        Args:
            file_path: The path of the file to upload
            remote_root_dir: The directory on the storage where the file will be uploaded
            remote_filename: The name of the file on the storage. If not specified, the file will be uploaded with its
                original name.

        Returns:
            The path of the file on the storage, None if the file has not been uploaded
        """
        remote_root_dir = os.path.join(self._root_dir, remote_root_dir) if remote_root_dir else self._root_dir
        destination_path = os.path.join(
            remote_root_dir, os.path.basename(file_path) if remote_filename is None else remote_filename
        )
        if any([ex_file in destination_path for ex_file in self._excluded_files]):
            return None

        return self._upload_file(file_path, destination_path)

    @abstractmethod
    def _upload_file(self, file_path: str, destination_path: str) -> str:
        pass

    def download_file(self, file_path: str) -> bytes | None:
        """
        Download a single file from the storage and return its content as bytes.

        Args:
            file_path: The path of the file to download from the storage

        Returns:
            The path of the file on the storage, None if the file has not been downloaded
        """
        file_path = os.path.join(self._root_dir, file_path)
        return self._download_file(file_path)

    @abstractmethod
    def _download_file(self, file_path: str) -> bytes | None:
        pass

    def read_file(self, remote_filename: str, remote_root_dir: str | None = None) -> bytes | None:
        """
        Retrieves the content of a specified file from a remote directory as bytes.

        This method constructs the complete path to the file based on the specified remote filename and the provided
        root directory, or the class's default root directory if none is given. It then retrieves the file content in
        binary format.

        Args:
            remote_filename (str): The name of the file to retrieve.
            remote_root_dir (str | None): The root directory of the remote file system where the file is stored. If None, the class's default root directory is used.

        Returns:
            bytes: The binary content of the specified file, or None if the file does not exist.
        """
        remote_root_dir = os.path.join(self._root_dir, remote_root_dir) if remote_root_dir else self._root_dir
        destination_path = os.path.join(remote_root_dir, remote_filename)  # type: ignore[assignment]

        if not self.file_exists(remote_filename, remote_root_dir):
            return None

        return self._read_file(destination_path)  # type: ignore[return-value]

    @abstractmethod
    def _read_file(self, file_path: str) -> bytes:
        pass

    def write_file(self, file_content: str | bytes, remote_filename: str, remote_root_dir: str | None = None) -> bool:
        """
        Writes the content to a specified file to a remote directory as bytes.

        This method constructs the complete path to the file based on the specified remote filename and the provided
        root directory, or the class's default root directory if none is given. It then writes the file content in
        binary format.

        Args:
            file_content (str | bytes): The content to write to the file.
            remote_filename (str): The name of the file to write.
            remote_root_dir (str | None): The root directory of the remote file system where the file is stored. If None, the class's default root directory is used.

        Returns:
             bool: True if the file was written successfully, False otherwise.
        """
        remote_root_dir = os.path.join(self._root_dir, remote_root_dir) if remote_root_dir else self._root_dir
        destination_path = os.path.join(remote_root_dir, remote_filename)  # type: ignore[assignment]

        try:
            self._write_file(file_content, destination_path)  # type: ignore[call-arg]
            return True
        except Exception as e:
            log.error(f"Error while writing file {destination_path}: {e}")
            return False

    @abstractmethod
    def _write_file(self, file_content: str | bytes, file_path: str) -> None:
        pass

    def remove_file(self, file_path: str) -> bool:
        """
        Remove a single file with `file_path` path from the storage.

        Args:
            file_path: The name/path of the file to remove, contained on the storage

        Returns:
            True if the file has been removed, False otherwise
        """
        file_path = os.path.join(self._root_dir, file_path)
        if not self.file_exists(file_path):
            return False

        return self._remove_file(file_path)

    @abstractmethod
    def _remove_file(self, file_path: str) -> bool:
        pass

    def remove_folder(self, remote_root_dir: str) -> bool:
        """
        Remove the entire `remote_root_dir` directory from the storage. If not specified, the entire storage will be
        removed.

        Returns:
            True if the storage has been removed, False otherwise
        """
        remote_root_dir = os.path.join(self._root_dir, remote_root_dir) if remote_root_dir else self._root_dir
        return self._remove_folder(remote_root_dir)

    @abstractmethod
    def _remove_folder(self, remote_root_dir: str) -> bool:
        pass

    def list_files(self, remote_root_dir: str) -> List[FileResponse]:
        remote_root_dir = os.path.join(self._root_dir, remote_root_dir) if remote_root_dir else self._root_dir

        files = self._list_files(remote_root_dir)

        def excluded(file: FileResponse) -> bool:
            parts = set(Path(file.path).parts[:-1]) | {file.name}
            return file.name in self._excluded_files or bool(parts & set(self._excluded_dirs))

        return [file for file in files if not excluded(file)]

    @abstractmethod
    def _list_files(self, remote_root_dir: str) -> List[FileResponse]:
        pass

    def clone_folder(self, remote_root_dir_from: str, remote_root_dir_to: str) -> List[str]:
        """
        Clone the entire `remote_root_dir_from` directory on the storage to the `remote_root_dir_to`.

        Args:
            remote_root_dir_from: The directory on the storage where the files are contained
            remote_root_dir_to: The directory on the storage where the files will be cloned

        Returns:
            List of the paths of the files on the storage
        """
        remote_root_dir_from = (
            os.path.join(self._root_dir, remote_root_dir_from) if remote_root_dir_from else self._root_dir
        )
        remote_root_dir_to = (
            os.path.join(self._root_dir, remote_root_dir_to) if remote_root_dir_to else self._root_dir
        )
        return self._clone_folder(remote_root_dir_from, remote_root_dir_to)

    @abstractmethod
    def _clone_folder(self, remote_root_dir_from: str, remote_root_dir_to: str) -> List[str]:
        pass

    def upload_folder(self, local_dir: str, remote_root_dir: str) -> List[str]:
        """
        Upload a directory with all the contained files on the storage, within the directory specified by
        `remote_root_dir`.

        Args:
            local_dir: The path of the directory locally, containing the files to upload to the storage
            remote_root_dir: The directory on the storage where the files will be uploaded

        Returns:
            List of the paths of the files on the storage
        """
        local_dir = os.path.join(self._root_dir, local_dir)

        # the subfolders keep their place (e.g. the chats of an agent, the files kept by the RabbitHole)
        return [  # type: ignore[return-value]
            self.upload_file(os.path.join(root, file), remote_root_dir, os.path.relpath(os.path.join(root, file), local_dir))
            for root, _, files in os.walk(local_dir)
            for file in files
        ]

    def download_folder(self, local_dir: str, remote_root_dir: str) -> List[str]:
        """
        Download the directory specified by `remote_root_dir` with all the contained files from the storage to
        `local_dir`.

        Args:
            local_dir: The path where the directory will be downloaded locally
            remote_root_dir: The directory on the storage where the files are contained

        Returns:
            List of the paths of the files locally
        """
        local_dir = os.path.join(self._root_dir, local_dir)
        remote_dir = os.path.join(self._root_dir, remote_root_dir) if remote_root_dir else self._root_dir

        result = []
        # Downloads every file from storage, subfolders included (the chats of an agent, the files kept by the
        # RabbitHole), keeping its place under the folder; skipping excluded files
        for file in self._walk_files(remote_dir):
            relative = os.path.relpath(file.path, remote_dir) if file.path.startswith(remote_dir + os.sep) else file.name
            if file.name in self._excluded_files or "__pycache__" in Path(relative).parts:
                continue
            local_path = os.path.join(local_dir, relative)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            result.append(self._download_file_to_local(file.path, local_path))

        return result

    def _walk_files(self, remote_root_dir: str) -> List[FileResponse]:
        """Every file under ``remote_root_dir``, subfolders included. Default: the listing, recursive on the object
        stores listing by prefix; a storage listing one folder only overrides it."""
        return self._list_files(remote_root_dir)

    @abstractmethod
    def _download_file_to_local(self, file_path: str, local_path: str) -> str:
        pass

    def transfer(self, file_manager_from: "BaseFileManager", remote_root_dir: str) -> bool:
        """
        Transfer files from the file manager specified in the `file_manager_from` to the current one.

        Args:
            file_manager_from: The file manager to transfer the files from
            remote_root_dir: The directory on the storage where the files are contained
        """
        try:
            with tempfile.TemporaryDirectory() as tmp_folder_name:
                # try to download the files from the old file manager to the `tmp_folder_name`
                file_manager_from.download_folder(tmp_folder_name, remote_root_dir)

                # now, try to upload the files to the new storage
                self.upload_folder(tmp_folder_name, remote_root_dir)
                file_manager_from.remove_folder(remote_root_dir)

                return True
        except Exception as e:
            log.error(f"Error while transferring files from the old file manager to the new one: {e}")
            return False

    def file_exists(self, filename: str, remote_root_dir: str | None = None) -> bool:
        """
        Check if a file exists in the storage.

        Args:
            filename: The name or path of the file to check
            remote_root_dir: The directory on the storage where the file should be contained

        Returns:
            True if the file exists, False otherwise
        """
        if remote_root_dir is None:
            remote_root_dir = os.path.dirname(filename)
            filename = os.path.basename(filename)
        elif os.path.dirname(filename):
            # a file in a subfolder of remote_root_dir: listed in that subfolder
            remote_root_dir = os.path.join(remote_root_dir, os.path.dirname(filename))
            filename = os.path.basename(filename)

        if self._root_dir not in remote_root_dir:
            remote_root_dir = os.path.join(self._root_dir, remote_root_dir)  # type: ignore[assignment]

        return filename in [file.name for file in self._list_files(remote_root_dir)]  # type: ignore[return-value]


class DummyFileManager(BaseFileManager):
    def _download_file(self, file_path: str) -> bytes | None:
        pass

    def _upload_file(self, file_path: str, destination_path: str) -> str:
        return ""

    def _download_file_to_local(self, file_path: str, local_path: str) -> str:
        return ""

    def _remove_file(self, file_path: str) -> bool:
        return False

    def _remove_folder(self, remote_root_dir: str) -> bool:
        return False

    def _clone_folder(self, remote_root_dir_from: str, remote_root_dir_to: str) -> List[str]:
        return []

    def _list_files(self, remote_root_dir: str) -> List[FileResponse]:
        return []

    def _read_file(self, destination_path: str) -> bytes:
        pass  # type: ignore[empty-body]
    
    def _write_file(self, file_content: str | bytes, file_path: str) -> None:
        pass

    def _eq(self, other: "DummyFileManager") -> bool:  # type: ignore[override]
        return self.__class__.__name__ == other.__class__.__name__


class FileManagerConfig(BaseFactoryConfigModel, ABC):
    @classmethod
    def base_class(cls) -> Type[BaseFileManager]:
        return BaseFileManager

    @classmethod
    @abstractmethod
    def pyclass(cls) -> Type[BaseFileManager]:
        pass


class DummyFileManagerConfig(FileManagerConfig):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def pyclass(cls) -> Type[DummyFileManager]:
        return DummyFileManager
