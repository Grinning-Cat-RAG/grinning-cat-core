import mimetypes
import os
import shutil
import uuid
from pathlib import Path
from slugify import slugify

from cat.log import log
from cat.services.atomic_fs import atomic_replace_dir
from cat.services.python_security import ast_scan, SecurityError
from cat.utils import get_allowed_plugins_mime_types


class PluginExtractor:
    def __init__(self, path: str):
        allowed_mime_types = get_allowed_plugins_mime_types()

        content_type = mimetypes.guess_type(path)[0]
        if content_type not in allowed_mime_types:
            raise Exception(
                f"Invalid package extension. Valid extensions are: {allowed_mime_types}"
            )

        self._extension = "tar" if content_type == "application/x-tar" else "zip"
        self._path = path

        # this will be plugin folder name (its id for the mad hatter)
        self._id = self.create_plugin_id()

    @property
    def path(self):
        return self._path

    @property
    def id(self):
        return self._id

    @property
    def extension(self):
        return self._extension

    def create_plugin_id(self):
        file_name = os.path.basename(self._path)
        file_name_no_extension = os.path.splitext(file_name)[0]
        return slugify(file_name_no_extension, separator="_")

    @staticmethod
    def _is_safe_plugin(folder_path: str) -> bool:
        """
        Check all Python files in the plugin folder for safety.

        Args:
            folder_path (str): Path to the plugin folder.

        Returns:
            bool: True if all Python files are safe, False if any file contains malicious code.
        """
        def is_safe_python_file() -> bool:
            try:
                ast_scan(file_path)  # AST scan + SecurityError if malicious code is detected
                return True
            except SecurityError as e:
                log.error(f"Malicious code detected: {e}")
                return False
            except Exception as e:  # SyntaxError, IOError, ecc.
                log.debug(f"Could not scan {file_path}: {e}")
                return False

        for root, _, files in os.walk(folder_path):
            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                if not is_safe_python_file():
                    return False
        return True

    def extract(self, to: str) -> str:
        # extract into a temporary folder on the SAME filesystem as the target: the final rename must be atomic,
        # and os.replace cannot move a directory across devices
        tmp_root = os.path.join(to, ".tmp")
        os.makedirs(tmp_root, exist_ok=True)
        tmp_folder_name = os.path.join(tmp_root, uuid.uuid4().hex)
        os.makedirs(tmp_folder_name)

        # extract into tmp directory
        shutil.unpack_archive(self._path, tmp_folder_name, self._extension)
        # what was extracted?
        contents = os.listdir(tmp_folder_name)

        tmp_folder_to = (
            os.path.join(tmp_folder_name, contents[0])
            if len(contents) == 1 and os.path.isdir(os.path.join(tmp_folder_name, contents[0]))
            else tmp_folder_name
        )

        try:
            # check if plugin is safe
            if not self._is_safe_plugin(tmp_folder_to):
                raise ValueError("Plugin contains unsafe Python files")

            # proceed with installation if checks pass
            # swap the plugin folder in with an atomic rename: a concurrent reader sees the old version or the new
            # one, never a missing or half-written folder
            folder_to = os.path.join(to, self._id)
            atomic_replace_dir(tmp_folder_to, folder_to, os.path.join(to, ".trash"))

            return folder_to
        except Exception as e:
            log.error(f"Error during plugin extraction: {e}")
            raise e
        finally:
            # Cleanup temporary directory
            if os.path.exists(tmp_folder_name):
                shutil.rmtree(tmp_folder_name, ignore_errors=True)

            # Remove zip after extraction
            if os.path.exists(self._path):
                os.remove(self._path)
