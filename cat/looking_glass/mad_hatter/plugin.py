import asyncio
import fcntl
import glob
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from inspect import getmembers, isabstract, isawaitable
from pathlib import Path
from typing import Dict, List, Tuple, Any, Type
from pydantic import BaseModel, ValidationError

from cat.db.cruds import plugins as crud_plugins
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.log import log
from cat.looking_glass.mad_hatter.decorators.experimental.form import CatForm
from cat.looking_glass.mad_hatter.decorators.experimental.mcp_client import CatMcpClient
from cat.looking_glass.mad_hatter.decorators.endpoint import CatEndpoint
from cat.looking_glass.mad_hatter.decorators.hook import CatHook
from cat.looking_glass.mad_hatter.decorators.plugin_decorator import CatPluginDecorator
from cat.looking_glass.mad_hatter.decorators.tool import CatTool
from cat.looking_glass.mad_hatter.procedures import CatProcedure
from cat.looking_glass.models import PluginSettingsModel, PluginManifest
from cat.utils import inspect_calling_agent, get_base_path, to_camel_case


class Plugin:
    def __init__(self, plugin_path: str):
        if not os.path.exists(plugin_path) or not os.path.isdir(plugin_path):
            raise Exception(
                f"{plugin_path} does not exist or is not a folder. Cannot create Plugin."
            )

        # where the plugin is on disk
        self._path: str = plugin_path

        # search for .py files in folder
        py_files_path = os.path.join(self._path, "**/*.py")
        self._py_files = glob.glob(py_files_path, recursive=True)

        if len(self._py_files) == 0:
            raise Exception(
                f"{plugin_path} does not contain any python files. Cannot create Plugin."
            )

        # plugin id is just the folder name
        self._id: str = os.path.basename(os.path.normpath(plugin_path))

        # plugin manifest (name, description, thumb, etc.)
        self._manifest = self._load_manifest()

        self._hooks: List[CatHook] = []  # list of plugin hooks
        self._procedures: List[CatProcedure] = []  # list of plugin procedures (tools + forms)
        self._endpoints: List[CatEndpoint] = []  # list of plugin endpoints

        # list of @plugin decorated functions overriding default plugin behaviour
        self._plugin_overrides: Dict[str, CatPluginDecorator] = {}

        # plugin starts deactivated
        self._active = False

    async def activate(self, agent_id: str):
        # install plugin requirements on activation (non-blocking: uses asyncio subprocess)
        await self._install_requirements()

        # load hooks and tools
        self._load_decorated_functions()

        await self.activate_settings(agent_id)
        self._active = True

        # run custom activation from @plugin
        if "activated" in self.overrides:
            self.overrides["activated"].function(self)

    async def activate_settings(self, agent_id: str) -> bool:
        # by default, plugin settings are saved inside the Redis database
        setting = await crud_plugins.get_setting(agent_id, self._id)

        # store the new settings incrementally, without losing the values of the configurations still supported
        # the new setting coming from the model to be activated
        new_setting = self._get_settings_from_model()

        if not setting and not new_setting:
            # no settings to migrate
            return True

        if setting is not None and new_setting and setting == new_setting:
            # settings are the same, no need to migrate
            return True

        try:
            # store the new settings incrementally, without losing the values of the configurations still supported
            finalized_setting = {k: setting.get(k, v) for k, v in new_setting.items()} if setting else new_setting  # type: ignore[union-attr]

            # Validate the settings
            self.settings_model().model_validate(finalized_setting)
        except:
            finalized_setting = new_setting

        if setting == finalized_setting:
            # no settings to migrate
            return True

        # try to create the new incremental settings into the Redis database
        await crud_plugins.set_setting(agent_id, self._id, finalized_setting)  # type: ignore[arg-type]
        log.info(f"Plugin {self._id} for agent '{agent_id}', settings migrated from {setting} to {finalized_setting}")

        return True

    async def deactivate(self, agent_id: str):
        # run custom deactivation from @plugin
        if "deactivated" in self.overrides:
            self.overrides["deactivated"].function(self)

        # Remove the imported modules
        for py_file in self._py_files:
            py_filename = py_file.replace("/", ".").replace(".py", "")

            # If the module is imported it is removed
            if py_filename not in sys.modules:
                continue
            log.debug(f"Remove module {py_filename}")
            sys.modules.pop(py_filename)

        self._unload_decorated_functions()
        self._active = False
        await self.deactivate_settings(agent_id)

    async def deactivate_settings(self, agent_id: str):
        # remove the settings
        await crud_plugins.delete_setting(agent_id, self._id)

    # get plugin settings JSON schema
    def settings_schema(self) -> Dict[str, Any]:
        # is "settings_schema" hook defined in the plugin?
        if "settings_schema" in self.overrides:
            return self.overrides["settings_schema"].function()

        # otherwise, if the "settings_schema" is not defined but "settings_model" is, it gets the schema from the model
        if "settings_model" in self.overrides:
            return self.overrides["settings_model"].function().model_json_schema()

        # default schema (empty)
        return PluginSettingsModel.model_json_schema()

    def _get_py_filename_dotted_notation(self, py_file: str) -> str:
        base_path_dotted_notation = get_base_path().replace("/", ".")
        return (
            py_file.replace(".py", "")
            .replace("/", ".")
            .replace(base_path_dotted_notation, "cat.")
        )

    # get plugin settings Pydantic model
    def settings_model(self) -> Type[BaseModel]:
        # is "settings_model" hook defined in the plugin?
        if "settings_model" in self.overrides:
            return self.overrides["settings_model"].function()

        # is "settings_schema" hook defined in the plugin?
        if "settings_schema" in self.overrides:
            schema = self.overrides["settings_schema"].function()
            pydantic_class = schema.get("__pydantic_model__", schema.get("title"))
            if not pydantic_class:
                return PluginSettingsModel

            if isinstance(pydantic_class, BaseModel):
                return pydantic_class  # type: ignore[return-value]

            if not isinstance(pydantic_class, str):
                log.error(f"Invalid settings class {pydantic_class} from `settings_schema` hook in plugin {self.id}")
                return PluginSettingsModel

            # if the pydantic class is a string, try to load it from the plugin
            # find where a Pydantic model with name pydantic_class is defined within the folder self._path
            for py_file in self._py_files:
                py_filename = self._get_py_filename_dotted_notation(py_file)
                try:
                    module = importlib.import_module(py_filename)
                    if hasattr(module, pydantic_class):
                        return getattr(module, pydantic_class)
                except Exception:
                    pass

            # if the pydantic class is not found, return the default PluginSettingsModel
            log.error(f"Unable to find settings model {pydantic_class} from `settings_schema` hook in plugin {self.id}")

        # default schema (empty)
        return PluginSettingsModel

    # load plugin settings
    async def load_settings(self, agent_id: str | None = None) -> Dict[str, Any]:
        if agent_id is None:
            try:
                calling_agent = inspect_calling_agent()
                agent_id = calling_agent.agent_key
            except Exception as e:
                log.error(f"Error loading plugin {self._id} settings. Getting default settings: {e}")
                log.warning(self.plugin_specific_error_message())
                agent_id = DEFAULT_SYSTEM_KEY

        # is "load_settings" hook defined in the plugin?
        if "load_settings" in self.overrides:
            # the override is called plain: when it is an async function the
            # coroutine must be awaited, otherwise it would be returned as-is
            # (plugin settings that never resolve) — so the plugin can rely on
            # the async official ``cat.db.crud`` API instead of sync Redis clients
            result = self.overrides["load_settings"].function(self._id, agent_id)
            if isawaitable(result):
                result = await result
            return result

        # by default, plugin settings are saved inside the Redis database
        settings = await crud_plugins.get_setting(agent_id, self._id) or self._get_settings_from_model()  # type: ignore[arg]
        if settings is None:
            log.debug(f"Agent {agent_id} - Plugin {self._id} settings model is not stored or has no default values, returning empty settings")
            return {}

        try:
            # Validate the settings
            self.settings_model().model_validate(settings)
            return settings
        except Exception as e:
            log.error(f"Agent {agent_id} - Unable to load plugin {self._id} settings: {e}")
            log.warning(self.plugin_specific_error_message())
            raise e

    # save plugin settings
    async def save_settings(self, settings: Dict, agent_id: str) -> Dict[str, Any]:
        # is "settings_save" hook defined in the plugin?
        if "save_settings" in self.overrides:
            # same plain-call handling as load_settings: await only when the
            # override is actually async
            result = self.overrides["save_settings"].function(self._id, settings, agent_id)
            if isawaitable(result):
                result = await result
            return result

        try:
            # overwrite settings over old ones
            # write settings into the Redis database
            return await crud_plugins.update_setting(agent_id, self._id, settings)
        except Exception as e:
            log.error(f"Unable to save plugin {self._id} settings: {e}")
            log.warning(self.plugin_specific_error_message())
            return {}

    def missing_dependencies(self, available_plugins: List[str]) -> List[str]:
        if not self.manifest.dependencies:
            return []

        missing_dependencies = [
            dependency for dependency in self.manifest.dependencies if dependency not in available_plugins
        ]
        if missing_dependencies:
            log.error(f"Plugin {self.id} is missing dependencies: {missing_dependencies}")

        return missing_dependencies

    def _get_settings_from_model(self) -> Dict | None:
        try:
            model = self.settings_model()
            # if some settings have no default value this will raise a ValidationError
            settings = model().model_dump()

            return settings
        except ValidationError:
            return None

    def _load_manifest(self) -> PluginManifest:
        plugin_json_metadata_file_name = "plugin.json"
        plugin_json_metadata_file_path = os.path.join(
            self._path, plugin_json_metadata_file_name
        )
        json_file_data = {}

        if os.path.isfile(plugin_json_metadata_file_path):
            try:
                json_file = open(plugin_json_metadata_file_path)
                json_file_data = json.load(json_file)
                json_file.close()
            except Exception:
                log.error(
                    f"Loading plugin {self._path} metadata, defaulting to generated values"
                )

        json_file_data["id"] = self._id
        json_file_data["name"] = json_file_data.get("name", to_camel_case(self._id))
        return PluginManifest(**json_file_data)

    @staticmethod
    def _hash_file(path: str) -> str:
        """SHA256 hash of file contents."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    async def _install_requirements(self):
        req_file = os.path.join(self.path, "requirements.txt")
        if not os.path.exists(req_file):
            return

        current_hash = self._hash_file(req_file)
        hash_file = os.path.join(self.path, ".requirements_hash")

        # Fast path: requirements haven't changed since last install
        if os.path.exists(hash_file):
            with open(hash_file, "r") as hf:
                if hf.read().strip() == current_hash:
                    return

        loop = asyncio.get_running_loop()
        lock_file = os.path.join(self.path, ".install.lock")

        # Try to become the installer via a non-blocking flock.
        # Only ONE worker succeeds — the others poll for the hash to appear.
        lf = open(lock_file, "w")
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lf.close()
            # Another worker is installing → poll asynchronously
            for _ in range(60):  # up to 6 seconds
                await asyncio.sleep(0.1)
                if os.path.exists(hash_file):
                    with open(hash_file, "r") as hf:
                        if hf.read().strip() == current_hash:
                            return
            # Poll timed out — fallback to blocking install
            await loop.run_in_executor(None, self._install_requirements_sync, req_file, current_hash)
            return

        # We hold the lock → run the install in a thread
        try:
            await loop.run_in_executor(None, self._install_requirements_sync, req_file, current_hash, lf)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()

    def _install_requirements_sync(
        self, req_file: str, current_hash: str, lock_fh=None
    ):
        """Synchronous install with cross-process serialization via fcntl.flock.

        If *lock_fh* is given the lock is already held by the caller;
        otherwise a blocking flock is acquired.
        """
        hash_file = os.path.join(self.path, ".requirements_hash")

        if lock_fh is None:
            lock_file = os.path.join(self.path, ".install.lock")
            lf = open(lock_file, "w")
            fcntl.flock(lf, fcntl.LOCK_EX)
            own_lock = True
        else:
            lf = lock_fh
            own_lock = False

        try:
            # Double-check: another worker may have installed while we waited
            if os.path.exists(hash_file):
                with open(hash_file, "r") as hf:
                    if hf.read().strip() == current_hash:
                        return

            start_time = time.time()
            log.info(f"Installing requirements for plugin {self.id}")

            install_cmd = ["uv", "pip", "install", "--no-cache", "--no-upgrade", "-r", req_file]
            try:
                proc = subprocess.Popen(
                    install_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if proc.stdout:
                    for line in proc.stdout:
                        log.debug(line.strip())
                proc.wait()
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode or 1, install_cmd)

                duration = time.time() - start_time
                log.info(f"Installation of requirements for {self.id} completed in {duration:.2f}s")

                # Persist hash so future calls take the fast path
                with open(hash_file, "w") as hf:
                    hf.write(current_hash)
            except subprocess.CalledProcessError as e:
                log.error(f"Error while installing plugin {self.id} requirements: {e}")
                log.info(f"Uninstalling requirements for: {self.id}")
                uninstall_cmd = ["uv", "pip", "uninstall", "-r", req_file]
                try:
                    proc = subprocess.Popen(
                        uninstall_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    if proc.stdout:
                        for line in proc.stdout:
                            log.debug(line.strip())
                    proc.wait()
                    if proc.returncode != 0:
                        raise subprocess.CalledProcessError(proc.returncode or 1, uninstall_cmd)
                except Exception as e_:
                    log.error(f"Error while uninstalling plugin {self.id} requirements: {e_}")
                raise Exception(f"Error while installing plugin {self.id} requirements")
        finally:
            if own_lock:
                fcntl.flock(lf, fcntl.LOCK_UN)
                lf.close()
            # Clean __pycache__ directories (cross-platform approach)
            for pycache in Path("/app").rglob("__pycache__"):
                shutil.rmtree(pycache, ignore_errors=True)

    # lists of hooks and tools
    def _load_decorated_functions(self):
        hooks = []
        procedures = []
        endpoints = []
        plugin_overrides = []

        load_done = True
        for py_file in self._py_files:
            py_filename = self._get_py_filename_dotted_notation(py_file)

            log.debug(f"Import module {py_filename}")

            # save a reference to decorated functions
            try:
                plugin_module = importlib.import_module(py_filename)
                importlib.reload(plugin_module)

                hooks += getmembers(plugin_module, lambda obj: isinstance(obj, CatHook))
                procedures += (
                        getmembers(plugin_module, lambda obj: isinstance(obj, CatTool)) +
                        getmembers(plugin_module, lambda obj: isinstance(obj, CatForm) and not isabstract(obj) and obj.autopilot) +
                        getmembers(plugin_module, lambda obj: isinstance(obj, CatMcpClient) and not isabstract(obj))
                )
                endpoints += getmembers(plugin_module, lambda obj: isinstance(obj, CatEndpoint))
                plugin_overrides += getmembers(plugin_module, lambda obj: isinstance(obj, CatPluginDecorator))
            except Exception as e:
                log.error(
                    f"Error in {py_filename}: {str(e)}. Unable to load plugin {self._id}"
                )
                log.warning(self.plugin_specific_error_message())
                load_done = False
                break

        if not load_done:
            self._unload_decorated_functions()
            raise Exception(f"Error loading plugin {self._id}")

        # clean and enrich instances
        self._hooks = list(map(self._clean_hook, hooks))
        self._procedures = list(map(self._clean_procedure, procedures))
        self._endpoints = list(map(self._clean_endpoint, endpoints))

        self._plugin_overrides = {override.name: override for _, override in plugin_overrides}

    def _unload_decorated_functions(self):
        self._hooks = []
        self._procedures = []
        self._endpoints = []
        self._plugin_overrides = {}

    def plugin_specific_error_message(self) -> str:
        name = self.manifest.name
        url = self.manifest.plugin_url
        if url:
            return f"To resolve any problem related to {name} plugin, contact the creator using github issue at the link {url}"
        return f"Error in {name} plugin, contact the creator"

    def _clean_hook(self, hook: Tuple[str, CatHook]) -> CatHook:
        # getmembers returns a tuple
        _, h = hook
        h.plugin_id = self._id
        return h

    def _clean_procedure(self, procedure: Tuple[str, CatProcedure]) -> CatProcedure:
        # getmembers returns a tuple
        _, p = procedure
        p.plugin_id = self._id
        return p

    def _clean_endpoint(self, endpoint: Tuple[str, CatEndpoint]) -> CatEndpoint:
        # getmembers returns a tuple
        _, e = endpoint
        e.plugin_id = self._id
        return e

    @property
    def path(self) -> str:
        return self._path

    @property
    def id(self) -> str:
        return self._id

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    @property
    def active(self) -> bool:
        return self._active

    @property
    def hooks(self) -> List[CatHook]:
        return self._hooks

    @property
    def procedures(self) -> List[CatProcedure]:
        return self._procedures

    @property
    def tools(self) -> List[CatTool]:
        return [p for p in self._procedures if isinstance(p, CatTool)]

    @property
    def forms(self) -> List[CatForm]:
        return [p for p in self._procedures if isinstance(p, CatForm)]

    @property
    def mcp_clients(self) -> List[CatMcpClient]:
        return [p for p in self._procedures if isinstance(p, CatMcpClient)]

    @property
    def endpoints(self) -> List[CatEndpoint]:
        return self._endpoints

    @property
    def overrides(self) -> Dict[str, CatPluginDecorator]:
        return self._plugin_overrides
