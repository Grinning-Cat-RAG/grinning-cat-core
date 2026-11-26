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
import re
import tempfile
import time
from collections import defaultdict
from inspect import getmembers, isabstract, isawaitable
from pathlib import Path
from typing import Dict, List, Tuple, Any, Type
import tomli
import tomlkit
from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
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
from cat.utils import (
    inspect_calling_agent, get_base_path, get_core_plugins_path, get_plugins_path, get_project_path, to_camel_case,
)

# Records whether a plugin's uv.lock is the plugin's own ("shipped") or was compiled by the core
# ("generated:<sha256 of the lock>"): see Plugin._shipped_lock
LOCK_ORIGIN_FILE = ".uv.lock.origin"


class Plugin:
    def __init__(self, plugin_path: str):
        if not os.path.exists(plugin_path) or not os.path.isdir(plugin_path):
            raise Exception(
                f"{plugin_path} does not exist or is not a folder. Cannot create Plugin."
            )

        # where the plugin is on disk
        self._path: str = plugin_path

        # search for .py files in folder, skipping the plugin's own tests: they import only under pytest,
        # and a single failed import would prevent the whole plugin from loading
        py_files_path = os.path.join(glob.escape(self._path), "**/*.py")
        self._py_files = [
            py_file for py_file in glob.glob(py_files_path, recursive=True) if not self._is_test_file(py_file)
        ]

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

    def _is_test_file(self, py_file: str) -> bool:
        """`conftest.py` files and anything under a `tests` folder inside the plugin (not above it)."""
        parts = Path(py_file).relative_to(self._path).parts
        return parts[-1] == "conftest.py" or "tests" in parts[:-1]

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
            # Validate the settings: they are read as the model reads them (e.g. the ones stored as they were sent)
            return self.normalized_settings(settings)
        except Exception as e:
            log.error(f"Agent {agent_id} - Unable to load plugin {self._id} settings: {e}")
            log.warning(self.plugin_specific_error_message())
            raise e

    def normalized_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        The settings as the settings model reads them (e.g. ``"false"`` is ``False``, an enum is its value), field by
        field: the keys that the model does not declare are kept as they are. ValidationError if they are not valid.
        """
        validated = self.settings_model().model_validate(settings).model_dump(mode="json")
        return {**settings, **{key: validated[key] for key in settings if key in validated}}

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
    def _hash_files(paths: List[str]) -> str:
        """SHA256 hash of the concatenated contents of the given files, in order."""
        h = hashlib.sha256()
        for path in paths:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _read_pyproject_dependencies(pyproject_file: str) -> List[str]:
        with open(pyproject_file, "rb") as f:
            data = tomli.load(f)
        return data.get("project", {}).get("dependencies", [])

    @staticmethod
    def _root_lock_path() -> str:
        return os.path.join(get_project_path(), "uv.lock")

    @classmethod
    def _root_lock_constraints(cls) -> List[str]:
        """PEP 508 constraints pinning every package in the root project's
        uv.lock, so a plugin's dependency resolution can never silently
        upgrade or downgrade a system library the core app depends on."""
        root_lock = cls._root_lock_path()
        if not os.path.exists(root_lock):
            return []
        with open(root_lock, "rb") as f:
            data = tomli.load(f)
        return [
            f"{pkg['name']}=={pkg['version']}"
            for pkg in data.get("package", [])
            if "version" in pkg
        ]

    def _other_plugins_constraints(self, pinned: set[str]) -> List[str]:
        """PEP 508 constraints pinning the packages locked by the other
        installed plugins (their uv.lock), except the ones in *pinned* (the
        root project's). The plugins share one virtual environment: a plugin
        can never replace the version of a library another plugin installed —
        it fails to resolve instead, at its installation.

        Only the locks of the plugins installed in this environment count: the
        ones whose `.requirements_hash` matches their pyproject.toml and the
        current root uv.lock (not a lock shipped in a plugin's archive, nor one
        compiled against an older root uv.lock)."""
        own_lock = os.path.abspath(os.path.join(self.path, "uv.lock"))
        locks = sorted(
            glob.glob(os.path.join(get_core_plugins_path(), "*", "uv.lock"))
            + glob.glob(os.path.join(get_plugins_path(), "*", "uv.lock"))
        )
        constraints: Dict[str, str] = {}
        for lock in locks:
            if os.path.abspath(lock) == own_lock or not self._is_installed_here(os.path.dirname(lock)):
                continue
            try:
                with open(lock, "rb") as f:
                    packages = tomli.load(f).get("package", [])
            except Exception as e:
                log.warning(f"Ignoring unreadable {lock}: {e}")
                continue
            for pkg in packages:
                # the plugin project itself (virtual / editable source) is not a library
                if "version" not in pkg or {"virtual", "editable"} & set(pkg.get("source", {})):
                    continue
                if pkg["name"] in pinned or pkg["name"] in constraints:
                    continue
                constraints[pkg["name"]] = f"{pkg['name']}=={pkg['version']}"
        return list(constraints.values())

    @classmethod
    def _is_installed_here(cls, plugin_dir: str) -> bool:
        """The plugin's dependencies were installed against the current root uv.lock (and, when the plugin ships its own
        uv.lock, from that very lock)."""
        if not os.path.exists(os.path.join(plugin_dir, "pyproject.toml")):
            return False
        return cls._should_exit_from_installing_requirements(
            os.path.join(plugin_dir, ".requirements_hash"), cls._hash_files(cls._requirements_hash_inputs(plugin_dir)),
        )

    @classmethod
    def _requirements_hash_inputs(cls, plugin_dir: str) -> List[str]:
        """The files whose change requires installing the plugin's dependencies again: its pyproject.toml, the uv.lock it
        ships (if any) and the root uv.lock."""
        inputs = [os.path.join(plugin_dir, "pyproject.toml")]
        shipped_lock = cls._shipped_lock(plugin_dir)
        if shipped_lock is not None:
            inputs.append(shipped_lock)
        if os.path.exists(cls._root_lock_path()):
            inputs.append(cls._root_lock_path())
        return inputs

    @classmethod
    def _shipped_lock(cls, plugin_dir: str) -> str | None:
        """The uv.lock the plugin ships, or None when it ships none.

        The core also writes a uv.lock in the plugin folder, the one it compiles for a plugin shipping none, and records
        it in `.uv.lock.origin` as `generated:<sha256 of the lock>`. A uv.lock is the plugin's own when that record says
        `shipped`, or when the lock no longer matches the recorded digest (e.g. a new version of the plugin brought its
        own). With no record at all, a uv.lock next to a `.requirements_hash` was compiled by a core older than the
        record, which always replaced the plugin's uv.lock with its own: it is not the plugin's."""
        lock = os.path.join(plugin_dir, "uv.lock")
        if not os.path.exists(lock):
            return None
        origin_file = os.path.join(plugin_dir, LOCK_ORIGIN_FILE)
        if os.path.exists(origin_file):
            with open(origin_file, "r") as f:
                origin = f.read().strip()
            if origin.startswith("generated:"):
                return None if origin == f"generated:{cls._hash_files([lock])}" else lock
            return lock
        if os.path.exists(os.path.join(plugin_dir, ".requirements_hash")):
            return None
        return lock

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    @staticmethod
    def _marker_applies(marker: str | None) -> bool:
        """Whether a PEP 508 marker holds in this environment (an unreadable marker is assumed to)."""
        if not marker:
            return True
        try:
            return Marker(marker).evaluate({**default_environment(), "extra": ""})
        except Exception:
            return True

    @classmethod
    def _lock_root_package(cls, lock_data: Dict) -> Dict | None:
        """The plugin project itself in a uv.lock (its virtual or editable source)."""
        for pkg in lock_data.get("package", []):
            source = pkg.get("source", {})
            if source.get("virtual") == "." or source.get("editable") == ".":
                return pkg
        return None

    @classmethod
    def _stale_lock_reasons(cls, lock_data: Dict, project: Dict) -> List[str]:
        """Why a shipped uv.lock does not match the plugin's pyproject.toml (empty when it does).

        The dependencies the lock was resolved for (`requires-dist` of the plugin project) are compared with
        `[project].dependencies`: same name, extras and version specifier, and the same outcome of their markers in this
        environment (uv rewrites markers, e.g. `python_version` as `python_full_version`, so their text may differ)."""
        root = cls._lock_root_package(lock_data)
        if root is None:
            return ["the lock does not contain the plugin project (no package with a virtual or editable source)"]
        if cls._normalize_name(root.get("name", "")) != cls._normalize_name(project.get("name", "")):
            return [f"the lock is for project {root.get('name')!r}, the pyproject.toml for {project.get('name')!r}"]

        def key(name, extras, specifier, marker):
            return cls._normalize_name(name), tuple(sorted(extras)), str(SpecifierSet(specifier or "")), cls._marker_applies(marker)

        locked = sorted(
            key(d["name"], d.get("extras", []), d.get("specifier"), d.get("marker"))
            for d in root.get("metadata", {}).get("requires-dist", [])
        )
        declared = []
        for dependency in project.get("dependencies", []):
            req = Requirement(dependency)
            declared.append(key(req.name, req.extras, str(req.specifier), str(req.marker) if req.marker else None))
        declared.sort()
        reasons = [f"declared in pyproject.toml, not locked: {d[0]}{d[2]}" for d in declared if d not in locked]
        reasons += [f"locked, no longer declared in pyproject.toml: {d[0]}{d[2]}" for d in locked if d not in declared]
        return reasons

    @classmethod
    def _locked_runtime_packages(cls, lock_data: Dict) -> Dict[str, set]:
        """`{name: {versions}}` of the packages a uv.lock installs in this environment for the plugin project: its
        dependencies and theirs (with the requested extras), skipping the ones whose markers do not hold here and every
        dependency group (dev included)."""
        root = cls._lock_root_package(lock_data)
        if root is None:
            return {}
        by_name: Dict[str, List[Dict]] = defaultdict(list)
        for pkg in lock_data.get("package", []):
            by_name[cls._normalize_name(pkg.get("name", ""))].append(pkg)

        installed: Dict[str, set] = defaultdict(set)
        seen = set()
        stack = [(root, ())]
        while stack:
            pkg, extras = stack.pop()
            name = cls._normalize_name(pkg.get("name", ""))
            node = (name, pkg.get("version"), extras)
            if node in seen:
                continue
            seen.add(node)
            if pkg is not root and "version" in pkg:
                installed[name].add(pkg["version"])
            edges = list(pkg.get("dependencies", []))
            for extra in extras:
                edges += pkg.get("optional-dependencies", {}).get(extra, [])
            for edge in edges:
                if not cls._marker_applies(edge.get("marker")):
                    continue
                targets = by_name.get(cls._normalize_name(edge["name"]), [])
                if "version" in edge:
                    targets = [t for t in targets if t.get("version") == edge["version"]]
                for target in targets:
                    stack.append((target, tuple(sorted(edge.get("extra", [])))))
        return dict(installed)

    def _requirements_source(self) -> str | None:
        """Path to the plugin's pyproject.toml, or None if the plugin has no
        dependencies to install. requirements.txt is not supported: plugins
        install exclusively through pyproject.toml + uv."""
        pyproject_file = os.path.join(self.path, "pyproject.toml")
        if os.path.exists(pyproject_file) and self._read_pyproject_dependencies(pyproject_file):
            return pyproject_file
        return None

    @staticmethod
    def _should_exit_from_installing_requirements(hash_file: str, current_hash: str) -> bool:
        """Check if the plugin's requirements have been installed."""
        if os.path.exists(hash_file):
            with open(hash_file, "r") as hf:
                return hf.read().strip() == current_hash
        return False

    async def _install_requirements(self):
        pyproject_file = self._requirements_source()
        if pyproject_file is None:
            return

        current_hash = self._hash_files(self._requirements_hash_inputs(self.path))
        hash_file = os.path.join(self.path, ".requirements_hash")

        # Fast path: neither the plugin's dependencies nor the root project's
        # locked versions have changed since last install
        if self._should_exit_from_installing_requirements(hash_file, current_hash):
            log.info(f"Plugin {self.id} requirements already installed")
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
                if self._should_exit_from_installing_requirements(hash_file, current_hash):
                    return
            # Poll timed out — fallback to blocking install
            await loop.run_in_executor(None, self._install_requirements_sync, pyproject_file, current_hash)
            return

        # We hold the lock → run the install in a thread
        try:
            await loop.run_in_executor(None, self._install_requirements_sync, pyproject_file, current_hash, lf)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()

    def _compile_and_install(self, pyproject_file: str, constraints: List[str], env: Dict[str, str]) -> None:
        """Compile the plugin's pyproject.toml into a uv.lock constrained by *constraints*, install from it and keep it
        next to the pyproject.toml, recorded as generated by the core."""
        with open(pyproject_file, "r") as f:
            document = tomlkit.parse(f.read())
        # dependency groups (dev included) are never installed: they must not take part in the resolution either, or a
        # test-only pin of the plugin could make it uninstallable next to the root project
        document.pop("dependency-groups", None)
        tool = document.setdefault("tool", tomlkit.table(is_super_table=True))
        uv = tool.setdefault("uv", tomlkit.table())
        uv.pop("dev-dependencies", None)
        uv["constraint-dependencies"] = list(uv.get("constraint-dependencies", [])) + constraints

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "pyproject.toml"), "w") as f:
                f.write(tomlkit.dumps(document))

            lock_cmd = ["uv", "lock", "--project", tmp_dir, "--no-cache"]
            if self._run_streamed(lock_cmd, env) != 0:
                raise subprocess.CalledProcessError(1, lock_cmd)

            sync_cmd = [
                "uv", "sync", "--project", tmp_dir,
                "--no-install-project", "--active", "--frozen", "--inexact", "--no-default-groups",
                "--link-mode=copy", "--no-cache",
            ]
            if self._run_streamed(sync_cmd, env) != 0:
                raise subprocess.CalledProcessError(1, sync_cmd)

            # Persist the compiled lock next to the plugin's pyproject.toml, replacing any existing one (generated by the
            # core: a uv.lock the plugin ships is never replaced, see _install_requirements_sync)
            lock = os.path.join(self.path, "uv.lock")
            shutil.copyfile(os.path.join(tmp_dir, "uv.lock"), lock)
            with open(os.path.join(self.path, LOCK_ORIGIN_FILE), "w") as f:
                f.write(f"generated:{self._hash_files([lock])}")

    def _install_from_shipped_lock(
        self, pyproject_file: str, lock: str, constraints: List[str], env: Dict[str, str]
    ) -> None:
        """Install the plugin's dependencies strictly from the uv.lock it ships, without regenerating it.

        Refused, with the reasons, when the lock does not match the plugin's pyproject.toml or would install a version
        of a library other than the one pinned in *constraints* (the root project's and the other plugins' locks). Its
        dependency groups (dev included) are not installed."""
        with open(lock, "rb") as f:
            lock_data = tomli.load(f)
        with open(pyproject_file, "rb") as f:
            project = tomli.load(f).get("project", {})

        problems = [f"uv.lock is out of date: {reason}" for reason in self._stale_lock_reasons(lock_data, project)]
        pinned = {self._normalize_name(c.split("==")[0]): c.split("==")[1] for c in constraints}
        for name, versions in sorted(self._locked_runtime_packages(lock_data).items()):
            for version in sorted(versions - {pinned[name]} if name in pinned else set()):
                problems.append(
                    f"uv.lock installs {name}=={version}, but {name}=={pinned[name]} is pinned by the core "
                    f"or by another installed plugin"
                )
        if problems:
            log.error(
                f"Plugin {self.id} ships a uv.lock that cannot be installed here; it is not regenerated, update it:\n"
                + "\n".join(f"  - {p}" for p in problems)
            )
            raise subprocess.CalledProcessError(1, ["uv", "sync", "--frozen"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copyfile(pyproject_file, os.path.join(tmp_dir, "pyproject.toml"))
            shutil.copyfile(lock, os.path.join(tmp_dir, "uv.lock"))
            sync_cmd = [
                "uv", "sync", "--project", tmp_dir,
                "--no-install-project", "--active", "--frozen", "--inexact", "--no-default-groups",
                "--link-mode=copy", "--no-cache",
            ]
            if self._run_streamed(sync_cmd, env) != 0:
                raise subprocess.CalledProcessError(1, sync_cmd)
        with open(os.path.join(self.path, LOCK_ORIGIN_FILE), "w") as f:
            f.write("shipped")

    @staticmethod
    def _run_streamed(cmd: List[str], env: Dict[str, str]) -> int:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        output = []
        if proc.stdout:
            for line in proc.stdout:
                log.debug(line.strip())
                output.append(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            # e.g. the dependency conflicting with the root project or with another plugin
            log.error(f"`{' '.join(cmd[:2])}` failed:\n" + "\n".join(output[-50:]))
        return proc.returncode

    def _install_requirements_sync(
        self, pyproject_file: str, current_hash: str, lock_fh=None
    ):
        """Synchronous install with cross-process serialization via fcntl.flock.

        A plugin that ships its own uv.lock is installed strictly from it,
        which is never regenerated: the lock must match the plugin's
        pyproject.toml and must not change the version of a library pinned by
        the root project's uv.lock or by another installed plugin, otherwise
        the installation fails and says why (see _install_from_shipped_lock).

        Otherwise the plugin's pyproject.toml, without its dependency groups
        (dev included: they are not installed), is compiled into a fresh
        uv.lock constrained against the root project's own uv.lock, so uv
        refuses to resolve a dangerous upgrade or downgrade of a system
        library, and against the uv.lock of the other installed plugins, so it
        never replaces a library another plugin installed. Dependencies are
        then installed strictly from that lock into the active virtual
        environment, and the lock is kept next to the pyproject.toml.

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
            if self._should_exit_from_installing_requirements(hash_file, current_hash):
                return

            start_time = time.time()
            log.info(f"Compiling and installing requirements for plugin {self.id}")

            constraints = self._root_lock_constraints()
            constraints += self._other_plugins_constraints({c.split("==")[0] for c in constraints})

            venv_dir = os.path.join(get_project_path(), ".venv")
            env = {**os.environ, "VIRTUAL_ENV": venv_dir}

            try:
                shipped_lock = self._shipped_lock(self.path)
                if shipped_lock is not None:
                    self._install_from_shipped_lock(pyproject_file, shipped_lock, constraints, env)
                else:
                    self._compile_and_install(pyproject_file, constraints, env)

                duration = time.time() - start_time
                log.info(f"Installation of requirements for {self.id} completed in {duration:.2f}s")

                # Persist hash so future calls take the fast path
                with open(hash_file, "w") as hf:
                    hf.write(current_hash)
            except subprocess.CalledProcessError as e:
                log.error(f"Error while installing plugin {self.id} requirements: {e}")
                raise Exception(f"Error while installing plugin {self.id} requirements")
        finally:
            if own_lock:
                fcntl.flock(lf, fcntl.LOCK_UN)
                lf.close()
            # Clean the plugin's own __pycache__ directories: its bytecode is the only one a reinstall can make stale.
            # Never the whole app: it holds the virtual environment (shared by the instances of a cluster) and the data
            for pycache in Path(self.path).rglob("__pycache__"):
                shutil.rmtree(pycache, ignore_errors=True)

    # lists of hooks and tools
    def _load_decorated_functions(self):
        hooks = []
        procedures = []
        endpoints = []
        plugin_overrides = []

        # every module is imported once: reloading them one by one, in the order of the files, left the module importing a
        # sibling with the classes (and the code) of the previous execution of the sibling. The modules of an installed
        # plugin loaded before (e.g. of a previous version) are forgotten first; the core plugins never change while the
        # Cat runs, and the core imports some of their modules directly: they are imported once and kept
        if not os.path.abspath(self._path).startswith(os.path.abspath(get_core_plugins_path())):
            package = self._get_py_filename_dotted_notation(os.path.join(self._path, "__init__.py")).removesuffix(".__init__")
            for name in [m for m in sys.modules if m == package or m.startswith(f"{package}.")]:
                del sys.modules[name]
            importlib.invalidate_caches()

        load_done = True
        for py_file in self._py_files:
            py_filename = self._get_py_filename_dotted_notation(py_file)

            log.debug(f"Import module {py_filename}")

            # save a reference to decorated functions
            try:
                plugin_module = importlib.import_module(py_filename)

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
