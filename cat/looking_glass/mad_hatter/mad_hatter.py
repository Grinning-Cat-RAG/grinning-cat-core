import glob
from inspect import iscoroutinefunction
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from cat import utils
from cat.db.cruds import plugins as crud_plugins, settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.db.models import Setting
from cat.exceptions import ManagementModeException
from cat.log import log
from cat.looking_glass.mad_hatter.decorators.endpoint import CatEndpoint
from cat.looking_glass.mad_hatter.decorators.hook import CatHook
from cat.looking_glass.mad_hatter.plugin import Plugin
from cat.looking_glass.mad_hatter.plugin_extractor import PluginExtractor
from cat.looking_glass.mad_hatter.procedures import CatProcedure


class LoadedPlugin(BaseModel):
    plugin: Plugin | None = None
    missing_dependencies: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MadHatter:
    """
    This is the abstract class that defines the methods that the plugin managers should implement.
    """
    def __init__(self, agent_key: str):
        self._agent_key = agent_key
        self._skip_folders = ["__pycache__", "lost+found"]

        self.plugins: Dict[str, Plugin] = {}
        # a unified registry for all procedures (local tools, forms, remote clients)
        self.procedures_registry: Dict[str, CatProcedure] = {}
        # dict of active plugins hooks (hook_name -> [CatHook, CatHook, ...])
        self.hooks: Dict[str, List[CatHook]] = {}
        # list of active plugins endpoints
        self.endpoints: List[CatEndpoint] = []

        self.active_plugins: List[str] = []

    # discover all plugins
    async def discover_plugins(self):
        # emptying the plugin dictionary, plugins will be discovered from the disk
        # and stored in a dictionary plugin_id -> plugin_obj
        self.plugins = {}

        # plugins are found in the plugins folder,
        # plus the default core plugin (where default hooks and tools are defined)
        # plugin folder is "cat/plugins/" in production, "tests/mocks/mock_plugin_folder/" during tests
        self.active_plugins = await self.load_active_plugins_ids_from_db()

        await self._on_discovering_plugins()
        await self._on_finish_discovering_plugins()

    async def load_active_plugins_ids_from_db(self) -> List[str]:
        active_plugins_from_db = await crud_settings.get_setting_by_name(self.agent_key, "active_plugins")
        active_plugins: List[str] = [] if active_plugins_from_db is None else active_plugins_from_db["value"]

        if not active_plugins:
            # the core plugins should be appended when the agent is created, i.e. has no active plugins in db
            active_plugins.extend(self.get_core_plugins_ids)

        # ensure all core untoggling plugins (e.g. base_plugin and other always-active plugins) are active
        active_plugins.extend(self.get_untoggling_plugin_ids)

        # Remove duplicates
        active_plugins = list(set(active_plugins))

        # Ensure base_factory is first
        if self.get_base_core_plugin_id in active_plugins:
            active_plugins.remove(self.get_base_core_plugin_id)
        active_plugins.insert(0, self.get_base_core_plugin_id)

        return active_plugins

    async def install_plugin(self, package_plugin: str) -> str:
        """
        Install a plugin from a package file (zip/tar).

        Args:
            package_plugin (str): The path to the plugin package file.

        Returns:
            str: The ID of the installed plugin.
        """
        # extract zip/tar file into plugin folder
        extractor = PluginExtractor(package_plugin)
        plugin_path = extractor.extract(utils.get_plugins_path())
        plugin_id = extractor.id

        if missing_deps := (await self.load_plugin(plugin_id, with_deactivation=False)).missing_dependencies:
            # remove plugin folder
            shutil.rmtree(plugin_path)
            raise Exception(f"Cannot install plugin {plugin_id} because of missing dependencies: {missing_deps}")

        # install the extracted plugin
        return await self.install_extracted_plugin(plugin_id)

    async def install_extracted_plugin(self, plugin_id: str) -> str:
        """
        Installs and activates a plugin if it is not already activated. This method verifies if the given plugin ID
        exists among core plugin IDs, and if not, activates the corresponding plugin.

        Args:
            plugin_id: Unique identifier for the plugin to be installed.

        Returns:
            The plugin ID as a string, whether it was already installed or newly activated.
        """
        # create plugin obj, and eventually activate it
        if plugin_id in self.get_core_plugins_ids:
            return plugin_id

        await self.activate_plugin(plugin_id)

        return plugin_id

    async def uninstall_plugin(self, plugin_id: str):
        """
        Uninstalls the specified plugin by its ID. This includes removing the plugin folder, deactivating the plugin if
        it is active, and clearing it from any stored metadata. If the plugin is a dependency of other plugins, the
        operation is not performed and an exception is raised.

        Args:
            plugin_id (str): The unique identifier of the plugin to be uninstalled.

        Raises:
            Exception: If the plugin is a dependency of other plugins, an exception is raised with the list of dependent
                plugins.
        """
        if not self.plugin_exists(plugin_id) or plugin_id in self.get_core_plugins_ids:
            return

        # if the plugin is within the dependencies of other plugins, raise an exception
        dependent_plugins = self._get_plugins_depending_on(plugin_id)
        if dependent_plugins:
            raise Exception(
                f"Cannot uninstall plugin {plugin_id} because it is a dependency of the following plugins: "
                f"{', '.join(dependent_plugins)}"
            )

        plugin_path = self.plugins[plugin_id].path

        # deactivate plugin if it is active (will sync cache)
        if plugin_id in self.active_plugins:
            await self.deactivate_plugin(plugin_id)

        # remove plugin folder
        shutil.rmtree(plugin_path)

        await crud_plugins.destroy_plugin(plugin_id)

    async def activate_plugin(self, plugin_id: str):
        if not self.plugin_exists(plugin_id):
            raise Exception(f"Plugin {plugin_id} not present in plugins folder")

        if await self._on_plugin_activation(plugin_id=plugin_id):
            # Add the plugin in the list of active plugins
            self.active_plugins.append(plugin_id)

        await self._on_finish_discovering_plugins()

    async def deactivate_plugin(self, plugin_id: str):
        if not self.plugin_exists(plugin_id):
            raise Exception(f"Plugin {plugin_id} not present in plugins folder")

        if plugin_id not in self.active_plugins or plugin_id in self.get_untoggling_plugin_ids:
            return

        # if the plugin is within the dependencies of other plugins, it cannot be deactivated (raise an exception)
        dependent_plugins = self._get_plugins_depending_on(plugin_id)
        if dependent_plugins:
            raise Exception(
                f"Cannot deactivate plugin {plugin_id} because it is a dependency of the following plugins: "
                f"{', '.join(dependent_plugins)}"
            )

        # Deactivate the plugin
        log.warning(f"Toggle plugin '{plugin_id}' for agent '{self.agent_key}': Deactivate")
        try:
            if self.agent_key == DEFAULT_SYSTEM_KEY:
                await self.plugins[plugin_id].deactivate(self.agent_key)
            else:
                await self.plugins[plugin_id].deactivate_settings(self.agent_key)

            # Remove the plugin from the list of active plugins
            self.active_plugins.remove(plugin_id)
            self.plugins.pop(plugin_id, None)

            await self._on_finish_discovering_plugins()
        except Exception as e:
            log.error(f"Could not deactivate plugin {plugin_id}: {e}")

    # activate / deactivate plugin
    async def toggle_plugin(self, plugin_id: str):
        if plugin_id in self.get_untoggling_plugin_ids:
            raise Exception(f"{', '.join(self.get_untoggling_plugin_ids)} cannot be deactivated")

        if not self.plugin_exists(plugin_id):
            raise Exception(f"Plugin {plugin_id} not active in the system")

        if plugin_id in self.active_plugins:
            await self.deactivate_plugin(plugin_id)
            return

        await self.activate_plugin(plugin_id)

    async def _on_finish_discovering_plugins(self):
        # store active plugins in db
        active_plugins = list(set(self.active_plugins))
        await crud_settings.upsert_setting_by_name(self.agent_key, Setting(name="active_plugins", value=active_plugins))

        log.debug(f"Agent '{self.agent_key}' - ACTIVE PLUGINS:")
        log.debug(self.active_plugins)

        # update cache and embeddings
        self.hooks = {}
        self.procedures_registry = {}
        self.endpoints = []

        for plugin_id in self.active_plugins:
            plugin = self.plugins[plugin_id]
            # Load local tools, forms and mcp clients as procedures
            self.procedures_registry |= {p.name: p for p in plugin.procedures}
            self.endpoints += plugin.endpoints

            # cache hooks (indexed by hook name)
            for h in plugin.hooks:
                self.hooks.setdefault(h.name, []).append(h)

        # sort each hooks list by priority
        for hook_name in self.hooks.keys():
            self.hooks[hook_name].sort(key=lambda x: x.priority, reverse=True)

    async def execute_hook(self, hook_name: str, *args, caller: "ContextMixin") -> Any:  # type: ignore[override, name-defined]
        """
        Execute a hook from an **async** call site.

        Async plugin hooks are **awaited directly** on the running event loop —
        no new OS thread, no new event loop, no `thread.join()` blocking.
        Synchronous plugin hooks are called in-line (safe for typical I/O-light bodies).

        Use this instead of `execute_hook` whenever the caller is already inside
        an `async` function.  Keep `execute_hook` for sync call sites (e.g. ``__init__``).
        """
        if hook_name not in self.hooks:
            raise Exception(f"Hook {hook_name} not present in any plugin")

        tea_cup = utils.safe_deepcopy(args[0]) if args else None
        for hook in self.hooks[hook_name]:
            try:
                log.debug(f"Executing {hook.plugin_id}::{hook.name} with priority {hook.priority}")
                hook_args = (utils.safe_deepcopy(tea_cup), *utils.safe_deepcopy(args[1:])) if args else ()
                kwargs = {self.context_execute_hook: caller}

                tea_spoon = (
                    await hook.function(*hook_args, **kwargs)
                    if iscoroutinefunction(hook.function)
                    else hook.function(*hook_args, **kwargs)
                )
                if tea_spoon is not None:
                    tea_cup = tea_spoon
            except ManagementModeException:
                # a deliberate management gate (mgmt_message plugin): not an
                # error — propagate to the caller (ConnectionAuth -> startup
                # handler -> 403) without logging and without swallowing
                raise
            except Exception as e:
                log.error(f"Error in plugin {hook.plugin_id}::{hook.name}: {e}")
                log.warning(self.plugins[hook.plugin_id].plugin_specific_error_message())  # type: ignore[union-attr, index]

        return tea_cup

    def get_plugin(self):
        name = utils.inspect_calling_folder()
        return self.plugins[name]

    def _get_plugin_folder_path(self, plugin_id: str) -> str:
        return os.path.join(
            utils.get_core_plugins_path() if plugin_id in self.get_core_plugins_ids else utils.get_plugins_path(),
            plugin_id
        )

    async def load_plugin(self, plugin_id: str, with_deactivation: bool = True) -> LoadedPlugin:
        try:
            folder = self._get_plugin_folder_path(plugin_id)
            plugin = Plugin(folder)

            if deps := plugin.missing_dependencies(self.load_active_plugins_ids_from_folders()):
                if with_deactivation:
                    await self.deactivate_plugin(plugin_id)
                return LoadedPlugin(plugin=None, missing_dependencies=deps)

            return LoadedPlugin(plugin=plugin)
        except Exception as e:
            log.error(str(e))
            return LoadedPlugin()

    def load_active_plugins_ids_from_folders(self):
        all_plugin_folders = list(set(
            glob.glob(f"{utils.get_core_plugins_path()}/*/") + glob.glob(f"{utils.get_plugins_path()}/*/")
        ))

        plugins = [
            plugin_id
            for folder in all_plugin_folders
            if ((plugin_id := os.path.basename(os.path.normpath(folder))) not in self._skip_folders)
        ]

        # Ensure base_factory is first
        if self.get_base_core_plugin_id in plugins:
            plugins.remove(self.get_base_core_plugin_id)
        plugins.insert(0, self.get_base_core_plugin_id)

        return plugins

    # check if plugin exists
    def plugin_exists(self, plugin_id: str):
        return plugin_id in self.load_active_plugins_ids_from_folders()

    def _get_plugins_depending_on(self, plugin_id: str) -> List[str]:
        dependent_plugins = []
        for p_id, plugin in self.plugins.items():
            if plugin_id in plugin.manifest.dependencies:
                dependent_plugins.append(p_id)
        return dependent_plugins

    async def _on_discovering_plugins(self):
        if self.agent_key == DEFAULT_SYSTEM_KEY:
            if not self.active_plugins:
                self.active_plugins = self.load_active_plugins_ids_from_folders()

            for plugin_id in self.active_plugins:
                plugin = (await self.load_plugin(plugin_id)).plugin
                if not plugin:
                    log.error(f"Plugin {plugin_id} could not be loaded")
                    continue

                self.plugins[plugin.id] = plugin
                try:
                    await self._on_plugin_activation(plugin_id)
                except Exception as e:
                    # Couldn't activate the plugin -> Deactivate it
                    await self.deactivate_plugin(plugin_id)
                    self.active_plugins.remove(plugin_id)
                    raise e

        # plugins are already loaded when BillTheLizard is created; since its plugin manager scans the plugins folder
        # then, we just need to grab the plugins from there
        for plugin_id, plugin in (await self.available_plugins()).items():
            if plugin_id not in self.active_plugins:
                continue

            if plugin_id not in self.plugins.keys():
                self.plugins[plugin_id] = plugin
            try:
                await self.plugins[plugin_id].activate_settings(self.agent_key)
            except Exception as e:
                # Couldn't activate the plugin -> Deactivate it
                await self.toggle_plugin(plugin_id)
                raise e

    async def _on_plugin_activation(self, plugin_id: str) -> bool:
        plugin = (
            (await self.load_plugin(plugin_id)).plugin
            if self.agent_key == DEFAULT_SYSTEM_KEY
            else (await self.available_plugins()).get(plugin_id)
        )
        if not plugin:
            return False

        self.plugins[plugin_id] = plugin
        try:
            if self.agent_key == DEFAULT_SYSTEM_KEY:
                await self.plugins[plugin_id].activate(self.agent_key)
            else:
                await self.plugins[plugin_id].activate_settings(self.agent_key)
            return True
        except Exception as e:
            log.error(f"Could not activate plugin {plugin_id}: {e}")
            self.plugins.pop(plugin_id, None)
            return False

    @property
    def procedures(self) -> List[CatProcedure]:
        return list(self.procedures_registry.values())

    @property
    def get_base_core_plugin_id(self) -> str:
        return "base_plugin"

    @property
    def get_untoggling_plugin_ids(self) -> List[str]:
        # always-on core plugins: not only can they not be deactivated, they are
        # force-added to every agent's active_plugins even when the stored list
        # was customized without them (see load_active_plugins_ids_from_db).
        # ingestion_status is here for the same reason; multimodal_ingestion is
        # here so the extracted-image deletion cascade (before_file_manager_file_delete)
        # is registered on pre-existing agents too, not only on freshly-created ones.
        return [self.get_base_core_plugin_id] + [
            "ingestion_status", "white_rabbit", "march_hare", "multimodal_ingestion",
        ]

    @property
    def get_core_plugins_ids(self) -> List[str]:
        path = Path(utils.get_core_plugins_path())
        core_plugins = [p.name for p in path.iterdir() if p.is_dir()]
        return core_plugins

    @property
    def agent_key(self) -> str:
        return self._agent_key

    async def available_plugins(self) -> Dict[str, Plugin]:
        from cat.looking_glass.bill_the_lizard import BillTheLizard
        if self.agent_key == DEFAULT_SYSTEM_KEY:
            plugins_ids = self.load_active_plugins_ids_from_folders()

            result = {}
            for plugin_id in plugins_ids:
                plugin = (await self.load_plugin(plugin_id)).plugin
                if plugin:
                    result[plugin.id] = plugin

            return result

        # the `plugins` property of the plugin manager of BillTheLizard contains only the globally active plugins
        return BillTheLizard().plugin_manager.plugins

    @property
    def context_execute_hook(self) -> str:
        return "lizard" if self.agent_key == DEFAULT_SYSTEM_KEY else "cat"
