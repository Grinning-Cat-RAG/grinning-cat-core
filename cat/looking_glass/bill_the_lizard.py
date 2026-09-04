import asyncio
from typing import List, Dict
from fastapi import FastAPI

from cat.auth.auth_utils import DEFAULT_ADMIN_USERNAME, hash_password
from cat.auth.permissions import get_full_permissions
from cat.db import crud
from cat.db.cruds import plugins as crud_plugins
from cat.db.cruds import settings as crud_settings
from cat.db.cruds import users as crud_users
from cat.db.database import DEFAULT_CONVERSATIONS_KEY, DEFAULT_SYSTEM_KEY
from cat.db.models import Setting
from cat.env import get_env
from cat.log import log
from cat.looking_glass.cheshire_cat import CheshireCat
from cat.looking_glass.mad_hatter.mad_hatter import MadHatter
from cat.looking_glass.mad_hatter.registry import PluginRegistry
from cat.mixins import NonCopyableMixin, OrchestratorMixin
from cat.rabbit_hole import RabbitHole
from cat.services.factory.auth_handler import CoreAuthHandler
from cat.services.websocket_manager import WebSocketManager
from cat.utils import safe_deepcopy, sanitize_permissions, singleton


@singleton
class BillTheLizard(OrchestratorMixin, NonCopyableMixin):
    """
    Singleton class that manages the Cheshire Cats and their strays.

    The Cheshire Cats are the agents that are currently active and have users to attend.
    The strays are the users that are waiting for an agent to attend them.

    The Bill The Lizard Manager is responsible for:
    - Creating and deleting Cheshire Cats
    - Adding and removing strays from Cheshire Cats
    - Getting the Cheshire Cat of a stray
    - Getting the strays of a Cheshire Cat
    """
    def __init__(self):
        """
        Bill the Lizard initialization.

        The constructor is intentionally kept **synchronous and side-effect-free**: it only initialises the plugin manager
        so that the singleton exists immediately.
        All async bootstrap work (plugin discovery, hook execution, service setup) is deferred to :meth:`bootstrap`,
        which is called by `startup_app` inside uvicorn's event loop.
        """
        self._plugin_registry = None
        self._fastapi_app = None
        self._pending_endpoints = []

        # Minimal sync setup — plugin manager is created but NOT yet discovered.
        self.plugin_manager = MadHatter(self.agent_key)  # type: ignore[arg-type]

        # These are populated in bootstrap(); set to None so callers can detect
        # that bootstrap hasn't run yet.
        self.websocket_manager = None
        self.rabbit_hole = None
        self.core_auth_handler = None

    async def bootstrap(self):
        """
        Fully initialise the lizard inside uvicorn's running event loop.

        Must be awaited from an async context (e.g. the lifespan coroutine) so that:
        - `discover_plugins` can await Redis calls.
        - Hook functions that call `asyncio.ensure_future` schedule tasks on the
          **correct** (uvicorn) event loop instead of a transient side-thread loop.
        """
        # Discover and load all plugins (async: reads active_plugins from Redis)
        await self.plugin_manager.discover_plugins()

        # Store endpoints for later activation (after fastapi_app is set)
        self._pending_endpoints = safe_deepcopy(self.plugin_manager.endpoints)
        if self._fastapi_app is not None:
            self._activate_pending_endpoints()

        # Allow plugins to act before the remaining cat components are created
        await self.plugin_manager.execute_hook("before_lizard_bootstrap", caller=self)

        self.websocket_manager = WebSocketManager()
        self.rabbit_hole = RabbitHole()
        self.core_auth_handler = CoreAuthHandler()

        await self.service_provider.bootstrap_services(self.agent_key, self.plugin_manager)

        # Initialize the default admin if not present
        if not await crud_users.get_users(DEFAULT_SYSTEM_KEY, limit=1):
            permissions = sanitize_permissions(get_full_permissions(), DEFAULT_SYSTEM_KEY)

            await crud_users.create_user(DEFAULT_SYSTEM_KEY, {
                "username": DEFAULT_ADMIN_USERNAME,
                "password": hash_password(get_env("CAT_ADMIN_DEFAULT_PASSWORD")),
                "permissions": permissions,  # base admin has all permissions, but CHAT
            })

        # Start Redis Pub/Sub listener so WebSocket messages are delivered
        # cross-replica in Docker Swarm deployments.  Degrades gracefully to
        # local-only mode if Redis pub/sub is unavailable.
        await self.websocket_manager.start()

        await self.plugin_manager.execute_hook("after_lizard_bootstrap", caller=self)

    async def create_cheshire_cat(self, agent_id: str, metadata: Dict | None = None) -> CheshireCat:
        """
        Create the Cheshire Cat with the given id, directly from db.

        Args:
            agent_id: The id of the agent to get
            metadata: The metadata of the agent to create

        Returns:
            The Cheshire Cat with the given id, or None if it doesn't exist
        """
        if agent_id == DEFAULT_SYSTEM_KEY:
            raise ValueError(f"{agent_id} is not allowed as name for agents")

        if agent_id in await crud_settings.get_agents_main_keys():
            return await self.get_cheshire_cat(agent_id)  # type: ignore[return-value]

        ccat = None
        try:
            ccat = await CheshireCat.create(agent_id)
            if metadata is not None:
                await crud_settings.upsert_setting_by_name(
                    ccat.agent_key,
                    Setting(name="metadata", value=metadata),
                )

            embedder = await self.embedder()
            await ccat.vector_memory_handler.initialize(embedder.name, embedder.size)
            await ccat.embed_procedures()

            await self.plugin_manager.execute_hook("after_cheshire_cat_creation", ccat, caller=self)

            return ccat
        except Exception as e:
            log.error(f"Error creating Cheshire Cat `{agent_id}`: {e}")
            await self.rollback_cheshire_cat_creation(agent_id, ccat)

            raise

    async def rollback_cheshire_cat_creation(self, agent_id: str, cat: CheshireCat | None) -> None:
        """
        Rollback the creation of a Cheshire Cat with the given id.

        Args:
            agent_id: The id of the agent to rollback
            cat: The Cheshire Cat to rollback

        Returns:
            None
        """
        # rollback
        if cat:
            await cat.destroy()
            return

        await crud.delete(agent_id)

    async def _get_cheshire_cat_on_plugin_event(self, agent_id: str, plugin_id: str) -> CheshireCat | None:
        """
        Determines and retrieves the CheshireCat object associated with a specific plugin event for a given agent if the
        plugin is active.

        Args:
            agent_id (str): The unique identifier for the agent.
            plugin_id (str): The unique identifier for the plugin.

        Returns:
            CheshireCat | None: The CheshireCat object if the plugin is active, otherwise None.
        """
        active_plugins = await crud_plugins.get_active_plugins_from_db(agent_id)
        if not active_plugins or plugin_id not in active_plugins:
            return None

        return await self.get_cheshire_cat(agent_id)

    @staticmethod
    async def get_cheshire_cat(agent_id: str) -> CheshireCat | None:
        """
        Gets the Cheshire Cat with the given id, directly from db.

        Args:
            agent_id: The id of the agent to get

        Returns:
            The Cheshire Cat with the given id, or None if it doesn't exist
        """
        if agent_id == DEFAULT_SYSTEM_KEY:
            log.debug("The system agent has been requested: returning null value.")
            return None

        if agent_id not in await crud_settings.get_agents_main_keys():
            log.debug(f"Requested not existing `{agent_id}`")
            raise ValueError("Bad Request")

        agent_settings = await crud_settings.get_settings(agent_id)
        if not agent_settings:
            log.debug(f"Agent `{agent_id}` has no settings")
            return None

        return await CheshireCat.create(agent_id)

    async def clone_cheshire_cat(self, ccat: CheshireCat, new_agent_id: str) -> CheshireCat:
        """
        Clone a Cheshire Cat into a new one.

        Args:
            ccat: The Cheshire Cat to clone.
            new_agent_id: The new agent id to clone into.

        Returns:
            The cloned Cheshire Cat.
        """
        # clone the settings from the provided agent
        log.info(f"Cloning settings from agent {ccat.agent_key} to agent {new_agent_id}")
        await crud_settings.clone_agent(ccat.agent_key, new_agent_id, [DEFAULT_CONVERSATIONS_KEY])

        # delegate cloning of in-memory data/resources from the source Cheshire Cat to the new one
        cloned_ccat = await self.get_cheshire_cat(new_agent_id)
        await cloned_ccat.clone_from(ccat)  # type: ignore[arg-type]

        return cloned_ccat  # type: ignore[return-value]

    async def embed_all_in_cheshire_cats(self) -> None:
        """Re-embeds all the stored files and procedures in all the Cheshire Cats using the current embedder.

        Upstream-parity implementation with one fix: the upstream version runs
        ``embed_with_limit`` inside the per-agent initialization loop, i.e. the
        whole re-embed pass is executed once PER AGENT (N passes for N agents).
        Here every database is initialized first (serialized), then the
        re-embeds run exactly once under the concurrency cap.
        """

        async def embed_with_limit(entry_):
            async with semaphore:
                # re-embed all the stored files
                tasks = [
                    entry_["ccat"].embed_stored_sources(collection_name, sources)
                    for collection_name, sources in entry_["stored_sources"].items()
                    if sources
                ] + [entry_["ccat"].embed_procedures()]
                await asyncio.gather(*tasks)

        success = False
        try:
            embedder = await self.embedder()
            embedder_name = embedder.name
            embedder_size = embedder.size

            ccat_ids = await crud_settings.get_agents_main_keys()
            stored_files_by_ccat: List[Dict] = []
            # first, I need to get all the stored files from all the Cheshire Cats with the metadata stored
            # within the vector memory; I do not remove anything from the latter to avoid any race condition
            for ccat_id in ccat_ids:
                if (ccat := await self.get_cheshire_cat(ccat_id)) is None:
                    continue

                stored_files_by_ccat.append({
                    "ccat": ccat,
                    "stored_sources": await ccat.get_stored_sources_with_metadata(),
                })

            # re-initialize all the vector databases in a serialized way, outside threads to avoid
            # race conditions
            for entry in stored_files_by_ccat:
                await entry["ccat"].vector_memory_handler.initialize(embedder_name, embedder_size)

            # finally, re-embed all the stored files in an asynchronous way
            # limit concurrent embeddings to avoid overwhelming resources
            semaphore = asyncio.Semaphore(5)  # Max 5 concurrent
            await asyncio.gather(*[embed_with_limit(entry) for entry in stored_files_by_ccat])

            success = True
        except Exception as e:
            log.error(f"Error embedding all stored files: {e}")

        await self.plugin_manager.execute_hook(
            "after_all_cheshire_cats_embedded", success, caller=self,
        )

    def is_custom_endpoint(self, path: str, methods: List[str] | None = None):
        """
        Check if the given path and methods correspond to a custom endpoint.

        Args:
            path (str): The path of the endpoint to check.
            methods (List[str] | None): The HTTP methods of the endpoint to check. If None, checks all methods.

        Returns:
            bool: True if the endpoint is a custom endpoint, False otherwise.
        """
        return any(
            ep.real_path == path and (methods is None or set(ep.methods) == set(methods))
            for ep in self.plugin_manager.endpoints
        )

    async def install_plugin(self, plugin_path: str) -> str:
        try:
            plugin_id = await self.plugin_manager.install_plugin(plugin_path)

            await self.on_plugin_activate(plugin_id)
            await self.plugin_manager.execute_hook(
                "lizard_notify_plugin_installation", plugin_id, plugin_path, caller=self,
            )

            return plugin_id
        except Exception as e:
            log.error(f"Could not install plugin from {plugin_path}: {e}")
            raise

    async def uninstall_plugin(self, plugin_id: str, dispatch_event: bool = True):
        try:
            # deactivate plugins in the Cheshire Cats
            if plugin_id in self.plugin_manager.active_plugins:
                await self.on_plugin_deactivate(plugin_id)

            await self.plugin_manager.uninstall_plugin(plugin_id)
        except Exception as e:
            log.error(f"Could not uninstall plugin {plugin_id}: {e}")
            raise
        finally:
            if dispatch_event:
                await self.plugin_manager.execute_hook(
                    "lizard_notify_plugin_uninstallation", plugin_id, caller=self,
                )

    async def toggle_plugin(self, plugin_id: str):
        # the plugin is active, and evidently I am deactivating it: deactivate it in the Cheshire Cats before
        # deactivating it on a system level
        if plugin_id in self.plugin_manager.active_plugins:
            await self.on_plugin_deactivate(plugin_id)

        # toggle (activate or deactivate) the plugin
        await self.plugin_manager.toggle_plugin(plugin_id)

        # if the plugin is now active, activate it in the Cheshire Cats
        if plugin_id in self.plugin_manager.active_plugins:
            await self.on_plugin_activate(plugin_id)

        await self.plugin_manager.execute_hook("after_plugin_toggling_on_system", plugin_id, caller=self)

    def activate_plugin_endpoints(self, plugin_id: str):
        # Store endpoints for later activation
        self._pending_endpoints = safe_deepcopy(self.plugin_manager.plugins[plugin_id].endpoints)
        self._activate_pending_endpoints()

    async def on_plugin_activate(self, plugin_id: str) -> None:
        self.activate_plugin_endpoints(plugin_id)

        # if I already installed and activated the plugin and I am now re-installing it, then migrate plugin settings in
        # the Cheshire Cats to incrementally apply the new settings
        for ccat_id in await crud_plugins.get_agents_plugin_keys(plugin_id):
            # if the plugin is not active for the Cheshire Cat, then skip it
            if (ccat := await self._get_cheshire_cat_on_plugin_event(ccat_id, plugin_id)) is None:
                continue
            await ccat.plugin_manager.activate_plugin(plugin_id)

    async def on_plugin_deactivate(self, plugin_id: str):
        # deactivate the endpoints from the plugin
        if endpoints := self.plugin_manager.plugins[plugin_id].endpoints:
            for endpoint in endpoints:
                endpoint.deactivate(self.fastapi_app)

        for ccat_id in await crud_plugins.get_agents_plugin_keys(plugin_id):
            # if the plugin is not active for the Cheshire Cat, then skip it
            if (ccat := await self._get_cheshire_cat_on_plugin_event(ccat_id, plugin_id)) is None:
                continue
            await ccat.plugin_manager.deactivate_plugin(plugin_id)

    def _activate_pending_endpoints(self) -> None:
        for endpoint in self._pending_endpoints:
            endpoint.activate(self.fastapi_app)
        self._pending_endpoints.clear()

    async def shutdown(self) -> None:
        """
        Shuts down the Bill The Lizard Manager. It closes all the strays' connections and stops the scheduling system.

        Returns:
            None
        """
        await self.plugin_manager.execute_hook("before_lizard_shutdown", caller=self)
        if self.websocket_manager:
            await self.websocket_manager.close_connections()

        endpoints = [
            endpoint
            for plugin_id in self.plugin_manager.active_plugins
            for endpoint in self.plugin_manager.plugins[plugin_id].endpoints
        ]
        for endpoint in endpoints:
            endpoint.deactivate(self.fastapi_app)

        self.core_auth_handler = None
        self.plugin_manager = None
        self.rabbit_hole = None
        self.websocket_manager = None
        self.fastapi_app = None

    @property
    def fastapi_app(self):
        return self._fastapi_app

    @fastapi_app.setter
    def fastapi_app(self, app: FastAPI | None = None):
        self._fastapi_app = app

    @property
    def plugin_registry(self) -> PluginRegistry:
        return self._plugin_registry

    @plugin_registry.setter
    def plugin_registry(self, registry: PluginRegistry):
        self._plugin_registry = registry

    @property
    def agent_key(self):
        return DEFAULT_SYSTEM_KEY
