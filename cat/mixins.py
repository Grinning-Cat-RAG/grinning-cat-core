import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Dict

from cat.core_plugins.white_rabbit.white_rabbit import WhiteRabbit
from cat.log import log
from cat.looking_glass.mad_hatter.mad_hatter import MadHatter
from cat.rabbit_hole import RabbitHole
from cat.services.factory.agentic_workflow import BaseAgenticWorkflowHandler
from cat.services.factory.auth_handler import BaseAuthHandler
from cat.services.factory.chunker import BaseChunker
from cat.services.factory.context_retriever import BaseContextRetriever
from cat.services.factory.embedder import Embeddings
from cat.services.factory.file_manager import BaseFileManager
from cat.services.factory.llm import LargeLanguageModel
from cat.services.factory.vector_db import BaseVectorDatabaseHandler
from cat.services.service_provider import ServiceProvider


class ContextMixin(ABC):
    """
    Mixin for shared methods between all the classes managing settings of the agents.
    Provides access to chat request/response, user info, and core subsystems.
    """
    _white_rabbit: WhiteRabbit = None
    _service_provider: ServiceProvider = None

    @property
    def service_provider(self):
        if not self._service_provider:
            self._service_provider = ServiceProvider()
        return self._service_provider

    @property
    def mad_hatter(self) -> MadHatter:
        """
        Gives access to the `MadHatter` plugin manager.

        Returns:
            mad_hatter (MadHatter): Module to manage plugins.

        Examples
        --------
        Obtain the path in which your plugin is located
        >> cat.mad_hatter.get_plugin().path
        /app/cat/plugins/my_plugin
        Obtain plugin settings
        >> cat.mad_hatter.get_plugin().load_settings()
        {"num_cats": 44, "rows": 6, "remainder": 0}
        """
        return getattr(self, "plugin_manager")

    @property
    def white_rabbit(self) -> WhiteRabbit:
        return self._white_rabbit

    @white_rabbit.setter
    def white_rabbit(self, white_rabbit: WhiteRabbit):
        if self._white_rabbit is None:
            self._white_rabbit = white_rabbit

    @property
    @abstractmethod
    def agent_key(self) -> str:
        """The agent's unique identifier, if applicable."""
        pass

    @abstractmethod
    def embedder(self) -> Embeddings:
        """
        Langchain `Embeddings` object.

        Returns:
            embedder: Langchain `Embeddings`. Langchain embedder to turn text into a vector.

        Examples
        --------
        >> cat.lizard.embedder.embed_query("Oh dear!")
        [0.2, 0.02, 0.4, ...]
        """
        pass

    @abstractmethod
    async def toggle_plugin(self, plugin_id: str):
        """
        Toggles the state of a plugin with the given ID.

        Args:
            plugin_id (str): The unique identifier of the plugin to be toggled.
        """
        pass


class OrchestratorMixin(ContextMixin, ABC):
    """
    Mixin for shared methods for the orchestrator class.
    Provides access to chat request/response, user info, and core subsystems.
    """
    async def embedder(self) -> Embeddings:
        return await self.service_provider.get_embedder(self.agent_key, self.mad_hatter)


class BotMixin(ContextMixin, ABC):
    """
    Mixin for shared methods between StrayCat and CheshireCat.
    Provides access to chat request/response, user info, and core subsystems.
    """
    _agentic_workflow = None
    _chunker = None
    _context_retriever = None
    _custom_auth_handler = None
    _file_manager = None
    _large_language_model = None
    _vector_memory_handler = None

    @property
    def lizard(self) -> "BillTheLizard":  # type: ignore[name-defined]
        """
        Instance of `BillTheLizard`. Use it to access the main components of the Cat.

        Returns:
            lizard: BillTheLizard
                Instance of langchain `BillTheLizard`.
        """
        from cat.looking_glass.bill_the_lizard import BillTheLizard

        return BillTheLizard()

    async def embedder(self) -> Embeddings:
        return await self.lizard.embedder()

    @property
    def agentic_workflow(self) -> BaseAgenticWorkflowHandler:
        return self._agentic_workflow

    @agentic_workflow.setter
    def agentic_workflow(self, value: BaseAgenticWorkflowHandler):
        self._agentic_workflow = value

    @property
    def chunker(self) -> BaseChunker:
        return self._chunker

    @chunker.setter
    def chunker(self, value: BaseChunker):
        self._chunker = value

    @property
    def context_retriever(self) -> BaseContextRetriever:
        return self._context_retriever

    @context_retriever.setter
    def context_retriever(self, value: BaseContextRetriever):
        self._context_retriever = value

    @property
    def custom_auth_handler(self) -> BaseAuthHandler:
        return self._custom_auth_handler

    @custom_auth_handler.setter
    def custom_auth_handler(self, value: BaseAuthHandler):
        self._custom_auth_handler = value

    @property
    def file_manager(self) -> BaseFileManager:
        return self._file_manager

    @file_manager.setter
    def file_manager(self, value: BaseFileManager):
        self._file_manager = value

    @property
    def large_language_model(self) -> LargeLanguageModel:
        return self._large_language_model

    @large_language_model.setter
    def large_language_model(self, value: LargeLanguageModel):
        self._large_language_model = value

    @property
    def vector_memory_handler(self) -> BaseVectorDatabaseHandler:
        return self._vector_memory_handler

    @vector_memory_handler.setter
    def vector_memory_handler(self, value: BaseVectorDatabaseHandler):
        self._vector_memory_handler = value

    @property
    def rabbit_hole(self) -> RabbitHole:
        """
        Gives access to the `RabbitHole`, to upload documents and URLs into the vector DB.

        Returns:
            rabbit_hole (RabbitHole): Module to ingest documents and URLs for RAG.

        Examples
        --------
        >> cat.rabbit_hole.ingest_file(...)
        """
        return self.lizard.rabbit_hole

    # each time we access the file handlers, plugins can intervene
    async def file_handlers(self) -> Dict:
        return await self.mad_hatter.execute_hook("rabbithole_instantiates_parsers", {}, caller=self)

    def __del__(self):
        if not inspect.iscoroutinefunction(self.shutdown):
            self.shutdown()
            return

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.shutdown())
            else:
                loop.run_until_complete(self.shutdown())
        except Exception:
            message = f"Error while shutting down {self.__class__.__name__} '{self.agent_key}'"
            if hasattr(self, "id"):
                message += f" - id '{getattr(self, 'id')}'"
            log.warning(message)
            pass

    async def shutdown(self) -> None:
        setattr(self, "plugin_manager", None)
        self._agentic_workflow = None
        self._chunker = None
        self._custom_auth_handler = None
        self._file_manager = None
        self._large_language_model = None
        if self._vector_memory_handler:
            try:
                await self._vector_memory_handler.close()
            except:
                pass
            self._vector_memory_handler = None
        self._service_provider = None


class NonCopyableMixin:
    """
    Class that prevents copying operations.

    This class is designed to prevent both shallow and deep copying operations. Attempting to
    copy an instance of this class using either the `copy` or `deepcopy` methods will return
    the instance itself, effectively making it non-copyable. This is useful in scenarios where
    objects need to maintain unique identity and state integrity, avoiding unintended duplication.

    Methods:
        __copy__: Overrides shallow copy behavior to return the same instance.
        __deepcopy__: Overrides deep copy behavior to return the same instance and registers
                      it in the memo dictionary to prevent circular reference issues.
    """
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self  # register in the memo to avoid loops on circular references
        return self
