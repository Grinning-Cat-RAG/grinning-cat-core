from typing import Any, Dict
from pydantic import BaseModel

from cat.db import models
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.looking_glass.mad_hatter.mad_hatter import MadHatter
from cat.services.factory.agentic_workflow import BaseAgenticWorkflowHandler
from cat.services.factory.auth_handler import BaseAuthHandler
from cat.services.factory.context_retriever import BaseContextRetriever
from cat.services.factory.embedder import Embeddings
from cat.services.factory.ingestion import BaseIngestionEngine
from cat.services.factory.llm import LargeLanguageModel
from cat.services.service_factory import ServiceFactory
from cat.services.factory.chunker import BaseChunker
from cat.services.factory.file_manager import BaseFileManager
from cat.services.factory.vector_db import BaseVectorDatabaseHandler
from cat.utils import singleton


class FactoryParams(BaseModel):
    factory_allowed_handler_name: str
    setting_category: str
    schema_name: str


@singleton
class ServiceProvider:
    @staticmethod
    async def _create_service_object(agent_key: str, factory: ServiceFactory):
        # if no config is saved, use default one and save to db
        # create the settings for the factory
        await crud_settings.upsert_setting_by_name(
            agent_key,
            models.Setting(
                name=factory.default_config_class.__name__,
                category=factory.setting_category,
                value=factory.default_config,
            ),
        )

    async def _get_service_object(self, agent_key: str, plugin_manager: MadHatter, factory_params: FactoryParams) -> Any:
        factory = ServiceFactory(agent_key, plugin_manager, **factory_params.model_dump())

        if not (selected_config := await crud_settings.get_settings_by_category(agent_key, factory.setting_category)):
            # if no config is saved, use the default one and save to db
            # create the settings for the factory
            await self._create_service_object(agent_key, factory)

            # reload from db and return
            selected_config = await crud_settings.get_settings_by_category(agent_key, factory.setting_category)

        return await factory.get_from_config_name(selected_config["name"])  # type: ignore[index]

    async def get_embedder(self, agent_key: str, plugin_manager: MadHatter) -> Embeddings:
        return await self._get_service_object(agent_key, plugin_manager, self._factory_services_params["embedder"])

    async def get_ingestion(self, agent_key: str, plugin_manager: MadHatter) -> BaseIngestionEngine:
        return await self._get_service_object(agent_key, plugin_manager, self._factory_services_params["ingestion"])

    async def get_large_language_model(self, agent_key: str, plugin_manager: MadHatter) -> LargeLanguageModel:
        return await self._get_service_object(
            agent_key, plugin_manager, self._factory_services_params["large_language_model"]
        )

    async def get_custom_auth_handler(self, agent_key: str, plugin_manager: MadHatter) -> BaseAuthHandler:
        return await self._get_service_object(
            agent_key, plugin_manager, self._factory_services_params["auth_handler"]
        )

    async def get_file_manager(self, agent_key: str, plugin_manager: MadHatter) -> BaseFileManager:
        return await self._get_service_object(agent_key, plugin_manager, self._factory_services_params["file_manager"])

    async def get_chunker(self, agent_key: str, plugin_manager: MadHatter) -> BaseChunker:
        return await self._get_service_object(agent_key, plugin_manager, self._factory_services_params["chunker"])

    async def get_context_retriever(self, agent_key: str, plugin_manager: MadHatter) -> BaseContextRetriever:
        context_retriever = await self._get_service_object(
            agent_key, plugin_manager, self._factory_services_params["context_retriever"],
        )
        context_retriever.vector_memory_handler = await self.get_vector_memory_handler(agent_key, plugin_manager)
        return context_retriever

    async def get_vector_memory_handler(self, agent_key: str, plugin_manager: MadHatter) -> BaseVectorDatabaseHandler:
        return await self._get_service_object(
            agent_key, plugin_manager, self._factory_services_params["vector_memory_handler"]
        )

    async def get_agentic_workflow(self, agent_key: str, plugin_manager: MadHatter) -> BaseAgenticWorkflowHandler:
        return await self._get_service_object(
            agent_key, plugin_manager, self._factory_services_params["agentic_workflow"]
        )

    async def bootstrap_services(self, agent_key: str, plugin_manager: MadHatter):
        list_factory_params = (
            self._orchestrator_factory_services_params
            if agent_key == DEFAULT_SYSTEM_KEY
            else self._bot_factory_services_params
        )

        for _, factory_params in list_factory_params.items():
            factory = ServiceFactory(agent_key, plugin_manager, **factory_params.model_dump())
            selected_config = await crud_settings.get_settings_by_category(agent_key, factory.setting_category)
            if selected_config:
                continue

            # if no config is saved, use default one and save to db
            # create the settings for the factory
            await self._create_service_object(agent_key, factory)

    @property
    def _orchestrator_factory_services_params(self) -> Dict[str, FactoryParams]:
        return {
            "embedder": FactoryParams(
                factory_allowed_handler_name="factory_allowed_embedders",
                setting_category="embedder",
                schema_name="languageEmbedderName",
            ),
            "ingestion": FactoryParams(
                factory_allowed_handler_name="factory_allowed_ingestions",
                setting_category="ingestion",
                schema_name="ingestionName",
            ),
        }

    @property
    def _bot_factory_services_params(self) -> Dict[str, FactoryParams]:
        return {
            "agentic_workflow": FactoryParams(
                factory_allowed_handler_name="factory_allowed_agentic_workflows",
                setting_category="agentic_workflow",
                schema_name="agenticWorkflowName",
            ),
            "large_language_model": FactoryParams(
                factory_allowed_handler_name="factory_allowed_llms",
                setting_category="llm",
                schema_name="languageModelName",
            ),
            "auth_handler": FactoryParams(
                factory_allowed_handler_name="factory_allowed_auth_handlers",
                setting_category="auth_handler",
                schema_name="authorizatorName",
            ),
            "file_manager": FactoryParams(
                factory_allowed_handler_name="factory_allowed_file_managers",
                setting_category="file_manager",
                schema_name="fileManagerName",
            ),
            "chunker": FactoryParams(
                factory_allowed_handler_name="factory_allowed_chunkers",
                setting_category="chunker",
                schema_name="chunkerName",
            ),
            "context_retriever": FactoryParams(
                factory_allowed_handler_name="factory_allowed_context_retrievers",
                setting_category="context_retriever",
                schema_name="contextRetrieverName",
            ),
            "vector_memory_handler": FactoryParams(
                factory_allowed_handler_name="factory_allowed_vector_databases",
                setting_category="vector_database",
                schema_name="vectorDatabaseName",
            ),
        }

    @property
    def _factory_services_params(self) -> Dict[str, FactoryParams]:
        return self._orchestrator_factory_services_params | self._bot_factory_services_params
