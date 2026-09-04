from typing import Any, Dict, List, Literal, Type

from pydantic import BaseModel

from cat.db.cruds import settings as crud_settings
from cat.exceptions import CustomValidationException
from cat.log import log
from cat.looking_glass.mad_hatter.mad_hatter import MadHatter
from cat.routes.routes_utils import GetSettingResponse, GetSettingsResponse, mask_secret_values
from cat.services.factory.agentic_workflow import CoreAgenticWorkflowConfig
from cat.services.factory.auth_handler import CoreAuthConfig
from cat.services.factory.chunker import RecursiveTextChunkerSettings
from cat.services.factory.context_retriever import DefaultContextRetrieverSettings
from cat.services.factory.embedder import EmbedderDumbConfig
from cat.services.factory.file_manager import DummyFileManagerConfig
from cat.services.factory.ingestion import BaseIngestionConfiguration
from cat.services.factory.llm import LLMDefaultConfig
from cat.services.factory.models import BaseFactoryConfigModel
from cat.services.factory.vector_db import QdrantConfig
from cat.utils import SUFFIX_TO_CRYPT


class ServiceFactory:
    def __init__(
        self,
        agent_key: str,
        hook_manager: MadHatter,
        factory_allowed_handler_name: str,
        setting_category: Literal[
            "auth_handler",
            "chunker",
            "context_retriever",
            "embedder",
            "file_manager",
            "llm",
            "vector_database",
            "agentic_workflow",
            "ingestion",
        ],
        schema_name: str,
    ):
        self._agent_key = agent_key
        self._hook_manager = hook_manager
        self.factory_allowed_handler_name = factory_allowed_handler_name
        self.setting_category = setting_category
        self.default_config_class = self.default_config_classes[setting_category]
        self.schema_name = schema_name

    @property
    def default_config_classes(self) -> Dict[str, Type[BaseFactoryConfigModel]]:
        return {
            "agentic_workflow": CoreAgenticWorkflowConfig,
            "auth_handler": CoreAuthConfig,
            "chunker": RecursiveTextChunkerSettings,
            "context_retriever": DefaultContextRetrieverSettings,
            "embedder": EmbedderDumbConfig,
            "file_manager": DummyFileManagerConfig,
            "llm": LLMDefaultConfig,
            "vector_database": QdrantConfig,
            "ingestion": BaseIngestionConfiguration,
        }

    async def get_config_class_from_adapter(self, obj: Any) -> Type[BaseModel] | None:
        allowed_classes = await self._get_allowed_classes()
        return next(
            (config_class for config_class in allowed_classes if config_class.pyclass() is type(obj)),
            None
        )

    async def get_schemas(self) -> Dict:
        # schemas contain metadata to let any client know which fields are required to create the class.
        schemas = {}
        for config_class in await self._get_allowed_classes():
            schema = config_class.model_json_schema()
            # useful for clients in order to call the correct config endpoints
            schema[self.schema_name] = schema["title"]
            schemas[schema["title"]] = schema

        return schemas

    async def _get_factory_class(self, config_name: str):
        # get plugin file manager factory class
        classes = await self._get_allowed_classes()
        factory_class = next((cls for cls in classes if cls.__name__ == config_name), None)
        return factory_class

    async def get_from_config_name(self, config_name: str) -> Any:
        factory_class = await self._get_factory_class(config_name)
        if not factory_class:
            log.error(
                f"Class {config_name} not found in the list of allowed classes for setting "
                f"'{self.setting_category}' — falling back to DumbEmbedder with a different "
                f"embedding dimension."
            )
            return self._fallback_default()

        # get configuration and instantiate the finalized object by the factory
        selected_config = await crud_settings.get_setting_by_name(self._agent_key, config_name)
        try:
            obj = factory_class.get_from_config(selected_config["value"])  # type: ignore[index]
        except Exception as e:
            # Surface the real reason instead of silently replacing the configured embedder
            # with the Dumb one (which emits vectors of a different dimension than the
            # configured model, breaking Qdrant upserts with confusing dim-mismatch errors).
            log.error(
                f"Failed to instantiate {config_name} for agent '{self._agent_key}' "
                f"(category '{self.setting_category}'): {e!r}. "
                f"Falling back to the category default '{self.default_config_class.__name__}'. "
                f"For embedders this means a DIFFERENT embedding dimension and will corrupt "
                f"the agent's vector collections.",
            )
            return self._fallback_default()

        await self._set_agent_id(obj)
        return obj

    def _fallback_default(self):
        obj = self.default_config_class.get_from_config(self.default_config)
        if hasattr(obj, "agent_id"):
            obj.agent_id = self._agent_key
        return obj

    async def _set_agent_id(self, obj: Any) -> None:
        if hasattr(obj, "agent_id"):
            obj.agent_id = self._agent_key

    async def _get_allowed_classes(self) -> List[Type[BaseFactoryConfigModel]]:
        return await self._hook_manager.execute_hook(
            self.factory_allowed_handler_name, [self.default_config_class], caller=None
        )

    async def upsert_service(self, service_name: str, payload: Dict) -> Dict:
        from cat.services.service_updater import ServiceUpdater

        schemas = await self.get_schemas()

        allowed_configurations = list(schemas.keys())
        if service_name not in allowed_configurations:
            raise CustomValidationException(
                f"{service_name} not supported. Must be one of {allowed_configurations}")

        updater_service = ServiceUpdater(self)
        result = await updater_service.replace_service(service_name, payload)

        return result

    async def get_factory_settings(self, reveal: bool = True) -> GetSettingsResponse:
        async def get_class_value(class_name: str) -> Dict[str, Any]:
            if class_name != saved_settings["name"]:
                return {}
            factory_class = await self._get_factory_class(class_name)
            return factory_class.parse_config(saved_settings["value"])

        saved_settings = await crud_settings.get_settings_by_category(self._agent_key, self.setting_category)
        schemas = await self.get_schemas()

        settings = [GetSettingResponse(
            name=class_name,
            value=mask_secret_values(await get_class_value(class_name), reveal),
            scheme=scheme
        ) for class_name, scheme in schemas.items()]

        return GetSettingsResponse(settings=settings, selected_configuration=saved_settings["name"])  # type: ignore[index]

    async def get_factory_setting(self, configuration_name: str, reveal: bool = True) -> GetSettingResponse:
        schemas = await self.get_schemas()

        allowed_configurations = list(schemas.keys())
        if configuration_name not in allowed_configurations:
            raise CustomValidationException(
                f"{configuration_name} not supported. Must be one of {allowed_configurations}")

        factory_class = await self._get_factory_class(configuration_name)

        setting = await crud_settings.get_setting_by_name(self._agent_key, configuration_name)
        setting = {} if setting is None else factory_class.parse_config(setting["value"])
        setting = mask_secret_values(setting, reveal)

        scheme = schemas[configuration_name]

        return GetSettingResponse(name=configuration_name, value=setting, scheme=scheme)

    @property
    def default_config(self) -> Dict:
        return {
            k: (
                self.default_config_class.crypto.encrypt(v.default)
                if isinstance(v.default, str) and any(suffix in k for suffix in SUFFIX_TO_CRYPT) and v.default
                else v.default
            )
            for k, v in self.default_config_class.model_fields.items()
        }

    @property
    def agent_key(self):
        return self._agent_key

    @property
    def hook_manager(self):
        """The MadHatter instance used to execute hooks (e.g. by ServiceUpdater)."""
        return self._hook_manager
