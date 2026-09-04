from typing import List

from cat import (
    AgenticWorkflowConfig,
    AuthHandlerConfig,
    ChunkerSettings,
    ContextRetrieverSettings,
    EmbedderSettings,
    FileManagerConfig,
    LLMSettings,
    VectorDatabaseSettings,
    hook,
)
from cat.core_plugins.base_plugin.embedders.configs import EmbedderFakeConfig
from cat.core_plugins.base_plugin.file_managers.configs import LocalFileManagerConfig


@hook(priority=0)
def factory_allowed_ingestions(allowed, lizard):
    """Hook to extend the list of supported ingestion (re-embed) engines.

    The ``efficient_ingestion`` core plugin registers its
    ``EfficientIngestionConfiguration`` here. No-op default.
    """
    return allowed


@hook(priority=0)
def factory_allowed_llms(allowed: List[LLMSettings], cat) -> list:
    """
    Hook to extend support of LLMs.

    Args:
        allowed: List of LLMSettings classes
        cat: CheshireCat instance

    Returns:
        list of allowed LLMSettings classes for the allowed language models
    """
    return allowed


@hook(priority=0)
def before_llm_settings_update(payload: dict, language_model_name: str, cat) -> dict:
    """Hook to validate/enrich LLM settings before they are stored.

    Called by the core right before the LLM settings payload is upserted (PUT
    ``/llm/settings/{language_model_name}``). Plugins can use it to:

    * validate the payload against provider-specific constraints (e.g. that
      ``model`` is one of the models the provider actually serves);
    * enrich the stored settings with provider-derived metadata (capabilities,
      pricing) that the runtime needs (multimodal dispatch, context splitting,
      accounting).

    Args:
        payload: the settings dict about to be stored.
        language_model_name: the config class name being saved (e.g.
            ``LLMOpenRouterConfig``).
        cat: CheshireCat instance.

    Returns:
        The (possibly modified) payload that will be stored. If a plugin returns
        ``None`` the original payload is kept.
    """
    return payload


@hook(priority=0)
def factory_allowed_embedders(allowed: List[EmbedderSettings], lizard) -> List:
    """Hook to extend list of supported embedders.

    Args:
        allowed: List of EmbedderSettings classes
        lizard: BillTheLizard instance

    Returns:
        list of allowed EmbedderSettings classes for the allowed embedders
    """
    return allowed + [EmbedderFakeConfig]


@hook(priority=0)
def factory_allowed_auth_handlers(allowed: List[AuthHandlerConfig], cat) -> List:
    """Hook to extend list of supported auth handlers.

    Args:
        allowed: List of AuthHandlerConfig classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of AuthHandlerConfig classes for the allowed auth_handlers
    """
    return allowed


@hook(priority=0)
def factory_allowed_file_managers(allowed: List[FileManagerConfig], cat) -> List:
    """Hook to extend list of supported file managers.

    Args:
        allowed: List of FileManagerConfig classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of FileManagerConfig classes for the allowed file managers
    """
    return allowed + [LocalFileManagerConfig]


@hook(priority=0)
def factory_allowed_chunkers(allowed: List[ChunkerSettings], cat) -> List:
    """Hook to extend list of supported chunkers.

    Args:
        allowed: List of ChunkerSettings classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of ChunkerSettings classes for the allowed chunkers
    """
    return allowed


@hook(priority=0)
def factory_allowed_context_retrievers(allowed: List[ContextRetrieverSettings], cat) -> List:
    """Hook to extend list of supported context retrievers.

    Args:
        allowed: List of ContextRetrieverSettings classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of ContextRetrieverSettings classes for the allowed chunkers
    """
    return allowed


@hook(priority=0)
def factory_allowed_vector_databases(allowed: List[VectorDatabaseSettings], cat) -> List:
    """Hook to extend list of supported vector databases.

    Args:
        allowed: List of VectorDatabaseSettings classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of VectorDatabaseSettings classes for the allowed vector databases
    """
    return allowed


@hook(priority=0)
def factory_allowed_agentic_workflows(allowed: List[AgenticWorkflowConfig], cat) -> List:
    """Hook to extend list of supported agentic workflows.

    Args:
        allowed: List of AgenticWorkflowConfig classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of AgenticWorkflowConfig classes for the allowed agentic_workflows
    """
    return allowed
