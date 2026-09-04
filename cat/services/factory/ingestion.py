"""Replaceable ingestion engine (factory pattern, fork-policy level 3).

The PUT /embedder/settings route must re-embed every agent's stored sources
with the new embedder. Which engine performs the pass is decided through the
same ServiceFactory mechanism used for embedders: the core provides the base
implementation and plugins can register more efficient ones via the
``factory_allowed_ingestions`` hook (``efficient_ingestion`` registers
``EfficientIngestionConfiguration``).

The engine selection follows the embedder pattern: the active configuration is
the SINGLE setting of the ``ingestion`` category (``name`` = configuration class
name, value = its settings) stored in the global ``system:agent`` list. There is
NO separate selection entry (that would make the category non-unique). Without a
saved entry the first non-core configuration contributed by a plugin wins;
otherwise the base one is used.
"""


from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.log import log
from cat.services.factory.models import BaseFactoryConfigModel


class BaseIngestionEngine:
    """Interface for the ingestion engine.

    The engine is the SEAM between the core and the plugins for the whole
    ingestion lifecycle: the core routes (file upload, URL, batch) resolve the
    configured engine and call ``ingest_file`` instead of calling
    ``rabbit_hole.ingest_file`` directly, so a plugin can fully replace the
    ingestion flow. The re-embed pass on embedder change is exposed via ``run``.
    """

    async def run(self, lizard) -> bool:
        raise NotImplementedError

    async def ingest_file(
        self,
        cat,
        file,
        filename: str | None = None,
        metadata: dict | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ) -> None:
        raise NotImplementedError


class CoreIngestionEngine(BaseIngestionEngine):
    """Base implementation: the upstream-parity flow (core methods).

    ``ingest_file`` wraps the original ``rabbit_hole.ingest_file`` (upstream
    behavior, unchanged); ``run`` is the upstream-parity re-embed pass. This is
    the engine that is resolved when no plugin overrides ingestion.
    """

    async def run(self, lizard) -> bool:
        try:
            await lizard.embed_all_in_cheshire_cats()
            return True
        except Exception as e:  # noqa: BLE001 - parity with the core error handling
            log.error(f"Error embedding all stored files: {e}")
            return False

    async def ingest_file(
        self,
        cat,
        file,
        filename: str | None = None,
        metadata: dict | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ) -> None:
        """Delegate to the original core ingestion flow (upstream parity)."""
        # local import: avoids a circular import at module load (RabbitHole
        # imports factory concerns through the lizard, not the other way around;
        # and the engine must stay import-safe for the core plugin manager)
        from cat.rabbit_hole import RabbitHole

        rabbit_hole = getattr(cat, "rabbit_hole", None) or RabbitHole()
        await rabbit_hole.ingest_file(
            cat=cat,
            file=file,
            filename=filename,
            metadata=metadata or {},
            store_file=store_file,
            content_type=content_type,
        )


class BaseIngestionConfiguration(BaseFactoryConfigModel):
    """Base configuration: the upstream-parity re-embed flow (core)."""

    @classmethod
    def pyclass(cls) -> type:
        return CoreIngestionEngine

    @classmethod
    def base_class(cls) -> type:
        return BaseIngestionEngine


def build_factory(lizard) -> "ServiceFactory":
    """ServiceFactory for the ``ingestion`` category (system scope)."""
    # local import: avoids the circular service_factory <-> factory.ingestion
    from cat.services.service_factory import ServiceFactory

    return ServiceFactory(
        agent_key=lizard.agent_key,
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_ingestions",
        setting_category="ingestion",
        schema_name="ingestionConfigurationName",
    )


async def _allowed_classes(lizard) -> list[type[BaseFactoryConfigModel]]:
    return await build_factory(lizard)._get_allowed_classes()


async def resolved_config_name(lizard) -> str:
    """Effective engine selection: the saved config of the ``ingestion`` category,
    else the first plugin class, else the base."""
    classes = await _allowed_classes(lizard)

    saved = await crud_settings.get_settings_by_category(DEFAULT_SYSTEM_KEY, "ingestion")
    if saved and isinstance(saved.get("name"), str):
        saved_name = saved["name"]
        if any(c.__name__ == saved_name for c in classes):
            return saved_name

    non_core = [c for c in classes if c is not BaseIngestionConfiguration]
    if non_core:
        return non_core[0].__name__
    return BaseIngestionConfiguration.__name__


async def resolve_ingestion_engine(lizard) -> BaseIngestionEngine | None:
    """Resolve the configured engine instance; None when nothing can be built.

    Unlike ``ServiceFactory.get_from_config_name`` (which falls back to the
    base with a loud error log when the entry was never saved), a missing
    entry here is normal: the configuration model defaults are used.
    """
    name = await resolved_config_name(lizard)
    sf = build_factory(lizard)
    try:
        config_class = await sf._get_factory_class(name)
        if config_class is None:
            return None
        entry = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, name)
        value = (entry or {}).get("value") or {}
        engine = config_class.get_from_config(value)
        await sf._set_agent_id(engine)
        return engine
    except Exception as e:  # noqa: BLE001
        log.error(f"Failed to instantiate ingestion engine '{name}': {e!r}")
        return None
