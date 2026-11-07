from abc import abstractmethod, ABC
from typing import Type

from cat.log import log
from cat.services.factory.models import BaseFactoryConfigModel
from cat.services.ingestion.locks import IngestionLockManager, NoopLockManager
from cat.services.ingestion.pipeline import CoreSinglePassPipeline, IngestionPipeline
from cat.services.ingestion.profile import IngestionProfile
from cat.services.ingestion.recovery import IngestionRecovery, NoopRecovery
from cat.services.ingestion.state import CoreIngestionStateStore, IngestionStateStore
from cat.services.ingestion.tags import IngestionTagResolver, SettingsVersionTags


class BaseIngestionEngine(ABC):
    """Interface for the ingestion engine.

    The engine is the SEAM between the core and the plugins for the whole
    ingestion lifecycle: the core routes (file upload, URL, batch) resolve the
    configured engine and call ``ingest_file`` instead of calling
    ``rabbit_hole.ingest_file`` directly, so a plugin can fully replace the
    ingestion flow. The re-embed pass on embedder change is exposed via ``run``.

    ``profile`` is the bundle of the five ingestion parts for the resolved
    configuration, attached by the factory after construction. It is ``None``
    on an engine built directly (no behaviour change for such callers).
    """
    profile: IngestionProfile | None = None
    @abstractmethod
    async def run(self, lizard) -> bool:
        pass

    @abstractmethod
    async def ingest_file(
        self,
        cat,
        file,
        filename: str | None = None,
        metadata: dict | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ) -> None:
        pass


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


class BaseIngestionConfiguration(BaseFactoryConfigModel, ABC):
    @classmethod
    @abstractmethod
    def pyclass(cls) -> type:
        pass

    @classmethod
    def base_class(cls) -> type:
        return BaseIngestionEngine

    @classmethod
    def pipeline(cls) -> type[IngestionPipeline]:
        """The pipeline implementation for this configuration."""
        return CoreSinglePassPipeline

    @classmethod
    def state_store(cls) -> type[IngestionStateStore]:
        """The state-store implementation for this configuration."""
        return CoreIngestionStateStore

    @classmethod
    def recovery(cls) -> type[IngestionRecovery]:
        """The recovery implementation for this configuration."""
        return NoopRecovery

    @classmethod
    def locks(cls) -> type[IngestionLockManager]:
        """The lock-manager implementation for this configuration."""
        return NoopLockManager

    @classmethod
    def tags(cls) -> type[IngestionTagResolver]:
        """The tag-resolver implementation for this configuration."""
        return SettingsVersionTags

    @classmethod
    def profile(cls) -> IngestionProfile:
        """Instantiate the five parts into one per-configuration profile."""
        return IngestionProfile(
            pipeline=cls.pipeline()(),
            state_store=cls.state_store()(),
            recovery=cls.recovery()(),
            locks=cls.locks()(),
            tags=cls.tags()(),
        )

    @classmethod
    def get_from_config(cls, config) -> Type:
        """Build the engine, keeping the factory validation, then attach the profile.

        The pyclass/base_class validation is delegated unchanged to
        :meth:`BaseFactoryConfigModel.get_from_config`; only the profile
        attachment is added on top.
        """
        obj = super().get_from_config(config)
        obj.profile = cls.profile()
        return obj


class CoreIngestionConfiguration(BaseIngestionConfiguration):
    """Base configuration: the upstream-parity re-embed flow (core)."""
    @classmethod
    def pyclass(cls) -> Type:
        return CoreIngestionEngine

    @classmethod
    def pipeline(cls) -> type[IngestionPipeline]:
        return CoreSinglePassPipeline

    @classmethod
    def state_store(cls) -> type[IngestionStateStore]:
        return CoreIngestionStateStore

    @classmethod
    def recovery(cls) -> type[IngestionRecovery]:
        return NoopRecovery

    @classmethod
    def locks(cls) -> type[IngestionLockManager]:
        return NoopLockManager

    @classmethod
    def tags(cls) -> type[IngestionTagResolver]:
        return SettingsVersionTags
