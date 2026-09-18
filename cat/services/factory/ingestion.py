from abc import abstractmethod, ABC
from typing import Type

from cat.log import log
from cat.services.factory.models import BaseFactoryConfigModel


class BaseIngestionEngine(ABC):
    """Interface for the ingestion engine.

    The engine is the SEAM between the core and the plugins for the whole
    ingestion lifecycle: the core routes (file upload, URL, batch) resolve the
    configured engine and call ``ingest_file`` instead of calling
    ``rabbit_hole.ingest_file`` directly, so a plugin can fully replace the
    ingestion flow. The re-embed pass on embedder change is exposed via ``run``.
    """
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


class CoreIngestionConfiguration(BaseIngestionConfiguration):
    """Base configuration: the upstream-parity re-embed flow (core)."""
    @classmethod
    def pyclass(cls) -> Type:
        return CoreIngestionEngine
