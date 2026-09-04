"""Factory configuration for the efficient ingestion engine.

Registers ``EfficientIngestionConfiguration`` in the ``ingestion`` category
through the ``factory_allowed_ingestions`` hook, so the engine can be selected
(replaceable class pattern, like the embedders).
"""


from pydantic import model_validator

from cat.core_plugins.efficient_ingestion.reembed import EfficientIngestionEngine
from cat.services.factory.ingestion import BaseIngestionEngine
from cat.services.factory.models import BaseFactoryConfigModel


class EfficientIngestionConfiguration(BaseFactoryConfigModel):
    """Configuration of the efficient ingestion engine (category ``ingestion``)."""

    ingestion_max_concurrency: int = 5

    @model_validator(mode="before")
    @classmethod
    def _migrate_old_key(cls, data):
        """Accept the legacy field name ``reembed_max_concurrency``.

        The config now drives the whole ingestion lifecycle (not only the
        re-embed pass); a previously-saved ``reembed_max_concurrency`` value is
        mapped onto the new ``ingestion_max_concurrency`` so an existing DB
        entry keeps working with no manual re-save.
        """
        if isinstance(data, dict) and "reembed_max_concurrency" in data:
            data.setdefault("ingestion_max_concurrency", data.pop("reembed_max_concurrency"))
        return data

    @classmethod
    def pyclass(cls) -> type:
        return EfficientIngestionEngine

    @classmethod
    def base_class(cls) -> type:
        return BaseIngestionEngine
