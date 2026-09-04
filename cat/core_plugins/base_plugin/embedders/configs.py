from typing import Type
from langchain_core.embeddings import FakeEmbeddings
from pydantic import ConfigDict
from cat.services.factory.embedder import EmbedderSettings


class EmbedderFakeConfig(EmbedderSettings):
    size: int = 128

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Default Embedder",
            "description": "Configuration for default embedder. It just outputs random numbers.",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[FakeEmbeddings]:
        return FakeEmbeddings
