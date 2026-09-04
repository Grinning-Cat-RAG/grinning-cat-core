from typing import List, Type
from pydantic import ConfigDict
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cat import hook
from cat.services.factory.llm import LLMSettings


class FakeLLMConfig(LLMSettings):
    """Fake LLM for testing purposes."""
    responses: List[str] = ["I'm a fake LLM!"]

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Fake LLM",
            "description": "Fake LLM for testing purposes",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[FakeListChatModel]:
        return FakeListChatModel

@hook
def factory_allowed_llms(allowed, cat) -> List:
    allowed.append(FakeLLMConfig)
    return allowed
