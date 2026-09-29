import time
import uuid
from typing import Literal, List
from pydantic import BaseModel, Field, ConfigDict


class ModelInteraction(BaseModel):
    """
    Base class for interactions with models, capturing essential attributes common to all model interactions.

    Attributes
    ----------
    model_type: Literal["llm", "embedder"]
        The type of model involved in the interaction, either a large language model (LLM) or an embedder.
    source: str
        The source from which the interaction originates.
    prompt: List[str]
        The prompt or input provided to the model.
    input_tokens: int
        The number of input tokens processed by the model.
    started_at: float
        The timestamp when the interaction started. Defaults to the current time.
    id: str
        The identity of the interaction: two calls with the same prompt (e.g. of an agent), even at the same time, are
        two interactions.
    """
    model_type: Literal["llm", "embedder"]
    source: str
    prompt: List[str]
    input_tokens: int
    started_at: float = Field(default_factory=lambda: time.time())
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    model_config = ConfigDict(
        protected_namespaces=()
    )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModelInteraction):
            return NotImplemented
        return self.id == other.id
