from typing import Any, Dict, List, Optional

from cat.env import get_env, get_env_int

try:
    from contpress import ContextPress, TokenCounter
except Exception:
    # If contpress is not installed in the environment yet, define lightweight placeholders
    ContextPress = None  # type: ignore
    TokenCounter = None  # type: ignore


class ContextPressService:
    def __init__(
        self,
        model: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        reserve_output_tokens: Optional[int] = None,
        compression: Optional[str] = None,
    ) -> None:
        model = model or get_env("CAT_CP_MODEL")
        max_input_tokens = max_input_tokens or get_env_int("CAT_CP_MAX_INPUT_TOKENS")
        reserve_output_tokens = reserve_output_tokens or get_env_int("CAT_CP_RESERVE_OUTPUT_TOKENS")
        compression = compression or get_env("CAT_CP_COMPRESSION")

        self.model = model
        self.max_input_tokens = max_input_tokens
        self.reserve_output_tokens = reserve_output_tokens
        self.compression = compression

        if ContextPress is None:
            self._cp = None
            self._counter = None
        else:
            self._cp = ContextPress(
                model=self.model,
                max_input_tokens=self.max_input_tokens,
                reserve_output_tokens=self.reserve_output_tokens,
                compression=self.compression,
            )
            self._counter = TokenCounter(model=self.model)

    def optimize(
        self,
        task: str,
        context: str | List[str] | Dict[str, Any] | None = None,
        instructions: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        if self._cp is None:
            raise RuntimeError("ContextPress is not available. Install the 'contpress' package.")

        return self._cp.optimize(task=task, context=context, instructions=instructions or [], **kwargs)

    def count(self, text: str) -> int:
        if self._counter is None:
            raise RuntimeError("ContextPress TokenCounter is not available. Install the 'contpress' package.")
        return self._counter.count(text)

    def fits(self, text: str, budget: int) -> bool:
        if self._counter is None:
            raise RuntimeError("ContextPress TokenCounter is not available. Install the 'contpress' package.")
        return self._counter.fits(text, budget=budget)

    def trim(self, text: str, max_tokens: int) -> str:
        if self._counter is None:
            raise RuntimeError("ContextPress TokenCounter is not available. Install the 'contpress' package.")
        return self._counter.trim(text, max_tokens=max_tokens)

    def available(self) -> bool:
        return self._cp is not None and self._counter is not None

    def close(self) -> None:
        # ContextPress currently does not require clean shutdown, but keep method for future needs
        self._cp = None
        self._counter = None
