import json
import time
from collections.abc import Sequence
from typing import Dict, Any, List
from uuid import UUID
from langchain_core.agents import AgentAction
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

from cat.env import get_env_bool
from cat.services.notifier import NotifierService
from cat.utils import colored_text


class ToolMessage(BaseModel):
    tool: str
    input: str
    output: str | None = None
    duration: float | None = None


class ThinkingMessage(BaseModel):
    """Represents a single LLM thinking/reasoning chunk sent over WebSocket.

    Attributes
    ----------
    content:
        The partial or full thinking text produced by the model.
    step:
        Monotonically-increasing counter that groups tokens belonging to the
        same thinking block (resets to 0 at every new LLM call).
    """

    content: str
    step: int = 0


class ActiveRun(BaseModel):
    name: str
    input: str
    start_time: float


class WebSocketCallbackManager(AsyncCallbackHandler):
    def __init__(self, notifier: NotifierService):
        """
        Args:
            notifier: NotifierService instance
        """
        self.notifier = notifier
        self.active_runs: Dict[str, ActiveRun] = {}
        # ── thinking-state tracking ───────────────────────────────────────────
        # Used for DeepSeek-R1 / Qwen-style <think>…</think> tag streaming.
        self._in_thinking_block: bool = False
        self._thinking_step: int = 0
        # True when thinking content was already forwarded via on_llm_new_token
        # (streaming mode). on_llm_end uses this to skip extraction and avoid
        # sending the same content twice.
        self._thinking_streamed: bool = False

    # ── LLM lifecycle ────────────────────────────────────────────────────────

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs,
    ):
        # Reset thinking state for every new LLM call.
        self._in_thinking_block = False
        self._thinking_step = 0
        self._thinking_streamed = False

        if get_env_bool("CAT_DEBUG"):
            lc_prompt = messages[0] if isinstance(messages, list) else messages
            print(colored_text("\n============== LLM INPUT ===============", "green"))
            for m in lc_prompt:
                print(m if isinstance(m, str) else m.model_dump())
            print(colored_text("========================================", "green"))

    async def on_llm_new_token(self, token: str, *, chunk=None, **kwargs) -> None:
        """Route each streamed token to the correct WebSocket channel.

        Three strategies are applied in order:

        1. **Anthropic extended thinking** – detected via ``chunk.additional_kwargs
           ["thinking_blocks"]``.  The full thinking text is extracted from the
           chunk and sent as an ``llm_thinking`` WebSocket message.
        2. **DeepSeek-R1 / Qwen ``<think>`` tags** – the token stream is parsed
           for ``<think>`` / ``</think>`` delimiters; thinking content is sent as
           ``llm_thinking`` while normal content continues as ``chat_token``.
        3. **Regular token** – forwarded as ``chat_token`` unchanged.
        """
        # 1 ── Anthropic extended thinking (chunk carries thinking_blocks) ────
        if chunk is not None and hasattr(chunk, "additional_kwargs"):
            thinking_blocks = chunk.additional_kwargs.get("thinking_blocks", [])
            if thinking_blocks:
                for block in thinking_blocks:
                    thinking_text = block.get("thinking") or block.get("text", "")
                    if not thinking_text:
                        continue
                    msg = ThinkingMessage(content=thinking_text, step=self._thinking_step)
                    await self.notifier.send_llm_thinking(msg.model_dump_json())
                    self._thinking_streamed = True
                # Do NOT also emit a chat_token for this chunk.
                return

        # 2 ── DeepSeek / <think> tag style ───────────────────────────────────
        if "<think>" in token or self._in_thinking_block:
            await self._handle_think_tagged_token(token)
            return

        # 3 ── Regular chat token ─────────────────────────────────────────────
        await self.notifier.send_chat_token(token)

    async def _handle_think_tagged_token(self, token: str) -> None:
        """Parse a token that may begin, continue, or end a ``<think>`` block.

        This handles all edge-cases where a single token may contain both
        thinking and non-thinking content (e.g. ``"text<think>reasoning"``).
        """
        # Opening tag encountered while not yet inside a block
        if "<think>" in token and not self._in_thinking_block:
            self._in_thinking_block = True
            self._thinking_step += 1
            pre, _, after_open = token.partition("<think>")
            if pre:
                await self.notifier.send_chat_token(pre)
            token = after_open  # continue processing the remainder

        # Closing tag encountered while inside a block
        if "</think>" in token and self._in_thinking_block:
            self._in_thinking_block = False
            thinking_part, _, remainder = token.partition("</think>")
            if thinking_part:
                msg = ThinkingMessage(content=thinking_part, step=self._thinking_step)
                await self.notifier.send_llm_thinking(msg.model_dump_json())
                self._thinking_streamed = True
            if remainder:
                await self.notifier.send_chat_token(remainder)
            return

        if token:
            # Still inside a thinking block
            if not self._in_thinking_block:
                await self.notifier.send_chat_token(token)
                return

            msg = ThinkingMessage(content=token, step=self._thinking_step)
            await self.notifier.send_llm_thinking(msg.model_dump_json())
            self._thinking_streamed = True

    async def on_llm_end(self, response: LLMResult, **kwargs):
        """Extract and forward thinking content from non-streaming responses.

        Skipped entirely if thinking was already forwarded token-by-token during
        streaming (``_thinking_streamed`` flag), to avoid sending the same content
        twice.

        Handles:
        * **Anthropic extended thinking** – ``message.additional_kwargs["thinking_blocks"]``
        * **OpenAI o-series reasoning** – ``message.additional_kwargs["reasoning_content"]``
        """
        if not self._thinking_streamed:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg_obj = getattr(gen, "message", None)
                    if msg_obj is None or not hasattr(msg_obj, "additional_kwargs"):
                        continue
                    extra = msg_obj.additional_kwargs

                    # Anthropic extended thinking blocks
                    for block in extra.get("thinking_blocks", []):
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            t_msg = ThinkingMessage(content=thinking_text, step=self._thinking_step)
                            await self.notifier.send_llm_thinking(t_msg.model_dump_json())

                    # OpenAI o-series reasoning_content
                    reasoning = extra.get("reasoning_content", "")
                    if reasoning:
                        t_msg = ThinkingMessage(content=reasoning, step=self._thinking_step)
                        await self.notifier.send_llm_thinking(t_msg.model_dump_json())

        if get_env_bool("CAT_DEBUG"):
            print(colored_text("\n============== LLM OUTPUT ===============", "blue"))
            print(response)
            print(colored_text("========================================", "blue"))

    # ── Tool lifecycle ───────────────────────────────────────────────────────

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "Unknown Tool")
        self.active_runs[str(run_id)] = ActiveRun(name=tool_name, input=input_str, start_time=time.perf_counter())

        notification = ToolMessage(tool=tool_name, input=input_str)
        await self.notifier.send_tool_message(notification.model_dump_json())

    async def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        run_data = self.active_runs.pop(str(run_id), None)
        if run_data:
            tool_name = run_data.name
            duration = time.perf_counter() - run_data.start_time

            notification = ToolMessage(tool=tool_name, input=run_data.input, output=output, duration=duration)
            await self.notifier.send_tool_message(notification.model_dump_json())

    async def on_agent_action(self, action: AgentAction, **kwargs):
        await self.notifier.send_thought_message(action.log)

    async def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, **kwargs):
        await self.notifier.send_context(json.dumps([doc.page_content for doc in documents]))
