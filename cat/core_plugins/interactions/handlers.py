import time
import math
import weakref
from typing import List, Dict, Any
import tiktoken
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs.llm_result import LLMResult

from cat import log
from cat.core_plugins.interactions.models import LLMModelInteraction

# Registry of the strays of the running turns (the handlers are deep-copied by the hooks, the strays must not be):
# an entry goes away with its stray
_stray_registry: "weakref.WeakValueDictionary[int, Any]" = weakref.WeakValueDictionary()


class ModelInteractionHandler(BaseCallbackHandler):
    """
    Langchain callback handler for tracking model interactions: every LLM call made with the callbacks of a turn is
    recorded in the working memory of the turn (an agent may call the LLM several times).
    """
    def __init__(self, source: str):
        """
        Args:
            source: Source of the model interaction
        """
        self.source = source
        # Store the stray ID to survive serialization
        self.stray_id = None
        # the calls started and not ended yet, by run id
        self._running: Dict[Any, LLMModelInteraction] = {}

    def inject_stray_cat(self, stray: "StrayCat") -> None:
        """Inject the stray for registry lookup."""
        self.stray_id = id(stray)
        _stray_registry[self.stray_id] = stray

    def _start(self, run_id: Any, prompt: List[str], input_tokens: int) -> None:
        self._running[run_id] = LLMModelInteraction(
            source=self.source,
            prompt=prompt,
            reply="",
            input_tokens=input_tokens,
            output_tokens=0,
            ended_at=0,
        )

    def _count_tokens(self, text: str) -> int:
        # cl100k_base is the most common encoding for OpenAI models such as GPT-3.5, GPT-4 - what about other providers?
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))

    def _count_image_tokens(self, image_data: Dict[str, Any]) -> int:
        """
        Count tokens for image content based on OpenAI's vision pricing.

        Reference: https://platform.openai.com/docs/guides/vision/calculating-costs

        For images:
        - Base cost: 85 tokens
        - Each 512x512 tile: 170 tokens
        - Images are resized to fit within 2048x2048, maintaining aspect ratio
        - Then scaled to 512px on shortest side
        - Tiles are created from the scaled image

        Args:
            image_data: Dictionary containing image_url information

        Returns:
            Estimated token count for the image
        """
        try:
            # Default token count if we can't determine image size
            base_tokens = 85
            tile_tokens = 170

            # Try to extract image details if available
            image_url = image_data.get("image_url", {})
            if isinstance(image_url, str):
                image_url = {"url": image_url}

            detail = image_url.get("detail", "auto")

            # Low detail images always use base tokens
            if detail == "low":
                return base_tokens

            # For high detail, we need image dimensions
            # Since we don't have actual image dimensions here, we estimate
            # Based on "high" detail setting, typical estimate is ~765 tokens
            # This assumes an average image that would create ~4 tiles
            if detail in ("high", "auto"):
                # Conservative estimate: base + 4 tiles
                return base_tokens + (4 * tile_tokens)

            return base_tokens
        except Exception as e:
            log.warning(f"Error counting image tokens: {e}")
            # Return conservative estimate
            return 255  # base (85) + 1 tile (170)

    def _calculate_image_tiles(self, width: int, height: int) -> int:
        """
        Calculate the number of 512x512 tiles needed for an image.
        This is a more accurate implementation if image dimensions are available.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Number of tiles needed
        """
        # Step 1: Resize to fit within 2048x2048
        max_dim = 2048
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale shortest side to 512px
        min_dim = 512
        scale = min_dim / min(width, height)
        width = int(width * scale)
        height = int(height * scale)

        # Step 3: Count tiles needed
        tiles_width = math.ceil(width / 512)
        tiles_height = math.ceil(height / 512)

        return tiles_width * tiles_height

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs) -> None:
        """Track input tokens and prompt content."""
        input_tokens = 0
        input_prompt = []

        lc_prompt = messages[0] if isinstance(messages, list) else messages
        for m in lc_prompt:
            if isinstance(m.content, str):
                input_tokens += self._count_tokens(m.content)
                input_prompt.append(m.content)
                continue

            if isinstance(m.content, list):
                for c in m.content:
                    if isinstance(c, str):
                        input_tokens += self._count_tokens(c)
                        input_prompt.append(c)
                        continue

                    # Count text tokens
                    if c.get("type") == "text":
                        text_content = c.get("text", "")
                        input_tokens += self._count_tokens(text_content)
                        input_prompt.append(text_content)
                        continue

                    # Count image tokens
                    if c.get("type") == "image_url":
                        image_tokens = self._count_image_tokens(c)
                        input_tokens += image_tokens
                        input_prompt.append(f"[Image: ~{image_tokens} tokens]")
                        continue

                    log.warning(f"Could not count tokens for message type: {c.get('type', 'unknown')}")

        self._start(kwargs.get("run_id"), input_prompt, self._with_buffer(input_tokens))

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Track input tokens and prompt content of the (non chat) LLMs."""
        self._start(kwargs.get("run_id"), list(prompts), self._with_buffer(sum(self._count_tokens(p) for p in prompts)))

    @staticmethod
    def _with_buffer(tokens: int) -> int:
        # Store token count with small buffer for tokenization variations
        # Different models may tokenize slightly differently
        buffer_multiplier = 1.05  # 5% buffer instead of 20%
        return int(tokens * buffer_multiplier)

    @staticmethod
    def _text(content: Any) -> str:
        """The text of a reply: chat models may reply with a list of content blocks (e.g. text and tool calls)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block if isinstance(block, str) else str(block.get("text", ""))
                for block in content
                if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
            )
        return str(content or "")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Track output tokens and response content; the call is recorded in the working memory of the turn."""
        interaction = self._running.pop(kwargs.get("run_id"), None)
        if interaction is None:
            return

        generation = response.generations[0][0]

        if hasattr(generation, "message"):
            response_text = self._text(generation.message.content)
        else:
            response_text = generation.text

        interaction.output_tokens = self._count_tokens(response_text)
        interaction.reply = response_text
        interaction.ended_at = time.time()

        # Retrieve the correct stray from the registry
        stray = _stray_registry.get(self.stray_id) if self.stray_id is not None else None
        if stray is not None:
            stray.working_memory.model_interactions.add(interaction)

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Handle LLM errors: the failed call is not recorded."""
        log.error(f"LLM error in ModelInteractionHandler: {error}")
        self._running.pop(kwargs.get("run_id"), None)
