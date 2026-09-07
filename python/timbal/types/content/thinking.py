from typing import Any, Literal

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .base import BaseContent


class ThinkingContent(BaseContent):
    """Thinking content type for chat messages.

    Two provider-specific carriers ride along with the (possibly empty) visible text:

    - ``signature`` — Anthropic's signed thinking block. Replayed verbatim through
      ``to_anthropic_input`` so multi-step tool loops keep their chain of thought.
    - ``id`` + ``encrypted_content`` — OpenAI Responses reasoning item (``rs_…``),
      populated when the request asked for ``include: ["reasoning.encrypted_content"]``.
      Replayed as a top-level ``{"type": "reasoning", …}`` input item. OpenAI's
      reasoning models expect these back on every subsequent call of a tool loop;
      without them the model loses the chain of thought between steps and tool
      calling degrades (up to emitting tool calls as plain text).
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None
    id: str | None = None
    encrypted_content: str | None = None

    @property
    def is_openai_reasoning_item(self) -> bool:
        """True when this block can be replayed as an OpenAI Responses ``reasoning`` item."""
        return bool(self.id and self.encrypted_content)

    @override
    def to_openai_responses_input(self, role: str, **kwargs: Any) -> dict[str, Any] | None:
        """See base class.

        Assistant thinking that carries an OpenAI reasoning id + encrypted payload becomes a
        ``reasoning`` item (the caller places it at the top level of ``input``, not inside a
        message). Anything else keeps the historical behaviour — the visible text as a text
        part — except that an empty text part is dropped rather than sent as ``""``.
        """
        if role == "assistant" and self.is_openai_reasoning_item:
            item: dict[str, Any] = {
                "type": "reasoning",
                "id": self.id,
                "encrypted_content": self.encrypted_content,
                "summary": [{"type": "summary_text", "text": self.thinking}] if self.thinking else [],
            }
            return item
        if not self.thinking:
            return None
        type = "output_text" if role == "assistant" else "input_text"
        return {"type": type, "text": self.thinking}

    @override
    def to_openai_chat_completions_input(self, **kwargs: Any) -> None:
        """Not a chat-completions content block by itself.

        Use ``Message.to_openai_chat_completions_input(reasoning_as=...)``:
        ``reasoning_content`` for Moonshot/Fireworks-style providers, omit otherwise.
        """
        return None

    @override
    def to_anthropic_input(self, **kwargs: Any) -> dict[str, Any] | None:
        """See base class.

        Anthropic verifies replayed thinking against its signature and rejects a block
        without one. Thinking that came from another provider (an OpenAI reasoning item,
        raw xAI reasoning text) has no Anthropic signature, so on a cross-provider
        fallback it is dropped rather than sent as an invalid block.
        """
        if not self.signature:
            return None
        return {"type": "thinking", "thinking": self.thinking, "signature": self.signature}
