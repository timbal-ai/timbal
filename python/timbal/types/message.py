from typing import Any, Literal

from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    SerializationInfo,
    TypeAdapter,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
)
from pydantic_core import CoreSchema, core_schema

from .content import TextContent, ThinkingContent, ToolResultContent, ToolUseContent, content_factory

# Runtime-injected control messages (not human utterances). Consumers that paint
# chat transcripts should filter ``source == "runtime"`` rather than sniffing text.
RUNTIME_SOURCE = "runtime"
BACKGROUND_TASK_COMPLETED_KIND = "background_task_completed"


def _append_anthropic_content_blocks(
    target: list[dict[str, Any]], anthropic_input: dict[str, Any] | list[dict[str, Any]]
) -> None:
    """Append Anthropic content blocks, dropping empty text (API rejects text: '')."""
    if isinstance(anthropic_input, list):
        for block in anthropic_input:
            _append_anthropic_content_blocks(target, block)
        return
    if anthropic_input.get("type") == "text" and not anthropic_input.get("text"):
        return
    target.append(anthropic_input)


class Message:
    """A class representing a message in a conversation with an LLM.

    This class handles messages for both OpenAI and Anthropic formats, providing
    conversion methods between different message formats and validation logic.

    Attributes:
        role: The role of the message sender
        content: The content of the message, which can include text and tool interactions
        stop_reason: The reason the LLM stopped generating (only for assistant messages).
            Provider-specific values:
            - Anthropic: 'end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'pause_turn', 'refusal'
            - OpenAI Chat Completions: 'stop', 'length', 'tool_calls', 'content_filter', 'function_call'
            - OpenAI Responses: 'completed', 'max_output_tokens', 'content_filter', etc.
        metadata: Optional bag for runtime annotations (not sent to LLM providers).
            Background completion notices use ``{"source": "runtime", "kind": "..."}``.
    """

    __slots__ = ("role", "content", "stop_reason", "metadata", "_cached_dump", "_cached_dump_len")

    def __init__(
        self,
        role: Any,
        content: Any,
        stop_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a Message instance.

        Args:
            role: The role of the message sender
            content: The content of the message
            stop_reason: The reason the LLM stopped generating (optional, only for assistant messages)
            metadata: Optional runtime annotations (omitted from provider wire formats)
        """
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "stop_reason", stop_reason)
        object.__setattr__(self, "metadata", dict(metadata) if metadata else {})
        # Serialized-form cache (see timbal.utils.serialization). Long
        # conversations re-dump the same Message objects on every turn
        # (span input dump, memory dump, LLM input dump); messages are
        # immutable after construction except for in-place content appends
        # (e.g. synthesized server tool results), so the cache is validated
        # against len(content).
        object.__setattr__(self, "_cached_dump", None)
        object.__setattr__(self, "_cached_dump_len", -1)

    def __str__(self) -> str:
        parts = [f"role={self.role}", f"content={self.content}"]
        if self.stop_reason:
            parts.append(f"stop_reason={self.stop_reason}")
        if self.metadata:
            parts.append(f"metadata={self.metadata}")
        return f"Message({', '.join(parts)})"

    __repr__ = __str__

    def is_runtime(self) -> bool:
        """True when this message is a runtime control signal, not a human utterance."""
        return self.metadata.get("source") == RUNTIME_SOURCE

    def to_openai_responses_input(self) -> list[dict[str, Any]]:
        """Convert the message to OpenAI's responses api expected input format."""
        inputs = []
        message_content = []
        for content_item in self.content:
            if isinstance(content_item, ToolUseContent | ToolResultContent):
                item_input = content_item.to_openai_responses_input()
                if item_input is not None:
                    inputs.append(item_input)
            else:
                item_input = content_item.to_openai_responses_input(role=self.role)
                if item_input is not None:
                    message_content.append(item_input)
        if message_content:
            # Role here should only be 'user' or 'assistant'
            inputs.append({"role": self.role, "content": message_content})
        return inputs

    def to_openai_chat_completions_input(
        self,
        *,
        reasoning_as: Literal["omit", "reasoning_content"] = "omit",
    ) -> dict[str, Any] | None:
        """Convert the message to OpenAI's chat completions api expected input format.

        Args:
            reasoning_as: How to serialize ``ThinkingContent``:
                - ``"omit"`` (default): drop thinking from the outbound message. Matches
                  Vercel AI SDK / LiteLLM defaults — do not dump CoT into visible ``content``.
                - ``"reasoning_content"``: top-level ``reasoning_content`` string for
                  providers that round-trip it (Moonshot, Fireworks, DeepSeek-style, etc.).
        """
        role = self.role
        # OpenAI chat completions api expects tool calls to be in a separate field in the message.
        content = []
        tool_calls = []
        reasoning_parts: list[str] = []
        for content_item in self.content:
            if isinstance(content_item, ToolUseContent):
                tool_call = content_item.to_openai_chat_completions_input()
                if tool_call is not None:
                    tool_calls.append(tool_call)
            elif isinstance(content_item, ToolResultContent):
                return content_item.to_openai_chat_completions_input()
            elif isinstance(content_item, ThinkingContent):
                if content_item.thinking and reasoning_as == "reasoning_content":
                    reasoning_parts.append(content_item.thinking)
            else:
                openai_input = content_item.to_openai_chat_completions_input()
                if openai_input is None:
                    continue
                # Enabling splitting files into multiple pages or chunks.
                if isinstance(openai_input, list):
                    content.extend(openai_input)
                else:
                    content.append(openai_input)
        openai_input = {
            "role": role,
        }
        if len(content):
            openai_input["content"] = content
        if len(tool_calls):
            openai_input["tool_calls"] = tool_calls
        if reasoning_parts:
            openai_input["reasoning_content"] = "".join(reasoning_parts)
        if len(openai_input) == 1:
            # Every content item was skipped (e.g. server-side tool blocks on
            # cross-provider replay, or thinking-only turns with omit). A bare
            # role dict is invalid for the API — drop the turn entirely.
            return None
        return openai_input

    def to_anthropic_input(self) -> dict[str, Any]:
        """Convert the message to Anthropic's expected input format."""
        content: list[dict[str, Any]] = []
        for content_item in self.content:
            _append_anthropic_content_blocks(content, content_item.to_anthropic_input())
        # Anthropic doesn't accept the tool role. We must send this under the user role.
        role = self.role
        if role == "tool":
            role = "user"
        return {
            "role": role,
            "content": content,
        }

    async def load(self, client: Any = None) -> None:
        """Eagerly load all file content in this message concurrently.

        Args:
            client: Optional shared httpx.AsyncClient for connection reuse across messages.
        """
        import asyncio

        from .content import FileContent

        unloaded = [
            c.file
            for c in self.content
            if isinstance(c, FileContent) and object.__getattribute__(c.file, "__fileobj__") is None
        ]
        if not unloaded:
            return
        if client is not None:
            await asyncio.gather(*(f.load(client=client) for f in unloaded))
        else:
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as own_client:
                await asyncio.gather(*(f.load(client=own_client) for f in unloaded))

    def collect_text(self) -> str:
        """Collect all text from the message content."""
        message_text = ""
        for content in self.content:
            if isinstance(content, TextContent):
                message_text += content.text + "\n\n"
        return message_text.strip()

    def without_empty_text_blocks(self) -> "Message | None":
        """Drop ``TextContent`` with empty text (invalid for Anthropic; see DEBUG2)."""
        kept = [c for c in self.content if not (isinstance(c, TextContent) and c.text == "")]
        if not kept:
            return None
        if len(kept) == len(self.content):
            return self
        return Message(
            role=self.role,
            content=kept,
            stop_reason=self.stop_reason,
            metadata=self.metadata or None,
        )

    @classmethod
    def validate(cls, value: ValidatorFunctionWrapHandler, _info: dict | ValidationInfo | None = None) -> "Message":
        """Validate and convert inputs into a Message instance."""
        # Don't recurse if we're already dealing with a Message instance
        if isinstance(value, Message):
            return value
        if isinstance(value, dict):
            # Only treat a dict as a message envelope when it actually looks like one.
            # Every internal envelope producer (Message.serialize, collectors, agent)
            # emits both "role" and "content", so require both to avoid misclassifying
            # arbitrary payload dicts (which commonly carry a lone "content"/"role" key).
            if "role" in value and "content" in value:
                role = value.get("role", "user")
                content = value.get("content", None)
                stop_reason = value.get("stop_reason", None)
                metadata = value.get("metadata", None)
                if not isinstance(content, list):
                    content = [content]
                content = [content_factory(item) for item in content]
                if metadata is not None and not isinstance(metadata, dict):
                    metadata = None
                return cls(role=role, content=content, stop_reason=stop_reason, metadata=metadata)
            # Arbitrary payload (e.g. a tool's dict output wired straight into a prompt):
            # stringify the whole dict via content_factory instead of silently dropping
            # it to the literal "None" (which is what the envelope path used to produce).
            return cls(role="user", content=[content_factory(value)])
        return cls.validate(
            {
                "role": "user",
                "content": value,
            }
        )

    @classmethod
    def serialize(cls, value: Any, _info: dict | SerializationInfo | None = None) -> str:
        """Serialize a Message instance into a dictionary format."""
        # When creating a model with fields with File type that are nullable,
        # pydantic will pass None as the value to File.serialize.
        if value is None:
            return None
        if not isinstance(value, cls):
            raise ValueError("Cannot serialize a non-message object.")
        result = {
            "role": value.role,
            "content": value.content,
        }
        # Only include optional fields when set (avoid breaking existing serialization)
        if value.stop_reason is not None:
            result["stop_reason"] = value.stop_reason
        if value.metadata:
            result["metadata"] = value.metadata
        return result

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema: CoreSchema, _handler: GetJsonSchemaHandler) -> dict[str, Any]:
        """Defines what this type should be in openapi.json."""
        # https://docs.pydantic.dev/2.8/errors/usage_errors/#custom-json-schema
        json_schema = {
            "title": "TimbalMessage",  # This becomes the type name in most generators
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "tool", "system"],
                },
                "content": {
                    "type": "array",
                    "items": {},  # Keep it open/generic for now.
                },
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
        }
        return json_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: type[Any], _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Defines how to serialize this type in the core schema."""
        return core_schema.with_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.serialize,
                info_arg=True,
                when_used="always",
            ),
        )


message_model_schema = TypeAdapter(Message).json_schema()
