import pathlib

import pytest
from pydantic import ValidationError
from timbal.types import File, Message
from timbal.types.content import FileContent, TextContent, ThinkingContent, ToolResultContent, ToolUseContent


def test_message_text_validation() -> None:
    message = Message(role="assistant", content=[TextContent(text="Hello, World!")])
    assert isinstance(message, Message)
    assert message.role == "assistant"
    assert len(message.content) == 1
    assert message.content[0].type == "text"
    assert message.content[0].text == "Hello, World!"
    assert message.content == [TextContent(text="Hello, World!")]

    # text must be a string
    with pytest.raises(ValueError):
        Message.validate({"role": "assistant", "content": [{"type": "text", "text": 123}]})


def test_message_non_envelope_dict_is_stringified() -> None:
    # A payload dict (no role/content keys, e.g. a tool's output wired into a prompt)
    # must be stringified whole, NOT silently dropped to the literal "None".
    payload = {"emails": [{"subject": "hi"}]}
    message = Message.validate(payload)
    assert message.role == "user"
    assert len(message.content) == 1
    assert message.content[0].type == "text"
    assert message.content[0].text == str(payload)
    assert message.content[0].text != "None"


def test_message_partial_envelope_dict_is_stringified() -> None:
    # A dict with only "role" (or only "content") is NOT a valid envelope, since every
    # real envelope carries both. Treat it as a payload and stringify it whole.
    role_only = {"role": "user"}
    message = Message.validate(role_only)
    assert message.role == "user"
    assert message.content == [TextContent(text=str(role_only))]

    content_only = {"content": "hi"}
    message = Message.validate(content_only)
    assert message.role == "user"
    assert message.content == [TextContent(text=str(content_only))]


def test_message_full_envelope_dict_is_parsed() -> None:
    # A dict with both "role" and "content" takes the envelope path.
    message = Message.validate({"role": "assistant", "content": "hi"})
    assert message.role == "assistant"
    assert message.content == [TextContent(text="hi")]


def test_message_metadata_roundtrip() -> None:
    from timbal.types.message import BACKGROUND_TASK_COMPLETED_KIND, RUNTIME_SOURCE

    message = Message(
        role="user",
        content=[TextContent(text="notice")],
        metadata={"source": RUNTIME_SOURCE, "kind": BACKGROUND_TASK_COMPLETED_KIND},
    )
    assert message.is_runtime()
    dumped = Message.serialize(message)
    assert dumped["metadata"] == {"source": "runtime", "kind": "background_task_completed"}
    restored = Message.validate(dumped)
    assert restored.is_runtime()
    assert restored.metadata == message.metadata
    # Provider wire formats omit metadata.
    assert "metadata" not in message.to_anthropic_input()
    assert "metadata" not in message.to_openai_chat_completions_input()


@pytest.mark.asyncio
async def test_message_metadata_survives_trace_dump() -> None:
    from timbal.utils import dump

    message = Message(
        role="user",
        content=[TextContent(text="notice")],
        metadata={"source": "runtime", "kind": "background_task_completed"},
    )
    dumped = await dump(message)
    assert dumped["metadata"] == {"source": "runtime", "kind": "background_task_completed"}
    ordinary = Message(role="user", content=[TextContent(text="hi")])
    ordinary_dump = await dump(ordinary)
    assert "metadata" not in ordinary_dump


def test_message_metadata_omitted_when_empty() -> None:
    message = Message(role="user", content=[TextContent(text="hi")])
    assert message.metadata == {}
    assert not message.is_runtime()
    assert "metadata" not in Message.serialize(message)
    with pytest.raises(TypeError):
        message.metadata["source"] = "runtime"


def test_message_text_to_openai_chat_completions_input() -> None:
    message = Message(role="assistant", content=[TextContent(text="Hello, World!")])
    assert message.to_openai_chat_completions_input() == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello, World!"}],
    }


def test_message_thinking_omitted_by_default() -> None:
    """Default path omits CoT (Vercel/LiteLLM) — do not dump thinking into visible content."""
    message = Message(
        role="assistant",
        content=[
            ThinkingContent(thinking="step 1"),
            ThinkingContent(thinking=" step 2"),
            TextContent(text="answer"),
        ],
    )
    assert message.to_openai_chat_completions_input() == {
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
    }


def test_message_thinking_to_openai_chat_completions_reasoning_content() -> None:
    message = Message(
        role="assistant",
        content=[
            ThinkingContent(thinking="step 1"),
            ThinkingContent(thinking=" step 2"),
            TextContent(text="answer"),
        ],
    )
    assert message.to_openai_chat_completions_input(reasoning_as="reasoning_content") == {
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
        "reasoning_content": "step 1 step 2",
    }


def test_thinking_content_not_a_chat_completions_content_block() -> None:
    assert ThinkingContent(thinking="secret plan").to_openai_chat_completions_input() is None


def test_message_text_to_anthropic_input() -> None:
    message = Message(role="assistant", content=[TextContent(text="Hello, World!")])
    assert message.to_anthropic_input() == {"role": "assistant", "content": [{"type": "text", "text": "Hello, World!"}]}


def test_message_to_anthropic_input_omits_empty_text() -> None:
    message = Message(
        role="assistant",
        content=[
            TextContent(text=""),
            TextContent(text="visible"),
        ],
    )
    payload = message.to_anthropic_input()
    assert payload["content"] == [{"type": "text", "text": "visible"}]


def test_message_file_validation(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        "89504e470d0a1a0a"  # PNG signature
    )
    test_file.write_bytes(png_content)
    file_content = FileContent(file=File.validate(str(test_file)))
    message = Message(role="assistant", content=[file_content])
    assert isinstance(message, Message)
    assert message.role == "assistant"
    assert isinstance(message.content[0], FileContent)

    # file must be a File
    with pytest.raises(ValueError):
        Message.validate({"role": "assistant", "content": [{"type": "file", "file": {"url": "not a file"}}]})


def test_message_tool_use_validation() -> None:
    message = Message(
        role="assistant", content=[ToolUseContent(id="123", name="get_weather", input={"city": "London"})]
    )
    assert isinstance(message, Message)
    assert message.role == "assistant"
    assert message.content == [ToolUseContent(id="123", name="get_weather", input={"city": "London"})]

    with pytest.raises(ValidationError):
        Message.validate(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "123", "name": "get_weather", "input": "not a dict"}],
            }
        )


def test_message_with_tool_use_to_openai_chat_completions_input() -> None:
    message = Message(
        role="assistant", content=[ToolUseContent(id="123", name="get_weather", input={"city": "London"})]
    )
    assert message.to_openai_chat_completions_input() == {
        "role": "assistant",
        "tool_calls": [
            {"id": "123", "type": "function", "function": {"arguments": '{"city": "London"}', "name": "get_weather"}}
        ],
    }


def test_message_with_tool_use_to_anthropic_input() -> None:
    message = Message(role="user", content=[ToolUseContent(id="123", name="get_weather", input={"city": "London"})])
    assert message.to_anthropic_input() == {
        "role": "user",
        "content": [{"type": "tool_use", "id": "123", "name": "get_weather", "input": {"city": "London"}}],
    }


def test_message_tool_result_validation() -> None:
    message = Message(
        role="assistant", content=[ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])]
    )
    assert isinstance(message, Message)
    assert message.role == "assistant"
    assert message.content == [ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])]

    Message.validate({"role": "assistant", "content": [{"type": "tool_result", "id": "123", "content": 123}]})


def test_message_with_tool_result_to_openai_chat_completions_input() -> None:
    message = Message(role="user", content=[ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])])
    assert message.to_openai_chat_completions_input() == {
        "role": "tool",
        "tool_call_id": "123",
        "content": [{"type": "text", "text": "Hello, World!"}],
    }


def test_message_with_tool_result_to_anthropic_input() -> None:
    message = Message(role="user", content=[ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])])
    assert message.to_anthropic_input() == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "123", "content": [{"type": "text", "text": "Hello, World!"}]}
        ],
    }


# --- Cross-provider replay of server-side tool blocks -----------------------
# Anthropic memory keeps server_tool_use (ToolUseContent) and
# web_search_tool_result (CustomContent) blocks. When replayed to another
# API shape (e.g. after a fallback-model switch) they must be skipped, not
# raise or leak Anthropic-only block types.


def _anthropic_server_tool_message() -> Message:
    from timbal.types.content import CustomContent

    return Message(
        role="assistant",
        content=[
            ToolUseContent(id="srvtoolu_1", name="web_search", input={"query": "weather"}, is_server_tool_use=True),
            CustomContent(value={"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}),
            TextContent(text="It is sunny. [[weather.com](https://weather.com)]"),
        ],
    )


def test_server_tool_blocks_skipped_in_openai_responses_input() -> None:
    inputs = _anthropic_server_tool_message().to_openai_responses_input()
    assert inputs == [
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "It is sunny. [[weather.com](https://weather.com)]"}],
        }
    ]


def test_server_tool_blocks_skipped_in_openai_chat_completions_input() -> None:
    result = _anthropic_server_tool_message().to_openai_chat_completions_input()
    assert result == {
        "role": "assistant",
        "content": [{"type": "text", "text": "It is sunny. [[weather.com](https://weather.com)]"}],
    }
    assert "tool_calls" not in result


def test_server_tool_only_message_drops_turn_for_openai() -> None:
    """An assistant turn with ONLY server-tool blocks (no text) must not become
    a bare {"role": "assistant"} dict — OpenAI rejects assistant messages with
    neither content nor tool_calls."""
    from timbal.types.content import CustomContent

    message = Message(
        role="assistant",
        content=[
            ToolUseContent(id="srvtoolu_1", name="web_search", input={"query": "x"}, is_server_tool_use=True),
            CustomContent(value={"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}),
        ],
    )
    assert message.to_openai_chat_completions_input() is None
    assert message.to_openai_responses_input() == []


def test_thinking_only_message_drops_turn_for_chat_completions() -> None:
    """Thinking-only turns serialized with reasoning_as="omit" have no payload either."""
    message = Message(role="assistant", content=[ThinkingContent(thinking="secret plan")])
    assert message.to_openai_chat_completions_input(reasoning_as="omit") is None
    # But with reasoning_content round-tripping the turn survives
    assert message.to_openai_chat_completions_input(reasoning_as="reasoning_content") == {
        "role": "assistant",
        "reasoning_content": "secret plan",
    }


def test_server_tool_blocks_preserved_in_anthropic_input() -> None:
    result = _anthropic_server_tool_message().to_anthropic_input()
    assert result["content"][0] == {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "weather"},
    }
    assert result["content"][1] == {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}
