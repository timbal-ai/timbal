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
    assert dumped["metadata"] is not message.metadata
    dumped["metadata"]["source"] = "forged"
    assert message.is_runtime()
    restored = Message.validate(dumped)
    assert not restored.is_runtime()
    assert message.metadata["source"] == RUNTIME_SOURCE
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
    assert dumped["metadata"] is not message.metadata
    dumped["metadata"]["source"] = "forged"
    assert message.is_runtime()
    ordinary = Message(role="user", content=[TextContent(text="hi")])
    ordinary_dump = await dump(ordinary)
    assert "metadata" not in ordinary_dump


def test_message_metadata_omitted_when_empty() -> None:
    from copy import deepcopy

    message = Message(role="user", content=[TextContent(text="hi")])
    assert message.metadata == {}
    assert not message.is_runtime()
    assert "metadata" not in Message.serialize(message)
    with pytest.raises(TypeError):
        message.metadata["source"] = "runtime"
    # Tracing providers deepcopy live Message graphs — empty metadata must
    # not store a MappingProxyType on the instance.
    cloned = deepcopy(message)
    assert cloned.metadata == {}
    assert not cloned.is_runtime()
    tagged = Message(
        role="user",
        content=[TextContent(text="notice")],
        metadata={"source": "runtime", "kind": "background_task_completed"},
    )
    cloned_tagged = deepcopy(tagged)
    assert cloned_tagged.is_runtime()
    assert cloned_tagged.metadata == tagged.metadata
    assert cloned_tagged.metadata is not tagged.metadata


def test_message_collect_text() -> None:
    message = Message(
        role="assistant",
        content=[
            TextContent(text="Hello"),
            ToolUseContent(id="1", name="edit", input={}),
            TextContent(text="World"),
        ],
    )
    assert message.collect_text() == "Hello\n\nWorld"


def test_message_closing_text_is_trailing_contiguous_run() -> None:
    message = Message(
        role="assistant",
        content=[
            TextContent(text="narration before tools"),
            ToolUseContent(id="1", name="write", input={"path": "a.py"}),
            ToolUseContent(id="2", name="bash", input={"cmd": "ls"}),
            TextContent(text="Contract: /api/users"),
            TextContent(text="Done."),
        ],
    )
    assert message.closing_text() == "Contract: /api/users\n\nDone."
    assert message.collect_text() == "narration before tools\n\nContract: /api/users\n\nDone."


def test_message_closing_text_still_reports_when_turn_ends_on_tool_call() -> None:
    """Trailing tool_use (token limit / cancel) must not erase the closing prose."""
    message = Message(
        role="assistant",
        content=[
            TextContent(text="Wrote the route contract."),
            ToolUseContent(id="1", name="edit", input={"path": "routes.ts"}),
            ToolUseContent(id="2", name="bash", input={"cmd": "pytest"}),
        ],
    )
    assert message.closing_text() == "Wrote the route contract."


def test_message_closing_text_empty_without_trailing_text() -> None:
    message = Message(
        role="assistant",
        content=[ToolUseContent(id="1", name="edit", input={})],
    )
    assert message.closing_text() == ""


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


def test_openai_reasoning_item_is_top_level_and_precedes_its_function_call() -> None:
    """An assistant turn `reasoning → function_call` replays as two top-level items in that
    order — the shape OpenAI's reasoning models need to keep their chain of thought across a
    tool loop. Nothing is wrapped in a `message`."""
    message = Message(
        role="assistant",
        content=[
            ThinkingContent(thinking="", id="rs_1", encrypted_content="enc-1"),
            ToolUseContent(id="call_1", name="search", input={"q": "x"}),
        ],
    )
    items = message.to_openai_responses_input()
    assert [i["type"] for i in items] == ["reasoning", "function_call"]
    assert items[0] == {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1", "summary": []}
    assert items[1]["call_id"] == "call_1"


def test_openai_reasoning_item_with_text_keeps_text_in_a_message() -> None:
    message = Message(
        role="assistant",
        content=[
            ThinkingContent(thinking="thought", id="rs_1", encrypted_content="enc-1"),
            TextContent(text="Here you go."),
        ],
    )
    items = message.to_openai_responses_input()
    assert items[0]["type"] == "reasoning"
    assert items[0]["summary"] == [{"type": "summary_text", "text": "thought"}]
    assert items[1] == {"role": "assistant", "content": [{"type": "output_text", "text": "Here you go."}]}


def test_openai_reasoning_items_one_per_step_all_replayed() -> None:
    """Parallel tool calls: each reasoning item stays adjacent to the call it produced."""
    message = Message(
        role="assistant",
        content=[
            ThinkingContent(thinking="", id="rs_1", encrypted_content="enc-1"),
            ToolUseContent(id="call_1", name="a", input={}),
            ThinkingContent(thinking="", id="rs_2", encrypted_content="enc-2"),
            ToolUseContent(id="call_2", name="b", input={}),
        ],
    )
    items = message.to_openai_responses_input()
    assert [(i["type"], i.get("id") or i.get("call_id")) for i in items] == [
        ("reasoning", "rs_1"),
        ("function_call", "call_1"),
        ("reasoning", "rs_2"),
        ("function_call", "call_2"),
    ]


def test_legacy_thinking_without_payload_still_rides_inside_the_message() -> None:
    """No id/encrypted payload (xAI raw reasoning, pre-fix memory): historical behaviour."""
    message = Message(
        role="assistant",
        content=[ThinkingContent(thinking="raw"), TextContent(text="answer")],
    )
    assert message.to_openai_responses_input() == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "raw"}, {"type": "output_text", "text": "answer"}]}
    ]


def test_empty_legacy_thinking_does_not_produce_an_empty_text_part() -> None:
    message = Message(role="assistant", content=[ThinkingContent(thinking=""), TextContent(text="answer")])
    assert message.to_openai_responses_input() == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}
    ]


def test_reasoning_only_assistant_turn_replays_as_nothing() -> None:
    """A turn cut off after reasoning (max_tokens, or a leaked tool call with nothing
    recoverable): a trailing reasoning item is rejected by the API ("provided without
    its required following item"), so it is not sent — and neither is an empty message."""
    message = Message(role="assistant", content=[ThinkingContent(thinking="", id="rs_1", encrypted_content="enc-1")])
    assert message.to_openai_responses_input() == []


def test_trailing_reasoning_items_are_dropped_but_earlier_ones_kept() -> None:
    message = Message(
        role="assistant",
        content=[
            ThinkingContent(thinking="", id="rs_1", encrypted_content="enc-1"),
            ToolUseContent(id="call_1", name="search", input={}),
            ThinkingContent(thinking="", id="rs_2", encrypted_content="enc-2"),
            ThinkingContent(thinking="", id="rs_3", encrypted_content="enc-3"),
        ],
    )
    items = message.to_openai_responses_input()
    assert [(i["type"], i.get("id") or i.get("call_id")) for i in items] == [("reasoning", "rs_1"), ("function_call", "call_1")]


# --- assistant `phase` and wire order (GPT-5.4+ preambles) ----------------------------


def test_assistant_text_phase_rides_on_the_message_item() -> None:
    message = Message(role="assistant", content=[TextContent(text="Done.", phase="final_answer")])
    assert message.to_openai_responses_input() == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "Done."}], "phase": "final_answer"}
    ]


def test_user_text_never_carries_a_phase() -> None:
    message = Message(role="user", content=[TextContent(text="hi", phase="final_answer")])
    assert message.to_openai_responses_input() == [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]


def test_preamble_replays_before_the_function_call_it_introduced() -> None:
    """Wire order is content order: `message(commentary) → reasoning → function_call →
    message(final_answer)`, not "all function_calls, then one message with every text"."""
    message = Message(
        role="assistant",
        content=[
            TextContent(text="Checking the schema first.", phase="commentary"),
            ThinkingContent(thinking="", id="rs_1", encrypted_content="enc-1"),
            ToolUseContent(id="call_1", name="get_schema", input={}),
            TextContent(text="Here is the schema.", phase="final_answer"),
        ],
    )
    items = message.to_openai_responses_input()
    assert [i.get("type") or i.get("phase") for i in items] == ["commentary", "reasoning", "function_call", "final_answer"]
    assert items[0] == {"role": "assistant", "content": [{"type": "output_text", "text": "Checking the schema first."}], "phase": "commentary"}
    assert items[3] == {"role": "assistant", "content": [{"type": "output_text", "text": "Here is the schema."}], "phase": "final_answer"}


def test_adjacent_text_with_the_same_phase_shares_one_message() -> None:
    message = Message(
        role="assistant",
        content=[TextContent(text="a", phase="final_answer"), TextContent(text="b", phase="final_answer")],
    )
    assert message.to_openai_responses_input() == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "a"}, {"type": "output_text", "text": "b"}], "phase": "final_answer"}
    ]


def test_adjacent_text_with_different_phases_splits_into_two_messages() -> None:
    message = Message(
        role="assistant",
        content=[TextContent(text="one sec", phase="commentary"), TextContent(text="done", phase="final_answer")],
    )
    items = message.to_openai_responses_input()
    assert [i["phase"] for i in items] == ["commentary", "final_answer"]


def test_text_without_phase_replays_as_before() -> None:
    message = Message(role="assistant", content=[TextContent(text="plain"), ToolUseContent(id="c", name="t", input={})])
    items = message.to_openai_responses_input()
    assert items[0] == {"role": "assistant", "content": [{"type": "output_text", "text": "plain"}]}
    assert items[1]["type"] == "function_call"


def test_server_tool_blocks_preserved_in_anthropic_input() -> None:
    result = _anthropic_server_tool_message().to_anthropic_input()
    assert result["content"][0] == {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "weather"},
    }
    assert result["content"][1] == {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}
