import json

import pytest
from pydantic import ValidationError
from timbal.types.content import ToolUseContent, content_factory


def test_basic_tool_use_content_validation() -> None:
    content = content_factory({"type": "tool_use", "id": "123", "name": "tool_name", "input": {"city": "London"}})
    assert isinstance(content, ToolUseContent)
    assert content.id == "123"
    assert content.name == "tool_name"
    assert content.input == {"city": "London"}
    assert content.type == "tool_use"

    with pytest.raises(ValidationError):
        content_factory({"type": "tool_use", "id": "123", "name": "get_weather", "input": "not a dict"})

def test_tool_use_to_openai_chat_completions_input() -> None:
    tool_use_content = ToolUseContent(id="123",
            name="get_weather",
            input={"city": "London"})
    assert tool_use_content.to_openai_chat_completions_input() == {
            "id": "123",
            "type": "function",
            "function": {
                "arguments": '{"city": "London"}',
                "name": "get_weather"
            }
        }
        
def test_tool_use_to_anthropic_input() -> None:
    tool_use_content = ToolUseContent(id="123", name="get_weather", input={"city": "London"})
    assert tool_use_content.to_anthropic_input() == {"type": "tool_use", "id": "123", "name": "get_weather", "input": {"city": "London"}}


def test_a_reloaded_input_replays_byte_identical() -> None:
    """The model's key order and a store's (jsonb sorts by length) must send the same input."""
    as_emitted = ToolUseContent(
        id="1", name="find_code", input={"query": "Total", "path": "ui/src", "opts": {"glob": "*.ts", "context": 2}}
    )
    as_reloaded = ToolUseContent(
        id="1", name="find_code", input={"opts": {"glob": "*.ts", "context": 2}, "path": "ui/src", "query": "Total"}
    )
    for convert in ("to_anthropic_input", "to_openai_responses_input", "to_openai_chat_completions_input"):
        emitted, reloaded = getattr(as_emitted, convert)(), getattr(as_reloaded, convert)()
        assert json.dumps(emitted) == json.dumps(reloaded), convert
    assert list(as_emitted.to_anthropic_input()["input"]) == ["opts", "path", "query"]
    assert list(as_emitted.to_anthropic_input()["input"]["opts"]) == ["context", "glob"]
    # Memory keeps what the model sent; only the outbound copy is ordered.
    assert list(as_emitted.input) == ["query", "path", "opts"]
