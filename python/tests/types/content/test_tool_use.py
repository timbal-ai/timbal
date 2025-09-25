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