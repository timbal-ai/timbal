from timbal.types.content import ToolUseContent, content_factory


def test_basic_tool_use_content_validation() -> None:
    content = content_factory({"type": "tool_use", "id": "123", "name": "tool_name", "input": {"city": "London"}})
    assert isinstance(content, ToolUseContent)
    assert content.id == "123"
    assert content.name == "tool_name"
    assert content.input == {"city": "London"}
    assert content.type == "tool_use"

    # input that can't be parsed becomes empty dict
    content = content_factory({"type": "tool_use", "id": "123", "name": "get_weather", "input": "not a dict"})
    assert isinstance(content, ToolUseContent)
    assert content.input == {}

def test_tool_use_to_openai_input() -> None:
    tool_use_content = ToolUseContent(id="123",
            name="get_weather",
            input={"city": "London"})
    assert tool_use_content.to_openai_input() == {
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