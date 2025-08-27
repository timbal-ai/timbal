from timbal.types.content import TextContent, ToolResultContent, content_factory


def test_basic_tool_result_content_validation() -> None:
    content = content_factory({"type": "tool_result", "id": "123", "content": [TextContent(text="Hello, World!")]})
    assert isinstance(content, ToolResultContent)
    assert content.id == "123"
    assert content.content == [TextContent(text="Hello, World!")]
    assert content.content[0].text == "Hello, World!"
    assert content.type == "tool_result"

def test_tool_result_to_openai_input() -> None:
    tool_result_content = ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])
    assert tool_result_content.to_openai_input() == {"role": "tool", "content": [{"type": "text", "text": "Hello, World!"}], "tool_call_id": "123"}

def test_tool_result_to_anthropic_input() -> None:
    tool_result_content = ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])
    assert tool_result_content.to_anthropic_input() == {"type": "tool_result", "tool_use_id": "123", "content": [{"type": "text", "text": "Hello, World!"}]}
