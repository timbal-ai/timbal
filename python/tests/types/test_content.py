import base64
import pathlib

import pytest
from anthropic.types import (
    TextBlock as AnthropicTextBlock,
)
from anthropic.types import (
    ToolUseBlock as AnthropicToolUseBlock,
)
from json.decoder import JSONDecodeError
from openai.types.chat import (
    ChatCompletionMessageToolCall as OpenAIToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function as OpenAIFunction
from timbal.types import Content, File, FileContent, TextContent, ToolResultContent, ToolUseContent


def test_content_validation_with_text() -> None:
    content = Content.model_validate({"type": "text", "text": "Hello, World!"})
    assert isinstance(content, TextContent)
    assert content.text == "Hello, World!"
    assert content.type == "text"

    # text must be a string
    with pytest.raises(ValueError):
        Content.model_validate({"type": "text", "text": 123})


@pytest.mark.asyncio
async def test_text_to_openai_input() -> None:
    text_content = TextContent(text="Hello, World!")
    assert await text_content.to_openai_input() == {"type": "text", "text": "Hello, World!"}

@pytest.mark.asyncio
async def test_text_to_anthropic_input() -> None:
    text_content = TextContent(text="Hello, World!")
    assert await text_content.to_anthropic_input() == {"type": "text", "text": "Hello, World!"}


def test_text_from_anthropic_input() -> None:
    text_block = AnthropicTextBlock(
        type="text",
        text="Hello, World!"
    )
    text_content = Content.model_validate(text_block)
    assert isinstance(text_content, TextContent)
    assert text_content.text == "Hello, World!"
    assert text_content.type == "text"


def test_content_validation_with_file(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
    )
    test_file.write_bytes(png_content)
    content = Content.model_validate({"type": "file", "file": File.validate(str(test_file))})
    assert isinstance(content, FileContent)
    assert isinstance(content.file, File)
    assert content.type == "file"

    # file must be a File
    with pytest.raises(ValueError):
        Content.model_validate({"type": "file", "file": "not a file"})

@pytest.mark.asyncio
async def test_file_to_openai_input(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
    )
    test_file.write_bytes(png_content) 
    file = File.validate(str(test_file))
    file_content = FileContent(file=file)

    with open(test_file, "rb") as f:
        png_content = f.read()
        base64_content = base64.b64encode(png_content).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_content}"

    assert await file_content.to_openai_input() == {"type": "image_url", "image_url": {"url": data_url}}


@pytest.mark.asyncio
async def test_file_to_anthropic_input(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
    )
    test_file.write_bytes(png_content) 
    file = File.validate(str(test_file))
    file_content = FileContent(file=file)
    assert await file_content.to_anthropic_input() == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png_content).decode("utf-8")}}


def test_content_validation_with_tool_use() -> None:
    content = Content.model_validate({"type": "tool_use", "id": "123", "name": "tool_name", "input": {"city": "London"}})
    assert isinstance(content, ToolUseContent)
    assert content.id == "123"
    assert content.name == "tool_name"
    assert content.input == {"city": "London"}
    assert content.type == "tool_use"

    # input must be a dict
    with pytest.raises(JSONDecodeError):
        Content.model_validate({"type": "tool_use", "id": "123", "name": "get_weather", "input": "not a dict"})


@pytest.mark.asyncio
async def test_tool_use_to_openai_input() -> None:
    tool_use_content = ToolUseContent(id="123",
            name="get_weather",
            input={"city": "London"})
    assert await tool_use_content.to_openai_input() == {
            "id": "123",
            "type": "function",
            "function": {
                "arguments": '{"city": "London"}',
                "name": "get_weather"
            }
        }
    

def test_tool_use_from_openai_input() -> None:
    tool_use_call = OpenAIToolCall(
        id="call_62136354",
        type="function",
        function=OpenAIFunction(arguments='{"order_id":"order_12345"}', name='get_delivery_date')
    )
    tool_use_content = Content.model_validate(tool_use_call)
    assert isinstance(tool_use_content, ToolUseContent)

        
@pytest.mark.asyncio
async def test_tool_use_to_anthropic_input() -> None:
    tool_use_content = ToolUseContent(id="123", name="get_weather", input={"city": "London"})
    assert await tool_use_content.to_anthropic_input() == {"type": "tool_use", "id": "123", "name": "get_weather", "input": {"city": "London"}}


def test_tool_use_from_anthropic_input() -> None:
    tool_use_block = AnthropicToolUseBlock(
        id="123",
        type="tool_use",
        name="get_weather",
        input={"city": "London"}
    )
    tool_use_content = Content.model_validate(tool_use_block)
    assert isinstance(tool_use_content, ToolUseContent)


def test_content_validation_with_tool_result() -> None:
    content = Content.model_validate({"type": "tool_result", "id": "123", "content": [TextContent(text="Hello, World!")]})
    assert isinstance(content, ToolResultContent)
    assert content.id == "123"
    assert content.content == [TextContent(text="Hello, World!")]
    assert content.content[0].text == "Hello, World!"
    assert content.type == "tool_result"


@pytest.mark.asyncio
async def test_tool_result_to_openai_input() -> None:
    tool_result_content = ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])
    assert await tool_result_content.to_openai_input() == {"role": "tool", "content": [{"type": "text", "text": "Hello, World!"}], "tool_call_id": "123"}


@pytest.mark.asyncio
async def test_tool_result_to_anthropic_input() -> None:
    tool_result_content = ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])
    assert await tool_result_content.to_anthropic_input() == {"type": "tool_result", "tool_use_id": "123", "content": [{"type": "text", "text": "Hello, World!"}]}
