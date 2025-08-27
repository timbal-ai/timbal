import base64
import pathlib

import pytest
from timbal.types import File, Message
from timbal.types.content import FileContent, TextContent, ToolResultContent, ToolUseContent


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


def test_message_text_to_openai_input() -> None:
    message = Message(role="assistant", content=[TextContent(text="Hello, World!")])
    assert message.to_openai_input() == {"role": "assistant", "content": [{"type": "text", "text": "Hello, World!"}]}


def test_message_text_to_anthropic_input() -> None:
    message = Message(role="assistant", content=[TextContent(text="Hello, World!")])
    assert message.to_anthropic_input() == {"role": "assistant", "content": [{"type": "text", "text": "Hello, World!"}]}


def test_message_file_validation(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
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
    

def test_message_file_to_openai_input(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
    )
    test_file.write_bytes(png_content)

    encoded_image = base64.b64encode(png_content).decode("utf-8")

    message = Message(role="assistant", content=[FileContent(file=File.validate(str(test_file)))])
    assert message.to_openai_input() == {
        "role": "assistant", 
        "content": [{
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
        }]
    }


def test_message_file_to_anthropic_input(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
    )
    test_file.write_bytes(png_content)
    message = Message(role="assistant", content=[FileContent(file=File.validate(str(test_file)))])
    assert message.to_anthropic_input() == {"role": "assistant", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png_content).decode("utf-8")}}]}


def test_message_tool_use_validation() -> None:
    message = Message(role="assistant", content=[ToolUseContent(id="123", name="get_weather", input={"city": "London"})])
    assert isinstance(message, Message)
    assert message.role == "assistant"
    assert message.content == [ToolUseContent(id="123", name="get_weather", input={"city": "London"})]

    # input that can't be parsed becomes empty dict
    message = Message.validate({"role": "assistant", "content": [{"type": "tool_use", "id": "123", "name": "get_weather", "input": "not a dict"}]})
    assert isinstance(message, Message)
    assert message.content[0].input == {}


def test_message_with_tool_use_to_openai_input() -> None:
    message = Message(role="assistant", content=[ToolUseContent(id="123", name="get_weather", input={"city": "London"})])
    assert message.to_openai_input() == {"role": "assistant", "tool_calls": [{"id": "123", "type": "function", "function": {"arguments": '{"city": "London"}', "name": "get_weather"}}]}


def test_message_with_tool_use_to_anthropic_input() -> None:
    message = Message(role="user", content=[ToolUseContent(id="123", name="get_weather", input={"city": "London"})])
    assert message.to_anthropic_input() == {"role": "user", "content": [{"type": "tool_use", "id": "123", "name": "get_weather", "input": {"city": "London"}}]}


def test_message_tool_result_validation() -> None:
    message = Message(role="assistant", content=[ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])])
    assert isinstance(message, Message)
    assert message.role == "assistant"
    assert message.content == [ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])]

    Message.validate({"role": "assistant", "content": [{"type": "tool_result", "id": "123", "content": 123}]})


def test_message_with_tool_result_to_openai_input() -> None:
    message = Message(role="user", content=[ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])])
    assert message.to_openai_input() == {"role": "tool", "tool_call_id": "123", "content": [{"type": "text", "text": "Hello, World!"}]}


def test_message_with_tool_result_to_anthropic_input() -> None:
    message = Message(role="user", content=[ToolResultContent(id="123", content=[TextContent(text="Hello, World!")])])
    assert message.to_anthropic_input() == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "123", "content": [{"type": "text", "text": "Hello, World!"}]}]}
