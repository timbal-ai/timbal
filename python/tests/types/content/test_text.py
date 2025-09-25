import pytest
from timbal.types.content import TextContent, content_factory


def test_basic_text_content_validation() -> None:
    content = content_factory({"type": "text", "text": "Hello, World!"})
    assert isinstance(content, TextContent)
    assert content.text == "Hello, World!"
    assert content.type == "text"

    # text must be a string
    with pytest.raises(ValueError):
        content_factory({"type": "text", "text": 123})

def test_text_to_openai_chat_completions_input() -> None:
    text_content = TextContent(text="Hello, World!")
    assert text_content.to_openai_chat_completions_input() == {"type": "text", "text": "Hello, World!"}

def test_text_to_anthropic_input() -> None:
    text_content = TextContent(text="Hello, World!")
    assert text_content.to_anthropic_input() == {"type": "text", "text": "Hello, World!"}
