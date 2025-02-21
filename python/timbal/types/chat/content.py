"""
Defines the content types for chat messages in Timbal.

Types:
- TextContent: Plain text messages
- FileContent: File attachments
- ToolUseContent: Request to use a specific tool
- ToolResultContent: Result returned by a tool

All message content must be an instance of one of these types.

Usage:

1. Validating a content:
   >>> text_content = Content.model_validate(text_block)
   >>> tool_use_content = Content.model_validate(tool_use_block)

2. Converting a content to the input format required by OpenAI and Anthropic:
   >>> file_content = FileContent(file=File.validate(data_url))
   >>> file_content.to_openai_input()
   >>> file_content.to_anthropic_input()

"""

import base64
import json
import mimetypes
from ast import literal_eval
from typing import Any, Literal

from anthropic.types import (
    TextBlock as AnthropicTextBlock,
)
from anthropic.types import (
    ToolUseBlock as AnthropicToolUseBlock,
)
from openai.types.chat import (
    ChatCompletionMessageToolCall as OpenAIToolCall,
)
from pydantic import BaseModel

from ..file import File


class Content(BaseModel):
    """
    A class representing the content of a chat message.
    """

    type: Literal["text", "file", "tool_use", "tool_result"]

    @classmethod 
    def model_validate(cls, value: Any, *args: Any, **kwargs: Any) -> "Content":
        """Validate and convert input formats into a Content instance."""
        # Don't recurse if we're already dealing with a Content instance
        if isinstance(value, Content):
            return value
        
        # cls will be diferent from Content when we call model_validate on one of the subclasses
        if cls is not Content:
            return super().model_validate(value, *args, **kwargs)
        
        if isinstance(value, str):
            return TextContent(text=value)

        if isinstance(value, File):
            return FileContent(file=value)
        
        if isinstance(value, AnthropicTextBlock):
            return TextContent(text=value.text)

        if isinstance(value, AnthropicToolUseBlock):
            return ToolUseContent(
                id=value.id,
                name=value.name,
                input=value.input,
            )

        if isinstance(value, OpenAIToolCall):
            return ToolUseContent(
                id=value.id,
                name=value.function.name,
                input=literal_eval(value.function.arguments),
            )
        
        # TODO Review
        if isinstance(value, dict):
            content_type = value.get("type", None)

            if content_type == "text":
                return TextContent(text=value.get("text"))
            
            # Anthropic's file content type.
            if content_type == "file":
                return FileContent(file=File.validate(value.get("file")))
            
            # OpenAI's file content type.
            if content_type == "image_url":
                return FileContent(file=File.validate(value.get("image_url")['url']))
            
            if content_type == "input_audio":
                return FileContent(file=File.validate(value.get("input_audio")['data']))

            # Anthropic's tool use content.
            if content_type == "tool_use":
                input_value = value.get("input") 
                if isinstance(input_value, str): 
                    input_value = json.loads(input_value)
                return ToolUseContent(
                    id=value.get("id"), 
                    name=value.get("name"), 
                    input=input_value,
                )
            
            # OpenAI's tool use content.
            if content_type == "function":
                return ToolUseContent(
                    id=value.get("id"),  
                    name=value["function"]["name"],
                    input=literal_eval(value["function"]["arguments"]),
                )
            
            # Anthropic's tool result content.
            if content_type == "tool_result":
                tool_result_content = value.get("content", [])
                if not isinstance(tool_result_content, list):
                    tool_result_content = [tool_result_content]
                return ToolResultContent(
                    id=value.get("tool_use_id") or value.get("id"), 
                    content=[cls.model_validate(item) for item in tool_result_content],
                )
        
        raise ValueError(f"Invalid content: {value}")


class FileContent(Content):
    """
    This class represents a file content in a chat message. 
    It also provides methods to convert the file content to the input format required by OpenAI and Anthropic.
    """
    type: Literal["file"] = "file"
    file: File


    def to_openai_input(self) -> dict[str, Any]:
        """Convert the file content to the input format required by OpenAI."""
        # Get mime type. 
        # source schemes: bytes, local_path, data, url, s3
        if self.file.__source_scheme__ == "data":
            mime = self.file.__source__.split(";")[0].split(":")[1]
        elif self.file.__source_scheme__ == "bytes":
            raise ValueError("Cannot convert bytes-source file to OpenAI message content.")
        elif self.file.__source_scheme__ == "s3":
            raise NotImplementedError("Converting S3 images to OpenAI message content is not supported.")
        else:
            mime, _ = mimetypes.guess_type(str(self.file.__source__))
        
        # Ensure the file pointer is at the start of the file if we need to read it.
        current_position = self.file.tell()
        if current_position != 0:
            self.file.seek(0)

        if mime and mime.startswith("image/"):
            if self.file.__source_scheme__ == "url": 
                url = str(self.file)
            else:
                base64_data = base64.b64encode(self.file.read()).decode("utf-8")
                url = f"data:{mime};base64,{base64_data}"
            return {"type": "image_url", "image_url": {"url": url}}

        elif mime and mime.startswith("audio/"):
            if self.file.__source_scheme__ == "data":
                base64_data = self.file.__source__.split(",", 1)[1]
            else:
                base64_data = base64.b64encode(self.file.read()).decode("utf-8")

            if "mp3" not in mime and "wav" not in mime:
                raise ValueError(f"Unsupported audio format: {mime}. Must be one of: mp3, wav")

            return {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_data, 
                    "format": "wav" if "wav" in mime else "mp3",
                },
            }

        raise ValueError(f"Unsupported file type: {mime}. Must be an image or audio (wav/mp3) file.")


    def to_anthropic_input(self) -> dict[str, Any]:
        """Convert the file content to the input format required by Anthropic."""
        # Get mime type
        if self.file.__source_scheme__ == "data":
            mime = self.file.__source__.split(";")[0].split(":")[1]
        else:
            mime, _ = mimetypes.guess_type(str(self.file.__source__))

        if mime not in ["image/png", "image/jpeg", "image/webp", "image/gif"]:
            raise ValueError(f"Unsupported image format: {mime}. Must be one of: png, jpeg, webp, gif")

        # Get base64 data
        current_position = self.file.tell()
        if current_position != 0:
            self.file.seek(0)

        if self.file.__source_scheme__ == "data":
            # Extract base64 data after the comma for data URLs
            base64_data = self.file.__source__.split(",", 1)[1]
        else:
            # Convert file content to base64
            base64_data = base64.b64encode(self.file.read()).decode("utf-8")

        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": base64_data}}


class TextContent(Content):
    """
    This class represents a text content in a chat message.
    It also provides methods to convert the text content to the input format required by OpenAI and Anthropic.
    """

    type: Literal["text"] = "text"
    text: str 


    def to_openai_input(self) -> dict[str, Any]:
        """Convert the text content to the input format required by OpenAI."""
        return {
            "type": "text", 
            "text": self.text
        }


    def to_anthropic_input(self) -> dict[str, Any]:
        """Convert the text content to the input format required by Anthropic."""
        return {
            "type": "text", 
            "text": self.text
        }


class ToolUseContent(Content):
    """
    This class represents a tool use content in a chat message.
    It also provides methods to convert the tool use content to the input format required by OpenAI and Anthropic.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


    def to_openai_input(self) -> dict[str, Any]:
        """Convert the tool use content to the input format required by OpenAI."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "arguments": json.dumps(self.input),
                "name": self.name
            }
        }


    def to_anthropic_input(self) -> dict[str, Any]:
        """Convert the tool use content to the input format required by Anthropic."""
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


class ToolResultContent(Content):
    """
    This class represents a tool result content in a chat message.
    It also provides methods to convert the tool result content to the input format required by OpenAI and Anthropic.
    """

    type: Literal["tool_result"] = "tool_result"
    id: str
    content: list[TextContent | FileContent]


    def to_openai_input(self) -> dict[str, Any]:
        """Convert the tool result content to the input format required by OpenAI."""
        return {
            "role": "tool",
            "content": [item.to_openai_input() for item in self.content],
            "tool_call_id": self.id
        }

    def to_anthropic_input(self) -> dict[str, Any]:
        """Convert the tool result content to the input format required by Anthropic."""
        return {
            "type": "tool_result",
            "tool_use_id": self.id,
            "content": [item.to_anthropic_input() for item in self.content],
        }
