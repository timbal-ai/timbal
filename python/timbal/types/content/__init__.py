import json
from ast import literal_eval
from typing import Any

import structlog

from ..file import File
from .base import BaseContent
from .file import FileContent
from .text import TextContent
from .tool_result import ToolResultContent
from .tool_use import ToolUseContent

logger = structlog.get_logger("timbal.types.content")


def _parse_tool_use_input(input: Any) -> dict[str, Any]:
    """Aux function to parse tool use input (output from LLMs) into python objects."""
    input_arguments = {}
    if isinstance(input, dict):
        input_arguments = input
    else:
        try:
            input_arguments = json.loads(input)
        except Exception:
            try:
                input_arguments = literal_eval(input)
            except Exception:
                logger.error(
                    "Both json.loads and literal_eval failed when parsing tool_use input", 
                    input=input,
                    exc_info=True
                )
    return input_arguments


def content_factory(value: Any) -> BaseContent:
    """Factory method to interpret any python object into a BaseContent object that can be sent to LLMs."""
    if isinstance(value, BaseContent):
        return value
    elif isinstance(value, str):
        return TextContent(text=value)
    elif isinstance(value, File):
        return FileContent(file=value)
    elif isinstance(value, dict):
        content_type = value.get("type", None)
        if content_type == "text":
            return TextContent(text=value.get("text"))
        elif content_type == "file":
            return FileContent(file=File.validate(value.get("file")))
        elif content_type == "tool_use":
            return ToolUseContent(
                id=value.get("id"), 
                name=value.get("name"), 
                input=_parse_tool_use_input(value.get("input")),
            )
        elif content_type == "tool_result":
            tool_result_content = value.get("content")
            if not isinstance(tool_result_content, list):
                tool_result_content = [tool_result_content]
            return ToolResultContent(
                id=value.get("id"), 
                content=[content_factory(item) for item in tool_result_content],
            )
    # By default try to convert whatever python object we have into a string.
    return TextContent(text=str(value))
