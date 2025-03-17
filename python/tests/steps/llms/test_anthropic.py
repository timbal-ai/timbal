import inspect
import json

import anthropic
import pytest
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextDelta,
    ToolUseBlock,
)
from timbal.steps.llms.gateway import handler
from timbal.types import Content, File, Message
from timbal.types.models import create_model_from_argspec


@pytest.mark.asyncio
async def test_basic_response():
    memory = {
        "role": "user", 
        "content": [{"type": "text", "text": "What is 2+2?"}]
    }

    argspec = inspect.getfullargspec(handler)
    model = create_model_from_argspec("TestModel", argspec)
    model_args = {
        "memory": [memory],
        "model": "claude-3-haiku-20240307"
    }
    model_args = model.model_validate(model_args)
    
    response_chunks = ""
    async for chunk in handler(**dict(model_args)):
        if isinstance(chunk, RawContentBlockDeltaEvent):
            if isinstance(chunk.delta, TextDelta):
                response_chunks += chunk.delta.text    
    
    assert response_chunks, "Response should not be empty"
    assert isinstance(response_chunks, str), "Response should be a string"
    assert "4" in response_chunks, "Response should contain the number 4"


@pytest.mark.asyncio
async def test_image_upload():
    url = "https://content.timbal.ai/assets/Three-Australian-Shepherd-puppies-sitting-in-a-field.jpg"
    file = File.validate(url)
    memory = {
        "role": "user", 
        "content": [{"type": "text", "text": "What is in this image?"}, {"type": "file", "file": file}]
    }

    argspec = inspect.getfullargspec(handler)
    model = create_model_from_argspec("TestModel", argspec)
    model_args = {
        "memory": [memory],
        "model": "claude-3-haiku-20240307"
    }
    model_args = model.model_validate(model_args)
    
    response_chunks = ""
    async for chunk in handler(**dict(model_args)):
        if isinstance(chunk, RawContentBlockDeltaEvent):
            if isinstance(chunk.delta, TextDelta):
                response_chunks += chunk.delta.text    

    assert response_chunks, "Response should not be empty"
    assert isinstance(response_chunks, str), "Response should be a string"
    assert "dog" in response_chunks.lower() or "pup" in response_chunks.lower(), \
        "Response should mention that there are dogs or puppies in the image"


@pytest.mark.asyncio
async def test_tools():
    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    memory = {
        "role": "user",
        "content": [{"type": "text", "text": "What's the weather in London?"}]
    }

    argspec = inspect.getfullargspec(handler)
    model = create_model_from_argspec("TestModel", argspec)
    model_args = {
        "memory": [memory],
        "system_prompt": "You are a helpful assistant.",
        "model": "claude-3-haiku-20240307",
        "tools": tools,
        "tool_choice": "auto"
    }
    model_args = model.model_validate(model_args)
    
    response_chunks = []
    async for chunk in handler(**dict(model_args)):
        # Start of a tool use or text message
        if isinstance(chunk, RawContentBlockStartEvent):
            if isinstance(chunk.content_block, ToolUseBlock):
                response_chunks.append({
                    "type": "tool_use",
                    "id": chunk.content_block.id,
                    "name": chunk.content_block.name,
                    "input": ""
                })
            else:
                response_chunks.append({
                    "type": "text",
                    "text": chunk.content_block.text
                })
        # Add content to the last message
        elif isinstance(chunk, RawContentBlockDeltaEvent):
            if isinstance(chunk.delta, InputJSONDelta):
                response_chunks[-1]["input"] += chunk.delta.partial_json
            elif isinstance(chunk.delta, TextDelta):
                response_chunks[-1]["text"] += chunk.delta.text
        # Finish a tool use or text message
        elif isinstance(chunk, RawContentBlockStopEvent):
            last_message = response_chunks[-1]
            if last_message["type"] == "tool_use":
                last_message["input"] = json.loads(last_message["input"])
            last_message = Content.model_validate(last_message)
            response_chunks[-1] = last_message

    assert response_chunks, "Response should not be empty"
    assert isinstance(response_chunks, list), "Response should be a list"
    weather_tool = next(
        (msg for msg in response_chunks if msg.type == "tool_use" and msg.name == "get_weather"),
        None
    )
    assert weather_tool is not None, "Weather tool call not found"
    assert isinstance(weather_tool.id, str) and weather_tool.id, "Tool call should have a non-empty ID"
    assert isinstance(weather_tool.input, dict), "Tool input should be a dictionary"
    assert weather_tool.input.get("location") == "London", "Tool input should contain 'London'"


@pytest.mark.asyncio
async def test_multiple_tools():
    tools=[
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "get_time",
            "description": "Get the current time in a given time zone",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The IANA time zone name, e.g. America/Los_Angeles"
                    }
                },
                "required": ["timezone"]
            }
        }
    ]

    memory = {
        "role": "user",
        "content": [{"type": "text", "text": "What is the weather like right now in New York? Also what time is it there?"}]
    }

    argspec = inspect.getfullargspec(handler)
    model = create_model_from_argspec("TestModel", argspec)
    model_args = {
        "memory": [memory],
        "system_prompt": "You are a helpful assistant. When a user asks multiple questions that require different tools, make sure to use all necessary tools to provide complete information.",
        "model": "claude-3-5-sonnet-20241022",
        "tools": tools,
        "tool_choice": {"type":"auto"},
    }
    model_args = model.model_validate(model_args)

    response_chunks = []
    async for chunk in handler(**dict(model_args)):
        # Start of a tool use or text message
        if isinstance(chunk, RawContentBlockStartEvent):
            if isinstance(chunk.content_block, ToolUseBlock):
                response_chunks.append({
                    "type": "tool_use",
                    "id": chunk.content_block.id,
                    "name": chunk.content_block.name,
                    "input": ""
                })
            else:
                response_chunks.append({
                    "type": "text",
                    "text": chunk.content_block.text
                })

        # Add content to the last message
        elif isinstance(chunk, RawContentBlockDeltaEvent):
            if isinstance(chunk.delta, InputJSONDelta):
                response_chunks[-1]["input"] += chunk.delta.partial_json
            elif isinstance(chunk.delta, TextDelta):
                response_chunks[-1]["text"] += chunk.delta.text
        
        # Finish a tool use or text message
        elif isinstance(chunk, RawContentBlockStopEvent):
            last_message = response_chunks[-1]
            if last_message["type"] == "tool_use":
                last_message["input"] = json.loads(last_message["input"])
            last_message = Content.model_validate(last_message)
            response_chunks[-1] = last_message

    assert response_chunks, "Response should not be empty"
    assert isinstance(response_chunks, list), "Response should be a list"
    assert any(
        msg.type == "tool_use" and msg.name == "get_weather" 
        for msg in response_chunks
    ), "Weather tool call not found"
    assert any(
        msg.type == "tool_use" and msg.name == "get_time" 
        for msg in response_chunks
    ), "Time tool call not found"


@pytest.mark.asyncio
async def test_response_format():
    json_schema = {
        "title": {"type": "string"},
        "author": {"type": "string"},
        "genre": {"type": "string"},
        "publication_year": {"type": "integer"},
        "summary": {"type": "string"}
    }  

    memory = {
        "role": "user",
        "content": [{"type": "text", "text": "Summarize The Eras Tour by Taylor Swift with fields for title, author, genre, publication year, and a short summary."}]
    }

    argspec = inspect.getfullargspec(handler)
    model = create_model_from_argspec("TestModel", argspec)
    model_args = {
        "memory": [memory],
        "system_prompt": "You are a helpful assistant.",
        "model": "claude-3-haiku-20240307",
        "json_schema": json_schema
    }
    model_args = model.model_validate(model_args)
    
    response_chunks = ""
    async for chunk in handler(**dict(model_args)):
        if isinstance(chunk, RawContentBlockDeltaEvent):
            if isinstance(chunk.delta, InputJSONDelta):
                response_chunks += chunk.delta.partial_json

    assert response_chunks, "Response should not be empty"
    assert isinstance(response_chunks, str), "Response should be a string"
    assert json.loads(response_chunks), "Response should match the JSON schema"


@pytest.mark.asyncio
async def test_invalid_model():
    memory = Message.validate({
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}]
    })
    argspec = inspect.getfullargspec(handler)
    model = create_model_from_argspec("TestModel", argspec)
    model_args = {
        "memory": [memory],
        "system_prompt": "You are a helpful assistant.",
        "model": "claude-invalid-model"
    }
    model_args = model.model_validate(model_args)

    with pytest.raises(anthropic.NotFoundError):
        async for _ in handler(**dict(model_args)):
            pass