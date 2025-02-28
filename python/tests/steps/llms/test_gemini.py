import base64
import inspect
import json
import uuid

import openai
import pytest
import requests
from openai.types.chat import ChatCompletionChunk
from timbal.steps.llms.gateway import handler
from timbal.types.file import File
from timbal.types.message import Message
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
        "model": "gemini-2.0-flash-lite-preview-02-05"
    }
    model_args = model.model_validate(model_args)
    
    response_chunks = ""
    async for chunk in handler(**dict(model_args)): 
        if isinstance(chunk, ChatCompletionChunk):
            if chunk.choices and chunk.choices[0].delta.content:
                response_chunks += chunk.choices[0].delta.content or "" 

    assert response_chunks, "Response should not be empty"
    assert isinstance(response_chunks, str), "Response should be a string"
    assert "4" in response_chunks, "Response should contain the number 4"


@pytest.mark.asyncio
async def test_image_upload():
    url = "https://content.timbal.ai/assets/Three-Australian-Shepherd-puppies-sitting-in-a-field.jpg"
    response = requests.get(url)
    image_data = response.content
    base64_image = base64.b64encode(image_data).decode('utf-8')
    file = File.validate(f"data:image/jpeg;base64,{base64_image}")
    
    memory = {
        "role": "user", 
        "content": [{"type": "text", "text": "What is in this image?"}, {"type": "file", "file": file}]
    }

    argspec = inspect.getfullargspec(handler)
    model = create_model_from_argspec("TestModel", argspec)
    model_args = {
        "memory": [memory],
        "model": "gemini-2.0-flash-lite-preview-02-05"
    }
    model_args = model.model_validate(model_args)
    
    response_chunks = ""
    async for chunk in handler(**dict(model_args)): 
        if isinstance(chunk, ChatCompletionChunk):
            if chunk.choices and chunk.choices[0].delta.content:
                response_chunks += chunk.choices[0].delta.content or ""   

    assert response_chunks, "Response should not be empty"
    assert isinstance(response_chunks, str), "Response should be a string"
    assert "dog" in response_chunks.lower() or "pup" in response_chunks.lower(), \
        "Response should mention that there are dogs or puppies in the image"


@pytest.mark.asyncio
async def test_audio_upload():
    url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
    file = File.validate(url)
    
    memory = {
        "role": "user", 
        "content": [
            {"type": "text", "text": "What is in this recording?"}, 
            {"type": "file", "file": file}
        ]
    }

    argspec = inspect.getfullargspec(handler)
    model = create_model_from_argspec("TestModel", argspec)

    model_args = {
        "memory": [memory],
        "model": "gemini-2.0-flash-lite-preview-02-05"
    }
    model_args = model.model_validate(model_args)

    response_chunks = ""
    async for chunk in handler(**dict(model_args)): 
        if isinstance(chunk, ChatCompletionChunk):
            if chunk.choices and chunk.choices[0].delta.content:
                response_chunks += chunk.choices[0].delta.content or ""

    assert response_chunks, "Response should not be empty"
    assert isinstance(response_chunks, str), "Response should be a string"
    assert "sun" in response_chunks.lower(), "Response should mention the sun"


@pytest.mark.asyncio
async def test_tools():
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
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
    }]

    memory = {
        "role": "user",
        "content": [{"type": "text", "text": "What's the weather in London?"}]
    }

    argspec = inspect.getfullargspec(handler)
    model = create_model_from_argspec("TestModel", argspec)
    model_args = {
        "memory": [memory],
        "system_prompt": "You are a helpful assistant.",
        "model": "gemini-2.0-flash-lite-preview-02-05",
        "tools": tools,
        "tool_choice": {"type": "auto"}
    }
    model_args = model.model_validate(model_args)
    
    response_chunks = []
    async for chunk in handler(**dict(model_args)):
        if isinstance(chunk, ChatCompletionChunk):
            # Handle tool calls
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                tool_call = chunk.choices[0].delta.tool_calls[0]
                # If this is a new tool call, create a new entry
                if tool_call.id is not None:
                    generated_id = str(uuid.uuid4())
                    current_tool_call = {
                        "type": "tool_use",
                        "id": generated_id,
                        "name": tool_call.function.name,
                        "input": ""
                    }
                    response_chunks.append(current_tool_call)
                # Accumulate arguments if present
                if tool_call.function.arguments:
                    current_tool_call["input"] += tool_call.function.arguments
            # Handle regular text content
            elif chunk.choices and chunk.choices[0].delta.content:
                if not response_chunks or "text" not in response_chunks[-1]:
                    response_chunks.append({"type": "text", "text": ""})
                response_chunks[-1]["text"] += chunk.choices[0].delta.content

    assert response_chunks, "Response should not be empty"
    assert isinstance(response_chunks, list), "Response should be a list"
    weather_tool = next(
        (chunk for chunk in response_chunks 
        if chunk.get("type") == "tool_use" and chunk.get("name") == "get_weather"),
        None
    )
    assert weather_tool is not None, "Weather tool call not found"
    assert isinstance(weather_tool["id"], str) and weather_tool["id"], "Tool call should have a non-empty ID"
    assert isinstance(weather_tool["input"], str), "Tool input should be a string"
    assert "London" in weather_tool["input"], "Tool input should contain 'London'"


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
        "model": "gemini-2.0-flash-lite-preview-02-05",
        "json_schema": json_schema
    }
    model_args = model.model_validate(model_args)

    response_chunks = ""
    async for chunk in handler(**dict(model_args)):
        if isinstance(chunk, ChatCompletionChunk):
            if chunk.choices and chunk.choices[0].delta.content:
                response_chunks += chunk.choices[0].delta.content or ""  

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
        "model": "gemini-invalid-model"
    }
    model_args = model.model_validate(model_args)

    with pytest.raises(openai.NotFoundError):
        async for _ in handler(**dict(model_args)):
            pass
