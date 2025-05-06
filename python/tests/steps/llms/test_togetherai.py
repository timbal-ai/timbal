import inspect
import json

import pytest
from openai.types.chat import ChatCompletionChunk
from timbal.steps.llms.gateway import handler
from timbal.types.file import File
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
        "model": "deepseek-ai/DeepSeek-V3",
    }
    model_args = model.model_validate(model_args)
        
    response_chunks = ""
    async for chunk in handler(**dict(model_args)):
       if chunk.choices and chunk.choices[0].delta.content:
           response_chunks += chunk.choices[0].delta.content or ""

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
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
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
async def test_tools():
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a location",
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
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "tools": tools,
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
                    current_tool_call = {
                        "type": "tool_use",
                        "id": tool_call.id,
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
        (chunk for chunk in response_chunks if chunk.get("type") == "tool_use" and chunk.get("name") == "get_current_weather"), None
    )
    assert weather_tool is not None, "Weather tool call not found"
    assert isinstance(weather_tool["id"], str) and weather_tool["id"], "Tool call should have a non-empty ID"
    assert isinstance(weather_tool["input"], str), "Tool input should be a string"
    assert "London" in weather_tool["input"], "Tool input should contain 'London'"


@pytest.mark.asyncio
async def test_multiple_tools():
    tools= [{
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
    },
    {"type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a given time zone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The IANA time zone name, e.g. America/Los_Angeles"
                    }
                },
                "required": ["timezone"]
            }
        }}
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
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "tools": tools,
        "tool_choice": {"type":"auto"},
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
                    current_tool_call = {
                        "type": "tool_use",
                        "id": tool_call.id,
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
    assert any(
        msg["type"] == "tool_use" and msg["name"] == "get_weather" 
        for msg in response_chunks
    ), "Weather tool call not found"
    assert any(
        msg["type"] == "tool_use" and msg["name"] == "get_time" 
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
        "system_prompt": "Respond in JSON format",
        "memory": [memory],
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
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