import os
from typing import Literal

from openai import AsyncOpenAI
from pydantic import Field

from ...errors import APIKeyNotFoundError
from ...types.file import File
from ...utils import resolve_default


async def tts(
    text: str = Field(
        ...,
        description="The text to convert to speech.",
    ),
    voice: Literal[
        "alloy", "ash", "ballad", "coral", "echo", 
        "fable", "nova", "onyx", "sage", "shimmer"
    ] = Field(
        "alloy",
        description="The voice to use for text-to-speech.",
    ),
    model_id: Literal["gpt-4o-mini-tts", "tts-1", "tts-1-hd"] = Field(
        "gpt-4o-mini-tts",
        description="The model to use for text-to-speech.",
    ),
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]= Field(
        "mp3",
        description="The audio format of the output.",
    ),
    instructions: str | None = Field(
        None,
        description="Instructions to guide the model's speech generation (only supported for gpt-4o-mini-tts model).",
    ),
    # model -> we default to gpt-4o-mini-tts, but support all TTS models
    # user -> not used
) -> File:
    voice = resolve_default("voice", voice)
    model_id = resolve_default("model_id", model_id)
    response_format = resolve_default("response_format", response_format)
    instructions = resolve_default("instructions", instructions)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise APIKeyNotFoundError("OPENAI_API_KEY not found")

    client = AsyncOpenAI(api_key=api_key)

    kwargs = {
        "model": model_id,
        "voice": voice,
        "response_format": response_format,
        "input": text,
    }

    # Add instructions only for gpt-4o-mini-tts model
    if kwargs["model"] == "gpt-4o-mini-tts" and instructions is not None:
        kwargs["instructions"] = instructions

    response = await client.audio.speech.create(**kwargs)

    # Determine file extension based on response format
    extension_map = {
        "mp3": ".mp3",
        "opus": ".opus", 
        "aac": ".aac",
        "flac": ".flac",
        "wav": ".wav",
        "pcm": ".pcm",
    }
    
    extension = extension_map.get(kwargs["response_format"], ".mp3")

    return File.validate(
        response.content,
        {"extension": extension}
    ) 