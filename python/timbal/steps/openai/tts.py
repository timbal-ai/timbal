import os

from openai import AsyncOpenAI

from ...errors import APIKeyNotFoundError
from ...types.field import Field, resolve_default
from ...types.file import File


async def tts(
    text: str = Field(
        description="The text to convert to speech.",
    ),
    voice: str = Field(
        default=None,
        description="The voice to use for text-to-speech.",
        choices=[
            "alloy", "ash", "ballad", "coral", "echo", 
            "fable", "nova", "onyx", "sage", "shimmer"
        ],
    ),
    model_id: str = Field(
        default=None,
        description="The model to use for text-to-speech.",
        choices=["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
    ),
    response_format: str = Field(
        default=None,
        description="The audio format of the output.",
        choices=["mp3", "opus", "aac", "flac", "wav", "pcm"],
    ),
    instructions: str = Field(
        default=None,
        description="Instructions to guide the model's speech generation (only supported for gpt-4o-mini-tts model).",
    ),
    # model -> we default to gpt-4o-mini-tts, but support all TTS models
    # user -> not used
) -> File:

    text = resolve_default("text", text)
    voice = resolve_default("voice", voice)
    model_id = resolve_default("model_id", model_id)
    response_format = resolve_default("response_format", response_format)
    instructions = resolve_default("instructions", instructions)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise APIKeyNotFoundError("OPENAI_API_KEY not found")

    client = AsyncOpenAI(api_key=api_key)

    kwargs = {}

    if text is not None:
        kwargs["input"] = text
    if voice is not None:
        kwargs["voice"] = voice
    else:
        kwargs["voice"] = "alloy"  # Default voice
    if model_id is not None:
        kwargs["model"] = model_id
    else:
        kwargs["model"] = "gpt-4o-mini-tts"  # Default model
    if response_format is not None:
        kwargs["response_format"] = response_format
    else:
        kwargs["response_format"] = "mp3"  # Default format
    
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