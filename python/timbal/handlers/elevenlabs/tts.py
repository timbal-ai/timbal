import os
from typing import Literal

import httpx
from pydantic import Field

from ...errors import APIKeyNotFoundError
from ...types.file import File
from ...utils import resolve_default


async def tts(
    text: str = Field(
        ...,
        description="The text to convert to speech.",
    ),
    voice_id: str = Field(
        ...,
        description="The voice ID to use for text-to-speech.",
    ),
    model_id: Literal["eleven_flash_v2_5", "eleven_multilingual_v2"] = Field(
        "eleven_flash_v2_5",
        description="The model to use for text-to-speech.",
    ),
    # TODO Add more fields.
) -> File:
    text = resolve_default("text", text)
    voice_id = resolve_default("voice_id", voice_id)
    model_id = resolve_default("model_id", model_id)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise APIKeyNotFoundError("ELEVENLABS_API_KEY not found")

    async with httpx.AsyncClient() as client:
        headers = {"xi-api-key": api_key}

        data = {
            "text": text,
            "model_id": model_id,
        }

        res = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128",
            headers=headers,
            json=data,
        )

        res.raise_for_status()

        return File.validate(
            res.content, 
            {"extension": ".mp3"}
        )
