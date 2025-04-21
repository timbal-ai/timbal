import os

import httpx

from ...errors import APIKeyNotFoundError
from ...types.field import Field
from ...types.file import File


async def tts(
    text: str = Field(
        description="The text to convert to speech.",
    ),
    voice_id: str = Field(
        description="The voice ID to use for text-to-speech.",
    ),
    model_id: str = Field(
        default="eleven_flash_v2_5",
        description="The model to use for text-to-speech.",
        choices=["eleven_flash_v2_5", "eleven_multilingual_v2"],
    ),
    # TODO Add more fields.
) -> File:

    # Enable calling this step without pydantic model_validate()
    text = text.default if hasattr(text, "default") else text
    voice_id = voice_id.default if hasattr(voice_id, "default") else voice_id
    model_id = model_id.default if hasattr(model_id, "default") else model_id

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
