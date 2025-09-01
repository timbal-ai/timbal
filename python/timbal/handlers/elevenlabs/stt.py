import os
from typing import Literal

import httpx
from pydantic import Field

from ...errors import APIKeyNotFoundError
from ...types.file import File
from ...utils import resolve_default


async def stt(
    audio_file: File = Field(
        ...,
        description=(
            "The audio file to transcribe. "
            "All major audio and video formats are supported. "
            "The file size must be less than 1GB."
        ),
    ),
    model_id: Literal["scribe_v1", "scribe_v1_experimental"] = Field(
        "scribe_v1", 
        description="The elevenlabs model to use for the STT.",
    ),
    # TODO Add more fields.
) -> str: # ? Should we return the other elements.
    audio_file = resolve_default("audio_file", audio_file)
    model_id = resolve_default("model_id", model_id)

    if not audio_file.__content_type__.startswith("audio"):
        raise ValueError(f"STT expected an audio file content type. Got {audio_file.__content_type__}")

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise APIKeyNotFoundError("ELEVENLABS_API_KEY not found")

    # Ensure the pointer is at the beginning of the file
    audio_file.seek(0)

    async with httpx.AsyncClient() as client:
        headers = {"xi-api-key": api_key}

        files = {
            "file": (
                f"audio{audio_file.__source_extension__}", 
                audio_file.read(), 
                audio_file.__content_type__
            ),
            "model_id": (None, model_id),
        }

        res = await client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers=headers,
            files=files,
        )

        res.raise_for_status()

        res_body = res.json()
        # {
        #     'language_code': 'eng',
        #     'language_probability': 0.9866735935211182,
        #     'text': '...',
        #     'words': [{'text': '...', 'start': 0.079, 'end': 0.219, 'type': 'word'}, ...],
        # }
        return res_body["text"]
