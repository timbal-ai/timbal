import os

import httpx

from ...errors import APIKeyNotFoundError
from ...types.field import Field
from ...types.file import File


async def stt(
    audio_file: File = Field(
        description=(
            "The audio file to transcribe. "
            "All major audio and video formats are supported. "
            "The file size must be less than 1GB."
        ),
    ),
    model_id: str = Field(
        default="scribe_v1", 
        description="The elevenlabs model to use for the STT.",
        choices=["scribe_v1", "scribe_v1_experimental"],
    ),
    # TODO Add more fields.
) -> str: # ? Should we return the other elements.

    # Enable calling this step without pydantic model_validate()
    audio_file = audio_file.default if hasattr(audio_file, "default") else audio_file
    model_id = model_id.default if hasattr(model_id, "default") else model_id

    if not audio_file.__content_type__.startswith("audio"):
        raise ValueError(f"STT expected an audio file content type. Got {audio_file.__content_type__}")

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise APIKeyNotFoundError("ELEVENLABS_API_KEY not found")

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
