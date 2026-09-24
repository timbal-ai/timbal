"""OpenAI file transcription and speech generation with standard tool credentials."""

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from pydantic import Field, SecretStr

from .._openai_audio import STT_MODELS, validate_speech
from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key


class OpenAITextToSpeech(Tool):
    name: str = "openai_text_to_speech"
    description: str | None = "Generate speech from text using OpenAI. Returns base64 audio and its content type."
    integration: Annotated[str, Integration("openai")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _text_to_speech(
            text: str = Field(..., min_length=1, max_length=4096, description="Text to speak."),
            model: str = Field("gpt-4o-mini-tts", description="Speech model: gpt-4o-mini-tts, tts-1 or tts-1-hd."),
            voice: str = Field("coral", description="OpenAI voice name, e.g. coral, alloy, marin or cedar."),
            response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
                "mp3", description="Audio format. PCM is raw 24 kHz signed 16-bit little-endian mono."
            ),
            speed: float = Field(1.0, ge=0.25, le=4.0, description="Speech speed multiplier."),
            instructions: str | None = Field(
                None, description="Speech style instructions; supported by gpt-4o-mini-tts, not tts-1 models."
            ),
        ) -> Any:
            key = await resolve_api_key(tool=self, provider_name="OpenAI", env_var="OPENAI_API_KEY")
            validate_speech(model, voice, instructions)
            payload = {
                "model": model,
                "voice": voice,
                "input": text,
                "response_format": response_format,
                "speed": speed,
            }
            if instructions:
                payload["instructions"] = instructions
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/speech", headers={"Authorization": f"Bearer {key}"}, json=payload
                )
                response.raise_for_status()
                return {
                    "audio_base64": base64.b64encode(response.content).decode("ascii"),
                    "content_type": response.headers.get("content-type", "application/octet-stream"),
                    "model": model,
                    "voice": voice,
                    "response_format": response_format,
                }

        super().__init__(handler=_text_to_speech, **kwargs)


class OpenAISpeechToText(Tool):
    name: str = "openai_speech_to_text"
    description: str | None = (
        "Transcribe a local audio file or base64 audio with OpenAI. Returns transcript text. Provide exactly one audio source."
    )
    integration: Annotated[str, Integration("openai")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _speech_to_text(
            audio_file_path: str | None = Field(
                None, description="Path to a local audio file; recommended for larger files."
            ),
            audio_file_base64: str | None = Field(
                None, description="Base64 audio data. Supply filename to identify the format."
            ),
            filename: str | None = Field(
                None, description="Audio filename, e.g. speech.wav. Required for base64 input."
            ),
            model: str = Field(
                "gpt-transcribe",
                description="Transcription model: gpt-transcribe (default), or legacy gpt-4o/whisper models.",
            ),
            language: str | None = Field(
                None, description="ISO-639-1 language code, e.g. en or es. Omit for auto-detection."
            ),
            prompt: str | None = Field(None, description="Vocabulary or context to guide transcription."),
        ) -> Any:
            if model not in STT_MODELS:
                raise ValueError(f"Unsupported OpenAI file transcription model: {model!r}")
            if bool(audio_file_path) == bool(audio_file_base64):
                raise ValueError("Provide exactly one of audio_file_path or audio_file_base64.")
            if audio_file_path:
                data = await asyncio.to_thread(Path(audio_file_path).read_bytes)
                name = filename or Path(audio_file_path).name
            else:
                if not filename:
                    raise ValueError("filename is required for base64 audio.")
                data = base64.b64decode(audio_file_base64, validate=True)
                name = filename
            key = await resolve_api_key(tool=self, provider_name="OpenAI", env_var="OPENAI_API_KEY")
            fields = {"model": model, "response_format": "json"}
            if language:
                fields["languages[]" if model == "gpt-transcribe" else "language"] = language
            if prompt:
                fields["prompt"] = prompt
            async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    data=fields,
                    files={"file": (name, data, mimetypes.guess_type(name)[0] or "application/octet-stream")},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_speech_to_text, **kwargs)
