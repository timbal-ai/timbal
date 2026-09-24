"""Deepgram Aura audio generation and voice discovery tools."""

import base64
from typing import Annotated, Any, Literal

import httpx
from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

_BASE_URL = "https://api.deepgram.com/v1"


class DeepgramTextToSpeech(Tool):
    name: str = "deepgram_text_to_speech"
    description: str | None = (
        "Generate spoken audio from text using a Deepgram Aura voice model. "
        "Returns base64 audio and its content type. Use DeepgramListVoices to discover voice model ids."
    )
    integration: Annotated[str, Integration("deepgram")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _text_to_speech(
            text: str = Field(..., min_length=1, description="The text to synthesize."),
            model: str = Field("aura-2-thalia-en", description="Full Aura voice model id, e.g. aura-2-thalia-en."),
            encoding: Literal["mp3", "linear16", "mulaw", "alaw", "opus", "flac", "aac"] = Field(
                "mp3", description="Output audio encoding."
            ),
            sample_rate: int | None = Field(
                None, description="Sample rate in Hz; omit for fixed-rate encodings like MP3."
            ),
            container: Literal["wav", "ogg", "none"] | None = Field(
                None, description="Audio container, if supported by the encoding."
            ),
            bit_rate: int | None = Field(None, description="Bitrate in bits/sec, if supported by the encoding."),
            speed: float = Field(
                1.0, ge=0.7, le=1.5, description="Speaking rate multiplier (language-dependent support)."
            ),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="Deepgram", env_var="DEEPGRAM_API_KEY")
            params = {"model": model, "encoding": encoding, "speed": speed}
            for key, value in (("sample_rate", sample_rate), ("container", container), ("bit_rate", bit_rate)):
                if value is not None:
                    params[key] = value
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/speak",
                    headers={"Authorization": f"Token {api_key}"},
                    params=params,
                    json={"text": text},
                )
                response.raise_for_status()
                return {
                    "audio_base64": base64.b64encode(response.content).decode("ascii"),
                    "content_type": response.headers.get("content-type", "application/octet-stream"),
                    "model": model,
                    "encoding": encoding,
                }

        super().__init__(handler=_text_to_speech, **kwargs)


class DeepgramListVoices(Tool):
    name: str = "deepgram_list_voices"
    description: str | None = (
        "List Deepgram Aura voices, with canonical model ids, languages, accents and audio sample URLs. "
        "Use a canonical_name as the model for Deepgram text-to-speech."
    )
    integration: Annotated[str, Integration("deepgram")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_voices(
            language: str | None = Field(None, description="Filter by language, e.g. en, en-US, es."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="Deepgram", env_var="DEEPGRAM_API_KEY")
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(f"{_BASE_URL}/models", headers={"Authorization": f"Token {api_key}"})
                response.raise_for_status()
                voices = [v for v in response.json().get("tts", []) if v.get("canonical_name", "").startswith("aura-")]
                if language:
                    lang = language.strip().lower()
                    voices = [
                        v
                        for v in voices
                        if any(
                            code.lower() == lang or code.lower().startswith(lang + "-")
                            for code in v.get("languages", [])
                        )
                    ]
                return {"voices": voices}

        super().__init__(handler=_list_voices, **kwargs)
