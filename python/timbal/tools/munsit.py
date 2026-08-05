from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

# Global endpoint. UAE data-residency deployments use https://ae.api.faseeh.ai/api/v1
# with a key from ae.app.munsit.com (keys are region-specific).
_BASE_URL = "https://api.munsit.com/api/v1"

Dialect = Literal["auto", "emirati", "fusha"]


class _MunsitTool(Tool):
    integration: Annotated[str, Integration("munsit")] | None = None
    api_key: SecretStr | None = None
    base_url: str = _BASE_URL

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "api_key": self.api_key,
                    "base_url": self.base_url,
                }
            ),
        }

    async def _resolve_api_key(self) -> str:
        return await resolve_api_key(tool=self, provider_name="Munsit", env_var="MUNSIT_API_KEY")


class MunsitTextToSpeech(_MunsitTool):
    name: str = "munsit_text_to_speech"
    description: str | None = (
        "Generate natural Arabic speech from text using Munsit (Faseeh) TTS. "
        "Supports multiple Arabic dialects (Fusha, Emirati, Saudi Najdi/Hijazi, ...) and "
        "Arabic-English code-switching voices. Get voice IDs from munsit_list_voices. "
        "Returns the audio as a base64-encoded WAV file."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _text_to_speech(
            text: str = Field(..., description="The Arabic text to convert to speech."),
            voice_id: str = Field(
                ...,
                description="Munsit voice ID (opaque string from munsit_list_voices, e.g. 'ar-najdi-male-2').",
            ),
            model_id: str = Field("faseeh-v1-preview", description="TTS model ID; list options with munsit_list_models."),
            stability: float = Field(0.5, description="Voice stability (0.0-1.0). Higher is more consistent."),
            speed: float = Field(1.0, description="Speech speed (0.7-1.2)."),
            sample_rate: int = Field(
                48000,
                description="Output sample rate in Hz (8000-48000). 48000 is the engine-native rate.",
            ),
            dialect: Dialect = Field("auto", description="Dialect hint for synthesis: 'auto', 'emirati', or 'fusha'."),
        ) -> Any:
            api_key = await self._resolve_api_key()
            import base64

            import httpx

            payload: dict[str, Any] = {
                "voice_id": voice_id,
                "text": text,
                "stability": stability,
                "speed": speed,
                "streaming": False,
                "sample_rate": sample_rate,
                "dialect": dialect,
            }

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                response = await client.post(
                    f"{self.base_url}/text-to-speech/{model_id}",
                    headers={"x-api-key": api_key},
                    json=payload,
                )
                response.raise_for_status()
                audio_b64 = base64.b64encode(response.content).decode("utf-8")
                return {
                    "audio_base64": audio_b64,
                    "content_type": response.headers.get("content-type", "audio/wav"),
                    "sample_rate": sample_rate,
                }

        super().__init__(handler=_text_to_speech, **kwargs)


class MunsitListVoices(_MunsitTool):
    name: str = "munsit_list_voices"
    description: str | None = (
        "List all available Munsit TTS voices with their voice_id, name, gender, age, "
        "languages, dialects (fusha, emirati, najdi, hijazi, ...), and a sample audio URL. "
        "Use a returned voice_id in munsit_text_to_speech."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_voices() -> Any:
            api_key = await self._resolve_api_key()
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                response = await client.get(
                    f"{self.base_url}/voices",
                    headers={"x-api-key": api_key},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_voices, **kwargs)


class MunsitListModels(_MunsitTool):
    name: str = "munsit_list_models"
    description: str | None = (
        "List available Munsit TTS models. Use a returned model_id in munsit_text_to_speech."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_models() -> Any:
            api_key = await self._resolve_api_key()
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"x-api-key": api_key},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_models, **kwargs)
