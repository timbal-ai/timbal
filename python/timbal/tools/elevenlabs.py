import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

_ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"


class ElevenLabsTextToSpeech(Tool):
    name: str = "elevenlabs_text_to_speech"
    description: str | None = (
        "Convert text to speech using a specified ElevenLabs voice. "
        "Returns the audio encoded as a base64 string."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _text_to_speech(
            text: str = Field(..., description="The text to synthesize."),
            voice_id: str = Field(..., description="ElevenLabs voice ID to use."),
            model_id: str = Field("eleven_multilingual_v2", description="Model to use, e.g. 'eleven_multilingual_v2', 'eleven_turbo_v2'."),
            output_format: str = Field("mp3_44100_128", description="Audio format, e.g. 'mp3_44100_128', 'pcm_16000', 'ulaw_8000'."),
            stability: float = Field(0.5, description="Voice stability (0.0–1.0)."),
            similarity_boost: float = Field(0.75, description="Voice similarity boost (0.0–1.0)."),
            style: float = Field(0.0, description="Style exaggeration (0.0–1.0, only for v2 models)."),
            use_speaker_boost: bool = Field(True, description="Improve speaker clarity (slight latency increase)."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64

            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/text-to-speech/{voice_id}",
                    headers={"xi-api-key": api_key},
                    params={"output_format": output_format},
                    json={
                        "text": text,
                        "model_id": model_id,
                        "voice_settings": {
                            "stability": stability,
                            "similarity_boost": similarity_boost,
                            "style": style,
                            "use_speaker_boost": use_speaker_boost,
                        },
                    },
                )
                response.raise_for_status()
                audio_b64 = base64.b64encode(response.content).decode("utf-8")
                return {
                    "audio_base64": audio_b64,
                    "content_type": response.headers.get("content-type", "audio/mpeg"),
                    "output_format": output_format,
                }

        super().__init__(handler=_text_to_speech, **kwargs)


class ElevenLabsSpeechToText(Tool):
    name: str = "elevenlabs_speech_to_text"
    description: str | None = (
        "Transcribe an audio (or video) file to text using ElevenLabs Scribe. "
        "Use audio_file_path for local files (recommended); use audio_file_base64 only for small pasted samples. "
        "Returns the transcript text plus detected language and (optionally) word-level timestamps and speaker labels."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _speech_to_text(
            audio_file_path: str | None = Field(
                None,
                description="Path to a local audio/video file (e.g. '/path/to/audio.mp3'). Preferred over base64 for large files.",
            ),
            audio_file_base64: str | None = Field(
                None,
                description="Audio/video file content as a base64 string. Use audio_file_path instead for large files.",
            ),
            filename: str | None = Field(None, description="Original filename, used to infer the media type (e.g. 'audio.mp3')."),
            model_id: str = Field("scribe_v1", description="STT model to use, e.g. 'scribe_v1' or 'scribe_v1_experimental'."),
            language_code: str | None = Field(None, description="ISO-639-1/3 language code to force (e.g. 'en', 'es'). Auto-detected if omitted."),
            tag_audio_events: bool = Field(True, description="Tag non-speech audio events such as (laughter) or (applause)."),
            diarize: bool = Field(False, description="Annotate which speaker said each word (speaker diarization)."),
            num_speakers: int | None = Field(None, description="Hint for the maximum number of distinct speakers (only used with diarize)."),
            timestamps_granularity: str = Field("word", description="Timestamp granularity: 'none', 'word', or 'character'."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64

            import httpx

            if audio_file_path:
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()
                resolved_name = filename or os.path.basename(audio_file_path)
            elif audio_file_base64:
                audio_bytes = base64.b64decode(audio_file_base64)
                resolved_name = filename or "audio"
            else:
                raise ValueError("Provide either audio_file_path or audio_file_base64.")

            data: dict[str, Any] = {
                "model_id": model_id,
                "tag_audio_events": str(tag_audio_events).lower(),
                "diarize": str(diarize).lower(),
                "timestamps_granularity": timestamps_granularity,
            }
            if language_code:
                data["language_code"] = language_code
            if num_speakers is not None:
                data["num_speakers"] = num_speakers

            files = {"file": (resolved_name, audio_bytes, "application/octet-stream")}

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/speech-to-text",
                    headers={"xi-api-key": api_key},
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_speech_to_text, **kwargs)


class ElevenLabsSpeechToSpeech(Tool):
    name: str = "elevenlabs_speech_to_speech"
    description: str | None = (
        "Convert speech in an audio file into a different ElevenLabs voice while preserving the "
        "delivery, emotion, and timing (voice changer). "
        "Use audio_file_path for local files (recommended); use audio_file_base64 only for small pasted samples. "
        "Returns the converted audio encoded as a base64 string."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _speech_to_speech(
            voice_id: str = Field(..., description="Target ElevenLabs voice ID to convert the speech into."),
            audio_file_path: str | None = Field(
                None,
                description="Path to a local audio file (e.g. '/path/to/audio.mp3'). Preferred over base64 for large files.",
            ),
            audio_file_base64: str | None = Field(
                None,
                description="Audio file content as a base64 string. Use audio_file_path instead for large files.",
            ),
            filename: str | None = Field(None, description="Original filename (e.g. 'audio.mp3')."),
            model_id: str = Field("eleven_multilingual_sts_v2", description="STS model to use, e.g. 'eleven_multilingual_sts_v2' or 'eleven_english_sts_v2'."),
            output_format: str = Field("mp3_44100_128", description="Audio format, e.g. 'mp3_44100_128', 'pcm_16000', 'ulaw_8000'."),
            stability: float = Field(0.5, description="Voice stability (0.0–1.0)."),
            similarity_boost: float = Field(0.75, description="Voice similarity boost (0.0–1.0)."),
            style: float = Field(0.0, description="Style exaggeration (0.0–1.0)."),
            use_speaker_boost: bool = Field(True, description="Improve speaker clarity (slight latency increase)."),
            remove_background_noise: bool = Field(False, description="Remove background noise from the source audio before conversion."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64
            import json

            import httpx

            if audio_file_path:
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()
                resolved_name = filename or os.path.basename(audio_file_path)
            elif audio_file_base64:
                audio_bytes = base64.b64decode(audio_file_base64)
                resolved_name = filename or "audio"
            else:
                raise ValueError("Provide either audio_file_path or audio_file_base64.")

            data: dict[str, Any] = {
                "model_id": model_id,
                "remove_background_noise": str(remove_background_noise).lower(),
                "voice_settings": json.dumps({
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                    "style": style,
                    "use_speaker_boost": use_speaker_boost,
                }),
            }
            files = {"audio": (resolved_name, audio_bytes, "application/octet-stream")}

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/speech-to-speech/{voice_id}",
                    headers={"xi-api-key": api_key},
                    params={"output_format": output_format},
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                audio_b64 = base64.b64encode(response.content).decode("utf-8")
                return {
                    "audio_base64": audio_b64,
                    "content_type": response.headers.get("content-type", "audio/mpeg"),
                    "output_format": output_format,
                }

        super().__init__(handler=_speech_to_speech, **kwargs)


class ElevenLabsSoundGeneration(Tool):
    name: str = "elevenlabs_sound_generation"
    description: str | None = (
        "Generate a sound effect from a text prompt (e.g. 'glass shattering', 'cinematic whoosh'). "
        "Returns the audio encoded as a base64 string."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _sound_generation(
            text: str = Field(..., description="Text prompt describing the sound effect to generate."),
            duration_seconds: float | None = Field(
                None,
                description="Desired duration in seconds (0.5–30). If omitted, the model picks an optimal length.",
            ),
            prompt_influence: float = Field(0.3, description="How strictly to follow the prompt (0.0–1.0). Higher is more literal."),
            model_id: str = Field("eleven_text_to_sound_v2", description="Sound generation model to use."),
            loop: bool = Field(False, description="Generate a smoothly looping sound effect (only for 'eleven_text_to_sound_v2')."),
            output_format: str = Field("mp3_44100_128", description="Audio format, e.g. 'mp3_44100_128', 'pcm_16000'."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64

            import httpx

            payload: dict[str, Any] = {
                "text": text,
                "model_id": model_id,
                "loop": loop,
                "prompt_influence": prompt_influence,
            }
            if duration_seconds is not None:
                payload["duration_seconds"] = duration_seconds

            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/sound-generation",
                    headers={"xi-api-key": api_key},
                    params={"output_format": output_format},
                    json=payload,
                )
                response.raise_for_status()
                audio_b64 = base64.b64encode(response.content).decode("utf-8")
                return {
                    "audio_base64": audio_b64,
                    "content_type": response.headers.get("content-type", "audio/mpeg"),
                    "output_format": output_format,
                }

        super().__init__(handler=_sound_generation, **kwargs)


class ElevenLabsAudioIsolation(Tool):
    name: str = "elevenlabs_audio_isolation"
    description: str | None = (
        "Isolate speech from an audio file by stripping background noise/music. "
        "Use audio_file_path for local files (recommended); use audio_file_base64 only for small pasted samples. "
        "Returns the cleaned audio encoded as a base64 string."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _audio_isolation(
            audio_file_path: str | None = Field(
                None,
                description="Path to a local audio file (e.g. '/path/to/audio.mp3'). Preferred over base64 for large files.",
            ),
            audio_file_base64: str | None = Field(
                None,
                description="Audio file content as a base64 string. Use audio_file_path instead for large files.",
            ),
            filename: str | None = Field(None, description="Original filename (e.g. 'audio.mp3')."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64

            import httpx

            if audio_file_path:
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()
                resolved_name = filename or os.path.basename(audio_file_path)
            elif audio_file_base64:
                audio_bytes = base64.b64decode(audio_file_base64)
                resolved_name = filename or "audio"
            else:
                raise ValueError("Provide either audio_file_path or audio_file_base64.")

            files = {"audio": (resolved_name, audio_bytes, "application/octet-stream")}

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/audio-isolation",
                    headers={"xi-api-key": api_key},
                    files=files,
                )
                response.raise_for_status()
                audio_b64 = base64.b64encode(response.content).decode("utf-8")
                return {
                    "audio_base64": audio_b64,
                    "content_type": response.headers.get("content-type", "audio/mpeg"),
                }

        super().__init__(handler=_audio_isolation, **kwargs)


class ElevenLabsListPhoneNumbers(Tool):
    name: str = "elevenlabs_list_phone_numbers"
    description: str | None = (
        "List all phone numbers registered with ElevenLabs (Twilio, etc.). "
        "Returns phone_number_id, phone_number, label, and assigned_agent. "
        "Use phone_number_id as agent_phone_number_id when making outbound calls."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_phone_numbers() -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/convai/phone-numbers",
                    headers={"xi-api-key": api_key},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_phone_numbers, **kwargs)


class ElevenLabsMakeOutboundCall(Tool):
    name: str = "elevenlabs_make_outbound_call"
    description: str | None = (
        "Initiate an outbound phone call via Twilio using an ElevenLabs conversational agent. "
        "Requires agent_phone_number_id from list_phone_numbers (not a raw phone number)."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _make_outbound_call(
            agent_id: str = Field(..., description="The ElevenLabs conversational agent ID."),
            agent_phone_number_id: str = Field(
                ...,
                description="Phone number ID from list_phone_numbers. Must be a number registered in ElevenLabs.",
            ),
            to_number: str = Field(
                ...,
                description="Destination phone number in E.164 format, e.g. '+14155552671'.",
            ),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            payload: dict[str, Any] = {
                "agent_id": agent_id,
                "agent_phone_number_id": agent_phone_number_id,
                "to_number": to_number,
            }

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/convai/twilio/outbound-call",
                    headers={"xi-api-key": api_key},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_make_outbound_call, **kwargs)


class ElevenLabsListAgents(Tool):
    name: str = "elevenlabs_list_agents"
    description: str | None = (
        "List all ElevenLabs conversational AI agents. Returns agent_id, name, and other metadata for each agent."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_agents() -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/convai/agents",
                    headers={"xi-api-key": api_key},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_agents, **kwargs)


class ElevenLabsGetModels(Tool):
    name: str = "elevenlabs_get_models"
    description: str | None = "List all available ElevenLabs TTS models with their capabilities."
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_models() -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/models",
                    headers={"xi-api-key": api_key},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_models, **kwargs)


class ElevenLabsGetVoicesWithDescriptions(Tool):
    name: str = "elevenlabs_get_voices_with_descriptions"
    description: str | None = (
        "Fetch all available voices from ElevenLabs including metadata "
        "such as name, gender, accent, age, and category."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_voices_with_descriptions(
            show_legacy: bool = Field(False, description="Include legacy pre-XTTS voices in the response."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/voices",
                    headers={"xi-api-key": api_key},
                    params={"show_legacy": str(show_legacy).lower()},
                )
                response.raise_for_status()
                data = response.json()

            voices = data.get("voices", [])
            enriched = []
            for voice in voices:
                labels = voice.get("labels", {})
                enriched.append({
                    "voice_id": voice.get("voice_id"),
                    "name": voice.get("name"),
                    "category": voice.get("category"),
                    "gender": labels.get("gender"),
                    "age": labels.get("age"),
                    "accent": labels.get("accent"),
                    "description": labels.get("description"),
                    "use_case": labels.get("use case"),
                    "preview_url": voice.get("preview_url"),
                    "fine_tuning_requested": voice.get("fine_tuning", {}).get("is_allowed_to_fine_tune"),
                })
            return {"voices": enriched, "total": len(enriched)}

        super().__init__(handler=_get_voices_with_descriptions, **kwargs)


class ElevenLabsListHistory(Tool):
    name: str = "elevenlabs_list_history"
    description: str | None = (
        "List generated audio history items (TTS, ConvAI, etc.). "
        "Returns history_item_id, text, voice_name, date_unix, source for each item. "
        "Use history_item_id with get_audio_from_history_item or download_history_items."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_history(
            page_size: int = Field(100, description="Max items to return (default 100)."),
            source: str | None = Field(None, description="Filter by source: 'TTS', 'STS', or 'ConvAI'."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            params: dict[str, Any] = {"page_size": page_size}
            if source:
                params["source"] = source

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/history",
                    headers={"xi-api-key": api_key},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_history, **kwargs)


class ElevenLabsGetAudioFromHistoryItem(Tool):
    name: str = "elevenlabs_get_audio_from_history_item"
    description: str | None = (
        "Retrieve the audio of a history item and return it as a base64-encoded file."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_audio_from_history_item(
            history_item_id: str = Field(..., description="The ID of the history item to retrieve audio for."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64

            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/history/{history_item_id}/audio",
                    headers={"xi-api-key": api_key},
                )
                response.raise_for_status()
                audio_b64 = base64.b64encode(response.content).decode("utf-8")
                return {
                    "history_item_id": history_item_id,
                    "audio_base64": audio_b64,
                    "content_type": response.headers.get("content-type", "audio/mpeg"),
                }

        super().__init__(handler=_get_audio_from_history_item, **kwargs)


class ElevenLabsDownloadHistoryItems(Tool):
    name: str = "elevenlabs_download_history_items"
    description: str | None = (
        "Download one or more history items. A single item returns an audio file (base64). "
        "Multiple items are packed into a .zip file and returned as base64."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_history_items(
            history_item_ids: list[str] = Field(..., description="List of history item IDs to download. Single ID returns audio; multiple IDs return a .zip."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64
            import io
            import zipfile

            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/history/download",
                    headers={"xi-api-key": api_key},
                    json={"history_item_ids": history_item_ids},
                )
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            is_zip = "zip" in content_type or len(history_item_ids) > 1

            if is_zip:
                zip_b64 = base64.b64encode(response.content).decode("utf-8")
                buf = io.BytesIO(response.content)
                with zipfile.ZipFile(buf) as zf:
                    filenames = zf.namelist()
                return {
                    "type": "zip",
                    "file_base64": zip_b64,
                    "filenames": filenames,
                    "content_type": "application/zip",
                }

            audio_b64 = base64.b64encode(response.content).decode("utf-8")
            return {
                "type": "audio",
                "audio_base64": audio_b64,
                "content_type": content_type or "audio/mpeg",
            }

        super().__init__(handler=_download_history_items, **kwargs)


class ElevenLabsCreateAgent(Tool):
    name: str = "elevenlabs_create_agent"
    description: str | None = (
        "Create a new ElevenLabs conversational AI agent with a given name, "
        "prompt, voice, and optional first message."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_agent(
            name: str = Field(..., description="Display name for the agent."),
            voice_id: str = Field(..., description="ElevenLabs voice ID the agent will use."),
            system_prompt: str = Field(..., description="The LLM system prompt defining agent behaviour."),
            first_message: str | None = Field(None, description="Optional opening message the agent speaks first."),
            language: str = Field("en", description="BCP-47 language code, e.g. 'en', 'es', 'fr'."),
            model_id: str = Field("eleven_turbo_v2", description="ElevenLabs model ID to use. e.g. 'eleven_turbo_v2'"),
            max_duration_seconds: int = Field(600, description="Maximum call/conversation length in seconds."),
            turn_timeout: int = Field(20, description="Seconds of silence before the agent ends its turn."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            payload: dict[str, Any] = {
                "name": name,
                "conversation_config": {
                    "agent": {
                        "prompt": {"prompt": system_prompt},
                        "language": language,
                    },
                    "tts": {
                        "voice_id": voice_id,
                        "model_id": model_id,
                    },
                    "turn": {
                        "turn_timeout": turn_timeout,
                    },
                    "conversation": {
                        "max_duration_seconds": max_duration_seconds,
                    },
                },
            }
            if first_message:
                payload["conversation_config"]["agent"]["first_message"] = first_message

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/convai/agents/create",
                    headers={"xi-api-key": api_key},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_agent, **kwargs)


class ElevenLabsAddVoice(Tool):
    name: str = "elevenlabs_add_voice"
    description: str | None = (
        "Clone or add a new voice to ElevenLabs from audio files. "
        "Use audio_file_paths for local files (recommended); use audio_files_base64 only for small pasted samples."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_voice(
            name: str = Field(..., description="Display name for the new voice."),
            audio_file_paths: list[str] | None = Field(
                None,
                description="Paths to local audio files (e.g. ['/path/to/sample.mp3']). Preferred over base64 for large files.",
            ),
            audio_files_base64: list[str] | None = Field(
                None,
                description="List of audio file contents as base64 strings. Use audio_file_paths instead for large files.",
            ),
            filenames: list[str] | None = Field(None, description="Optional list of original filenames (e.g. ['sample1.mp3']). Must match the length of audio files if provided."),
            description: str | None = Field(None, description="Optional text description of the voice."),
            labels: dict[str, str] | None = Field(
                None,
                description="Optional metadata. Keys: language (BCP-47: 'ca', 'es', 'en', 'fr'), accent, gender, age. E.g. {'language': 'ca', 'accent': 'British'}.",
            ),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64

            import httpx

            if audio_file_paths:
                audio_chunks: list[bytes] = []
                for p in audio_file_paths:
                    with open(p, "rb") as f:
                        audio_chunks.append(f.read())
                resolved_names = filenames or [os.path.basename(p) for p in audio_file_paths]
            elif audio_files_base64:
                audio_chunks = [base64.b64decode(b64) for b64 in audio_files_base64]
                resolved_names = filenames or [f"sample_{i}.mp3" for i in range(len(audio_files_base64))]
            else:
                raise ValueError("Provide either audio_file_paths or audio_files_base64.")

            files: list[tuple[str, Any]] = []
            for i, audio_bytes in enumerate(audio_chunks):
                fname = resolved_names[i] if i < len(resolved_names) else f"sample_{i}.mp3"
                files.append(("files", (fname, audio_bytes, "audio/mpeg")))

            data: dict[str, Any] = {"name": name}
            if description:
                data["description"] = description
            if labels:
                import json
                data["labels"] = json.dumps(labels)

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/voices/add",
                    headers={"xi-api-key": api_key},
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_add_voice, **kwargs)


class ElevenLabsDeleteVoice(Tool):
    name: str = "elevenlabs_delete_voice"
    description: str | None = (
        "Delete a voice from your ElevenLabs voice library by voice_id. "
        "Useful for cleaning up cloned/designed voices that count against your quota."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_voice(
            voice_id: str = Field(..., description="The ID of the voice to delete."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.delete(
                    f"{_ELEVENLABS_BASE}/voices/{voice_id}",
                    headers={"xi-api-key": api_key},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_delete_voice, **kwargs)


class ElevenLabsCreateVoicePreviews(Tool):
    name: str = "elevenlabs_create_voice_previews"
    description: str | None = (
        "Design new voice candidates from a text description (voice design). "
        "Returns a list of previews, each with a generated_voice_id and base64 audio sample. "
        "Pass a chosen generated_voice_id to create_voice_from_preview to save it permanently."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_voice_previews(
            voice_description: str = Field(..., description="Description of the desired voice (e.g. 'a calm middle-aged British man')."),
            text: str | None = Field(
                None,
                description="Sample text to speak in the previews (100–1000 chars). Omit and set auto_generate_text=True to auto-generate.",
            ),
            auto_generate_text: bool = Field(False, description="Auto-generate sample text suited to the description instead of providing 'text'."),
            model_id: str = Field("eleven_multilingual_ttv_v2", description="Voice design model: 'eleven_multilingual_ttv_v2' or 'eleven_ttv_v3'."),
            loudness: float = Field(0.5, description="Volume level (-1 quietest, 1 loudest, 0 ≈ -24 LUFS)."),
            quality: float = Field(0.9, description="Higher quality = better output but less variety (0.0–1.0)."),
            guidance_scale: float = Field(5.0, description="How closely to follow the prompt. Higher sticks to prompt; too high can sound robotic."),
            should_enhance: bool = Field(False, description="Let the AI expand a short description into a more detailed one before generating."),
            seed: int | None = Field(None, description="Optional seed for reproducible generation."),
            output_format: str = Field("mp3_44100_128", description="Audio format for the preview samples."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            payload: dict[str, Any] = {
                "voice_description": voice_description,
                "model_id": model_id,
                "auto_generate_text": auto_generate_text,
                "loudness": loudness,
                "quality": quality,
                "guidance_scale": guidance_scale,
                "should_enhance": should_enhance,
            }
            if text:
                payload["text"] = text
            if seed is not None:
                payload["seed"] = seed

            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/text-to-voice/design",
                    headers={"xi-api-key": api_key},
                    params={"output_format": output_format},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_voice_previews, **kwargs)


class ElevenLabsCreateVoiceFromPreview(Tool):
    name: str = "elevenlabs_create_voice_from_preview"
    description: str | None = (
        "Save a designed voice preview to your voice library permanently. "
        "Use a generated_voice_id returned by create_voice_previews."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_voice_from_preview(
            voice_name: str = Field(..., description="Name for the new voice."),
            voice_description: str = Field(..., description="Description for the new voice (20–1000 chars)."),
            generated_voice_id: str = Field(..., description="The generated_voice_id from a create_voice_previews response."),
            labels: dict[str, str] | None = Field(None, description="Optional metadata labels, e.g. {'language': 'en', 'accent': 'British'}."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            payload: dict[str, Any] = {
                "voice_name": voice_name,
                "voice_description": voice_description,
                "generated_voice_id": generated_voice_id,
            }
            if labels:
                payload["labels"] = labels

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/text-to-voice",
                    headers={"xi-api-key": api_key},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_voice_from_preview, **kwargs)


class ElevenLabsCreateDubbing(Tool):
    name: str = "elevenlabs_create_dubbing"
    description: str | None = (
        "Translate and dub an audio/video file into another language while preserving the speaker's voice. "
        "Provide either source_url or a local file (audio_file_path / audio_file_base64). "
        "Returns a dubbing_id; this runs asynchronously — poll status with get_dubbing, then fetch output with get_dubbed_audio."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_dubbing(
            target_lang: str = Field(..., description="Target language code (ISO 639-1/3), e.g. 'es', 'fr'."),
            source_url: str | None = Field(None, description="URL of the source video/audio to dub. Use this OR a local file."),
            audio_file_path: str | None = Field(None, description="Path to a local audio/video file to dub."),
            audio_file_base64: str | None = Field(None, description="Audio/video file content as base64. Use audio_file_path for large files."),
            filename: str | None = Field(None, description="Original filename (e.g. 'video.mp4'), used when uploading a file."),
            name: str | None = Field(None, description="Optional name for the dubbing project."),
            source_lang: str = Field("auto", description="Source language code (ISO 639-1/3), or 'auto' to detect."),
            num_speakers: int = Field(0, description="Number of speakers. 0 = auto-detect."),
            watermark: bool = Field(False, description="Add a watermark to the output."),
            highest_resolution: bool = Field(False, description="Use the highest available resolution for video output."),
            drop_background_audio: bool = Field(False, description="Drop the original background audio from the output."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64

            import httpx

            data: dict[str, Any] = {
                "target_lang": target_lang,
                "source_lang": source_lang,
                "num_speakers": num_speakers,
                "watermark": str(watermark).lower(),
                "highest_resolution": str(highest_resolution).lower(),
                "drop_background_audio": str(drop_background_audio).lower(),
            }
            if name:
                data["name"] = name

            files = None
            if source_url:
                data["source_url"] = source_url
            elif audio_file_path:
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()
                resolved_name = filename or os.path.basename(audio_file_path)
                files = {"file": (resolved_name, audio_bytes, "application/octet-stream")}
            elif audio_file_base64:
                audio_bytes = base64.b64decode(audio_file_base64)
                files = {"file": (filename or "source", audio_bytes, "application/octet-stream")}
            else:
                raise ValueError("Provide source_url, audio_file_path, or audio_file_base64.")

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/dubbing",
                    headers={"xi-api-key": api_key},
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_dubbing, **kwargs)


class ElevenLabsGetDubbing(Tool):
    name: str = "elevenlabs_get_dubbing"
    description: str | None = (
        "Get the metadata and status of a dubbing project by dubbing_id "
        "(status is 'dubbing', 'dubbed', or 'failed'). Use get_dubbed_audio to fetch the result once dubbed."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_dubbing(
            dubbing_id: str = Field(..., description="The ID of the dubbing project."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/dubbing/{dubbing_id}",
                    headers={"xi-api-key": api_key},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_dubbing, **kwargs)


class ElevenLabsListDubbings(Tool):
    name: str = "elevenlabs_list_dubbings"
    description: str | None = (
        "List your dubbing projects with their ids, names, statuses, and target languages."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_dubbings(
            page_size: int = Field(100, description="Max items to return (default 100)."),
            dubbing_status: str | None = Field(None, description="Filter by status: 'dubbing', 'dubbed', or 'failed'."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import httpx

            params: dict[str, Any] = {"page_size": page_size}
            if dubbing_status:
                params["dubbing_status"] = dubbing_status

            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/dubbing",
                    headers={"xi-api-key": api_key},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_dubbings, **kwargs)


class ElevenLabsGetDubbedAudio(Tool):
    name: str = "elevenlabs_get_dubbed_audio"
    description: str | None = (
        "Download the dubbed audio/video output for a completed dubbing project in a given language. "
        "Returns the file encoded as a base64 string. The dubbing must be in 'dubbed' status."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_dubbed_audio(
            dubbing_id: str = Field(..., description="The ID of the dubbing project."),
            language_code: str = Field(..., description="Language code of the dubbed track to download, e.g. 'es'."),
        ) -> Any:
            api_key = await resolve_api_key(tool=self, provider_name="ElevenLabs", env_var="ELEVENLABS_API_KEY")
            import base64

            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/dubbing/{dubbing_id}/audio/{language_code}",
                    headers={"xi-api-key": api_key},
                )
                response.raise_for_status()
                file_b64 = base64.b64encode(response.content).decode("utf-8")
                return {
                    "dubbing_id": dubbing_id,
                    "language_code": language_code,
                    "file_base64": file_b64,
                    "content_type": response.headers.get("content-type", "application/octet-stream"),
                }

        super().__init__(handler=_get_dubbed_audio, **kwargs)
