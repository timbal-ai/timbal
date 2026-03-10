import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"


async def _resolve_api_key(tool: Any) -> str:
    """Resolve ElevenLabs API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("ELEVENLABS_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "ElevenLabs API key not found. Set ELEVENLABS_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class TextToSpeech(Tool):
    name: str = "elevenlabs_text_to_speech"
    description: str | None = (
        "Convert text to speech using a specified ElevenLabs voice. "
        "Returns the audio encoded as a base64 string."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
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
            api_key = await _resolve_api_key(self)
            import base64
            import httpx

            async with httpx.AsyncClient() as client:
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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "ElevenLabs/TextToSpeech"
        super().__init__(handler=_text_to_speech, metadata=metadata, **kwargs)


class MakeOutboundCall(Tool):
    name: str = "elevenlabs_make_outbound_call"
    description: str | None = (
        "Initiate an outbound phone call via Twilio using an ElevenLabs conversational agent."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _make_outbound_call(
            agent_id: str = Field(..., description="The ElevenLabs conversational agent ID."),
            to_number: str = Field(..., description="Destination phone number in E.164 format, e.g. '+14155552671'."),
            from_number: str = Field(..., description="Twilio phone number in E.164 format to place the call from."),
            agent_phone_number_id: str | None = Field(None, description="Optional phone number ID registered on the agent."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "agent_id": agent_id,
                "to": to_number,
                "from": from_number,
            }
            if agent_phone_number_id:
                payload["agent_phone_number_id"] = agent_phone_number_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/convai/twilio/outbound-call",
                    headers={"xi-api-key": api_key},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "ElevenLabs/MakeOutboundCall"
        super().__init__(handler=_make_outbound_call, metadata=metadata, **kwargs)


class GetModels(Tool):
    name: str = "elevenlabs_get_models"
    description: str | None = "List all available ElevenLabs TTS models with their capabilities."
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_models() -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/models",
                    headers={"xi-api-key": api_key},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "ElevenLabs/GetModels"
        super().__init__(handler=_get_models, metadata=metadata, **kwargs)


class GetVoicesWithDescriptions(Tool):
    name: str = "elevenlabs_get_voices_with_descriptions"
    description: str | None = (
        "Fetch all available voices from ElevenLabs including metadata "
        "such as name, gender, accent, age, and category."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_voices_with_descriptions(
            show_legacy: bool = Field(False, description="Include legacy pre-XTTS voices in the response."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "ElevenLabs/GetVoicesWithDescriptions"
        super().__init__(handler=_get_voices_with_descriptions, metadata=metadata, **kwargs)


class GetAudioFromHistoryItem(Tool):
    name: str = "elevenlabs_get_audio_from_history_item"
    description: str | None = (
        "Retrieve the audio of a history item and return it as a base64-encoded file."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_audio_from_history_item(
            history_item_id: str = Field(..., description="The ID of the history item to retrieve audio for."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import base64
            import httpx

            async with httpx.AsyncClient() as client:
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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "ElevenLabs/GetAudioFromHistoryItem"
        super().__init__(handler=_get_audio_from_history_item, metadata=metadata, **kwargs)


class DownloadHistoryItems(Tool):
    name: str = "elevenlabs_download_history_items"
    description: str | None = (
        "Download one or more history items. A single item returns an audio file (base64). "
        "Multiple items are packed into a .zip file and returned as base64."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_history_items(
            history_item_ids: list[str] = Field(..., description="List of history item IDs to download. Single ID returns audio; multiple IDs return a .zip."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import base64
            import httpx
            import io
            import zipfile

            async with httpx.AsyncClient() as client:
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

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "ElevenLabs/DownloadHistoryItems"
        super().__init__(handler=_download_history_items, metadata=metadata, **kwargs)


class CreateAgent(Tool):
    name: str = "elevenlabs_create_agent"
    description: str | None = (
        "Create a new ElevenLabs conversational AI agent with a given name, "
        "prompt, voice, and optional first message."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
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
            api_key = await _resolve_api_key(self)
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

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/convai/agents/create",
                    headers={"xi-api-key": api_key},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "ElevenLabs/CreateAgent"
        super().__init__(handler=_create_agent, metadata=metadata, **kwargs)


class AddVoice(Tool):
    name: str = "elevenlabs_add_voice"
    description: str | None = (
        "Clone or add a new voice to ElevenLabs from one or more base64-encoded audio files."
    )
    integration: Annotated[str, Integration("elevenlabs")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_voice(
            name: str = Field(..., description="Display name for the new voice."),
            audio_files_base64: list[str] = Field(..., description="List of audio file contents encoded as base64 strings."),
            filenames: list[str] | None = Field(None, description="Optional list of original filenames (e.g. ['sample1.mp3']). Must match the length of audio_files_base64 if provided."),
            description: str | None = Field(None, description="Optional text description of the voice."),
            labels: dict[str, str] | None = Field(None, description="Optional key-value metadata, e.g. {'accent': 'British', 'gender': 'female'}."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            resolved_names = filenames or [f"sample_{i}.mp3" for i in range(len(audio_files_base64))]

            files: list[tuple[str, Any]] = []
            for i, b64 in enumerate(audio_files_base64):
                audio_bytes = base64.b64decode(b64)
                fname = resolved_names[i] if i < len(resolved_names) else f"sample_{i}.mp3"
                files.append(("files", (fname, audio_bytes, "audio/mpeg")))

            data: dict[str, Any] = {"name": name}
            if description:
                data["description"] = description
            if labels:
                import json
                data["labels"] = json.dumps(labels)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/voices/add",
                    headers={"xi-api-key": token},
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "ElevenLabs/AddVoice"
        super().__init__(handler=_add_voice, metadata=metadata, **kwargs)
