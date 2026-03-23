import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"


async def _resolve_api_key(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("ELEVENLABS_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "ElevenLabs API key not found. Set ELEVENLABS_API_KEY, pass api_key, or configure an integration."
    )


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

        super().__init__(handler=_text_to_speech, **kwargs)


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
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
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
            api_key = await _resolve_api_key(self)
            import httpx

            payload: dict[str, Any] = {
                "agent_id": agent_id,
                "agent_phone_number_id": agent_phone_number_id,
                "to_number": to_number,
            }

            async with httpx.AsyncClient() as client:
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
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
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
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
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
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {"page_size": page_size}
            if source:
                params["source"] = source

            async with httpx.AsyncClient() as client:
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
            api_key = await _resolve_api_key(self)
            import base64
            import io
            import zipfile

            import httpx

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
            api_key = await _resolve_api_key(self)
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

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/voices/add",
                    headers={"xi-api-key": api_key},
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_add_voice, **kwargs)
