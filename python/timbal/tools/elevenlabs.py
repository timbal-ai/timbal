import base64
import io
import zipfile
from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"


class TextToSpeech(Tool):
    name: str = "elevenlabs_text_to_speech"
    description: str | None = (
        "Convert text to speech using a specified ElevenLabs voice. "
        "Returns the audio encoded as a base64 string."
    )
    integration: Annotated[str, Integration("elevenlabs")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _text_to_speech(
            text: str,
            voice_id: str,
            model_id: str = "eleven_multilingual_v2",
            output_format: str = "mp3_44100_128",
            stability: float = 0.5,
            similarity_boost: float = 0.75,
            style: float = 0.0,
            use_speaker_boost: bool = True,
        ) -> Any:
            """
            text: the text to synthesize.
            voice_id: ElevenLabs voice ID to use.
            model_id: model to use, e.g. "eleven_multilingual_v2", "eleven_turbo_v2".
            output_format: audio format, e.g. "mp3_44100_128", "pcm_16000", "ulaw_8000".
            stability: voice stability (0.0–1.0).
            similarity_boost: voice similarity boost (0.0–1.0).
            style: style exaggeration (0.0–1.0, only for v2 models).
            use_speaker_boost: improve speaker clarity (slight latency increase).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/text-to-speech/{voice_id}",
                    headers={"xi-api-key": token},
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
    integration: Annotated[str, Integration("elevenlabs")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _make_outbound_call(
            agent_id: str,
            to_number: str,
            from_number: str,
            agent_phone_number_id: str | None = None,
        ) -> Any:
            """
            agent_id: the ElevenLabs conversational agent ID.
            to_number: destination phone number in E.164 format, e.g. "+14155552671".
            from_number: Twilio phone number in E.164 format to place the call from.
            agent_phone_number_id: optional phone number ID registered on the agent.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"xi-api-key": token},
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
    integration: Annotated[str, Integration("elevenlabs")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_models() -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/models",
                    headers={"xi-api-key": token},
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
    integration: Annotated[str, Integration("elevenlabs")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_voices_with_descriptions(
            show_legacy: bool = False,
        ) -> Any:
            """
            show_legacy: include legacy pre-XTTS voices in the response.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/voices",
                    headers={"xi-api-key": token},
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
    integration: Annotated[str, Integration("elevenlabs")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_audio_from_history_item(
            history_item_id: str,
        ) -> Any:
            """
            history_item_id: the ID of the history item to retrieve audio for.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ELEVENLABS_BASE}/history/{history_item_id}/audio",
                    headers={"xi-api-key": token},
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
    integration: Annotated[str, Integration("elevenlabs")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_history_items(
            history_item_ids: list[str],
        ) -> Any:
            """
            history_item_ids: list of history item IDs to download.
                              Single ID returns audio; multiple IDs return a .zip.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/history/download",
                    headers={"xi-api-key": token},
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
    integration: Annotated[str, Integration("elevenlabs")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_agent(
            name: str,
            voice_id: str,
            system_prompt: str,
            first_message: str | None = None,
            language: str = "en",
            model_id: str = "eleven_turbo_v2",
            max_duration_seconds: int = 600,
            turn_timeout: int = 20,
        ) -> Any:
            """
            name: display name for the agent.
            voice_id: ElevenLabs voice ID the agent will use.
            system_prompt: the LLM system prompt defining agent behaviour.
            first_message: optional opening message the agent speaks first.
            language: BCP-47 language code, e.g. "en", "es", "fr".
            model_id: TTS model to use, e.g. "eleven_turbo_v2".
            max_duration_seconds: maximum call/conversation length in seconds.
            turn_timeout: seconds of silence before the agent ends its turn.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
                    headers={"xi-api-key": token},
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
    integration: Annotated[str, Integration("elevenlabs")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _add_voice(
            name: str,
            audio_files_base64: list[str],
            filenames: list[str] | None = None,
            description: str | None = None,
            labels: dict[str, str] | None = None,
        ) -> Any:
            """
            name: display name for the new voice.
            audio_files_base64: list of audio file contents encoded as base64 strings.
            filenames: optional list of original filenames (e.g. ["sample1.mp3"]).
                       Must match the length of audio_files_base64 if provided.
            description: optional text description of the voice.
            labels: optional key-value metadata, e.g. {"accent": "British", "gender": "female"}.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

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
