import asyncio
import sys
import wave
from collections.abc import AsyncGenerator
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override
from timbal.collectors.base import BaseCollector
import websockets

from dotenv import load_dotenv
load_dotenv()


import base64
import httpx
import os
import json
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

from timbal import Agent
from pydantic import PrivateAttr

class SuperAgent(Agent):
    _openai_transcribe_ws = PrivateAttr(default=None)
    _openai_transcribe_ws_listener_task = PrivateAttr(default=None)
    _openai_transcribe_session_ready = PrivateAttr(default=False)
    _openai_transcribe_event_buffer = PrivateAttr(default_factory=list)

    _agent_call_task = PrivateAttr(default=None)

    async def _init_openai_transcribe_ws(self) -> None:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.openai.com/v1/realtime/transcription_sessions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "assistants=v2",
                },
                json={
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "gpt-4o-mini-transcribe",
                        "language": "en"
                    },
                    "turn_detection": {"type": "server_vad", "threshold": 0.6, "prefix_padding_ms": 300, "silence_duration_ms": 500},
                    "input_audio_noise_reduction": {"type": "far_field"},
                }
            )
            res.raise_for_status()
            res_json = res.json()
            ephemeral_token = res_json["client_secret"]["value"]
            print("Ephemeral token: ", ephemeral_token)

        self._openai_transcribe_ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime",
            additional_headers={
                "Authorization": f"Bearer {ephemeral_token}",
                "OpenAI-Beta": "realtime=v1",
            },
            ping_interval=20,
            ping_timeout=10
        )
        session_config = {
            "type": "transcription_session.update",
            "session": {
                # "input_audio_format": "g711_ulaw",
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gpt-4o-mini-transcribe",
                    "language": "en"
                },
                "turn_detection": {"type": "server_vad", "threshold": 0.6, "prefix_padding_ms": 300, "silence_duration_ms": 500},
                "input_audio_noise_reduction": {"type": "far_field"},
            },
        }
        await self._openai_transcribe_ws.send(json.dumps(session_config))
        print("OpenAI transcription WebSocket initialized")
        self._openai_transcribe_ws_listener_task = asyncio.create_task(self._listen_openai_transcribe_ws())

    async def _process_openai_transcribe_buffered_events(self) -> None:
        for event in self._openai_transcribe_event_buffer:
            await self._openai_transcribe_ws.send(json.dumps(event))
        print(f"Sent {len(self._openai_transcribe_event_buffer)} buffered events")
        self._openai_transcribe_event_buffer.clear()

    async def _handle_openai_transcribe_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "transcription_session.updated":
            print("Session ready! Processing buffered audio...")
            self._openai_transcribe_session_ready = True
            await self._process_openai_transcribe_buffered_events()
        elif event_type == "conversation.item.input_audio_transcription.completed":
            print("Transcription completed: ", event)
            # async def ...:
            #     # Make sure the elevenlabs ws is open and ready
            #     async for event in superagent(prompt="Tell me a story"):
            #         if event.type == "CHUNK":
            #             await tts_ws.send(json.dumps({"text": event.chunk}))
            self._agent_call_task = asyncio.create_task(self(prompt=event["transcript"]).collect())
        else:
            print("Unhandled event: ", event)

    async def _listen_openai_transcribe_ws(self) -> None:
        try:
            async for event in self._openai_transcribe_ws:
                try:
                    event = json.loads(event)
                    await self._handle_openai_transcribe_event(event)
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {event}")
        except websockets.exceptions.ConnectionClosedError:
            print("OpenAI transcription WebSocket connection closed")
        except Exception as e:
            print(f"Error listening to OpenAI transcription WebSocket: {e}")

    async def listen(self, async_gen: AsyncGenerator[Any, None]) -> None:
        if self._openai_transcribe_ws is None:
            await self._init_openai_transcribe_ws()
        # Base64-encode the audio chunk.
        async for chunk in async_gen:
            audio_chunk = base64.b64encode(chunk).decode("utf-8")
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": audio_chunk
            }
            if self._openai_transcribe_session_ready:
                await self._openai_transcribe_ws.send(json.dumps(audio_event))
            else:
                self._openai_transcribe_event_buffer.append(audio_event)

        # TODO: This is a hack to wait for the agent call to finish
        if self._agent_call_task:
            await self._agent_call_task
            
async def stream_audio_file(audio_file: str = "audio.wav", chunk_duration_ms: int = 100) -> AsyncGenerator[bytes, None]:
    with wave.open(audio_file, "rb") as wav_file:
        frames_per_second = wav_file.getframerate()
        frames_per_chunk = int(frames_per_second * chunk_duration_ms / 1000)
        chunk_delay = chunk_duration_ms / 1000  # Convert to seconds
        while True:
            audio_data = wav_file.readframes(frames_per_chunk)
            if not audio_data:
                break
            yield audio_data
            # Wait for the duration of this chunk before sending the next one
            await asyncio.sleep(chunk_delay)


async def test_stream_output():
    tts_ws = await websockets.connect(
        "wss://api.elevenlabs.io/v1/text-to-speech/56AoDkrOh6qfVPDXZ7Pt/stream-input?model_id=eleven_flash_v2_5&language_code=en&output_format=pcm_16000",
        additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
        ping_interval=20,
        ping_timeout=10
    )
    audio_buffer = []
    async def listen_tts_ws() -> None:
        try:
            async for event in tts_ws:
                try:
                    event = json.loads(event)
                    audio = event.get("audio")
                    audio_bytes = base64.b64decode(audio)
                    audio_buffer.append(audio_bytes)
                    # is_final = event.get("isFinal")
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {event}")
        except websockets.exceptions.ConnectionClosedError:
            print("Elevenlabs WebSocket connection closed")
        except Exception as e:
            print(f"Error listening to Elevenlabs WebSocket: {e}")
    asyncio.create_task(listen_tts_ws())
    await tts_ws.send(json.dumps({"text": " "})) # Weird elevenlabs things
    async for event in superagent(prompt="Tell me a story"):
        if event.type == "CHUNK":
            await tts_ws.send(json.dumps({"text": event.chunk}))
    audio_data = b"".join(audio_buffer)
    audio_buffer.clear()
    with wave.open("output.wav", "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit (2 bytes)
        wav_file.setframerate(16000)  # 16kHz sample rate (based on PCM_16000 format)
        wav_file.writeframes(audio_data)


async def main():
    superagent = SuperAgent(
        name="superagent",
        model="openai/gpt-4o-mini",
    )
    async_gen = stream_audio_file(sys.argv[1])
    await superagent.listen(async_gen)
    

if __name__ == "__main__":
    asyncio.run(main())
