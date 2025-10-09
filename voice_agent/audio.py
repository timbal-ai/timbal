import asyncio
import base64
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

import structlog
import websockets
from dotenv import load_dotenv
from pydantic import PrivateAttr
from timbal.core.agent import Agent
from timbal.errors import APIKeyNotFoundError

from timbal.state import RunContext, set_run_context

load_dotenv()
logger = structlog.get_logger(__name__)


class VoiceAgent(Agent):
    # OpenAI
    _openai_transcribe_ws = PrivateAttr(default=None)
    _openai_transcribe_ws_listener_task = PrivateAttr(default=None)

    # Elevenlabs
    _elevenlabs_tts_ws = PrivateAttr(default=None)
    _elevenlabs_tts_task = PrivateAttr(default=None)
    _elevenlabs_keepalive_task = PrivateAttr(default=None)
    _elevenlabs_audio_queue = PrivateAttr(default=None)

    _twilio_ws = PrivateAttr(default=None)
    _twilio_stream_sid = PrivateAttr(default=None)

    _agent_call_task = PrivateAttr(default=None)
    _run_id: str = PrivateAttr(default=None)

    _speech_started_item_id: str | None = PrivateAttr(default=None)
    _speech_transcript: str = PrivateAttr(default="")


    def __init__(
        self,
        openai_model_ws: str = "gpt-4o-mini-transcribe",
        language: str = "en",
        vad_threshold: float = 0.6,
        vad_prefix_padding_ms: int = 300,
        vad_silence_duration_ms: int = 500,
        audio_format: str = "pcm16",
        noise_reduction_type: str = "far_field",
        elevenlabs_voice_type: str = "21m00Tcm4TlvDq8ikWAM",
        elevenlabs_model_id: str = "eleven_flash_v2_5",
        elevenlabs_output_format: str = "ulaw_8000",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.openai_model = openai_model_ws
        self.language = language
        self.vad_threshold = vad_threshold
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.audio_format = audio_format
        self.noise_reduction_type = noise_reduction_type
        self.elevenlabs_voice_type = elevenlabs_voice_type
        self.elevenlabs_model_id = elevenlabs_model_id
        self.elevenlabs_output_format = elevenlabs_output_format

        
    async def _elevenlabs_keepalive(self):
        while True:
            await asyncio.sleep(10)  
            if self._elevenlabs_tts_ws:
                await self._elevenlabs_tts_ws.send(json.dumps({"text": " "}))


    async def _init_openai_transcribe_ws(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise APIKeyNotFoundError("OPENAI_API_KEY not found.")
        self._openai_transcribe_ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime?intent=transcription",
            additional_headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            ping_interval=20,
            ping_timeout=10,
            compression=None,
            max_queue=1
        )
        session_config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": self.audio_format,
                "input_audio_transcription": {
                    "model": self.openai_model,
                    "language": self.language
                },
                "turn_detection": {"type": "server_vad", "threshold": self.vad_threshold, "prefix_padding_ms": self.vad_prefix_padding_ms, "silence_duration_ms": self.vad_silence_duration_ms},
                "input_audio_noise_reduction": {"type": self.noise_reduction_type},
            },
        }
        await self._openai_transcribe_ws.send(json.dumps(session_config))
        self._openai_transcribe_ws_listener_task = asyncio.create_task(self._listen_openai_transcribe_ws())


    async def _init_elevenlabs_tts_ws(self) -> None:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise APIKeyNotFoundError("ELEVENLABS_API_KEY not found.")
        self._elevenlabs_tts_ws = await websockets.connect(
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_type}/stream-input?model_id={self.elevenlabs_model_id}&language_code={self.language}&output_format={self.elevenlabs_output_format}",
            additional_headers={"xi-api-key": api_key},
            ping_interval=20,
            ping_timeout=10,
            compression=None,
            max_queue=1
        )
        self._elevenlabs_tts_task = asyncio.create_task(self._listen_elevenlabs_tts_ws())
        await self._elevenlabs_tts_ws.send(json.dumps({"text": " "})) # Weird elevenlabs things


    async def _run_agent_and_tts(self, prompt: str) -> None:
        await self._init_elevenlabs_tts_ws()
        logger.info("run_agent_and_tts", parent_run_id=self._run_id)
        if self._run_id is None:
            run_context = RunContext()
        else:
            run_context = RunContext(parent_id=self._run_id)
        self._run_id = run_context.id
        set_run_context(run_context)
        async for event in self(prompt=prompt):
            if event.type == "CHUNK" and event.chunk and self._elevenlabs_tts_ws:
                await self._elevenlabs_tts_ws.send(json.dumps({"text": event.chunk}))
        await self._elevenlabs_tts_ws.send(json.dumps({"text": "", "flush": True}))
        await self._elevenlabs_tts_ws.close()


    async def _handle_openai_transcribe_event(self, event: dict) -> None:
        event_type = event.get("type")

        if event_type == "conversation.item.input_audio_transcription.completed":
            logger.info("transcription_completed", full_event=event)
            self._speech_transcript += f"{event['transcript']} "
            transcript = self._speech_transcript
            if self._speech_started_item_id != event["item_id"]:
                return
            self._speech_transcript = ""
            self._agent_call_task = asyncio.create_task(self._run_agent_and_tts(prompt=transcript.strip()))

        elif event_type == "input_audio_buffer.speech_started":
            logger.info("speech_started")
            self._speech_started_item_id = event["item_id"]
            if not self._agent_call_task:
                return

            self._audio_paused.clear()

            if self._agent_call_task and not self._agent_call_task.done():
                logger.info("cancelling_agent")
                self._agent_call_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._agent_call_task
            self._agent_call_task = None

            # Send clear 
            await self._twilio_ws.send_json({
                "event": "clear",
                "streamSid": self._twilio_stream_sid
            })
            logger.info("clear_sent")
            
            drained = 0
            while True:
                try:
                    self._elevenlabs_audio_queue.get_nowait()
                    drained += 1
                except asyncio.QueueEmpty:
                    break
            logger.info(f"drained {drained} queued chunks")
            
            self._audio_paused.set()
            logger.info("interruption_complete")
        else:
            logger.warning(f"Unhandled openai stt event: {event}")


    async def _listen_openai_transcribe_ws(self) -> None:
        try:
            async for event in self._openai_transcribe_ws:
                try:
                    event = json.loads(event)
                    await self._handle_openai_transcribe_event(event)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {event}")
        except websockets.exceptions.ConnectionClosedOK as e:
            print("OpenAI transcription WebSocket connection closed OK", e)
        except websockets.exceptions.ConnectionClosedError as e:
            print("OpenAI transcription WebSocket connection closed Error", e)
        except Exception as e:
            print(f"Error listening to OpenAI transcription WebSocket: {e}")


    async def _listen_elevenlabs_tts_ws(self) -> None:
        try:
            async for event in self._elevenlabs_tts_ws:
                try:
                    event = json.loads(event)
                    audio = event.get("audio")
                    if audio:
                        audio_bytes = base64.b64decode(audio) 
                        await self._elevenlabs_audio_queue.put(audio_bytes)
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {event}")
        except websockets.exceptions.ConnectionClosedOK as e:
            print("Elevenlabs WebSocket connection closed OK", e)
        except websockets.exceptions.ConnectionClosedError:
            print("Elevenlabs WebSocket connection closed")
        except Exception as e:
            print(f"Error listening to Elevenlabs WebSocket: {e}")


    async def listen(self, async_gen: AsyncGenerator[Any, None]) -> None:
        async for chunk in async_gen:
            audio_chunk = base64.b64encode(chunk).decode("ascii") # utf-8
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": audio_chunk
            }
            await self._openai_transcribe_ws.send(json.dumps(audio_event))
    

    @asynccontextmanager
    async def session(self, input_stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Any, None]:
        await self._init_openai_transcribe_ws()
        # await self._init_elevenlabs_tts_ws()
        self._elevenlabs_audio_queue = asyncio.Queue()
        self._audio_paused = asyncio.Event()
        self._audio_paused.set()  # Start unpaused
        self._elevenlabs_keepalive_task = asyncio.create_task(self._elevenlabs_keepalive())

        # Create a task to consume the input stream and send audio chunks to stt
        listen_task = asyncio.create_task(self.listen(input_stream))

        async def _consume_elevenlabs_audio_queue():
            chunk_buffer = bytearray()
            chunk_size_bytes = 160  # 20ms at 8kHz (8000 samples/sec * 0.02 sec = 160 bytes)
            sleep_time = 0.02  # 19ms for ~5% speedup (20ms / 1.05)
            while True:
                await self._audio_paused.wait()
                # Get audio from queue (this blocks until data is available)
                audio_chunk = await self._elevenlabs_audio_queue.get()
                chunk_buffer.extend(audio_chunk)
                # Send complete chunks at controlled pace
                while len(chunk_buffer) >= chunk_size_bytes:
                    chunk = bytes(chunk_buffer[:chunk_size_bytes])
                    chunk_buffer = chunk_buffer[chunk_size_bytes:]
                    yield chunk
                    # Sleep to maintain real-time pacing (slightly faster)
                    await asyncio.sleep(sleep_time)

        try:
            yield _consume_elevenlabs_audio_queue()
        finally:
            listen_task.cancel()
            self._agent_call_task.cancel()
            self._openai_transcribe_ws_listener_task.cancel()
            self._elevenlabs_tts_task.cancel()
            await self._openai_transcribe_ws.close()
            await self._elevenlabs_tts_ws.close()
            