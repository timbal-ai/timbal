import asyncio
import base64
import json
import wave
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any
import numpy as np
import audioop

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
        openai_model_ws: str = "gpt-4o-transcribe",
        language: str = "es",
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
        self.elevenlabs_buffer = bytearray()

        
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

        elif event_type == "input_audio_buffer.speech_stopped":
            await self._init_elevenlabs_tts_ws()
            # await self._elevenlabs_tts_ws.send(json.dumps({"text": "mmmmh...          "}))
            # TODO: keepalive elevenlabs until agent transcription completed. Kill the task then.

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
            # if self._twilio_ws:
            #     await self._twilio_ws.send_json({
            #         "event": "clear",
            #         "streamSid": self._twilio_stream_sid
            #     })
            #     logger.info("clear_sent")
            
            await self._elevenlabs_tts_ws.close()
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
            audio_chunk = base64.b64encode(chunk).decode("utf-8") # ascii
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": audio_chunk
            }
            await self._openai_transcribe_ws.send(json.dumps(audio_event))


    async def audio_chunk_generator(self, wave_file: str, samplerate: int=8000, samplewidth: int=1, channels: int=1, chunk_size=160):
        with wave.open(wave_file, "rb") as wf:
            print(wf.getframerate(), samplerate)
            print(wf.getsampwidth(), samplewidth)
            print(wf.getnchannels(), channels)
            # assert wf.getframerate() == samplerate, "Sample rates diferentes"
            # assert wf.getsampwidth() == samplewidth, "Profundidad de bits diferente"
            # assert wf.getnchannels() == channels, "Número de canales diferente"

            while True:
                chunk = wf.readframes(chunk_size // samplewidth)
                if len(chunk) < chunk_size:
                    # Not enough data: rewind and read the rest
                    wf.rewind()
                    remaining = chunk_size - len(chunk)
                    chunk += wf.readframes(remaining // samplewidth)
                yield chunk


    async def combine_samples(self, sample1: bytes, sample2: bytes, intensity: float=0.3):
       
        def decode_ulaw_sample(sample: bytes, is_ulaw: bool=True):
            if sample:
                # Convert ulaw to linear PCM16
                if is_ulaw:
                    linear = audioop.ulaw2lin(sample, 2)  # 2 bytes = 16-bit
                else:
                    linear = sample
                return np.frombuffer(linear, dtype=np.int16).astype(np.int32)
            return None

        def encode_ulaw_sample(sample: np.ndarray):
            # Clip to 16-bit range and convert to bytes
            pcm16 = np.clip(sample, -32768, 32767).astype(np.int16).tobytes()
            # Convert linear PCM16 to ulaw
            return audioop.lin2ulaw(pcm16, 2)  # 2 bytes = 16-bit

        s1 = decode_ulaw_sample(sample1)
        s2 = decode_ulaw_sample(sample2, is_ulaw=False)


        if s1 is None:
            return encode_ulaw_sample(s2 * intensity)
        
        mixed = s1 + s2 * intensity
        return encode_ulaw_sample(mixed)


    @asynccontextmanager
    async def session(self, input_stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Any, None]:
        await self._init_openai_transcribe_ws()
        self._elevenlabs_audio_queue = asyncio.Queue()
        self._audio_paused = asyncio.Event()
        self._audio_paused.set()  # Start unpaused
        self._elevenlabs_keepalive_task = asyncio.create_task(self._elevenlabs_keepalive())


        # Create a task to consume the input stream and send audio chunks to stt
        listen_task = asyncio.create_task(self.listen(input_stream))

        async def _consume_elevenlabs_audio_queue():
            bg_audio_gen = self.audio_chunk_generator("../typing-8khz.wav", samplewidth=1)
            chunk_buffer = bytearray()
            chunk_size_bytes = 160  # 20ms at 8kHz mulaw (8000 samples/sec * 0.02 sec * 1 byte/sample = 160 bytes)
            sleep_time = 0.0  # 19ms for ~5% speedup (20ms / 1.05)ge
            bg_intensity = 0.3

            async for bg_audio_chunk in bg_audio_gen:
                if not self._audio_paused.is_set():
                    chunk_buffer.clear()
                if self._audio_paused.is_set() and not self._elevenlabs_audio_queue.empty():
                    audio_chunk = await self._elevenlabs_audio_queue.get()
                    chunk_buffer.extend(audio_chunk)
                    print("received")
                if len(chunk_buffer) >= chunk_size_bytes:
                    chunk = bytes(chunk_buffer[:chunk_size_bytes])
                    del chunk_buffer[:chunk_size_bytes]
                    yield await self.combine_samples(chunk, bg_audio_chunk, bg_intensity)
                else:
                    yield await self.combine_samples(None, bg_audio_chunk, bg_intensity)
                await asyncio.sleep(sleep_time)                

        try:
            yield _consume_elevenlabs_audio_queue()
        finally:
            listen_task.cancel()
            with suppress(asyncio.CancelledError):
                await listen_task
            
            if self._agent_call_task:
                self._agent_call_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._agent_call_task
            
            if self._openai_transcribe_ws_listener_task:
                self._openai_transcribe_ws_listener_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._openai_transcribe_ws_listener_task
            
            if self._elevenlabs_tts_task:
                self._elevenlabs_tts_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._elevenlabs_tts_task
            
            if self._elevenlabs_keepalive_task:
                self._elevenlabs_keepalive_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._elevenlabs_keepalive_task
            
            # Close websockets gracefully
            if self._openai_transcribe_ws:
                with suppress(Exception):
                    await self._openai_transcribe_ws.close()
            
            if self._elevenlabs_tts_ws:
                with suppress(Exception):
                    await self._elevenlabs_tts_ws.close()
            