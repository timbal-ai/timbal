import asyncio
import base64
import json
import os
import wave
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
import websockets
from dotenv import load_dotenv
from pydantic import PrivateAttr
from timbal.core.agent import Agent
from timbal.errors import APIKeyNotFoundError

from timbal.state import RunContext, set_run_context

load_dotenv()


class VoiceAgent(Agent):
    # OpenAI
    _openai_transcribe_ws = PrivateAttr(default=None)
    _openai_transcribe_ws_listener_task = PrivateAttr(default=None)
    _openai_transcribe_session_ready = PrivateAttr(default=False)
    _openai_transcribe_event_buffer = PrivateAttr(default_factory=list)

    # Elevenlabs
    _elevenlabs_tts_ws = PrivateAttr(default=None)
    _elevenlabs_tts_task = PrivateAttr(default=None)
    _elevenlabs_audio_queue = PrivateAttr(default=None)

    # Streaming
    _interruption = PrivateAttr(default=False)
    _interruption_lock = PrivateAttr(default=None)  # NEW: Lock for interruption handling
    _clear_pending = PrivateAttr(default=False)     # NEW: Track if clear is in flight

    _twilio_ws = PrivateAttr(default=None)
    _twilio_stream_sid = PrivateAttr(default=None)

    _agent_call_task = PrivateAttr(default=None)
    _run_id: str = PrivateAttr(default=None)


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

        self._elevenlabs_keepalive_task = None
        

    async def _elevenlabs_keepalive(self):
        try:
            while True:
                await asyncio.sleep(10)  
                if self._elevenlabs_tts_ws:
                    await self._elevenlabs_tts_ws.send(json.dumps({"text": " "}))
        except asyncio.CancelledError:
            pass


    async def _init_openai_transcribe_ws(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise APIKeyNotFoundError("OPENAI_API_KEY not found.")

        self._openai_transcribe_ws = await websockets.connect(
            # "wss://api.openai.com/v1/realtime",
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
        print("OpenAI transcription WebSocket initialized")
        self._openai_transcribe_ws_listener_task = asyncio.create_task(self._listen_openai_transcribe_ws())


    async def _init_elevenlabs_tts_ws(self) -> None:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise APIKeyNotFoundError("ELEVENLABS_API_KEY not found.")
        self._elevenlabs_tts_ws = await websockets.connect(
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_type}/stream-input?model_id={self.elevenlabs_model_id}&language_code={self.language}&output_format={self.elevenlabs_output_format}",
            # f"wss://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_type}/stream-input?model_id={self.elevenlabs_model_id}&language_code={self.language}&output_format={self.elevenlabs_output_format}",
            additional_headers={"xi-api-key": api_key},
            ping_interval=20,
            ping_timeout=10,
            compression=None,
            max_queue=1
        )
        print("Elevenlabs TTS WebSocket initialized")
        self._elevenlabs_tts_task = asyncio.create_task(self._listen_elevenlabs_tts_ws())

        self._elevenlabs_keepalive_task = asyncio.create_task(self._elevenlabs_keepalive())



    async def _process_openai_transcribe_buffered_events(self) -> None:
        for event in self._openai_transcribe_event_buffer:
            await self._openai_transcribe_ws.send(json.dumps(event))
        self._openai_transcribe_event_buffer.clear()


    async def _run_agent_and_tts(self, prompt: str) -> None:
        async with self._interruption_lock:  # Use lock here
            self._interruption = False
            
        if self._run_id is None:
            run_context = RunContext()
        else:
            run_context = RunContext(parent_id=self._run_id)
        self._run_id = run_context.id
        set_run_context(run_context)
        try:
            await self._elevenlabs_tts_ws.send(json.dumps({"text": " "})) # Weird elevenlabs things
            async for event in self(prompt=prompt):
                # Sending to elevenlabs ws
                if event.type == "CHUNK" and event.chunk:
                    async with self._interruption_lock:
                        if not self._interruption:
                            print("event.chunk: ", event.chunk)
                            await self._elevenlabs_tts_ws.send(json.dumps({"text": event.chunk}))
        except asyncio.CancelledError:
            pass
        


    async def _handle_openai_transcribe_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "transcription_session.updated":
            self._openai_transcribe_session_ready = True
            await self._process_openai_transcribe_buffered_events()

        elif event_type == "conversation.item.input_audio_transcription.completed":
            async with self._interruption_lock:
                while self._interruption or self._clear_pending:
                    await asyncio.sleep(0)
            self._agent_call_task = asyncio.create_task(self._run_agent_and_tts(prompt=event["transcript"]))

        elif event_type == "input_audio_buffer.speech_started":
            async with self._interruption_lock:
                if self._interruption or self._clear_pending:
                    print("⚠️ Already interrupting")
                    return
                self._interruption = True
                self._clear_pending = True
            
            # STEP 1: CLEAR FIRST to drop buffer
            if self._twilio_ws and self._twilio_stream_sid:
                try:
                    # Send clear IMMEDIATELY
                    await self._twilio_ws.send_json({
                        "event": "clear",
                        "streamSid": self._twilio_stream_sid
                    })
                    print("Clear sent")
                                        

                    sample_rate = 8000
                    silence_duration_ms = 50  # 50ms chunks
                    silence_samples = (sample_rate * silence_duration_ms) // 1000
                    silence_chunk = bytes([0xFF] * silence_samples)
                    
                    print("🔇 FLOODING WITH SILENCE")
                    # Send 20 bursts = 1 full second of silence
                    for i in range(20):
                        encoded_silence = base64.b64encode(silence_chunk).decode('ascii')
                        await self._twilio_ws.send_json({
                            "event": "media",
                            "streamSid": self._twilio_stream_sid,
                            "media": {
                                "payload": encoded_silence
                            }
                        })
                        if i % 10 == 0: 
                            print(f"🔇 Silence burst {i+1}/20")
                    
                    print("✅ Silence flood complete")
                except Exception as e:
                    print(f"❌ Failed to interrupt: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ No Twilio WebSocket available!")
            
            # STEP 2: Cancel agent task
            if self._agent_call_task and not self._agent_call_task.done():
                print("❌ Cancelling agent")
                self._agent_call_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._agent_call_task
                self._agent_call_task = None
            
            # STEP 3: Drain queue
            drained = 0
            while True:
                try:
                    self._elevenlabs_audio_queue.get_nowait()
                    drained += 1
                except asyncio.QueueEmpty:
                    break
            print(f"🗑️ Drained {drained} queued chunks")
            
            # STEP 4: Flush ElevenLabs
            try:
                await self._elevenlabs_tts_ws.send(json.dumps({"text": "", "flush": True}))
            except Exception as e:
                print(f"⚠️ Failed to flush ElevenLabs: {e}")
            
            # STEP 5: Reset flags
            async with self._interruption_lock:
                self._interruption = False
                self._clear_pending = False
            
            print("✅ INTERRUPTION COMPLETE")

        # I1 (agent)
        # elif event_type == "input_audio_buffer.speech_started" and self._agent_call_task and not self._agent_call_task.done() and self._elevenlabs_audio_queue.empty():
        #     print("I1")
        #     # Detected new turn
        #     async with self._interruption_lock:
        #         if self._interruption or self._clear_pending:
        #             print("Already handling interruption")
        #             return
                
        #         self._interruption = True
        #         self._clear_pending = True

        #     self._agent_call_task.cancel()  # Interrupt agent call
        #     with suppress(asyncio.CancelledError):
        #         await self._agent_call_task  
        #         self._agent_call_task = None
        #     while True:
        #         try:
        #             self._elevenlabs_audio_queue.get_nowait()
        #         except asyncio.QueueEmpty:
        #             print("Queue empty")
        #             break
        #     # adding silence
        #     silence_duration_ms = 500
        #     sample_rate = 8000  # ulaw_8000 format
        #     silence_samples = (sample_rate * silence_duration_ms) // 1000
        #     silence = bytes([0xFF] * silence_samples)
        #     await self._elevenlabs_audio_queue.put(silence)

        #     await self._elevenlabs_audio_queue.put("clear")
        #     # await self._elevenlabs_tts_ws.send(json.dumps({"flush": True}))
        #     await asyncio.sleep(0)
        #     await self._elevenlabs_tts_ws.send(json.dumps({"text": " "}))
        #     async with self._interruption_lock:
        #         self._interruption = False
        
        # # I2 (agent + elevenlabs)
        # elif event_type == "input_audio_buffer.speech_started" and self._agent_call_task and not self._agent_call_task.done() and not self._elevenlabs_audio_queue.empty():
        #     print("I2")
        #     self._interruption = True
        #     self._agent_call_task.cancel()  # Interrupt agent call
        #     with suppress(asyncio.CancelledError):
        #         await self._agent_call_task  
        #         self._agent_call_task = None
        #     while True:
        #         try:
        #             self._elevenlabs_audio_queue.get_nowait()
        #         except asyncio.QueueEmpty:
        #             print("Queue empty")
        #             await self._elevenlabs_audio_queue.put("clear")
        #             break
        #     # await self._elevenlabs_tts_ws.send(json.dumps({"text": "", "flush": True})) 
        #     self._interruption = False

        # # I3 (elevenlabs + twilio)
        # elif event_type == "input_audio_buffer.speech_started" and not self._elevenlabs_audio_queue.empty():
        #     print("I3")
        #     self._interruption = True     
        #     while True:
        #         try:
        #             self._elevenlabs_audio_queue.get_nowait()
        #         except asyncio.QueueEmpty:
        #             print("Queue empty")
        #             await self._elevenlabs_audio_queue.put("clear")
        #             break
        #     # await self._elevenlabs_tts_ws.send(json.dumps({"text": "", "flush": True})) 
        #     await self._elevenlabs_tts_ws.send(json.dumps({"text": " "})) 
        #     self._interruption = False


    async def _listen_openai_transcribe_ws(self) -> None:
        try:
            async for event in self._openai_transcribe_ws:
                try:
                    event = json.loads(event)
                    await self._handle_openai_transcribe_event(event)
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {event}")
        except websockets.exceptions.ConnectionClosedOK as e:
            print("OpenAI transcription WebSocket connection closed OK", e)
        except websockets.exceptions.ConnectionClosedError as e:
            print("OpenAI transcription WebSocket connection closed Error", e)
        except Exception as e:
            print(f"Error listening to OpenAI transcription WebSocket: {e}")


    async def _listen_elevenlabs_tts_ws(self) -> None:
        print("listening elevenlabs")
        try:
            async for event in self._elevenlabs_tts_ws:
                print("new elevenlabs event")

                try:
                    event = json.loads(event)
                    audio = event.get("audio")
                    if audio:
                        audio_bytes = base64.b64decode(audio) 
                        async with self._interruption_lock:
                            if self._interruption or self._clear_pending:
                                print("skipping audio chunk due to interruption/clear")
                                continue
                        await self._elevenlabs_audio_queue.put(audio_bytes)
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {event}")
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
            if self._openai_transcribe_session_ready:
                await self._openai_transcribe_ws.send(json.dumps(audio_event))
            else:
                self._openai_transcribe_event_buffer.append(audio_event)

    
    @asynccontextmanager
    async def session(self, input_stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Any, None]:
        # Init websocket connections
        if self._openai_transcribe_ws is None:
            await self._init_openai_transcribe_ws()
        if self._elevenlabs_tts_ws is None:
            await self._init_elevenlabs_tts_ws()

        if self._interruption_lock is None:
            self._interruption_lock = asyncio.Lock()
        
        # Initialize the audio queue in the running event loop
        if self._elevenlabs_audio_queue is None:
            self._elevenlabs_audio_queue = asyncio.Queue(maxsize=50)

        listen_task = asyncio.create_task(self.listen(input_stream))

        async def audio_bytes():
            while True:
                try:
                    print("audio_bytes()")
                    audio_bytes = await self._elevenlabs_audio_queue.get() # wait until something is in the queue
                    yield audio_bytes
                except asyncio.CancelledError:
                    break
        try:
            yield audio_bytes()
        finally:
            # Cancel the listen task
            if listen_task:
                listen_task.cancel()
                with suppress(asyncio.CancelledError):
                    await listen_task

            # Cancel any running agent task
            if self._agent_call_task:
                self._agent_call_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._agent_call_task

            # Clear OpenAI websocket
            if self._openai_transcribe_ws_listener_task:
                self._openai_transcribe_ws_listener_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._openai_transcribe_ws_listener_task
            
            if self._openai_transcribe_ws:
                await self._openai_transcribe_ws.close()
                self._openai_transcribe_ws = None
                self._openai_transcribe_session_ready = False
            
            # Clear ElevenLabs websocket
            if self._elevenlabs_tts_task:
                self._elevenlabs_tts_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._elevenlabs_tts_task
            
            if self._elevenlabs_tts_ws:
                await self._elevenlabs_tts_ws.close()
                self._elevenlabs_tts_ws = None
            
            # Clear the queue for next session
            self._elevenlabs_audio_queue = None
            


async def stream_audio_file(audio_file: str = "audio.wav", chunk_duration_ms: int = 100) -> AsyncGenerator[bytes, None]:
    with wave.open(audio_file, "rb") as wav_file:
        frames_per_second = wav_file.getframerate()
        frames_per_chunk = int(frames_per_second * chunk_duration_ms / 1000)
        chunk_delay = chunk_duration_ms / 1000  # Convert to seconds
        try:
            while True:
                audio_data = wav_file.readframes(frames_per_chunk)
                if not audio_data:
                    break
                yield audio_data
                # Wait for the duration of this chunk before sending the next one
                await asyncio.sleep(chunk_delay)
        except asyncio.CancelledError:
            # Clean up and re-raise CancelledError
            raise



async def main():
    agent = VoiceAgent(
        name="voice_agent",
        model="openai/gpt-4o-mini"
    )

    try:
        audio_stream = stream_audio_file(audio_file="/Users/julia/Desktop/timbal/bruno_being_bruno_with_silence.wav")
        async with agent.session(input_stream=audio_stream) as session:
            async for event in session:
                print("event: ", event)
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    asyncio.run(main())