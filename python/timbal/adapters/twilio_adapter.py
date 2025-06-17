import asyncio
import base64
import json
import websockets
from datetime import datetime
import os
from typing import Dict, Optional, Any, AsyncGenerator, List
from collections import deque
import httpx
import re

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import PrivateAttr, Field, BaseModel, SecretStr
from timbal.state import RunContext
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, VoiceResponse
from timbal.types.message import Message
from timbal.errors import EarlyExit

from .base import BaseAdapter
import structlog
from uuid_extensions import uuid7

logger = structlog.get_logger("timbal.adapters.twilio_adapter")
from dotenv import load_dotenv
load_dotenv()

# Real-time processing configuration
MIN_TRANSCRIPTION_LENGTH = 2

# Audio format constants for ulaw_8000
AUDIO_SAMPLE_RATE = 8000  # Hz
AUDIO_BYTES_PER_SECOND = AUDIO_SAMPLE_RATE  # 8000 bytes per second for ulaw_8000


# --- Config Models ---

class ElevenLabsConfig(BaseModel):
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("ELEVENLABS_API_KEY")))
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    model_id: str = "eleven_multilingual_v2"
    optimize_streaming_latency: int = 2
    speed: float = 1.1
    stability: float = 0.5
    use_speaker_boost: bool = True

class OpenAIConfig(BaseModel):
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY")))
    model: str = "gpt-4o-realtime-preview-2025-06-03"
    transcription_model: str = "gpt-4o-transcribe"
    transcription_language: str = "es"

class TwilioConfig(BaseModel):
    account_sid: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("TWILIO_ACCOUNT_SID")))
    auth_token: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("TWILIO_AUTH_TOKEN")))
    from_phone_number: str = Field(default_factory=lambda: os.getenv("TWILIO_PHONE_NUMBER"))

class TwilioAdapterConfig(BaseModel):
    twilio: TwilioConfig
    openai: OpenAIConfig
    elevenlabs: ElevenLabsConfig
    websocket_url: str
    to_phone_number: str

class CallSession:
    def __init__(self, call_sid: str, stream_sid: str, adapter_config: TwilioAdapterConfig):
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.adapter_config = adapter_config
        self.conversation_count = 0
        self.start_time = datetime.now()
        self.openai_websocket = None
        self.openai_session_ready = False
        self.openai_task = None
        self.last_context_id: Optional[str] = None  # Track last context ID for parent_id chain
        self.text_buffer: str = "" # New state for sentence-based TTS generation
        self.last_request_ids: deque[str] = deque(maxlen=3)
        self.is_generating_audio: bool = False  # Lock to prevent input processing during audio generation

    async def initialize_openai_connection(self):
        try:
            headers = {
                "Authorization": f"Bearer {self.adapter_config.openai.api_key.get_secret_value()}",
                "OpenAI-Beta": "realtime=v1"
            }
            ws_url = f"wss://api.openai.com/v1/realtime?model={self.adapter_config.openai.model}"
            self.openai_websocket = await websockets.connect(ws_url, additional_headers=headers, ping_interval=20, ping_timeout=10)
            session_config = {
                "type": "session.update",
                "session": {
                    "input_audio_format": "g711_ulaw",
                    "input_audio_transcription": {
                        "model": self.adapter_config.openai.transcription_model,
                        "language": self.adapter_config.openai.transcription_language
                    },
                    "turn_detection": {"type": "server_vad", "threshold": 0.6, "prefix_padding_ms": 300, "silence_duration_ms": 500},
                    "input_audio_noise_reduction": {"type": "far_field"},
                },
            }
            await self.openai_websocket.send(json.dumps(session_config))
            self.openai_task = asyncio.create_task(self.handle_openai_events())
            return True
        except Exception as e:
            logger.error(f"Error initializing OpenAI connection: {e}")
            return False

    async def handle_openai_events(self):
        try:
            while self.openai_websocket:
                try:
                    message = await self.openai_websocket.recv()
                    event = json.loads(message)
                    await self.process_openai_event(event)
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            logger.error(f"Error in OpenAI event handler: {e}")

    async def process_openai_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "session.created":
            self.openai_session_ready = True
            logger.info(f"OpenAI session created for call {self.call_sid}")
            # For the first turn, trigger agent to start conversation
            await self._trigger_agent_with_transcription("")
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            logger.info(f"OpenAI transcript received for call {self.call_sid}: '{transcript}'")
            
            if self.validate_transcription(transcript):
                # Check if we're currently generating audio - if so, ignore this transcript
                if self.is_generating_audio:
                    logger.info(f"Ignoring transcript '{transcript}' while generating audio for call {self.call_sid}")
                    return
                
                # Set lock to prevent processing new transcripts while we handle this one
                self.is_generating_audio = True
                logger.info(f"Processing transcript '{transcript}' for call {self.call_sid} - audio generation lock activated")
                await self._trigger_agent_with_transcription(transcript)
            else:
                logger.info(f"Transcript validation failed for call {self.call_sid}: '{transcript}' (length: {len(transcript.strip()) if transcript else 0})")

    def validate_transcription(self, text: str) -> bool:
        if not text or not text.strip():
            # Allow empty transcript for the first turn to make the agent speak first
            is_valid = self.conversation_count == 0
            return is_valid
        
        is_valid = len(text.strip()) >= MIN_TRANSCRIPTION_LENGTH
        return is_valid

    async def _trigger_agent_with_transcription(self, transcript: str):
        """Trigger agent processing with a complete transcription from OpenAI VAD."""
        try:
            adapter = self._adapter_ref
            agent = adapter._agent
            new_context_id = uuid7(as_type="str")
            context = RunContext(
                id=new_context_id,
                parent_id=self.last_context_id  # None for first turn, context_id for subsequent turns
            )

            if transcript.strip():
                prompt = Message.validate({
                    "role": "user", 
                    "content": transcript
                })
                logger.info(f"Sending transcript to agent for call {self.call_sid}: '{transcript}'")
            else:
                prompt = Message.validate({
                    "role": "user", 
                    "content": "Inicia la conversación con un saludo."
                })

            # Run the agent with the transcription and process chunks in real-time
            chunk_count = 0
            agent_response = ""
            hang_up_detected = False
            total_audio_duration = 0.0  # Track total audio duration
            
            async for event in agent.run(context=context, prompt=prompt):
                if (hasattr(event, 'chunk') and hasattr(event, 'path') and 
                    event.path.startswith('agent.llm')):
                    chunk_text = str(event.chunk)
                    if chunk_text and chunk_text.strip():
                        chunk_count += 1
                        agent_response += chunk_text
                        
                        # Check if agent wants to hang up
                        if "{hang_up_call}" in agent_response:
                            hang_up_detected = True
                            logger.info(f"Detected hang_up_call command in agent response for call {self.call_sid}")
                            # Release the lock since we're hanging up
                            self.is_generating_audio = False
                            await adapter.hang_up_call(self.call_sid)
                            return
                        
                        # Process chunk immediately for TTS (only if no hang_up_call detected)
                        else:
                            chunk_duration = await self.process_text_chunk(chunk_text)
                            total_audio_duration += chunk_duration
            
            # Flush any remaining text in the buffer (only if no hang_up was detected)
            if not hang_up_detected:
                final_duration = await self.flush_remaining_text()
                total_audio_duration += final_duration
            
            # Update last context ID for next turn's parent_id and increment conversation count
            self.last_context_id = new_context_id
            self.conversation_count += 1
            
            # Wait for all audio to finish playing before releasing the lock
            if total_audio_duration > 0:
                logger.info(f"Waiting {total_audio_duration:.2f} seconds for all audio to finish playing for call {self.call_sid}")
                await asyncio.sleep(total_audio_duration)
            
            # Release the lock after all audio generation and playback is complete
            self.is_generating_audio = False
            logger.info(f"Audio generation lock released for call {self.call_sid} - ready for next transcript")
            
            if chunk_count == 0:
                logger.warning(f"No text chunks received from agent for call {self.call_sid}")
                
        except Exception as e:
            logger.error(f"Error triggering agent: {e}", exc_info=True)
            self.is_generating_audio = False
            logger.error(f"Audio generation lock released due to error for call {self.call_sid}")

    async def send_audio_chunk_to_openai(self, audio_bytes: bytes):
        if self.openai_websocket and self.openai_session_ready:
            try:
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                await self.openai_websocket.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
                logger.debug(f"Sent audio chunk to OpenAI for call {self.call_sid} (size: {len(audio_bytes)} bytes)")
            except Exception as e:
                logger.error(f"💥 Error sending audio to OpenAI: {e}")

    def add_audio_chunk(self, audio_bytes: bytes):
        """Add audio chunk and send to OpenAI for real-time transcription."""
        # Send to OpenAI if connection is ready
        if self.openai_websocket and self.openai_session_ready:
            logger.debug(f"Adding audio chunk for call {self.call_sid} (size: {len(audio_bytes)} bytes)")
            asyncio.create_task(self.send_audio_chunk_to_openai(audio_bytes))
        else:
            logger.debug(f"OpenAI connection not ready for call {self.call_sid} (websocket: {self.openai_websocket is not None}, session_ready: {self.openai_session_ready})")

    async def cleanup(self):
        logger.info(f"Cleaning up call session {self.call_sid}")
        if self.openai_task: 
            self.openai_task.cancel()
            try:
                await self.openai_task
            except asyncio.CancelledError:
                pass
        if self.openai_websocket: 
            try:
                await self.openai_websocket.close()
            except Exception as e:
                logger.error(f"Error closing OpenAI WebSocket for call {self.call_sid}: {e}")
            finally:
                self.openai_websocket = None
                self.openai_session_ready = False
        
        # Reset audio generation lock
        self.is_generating_audio = False

    def is_audio_generation_active(self) -> bool:
        """Check if audio generation is currently active (locked)."""
        return self.is_generating_audio

    async def process_text_chunk(self, chunk_text: str) -> float:
        """Process a text chunk, buffer it, and generate TTS when a sentence is complete.
        
        Returns:
            float: Total duration of audio generated from this chunk
        """
        self.text_buffer += chunk_text

        # Use a regex to find the end of a sentence or clause based on common punctuation.
        sentence_regex = r'([^.?!;:]*[.?!;:])'
        
        total_duration = 0.0
        while True:
            match = re.search(sentence_regex, self.text_buffer)
            if not match:
                logger.debug("No complete sentence found in buffer yet.")
                break # No complete sentence/clause found, wait for more text
                
            segment_to_send = match.group(0).strip()
            # Update buffer by removing the matched part
            self.text_buffer = self.text_buffer[match.end():]
            
            if segment_to_send:
                logger.info(f"Found text segment to process: '{segment_to_send}'")
                # Generate TTS for the segment and get duration
                duration = await self._generate_and_stream_tts(segment_to_send)
                total_duration += duration
        
        return total_duration

    async def flush_remaining_text(self) -> float:
        """Flush any remaining text in the buffer at the end of an agent turn.
        
        Returns:
            float: Duration of audio generated from remaining text
        """
        if self.text_buffer.strip():
            logger.info(f"Flushing remaining text: '{self.text_buffer.strip()}'")
            duration = await self._generate_and_stream_tts(self.text_buffer)
            self.text_buffer = ""
            return duration
        return 0.0

    async def _generate_and_stream_tts(self, text_segment: str) -> float:
        """Generates TTS using ElevenLabs REST API and streams it back.
        
        Returns:
            float: Duration of the audio in seconds
        """
        adapter = self._adapter_ref

        config = self.adapter_config.elevenlabs
        api_key = config.api_key.get_secret_value()
        
        # Correct endpoint for streaming, as per ElevenLabs documentation
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.voice_id}/stream"
        headers = {"xi-api-key": api_key}
        
        # Query parameters for the request
        params = {
            "output_format": "ulaw_8000",
            "optimize_streaming_latency": config.optimize_streaming_latency,
        }
        
        payload = {
            "text": text_segment.strip(),
            "model_id": config.model_id,
            "voice_settings": {
                "speed": config.speed,
                "stability": config.stability,
                "use_speaker_boost": config.use_speaker_boost
            }
        }
        if self.last_request_ids:
            payload["previous_request_ids"] = list(self.last_request_ids)

        try:
            response = await adapter._elevenlabs_client.post(
                url, params=params, json=payload, headers=headers, timeout=60
            )
            response.raise_for_status()
            
            request_id = response.headers.get("request-id")
            if request_id:
                self.last_request_ids.append(request_id)
            
            audio_data = response.content
            total_bytes = len(audio_data)
            
            # Stream audio in chunks for better performance
            chunk_size = 1024
            for i in range(0, total_bytes, chunk_size):
                chunk = audio_data[i:i + chunk_size]
                b64_audio = base64.b64encode(chunk).decode("utf-8")
                await adapter._send_audio_to_websocket({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": b64_audio}
                })
            
            duration = total_bytes / AUDIO_BYTES_PER_SECOND
            logger.debug(f"Streamed audio for request {request_id}, total bytes: {total_bytes}, duration: {duration:.2f}s")
            return duration
            
        except httpx.HTTPStatusError as e:
            logger.error(f"ElevenLabs API error: {e.response.status_code} - {e.response.text}")
            return 0.0

        except httpx.ReadTimeout:
            logger.error("Timeout reading from ElevenLabs API stream")
            return 0.0
        except Exception as e:
            logger.error(f"Error during TTS generation and streaming: {e}", exc_info=True)
            return 0.0


class TwilioCallAdapter(BaseAdapter):
    type: str = "twilio_adapter"
    config: TwilioAdapterConfig

    _active_calls: Dict[str, "CallSession"] = PrivateAttr(default_factory=dict)
    _twilio_client: Client = PrivateAttr()
    _call_initiated: bool = PrivateAttr(default=False)
    _agent: Optional[Any] = PrivateAttr(default=None)
    _pending_audio_queue: asyncio.Queue = PrivateAttr(default_factory=asyncio.Queue)
    _elevenlabs_client: Optional[httpx.AsyncClient] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize adapter-specific resources after Pydantic validation."""
        self._twilio_client = Client(
            self.config.twilio.account_sid.get_secret_value(), 
            self.config.twilio.auth_token.get_secret_value()
        )
        # Initialize ElevenLabs HTTP client
        self._elevenlabs_client = httpx.AsyncClient()

    def set_agent(self, agent):
        """Set the agent instance for this adapter."""
        self._agent = agent

    async def stop(self):
        """Stop the adapter and clean up resources."""
        # Hang up all active calls
        for call_sid in list(self._active_calls.keys()):
            await self.hang_up_call(call_sid)
        
        # Close HTTPX client
        if self._elevenlabs_client:
            await self._elevenlabs_client.aclose()
            self._elevenlabs_client = None

    async def before_agent(self, context: RunContext) -> None:
        """Called before the agent processes each turn. Handles call initiation."""
        if not self._call_initiated:
            logger.info("Initiating outbound call...")
            call_sid = await self._initiate_call()
            if call_sid:
                self._call_initiated = True
                logger.info(f"Call initiated successfully. Call SID: {call_sid}")
            else:
                logger.error("Failed to initiate call")
            
            # Exit early after call initiation - wait for OpenAI VAD to trigger agent
            raise EarlyExit("Call initiated, waiting for OpenAI VAD to trigger agent")

    async def _initiate_call(self):
        """Initiates an outbound call with media streaming."""
        if not self.config.websocket_url:
            logger.error("WebSocket URL not provided. Cannot make call.")
            return None

        # Automatic conversion from HTTPS to WSS
        if self.config.websocket_url.startswith("https://"):
            wss_url = self.config.websocket_url.replace("https://", "wss://")
            if not wss_url.endswith("/ws"):
                wss_url += "/ws"
        elif self.config.websocket_url.startswith("wss://"):
             wss_url = self.config.websocket_url
             if not wss_url.endswith("/ws"):
                wss_url += "/ws"
        else:
            logger.error("Invalid URL. Must start with 'https://' or 'wss://'")
            return None

        response = VoiceResponse()
        connect = Connect()
        stream = connect.stream(url=wss_url)
        stream.parameter(name="track", value="both_tracks")
        response.append(connect)
        response.pause(length=100)
        
        twiml_str = str(response)
        logger.info(f"WebSocket URL: {wss_url}")
        
        try:
            logger.info(f"Initiating call to {self.config.to_phone_number}")
            call = self._twilio_client.calls.create(
                to=self.config.to_phone_number, 
                from_=self.config.twilio.from_phone_number, 
                twiml=twiml_str
            )
            return call.sid
        except Exception as e:
            logger.error(f"Error initiating call: {e}")
            return None

    async def handle_message(self, message_json: str) -> Optional[List[Dict[str, Any]]]:
        """Routes incoming messages to the appropriate handler and drains the audio queue."""
        try:
            message = json.loads(message_json)
            event_type = message.get("event")

            if event_type == "start":
                await self._handle_call_start(message)
            elif event_type == "media":
                await self._handle_media(message)
            elif event_type == "stop":
                await self._handle_call_stop(message)
                return [{"event": "close"}] # Return as list
            
            # After handling the incoming message, drain the pending audio queue
            pending_responses = []
            while not self._pending_audio_queue.empty():
                try:
                    audio_response = self._pending_audio_queue.get_nowait()
                    pending_responses.append(audio_response)
                except asyncio.QueueEmpty:
                    break
            
            if pending_responses:
                return pending_responses

            return None
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message_json}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
        
        return None

    async def _handle_call_start(self, message: dict):
        """Handles the start of a call."""
        start_data = message.get("start", {})
        call_sid = start_data.get("callSid")
        stream_sid = message.get("streamSid")

        call_session = self._create_call_session(call_sid, stream_sid)
        if not call_session:
            logger.error(f"Failed to create or retrieve call session for {call_sid}")
            return

        await call_session.initialize_openai_connection()
        
    async def _handle_media(self, message: dict):
        """Handles received media (audio) data."""
        media_data = message.get("media", {})
        payload = media_data.get("payload")
        track = media_data.get("track")
        
        if payload and track == "inbound":
            # Get call session from stream_sid in the message
            stream_sid = message.get("streamSid")
            call_session = self._get_call_session_by_stream_sid(stream_sid)
            
            if call_session:
                audio_bytes = base64.b64decode(payload)
                logger.debug(f"Received inbound audio for call {call_session.call_sid} (size: {len(audio_bytes)} bytes)")
                call_session.add_audio_chunk(audio_bytes)
            else:
                logger.warning(f"No call session found for stream_sid: {stream_sid}")
        elif payload and track == "outbound":
            logger.debug(f"Received outbound audio (size: {len(base64.b64decode(payload))} bytes)")
        else:
            logger.debug(f"Received media with no payload or wrong track: track={track}, has_payload={bool(payload)}")

    async def _handle_call_stop(self, message: dict):
        """Handles the end of a call."""
        stop_data = message.get("stop", {})
        call_sid = stop_data.get("callSid")
        logger.info(f"CALL ENDED - CallSid: {call_sid}")

        call_session = self._active_calls.get(call_sid)
        if call_session:
            await call_session.cleanup()
            self._remove_call_session(call_sid)

    async def hang_up_call(self, call_sid: str):
        """Initiates the process of hanging up a specific call."""
        logger.info(f"--- HANG UP PROCESS STARTED for {call_sid} ---")
        try:
            # First, try to update the call status to completed
            call = self._twilio_client.calls(call_sid)
            call.update(status="completed")
            logger.info(f"Successfully updated call {call_sid} status to completed")
            
            # Also try to terminate the call if it's still active
            try:
                call.update(status="busy")
                logger.info(f"Also set call {call_sid} status to busy")
            except Exception as e:
                logger.error(f"Could not set call {call_sid} to busy (might already be terminated): {e}")
                
        except Exception as e:
            if "Call is not in-progress" in str(e):
                logger.warning(f"Call {call_sid} already completed or hung up by other party.")
            elif "not found" in str(e).lower():
                logger.warning(f"Call {call_sid} not found - may have already ended.")
            else:
                logger.error(f"Twilio hang up error for {call_sid}: {e}")
        
        # Clean up the call session regardless of Twilio API result
        call_session = self._active_calls.get(call_sid)
        if call_session:
            logger.info(f"Cleaning up call session for {call_sid}")
            await call_session.cleanup()
            self._remove_call_session(call_sid)
        else:
            logger.warning(f"No call session found for {call_sid} during hang up")

    def _create_call_session(self, call_sid: str, stream_sid: str) -> "CallSession":
        call_session = CallSession(call_sid, stream_sid, self.config)
        # Set adapter reference so CallSession can trigger agent
        call_session._adapter_ref = self
        self._active_calls[call_sid] = call_session
        return call_session

    def _get_call_session_by_stream_sid(self, stream_sid: str) -> Optional["CallSession"]:
        """Get call session by stream_sid."""
        for call_session in self._active_calls.values():
            if call_session.stream_sid == stream_sid:
                return call_session
        return None

    def _remove_call_session(self, call_sid: str):
        if call_sid in self._active_calls:
            logger.info(f"Removing call session for CallSid: {call_sid}")
            del self._active_calls[call_sid]

    async def _send_audio_to_websocket(self, audio_response: dict):
        """Send audio response back to Twilio via WebSocket queue."""
        try:
            # The audio_response should be in Twilio WebSocket format
            if not audio_response:
                logger.warning("Received empty audio response to queue")
                return
                  
            # Add the response to the queue for streaming
            if self._pending_audio_queue.qsize() < 100: # Safety cap
                await self._pending_audio_queue.put(audio_response)
            else:
                logger.warning("Pending audio queue is full, dropping audio chunk.")

        except Exception as e:
            logger.error(f"Error queuing audio to WebSocket: {e}", exc_info=True)



 