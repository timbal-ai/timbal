import asyncio
import base64
import json
import websockets
from datetime import datetime
import os
import audioop
import sys
from typing import Dict, Optional, Any, AsyncGenerator, Callable, List

# Ensure UTF-8 encoding for the process
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('LC_ALL', 'en_US.UTF-8')

# Ensure UTF-8 encoding for stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import PrivateAttr, Field, BaseModel, SecretStr
from timbal.state import RunContext
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, VoiceResponse
from timbal.types.message import Message
from timbal.errors import APIKeyNotFoundError, EarlyExit

from .base import BaseAdapter
import structlog
from uuid_extensions import uuid7

logger = structlog.get_logger("timbal.adapters.twilio_adapter")
from dotenv import load_dotenv
load_dotenv()

# Real-time processing configuration
MIN_TRANSCRIPTION_LENGTH = 2


# --- Config Models ---

class ElevenLabsConfig(BaseModel):
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("ELEVENLABS_API_KEY")))
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    model_id: str = "eleven_multilingual_v2"
    optimize_streaming_latency: int = 2
    speed: float = 1.1
    stability: float = 0.5
    use_speaker_boost: bool = True
    chunk_length_schedule: list[int] = Field(default_factory=lambda: [50, 80, 120, 200])

class OpenAIConfig(BaseModel):
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY")))
    model: str = "gpt-4o-realtime-preview-2025-06-03"
    transcription_model: str = "gpt-4o-mini-transcribe"
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
    def __init__(self, call_sid: str, stream_sid: str, event_queue: asyncio.Queue, adapter_config: TwilioAdapterConfig):
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.adapter_config = adapter_config
        self.conversation_count = 0
        self.start_time = datetime.now()
        self.openai_websocket = None
        self.openai_session_ready = False
        self.openai_task = None
        self.last_context_id: Optional[str] = None  # Track last context ID for parent_id chain
        
    async def initialize_openai_connection(self):
        try:
            logger.info(f"🔗 Initializing OpenAI Realtime API connection for call {self.call_sid}")
            headers = {
                "Authorization": f"Bearer {self.adapter_config.openai.api_key.get_secret_value()}",
                "OpenAI-Beta": "realtime=v1"
            }
            ws_url = f"wss://api.openai.com/v1/realtime?model={self.adapter_config.openai.model}"
            self.openai_websocket = await websockets.connect(ws_url, additional_headers=headers, ping_interval=20, ping_timeout=10)
            logger.info(f"transcription_language: {self.adapter_config.openai.transcription_language}")
            session_config = {
                "type": "session.update",
                "session": {
                    "input_audio_format": "g711_ulaw",
                    "input_audio_transcription": {
                        "model": self.adapter_config.openai.transcription_model,
                        "language": self.adapter_config.openai.transcription_language
                    },
                    "turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500},
                    "input_audio_noise_reduction": {"type": "near_field"},
                },
            }
            await self.openai_websocket.send(json.dumps(session_config))
            self.openai_task = asyncio.create_task(self.handle_openai_events())
            return True
        except Exception as e:
            logger.error(f"💥 Error initializing OpenAI connection: {e}")
            return False

    async def handle_openai_events(self):
        try:
            logger.info("👂 Starting OpenAI event handler")
            while self.openai_websocket:
                try:
                    message = await self.openai_websocket.recv()
                    event = json.loads(message)
                    await self.process_openai_event(event)
                except websockets.exceptions.ConnectionClosed:
                    logger.info("🔌 OpenAI WebSocket connection closed")
                    break
        except Exception as e:
            logger.error(f"💥 Error in OpenAI event handler: {e}")

    async def process_openai_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "session.created":
            self.openai_session_ready = True
            # For the first turn, trigger agent to start conversation
            # logger.info("🎯 OpenAI session ready, triggering agent for initial greeting")
            await self._trigger_agent_with_transcription("")
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            # logger.info(f"✅ Transcription completed: '{transcript}'")
            if self.validate_transcription(transcript):
                # Trigger agent processing with complete transcription
                await self._trigger_agent_with_transcription(transcript)

    def validate_transcription(self, text: str) -> bool:
        if not text or not text.strip():
            # Allow empty transcript for the first turn to make the agent speak first
            return self.conversation_count == 0
        return len(text.strip()) >= MIN_TRANSCRIPTION_LENGTH

    async def _trigger_agent_with_transcription(self, transcript: str):
        """Trigger agent processing with a complete transcription from OpenAI VAD."""
        try:
            # Get the adapter reference (this CallSession needs access to the adapter)
            if not hasattr(self, '_adapter_ref'):
                logger.warning("No adapter reference available to trigger agent")
                return
                
            adapter = self._adapter_ref
            if not hasattr(adapter, '_agent') or not adapter._agent:
                logger.warning("No agent available for processing transcription")
                return
                
            agent = adapter._agent
            
            # Create context with proper parent_id chain for conversation memory
            # Generate new context ID for this turn
            new_context_id = uuid7(as_type="str")
            
            # Create context with parent_id to maintain conversation chain
            # For first turn: parent_id = None (no previous memory)
            # For subsequent turns: parent_id = last_context_id (previous turn's context)
            context = RunContext(
                id=new_context_id,
                parent_id=self.last_context_id  # None for first turn, context_id for subsequent turns
            )
            
            logger.debug(f"🧠 Memory chain: context_id={new_context_id}, parent_id={context.parent_id}, turn={self.conversation_count + 1}")
            
            # Add transcription as prompt for the agent
            if transcript.strip():
                context.data["prompt"] = Message.validate({
                    "role": "user", 
                    "content": transcript
                })
            else:
                # For initial greeting (empty transcript), let agent start
                context.data["prompt"] = Message.validate({
                    "role": "user", 
                    "content": "Inicia la conversación con un saludo."
                })
            
            # Run the agent with the transcription
            response_text = None
            async for event in agent.run(context=context):
                # Extract response text from final agent output event (not tool results)
                if (hasattr(event, 'output') and event.output and 
                    hasattr(event, 'path') and event.path == 'agent'):
                    if hasattr(event.output, 'content') and event.output.content:
                        # Extract text properly from TextContent objects
                        for content_item in event.output.content:
                            if hasattr(content_item, 'text'):
                                response_text = content_item.text
                                break
                    else:
                        response_text = str(event.output)
                # Continue iterating to let the agent complete and save snapshots
                # Don't break early - let the generator finish completely
            
            # Update last context ID for next turn's parent_id and increment conversation count
            self.last_context_id = new_context_id
            self.conversation_count += 1
            logger.debug(f"🔄 Updated last_context_id to {new_context_id}, conversation_count: {self.conversation_count}")
            
            if response_text:
                # Ensure proper UTF-8 encoding
                try:
                    response_text = response_text.encode('utf-8').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    response_text = response_text.encode('utf-8', errors='ignore').decode('utf-8')
                
                logger.info(f"🤖 Agent response: {response_text}")
                
                # Stream TTS response in real-time
                # logger.info(f"🔊 ELEVENLABS: Starting real-time TTS streaming for text: '{response_text[:50]}...'")
                audio_chunks_streamed = 0
                
                async for audio_response in adapter._generate_tts_response(response_text, self):
                    if audio_response:
                        audio_chunks_streamed += 1
                        # logger.info(f"🔊 ELEVENLABS: Streaming audio chunk #{audio_chunks_streamed} to WebSocket")
                        # Send each chunk immediately to WebSocket for real-time playback
                        await adapter._send_audio_to_websocket(audio_response)
                    else:
                        logger.error(f"🔊 ELEVENLABS: Received empty audio response chunk")
                
                if audio_chunks_streamed > 0:
                    logger.info(f"🔊 ELEVENLABS: ✅ Successfully streamed {audio_chunks_streamed} audio chunks in real-time")
                else:
                    logger.error(f"🔊 ELEVENLABS: ❌ No audio chunks were streamed")
                
        except Exception as e:
            logger.error(f"❌ Error triggering agent: {e}", exc_info=True)

    async def send_audio_chunk_to_openai(self, audio_bytes: bytes):
        if self.openai_websocket and self.openai_session_ready:
            try:
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                await self.openai_websocket.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
            except Exception as e:
                logger.error(f"💥 Error sending audio to OpenAI: {e}")

    def add_audio_chunk(self, audio_bytes: bytes):
        """Add audio chunk and send to OpenAI for real-time transcription."""
        # Send to OpenAI if connection is ready
        if self.openai_websocket and self.openai_session_ready:
            asyncio.create_task(self.send_audio_chunk_to_openai(audio_bytes))

    async def cleanup(self):
        logger.info(f"🧹 Cleaning up call session {self.call_sid}")
        if self.openai_task: 
            self.openai_task.cancel()
            try:
                await self.openai_task
            except asyncio.CancelledError:
                pass
        if self.openai_websocket: 
            await self.openai_websocket.close()


class TwilioCallAdapter(BaseAdapter):
    type: str = "twilio_adapter"
    config: TwilioAdapterConfig

    _active_calls: Dict[str, "CallSession"] = PrivateAttr(default_factory=dict)
    _twilio_client: Client = PrivateAttr()
    _call_initiated: bool = PrivateAttr(default=False)
    _websocket_handler_task: Optional[asyncio.Task] = PrivateAttr(default=None)
    _agent: Optional[Any] = PrivateAttr(default=None)
    _pending_audio_queue: asyncio.Queue = PrivateAttr(default_factory=asyncio.Queue)

    def model_post_init(self, __context: Any) -> None:
        """Initialize adapter-specific resources after Pydantic validation."""
        self._twilio_client = Client(
            self.config.twilio.account_sid.get_secret_value(), 
            self.config.twilio.auth_token.get_secret_value()
        )

    def set_agent(self, agent):
        """Set the agent instance for this adapter."""
        self._agent = agent

    async def stop(self):
        """Stop the adapter and clean up resources."""
        # Hang up all active calls
        for call_sid in list(self._active_calls.keys()):
            await self.hang_up_call(call_sid)
        
        # Cancel websocket handler if running
        if self._websocket_handler_task and not self._websocket_handler_task.done():
            self._websocket_handler_task.cancel()
            try:
                await self._websocket_handler_task
            except asyncio.CancelledError:
                pass

    async def before_agent(self, context: RunContext) -> None:
        """Called before the agent processes each turn. Handles call initiation."""     # Check if we need to initiate a call first
        if not self._call_initiated:
            logger.info("📞 Initiating outbound call...")
            call_sid = await self._initiate_call()
            if call_sid:
                self._call_initiated = True
                logger.info(f"Call initiated successfully. Call SID: {call_sid}")
            else:
                logger.error("Failed to initiate call")
            
            # Exit early after call initiation - wait for OpenAI VAD to trigger agent
            raise EarlyExit("Call initiated, waiting for OpenAI VAD to trigger agent")
        
        # If we reach here, the agent was triggered by _trigger_agent_with_transcription
        # The prompt should already be set in context.data["prompt"]
        if "prompt" not in context.data:
            logger.warning("No prompt found in context - this shouldn't happen")
            raise EarlyExit("No prompt to process")

    async def after_agent(self, context: RunContext) -> None:
        """Called after the agent completes each turn. Response handling is done in _trigger_agent_with_transcription."""
        # Response handling is already done in _trigger_agent_with_transcription
        # This method is kept for compatibility but doesn't need to do anything
        pass

    async def on_error(self, context: RunContext, error: dict[str, Any]) -> None:
        """Called when an error occurs during agent execution."""
        logger.error("Agent error occurred", error=error)
        # Could implement error-specific handling here, like sending an error message to the call

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
            logger.error("❌ Invalid URL. Must start with 'https://' or 'wss://'")
            return None

        response = VoiceResponse()
        connect = Connect()
        stream = connect.stream(url=wss_url)
        stream.parameter(name="track", value="both_tracks")
        response.append(connect)
        response.pause(length=100)
        
        twiml_str = str(response)
        logger.info(f"🔗 WebSocket URL: {wss_url}")
        logger.info(f"📋 TwiML being sent: {twiml_str}")
        
        try:
            logger.info(f"📞 Initiating call to {self.config.to_phone_number}")
            logger.info(f"📞 TwiML being sent: {twiml_str}")
            logger.info(f"📞 From phone number: {self.config.twilio.from_phone_number}")
            call = self._twilio_client.calls.create(
                to=self.config.to_phone_number, 
                from_=self.config.twilio.from_phone_number, 
                twiml=twiml_str
            )
            logger.info(f"📞 Call SID: {call.sid}")
            logger.info(f"📞 Call initiated to {self.config.to_phone_number}. Call SID: {call.sid}")
            return call.sid
        except Exception as e:
            logger.error(f"💥 Error initiating call: {e}")
            return None

    async def _send_tts_response(self, text: str, call_session: "CallSession"):
        """Synthesizes and sends the agent's response back to the call."""
        elevenlabs_ws = await self._create_elevenlabs_websocket()
        if not elevenlabs_ws:
            return

        # Start audio streaming task
        audio_stream_task = asyncio.create_task(
            self._stream_audio_from_elevenlabs(elevenlabs_ws, call_session)
        )

        # Send text to ElevenLabs for TTS
        await self._send_text_chunk_to_elevenlabs(elevenlabs_ws, text)
        await self._close_elevenlabs_stream(elevenlabs_ws)
        
        # Wait for audio streaming to complete
        await audio_stream_task

    async def handle_message(self, message_json: str):
        """Routes incoming messages to the appropriate handler."""
        try:
            message = json.loads(message_json)
            event_type = message.get("event")

            if event_type == "start":
                await self._handle_call_start(message)
                return None
            elif event_type == "media":
                return await self._handle_media(message)
            elif event_type == "stop":
                await self._handle_call_stop(message)
                return {"event": "close"}
            
            return None
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message_json}")
            return None
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
                call_session.add_audio_chunk(audio_bytes)
                     
        # Check if we have pending audio chunks to send
        try:
            audio_response = self._pending_audio_queue.get_nowait()
            return audio_response
        except asyncio.QueueEmpty:
            # No audio chunks available
            pass
                    
        return None

    async def _handle_call_stop(self, message: dict):
        """Handles the end of a call."""
        stop_data = message.get("stop", {})
        call_sid = stop_data.get("callSid")
        logger.info(f"📞 CALL ENDED - CallSid: {call_sid}")

        call_session = self.get_call_session(call_sid)
        if call_session:
            await call_session.cleanup()
            self._remove_call_session(call_sid)


    async def hang_up_call(self, call_sid: str):
        """Initiates the process of hanging up a specific call."""
        logger.info(f"--- HANG UP PROCESS STARTED for {call_sid} ---")
        try:
            self._twilio_client.calls(call_sid).update(status="completed")
        except Exception as e:
            if "Call is not in-progress" in str(e):
                logger.warning(f"Call {call_sid} already completed or hung up by other party.")
            else:
                logger.error(f"💥 Twilio hang up error for {call_sid}: {e}")

    def _create_call_session(self, call_sid: str, stream_sid: str) -> "CallSession":
        # Create a dummy event queue since we're not using it in the new approach
        event_queue = asyncio.Queue()
        call_session = CallSession(call_sid, stream_sid, event_queue, self.config)
        # Set adapter reference so CallSession can trigger agent
        call_session._adapter_ref = self
        self._active_calls[call_sid] = call_session
        return call_session

    def get_call_session(self, call_sid: str) -> Optional["CallSession"]:
        return self._active_calls.get(call_sid)

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

    async def _generate_tts_response(self, text: str, call_session: "CallSession") -> AsyncGenerator[dict, None]:
        """
        Generate TTS response and yield each audio chunk as it arrives for real-time streaming.
        This replaces the old _send_tts_response method.
        """
        try:
            # Create ElevenLabs WebSocket connection
            elevenlabs_ws = await self._create_elevenlabs_websocket()
            if not elevenlabs_ws:
                return

            # Send text to ElevenLabs
            await self._send_text_chunk_to_elevenlabs(elevenlabs_ws, text)
            await self._close_elevenlabs_stream(elevenlabs_ws)

            # Stream audio data as it arrives
            chunk_count = 0
            try:
                while True:
                    message = await elevenlabs_ws.recv()
                    event = json.loads(message)
                    
                    if event.get("audio"):
                        audio_chunk = base64.b64decode(event["audio"])
                        chunk_count += 1
 
                        # Convert to base64 and yield immediately
                        payload = base64.b64encode(audio_chunk).decode("utf-8")
                        
                        # Yield in Twilio WebSocket format
                        response = {
                            "event": "media",
                            "streamSid": call_session.stream_sid,
                            "media": {
                                "payload": payload
                            }
                        }
                        
                        yield response
                        
                    # Check both camelCase and snake_case for final marker
                    if event.get("is_final") or event.get("isFinal"):
                        break
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                pass
            finally:
                await elevenlabs_ws.close()

            if chunk_count == 0:
                logger.error(f"🔊 ELEVENLABS: ❌ No audio chunks received")
            
        except Exception as e:
            logger.error(f"🔊 ELEVENLABS: ❌ Error generating TTS response: {e}", exc_info=True)

    # --- ElevenLabs Helper Methods ---
    async def _create_elevenlabs_websocket(self):
        try:
            config = self.config.elevenlabs
            ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{config.voice_id}/stream-input?model_id={config.model_id}&output_format=ulaw_8000&optimize_streaming_latency={config.optimize_streaming_latency}"
            headers = {"xi-api-key": config.api_key.get_secret_value()}
            elevenlabs_ws = await websockets.connect(ws_url, additional_headers=headers)
            bos_message = {
                "text": " ",  # documentation says this is a small space
                "voice_settings": {
                    "speed": config.speed,
                    "stability": config.stability,
                    "use_speaker_boost": config.use_speaker_boost
                },
                "generation_config": {
                    "chunk_length_schedule": config.chunk_length_schedule
                },
            }
            await elevenlabs_ws.send(json.dumps(bos_message, ensure_ascii=False))
            return elevenlabs_ws
        except Exception as e:
            logger.error(f"❌ Error creating ElevenLabs WebSocket: {e}")
            return None

    async def _send_text_chunk_to_elevenlabs(self, elevenlabs_ws, text_chunk: str):
        try:
            if text_chunk.strip():
                # Ensure proper UTF-8 encoding for ElevenLabs
                try:
                    text_chunk = text_chunk.encode('utf-8').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    text_chunk = text_chunk.encode('utf-8', errors='ignore').decode('utf-8')
                
                message = {"text": text_chunk}
                # Ensure JSON is properly encoded
                json_message = json.dumps(message, ensure_ascii=False)
                await elevenlabs_ws.send(json_message)
        except Exception as e:
            logger.error(f"Error sending text to ElevenLabs: {e}")
            pass

    async def _close_elevenlabs_stream(self, elevenlabs_ws):
        try:
            # Send empty text to signal end of stream
            await elevenlabs_ws.send(json.dumps({"text": ""}, ensure_ascii=False))
        except Exception:
            pass

    async def _send_audio_to_websocket(self, audio_response: dict):
        """Send audio response back to Twilio via WebSocket queue."""
        try:
            # The audio_response should be in Twilio WebSocket format
            if not audio_response:
                return
                  
            # Add the response to the queue for streaming
            await self._pending_audio_queue.put(audio_response)

        except Exception as e:
            logger.error(f"🔊 ELEVENLABS: Error queuing audio to WebSocket: {e}", exc_info=True)

    async def _stream_audio_from_elevenlabs(self, elevenlabs_ws, session: "CallSession"):
        """This method is no longer used since we return audio data instead of streaming directly."""
        pass

 