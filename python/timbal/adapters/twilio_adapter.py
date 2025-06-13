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
        self.event_queue = event_queue
        self.adapter_config = adapter_config
        self.context = RunContext()
        self.processing = False
        self.conversation_count = 0
        self.conversation_history: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self.openai_websocket = None
        self.openai_session_ready = False
        self.openai_task = None
        self.audio_chunks: List[bytes] = []  # Store audio chunks for transcription
        
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
                self.conversation_history.append({
                    "speaker": "user",
                    "text": transcript,
                    "timestamp": datetime.now().isoformat()
                })
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
            
            # Create context with transcription
            from timbal.state.context import RunContext
            context = RunContext()
            context.data["transcription"] = transcript
            context.data["call_session"] = self
            
            # Run the agent with the transcription
            response_text = None
            async for event in agent.run(context=context):
                if hasattr(event, 'output') and event.output:
                    if hasattr(event.output, 'content') and event.output.content:
                        # Extract text properly from TextContent objects
                        for content_item in event.output.content:
                            if hasattr(content_item, 'text'):
                                response_text = content_item.text
                                break
                    else:
                        response_text = str(event.output)
                    break
            
            if response_text:
                # Ensure proper UTF-8 encoding
                try:
                    response_text = response_text.encode('utf-8').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    response_text = response_text.encode('utf-8', errors='ignore').decode('utf-8')
                
                logger.info(f"🤖 Agent response: {response_text}")
                
                # Add agent response to conversation history
                self.conversation_history.append({
                    "speaker": "agent",
                    "text": response_text,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Stream TTS response in real-time
                logger.info(f"🔊 ELEVENLABS: Starting real-time TTS streaming for text: '{response_text[:50]}...'")
                audio_chunks_streamed = 0
                
                async for audio_response in adapter._generate_tts_response(response_text, self):
                    if audio_response:
                        audio_chunks_streamed += 1
                        logger.info(f"🔊 ELEVENLABS: Streaming audio chunk #{audio_chunks_streamed} to WebSocket")
                        # Send each chunk immediately to WebSocket for real-time playback
                        await adapter._send_audio_to_websocket(audio_response)
                    else:
                        logger.error(f"🔊 ELEVENLABS: Received empty audio response chunk")
                
                if audio_chunks_streamed > 0:
                    logger.info(f"🔊 ELEVENLABS: ✅ Successfully streamed {audio_chunks_streamed} audio chunks in real-time")
                else:
                    logger.error(f"🔊 ELEVENLABS: ❌ No audio chunks were streamed")
                
                # Clear audio chunks after successful response
                self.audio_chunks.clear()
                
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
        # Store audio chunk for transcription
        self.audio_chunks.append(audio_bytes)
        
        # Also send to OpenAI if connection is ready (for compatibility)
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

    async def start(self):
        """Initialize the adapter and start the call."""
        if not self._call_initiated:
            await self._initiate_call()
            self._call_initiated = True

    async def listen(self):
        """Listen for user input (transcriptions) from active calls."""
        while True:
            # Wait for a call to be established
            while not self._active_calls:
                await asyncio.sleep(0.1)
            
            # Get the first active call session
            call_session = list(self._active_calls.values())[0]
            
            # Wait for transcription from the user
            transcription = await call_session.wait_for_transcription()
            
            # Yield the transcription to the agent
            if transcription.strip():
                yield transcription
            else:
                # For the first turn with empty transcription, let the agent start
                yield "Iniciar conversación"

    async def reply(self, response_generator):
        """Send agent responses back to the call via TTS."""
        async for event in response_generator:
            # Handle different types of events from the agent
            if hasattr(event, 'output') and hasattr(event.output, 'content'):
                # This is the final agent response
                response_text = ""
                for content_item in event.output.content:
                    if hasattr(content_item, 'text'):
                        response_text += content_item.text
                
                if response_text.strip() and self._active_calls:
                    call_session = list(self._active_calls.values())[0]
                    call_session.conversation_count += 1
                    
                    # Add agent response to conversation history
                    call_session.conversation_history.append({
                        "speaker": "agent",
                        "text": response_text,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Send the response via TTS
                    await self._send_tts_response(response_text, call_session)

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
        """Called before the agent processes each turn. Handles call initiation and transcription processing."""
        # Store adapter reference in context for tools to access
        context.data['adapter'] = self
        
        # Check if we need to initiate a call first
        if not self._call_initiated:
            logger.info("📞 Initiating outbound call...")
            call_sid = await self._initiate_call()
            if call_sid:
                self._call_initiated = True
                logger.info(f"✅ Call initiated successfully. Call SID: {call_sid}")
            else:
                logger.error("❌ Failed to initiate call")
            
            # Exit early after call initiation - wait for OpenAI VAD to trigger agent
            raise EarlyExit("Call initiated, waiting for OpenAI VAD to trigger agent")
        
        # Check if we have a transcription ready to process (triggered by OpenAI VAD)
        if "transcription" not in context.data:
            logger.debug("No transcription available, agent will exit early")
            raise EarlyExit("No transcription to process")
        
        transcription = context.data["transcription"]
        
        # For empty transcription (first turn), let agent start conversation
        if not transcription.strip():
            logger.info("🎯 First turn - agent will start conversation")
            context.data["prompt"] = Message.validate({
                "role": "user", 
                "content": "Iniciar conversación"
            })
        else:
            logger.info(f"🎤 Processing transcription: {transcription}")
            # Set the transcription as the prompt for the agent
            context.data["prompt"] = Message.validate({
                "role": "user", 
                "content": transcription
            })

    async def after_agent(self, context: RunContext) -> None:
        """Called after the agent completes each turn. Handles output (TTS and audio streaming)."""
        # Get the agent's response from the context
        if 'output' not in context.data:
            logger.warning("No agent output found in context")
            return
        
        agent_output = context.data['output']
        
        # Extract text from the agent's response
        response_text = ""
        if hasattr(agent_output, 'content') and isinstance(agent_output.content, list):
            for content_item in agent_output.content:
                if hasattr(content_item, 'text'):
                    response_text += content_item.text
        elif isinstance(agent_output, str):
            response_text = agent_output
        
        # Ensure proper UTF-8 encoding
        if response_text:
            try:
                # Normalize the text to ensure proper UTF-8 encoding
                response_text = response_text.encode('utf-8').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                logger.warning(f"Text encoding issue: {e}")
                # Fallback: remove problematic characters
                response_text = response_text.encode('utf-8', errors='ignore').decode('utf-8')
        
        if not response_text.strip():
            logger.warning("No text content found in agent response")
            return
        
        # Get the current call session
        if not self._active_calls:
            logger.warning("No active call session for sending response")
            return
        
        call_session = list(self._active_calls.values())[0]
        call_session.conversation_count += 1
        
        # Add agent response to conversation history
        call_session.conversation_history.append({
            "speaker": "agent",
            "text": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Log the response with proper encoding
        logger.info(f"🤖 Agent response: {response_text}")
        
        # Send the response via TTS
        await self._send_tts_response(response_text, call_session)

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
            call = self._twilio_client.calls.create(
                to=self.config.to_phone_number, 
                from_=self.config.twilio.from_phone_number, 
                twiml=twiml_str
            )
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
            logger.error(f"💥 Invalid JSON received: {message_json}")
            return None
        except Exception as e:
            logger.error(f"💥 Error handling message: {e}")
            return None

    async def _handle_call_start(self, message: dict):
        """Handles the start of a call."""
        start_data = message.get("start", {})
        call_sid = start_data.get("callSid")
        stream_sid = message.get("streamSid")
        # logger.info(f"📞 CALL STARTED - CallSid: {call_sid}")

        call_session = self._create_call_session(call_sid, stream_sid)
        if not call_session:
            logger.error(f"Failed to create or retrieve call session for {call_sid}")
            return

        await call_session.initialize_openai_connection()
        
        # Call session is now ready for agent interaction
        # logger.info(f"📞 Call session {call_sid} ready for agent interaction")

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
                
                # Just accumulate audio - OpenAI VAD will handle transcription and agent triggering
                # logger.debug(f"📡 Audio chunk received, total chunks: {len(call_session.audio_chunks)}")
        
        # Check if we have pending audio chunks to send
        try:
            audio_response = self._pending_audio_queue.get_nowait()
            logger.info(f"🔊 ELEVENLABS: Returning queued audio chunk")
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
        # logger.info(f"Creating new call session for CallSid: {call_sid}")
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

    def _get_call_session_by_websocket(self, websocket: WebSocket) -> Optional["CallSession"]:
        call_sid = None
        for sid, ws in self._active_websockets.items():
            if ws == websocket:
                call_sid = sid
                break
        
        if call_sid:
            return self.get_call_session(call_sid)
        
        return None

    def _remove_call_session(self, call_sid: str):
        if call_sid in self._active_calls:
            logger.info(f"Removing call session for CallSid: {call_sid}")
            del self._active_calls[call_sid]

    async def _try_get_transcription(self, call_session: "CallSession") -> Optional[str]:
        """
        Try to get transcription from accumulated audio chunks.
        Returns transcription if available, None otherwise.
        """
        try:
            # Check if we have enough audio data (simple heuristic)
            if len(call_session.audio_chunks) < 10:  # Need at least some chunks
                return None
                
            # Combine all audio chunks
            combined_audio = b''.join(call_session.audio_chunks)
            
            # Use OpenAI Whisper for transcription
            import tempfile
            import wave
            
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert μ-law to PCM and save as WAV
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(8000)  # 8kHz
                    
                    # Convert μ-law to linear PCM
                    import audioop
                    linear_audio = audioop.ulaw2lin(combined_audio, 2)
                    wav_file.writeframes(linear_audio)
                
                # Transcribe using OpenAI
                import openai
                client = openai.AsyncOpenAI(api_key=self.config.openai.api_key.get_secret_value())
                
                with open(temp_file.name, 'rb') as audio_file:
                    transcript = await client.audio.transcriptions.create(
                        model=self.config.openai.transcription_model,
                        file=audio_file,
                        language=self.config.openai.transcription_language
                    )
                
                # Clean up temp file
                import os
                os.unlink(temp_file.name)
                
                transcription_text = transcript.text.strip()
                if transcription_text and len(transcription_text) > 3:  # Minimum meaningful length
                    return transcription_text
                    
        except Exception as e:
            logger.error(f"❌ Error getting transcription: {e}", exc_info=True)
            
        return None

    async def _generate_tts_response(self, text: str, call_session: "CallSession") -> AsyncGenerator[dict, None]:
        """
        Generate TTS response and yield each audio chunk as it arrives for real-time streaming.
        This replaces the old _send_tts_response method.
        """
        try:
            logger.info(f"🔊 ELEVENLABS: Starting TTS generation for: '{text[:50]}...'")
            
            # Create ElevenLabs WebSocket connection
            elevenlabs_ws = await self._create_elevenlabs_websocket()
            if not elevenlabs_ws:
                logger.error("🔊 ELEVENLABS: ❌ Failed to create ElevenLabs WebSocket")
                return

            logger.info(f"🔊 ELEVENLABS: ✅ WebSocket connection established")

            # Send text to ElevenLabs
            await self._send_text_chunk_to_elevenlabs(elevenlabs_ws, text)
            await self._close_elevenlabs_stream(elevenlabs_ws)

            logger.info(f"🔊 ELEVENLABS: Text sent, streaming audio chunks in real-time...")

            # Stream audio data as it arrives
            chunk_count = 0
            try:
                while True:
                    message = await elevenlabs_ws.recv()
                    event = json.loads(message)
                    
                    if event.get("audio"):
                        audio_chunk = base64.b64decode(event["audio"])
                        chunk_count += 1
                        logger.info(f"🔊 ELEVENLABS: Streaming audio chunk #{chunk_count}, size: {len(audio_chunk)} bytes")
                        
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
                        
                        logger.info(f"🔊 ELEVENLABS: ✅ Yielding audio chunk #{chunk_count} for immediate streaming")
                        yield response
                        
                    # Check both camelCase and snake_case for final marker
                    if event.get("is_final") or event.get("isFinal"):
                        logger.info(f"🔊 ELEVENLABS: Received final marker, total chunks streamed: {chunk_count}")
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
                "text": " ",  # Small space as per documentation
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
            logger.info(f"🔊 ELEVENLABS: Queuing audio chunk for WebSocket")
            
            # The audio_response should be in Twilio WebSocket format
            if not audio_response:
                logger.error(f"🔊 ELEVENLABS: No audio response to send")
                return
                
            logger.info(f"🔊 ELEVENLABS: Audio response ready: {type(audio_response)}")
            logger.info(f"🔊 ELEVENLABS: Audio response keys: {audio_response.keys() if isinstance(audio_response, dict) else 'Not a dict'}")
            
            # Add the response to the queue for streaming
            await self._pending_audio_queue.put(audio_response)
            logger.info(f"🔊 ELEVENLABS: ✅ Audio chunk queued successfully")
            
        except Exception as e:
            logger.error(f"🔊 ELEVENLABS: Error queuing audio to WebSocket: {e}", exc_info=True)

    async def _stream_audio_from_elevenlabs(self, elevenlabs_ws, session: "CallSession"):
        """This method is no longer used since we return audio data instead of streaming directly."""
        pass

 