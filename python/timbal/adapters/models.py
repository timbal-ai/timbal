import asyncio
import base64
import json
import time
import websockets
from datetime import datetime
import audioop

from fastapi import WebSocket
from timbal.state import RunContext

from config import (
    logger, OPENAI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    MIN_TRANSCRIPTION_LENGTH
)


class CallSession:
    """Manages the state of an individual call with OpenAI real-time transcription."""
    
    def __init__(self, call_sid: str, stream_sid: str, websocket: WebSocket, agent):
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.websocket = websocket
        self.agent = agent
        self.context = RunContext()
        self.processing = False
        self.conversation_count = 0
        self.conversation_history = []
        self.start_time = datetime.now()
        self.total_audio_chunks_received = 0
        self.total_audio_bytes_received = 0
        self.audio_responses_sent = 0
        
        # OpenAI Real-time API components
        self.openai_websocket = None
        self.openai_session_ready = False
        self.current_transcription_buffer = ""
        self.transcription_chunks = []
        self.openai_task = None
        
        # ElevenLabs streaming components
        self.audio_task = None
        self.current_tts_task = None

    async def initialize_openai_connection(self):
        """Initialize connection to OpenAI Realtime API"""
        try:
            logger.info(f"🔗 Initializing OpenAI Realtime API connection for call {self.call_sid}")
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            ws_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2025-06-03"
            
            self.openai_websocket = await websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Send session configuration for transcription
            session_config = {
                "type": "session.update",
                "session": {
                    "input_audio_format": "g711_ulaw",
                    "input_audio_transcription": {
                        "model": "gpt-4o-mini-transcribe",
                        "language": "es"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    },
                    "input_audio_noise_reduction": {
                        "type": "near_field"
                    }
                }
            }
            
            logger.info(f"📤 Sending OpenAI session config: {json.dumps(session_config, indent=2)}")
            await self.openai_websocket.send(json.dumps(session_config))
            logger.info("✅ OpenAI session configuration sent")
            
            # Start event handler
            self.openai_task = asyncio.create_task(self.handle_openai_events())
            
            return True
            
        except Exception as e:
            logger.error(f"💥 Error initializing OpenAI connection: {e}")
            return False

    async def handle_openai_events(self):
        """Handle events from OpenAI Realtime API"""
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
                except json.JSONDecodeError as e:
                    logger.error(f"💥 Invalid JSON from OpenAI: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"💥 Error in OpenAI event handler: {e}")

    async def process_openai_event(self, event: dict):
        """Process individual OpenAI events with low latency"""
        event_type = event.get("type")
        
        # LOG ALL EVENTS to debug transcription issues
        # logger.info(f"🔍 OpenAI Event Received: {event_type}")
        # logger.debug(f"🔍 Full event data: {event}")
        
        if event_type == "session.created":
            logger.info("✅ OpenAI session created")
            self.openai_session_ready = True
            
        elif event_type == "input_audio_buffer.speech_started":
            logger.info("🗣️ OpenAI VAD: Speech detected - waiting for transcription...")
            self.current_transcription_buffer = ""
            # Cancel any ongoing processing to prioritize new speech
            if self.processing:
                logger.info("🔄 New speech detected - interrupting current processing")
            
        elif event_type == "input_audio_buffer.speech_stopped":
            logger.info("🤐 OpenAI VAD: Speech stopped - transcription should start soon...")
            
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            logger.info(f"✅ Transcription completed: '{transcript}'")
            
            if self.validate_transcription(transcript):
                logger.info(f"✅ Transcription validation PASSED: '{transcript}'")
                # Process immediately without waiting
                asyncio.create_task(self.process_complete_transcription(transcript))
            else:
                logger.info(f"🚫 Transcription validation FAILED: '{transcript}' (length: {len(transcript.strip())})")
                
        elif event_type == "conversation.item.input_audio_transcription.failed":
            error = event.get("error", {})
            logger.error(f"❌ OpenAI Transcription Failed: {error}")
            
        elif event_type == "conversation.item.created":
            logger.info(f"📝 OpenAI Conversation item created: {event.get('item', {}).get('type', 'unknown')}")
            
        elif event_type == "input_audio_buffer.committed":
            logger.info("💾 OpenAI Audio buffer committed for processing")
            
        elif event_type == "rate_limits.updated":
            logger.debug(f"📊 OpenAI Rate limits updated: {event.get('rate_limits', {})}")
            
        elif event_type == "error":
            error = event.get("error", {})
            logger.error(f"❌ OpenAI API Error: {error}")
            
        else:
            logger.debug(f"❓ Unhandled OpenAI event type: {event_type}")

    def validate_transcription(self, text: str) -> bool:
        """Validates if a transcription appears to be real speech"""
        logger.info(f"🔍 Validating transcription: '{text}' (raw length: {len(text) if text else 0})")
        
        if not text or not text.strip():
            logger.info(f"❌ Validation failed: Empty text or only whitespace")
            return False
        
        stripped_length = len(text.strip())
        logger.info(f"🔍 Stripped length: {stripped_length}, MIN_TRANSCRIPTION_LENGTH: {MIN_TRANSCRIPTION_LENGTH}")
        
        is_valid = stripped_length >= MIN_TRANSCRIPTION_LENGTH
        logger.info(f"{'✅' if is_valid else '❌'} Validation result: {is_valid}")
        
        return is_valid

    async def process_complete_transcription(self, transcript: str):
        """Process a complete transcription from OpenAI VAD with streaming response"""
        logger.info(f"🎯 PROCESS_COMPLETE_TRANSCRIPTION CALLED with: '{transcript}'")
        
        if self.processing:
            logger.info("🔇 Skipping transcription processing (agent busy)")
            return
        
        self.processing = True
        logger.info(f"🔒 Processing flag set to True for transcript: '{transcript}'")
        
        try:
            self.conversation_count += 1
            logger.info(f"📝 User #{self.conversation_count}: '{transcript}'")
            
            self.add_conversation_entry("user", transcript)
            
            # Process with agent and stream response in parallel
            logger.info(f"🚀 Starting agent processing task for: '{transcript}'")
            asyncio.create_task(self._process_and_respond(transcript))
            
        except Exception as e:
            logger.error(f"💥 Error processing transcription: {e}")
            asyncio.create_task(self._send_error_response())
        finally:
            # Don't block here - let processing continue in background
            logger.info(f"✅ Process_complete_transcription finished setup for: '{transcript}'")
            pass

    async def _process_and_respond(self, transcript: str):
        """Process transcript with agent and send streaming response using ElevenLabs streaming"""
        try:
            logger.info("🤖 Starting Agent streaming response...")
            
            # Create ElevenLabs WebSocket connection for streaming
            elevenlabs_ws = await self.create_elevenlabs_websocket()
            
            if not elevenlabs_ws:
                logger.error("❌ Failed to create ElevenLabs WebSocket")
                await self._send_error_response()
                return

            # Process agent response with streaming
            full_response = await self.process_agent_response_streaming(elevenlabs_ws, transcript)
            
            logger.info(f"✅ Agent streaming completed")
            logger.info(f"📝 Full response ({len(full_response)} chars): '{full_response}'")
            
            # Add to conversation history
            if full_response:
                self.add_conversation_entry("agent", full_response)
            
            # Ensure WebSocket is closed
            if elevenlabs_ws:
                await self.close_elevenlabs_stream(elevenlabs_ws)
                await elevenlabs_ws.close()
                logger.info("🔌 ElevenLabs WebSocket closed")
            
        except Exception as e:
            logger.error(f"💥 Error in streaming agent processing: {e}")
            await self._send_error_response()
        finally:
            self.processing = False

    async def process_agent_response_streaming(self, elevenlabs_ws, transcript: str) -> str:
        """Process agent response with streaming and send audio to ElevenLabs"""
        logger.info(f"⚡️ Processing agent response for: '{transcript}'")
        full_response = ""
        flow_output_event = None
        hang_up_triggered = False

        async def agent_runner():
            nonlocal full_response, flow_output_event, hang_up_triggered
            try:
                logger.info("🏃 Agent runner started")
                # Stream agent response, adapting to handle various event types from Timbal
                async for event in self.agent.run(prompt=transcript, context=self.context):
                    
                    # Capture the final output event to get the run_id for the next turn
                    if hasattr(event, 'type') and event.type == 'OUTPUT':
                        flow_output_event = event
                        continue # This is a final event, not a chunk for TTS

                    text_chunk = None
                    
                    # Handle events which are strings (likely direct text chunks)
                    if isinstance(event, str):
                        text_chunk = event
                    # Handle Timbal's structured chunk events
                    elif hasattr(event, 'chunk') and event.chunk:
                        chunk_content = event.chunk
                        if isinstance(chunk_content, str):
                            text_chunk = chunk_content
                        elif isinstance(chunk_content, dict):
                            if chunk_content.get("type") == "text":
                                text_chunk = chunk_content.get("text")
                            elif chunk_content.get("type") == "tool_use":
                                logger.info(f"🛠️ Agent using tool: {chunk_content.get('name')}")
                                if chunk_content.get('name') == 'hang_up_call':
                                    hang_up_triggered = True
                                # No text to speak for tool usage events, so we continue
                                continue
                    
                    if text_chunk and text_chunk.strip():
                        logger.info(f"➡️ Sending chunk to ElevenLabs: '{text_chunk}'")
                        await self.send_text_chunk_to_elevenlabs(elevenlabs_ws, text_chunk)
                        full_response += text_chunk

                logger.info("🏁 Agent runner finished")

            except Exception as e:
                logger.error(f"💥 Error in agent runner: {e}")
            finally:
                # Always ensure the EOS message is sent to cleanly end TTS
                if elevenlabs_ws:
                    await self.close_elevenlabs_stream(elevenlabs_ws)

        # Create tasks for concurrent execution
        agent_task = asyncio.create_task(agent_runner())
        audio_task = asyncio.create_task(self.stream_audio_from_elevenlabs(elevenlabs_ws))
        
        self.current_tts_task = audio_task
        
        await asyncio.gather(agent_task, audio_task)
        
        # If the agent signaled to hang up, execute the action now that TTS is complete.
        if hang_up_triggered:
            logger.info("📞 Hang up tool was triggered by agent. Terminating call.")
            # We use create_task to not block the main flow while the hangup (with its delays) happens
            asyncio.create_task(self.agent._hang_up_call())
        
        # Update the context for the next turn to chain the conversation
        if flow_output_event and hasattr(flow_output_event, 'run_id'):
            self.context = RunContext(parent_id=flow_output_event.run_id)
            logger.info(f"🔄 Context updated for next turn with parent_id: {flow_output_event.run_id}")
        else:
            logger.warning("⚠️ Could not find run_id in agent output, context not updated.")
        
        # Wait a moment for the mark message to be processed before the next turn
        await asyncio.sleep(0.1)
        
        return full_response

    async def create_elevenlabs_websocket(self):
        """Create and initialize ElevenLabs WebSocket connection for streaming"""
        try:
            # Use semaphore to control concurrent requests
            await self.agent.tts_semaphore.acquire()
            logger.info(f"🎵 TTS semaphore acquired for streaming")
            
            # Validate required parameters
            if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
                logger.error("⚠️ ElevenLabs credentials not configured")
                self.agent.tts_semaphore.release()
                return None
            
            # Construct WebSocket URL
            ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input?model_id=eleven_multilingual_v2&output_format=ulaw_8000&optimize_streaming_latency=2"
            
            # WebSocket headers
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY
            }
            
            logger.info(f"🔗 Creating ElevenLabs streaming WebSocket...")
            
            # Connect to ElevenLabs WebSocket
            elevenlabs_ws = await websockets.connect(ws_url, additional_headers=headers)
            
            # Send initialization message
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "speed": 1.2,
                    "stability": 0.5,
                    "use_speaker_boost": True
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 80, 120, 200]
                }
            }
            await elevenlabs_ws.send(json.dumps(bos_message))
            logger.info("📤 Sent BOS to ElevenLabs")

            return elevenlabs_ws

        except Exception as e:
            logger.error(f"❌ Error creating ElevenLabs WebSocket: {e}")
            self.agent.tts_semaphore.release()
            return None

    async def send_text_chunk_to_elevenlabs(self, elevenlabs_ws, text_chunk: str):
        """Send a text chunk to ElevenLabs for immediate processing"""
        try:
            if not text_chunk.strip():
                logger.warning(f"   ⚠️ Attempted to send empty chunk: '{text_chunk}'")
                return
                
            message = {
                "text": text_chunk,
                "try_trigger_generation": True  # Force immediate generation
            }
            
            message_json = json.dumps(message)
            logger.info(f"📤 Sending to ElevenLabs WebSocket: {text_chunk}")
            
            await elevenlabs_ws.send(message_json)
            logger.info(f"✅ Successfully sent chunk: '{text_chunk}'")
            
        except Exception as e:
            logger.error(f"❌ Error sending chunk to ElevenLabs: {e}")

    async def close_elevenlabs_stream(self, elevenlabs_ws):
        """Close the ElevenLabs stream"""
        try:
            close_message = {"text": ""}
            await elevenlabs_ws.send(json.dumps(close_message))
            logger.info("📤 Sent close message to ElevenLabs")
            
        except websockets.exceptions.ConnectionClosed:
            logger.info("ℹ️ ElevenLabs stream was already closed when attempting to send EOS.")
        except Exception as e:
            logger.error(f"❌ Error closing ElevenLabs stream: {e}")

    async def stream_audio_from_elevenlabs(self, elevenlabs_ws):
        """Continuously receive and forward audio chunks from ElevenLabs to Twilio"""
        audio_chunk_count = 0
        
        try:
            logger.info("🎧 Starting ElevenLabs audio streaming listener...")
            
            while True:
                try:
                    # Timeout for more responsive audio processing
                    response = await asyncio.wait_for(elevenlabs_ws.recv(), timeout=10.0)
                    
                    if response is None:
                        logger.warning("⚠️ Received None response from ElevenLabs")
                        continue
                        
                    message = json.loads(response)
                    logger.debug(f"📨 Received ElevenLabs message: {list(message.keys())}")
                    
                    # Check for errors
                    if "error" in message:
                        logger.error(f"❌ ElevenLabs Streaming Error: {message}")
                        break
                    
                    if "audio" in message:
                        audio_data = message["audio"]
                        if audio_data is None:
                            logger.debug("⚠️ Received audio message with None data")
                            continue
                            
                        if not isinstance(audio_data, str):
                            logger.warning(f"⚠️ Received audio data is not string: {type(audio_data)}")
                            continue
                        
                        # Decode and send audio immediately
                        try:
                            audio_chunk = base64.b64decode(audio_data)
                            audio_chunk_count += 1
                            
                            logger.info(f"🔊 Audio chunk #{audio_chunk_count}: {len(audio_chunk)} bytes")
                            
                            # Send to Twilio immediately
                            asyncio.create_task(self.send_audio_to_twilio(audio_chunk))
                            
                        except Exception as decode_error:
                            logger.error(f"❌ Error decoding audio chunk: {decode_error}")
                            continue
                    
                    if message.get("is_final"):
                        logger.info("🎤 ElevenLabs stream is final")
                        break
                        
                except asyncio.TimeoutError:
                    logger.info("⏰ Timeout in ElevenLabs audio streaming - stream may be complete")
                    break
                except websockets.exceptions.ConnectionClosed as e:
                    logger.info(f"🔌 ElevenLabs WebSocket connection closed gracefully: {e}")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"💥 Invalid JSON in audio stream: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"💥 Error in ElevenLabs audio streaming: {e}")
        finally:
            # Release the semaphore when streaming is complete
            self.agent.tts_semaphore.release()
            logger.info("🎵 TTS semaphore released after streaming")
            # After the loop, the stream is finished. Send the mark message here.
            await self.send_mark_message_to_twilio()

    async def _send_error_response(self):
        """Send a generic error message if something goes wrong."""
        try:
            error_response = "Disculpa, no pude entender. ¿Puedes repetirlo?"
            await self.send_agent_response(error_response)
        except Exception as e:
            logger.error(f"💥 Error sending error response: {e}")

    async def send_audio_chunk_to_openai(self, audio_bytes: bytes):
        """Send audio chunk to OpenAI"""
        if not self.openai_websocket:
            logger.warning("⚠️ OpenAI websocket not available for audio chunk")
            return
        
        if not self.openai_session_ready:
            logger.warning("⚠️ OpenAI session not ready for audio chunk")
            return
            
        try:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            await self.openai_websocket.send(json.dumps(audio_event))
            logger.debug(f"📤 Sent audio chunk to OpenAI: {len(audio_bytes)} bytes")
        except Exception as e:
            logger.error(f"💥 Error sending audio to OpenAI: {e}")

    def add_audio_chunk(self, audio_bytes: bytes):
        """Add audio chunk and forward to OpenAI with priority"""
        self.total_audio_chunks_received += 1
        self.total_audio_bytes_received += len(audio_bytes)
        
        # Log every 100 chunks to avoid spam but track progress
        if self.total_audio_chunks_received % 100 == 0:
            logger.info(f"🎵 Audio chunks received: {self.total_audio_chunks_received} (total bytes: {self.total_audio_bytes_received})")
        
        # Always forward audio immediately, even if processing
        if self.openai_websocket and self.openai_session_ready:
            # Use high priority task to avoid audio buffering delays
            task = asyncio.create_task(self.send_audio_chunk_to_openai(audio_bytes))
            # Don't await - let it run in parallel to avoid blocking
        else:
            if self.total_audio_chunks_received <= 5:  # Log first few misses
                logger.warning(f"⚠️ Chunk #{self.total_audio_chunks_received} not sent to OpenAI - websocket: {bool(self.openai_websocket)}, session_ready: {self.openai_session_ready}")

    def add_conversation_entry(self, speaker: str, text: str):
        """Add entry to conversation history"""
        self.conversation_history.append({"speaker": speaker, "text": text, "timestamp": datetime.now().isoformat()})

    async def send_agent_response(self, text: str):
        """Generate TTS for a text response and send it to Twilio."""
        try:
            audio_bytes = await self.generate_elevenlabs_tts(text)
            if audio_bytes:
                await self.send_audio_to_twilio(audio_bytes)
                await self.send_mark_message_to_twilio()
        except Exception as e:
            logger.error(f"💥 Error in send_agent_response: {e}")

    async def generate_elevenlabs_tts(self, text: str) -> bytes:
        """Generates audio from text using ElevenLabs REST API (for short responses)."""
        try:
            if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID or not text.strip():
                return self.generate_silence_placeholder()
            
            # Use semaphore to control concurrent requests
            async with self.agent.tts_semaphore:
                logger.info(f"🎵 Starting TTS for: '{text[:50]}...' (semaphore acquired)")
                
                # Use standard ElevenLabs streaming URL
                ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input?model_id=eleven_multilingual_v2&output_format=ulaw_8000"
                headers = {"xi-api-key": ELEVENLABS_API_KEY}
                
                try:
                    async with websockets.connect(ws_url, additional_headers=headers) as elevenlabs_ws:
                        # Initialize with standard settings
                        init_message = {
                            "text": " ",
                            "voice_settings": {
                                "speed": 1.2,
                                "stability": 0.5,
                                "use_speaker_boost": True
                            }
                        }
                        await elevenlabs_ws.send(json.dumps(init_message))
                        
                        # Send text for generation
                        text_message = {
                            "text": text.strip(), 
                            "try_trigger_generation": True
                        }
                        await elevenlabs_ws.send(json.dumps(text_message))
                        
                        # Close stream
                        close_message = {"text": ""}
                        await elevenlabs_ws.send(json.dumps(close_message))
                        
                        # Collect audio with shorter timeout for better responsiveness
                        audio_chunks = []
                        start_time = time.time()
                        while True:
                            try:
                                response = await asyncio.wait_for(elevenlabs_ws.recv(), timeout=8.0)
                                message = json.loads(response)
                                
                                if "error" in message:
                                    logger.error(f"ElevenLabs Error: {message}")
                                    break
                                    
                                if "audio" in message and message["audio"]:
                                    audio_chunk = base64.b64decode(message["audio"])
                                    audio_chunks.append(audio_chunk)
                                    
                                if message.get("isFinal", False):
                                    break
                                    
                                # Safety timeout after 15 seconds total
                                if time.time() - start_time > 15:
                                    logger.warning("TTS taking too long, breaking")
                                    break
                                    
                            except asyncio.TimeoutError:
                                logger.debug("TTS timeout - returning available audio")
                                break
                            except json.JSONDecodeError:
                                logger.warning("Invalid JSON from ElevenLabs")
                                continue
                        
                        result = b''.join(audio_chunks) if audio_chunks else self.generate_silence_placeholder()
                        logger.info(f"✅ TTS completed: {len(result)} bytes (semaphore released)")
                        return result
                        
                except websockets.exceptions.WebSocketException as ws_error:
                    logger.error(f"WebSocket error: {ws_error}")
                    return self.generate_silence_placeholder()
                
        except Exception as e:
            logger.error(f"💥 Error in ElevenLabs TTS: {e}")
            return self.generate_silence_placeholder()

    def generate_silence_placeholder(self) -> bytes:
        """Generate minimal silence as fallback"""
        try:
            duration_seconds = 0.1
            sample_rate = 8000
            samples = int(duration_seconds * sample_rate)
            silence_pcm = b'\x00' * (samples * 2)
            return audioop.lin2ulaw(silence_pcm, 2)
        except Exception:
            return b''

    async def send_audio_to_twilio(self, audio_data: bytes):
        """Sends audio data to Twilio as a media message."""
        try:
            # Base64 encode the audio data (mu-law)
            payload = base64.b64encode(audio_data).decode("utf-8")
            
            message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {
                    "payload": payload,
                },
            }
            
            await self.websocket.send_text(json.dumps(message))
            self.audio_responses_sent += 1
            
        except Exception as e:
            logger.error(f"💥 Error sending audio to Twilio: {e}")

    async def send_mark_message_to_twilio(self):
        """Sends a mark message to Twilio to indicate the end of a media segment."""
        try:
            mark_message = {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {
                    "name": f"end_of_agent_turn_{self.conversation_count}"
                }
            }
            logger.info(f"📤 Sending mark message to Twilio: {mark_message['mark']['name']}")
            await self.websocket.send_text(json.dumps(mark_message))
        except Exception as e:
            logger.error(f"💥 Error sending mark message to Twilio: {e}")

    def get_conversation_summary(self) -> dict:
        """Get a summary of the conversation."""
        return {
            "call_sid": self.call_sid,
            "start_time": self.start_time,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_exchanges": self.conversation_count,
            "conversation_history": self.conversation_history
        }

    async def cleanup(self):
        """Clean up resources"""
        logger.info(f"🧹 Cleaning up call session {self.call_sid}")
        
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()
            logger.info("❌ TTS task cancelled")
            
        if self.openai_task:
            self.openai_task.cancel()
            logger.info("❌ OpenAI task cancelled")
            
        if self.openai_websocket:
            await self.openai_websocket.close()
            logger.info("🔌 OpenAI WebSocket closed")
            
        logger.info(f"📊 Final TTS Semaphore status: {self.agent.tts_semaphore._value}/{self.agent.max_concurrent_tts} slots available") 