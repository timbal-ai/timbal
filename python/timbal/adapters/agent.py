import asyncio
from collections import deque
from datetime import datetime

from timbal import Agent
from timbal.core.agent.types.tool import Tool
from timbal.state.savers import InMemorySaver
from twilio.rest import Client

from config import logger, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
from models import CallSession
from tools import hang_up_call


class CallHandlerAgent(Agent):
    """Agent subclass that handles all Twilio call logic."""
    
    def __init__(self, **kwargs):
        # Initialize parent first (Pydantic model)
        super().__init__(**kwargs)
        
        # Now set our custom attributes after Pydantic initialization
        self.active_calls = {}
        self.current_call_session = None
        self.twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # ElevenLabs concurrency control
        self.max_concurrent_tts = 3  # Stay well below the 5 limit
        self.tts_semaphore = asyncio.Semaphore(self.max_concurrent_tts)
        self.tts_queue = deque()
        self.processing_tts = False
        
        # Define the tools available to the agent
        tools = [
            Tool(
                runnable=hang_up_call,
                description="Signals the intention to terminate the phone call. Use this after you have said the final goodbye message to the user.",
            )
        ]
        
        # Add the tool to the existing tools
        self._load_tools(tools)
    
    async def _hang_up_call(self) -> str:
        """Terminates the phone call immediately."""
        logger.info("--- HANG UP PROCESS STARTED ---")
        if self.current_call_session is None:
            logger.warning("⚠️ _hang_up_call invoked but no current call session found.")
            return "No active call to hang up."
        
        try:
            logger.info(f"🔚 Agent requested to hang up call: {self.current_call_session.call_sid}")
            
            # Mark that we're going to hang up to avoid more processing
            self.current_call_session.processing = True
            
            # Cancel any ongoing tasks
            if hasattr(self.current_call_session, 'current_tts_task') and self.current_call_session.current_tts_task and not self.current_call_session.current_tts_task.done():
                self.current_call_session.current_tts_task.cancel()
                logger.info("❌ TTS task cancelled for hang up")
            
            # Create hang up task with delay
            asyncio.create_task(self._execute_hang_up_after_delay(self.current_call_session.call_sid, 2.0))
            logger.info("✅ Hang up task programmed successfully")
            
            return "Hang up process initiated."
            
        except Exception as e:
            logger.error(f"💥 Error in _hang_up_call: {e}")
            return "Error initiating hang up."

    async def _execute_hang_up_after_delay(self, call_sid: str, delay_seconds: float):
        """Execute hang up after delay to allow final message to be sent"""
        try:
            logger.info(f"⏰ Waiting {delay_seconds} seconds before hanging up call {call_sid}...")
            await asyncio.sleep(delay_seconds)
            
            logger.info(f"🔚 EXECUTING HANG UP for call: {call_sid}")
            
            call_session = self.get_call_session(call_sid)
            if not call_session:
                logger.error(f"❌ Cannot execute hang up. No call session found for call_sid: {call_sid}")
                return

            # Import here to avoid circular import
            from utils import generate_full_call_transcript, analyze_call_transcript, save_agent_memory_to_file

            # Generate transcript first
            await generate_full_call_transcript(call_session)
            
            # Analyze appointment scheduling
            logger.info("🚗 Analyzing TIMBAL transcript...")
            await analyze_call_transcript(call_session)
            
            # Save agent memory to files
            logger.info("🧠 Saving agent memory and state...")
            memory_file = await save_agent_memory_to_file(call_session, call_sid)
            print(call_session.context)
            
            if memory_file:
                logger.info(f"✅ Agent memory saved to: {memory_file}")
                print("\n" + "🧠 " + "=" * 78)
                print("MEMORIA GUARDADA EXITOSAMENTE")
                print("=" * 80)
                print(f"📝 Archivo TXT: {memory_file}")
                print(f"📊 Total intercambios: {call_session.conversation_count}")
                print(f"⏱️  Duración: {(datetime.now() - call_session.start_time).total_seconds():.1f} segundos")
                print("=" * 80)
            else:
                logger.error("❌ Failed to save agent memory")
            
            # 1. Send stop message to WebSocket
            if call_session.websocket:
                try:
                    import json
                    stop_message = {
                        "event": "stop",
                        "streamSid": call_session.stream_sid
                    }
                    await call_session.websocket.send_text(json.dumps(stop_message))
                    logger.info("📢 STOP message sent to WebSocket")
                except Exception as ws_error:
                    logger.error(f"💥 Error sending stop message to WebSocket: {ws_error}")
            
            # 2. Use Twilio client to hang up call
            try:
                call = self.twilio_client.calls(call_sid).update(status='completed')
                logger.info(f"✅ Call {call_sid} hung up successfully via Twilio. Status: {call.status}")
            except Exception as twilio_error:
                # Check if call was already completed
                if "Call is not in-progress" in str(twilio_error):
                    logger.warning(f"Call {call_sid} was already completed or hung up by the other party.")
                else:
                    logger.error(f"💥 Error using Twilio to hang up {call_sid}: {twilio_error}")
            
            # 3. Clean up session
            logger.info(f"🧹 Cleaning up session for call {call_sid}.")
            await call_session.cleanup()
            self.remove_call_session(call_sid)
                
        except Exception as e:
            logger.error(f"💥 Unhandled error in _execute_hang_up_after_delay for call {call_sid}: {e}")

    def create_call_session(self, call_sid: str, stream_sid: str, websocket) -> CallSession:
        """Create a new call session"""
        call_session = CallSession(call_sid, stream_sid, websocket, self)
        self.active_calls[call_sid] = call_session
        self.current_call_session = call_session
        return call_session

    def get_call_session(self, call_sid: str) -> CallSession:
        """Get an existing call session"""
        return self.active_calls.get(call_sid)

    def remove_call_session(self, call_sid: str):
        """Remove a call session"""
        if call_sid in self.active_calls:
            del self.active_calls[call_sid]
        if self.current_call_session and self.current_call_session.call_sid == call_sid:
            self.current_call_session = None 