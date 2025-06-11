import base64
import json
from datetime import datetime

from fastapi import WebSocket

from config import logger
from utils import generate_full_call_transcript, analyze_call_transcript, save_agent_memory_to_file


async def handle_call_start(websocket: WebSocket, message: dict, agent):
    """Handle the start of a call."""
    start_data = message.get("start", {})
    call_sid = start_data.get("callSid")
    stream_sid = message.get("streamSid")
    
    logger.info(f"📞 CALL STARTED - CallSid: {call_sid}")
    
    # Create call session using the agent
    call_session = agent.create_call_session(call_sid, stream_sid, websocket)
    
    # Initialize OpenAI connection
    openai_success = await call_session.initialize_openai_connection()
    
    if openai_success:
        logger.info("✅ OpenAI connection established successfully. Triggering agent to start conversation...")
        # Trigger the agent to send its initial message based on the system prompt.
        # We pass an empty string to simulate the start of the conversation.
        import asyncio
        asyncio.create_task(call_session.process_complete_transcription(""))


async def handle_media(websocket: WebSocket, message: dict, agent):
    """Handle received media (audio) data."""
    media_data = message.get("media", {})
    payload = media_data.get("payload")
    track = media_data.get("track")
    
    if payload and track == "inbound":
        # Find active session by websocket
        call_session = None
        for session in agent.active_calls.values():
            if session.websocket == websocket:
                call_session = session
                break
        
        if call_session:
            audio_bytes = base64.b64decode(payload)
            call_session.add_audio_chunk(audio_bytes)


async def handle_call_stop(websocket: WebSocket, message: dict, agent):
    """Handle the end of a call."""
    stop_data = message.get("stop", {})
    call_sid = stop_data.get("callSid")
    
    logger.info(f"📞 CALL ENDED - CallSid: {call_sid}")
    
    call_session = agent.get_call_session(call_sid)
    
    if call_session:
        # Generate transcript first
        await generate_full_call_transcript(call_session)
        
        # Analyze survey results using our new function
        logger.info("📝 Analyzing survey transcript...")
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
        
        # Clean up call session
        await call_session.cleanup()
        agent.remove_call_session(call_sid)


async def send_welcome_message(call_session):
    """Send a welcome message at the start of the call."""
    welcome_text = "Hola, le llamamos de Timbal en relación a una factura pendiente."
    
    call_session.add_conversation_entry("agent", welcome_text)
    await call_session.send_agent_response(welcome_text)
    logger.info("✅ Welcome message sent") 