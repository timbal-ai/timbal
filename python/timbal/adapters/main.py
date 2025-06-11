import asyncio
import json
import os

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from timbal.state.savers import InMemorySaver

from config import logger, SERVER_PORT, TO_PHONE_NUMBER, TWILIO_PHONE_NUMBER
from agent import CallHandlerAgent
from call_handlers import handle_call_start, handle_media, handle_call_stop
from twilio_client import make_twilio_call

# Global variables
NGROK_WSS_URL = None

# Initialize the call handler agent
state_saver = InMemorySaver()

# Set the global reference
agent = CallHandlerAgent(
    model="gpt-4o-mini",
    system_prompt=(
        """
        ### CORE DIRECTIVES ###
        1. **Language:** Always respond in Spanish. Your persona is a friendly survey agent from "VICIO".
        2. **Persona:** Your tone should be polite, professional, and friendly.
        3. **Communication Style:** Be clear and concise.
        4. **Text Output for TTS:** Your output will be converted to speech. Spell out numbers and symbols (e.g., write "cinco" not "5").

        ### PRIMARY OBJECTIVE ###
        * Your goal is to complete a two-question customer satisfaction survey.

        ### CONVERSATION FLOW ###
        1. **Introduction:**
           * Start with a friendly greeting: "¡Buenas! Te habla el equipo de VICIO. Queremos saber qué tal tu última experiencia con nosotros. ¿Nos regalas un minuto para un par de preguntas?"
           * If the user agrees, proceed. If they decline, thank them and use the `hang_up_call` tool.

        2. **Question 1: Satisfaction Rating:**
           * Ask the first question: "En una escala del uno al cinco, ¿qué tan satisfecho está con vicio en su ultima visita?"

        3. **Question 2: Reason for Rating (Conditional):**
           * **Analyze the user's response.** If they only give a number (e.g., "un tres"), then ask for the reason: "Entendido, gracias. ¿Podría decirnos brevemente el motivo de su puntuación?"
           * **If the user provides a reason along with their score** (e.g., "un tres, porque la hamburguesa estaba fría"), **do NOT ask again.** Simply acknowledge the reason and proceed to the conclusion.

        4. **Conclusion and Hang Up:**
           * Once the survey is complete, thank them for their time: "Agradecemos mucho sus comentarios. ¡Que tenga un buen día!"
           * Immediately after thanking them, you MUST use the `hang_up_call` tool to end the conversation.

        ### CRITICAL PROTOCOL ###
        * **Tool Syntax:** Your responses to the user MUST ONLY be natural, spoken language. NEVER include tool calls like `{hang_up_call}` in your text.
        *   **Correct Example:** First, say "Adiós!". Then, in a separate action, use the `hang_up_call` tool.
        *   **Incorrect Example:** Do NOT say "Adiós! {hang_up_call}".
        * **Hang Up:** Always use the `hang_up_call` tool to end the conversation once the survey is complete or the user declines to participate.
        """
    ),
    state_saver=state_saver,
    stream=True
)

# Update the global reference for the hang_up_call function
import config
config.call_agent = agent

# --- FastAPI App ---
app = FastAPI()

@app.get("/")
async def health_check():
    """Simple endpoint to verify the server is working."""
    return {"status": "ok", "message": "Twilio WebSocket server running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Twilio calls."""
    logger.info("🔗 New WebSocket connection request")
    await websocket.accept()
    logger.info("✅ WebSocket connection established")
    
    try:
        while True:
            message_json = await websocket.receive_text()
            message = json.loads(message_json)
            event_type = message.get("event")
            
            if event_type == "start":
                print("Start Message: ", message)
                await handle_call_start(websocket, message, agent)
            elif event_type == "media":
                await handle_media(websocket, message, agent)
            elif event_type == "stop":
                await handle_call_stop(websocket, message, agent)
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket connection closed by client")
    except Exception as e:
        logger.error(f"💥 WebSocket error: {e}")
    finally:
        logger.info("🧹 WebSocket connection terminated")


async def main():
    global NGROK_WSS_URL

    print("🔄 Starting FastAPI server...")
    
    config = uvicorn.Config(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    
    logger.info(f"🚀 FastAPI server started at http://0.0.0.0:{SERVER_PORT}")
    await asyncio.sleep(3)

    ngrok_input_url = os.getenv("NGROK_DOMAIN", "").strip()
    if not ngrok_input_url:
        logger.error("❌ NGROK_URL not found in .env file")
        server.should_exit = True
        await server_task
        return

    # Automatic conversion from HTTPS to WSS
    if ngrok_input_url.startswith("https://"):
        wss_url = ngrok_input_url.replace("https://", "wss://")
        if not wss_url.endswith("/ws"):
            wss_url += "/ws"
        NGROK_WSS_URL = wss_url
        print(f"🔄 Automatically converted to: {NGROK_WSS_URL}")
    elif ngrok_input_url.startswith("wss://"):
        if not ngrok_input_url.endswith("/ws"):
            ngrok_input_url += "/ws"
        NGROK_WSS_URL = ngrok_input_url
        print(f"✅ Valid WSS URL: {NGROK_WSS_URL}")
    else:
        logger.error("❌ Invalid URL. Must start with 'https://' or 'wss://'")
        server.should_exit = True
        await server_task
        return
    
    call_sid = await make_twilio_call(TO_PHONE_NUMBER, TWILIO_PHONE_NUMBER, NGROK_WSS_URL)

    try:
        await server_task
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
        server.should_exit = True
        await server_task
    finally:
        logger.info("👋 System finished")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Program stopped by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
