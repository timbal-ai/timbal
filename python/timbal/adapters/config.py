import logging
import os
from dotenv import find_dotenv, load_dotenv

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TwilioMediaStream")

# --- Load environment variables ---
load_dotenv(find_dotenv())

# Environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TO_PHONE_NUMBER = "+34639038963"
SERVER_PORT = 8080

# Real-time processing configuration
MIN_TRANSCRIPTION_LENGTH = 2

# Validate that credentials are loaded
required_credentials = {
    "Twilio": [TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER],
    "OpenAI": [OPENAI_API_KEY],
    "ElevenLabs": [ELEVENLABS_API_KEY]
}

for service, credentials in required_credentials.items():
    if not all(credentials):
        logger.error(f"Error: Missing {service} credentials in .env file")
        exit(1)

logger.info(f"🎤 Using ElevenLabs Voice ID: {ELEVENLABS_VOICE_ID}")
logger.info(f"🔑 ElevenLabs API Key configured: {'Yes' if ELEVENLABS_API_KEY else 'No'}")

# Global variables
NGROK_WSS_URL = None
call_agent = None 