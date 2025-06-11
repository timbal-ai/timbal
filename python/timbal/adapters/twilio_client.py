from twilio.rest import Client
from twilio.twiml.voice_response import Connect, VoiceResponse

from config import logger, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN


def create_twilio_client():
    """Create and return a Twilio client instance."""
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


async def make_twilio_call(target_phone_number: str, twilio_phone_number: str, websocket_url: str):
    """Initiate a Twilio call that streams to a WebSocket."""
    if not websocket_url:
        logger.error("WebSocket URL not provided. Cannot make call.")
        return

    response = VoiceResponse()
    connect = Connect()
    stream = connect.stream(url=websocket_url)
    stream.parameter(name="track", value="both_tracks")
    response.append(connect)
    response.pause(length=150)

    try:
        twilio_client = create_twilio_client()
        call = twilio_client.calls.create(
            to=target_phone_number,
            from_=twilio_phone_number,
            twiml=str(response)
        )
        logger.info(f"📞 Call initiated to {target_phone_number}. Call SID: {call.sid}")
        return call.sid
    except Exception as e:
        logger.error(f"💥 Error initiating call: {e}")
        return None 