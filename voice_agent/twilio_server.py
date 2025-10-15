import asyncio
import base64
import json
import os
from collections.abc import AsyncGenerator
from contextlib import suppress

from audio_bg import VoiceAgent
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from timbal.core.agent import logger
from twilio.twiml.voice_response import Connect, Stream, VoiceResponse

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

app = FastAPI()


os.environ["TIMBAL_ORG_ID"] = "1"
# os.environ["TIMBAL_APP_ID"] = "125"



@app.post("/incoming-call")
async def incoming_call():
    """
    Twilio webhook endpoint for incoming calls.
    Returns TwiML that connects the call to our WebSocket.
    """
    response = VoiceResponse()
    connect = Connect()
    
    # Create a bidirectional audio stream to our WebSocket endpoint
    stream = Stream(url='wss://ta-stratospherical-supersentimentally.ngrok-free.dev/media-stream')
    connect.append(stream)
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")


async def twilio_audio_stream(websocket: WebSocket) -> AsyncGenerator[bytes, None]:
    while True:
        message = await websocket.receive_text()
        data = json.loads(message)
        if data['event'] == 'start':
            print("Twilio stream started")
        elif data['event'] == 'media':
            # Twilio sends base64-encoded mulaw audio
            yield base64.b64decode(data['media']['payload'])
        elif data['event'] == 'stop':
            print("Twilio stream stopped - but continuing to listen for more audio")
            # Don't break here - continue listening for more audio
        else:
            logger.warning(f"Unknown event: {data}")



async def send_audio_to_twilio(websocket: WebSocket, audio_stream: AsyncGenerator[bytes, None], sid_getter: callable):
    try:
        async for audio_chunk in audio_stream:
            sid = sid_getter()
            encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
            await websocket.send_json({
                "event": "media",
                "streamSid": sid,
                "media": {
                    "payload": encoded_audio
                }
            })
    except asyncio.CancelledError:
        print("Audio transmission to Twilio was cancelled")
    except Exception as e:
        print(f"Error sending audio to Twilio: {e}")
        raise


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    Handles bidirectional audio streaming between Twilio and VoiceAgent.
    
    Flow:
    1. Accept WebSocket connection
    2. Receive 'start' event from Twilio
    3. Create bidirectional audio pipeline:
       - Twilio → VoiceAgent (mulaw 8kHz → PCM16 16kHz)
       - VoiceAgent → Twilio (ulaw_8000 base64 encoded)
    4. Maintain continuous streaming throughout the call
    """
    await websocket.accept()
    
    stream_sid = None
    voice_agent_instance = None

    def sid_getter():
        return stream_sid
    
    try:
        # Read the first 'start' message from Twilio
        message = await websocket.receive_text()
        data = json.loads(message)
        print(f"First message event: {data['event']}")
        
        if data['event'] == 'connected':
            message = await websocket.receive_text()
            data = json.loads(message)

        if data['event'] != 'start':
            raise RuntimeError("Expected 'start' event")
        

        stream_sid = data['start']['streamSid']
        call_sid = data['start']['callSid']
        print(f"📞 Call started - Stream: {stream_sid}, Call: {call_sid}")

        # Agent info
        system_prompt = open("system_prompt.txt", encoding="utf-8").read()
        
        # Create a new VoiceAgent instance for this call
        voice_agent_instance = VoiceAgent(
            name="twilio_voice_agent",
            model="openai/gpt-4o-mini",
            system_prompt=system_prompt,
            language="en",
            audio_format="g711_ulaw",
            vad_prefix_padding_ms=250,
            vad_silence_duration_ms=250,
            elevenlabs_voice_type="H6bZE3vdUcn6ksY6zH1x",
            elevenlabs_output_format="ulaw_8000"
        )

        voice_agent_instance._twilio_ws = websocket
        voice_agent_instance._twilio_stream_sid = stream_sid
        
        # Input Stream
        input_stream = twilio_audio_stream(websocket)

        # Start the voice agent session and keep it running for the entire call
        async with voice_agent_instance.session(input_stream=input_stream) as agent_output:
            # Create a task to send agent audio to Twilio
            await send_audio_to_twilio(websocket, agent_output, sid_getter)
        print(f"✅ Call ended - Stream: {stream_sid}")
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {stream_sid}")
    except Exception as e:
        print(f"❌ Error in media stream {stream_sid}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close websocket gracefully if still open
        with suppress(Exception):
            await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

