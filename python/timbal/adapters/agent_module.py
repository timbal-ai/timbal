"""
Agent module for Twilio adapter integration.
This module exports a configured agent that can be loaded by the HTTP server.
"""

import os
from timbal import Agent
from timbal.state.savers import InMemorySaver
from timbal.core.agent.types.tool import Tool
from dotenv import find_dotenv, load_dotenv

from timbal.adapters.twilio_adapter import (
    TwilioCallAdapter,
    TwilioAdapterConfig,
    TwilioConfig,
    OpenAIConfig,
    ElevenLabsConfig,
)
from timbal.adapters.tools import hang_up_call

# Load environment variables
load_dotenv(find_dotenv())

# Initialize Adapter and Agent
adapter_config = TwilioAdapterConfig(
    twilio=TwilioConfig(
        from_phone_number=os.getenv("TWILIO_PHONE_NUMBER"),
    ),
    openai=OpenAIConfig(),
    elevenlabs=ElevenLabsConfig(
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
    ),
    websocket_url="", # Will be set by HTTP server when ngrok is enabled
    to_phone_number="+34639038963",
)

state_saver = InMemorySaver()
twilio_adapter = TwilioCallAdapter(type="twilio_adapter", config=adapter_config)

# Create the agent
agent = Agent(
    model="gpt-4o-mini",
    system_prompt=(
        """
        Eres un agente telefónico que habla español. 
        Tu objetivo es preguntar que tal va el día.
        """
    ),
    # tools=[
    #    Tool(
    #        runnable=hang_up_call,
    #        description="Terminate the phone call.",
    #    )
    #],
    state_saver=state_saver,
    stream=True,
    adapters=[twilio_adapter],
)

# Export the agent for the HTTP server to load
__all__ = ["agent"] 