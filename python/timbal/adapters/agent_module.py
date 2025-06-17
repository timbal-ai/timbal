"""
Agent module for Twilio adapter integration.
This module exports a configured agent that can be loaded by the HTTP server.
"""

import os
from datetime import datetime
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


def get_service_price(service: str) -> str:
    """Get the price for a specific workshop service."""
    return f"El servicio {service} tiene un precio de cincuenta euros"


def check_availability(date: str) -> str:
    """Check workshop availability for a specific date."""
    return f"Para el {date} tenemos disponibilidad a las diez de la mañana"


# Create the agent
agent = Agent(
    model="gpt-4o-mini",
    system_prompt=(
        """
        Eres un agendador profesional de citas de taller mecánico. 
        Manten tus respuestas cortas y concisas. No mas de 1 o 2 frases.
        Tu objetivo es ayudar a los clientes a agendar citas para servicios de taller de manera eficiente y profesional.
        
        Instrucciones específicas:
        1. Saluda de manera cordial y profesional, no especifiques la hora del día.
        2. Pregunta qué servicio necesita el cliente para su vehículo.
        3. Mantén un tono conversacional y natural, evitando sonar robótico.
        4. Adapta tu lenguaje según la respuesta del cliente.
        5. Usa un lenguaje claro y fácil de entender.
        6. Cuando te pregunten por precios de servicios, usa la herramienta get_service_price para obtener la información precisa.
        7. Cuando necesites verificar disponibilidad horaria, usa la herramienta check_availability con el día específico (lunes, martes, etc.).
        8. Retorna números y unidades en formato texto, no en números.
        9. Si el cliente quiere agendar una cita, recopila la información necesaria: servicio, fecha y hora preferida.
        10. Confirma los detalles de la cita antes de finalizar.
        11. Cuando uses check_availability, especifica solo el día (ej: "martes", "lunes") no la hora completa.
        12. Si el cliente menciona una hora específica que no está disponible, sugiere las alternativas disponibles.
        13. Una vez confirmada la cita, pregunta si necesita información sobre precios o si hay algo más en lo que puedas ayudarle.
        
        
        Logica de colgar llamada {hang_up_call}:
        - Si el cliente parece ocupado, no quiere agendar, pide que termines la llamada, o la conversación ha llegado a su fin natural, responde con "{hang_up_call}" para terminar la llamada.
 
        Recuerda: Tu objetivo es agendar citas de manera eficiente y profesional, proporcionando un excelente servicio al cliente.
        """
    ),
    # tools=[
    #    Tool(
    #        runnable=hang_up_call,
    #        description="Terminate the phone call.",
    #    )
    #],
    tools=[
        Tool(
            runnable=get_service_price,
            description="Get the price for a specific workshop service.",
        ),
        Tool(
            runnable=check_availability,
            description="Check workshop availability for a specific date.",
        )
    ],
    state_saver=state_saver,
    stream=True,
    adapters=[twilio_adapter],
)

# Export the agent for the HTTP server to load
__all__ = ["agent"]