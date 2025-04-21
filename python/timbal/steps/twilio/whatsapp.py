import json
import os
from typing import Any

import httpx

from ...types.field import Field


async def send_whatsapp_message(
    to: str = Field(description="The WhatsApp account to send the message to in E.164 format"),
    message: str = Field(
        description=(
            "The message to send. "
            "The maximum length is 1600 characters. "
            "If the message is longer, it will be split into multiple messages."
        ),
    ),
) -> None:
    """A WhatsApp session begins when a user sends a message to your app. 
    Sessions are valid for 24 hours after the most recently received message, 
    during which you can communicate with customers using free-form messages. 
    To send a message outside the 24-hour session window, you must use a pre-approved message template.
    """

    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from = os.getenv("TWILIO_FROM")
    
    url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_account_sid}/Messages.json"

    data = {
        "To": f"whatsapp:{to}",
        "From": f"whatsapp:{twilio_from}",
        "Body": message
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            data=data,
            auth=(twilio_account_sid, twilio_auth_token)
        )
        response.raise_for_status()


async def send_whatsapp_template(
    to: str = Field(description="The WhatsApp account to send the message to in E.164 format"),
    template_sid: str = Field(description="The template SID to send the message to"),
    template_params: dict[str, Any] = Field(
        description=(
            "The parameters to send to the template. "
            "The parameters must match the template parameters exactly."
        ),
    ),
) -> None:
    """A WhatsApp session begins when a user sends a message to your app. 
    Sessions are valid for 24 hours after the most recently received message, 
    during which you can communicate with customers using free-form messages. 
    To send a message outside the 24-hour session window, you must use a pre-approved message template.
    """

    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from = os.getenv("TWILIO_FROM")
    
    url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_account_sid}/Messages.json"

    data = {
        "To": f"whatsapp:{to}",
        "From": f"whatsapp:{twilio_from}",
        "ContentSid": template_sid,
        "ContentVariables": json.dumps(template_params),
    }
    
    async with httpx.AsyncClient() as client: # TODO Think timeout
        response = await client.post(
            url,
            data=data,
            auth=(twilio_account_sid, twilio_auth_token)
        )
        response.raise_for_status()
