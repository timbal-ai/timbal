"""
WhatsApp Messages Module

Setup:
1. Create app at https://developers.facebook.com/apps/
2. Configure WhatsApp Business API
3. Set environment variables:
   - WHATSAPP_ACCESS_TOKEN: the access token for WhatsApp Business API
   - WHATSAPP_PHONE_NUMBER_ID: the phone number ID from WhatsApp Business API

Example Usage:
>>> send_whatsapp_message(to="34639038963", message="Hello, world!")
>>> send_whatsapp_template_message(to="34639038963", template_name="hello_world", language_code="en")

Important Notes:
- Customer must initiate conversation first
- 24-hour window restriction applies
- Only template messages can be sent outside the 24-hour window
"""
import os
import httpx
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from timbal.errors import APIKeyNotFoundError

import logging 
logger = logging.getLogger(__name__)

# Global storage for customer-initiated conversations
_customer_initiated_chats: Dict[str, Dict[str, Any]] = {}


def mark_customer_initiated(phone_number: str) -> None:
    """Mark that a customer has initiated a conversation."""
    _customer_initiated_chats[phone_number] = {
        "initiated_at": datetime.now(),
        "can_receive_messages": True
    }
    print(f"✅ Customer {phone_number} conversation window opened/renewed")

def can_send_message(phone_number: str) -> bool:
    """Check if we can send a message to this phone number."""
    chat_info = _customer_initiated_chats.get(phone_number)
    
    if not chat_info:
        print(f"❌ Cannot send to {phone_number}: Customer has not initiated conversation")
        return False

    # Check 24-hour window
    now = datetime.now()
    time_diff = now - chat_info["initiated_at"]
    hours_elapsed = time_diff.total_seconds() / 3600
    
    if hours_elapsed > 24:
        print(f"❌ Cannot send to {phone_number}: 24-hour window expired ({hours_elapsed:.1f}h ago)")
        chat_info["can_receive_messages"] = False
        return False

    print(f"✅ Can send to {phone_number}: {24 - hours_elapsed:.1f}h remaining")
    return True

def get_conversation_status(phone_number: str) -> Dict[str, Any]:
    """Get the conversation status for a phone number."""
    chat_info = _customer_initiated_chats.get(phone_number)
    
    if not chat_info:
        return {
            "can_send": False,
            "reason": "Cliente debe iniciar la conversación",
            "hours_remaining": 0
        }

    now = datetime.now()
    time_diff = now - chat_info["initiated_at"]
    hours_elapsed = time_diff.total_seconds() / 3600
    hours_remaining = max(0, 24 - hours_elapsed)
    
    return {
        "can_send": hours_remaining > 0,
        "reason": "Ventana activa" if hours_remaining > 0 else "Ventana de 24h expirada",
        "hours_remaining": round(hours_remaining, 1),
        "initiated_at": chat_info["initiated_at"]
    }

def clean_expired_conversations() -> int:
    """Clean expired conversation windows."""
    now = datetime.now()
    cleaned = 0
    
    expired_numbers = []
    for phone_number, chat_info in _customer_initiated_chats.items():
        time_diff = now - chat_info["initiated_at"]
        hours_elapsed = time_diff.total_seconds() / 3600
        
        if hours_elapsed > 24:
            expired_numbers.append(phone_number)
    
    for phone_number in expired_numbers:
        del _customer_initiated_chats[phone_number]
        cleaned += 1
    
    if cleaned > 0:
        print(f"🧹 Cleaned {cleaned} expired conversation windows")
    
    return cleaned


def send_whatsapp_message(
    to: str,
    message: str
) -> dict:
    """Send a text message via WhatsApp Business API."""
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    
    if not access_token:
        raise APIKeyNotFoundError("WHATSAPP_ACCESS_TOKEN not found")
    if not phone_number_id:
        raise APIKeyNotFoundError("WHATSAPP_PHONE_NUMBER_ID not found")
    
    try:
        url = f"https://graph.facebook.com/v20.0/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "text": {"body": message},
        }
        response = httpx.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        resp = e.response
        error_msg = f"HTTP {resp.status_code}: {resp.text}"
        raise ValueError(f"Failed to send WhatsApp message: {error_msg}")
    except httpx.RequestError as e:
        raise ValueError(f"Network error sending WhatsApp message: {e}")


def send_whatsapp_template_message(
    to: str,
    template_name: str,
    language_code: str,
    components: Optional[list]
) -> dict:
    """Send a template message via WhatsApp Business API."""
    
    # Enable calling this step without pydantic model_validate()
    to = to.default if hasattr(to, "default") else to
    template_name = template_name.default if hasattr(template_name, "default") else template_name
    language_code = language_code.default if hasattr(language_code, "default") else language_code
    components = components.default if hasattr(components, "default") else components

    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    
    if not access_token:
        raise APIKeyNotFoundError("WHATSAPP_ACCESS_TOKEN not found")
    if not phone_number_id:
        raise APIKeyNotFoundError("WHATSAPP_PHONE_NUMBER_ID not found")
    
    url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {
                "code": language_code
            }
        }
    }
    
    if components:
        payload["template"]["components"] = components
    
    response = httpx.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()



def send_typing_and_read(message_id: str) -> dict:
    """Marca un mensaje como leído y envía el indicador de 'escribiendo' en WhatsApp."""

    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

    if not access_token or not phone_number_id:
        raise ValueError("Faltan credenciales de WhatsApp API")

    url = f"https://graph.facebook.com/v19.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    with httpx.Client() as client:
        # 1. Marcar como leído (doble check azul)
        read_payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        client.post(url, headers=headers, json=read_payload)

        # 2. Enviar indicador "escribiendo..."
        typing_payload = {
            "messaging_product": "whatsapp",
            "typing_indicator": {"type": "typing_on"}
        }
        response = client.post(url, headers=headers, json=typing_payload)

        return response.json()

def mark_whatsapp_message_as_read(
    message_id: str
) -> dict:
    """Mark a WhatsApp message as read."""
    
    # Enable calling this step without pydantic model_validate()
    message_id = message_id.default if hasattr(message_id, "default") else message_id

    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    
    if not access_token:
        raise APIKeyNotFoundError("WHATSAPP_ACCESS_TOKEN not found")
    if not phone_number_id:
        raise APIKeyNotFoundError("WHATSAPP_PHONE_NUMBER_ID not found")
    
    url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id
    }
    
    response = httpx.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()

