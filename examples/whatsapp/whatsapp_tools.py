
import httpx
from timbal.errors import APIKeyNotFoundError
import os

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
