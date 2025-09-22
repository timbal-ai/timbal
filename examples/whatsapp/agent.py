import asyncio
from datetime import datetime
import datetime as dt

from timbal import Agent
from timbal.state import get_run_context
import os
import json
from whatsapp_tools import send_whatsapp_message

## this is just in case you are using timbal platform
from timbal.types import Message
from timbal.errors import bail
from media import process_whatsapp


from dotenv import load_dotenv

import logging

logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUR_BRAND = "Timbal AI"


def get_datetime() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")



SYSTEM_PROMPT = f"""
    title: "WHATSAPP AGENT from {YOUR_BRAND}",
    description: "You are a whatsapp agent from {YOUR_BRAND}. 
    Your goal is to help users find the perfect vehicle and manage all their automotive needs.",

    ## WHATSAPP MESSAGE FORMAT
    You are responding to messages in WhatsApp, so you can use the specific syntax format that WhatsApp will render automatically. Use these formats when appropriate to improve readability and emphasis:

    Text Format:
    - *text* = bold (for important information)
    - _text_ = italic (for clarifications or soft emphasis)
    - ~text~ = strikethrough (for corrections or obsolete information)
    - `text` = monospace (for code, commands or technical text)
    - ```code``` = code block (for long fragments)
    - > text = quote (for references or highlighted information)
    - Important: NUNCA uses  [View here](url)  ni ![Image](url), simply send the image link or the vehicle link. Whatsapp does not render these formats.

    ## Important rules:
    - The symbols must be attached to the text (without spaces)
    - Each opening symbol must have its closing symbol
    - You can combine formats: *_text_* for bold+cursive
    - Use with moderation, only when adding value to the response
    - For lists use emojis (•), numbers (1.) or hyphens (-)

    Current time: {get_datetime()}.

    <the rest of your system prompt> 
"""



def _get_value_from_webhook(payload: dict) -> dict | None:
    """Safely extract the WhatsApp 'value' payload from the webhook."""
    try:
        return payload["entry"][0]["changes"][0]["value"]
    except Exception:
        return None


def _build_contact_map(contacts: list[dict]) -> dict[str, str]:
    contact_map: dict[str, str] = {}
    for contact in contacts:
        wa_id = contact.get("wa_id")
        profile = contact.get("profile", {})
        name = profile.get("name", wa_id)
        if wa_id:
            contact_map[wa_id] = name
    return contact_map


def _extract_prompt_from_message(message: dict) -> str:
    """Create a minimal prompt string from the WhatsApp message."""
    message_type = message.get("type")
    if message_type == "text":
        return (message.get("text") or {}).get("body", "").strip()
    if message_type == "image":
        media_id = (message.get("image") or {}).get("id")
        caption = (message.get("image") or {}).get("caption", "")
        return f"[image:{media_id}] {caption}".strip()
    if message_type == "audio":
        media_id = (message.get("audio") or {}).get("id")
        return f"[audio:{media_id}]"
    if message_type == "document":
        filename = (message.get("document") or {}).get("filename", "document")
        return f"[document:{filename}]"
    return f"[{message_type or 'unknown'}]"


def _append_jsonl(path: str, record: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as err:
        logger.error(f"Failed to write record to {path}: {err}")


def _load_recent_history(path: str, user_phone: str, limit: int = 10) -> list[str]:
    lines: list[str] = []
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            # Read all lines and filter for user; keep last N
            all_records = [json.loads(l) for l in f if l.strip()]
            for r in reversed(all_records):
                if r.get("user_phone") == user_phone:
                    try:
                        payload = json.loads(r.get("message") or "{}")
                        lines.append(f"{r.get('direction')}: {payload.get('content','')}")
                    except Exception:
                        continue
                    if len(lines) >= limit:
                        break
            return list(reversed(lines))
    except Exception as err:
        logger.error(f"Failed to read history from {path}: {err}")
    return []


def pre_hook():
    """Build professional prompt from WhatsApp metadata, message, and optional history."""
    trace = get_run_context().current_trace()
    payload = trace.input.get("_webhook")
    if not payload:
        bail("No payload found in webhook")

    value = _get_value_from_webhook(payload)
    if not value:
        bail("No value found in webhook")

    messages = value.get("messages", [])
    contacts = value.get("contacts", [])
    if not messages:
        bail("No messages found in webhook")

    contact_map = _build_contact_map(contacts)
    msg = messages[0]
    from_number = msg.get("from")
    name = contact_map.get(from_number, from_number)
    phone_number_id = (value.get("metadata") or {}).get("phone_number_id")

    prompt_text = _extract_prompt_from_message(msg)

    # Lightweight history context (optional) - fetch BEFORE persisting current message
    history_lines = _load_recent_history("whatsapp_messages.jsonl", user_phone=from_number, limit=10)

    # Construct professional prompt
    prompt_parts: list[str] = []
    if name and name != from_number:
        prompt_parts.append(f"From: {name} ({from_number})")
    else:
        prompt_parts.append(f"From: {from_number}")
    prompt_parts.append(f"Message: {prompt_text}")
    if history_lines:
        prompt_parts.append("Previous conversation:")
        prompt_parts.append("\n".join(history_lines))

    assembled_prompt = "\n".join(prompt_parts)

    # Provide prompt and metadata to the agent
    trace.input["prompt"] = assembled_prompt
    trace.input["whatsapp_from_number"] = from_number
    trace.input["name"] = name

    # Persist inbound to JSONL after prompt is constructed, so current message isn't duplicated in history
    inbound_record = {
        "id": msg.get("id") or f"inbound_{from_number}_{msg.get('timestamp')}",
        "user_phone": from_number,
        "direction": "inbound",
        "message_type": msg.get("type"),
        "message": json.dumps({"type": msg.get("type"), "content": prompt_text}, ensure_ascii=False),
        "timestamp": msg.get("timestamp"),
        "conversation_id": f"{phone_number_id}_{from_number}",
        "user_name": name,
    }
    _append_jsonl("whatsapp_messages.jsonl", inbound_record)


def post_hook():
    """Send WhatsApp response and persist outbound to JSONL."""
    trace = get_run_context().current_trace()

    # Extract assistant text output (best-effort)
    response_text = ""
    try:
        response_text = (trace.output.content[0].text or "").strip()
    except Exception:
        raise ValueError("No response text found")
    
    logger.info(f"Response text: {response_text}")
    if not response_text:
        return

    from_number = trace.input.get("whatsapp_from_number")
    if not from_number:
        return

    try:
        send_whatsapp_message(to=from_number, message=response_text)
    except Exception as err:
        logger.error(f"Failed to send WhatsApp message: {err}")

    outbound_record = {
        "id": f"outbound_{from_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "user_phone": from_number,
        "direction": "outbound",
        "message_type": "agent",
        "message": json.dumps({"type": "text", "content": response_text}, ensure_ascii=False),
        "timestamp": datetime.now().isoformat(),
        "conversation_id": "",
        "user_name": trace.input.get("name", ""),
    }
    _append_jsonl("whatsapp_messages.jsonl", outbound_record)

agent = Agent(
    name="whatsapp-template",
    description="This is a template for a WhatsApp agent",
    model="openai/gpt-4.1-mini",
    pre_hook=pre_hook,
    post_hook=post_hook,
    system_prompt=SYSTEM_PROMPT
)


async def main():
    while True:
        prompt = input("User: ")
        if prompt == "q":
            break
        agent_output_event = await agent(prompt=prompt).collect()
        print(f"Agent: {agent_output_event.output}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        print("Goodbye!")
