import asyncio
import datetime as dt
from datetime import datetime
import json
import logging
import os

from dotenv import load_dotenv

from timbal import Agent
from timbal.errors import bail
from timbal.state import get_run_context
from timbal.types import Message

from whatsapp_tools import send_whatsapp_message
from system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _prompt_from_message(msg: dict) -> str:
    t = msg.get("type")
    if t == "text":
        return (msg.get("text") or {}).get("body", "").strip()
    if t == "image":
        mid = (msg.get("image") or {}).get("id")
        cap = (msg.get("image") or {}).get("caption", "")
        return f"[image:{mid}] {cap}".strip()
    if t == "audio":
        mid = (msg.get("audio") or {}).get("id")
        return f"[audio:{mid}]"
    if t == "document":
        fn = (msg.get("document") or {}).get("filename", "document")
        return f"[document:{fn}]"
    return f"[{t or 'unknown'}]"

def _append_jsonl(path: str, rec: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _recent_history(path: str, user_phone: str, limit: int = 10) -> list[str]:
    out = []
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
            for r in reversed(rows):
                if r.get("user_phone") == user_phone:
                    try:
                        payload = json.loads(r.get("message") or "{}")
                        out.append(f"{r.get('direction')}: {payload.get('content','')}")
                    except Exception:
                        continue
                    if len(out) >= limit:
                        break
        return list(reversed(out))
    except Exception:
        return []

def pre_hook():
    """Build professional prompt from WhatsApp metadata, message, and optional history."""
    trace = get_run_context().current_trace()
    payload = trace.input.get("_webhook")
    if not payload:
        bail("No payload found in webhook")

    value = payload["entry"][0]["changes"][0]["value"]
    if not value:
        bail("No value found in webhook")

    messages = value.get("messages", [])
    if not messages:
        bail("No messages found in webhook")

    contacts = value.get("contacts", [])
    profile = contacts[0].get("profile", {})
    name = profile.get("name", "")
    msg = messages[0]
    from_number = msg.get("from")
    message_id = msg.get("id")
    phone_number_id = value.get("metadata").get("phone_number_id")
    prompt = _prompt_from_message(msg)
    msg_type = msg.get("type")

    rec = {
        "id": msg.get("id"),
        "user_phone": from_number,
        "direction": "inbound",
        "message_type": msg_type,
        "message": json.dumps({"type": msg_type, "content": prompt}, ensure_ascii=False),
        "timestamp": msg.get("timestamp"),
        "conversation_id": f"{phone_number_id}_{from_number}",
        "user_name": name,
    }
    _append_jsonl("whatsapp_messages.jsonl", rec)

    trace.input["prompt"] = _recent_history("whatsapp_messages.jsonl", user_phone=from_number, limit=10)
    trace.input["whatsapp_from_number"] = from_number
    trace.input["name"] = name


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
