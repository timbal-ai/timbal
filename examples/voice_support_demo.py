"""Local Norte Market support demo. See ``examples/voice_support_demo.md``.

Launch with ``examples/voice_support_demo_livekit.py``. That starts the
playground, which is the process that mints the LiveKit dial.

The microphone, Flux, ``openai/gpt-5.4-mini``, and ElevenLabs are the live
providers. The shop is not. Orders, customers, tracking, delivery notes, and
tickets live in this process, copied per voice session. Nothing is emailed,
refunded, or sent to a carrier.

``language`` is ``es`` so speech-to-text is hinted Spanish.
``reply_language`` is ``follow`` so a full English or Catalan request is
answered in that language. The prompt does not name a reply language.

Speculation is the measured block for ``openai/gpt-5.4-mini`` only. Another
``TIMBAL_VOICE_DEMO_MODEL`` leaves it off.
"""

from __future__ import annotations

import asyncio
import copy
import os
from typing import Any

from timbal import Agent
from timbal.voice import current_voice_session
from timbal.voice.speculative import documented_speculative_preset

_MODEL = os.environ.get("TIMBAL_VOICE_DEMO_MODEL", "openai/gpt-5.4-mini")
SLOW_SECS = 4.0
FAST_SECS = 0.2

# Seed data. Tools copy this into the session bag and never write it back.
_SEED_ORDERS: dict[str, dict[str, str]] = {
    "1001": {
        "order_id": "1001",
        "customer": "Ana Ruiz",
        "city": "Barcelona",
        "status": "packed",
        "delivery_note": "Leave it with the neighbour in 2B.",
    },
    "1002": {
        "order_id": "1002",
        "customer": "Jordi Puig",
        "city": "Girona",
        "status": "processing",
        "delivery_note": "Morning delivery.",
    },
    "1003": {
        "order_id": "1003",
        "customer": "Marta Soler",
        "city": "Valencia",
        "status": "delivered",
        "delivery_note": "Handed to the customer.",
    },
    "4040": {
        "order_id": "4040",
        "customer": "Pol Serra",
        "city": "Lleida",
        "status": "packed",
        "delivery_note": "Call on arrival.",
    },
}
_SEED_TRACKING = {
    "1001": "NM1001, in transit, Barcelona hub.",
    "1002": "NM1002, label printed, not scanned yet.",
    "1003": "NM1003, delivered yesterday.",
    "4040": "NM4040, waiting for the warehouse scan.",
}


def demo_voice_config(model: str | None = None) -> dict[str, Any]:
    """Spanish hint, replies follow the caller, ship knobs otherwise."""
    preset = documented_speculative_preset(model or _MODEL)
    config: dict[str, Any] = {
        "language": "es",
        "reply_language": "follow",
        "ignore_backchannels": True,
        "speculative_generation": {"enabled": preset.enabled, "preemptive_tts": preset.preemptive_tts},
        "max_tts_pause_secs": 0.2,
        # Start the first spoken clause at a comma once it has four words, so a
        # long one-sentence reply is not held until its period.
        "first_flush_min_words": 4,
        "filler": {"enabled": True, "soft_timeout_secs": 0.8, "message": "Un momento."},
        "hangup_grace_secs": 1.2,
        "entity_patience": True,
    }
    from timbal.voice import VoiceConfig

    VoiceConfig.model_validate(config)
    return config


def _session_key() -> str:
    session = current_voice_session()
    if session is None or not getattr(session, "session_id", None):
        return "no-session"
    return str(session.session_id)


# One bag per voice session. A new playground connection does not see another
# call's notes or tickets. Restarting the process drops all of them.
_BAGS: dict[str, dict[str, Any]] = {}


def _bag() -> dict[str, Any]:
    key = _session_key()
    bag = _BAGS.get(key)
    if bag is None:
        bag = {
            "orders": copy.deepcopy(_SEED_ORDERS),
            "failures": {},
            "pending": None,
            "tickets": [],
        }
        _BAGS[key] = bag
    return bag


def reset_demo_store() -> None:
    """Drop every session bag. Tests and a manual reset use this."""
    _BAGS.clear()


def _norm_id(order_id: str) -> str:
    return "".join(ch for ch in str(order_id) if ch.isdigit()) or str(order_id).strip()


async def lookup_order(order_id: str) -> str:
    """Look up one simulated order. Unknown ids are not found. 1002 is slow. 4040 fails once."""
    oid = _norm_id(order_id)
    bag = _bag()
    if oid == "1002":
        await asyncio.sleep(SLOW_SECS)
    else:
        await asyncio.sleep(FAST_SECS)
    if oid == "4040":
        seen = bag["failures"].get(oid, 0)
        bag["failures"][oid] = seen + 1
        if seen == 0:
            return "Simulated lookup failed for 4040. Ask them if you should retry. Nothing was changed."
    order = bag["orders"].get(oid)
    if order is None:
        return f"No simulated order {oid}. Do not invent one."
    return (
        f"Simulated order {oid} for {order['customer']} in {order['city']}. "
        f"Status {order['status']}. Note: {order['delivery_note']}"
    )


async def lookup_customer(name: str) -> str:
    """Find a simulated customer by name. This does not search a real directory."""
    await asyncio.sleep(FAST_SECS)
    needle = " ".join(name.casefold().split())
    if not needle:
        return "Ask for the customer name."
    matches = [order for order in _bag()["orders"].values() if needle in order["customer"].casefold()]
    if not matches:
        return f"No simulated customer matching {name}."
    lines = [f"{order['customer']}, order {order['order_id']}, {order['city']}, {order['status']}" for order in matches]
    return "Simulated customers: " + "; ".join(lines)


async def track_shipment(order_id: str) -> str:
    """Read the simulated tracking line. It does not call a carrier."""
    await asyncio.sleep(FAST_SECS)
    oid = _norm_id(order_id)
    if oid not in _bag()["orders"]:
        return f"No simulated shipment for {oid}."
    return "Simulated tracking: " + _SEED_TRACKING.get(oid, "No scan yet.")


async def stage_delivery_change(order_id: str, detail: str) -> str:
    """Preview a delivery-note change. Does not save it."""
    oid = _norm_id(order_id)
    note = " ".join(str(detail).split())
    bag = _bag()
    if oid not in bag["orders"]:
        return f"No simulated order {oid}. Nothing staged."
    if not note:
        return "Ask what the delivery note should say. Nothing staged."
    bag["pending"] = {"kind": "delivery", "order_id": oid, "detail": note}
    return (
        f"Staged only, not saved: order {oid}, new note '{note}'. "
        "Read this back and wait. Call apply_pending only after a clear yes."
    )


async def stage_support_ticket(order_id: str, summary: str) -> str:
    """Preview a support ticket. Does not open it."""
    oid = _norm_id(order_id)
    text = " ".join(str(summary).split())
    bag = _bag()
    if oid not in bag["orders"]:
        return f"No simulated order {oid}. Nothing staged."
    if not text:
        return "Ask what the ticket is about. Nothing staged."
    bag["pending"] = {"kind": "ticket", "order_id": oid, "detail": text}
    return (
        f"Staged only, not opened: ticket for {oid}, '{text}'. "
        "Read this back and wait. Call apply_pending only after a clear yes."
    )


async def apply_pending() -> str:
    """Save the staged note or ticket in this call's memory. Refuses when nothing is staged."""
    bag = _bag()
    pending = bag.get("pending")
    if not isinstance(pending, dict):
        return "Nothing is staged. Do not say it was saved."
    bag["pending"] = None
    oid = pending["order_id"]
    if pending["kind"] == "delivery":
        bag["orders"][oid]["delivery_note"] = pending["detail"]
        return f"Saved in this call only: order {oid} note is now {pending['detail']}."
    ticket_id = f"T-{len(bag['tickets']) + 1}"
    bag["tickets"].append({"ticket_id": ticket_id, "order_id": oid, "summary": pending["detail"]})
    return f"Opened in this call only: ticket {ticket_id} for order {oid}."


async def cancel_pending() -> str:
    """Drop a staged change. Does not undo one that was already applied."""
    bag = _bag()
    if not bag.get("pending"):
        return "Nothing was staged."
    bag["pending"] = None
    return "Staged change discarded. Nothing was saved."


async def end_call() -> str:
    """Hang up only if the caller stays quiet. If this says they spoke again, keep helping."""
    session = current_voice_session()
    if session is None:
        return "No live call. Do not hang up."
    if not await session.commit_hangup():
        return "Do not hang up. Keep helping."
    return "Ending the call."


# Do not name a reply language. ``reply_language: follow`` is the policy.
_PROMPT = (
    "You are the phone agent for Norte Market. "
    "The order tools are a simulated in-memory shop for this demo. "
    "They do not send messages, take payment, refund, or talk to a carrier. "
    "One or two short spoken sentences. No lists, no markdown, no emoji. "
    "Do not invent orders, prices, or tracking numbers. "
    "When they ask about an order, call lookup_order before you answer. "
    "When they ask who a customer is, call lookup_customer. "
    "When they ask where a parcel is, call track_shipment. "
    "If they ask for the order and the tracking together, call lookup_order and track_shipment in the same step. "
    "Order 1002 takes several seconds. Stay quiet until the tool returns, apart from any waiting line the session speaks. "
    "If a lookup says it failed, say so. Retry that same lookup once if they ask you to. "
    "If an order or customer is unknown, say so. "
    "When they dictate a phone number or spell a name, wait until they finish and read the entire thing back once. "
    "To change a delivery note, call stage_delivery_change, read the preview, and wait. "
    "To open a ticket, call stage_support_ticket, read the preview, and wait. "
    "Call apply_pending only after they clearly say yes. If they say no, call cancel_pending. "
    "Never say a change is saved until apply_pending says it is saved. "
    "Say one short goodbye and call end_call only when the caller is ending the conversation "
    "or asking you to hang up. "
    "If they are thanking you, or telling you to continue, stay on the line and ask what they need. "
    "Do not speak a farewell unless you are calling end_call. "
    "If end_call says not to hang up, stay on the line and help them. "
    "If they end the conversation again, say one short goodbye and call end_call again."
)

voice_config = demo_voice_config()

agent = Agent(
    name="norte_market_support",
    model=_MODEL,
    system_prompt=_PROMPT,
    tools=[
        lookup_order,
        lookup_customer,
        track_shipment,
        stage_delivery_change,
        stage_support_ticket,
        apply_pending,
        cancel_pending,
        end_call,
    ],
    voice_config=voice_config,
)


if __name__ == "__main__":
    import json

    print(json.dumps({"model": _MODEL, "voice_config": voice_config}, indent=2))  # noqa: T201
