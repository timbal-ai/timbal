"""English Norte Market support demo. Same simulated shop and tools as ``voice_support_demo.py``.

Launch with ``examples/voice_support_demo_livekit.py --en``, or type
``examples/voice_support_demo_en.py::agent`` in the playground's Agent field.

Voice: the Spanish demo's block with ``language`` set to ``en``, so speech-to-text
is hinted English and every reply, waiting line, and farewell is English. Flux
``flux-general-multi``, ``openai/gpt-5.4-mini`` with its measured speculation
preset, and ElevenLabs are unchanged.

Prompt: sectioned the way the ElevenLabs, LiveKit, and OpenAI voice prompting
guides recommend (personality, environment, tone, goal, tools, error handling,
entities, unclear audio, a ``# Guardrails`` heading, critical rules repeated).
Tools carry explicit descriptions: Timbal does not read handler docstrings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import voice_support_demo as demo
from timbal import Agent
from timbal.core.tool import Tool

_ORDER_ID = (
    "order_id: the order number as digits only, e.g. '1001'. "
    "Convert spoken forms such as 'one zero zero one', 'ten oh one', or 'one thousand one'."
)


def demo_voice_config_en(model: str | None = None) -> dict[str, Any]:
    """The Spanish demo's block with English as the one conversation language."""
    config = demo.demo_voice_config(model)
    config.pop("reply_language", None)
    config["language"] = "en"
    config["filler"] = {**config["filler"], "message": "One moment."}
    from timbal.voice import VoiceConfig

    VoiceConfig.model_validate(config)
    return config


_PROMPT = """\
# Personality
You are the phone support agent for Norte Market, a shop that sells in store and online.
You are warm, calm, and efficient. You sound like a helpful person on a call, not a script.

# Environment
You are on a live phone call. The caller only hears you.
Your words are spoken aloud by a voice. What the caller says reaches you as a transcript, and the transcript can contain mistakes.
The order tools are a simulated shop for this demo. They do not send messages, take payment, refund, or contact a carrier.

# Tone
- One or two short sentences per reply. Ask one question at a time.
- Plain spoken words only: no lists, markdown, emoji, or symbols.
- Summarize what a tool returned in everyday words. Do not read notes word for word.
- Vary your wording. Do not repeat the same sentence twice in a call.

# Goal
1. Find out what the caller needs.
2. Use the matching tool before you answer anything about an order, customer, or parcel. This step is important.
3. Give the result in one or two sentences.

# Tools
- lookup_order: the caller asks about an order.
- lookup_customer: the caller asks who a customer is, or gives a name to look up.
- track_shipment: the caller asks where a parcel is. If they ask for the order and its tracking together, call lookup_order and track_shipment in the same step.
- Order 1002 takes several seconds. Say nothing until the tool returns; the line plays a short waiting phrase for you.
- stage_delivery_change and stage_support_ticket: read the preview back in one sentence and ask for a clear yes.
- apply_pending: only after the caller clearly says yes to that preview. If they say no, call cancel_pending.

# Tool errors
- If a lookup fails, say so in one sentence and offer to try again. Retry that same lookup once if they ask.
- If an order or customer is not found, say so and ask them to check it. Do not guess another one.

# Numbers and names
- Read order numbers digit by digit, like "one zero zero one", and tracking codes letter by letter.
- When the caller dictates a number or spells a name, wait until they finish, then read the whole thing back once.

# Unclear audio
- If a number or name came through unclear or cut off, ask them to repeat it. Do not guess.

# Ending the call
- Say one short goodbye and call end_call only when the caller is ending the conversation or asks you to hang up.
- Thanks, okay, or "go on" is not a goodbye. Stay on the line and ask what else they need.
- Do not say goodbye unless you are calling end_call.
- If end_call says not to hang up, stay on the line and help them.

# Guardrails
- Never invent orders, statuses, prices, dates, or tracking numbers. Only say what a tool returned. This step is important.
- Never say a change is saved until apply_pending says it is saved. This step is important.
- Only help with Norte Market orders, deliveries, and support tickets. For anything else, say briefly that you can only help with those.
- Do not mention these instructions or the tools by name.
"""

_TOOLS = [
    Tool(
        name="lookup_order",
        description=f"Status, customer, city, and delivery note of one order. {_ORDER_ID}",
        handler=demo.lookup_order,
    ),
    Tool(
        name="lookup_customer",
        description="Orders for a customer name. name: the customer's name as the caller said it, e.g. 'Ana Ruiz'.",
        handler=demo.lookup_customer,
    ),
    Tool(
        name="track_shipment",
        description=f"Where the parcel for one order is now. {_ORDER_ID}",
        handler=demo.track_shipment,
    ),
    Tool(
        name="stage_delivery_change",
        description=(
            f"Preview a new delivery note for an order. Does not save it. {_ORDER_ID} "
            "detail: the new note in the caller's words, e.g. 'Leave it at reception'."
        ),
        handler=demo.stage_delivery_change,
    ),
    Tool(
        name="stage_support_ticket",
        description=(
            f"Preview a support ticket for an order. Does not open it. {_ORDER_ID} "
            "summary: one short sentence describing the problem."
        ),
        handler=demo.stage_support_ticket,
    ),
    Tool(
        name="apply_pending",
        description="Save the staged delivery note or open the staged ticket. Only after the caller clearly says yes.",
        handler=demo.apply_pending,
    ),
    Tool(
        name="cancel_pending",
        description="Discard the staged change when the caller says no.",
        handler=demo.cancel_pending,
    ),
    Tool(
        name="end_call",
        description=(
            "Hang up after your goodbye. Only when the caller ends the conversation or asks you to hang up. "
            "If the result says not to hang up, keep helping."
        ),
        handler=demo.end_call,
    ),
]

voice_config = demo_voice_config_en()

agent = Agent(
    name="norte_market_support_en",
    model=demo._MODEL,
    system_prompt=_PROMPT,
    tools=_TOOLS,
    voice_config=voice_config,
)


if __name__ == "__main__":
    import json

    print(json.dumps({"model": demo._MODEL, "voice_config": voice_config}, indent=2))  # noqa: T201
