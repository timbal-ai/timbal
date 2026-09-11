"""Agents for exercising every voice_config knob in the /voice playground.

One module, several agents — pick one with the import spec's ``::name``. All
run on Groq (``qwen3.8-27b`` — fast enough that TTS never waits on the LLM),
override with ``TIMBAL_VOICE_DEMO_MODEL``. Keys come from the repo-root ``.env``
(the server ``load_dotenv()``s its cwd): ``GROQ_API_KEY`` + ``ELEVENLABS_API_KEY``
minimum; ``DEEPGRAM_API_KEY`` to A/B Flux from the playground.

From the repo root::

    uv run python -m timbal.server --import_spec examples/voice_playground_agents.py::plain --port 4444
    # then open http://127.0.0.1:4444/voice

or drive several at once from the standalone launcher (Target panel → Agent
field takes ``examples/voice_playground_agents.py::tools``, Start spawns it)::

    uv run python -m timbal.server.playground

Agents:

  plain      no thinking, no tools. Inbound + outbound openers, user-idle
             re-engagement, hang-up on a dead line. Flip *Simulate* to hear
             the two openers; go quiet after a reply to hear the check-in.
  thinking   reasoning ON, one slow tool — hear what 1–2s of silent thinking
             does to first audio (nothing covers it: the filler is
             tool-triggered; ask for the time to hear both stack up).
  tools      a 3–5s tool and a fast one, spoken filler with a follow-up,
             non-interruptible opener with a settle beat.
  outbound   an outbound campaign shape: silent on inbound (``greeting=""``),
             waits for the callee's "hello?" before opening, hangs up after
             25s of silence (voicemail). Use *Simulate → Outbound*.
  bare       declares no voice_config at all — the playground with nothing
             to show as "server default"; every knob comes from the browser.

Every value here is deliberately *declared* (not left default) so the Agent
panel lists it and each field's placeholder shows it. Override any of them
from the right-hand panel; ``Server default`` puts it back.
"""

from __future__ import annotations

import asyncio
import os
import random
from datetime import UTC, datetime

from timbal import Agent
from timbal.core.tool import Tool

_MODEL = os.environ.get("TIMBAL_VOICE_DEMO_MODEL", "groq/qwen/qwen3.8-27b")
_QWEN = "qwen" in _MODEL

# Groq's Qwen3 thinks by default; ``reasoning_effort`` is how it is switched.
# Other providers: ``model_params`` passes through untouched, so for Anthropic
# use {"thinking": {"type": "enabled", "budget_tokens": 1024}} and for OpenAI
# {"reasoning_effort": "low"} — set TIMBAL_VOICE_DEMO_MODEL accordingly.
_NO_THINKING = {"reasoning_effort": "none"} if _QWEN else {}
_THINKING = {"reasoning_effort": "default"} if _QWEN else {}

_STYLE = (
    "You are a warm, concise voice assistant on a phone call. One or two short spoken "
    "sentences per reply — no lists, no markdown, no emoji. Match the caller's language."
)


async def get_datetime() -> str:
    """Current UTC date and time. Slow on purpose — long enough for a spoken filler."""
    await asyncio.sleep(random.uniform(3.0, 5.0))
    return datetime.now(UTC).strftime("%A %d %B %Y, %H:%M UTC")


def roll_dice(sides: int = 6) -> int:
    """Roll a die with ``sides`` sides. Instant — no filler should fire for this."""
    return random.randint(1, max(2, sides))


# ---------------------------------------------------------------------------

plain = Agent(
    name="voice_plain",
    model=_MODEL,
    max_tokens=512,
    model_params=_NO_THINKING,
    system_prompt=f"{_STYLE} You are Sam from Northwind Dental.",
    tools=[],
    voice_config={
        "language": "en",
        "greeting": {
            "text": "Northwind Dental, this is Sam. How can I help you today?",
            "interruptible": True,
        },
        "outbound_greeting": {
            "text": "Hi, it's Sam from Northwind Dental calling about your appointment. Is now a good time?",
            "delay_ms": 300,
            "after_user_silence_secs": 2.0,
        },
        "user_idle": {
            "timeout_secs": 6.0,
            "text": ["Are you still there?", "Take your time — I'm here when you're ready."],
            "max_count": 2,
            "hangup_after_secs": 40.0,
        },
    },
)

thinking = Agent(
    name="voice_thinking",
    model=_MODEL,
    max_tokens=1024,
    model_params=_THINKING,
    system_prompt=(
        f"{_STYLE} You are a thoughtful tutor; reason carefully before answering. "
        "For date or time questions call get_datetime."
    ),
    tools=[Tool(handler=get_datetime, description="Return the current UTC date and time.")],
    voice_config={
        "language": "en",
        "greeting": "Hi, I'm your tutor. Ask me anything and I'll think it through with you.",
        # Thinking is dead air until first audio and nothing covers it — the
        # filler is tool-triggered (ask for the time to hear it fire on top of
        # the thinking). A turn timeout keeps a runaway think from stalling
        # the call.
        "filler": {"delay_secs": 1.2, "repeat_secs": 4.0, "model": "groq/qwen/qwen3.8-27b"},
        "turn_timeout_secs": 30.0,
        "turn_timeout_fallback": "Sorry, that one's taking me too long — let's try again.",
    },
)

tools = Agent(
    name="voice_tools",
    model=_MODEL,
    max_tokens=512,
    model_params=_NO_THINKING,
    system_prompt=(
        f"{_STYLE} For any date or time question you MUST call get_datetime and then speak "
        "the result. For anything about dice, luck or random numbers call roll_dice. "
        "Never claim you cannot do these."
    ),
    tools=[
        Tool(handler=get_datetime, description="Return the current UTC date and time. Always use for date/time."),
        Tool(handler=roll_dice, description="Roll a die. Use for dice, luck, or picking a random number."),
    ],
    voice_config={
        "language": "en",
        "greeting": {
            "text": "Hello, this is the concierge line. I can tell you the time or roll a die for you.",
            "interruptible": False,
            "delay_ms": 400,
        },
        "filler": {
            "delay_secs": 0.8,
            "repeat_secs": 2.5,
            "max_per_turn": 3,
            "model": "groq/qwen/qwen3.8-27b",
        },
        "user_idle": {"timeout_secs": 8.0, "instructions": "Ask if they still need anything, in one short line."},
    },
)

outbound = Agent(
    name="voice_outbound",
    model=_MODEL,
    max_tokens=512,
    model_params=_NO_THINKING,
    system_prompt=(
        f"{_STYLE} You are Alex from Riverside Clinic, calling to confirm tomorrow's 10am appointment. "
        "Confirm, reschedule, or take a message — then wrap up politely."
    ),
    tools=[],
    voice_config={
        "language": "en",
        # Nothing to say if someone dials *in* to this number...
        "greeting": "",
        # ...but on a call we placed: let them pick up first, open only if they don't.
        "outbound_greeting": {
            "text": "Hi, this is Alex from Riverside Clinic — I'm calling to confirm your appointment tomorrow at ten.",
            "interruptible": False,
            "delay_ms": 500,
            "after_user_silence_secs": 2.5,
        },
        # Voicemail / dead line: one nudge, then hang up.
        "user_idle": {"timeout_secs": 5.0, "text": "Hello? Can you hear me?", "max_count": 1, "hangup_after_secs": 25.0},
    },
)

bare = Agent(
    name="voice_bare",
    model=_MODEL,
    max_tokens=512,
    model_params=_NO_THINKING,
    system_prompt=_STYLE,
    tools=[],
)
