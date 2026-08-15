"""Demo agent for A/B'ing voice turn-detection modes in the /voice UI.

Usage (from repo root)::

    export ELEVENLABS_API_KEY=...
    export TIMBAL_VOICE_TURN_DETECTOR=lexical   # heuristic|provider|lexical|local
    export TIMBAL_RUNNABLE="$(pwd)/examples/voice_turn_modes.py::agent"
    uv run python -m timbal.server --port 4444

Then open http://127.0.0.1:4444/voice

Modes:
  heuristic  — holdless regex/similarity (opt-in; default is local)
  provider   — trust ElevenLabs VAD commits (minimal filtering)
  lexical    — punctuation/dangling HOLD (noticeable mid-thought pauses)
  local      — audio EOU (Smart Turn v3 ONNX with `timbal[voice]`; else == heuristic)
"""

from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, timezone

from timbal import Agent
from timbal.core.tool import Tool
from timbal.voice import resolve_turn_detector

_MODE = os.environ.get("TIMBAL_VOICE_TURN_DETECTOR", "local").strip().lower()
# Qwen3.6-27B on Groq: capable enough for phatic voice turns, fast enough
# that TTS is not waiting on a 70B. Thinking is ON by default on this model
# and would eat 1–2s of silence before first audio — force it off.
_MODEL = os.environ.get("TIMBAL_VOICE_DEMO_MODEL", "groq/qwen/qwen3.6-27b")


async def get_datetime() -> str:
    """Get the current UTC date and time (slow on purpose for voice tool-call testing)."""
    await asyncio.sleep(random.uniform(3.0, 5.0))
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


agent = Agent(
    name="voice_turn_modes",
    model=_MODEL,
    max_tokens=1024,  # required when switching to Anthropic from the playground
    model_params={"reasoning_effort": "none"} if "qwen" in _MODEL else {},
    system_prompt=(
        "You are a concise voice assistant. Keep replies to 1–2 short sentences. "
        "When the user asks for the date and/or time, you MUST call get_datetime "
        "and then speak the returned value clearly (include both date and time). "
        "Never claim you cannot tell the time. "
        f"(turn_detector mode: {_MODE})"
    ),
    tools=[
        Tool(
            handler=get_datetime,
            description="Return the current UTC date and time. Always use this for date/time questions.",
        )
    ],
)

# Server reads this on startup (instance or mode name). Client WS JSON cannot override it.
agent.voice_config = {
    "turn_detector": resolve_turn_detector(_MODE),
    "language": os.environ.get("TIMBAL_VOICE_LANGUAGE", "en"),
}
