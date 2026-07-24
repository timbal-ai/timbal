"""Demo agent for A/B'ing voice turn-detection modes in the /voice UI.

Usage (from repo root)::

    export ELEVENLABS_API_KEY=...
    export TIMBAL_VOICE_TURN_DETECTOR=lexical   # heuristic|provider|lexical|local
    export TIMBAL_RUNNABLE="$(pwd)/examples/voice_turn_modes.py::agent"
    uv run python -m timbal.server --port 4444

Then open http://127.0.0.1:4444/voice

Modes:
  heuristic  — default; today's regex/similarity behavior
  provider   — trust ElevenLabs VAD commits (minimal filtering)
  lexical    — punctuation/dangling HOLD (noticeable mid-thought pauses)
  local      — audio EOU (Smart Turn v3 ONNX with `timbal[voice]`; else == heuristic)
"""

from __future__ import annotations

import os

from timbal import Agent
from timbal.voice import resolve_turn_detector

_MODE = os.environ.get("TIMBAL_VOICE_TURN_DETECTOR", "heuristic").strip().lower()

agent = Agent(
    name="voice_turn_modes",
    model=os.environ.get("TIMBAL_VOICE_DEMO_MODEL", "groq/llama-3.1-8b-instant"),
    system_prompt=(
        "You are a concise voice assistant. Keep replies to 1–2 short sentences. "
        f"(turn_detector mode: {_MODE})"
    ),
    tools=[],
)

# Server reads this on startup (instance or mode name). Client WS JSON cannot override it.
agent.voice_config = {
    "turn_detector": resolve_turn_detector(_MODE),
    "language": os.environ.get("TIMBAL_VOICE_LANGUAGE", "en"),
}
