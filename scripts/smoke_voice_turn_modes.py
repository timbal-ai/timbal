#!/usr/bin/env python3
"""Offline smoke: print what each turn-detector mode decides (no mic / no API keys).

    uv run python scripts/smoke_voice_turn_modes.py
"""

from __future__ import annotations

import asyncio

from timbal.voice.turn_detection import (
    HeuristicTurnDetector,
    LexicalTurnDetector,
    LocalAudioTurnDetector,
    ProviderTurnDetector,
    TurnState,
    resolve_turn_detector,
)


def _state(**kw) -> TurnState:
    base = dict(
        assistant_active=False,
        audio_playing=False,
        assistant_text="",
        active_user_text="",
        seconds_since_turn_start=10.0,
        seconds_since_last_commit=10.0,
        partials_since_last_commit=2,
    )
    base.update(kw)
    return TurnState(**base)


CASES = [
    ("idle complete", "Tell me a story.", _state()),
    ("idle incomplete (dangling)", "I was wondering about the", _state()),
    (
        "mid-turn short fragment",
        "estás?",
        _state(
            assistant_active=True,
            active_user_text="Hola, ¿qué tal",
            seconds_since_turn_start=2.0,
            seconds_since_last_commit=1.0,
        ),
    ),
    (
        "mid-turn long incomplete + follow-up",
        "the weather tomorrow please",
        _state(
            assistant_active=True,
            active_user_text="I was wondering if you could tell me about",
            seconds_since_turn_start=3.0,
            seconds_since_last_commit=1.0,
        ),
    ),
    ("noise", "(music playing)", _state()),
]


async def _run() -> None:
    modes = {
        "heuristic": HeuristicTurnDetector(),
        "provider": ProviderTurnDetector(),
        "lexical": LexicalTurnDetector(),
        "local (no AudioEouModel)": LocalAudioTurnDetector(),
    }
    print(f"{'mode':28} {'case':40} action / reason")
    print("-" * 90)
    for name, det in modes.items():
        for case, text, state in CASES:
            d = await det.on_committed(text, state)
            print(f"{name:28} {case:40} {d.action.value} / {d.reason}")
        print()

    print("resolve_turn_detector check:")
    for key in ("heuristic", "provider", "lexical", "local"):
        print(f"  {key!r:12} -> {type(resolve_turn_detector(key)).__name__}")


if __name__ == "__main__":
    asyncio.run(_run())
