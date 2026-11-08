"""Callable #242 ship config. See ``examples/voice_fde_ship.md``.

The dict below is the measured production block (language pin, backchannels,
speculation with preemptive TTS, pause trim, soft-timeout filler, hang-up
grace, entity patience). Nothing else is set, so a playground left on
"Server default" is this config.

``TIMBAL_VOICE_LANGUAGE`` is the only switch: absent → ``en``; ``es`` (or
another BCP-47 tag) pins that; empty / ``unset`` / ``auto`` / ``none`` /
``off`` omits ``language`` so the reply follows the latest substantive
message. Do not also say "match the caller's language" in the prompt.
"""

from __future__ import annotations

import asyncio
import os

from timbal import Agent
from timbal.voice import VoiceConfig, current_voice_session

# Measured on openai/gpt-5.4-mini. Do not point this stack at
# groq/openai/gpt-oss-20b — that tail got worse. Override with
# TIMBAL_VOICE_DEMO_MODEL.
_MODEL = os.environ.get("TIMBAL_VOICE_DEMO_MODEL", "openai/gpt-5.4-mini")

_OMIT_LANGUAGE = frozenset({"", "unset", "auto", "none", "off"})


def ship_language() -> str | None:
    """Resolve ``TIMBAL_VOICE_LANGUAGE``. Absent means ``en``, not omit."""
    raw = os.environ.get("TIMBAL_VOICE_LANGUAGE", "en")
    token = raw.strip()
    if token.lower() in _OMIT_LANGUAGE:
        return None
    return token


def ship_voice_config(language: str | None = None) -> dict:
    """The #242 ship block. ``language`` is the only field that moves."""
    # language: "en" or "es"; omit to follow the latest substantive message.
    config: dict = {
        "ignore_backchannels": True,
        "speculative_generation": {"enabled": True, "preemptive_tts": True},
        "max_tts_pause_secs": 0.2,
        "filler": {"enabled": True, "soft_timeout_secs": 0.8, "message": "One moment."},
        "hangup_grace_secs": 1.2,
        "entity_patience": True,
    }
    if language is not None:
        config = {"language": language, **config}
    VoiceConfig.model_validate(config)
    return config


async def check_order(order_id: str = "1001") -> str:
    """Look up an order. Takes about four seconds. Status only, no delivery window."""
    # Async sleep. A blocking sleep never yields, so the soft-timeout filler cannot play.
    await asyncio.sleep(4.0)
    return f"Order {order_id} is packed. Status is in transit."


async def end_call() -> str:
    """Hang up only if the caller stays quiet. If this says they spoke again, keep helping."""
    # Await commit_hangup. session.end_call() alone requests teardown before a
    # retraction can cancel it. Real side effects (CRM, carrier) belong after True.
    session = current_voice_session()
    if session is None:
        return "No live call. Do not hang up."
    if not await session.commit_hangup():
        return "Do not hang up. Keep helping."
    return "Ending the call."


# Do not name a reply language here. A set ``VoiceConfig.language`` pins it;
# an omitted one follows the latest substantive message, and that instruction
# yields if this prompt already names a single language.
_PROMPT = (
    "You are a phone support agent. One or two short spoken sentences. "
    "No lists, no markdown, no emoji. "
    "When they ask about an order, call check_order before you answer and then say only what it returned. "
    "When they dictate a phone number or spell a name, wait until they finish and read the entire number "
    "or the entire name back once, digit by digit or letter by letter. "
    "When they say a clear goodbye or ask you to hang up, say one short goodbye and call end_call. "
    "Thanks, vamos, and vámonos are not a goodbye. "
    "If end_call says not to hang up, stay on the line and help them. "
    "If they say goodbye again, say one short goodbye and call end_call again."
)

voice_config = ship_voice_config(ship_language())

agent = Agent(
    name="voice_fde_ship",
    model=_MODEL,
    system_prompt=_PROMPT,
    tools=[check_order, end_call],
    voice_config=voice_config,
)


if __name__ == "__main__":
    import json

    print(json.dumps(voice_config, indent=2))  # noqa: T201
