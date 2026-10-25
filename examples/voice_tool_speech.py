"""Live listening harness for #226 / #231. See voice_tool_speech.md."""

import asyncio
import os
import time

import structlog
from dotenv import load_dotenv
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.message import Message

# Explicit path allows sharing credentials with your main checkout. Exported
# shell values take precedence; never copy credentials into this example.
load_dotenv(os.environ.get("VOICE_DEMO_ENV", ".env"), override=False)
logger = structlog.get_logger("voice_tool_demo")
beat = float(os.environ.get("VOICE_DEMO_BEAT", "0"))
mode = os.environ.get("VOICE_DEMO_MODE", "live").lower()
if mode not in {"live", "scripted"}:
    raise ValueError("VOICE_DEMO_MODE must be live or scripted")


async def _lookup(delay: float, label: str) -> str:
    started = time.monotonic()
    logger.info("demo_tool_started", scenario=label, simulated_delay=delay)
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        logger.info("demo_tool_cancelled", scenario=label)
        raise
    logger.info("demo_tool_finished", scenario=label, elapsed=round(time.monotonic() - started, 3))
    return "Order 101 has shipped and arrives tomorrow."


async def fast_lookup() -> str:
    """Look up demo order 101 in 150 milliseconds. Use for fast and silent tests."""
    return await _lookup(0.15, "fast")


async def slow_lookup() -> str:
    """Look up demo order 101 in eight seconds. Use for slow and interruption tests."""
    return await _lookup(8.0, "slow")


def scripted_reply(messages):
    """Deterministic non-streaming control, with real tools and real STT/TTS."""
    if messages[-1].role == "tool":
        return "Order 101 has shipped and arrives tomorrow."
    text = messages[-1].collect_text().lower()
    if "stop" in text:
        return "Stopped. What would you like to do next?"
    if any(word in text for word in ("fast", "slow", "silent")):
        content = []
        if "silent" not in text:
            content.append(TextContent(text="Okay, let me check that order for you…"))
        content.append(ToolUseContent(
            id=f"lookup_{time.monotonic_ns()}",
            name="slow_lookup" if "slow" in text else "fast_lookup",
            input={},
        ))
        return Message(role="assistant", content=content, stop_reason="tool_use")
    return "Say fast test, slow test, or silent test."


agent = Agent(
    name="tool_speech_demo",
    model=(TestModel(handler=scripted_reply) if mode == "scripted"
           else os.environ.get("VOICE_DEMO_MODEL", "openai/gpt-4.1-mini")),
    system_prompt=(
        "You are a voice timing test assistant using simulated order data. "
        "When the user says FAST TEST, say exactly 'Okay, let me check that order for you…' "
        "then call fast_lookup in that same response. "
        "For SLOW TEST, say the same phrase then call slow_lookup in that same response. "
        "For SILENT TEST, call fast_lookup immediately with NO text before the tool call. "
        "After a tool returns, say exactly 'Order 101 has shipped and arrives tomorrow.' "
        "Never give the result before calling the tool. Never repeat the pre-tool phrase. "
        "If interrupted with stop, acknowledge briefly and do not restart the lookup. "
        "For anything else, ask the user to say fast test, slow test, or silent test."
    ),
    tools=[fast_lookup, slow_lookup],
    tracing_provider=InMemoryTracingProvider,
    voice_config={
        "language": "en",
        "stt_provider": "deepgram-flux",
        "stt_model": "flux-general-multi",
        "tts_provider": "elevenlabs",
        "tts_model": "eleven_v4_turbo",
        "tts_extra": {"stability": 0.5, "similarity_boost": 0.75},
        "tool_beat_secs": beat,
        "turn_detector": "provider",
        "filler": None,
        "greeting": None,
        "user_idle": None,
    },
)
logger.info("demo_config", mode=mode, tool_beat_secs=beat, model=str(agent.model))
