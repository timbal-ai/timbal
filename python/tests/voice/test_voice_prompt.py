"""Voice turns keep a stable prompt prefix and a per-session cache key."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from timbal import Agent
from timbal.core.llm.prompt_cache import SystemPrompt, current_prompt_cache_key
from timbal.core.test_model import TestModel
from timbal.state import get_run_context
from timbal.voice import VoiceSession
from timbal.voice.session import _current_voice_session


@pytest.fixture(autouse=True)
def clean_voice_context():
    from timbal.core.llm.prompt_cache import bind_prompt_cache_key, reset_prompt_cache_key
    from timbal.state import _call_id, _run_context_var

    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    voice_token = _current_voice_session.set(None)
    cache_token = bind_prompt_cache_key(None)
    yield
    reset_prompt_cache_key(cache_token)
    _current_voice_session.reset(voice_token)
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)


def _session(**kwargs) -> VoiceSession:
    agent = Agent(name="voice_assistant", model=TestModel(responses=["ok"]), system_prompt="Be brief.")
    return VoiceSession(agent=agent, stt=MagicMock(), tts=MagicMock(), turn_detector="heuristic", **kwargs)


class TestPromptForTurn:
    def test_greeting_note_is_not_part_of_the_static_prefix(self):
        session = _session(call_context={"from": "+1555", "account": "42"})
        session._turn_prompt_now = datetime(2026, 11, 4, 9, 40, tzinfo=UTC)
        session._greeting_entry = SimpleNamespace(text="Hello, this is the clinic.")
        session._last_run_context = None
        first = session.prompt_for_turn("Be brief.")
        session._last_run_context = SimpleNamespace(id="turn-1")
        second = session.prompt_for_turn("Be brief.")
        assert first.static == second.static == "Be brief."
        assert "Hello, this is the clinic." in first.dynamic
        assert "Hello, this is the clinic." not in second.dynamic
        assert "from: +1555" in second.dynamic
        assert second.dynamic.index("from: +1555") < second.dynamic.index("Current date and time")

    @pytest.mark.asyncio
    async def test_handler_uses_the_session_prompt(self):
        seen: dict[str, SystemPrompt] = {}

        def hook():
            ctx = get_run_context()
            for span in ctx._trace.values():
                raw = span.input.get("system_prompt") if isinstance(span.input, dict) else None
                if isinstance(raw, SystemPrompt):
                    seen["prompt"] = raw

        agent = Agent(
            name="voice_assistant",
            model=TestModel(responses=["ok"]),
            system_prompt="Be brief.",
            post_hook=hook,
        )
        session = VoiceSession(agent=agent, stt=MagicMock(), tts=MagicMock(), turn_detector="heuristic")
        session._turn_prompt_now = datetime(2026, 11, 4, 9, 40, tzinfo=UTC)
        token = _current_voice_session.set(session)
        try:
            await agent(prompt="hi").collect()
        finally:
            _current_voice_session.reset(token)
        prompt = seen["prompt"]
        assert prompt.static == "Be brief."
        assert prompt.dynamic.startswith("Current date and time (UTC): 2026-11-04T09:40:00Z")

    @pytest.mark.asyncio
    async def test_run_binds_the_session_id(self):
        session = _session()
        session.stt.close = AsyncMock()
        session.tts.close = AsyncMock()
        session.session_id = "call-123"

        async def _stop():
            raise RuntimeError("stop")

        session.prepare = _stop
        seen: list[str | None] = []

        def spy(key):
            seen.append(current_prompt_cache_key())
            from timbal.core.llm.prompt_cache import bind_prompt_cache_key

            token = bind_prompt_cache_key(key)
            seen.append(current_prompt_cache_key())
            return token

        with patch("timbal.voice.session.bind_prompt_cache_key", spy):
            async for _event in session.run(_silent()):
                pass
        assert seen == [None, "call-123"]
        assert current_prompt_cache_key() is None


async def _silent():
    if False:
        yield b""
