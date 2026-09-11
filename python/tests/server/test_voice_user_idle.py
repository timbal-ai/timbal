"""``VoiceConfig.user_idle`` — re-engage a silent user, hang up on a dead line.

Same harness as the greeting tests: an STT that stays open until told to
finish, a tiny fake TTS, a ``TestModel`` agent.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice import (
    AgentTextDone,
    SessionEnded,
    TranscriptEvent,
    UserIdleConfig,
    VoiceConfig,
    VoiceSession,
    VoiceSessionEvent,
)

from .test_voice_greeting import GREETING, REPLY, _OpenSTT, _run
from .test_voice_ws import _make_tts_class


def _session(stt: _OpenSTT, **user_idle: object) -> VoiceSession:
    agent = Agent(name="idle_test", model=TestModel(responses=[REPLY]), tools=[])
    return VoiceSession(agent, stt, _make_tts_class()(), turn_detector="heuristic", user_idle=user_idle or None)


class TestUserIdleConfig:
    def test_defaults_and_rotation(self) -> None:
        cfg = UserIdleConfig(text=["Still there?", "Hello?"])
        assert cfg.timeout_secs == 8.0 and cfg.max_count == 2 and cfg.hangup_after_secs is None
        assert [cfg.line_for(i) for i in range(3)] == ["Still there?", "Hello?", "Still there?"]
        assert UserIdleConfig(text="One line").line_for(5) == "One line"
        assert UserIdleConfig(instructions="check in").line_for(0) is None  # → generated

    def test_rejects_configs_that_do_nothing(self) -> None:
        with pytest.raises(ValidationError):
            UserIdleConfig()  # no line, prompts enabled
        with pytest.raises(ValidationError):
            UserIdleConfig(max_count=0)  # nothing to say and nothing to hang up on
        UserIdleConfig(max_count=0, hangup_after_secs=30)  # hang-up only: fine
        with pytest.raises(ValidationError):
            UserIdleConfig(text="x", timeout_secs=0)
        with pytest.raises(ValidationError):
            VoiceConfig(user_idle={"text": "x", "unknown": 1})

    def test_rides_voice_config(self) -> None:
        cfg = VoiceConfig(user_idle={"text": "Are you still there?", "timeout_secs": 6, "hangup_after_secs": 30})
        assert isinstance(cfg.user_idle, UserIdleConfig)
        assert cfg.user_idle.hangup_after_secs == 30
        assert VoiceConfig().user_idle is None  # status quo: wait forever


class TestUserIdleBehaviour:
    async def test_prompts_after_silence_then_stops_at_max_count(self) -> None:
        stt = _OpenSTT()
        session = _session(stt, timeout_secs=0.2, text=["Still there?", "Hello?"], max_count=2)

        async def drive(_events: list[VoiceSessionEvent]) -> None:
            # Long enough for four timeouts; only two prompts may be spoken.
            await asyncio.sleep(1.4)

        events = await _run(session, stt, drive=drive)
        spoken = [e.text for e in events if isinstance(e, AgentTextDone)]
        assert spoken == ["Still there?", "Hello?"]
        assert [(e.role, e.text) for e in session.transcript] == [
            ("assistant", "Still there?"),
            ("assistant", "Hello?"),
        ]
        assert session.metrics == []  # prompts are not turns

    async def test_user_speech_resets_the_clock(self) -> None:
        stt = _OpenSTT()
        session = _session(stt, timeout_secs=0.3, text="Still there?", max_count=1)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await asyncio.sleep(0.2)
            await stt.inject(TranscriptEvent(type="committed", text="One sec."))
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.15)  # < timeout since the reply drained → no prompt yet

        events = await _run(session, stt, drive=drive)
        spoken = [e.text for e in events if isinstance(e, AgentTextDone)]
        assert spoken == [REPLY]

    async def test_hangup_after_secs_ends_the_session(self) -> None:
        stt = _OpenSTT()
        session = _session(stt, timeout_secs=0.15, text="Still there?", max_count=1, hangup_after_secs=0.6)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            # Do not finish the STT ourselves: the hang-up must end the session.
            while not any(isinstance(e, SessionEnded) for e in events):
                await asyncio.sleep(0.02)

        events = await _run(session, stt, drive=drive, timeout=3.0)
        spoken = [e.text for e in events if isinstance(e, AgentTextDone)]
        assert spoken == ["Still there?"]  # one prompt, then the line was dead
        assert session.closed
        assert isinstance(events[-1], SessionEnded)

    async def test_our_own_prompt_does_not_reset_the_hangup_clock(self) -> None:
        """Two prompts at 0.15s cadence would keep an agent-speech-reset clock
        alive forever; the hang-up counts from the user's last word."""
        stt = _OpenSTT()
        session = _session(stt, timeout_secs=0.15, text="Hello?", max_count=5, hangup_after_secs=0.7)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            while not any(isinstance(e, SessionEnded) for e in events):
                await asyncio.sleep(0.02)

        events = await _run(session, stt, drive=drive, timeout=3.0)
        assert session.closed
        assert len([e for e in events if isinstance(e, AgentTextDone)]) <= 4

    async def test_generated_prompt_uses_instructions(self) -> None:
        stt = _OpenSTT()
        agent = Agent(name="idle_test", model=TestModel(responses=[REPLY]), tools=[])
        session = VoiceSession(
            agent,
            stt,
            _make_tts_class()(),
            turn_detector="heuristic",
            user_idle={
                "timeout_secs": 0.2,
                "instructions": "Ask if they are still there.",
                "model": TestModel(responses=["Are you still with me?"]),
                "max_count": 1,
            },
        )

        async def drive(events: list[VoiceSessionEvent]) -> None:
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.02)

        events = await _run(session, stt, drive=drive)
        assert [e.text for e in events if isinstance(e, AgentTextDone)] == ["Are you still with me?"]

    async def test_greeting_and_idle_compose(self) -> None:
        """Opener first; the idle clock starts only once it has drained."""
        stt = _OpenSTT()
        agent = Agent(name="idle_test", model=TestModel(responses=[REPLY]), tools=[])
        session = VoiceSession(
            agent,
            stt,
            _make_tts_class()(),
            turn_detector="heuristic",
            greeting=GREETING,
            user_idle={"timeout_secs": 0.2, "text": "Still there?", "max_count": 1},
        )

        async def drive(events: list[VoiceSessionEvent]) -> None:
            while len([e for e in events if isinstance(e, AgentTextDone)]) < 2:
                await asyncio.sleep(0.02)

        events = await _run(session, stt, drive=drive)
        assert [e.text for e in events if isinstance(e, AgentTextDone)] == [GREETING, "Still there?"]
