"""Agent-speaks-first greeting: the opener a caller hears before saying anything.

Unit tests drive ``VoiceSession`` internals directly (TestModel + mock TTS);
config/merge tests cover the override channels a per-call opener arrives
through; integration tests go through ``/voice/ws`` like a real client.
"""

# ruff: noqa: ARG001, ARG002  (mock provider signatures are fixed by their callers)
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server import telephony as telephony_routes
from timbal.server import voice as voice_routes
from timbal.server.http import create_app
from timbal.voice import (
    AgentTextDone,
    AudioInputConfig,
    AudioOutput,
    GreetingConfig,
    SessionInterrupted,
    SessionStarted,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
    VoiceConfig,
    VoiceSession,
    VoiceSessionEvent,
)

from .test_voice_ws import _collect_ws_messages, _make_stt_class, _make_tts_class
from .voice_env import VOICE_ENV_KEYS

GREETING = "Hi, this is the clinic calling about your appointment."
SYSTEM_PROMPT = "You are the clinic's scheduling assistant."
REPLY = "Of course, let me pull that up."


class _OpenSTT(SpeechToText):
    """STT whose stream stays open until the test closes it.

    ``_make_stt_class`` ends its event stream on connect, which makes
    ``_process_stt_events`` call ``session.close()`` immediately — fine for a
    turn-driven test, fatal for an opener, which is exactly the thing that
    happens *before* any transcript.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()

    async def connect(self, config: AudioInputConfig) -> None:
        pass

    async def push_audio(self, chunk: bytes) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def inject(self, event: TranscriptEvent) -> None:
        await self._queue.put(event)

    async def finish(self) -> None:
        await self._queue.put(None)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            if item.text:
                yield item

    async def close(self) -> None:
        pass


class _PacedTTS(TextToSpeech):
    """Emits ``num_chunks`` of PCM, one every ``every_secs``.

    Chunks are half a second of audio each (16kHz mono PCM16) so synthesis
    outruns playback the way a real provider does — without that gap the
    playhead is always past everything emitted and no barge-in can land
    mid-word. ``marker`` makes each caller's audio identifiable in the stream.
    """

    def __init__(
        self,
        *,
        num_chunks: int = 4,
        every_secs: float = 0.05,
        chunk_bytes: int = 16_000,
        marker: Callable[[str], bytes] | None = None,
    ) -> None:
        self.num_chunks = num_chunks
        self.every_secs = every_secs
        self.chunk_bytes = chunk_bytes
        self.marker = marker or (lambda _text: b"\x00")

    async def connect(self, config) -> None:
        pass

    async def close(self) -> None:
        pass

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        for _ in range(self.num_chunks):
            await asyncio.sleep(self.every_secs)
            yield self.marker(text) * self.chunk_bytes


def _spy_on_system_prompts(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Record the ``system_prompt`` kwarg every agent run is invoked with.

    The override is a per-call kwarg (``Agent`` itself must not be mutated —
    it is shared across sessions), and ``TestModel.stream`` is handed only the
    messages, so the call boundary is where it is observable.
    """
    seen: list[str | None] = []
    original = Agent.__call__

    def spy(self: Agent, **kwargs: Any) -> Any:
        seen.append(kwargs.get("system_prompt"))
        return original(self, **kwargs)

    monkeypatch.setattr(Agent, "__call__", spy)
    return seen


def _make_session(
    *,
    greeting: dict | str | GreetingConfig | None = None,
    stt: SpeechToText | None = None,
    tts: TextToSpeech | None = None,
    model: Any = None,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> VoiceSession:
    agent = Agent(
        name="voice_test",
        model=model or TestModel(responses=[REPLY]),
        tools=[],
        system_prompt=system_prompt,
    )
    return VoiceSession(
        agent,
        stt or _make_stt_class()(),
        tts or _make_tts_class()(),
        turn_detector="heuristic",
        greeting=greeting,
    )


async def _run(
    session: VoiceSession,
    stt: _OpenSTT,
    *,
    drive: Callable[[list[VoiceSessionEvent]], Any] | None = None,
    settle: float = 0.0,
    timeout: float = 5.0,
) -> list[VoiceSessionEvent]:
    """Run ``session`` to completion, closing the STT once ``drive`` returns."""
    events: list[VoiceSessionEvent] = []

    async def _empty_audio() -> AsyncIterator[bytes]:
        return
        yield  # noqa: RET504 — make it an async generator

    async def _wait_for(predicate: Callable[[], bool]) -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    async def _driver() -> None:
        await _wait_for(lambda: any(isinstance(e, SessionStarted) for e in events))
        if drive is not None:
            await drive(events)
        if settle:
            await asyncio.sleep(settle)
        await stt.finish()

    async def _go() -> None:
        async with aclosing(session.run(_empty_audio())) as stream:
            driver = asyncio.create_task(_driver())
            async for event in stream:
                events.append(event)
            await driver

    await asyncio.wait_for(_go(), timeout=timeout)
    return events


def _spoken(events: list[VoiceSessionEvent]) -> list[str]:
    return [e.text for e in events if isinstance(e, AgentTextDone)]


def _drain_queue(session: VoiceSession) -> list[VoiceSessionEvent]:
    """Events a session emitted while being driven through its internals."""
    events: list[VoiceSessionEvent] = []
    while not session._event_queue.empty():
        events.append(session._event_queue.get_nowait())
    return events


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestGreetingConfig:
    def test_defaults(self) -> None:
        cfg = GreetingConfig(text=GREETING)
        assert cfg.text == GREETING
        assert cfg.instructions is None
        # Matches Vapi's firstMessageInterruptionsEnabled and LiveKit's on_enter.
        assert cfg.interruptible is False
        assert cfg.delay_ms == 0
        assert cfg.model is None

    def test_empty_block_fails(self) -> None:
        """``{}`` is a typo, not "no opener" — that is ``greeting=None``."""
        with pytest.raises(ValidationError, match="text"):
            GreetingConfig()

    def test_whitespace_only_text_fails(self) -> None:
        with pytest.raises(ValidationError):
            GreetingConfig(text="   ")

    def test_unknown_key_fails(self) -> None:
        with pytest.raises(ValidationError):
            GreetingConfig(text="hi", first_message="hi")

    def test_negative_delay_fails(self) -> None:
        with pytest.raises(ValidationError):
            GreetingConfig(text="hi", delay_ms=-1)

    def test_voice_config_default_is_off(self) -> None:
        assert VoiceConfig().greeting is None

    def test_bare_string_coerced_to_text(self) -> None:
        cfg = VoiceConfig(greeting=GREETING)
        assert isinstance(cfg.greeting, GreetingConfig)
        assert cfg.greeting.text == GREETING
        assert cfg.greeting.interruptible is False

    def test_empty_string_means_no_greeting(self) -> None:
        assert VoiceConfig(greeting="").greeting is None
        assert VoiceConfig(greeting="   ").greeting is None

    def test_dict_is_validated(self) -> None:
        cfg = VoiceConfig(greeting={"text": "hi", "interruptible": True, "delay_ms": 500})
        assert cfg.greeting.interruptible is True
        assert cfg.greeting.delay_ms == 500
        with pytest.raises(ValidationError):
            VoiceConfig(greeting={"delay_ms": 500})

    def test_session_coerces_bare_string(self) -> None:
        session = _make_session(greeting=GREETING)
        assert isinstance(session.greeting, GreetingConfig)
        assert session.greeting.text == GREETING

    def test_session_default_is_off(self) -> None:
        assert _make_session().greeting is None


@pytest.mark.usefixtures("clear_voice_env")
class TestGreetingEnvAndRunnableMerge:
    def test_unset_means_off(self) -> None:
        assert voice_routes.default_voice_config_from_env().greeting is None

    def test_env_sets_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_GREETING", GREETING)
        cfg = voice_routes.default_voice_config_from_env()
        assert cfg.greeting.text == GREETING

    def test_runnable_dict_deep_merges_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An agent tuning ``delay_ms`` must not drop the platform's opener."""
        monkeypatch.setenv("TIMBAL_VOICE_GREETING", GREETING)

        class R:
            voice_config = {"greeting": {"delay_ms": 400}}

        merged = voice_routes.merge_voice_config(R())
        assert merged.greeting.text == GREETING
        assert merged.greeting.delay_ms == 400

    def test_runnable_string_replaces_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_GREETING", "from env")

        class R:
            voice_config = {"greeting": GREETING}

        assert voice_routes.merge_voice_config(R()).greeting.text == GREETING

    def test_runnable_empty_string_disables_env_greeting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_GREETING", GREETING)

        class R:
            voice_config = {"greeting": ""}

        assert voice_routes.merge_voice_config(R()).greeting is None

    def test_voice_config_instance_merges_sparsely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_GREETING", GREETING)

        class R:
            voice_config = VoiceConfig(greeting={"interruptible": True, "text": "x"})

        merged = voice_routes.merge_voice_config(R())
        assert merged.greeting.interruptible is True
        assert merged.greeting.text == "x"

    def test_agent_only_greeting_without_env(self) -> None:
        class R:
            voice_config = {"greeting": {"instructions": "greet them by name"}}

        merged = voice_routes.merge_voice_config(R())
        assert merged.greeting.instructions == "greet them by name"
        assert merged.greeting.text is None

    def test_invalid_greeting_key_fails_fast(self) -> None:
        class R:
            voice_config = {"greeting": {"first_message": "nope"}}

        with pytest.raises(ValidationError, match="first_message"):
            voice_routes.merge_voice_config(R())


class TestClientGreetingOverrides:
    """Per-call openers: telephony ``<Parameter>``, LiveKit/RTC/WS hello."""

    def test_greeting_is_client_settable(self) -> None:
        assert "greeting" in voice_routes.CLIENT_SETTABLE_VOICE_FIELDS

    def test_telephony_param_allowlist_includes_greeting(self) -> None:
        assert "greeting" in telephony_routes._CONFIG_PARAM_KEYS

    def test_telephony_string_param_reaches_the_config(self) -> None:
        """End to end over the two hops a TeXML ``<Parameter>`` actually takes."""
        custom = {"greeting": GREETING, "recording": "/tmp/evil", "bogus": "x"}
        client_config = {
            k: v
            for k, v in custom.items()
            if k in telephony_routes._CONFIG_PARAM_KEYS and isinstance(v, str) and v
        }
        assert client_config == {"greeting": GREETING}

        out = voice_routes.merge_client_voice_overrides(VoiceConfig(), client_config)
        assert isinstance(out.greeting, GreetingConfig)
        assert out.greeting.text == GREETING
        assert out.greeting.interruptible is False

    def test_string_override_keeps_server_policy(self) -> None:
        """A bare string is a *text* override: the server keeps owning the rest."""
        base = VoiceConfig(greeting={"text": "server line", "interruptible": True, "delay_ms": 300})
        out = voice_routes.merge_client_voice_overrides(base, {"greeting": GREETING})
        assert out.greeting.text == GREETING
        assert out.greeting.interruptible is True
        assert out.greeting.delay_ms == 300

    def test_empty_string_override_disables(self) -> None:
        base = VoiceConfig(greeting=GREETING)
        assert voice_routes.merge_client_voice_overrides(base, {"greeting": ""}).greeting is None

    def test_partial_dict_override_keeps_server_text(self) -> None:
        base = VoiceConfig(greeting={"text": GREETING})
        out = voice_routes.merge_client_voice_overrides(base, {"greeting": {"delay_ms": 250}})
        assert out.greeting.text == GREETING
        assert out.greeting.delay_ms == 250

    def test_invalid_client_greeting_keeps_server_config(self) -> None:
        base = VoiceConfig(greeting={"text": GREETING})
        out = voice_routes.merge_client_voice_overrides(base, {"greeting": {"first_message": "no"}})
        assert out.greeting.text == GREETING

    def test_client_enables_when_server_has_none(self) -> None:
        out = voice_routes.merge_client_voice_overrides(
            VoiceConfig(), {"greeting": {"text": GREETING, "delay_ms": 100}}
        )
        assert out.greeting.text == GREETING
        assert out.greeting.delay_ms == 100

    def test_bogus_client_type_keeps_server_config(self) -> None:
        base = VoiceConfig(greeting={"text": GREETING})
        out = voice_routes.merge_client_voice_overrides(base, {"greeting": 42})
        assert out.greeting.text == GREETING


# ---------------------------------------------------------------------------
# Speaking it
# ---------------------------------------------------------------------------


class TestGreetingSpoken:
    """The opener reaches the caller with no user speech at all."""

    async def test_static_greeting_spoken_without_user_speech(self) -> None:
        stt = _OpenSTT()
        session = _make_session(greeting={"text": GREETING}, stt=stt)
        events = await _run(session, stt, settle=0.1)

        assert [(e.role, e.text) for e in session.transcript] == [("assistant", GREETING)]
        assert _spoken(events) == [GREETING]
        assert any(isinstance(e, AudioOutput) for e in events)
        # The opener is not a turn and must not fake one.
        assert session.metrics == []

    async def test_bare_string_greeting_spoken(self) -> None:
        stt = _OpenSTT()
        session = _make_session(greeting=GREETING, stt=stt)
        await _run(session, stt, settle=0.1)
        assert [e.text for e in session.transcript] == [GREETING]

    async def test_no_greeting_by_default_stays_silent(self) -> None:
        stt = _OpenSTT()
        session = _make_session(stt=stt)
        events = await _run(session, stt, settle=0.15)

        assert session.greeting is None
        assert session.transcript == []
        assert not any(isinstance(e, AudioOutput) for e in events)

    async def test_delay_ms_holds_the_line_before_speaking(self) -> None:
        stt = _OpenSTT()
        session = _make_session(greeting={"text": GREETING, "delay_ms": 300}, stt=stt)
        silent_at_100ms: list[bool] = []

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await asyncio.sleep(0.1)
            silent_at_100ms.append(not any(isinstance(e, AudioOutput) for e in events))
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.01)

        await _run(session, stt, drive=drive)

        assert silent_at_100ms == [True]
        assert [e.text for e in session.transcript] == [GREETING]

    async def test_user_speaking_first_inside_the_delay_skips_the_greeting(self) -> None:
        """The callee's "hello?" beats the opener out of the gate — speaking now
        would talk over their turn, and they no longer need prompting."""
        stt = _OpenSTT()
        session = _make_session(greeting={"text": GREETING, "delay_ms": 400}, stt=stt)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="Hello?"))
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.2)  # outlive the delay we skipped

        await _run(session, stt, drive=drive)

        assert [(e.role, e.text) for e in session.transcript] == [
            ("user", "Hello?"),
            ("assistant", REPLY),
        ]
        assert session._greeting_text == ""

    async def test_generated_greeting_uses_instructions_and_agent_prompt(self) -> None:
        stt = _OpenSTT()
        seen: list[str] = []

        def handler(messages):
            seen.append(messages[-1].collect_text())
            return '"Good morning, Bob — the clinic here."'

        session = _make_session(
            greeting={
                "instructions": "Greet the caller by name and say why you are calling.",
                "model": TestModel(handler=handler),
            },
            stt=stt,
        )
        await _run(session, stt, settle=0.1)

        # Surrounding quotes stripped, same as the filler generator.
        assert [e.text for e in session.transcript] == ["Good morning, Bob — the clinic here."]
        assert seen == ["Say your opening line."]
        prompt = session._greeting_agent.system_prompt
        assert SYSTEM_PROMPT in prompt  # knows who it is calling as
        assert "Greet the caller by name" in prompt
        assert "speak first" in prompt  # DEFAULT_GREETING_SYSTEM_PROMPT

    async def test_text_wins_over_instructions(self) -> None:
        stt = _OpenSTT()

        def handler(messages):
            raise AssertionError("the LLM path must not run when text is set")

        session = _make_session(
            greeting={
                "text": GREETING,
                "instructions": "improvise something",
                "model": TestModel(handler=handler),
            },
            stt=stt,
        )
        await _run(session, stt, settle=0.1)
        assert [e.text for e in session.transcript] == [GREETING]

    async def test_generation_failure_is_silent(self) -> None:
        """Silence at the top of the call is the status quo, not a SessionError."""
        stt = _OpenSTT()

        def _boom(messages):
            raise RuntimeError("generator down")

        session = _make_session(
            greeting={"instructions": "say hi", "model": TestModel(handler=_boom)}, stt=stt
        )
        events = await _run(session, stt, settle=0.15)

        assert session.transcript == []
        assert not any(isinstance(e, AudioOutput) for e in events)
        assert _spoken(events) == []


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestGreetingMemory:
    """The opener never enters agent memory; the first turn is told about it."""

    @staticmethod
    def _turns(stt: _OpenSTT, *texts: str) -> Callable:
        """Drive one turn per *text*, each after the previous reply is out."""

        async def drive(events: list[VoiceSessionEvent]) -> None:
            done = 0

            async def _wait(n: int) -> None:
                while sum(1 for e in events if isinstance(e, AgentTextDone)) < n:
                    await asyncio.sleep(0.01)

            for text in texts:
                done += 1
                await _wait(done)  # the opener, then each reply in turn
                await asyncio.sleep(0.02)
                await stt.inject(TranscriptEvent(type="committed", text=text))
            await _wait(done + 1)

        return drive

    async def test_first_turn_prompt_says_do_not_greet_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts = _spy_on_system_prompts(monkeypatch)
        stt = _OpenSTT()
        session = _make_session(greeting={"text": GREETING}, stt=stt)
        await _run(session, stt, drive=self._turns(stt, "Yes, hello."))

        [prompt] = prompts
        assert SYSTEM_PROMPT in prompt  # the agent's own prompt survives intact
        assert GREETING in prompt
        assert "Do not greet" in prompt
        # And the Agent itself was not mutated — it is shared across sessions.
        assert session.agent.system_prompt == SYSTEM_PROMPT

    async def test_greeting_never_enters_agent_memory(self) -> None:
        stt = _OpenSTT()
        session = _make_session(greeting={"text": GREETING}, stt=stt)
        await _run(session, stt, drive=self._turns(stt, "Yes, hello."))

        root = session._last_run_context.root_span()
        assert [m.role for m in root.memory] == ["user", "assistant"]
        assert GREETING not in " ".join(m.collect_text() for m in root.memory)

    async def test_transcript_records_the_opener_as_plain_assistant_text(self) -> None:
        """It is what reaches the recording manifest and the sessions ingest."""
        stt = _OpenSTT()
        session = _make_session(greeting={"text": GREETING}, stt=stt)
        await _run(session, stt, drive=self._turns(stt, "Yes, hello."))

        assert [(e.role, e.text) for e in session.transcript] == [
            ("assistant", GREETING),
            ("user", "Yes, hello."),
            ("assistant", REPLY),
        ]
        # Not a filler: fillers are dimmed / excluded from reply text downstream.
        assert all(e.filler is False for e in session.transcript)

    async def test_no_greeting_means_no_prompt_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing behaviour unchanged: the agent resolves its own prompt."""
        prompts = _spy_on_system_prompts(monkeypatch)
        stt = _OpenSTT()
        session = _make_session(stt=stt)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            await stt.inject(TranscriptEvent(type="committed", text="Yes, hello."))
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.01)

        await _run(session, stt, drive=drive)

        assert session.greeting is None
        assert prompts == [None]  # no override at all — Agent.handler resolves it

    async def test_second_turn_has_no_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Turn two chains through turn one's memory, which carries the note's
        effect — re-stating it every turn would be dead weight in the prompt."""
        prompts = _spy_on_system_prompts(monkeypatch)
        stt = _OpenSTT()
        session = _make_session(
            greeting={"text": GREETING},
            stt=stt,
            model=TestModel(responses=[REPLY, "It was at four."]),
        )
        await _run(session, stt, drive=self._turns(stt, "Yes, hello.", "What time was it again?"))

        first, second = prompts
        assert GREETING in first
        assert second is None

    async def test_prompt_note_survives_a_callable_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts = _spy_on_system_prompts(monkeypatch)
        stt = _OpenSTT()

        def dynamic_prompt() -> str:
            return "RESOLVED PROMPT"

        agent = Agent(
            name="voice_test",
            model=TestModel(responses=[REPLY]),
            tools=[],
            system_prompt=dynamic_prompt,
        )
        session = VoiceSession(
            agent, stt, _make_tts_class()(), turn_detector="heuristic", greeting=GREETING
        )
        await _run(session, stt, drive=self._turns(stt, "Yes, hello."))

        [prompt] = prompts
        assert prompt.startswith("RESOLVED PROMPT")
        assert GREETING in prompt


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------


class TestGreetingInterruptions:
    """``interruptible`` decides whether a barge-in may chop the opener."""

    async def test_non_interruptible_greeting_survives_barge_in(self) -> None:
        tts = _PacedTTS(num_chunks=4)
        session = _make_session(greeting={"text": GREETING}, tts=tts)
        speak = session._claim_greeting(GREETING)
        await asyncio.sleep(0.12)  # ~2 of 4 chunks out, playback far behind

        await session.interrupt()
        assert session._greeting_speaking is True
        assert session._cancel_turn.is_set() is False  # nothing was torn down
        await speak

        assert session._greeting_record[1] == 4 * 16_000  # every chunk went out
        assert [e.text for e in session.transcript] == [GREETING]

    async def test_interruptible_greeting_is_cut_to_what_was_heard(self) -> None:
        tts = _PacedTTS(num_chunks=4)
        session = _make_session(greeting={"text": GREETING, "interruptible": True}, tts=tts)
        speak = session._claim_greeting(GREETING)
        await asyncio.sleep(0.12)

        await session.interrupt()
        await asyncio.gather(speak, return_exceptions=True)

        assert session._greeting_speaking is False
        assert speak.cancelled()
        # The entry the caller heard a prefix of is rewritten to that prefix.
        [entry] = session.transcript
        assert entry.role == "assistant"
        assert entry.text != GREETING
        assert GREETING.startswith(entry.text)
        # And the client is told to rewrite the opener's bubble to match, then
        # to seal it — an open bubble would hang forever.
        events = _drain_queue(session)
        [interrupted] = [e for e in events if isinstance(e, SessionInterrupted)]
        assert interrupted.heard_text == entry.text
        assert _spoken(events) == [entry.text]

    async def test_completed_but_still_playing_greeting_is_rewritten(self) -> None:
        """The common barge-in shape: TTS finished, audio still queued."""
        tts = _PacedTTS(num_chunks=2, every_secs=0.0)
        session = _make_session(greeting={"text": GREETING, "interruptible": True}, tts=tts)
        await session._claim_greeting(GREETING)  # synthesis is instant here
        assert session.transcript[-1].text == GREETING  # recorded in full...

        await asyncio.sleep(0.2)  # ...of which ~0.2s of 1.0s has played
        await session.interrupt()

        [entry] = session.transcript  # ...then rewritten to the heard prefix
        assert entry.text != GREETING
        assert GREETING.startswith(entry.text)

    async def test_unheard_greeting_is_removed_from_the_transcript(self) -> None:
        """Interrupted before a single word landed → it was never said."""
        tts = _PacedTTS(num_chunks=2, every_secs=0.0)
        session = _make_session(greeting={"text": GREETING, "interruptible": True}, tts=tts)
        await session._claim_greeting(GREETING)

        await session.interrupt()

        assert session.transcript == []
        [interrupted] = [e for e in _drain_queue(session) if isinstance(e, SessionInterrupted)]
        assert interrupted.heard_text == ""  # client drops the bubble entirely

    async def test_close_cuts_a_non_interruptible_greeting(self) -> None:
        """A caller who hung up must not hold the box to the last syllable."""
        tts = _PacedTTS(num_chunks=40, every_secs=0.05)
        session = _make_session(greeting={"text": GREETING}, tts=tts)
        session._greeting_task = asyncio.create_task(session._greeting_flow())
        await asyncio.sleep(0.12)
        assert session._greeting_speaking is True
        speak = session._tts_tail

        await asyncio.wait_for(session.close(), timeout=1.0)
        await asyncio.gather(session._greeting_task, speak, return_exceptions=True)

        assert session._greeting_task.cancelled()
        assert speak.cancelled()
        # close() leaves the committed transcript alone (truncate_completed=False),
        # so what the caller did hear is not rewritten away on hangup.
        assert [e.text for e in session.transcript] == [GREETING]

    async def test_deferred_barge_in_reply_queues_behind_the_opener(self) -> None:
        """A barge-in during a non-interruptible opener is deferred, not dropped:
        the turn still runs, its audio just follows the greeting's."""
        stt = _OpenSTT()
        # 0.1s of audio per chunk, emitted every 0.05s: playback is genuinely
        # mid-opener when the commit lands, so this is a real barge-in.
        tts = _PacedTTS(
            num_chunks=3,
            every_secs=0.05,
            chunk_bytes=3_200,
            marker=lambda text: b"G" if text == GREETING else b"R",
        )
        session = _make_session(greeting={"text": GREETING}, stt=stt, tts=tts)

        async def drive(events: list[VoiceSessionEvent]) -> None:
            while not any(isinstance(e, AudioOutput) for e in events):
                await asyncio.sleep(0.005)
            await stt.inject(TranscriptEvent(type="committed", text="Yes, hello."))
            while sum(1 for e in events if isinstance(e, AgentTextDone)) < 2:
                await asyncio.sleep(0.01)

        events = await _run(session, stt, drive=drive)

        markers = [e.data[:1] for e in events if isinstance(e, AudioOutput)]
        assert markers == [b"G"] * 3 + [b"R"] * 3
        assert _spoken(events) == [GREETING, REPLY]
        assert [(e.role, e.text) for e in session.transcript] == [
            ("assistant", GREETING),
            ("user", "Yes, hello."),
            ("assistant", REPLY),
        ]

    async def test_pending_greeting_audio_seeds_a_textless_segment(self) -> None:
        """Truncation accounting: opener bytes still queued when the first turn
        starts must not be mapped onto the reply's words."""
        tts = _PacedTTS(num_chunks=2, every_secs=0.0)
        session = _make_session(greeting={"text": GREETING}, tts=tts)
        await session._claim_greeting(GREETING)

        pending = session._greeting_pending_bytes()
        assert pending > 0

        await session._run_turn("Yes, hello.")
        first, *rest = [(t, b) for t, b in session._turn_tts_segment_records]
        assert first[0] == ""  # text-less: those bytes carry no reply words
        assert first[1] == pytest.approx(pending, rel=0.2)
        assert [t for t, _ in rest] == [REPLY]

    async def test_no_pending_bytes_once_the_opener_has_played_out(self) -> None:
        tts = _PacedTTS(num_chunks=1, every_secs=0.0, chunk_bytes=32)
        session = _make_session(greeting={"text": GREETING}, tts=tts)
        await session._claim_greeting(GREETING)
        await asyncio.sleep(0.05)  # 32 bytes = 1ms of audio

        assert session._greeting_pending_bytes() == 0
        await session._run_turn("Yes, hello.")
        assert [t for t, _ in session._turn_tts_segment_records] == [REPLY]


# ---------------------------------------------------------------------------
# Over the wire
# ---------------------------------------------------------------------------


class TestGreetingOverWs:
    """The wire format a real client sees, through ``build_voice_session``."""

    def _run_ws(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        module_body: str,
        hello: dict,
    ) -> list[dict]:
        mod = tmp_path / "voice_agent.py"
        mod.write_text(module_body)
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{mod.resolve()}::agent")
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)

        stt_cls = _make_stt_class([TranscriptEvent(type="committed", text="Yes, hello.")])
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", stt_cls)
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({"turn_detector": "heuristic", **hello})
                return _collect_ws_messages(ws)

    _AGENT = (
        "from timbal import Agent\n"
        "from timbal.core.test_model import TestModel\n"
        f"agent = Agent(name='voice_test', model=TestModel(responses=[{REPLY!r}]), tools=[])\n"
    )

    def test_agent_voice_config_greeting_is_spoken_first(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        body = self._AGENT + f"agent.voice_config = {{'greeting': {GREETING!r}}}\n"
        messages = self._run_ws(monkeypatch, tmp_path, module_body=body, hello={})

        assert [m["text"] for m in messages if m["type"] == "agent_text_done"] == [GREETING, REPLY]
        transcript = next(m for m in messages if m["type"] == "session_transcript")["entries"]
        assert [e["text"] for e in transcript] == [GREETING, "Yes, hello.", REPLY]

    def test_client_hello_string_override_is_spoken(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Same shape a telephony ``<Parameter name="greeting">`` arrives in."""
        messages = self._run_ws(
            monkeypatch, tmp_path, module_body=self._AGENT, hello={"greeting": GREETING}
        )
        assert [m["text"] for m in messages if m["type"] == "agent_text_done"] == [GREETING, REPLY]

    def test_client_can_switch_off_a_server_greeting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        body = self._AGENT + f"agent.voice_config = {{'greeting': {GREETING!r}}}\n"
        messages = self._run_ws(monkeypatch, tmp_path, module_body=body, hello={"greeting": ""})
        assert [m["text"] for m in messages if m["type"] == "agent_text_done"] == [REPLY]

    def test_no_greeting_configured_stays_reactive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        messages = self._run_ws(monkeypatch, tmp_path, module_body=self._AGENT, hello={})
        assert [m["text"] for m in messages if m["type"] == "agent_text_done"] == [REPLY]
