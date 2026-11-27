"""Regressions from the review of the voice stack (timbal-ai/timbal#323), one per reported failure."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
from timbal import Agent
from timbal.core.llm.messages import prepare_messages_request
from timbal.core.test_model import TestModel
from timbal.core.voice_context import CURRENT_VOICE_SESSION
from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import TranscriptEvent
from timbal.voice.config import VoiceConfig
from timbal.voice.deepgram import DeepgramFluxSTT
from timbal.voice.speculative import AudioHold, SpeculativeAttempt, SpeculativeGate, transcripts_equivalent

from .test_deepgram import _FakeWs
from .test_predictive_eou import _stub_tts


async def _until(condition, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.01)


def _flux_session(monkeypatch: pytest.MonkeyPatch, ran: list[str], **config):
    """A Flux session with predictive end of turn and a ``cancel_booking`` tool the model calls on "booking"."""
    from timbal.server.voice import build_voice_session

    def cancel_booking(day: str) -> str:
        ran.append(day)
        return "cancelled"

    def reply(messages: list) -> Message | str:
        last = messages[-1]
        if last.role == "user" and "booking" in last.collect_text().lower():
            return Message(
                role="assistant",
                content=[ToolUseContent(id="call-1", name="cancel_booking", input={"day": "friday"})],
            )
        return "Okay."

    agent = Agent(name="voice_test", model=TestModel(handler=reply), tools=[cancel_booking])
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    session, _ = build_voice_session(agent, VoiceConfig(predict_during_speech=True, **config), {})
    _stub_tts(session)
    return session


def _amount_agent(got: list[str]) -> Agent:
    """An agent whose ``record`` tool gets the last word of the user's message as its argument."""

    def record(amount: str) -> str:
        got.append(amount)
        return "recorded"

    def reply(messages: list) -> Message | str:
        last = messages[-1]
        if last.role == "user":
            amount = last.collect_text().split()[-1]
            return Message(
                role="assistant",
                content=[ToolUseContent(id="call-1", name="record", input={"amount": amount})],
                stop_reason="tool_use",
            )
        return "Done."

    return Agent(name="voice_test", model=TestModel(handler=reply), tools=[record])


# Eager words, then a final transcript that differs only inside a number: its unit, or a separator that
# reads as thousands in one locale and as three decimals in another.
_CHANGED_QUANTITIES = [
    ("set discount to $12", "set discount to 12%"),
    ("set discount to 12", "set discount to 12%"),
    ("set discount to 12\u00a0€", "set discount to 12\u00a0%"),
    ("record 1250", "record 1.250"),
    ("record 1000", "record 1,000"),
]


def _protect(session) -> None:
    """A protected (``disable_during_tool``) tool runs: caller speech is deferred."""
    session._turn_tool_batch_inflight = 1
    session._turn_tool_batch_mode = "disable_during_tool"
    assert session._interruptions_suppressed()


class TestTempoSourceTimeout:
    async def test_a_source_timeout_propagates_instead_of_spinning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("av")
        import timbal.voice.tempo as tempo_module
        from timbal.voice.tempo import change_tempo

        # The failure mode never yields to the loop, so no asyncio timeout can stop it: count the flushes
        # instead, and fail the test after a handful (one is the most a single timeout may cause).
        real_blocks = tempo_module._blocks
        flushes = 0

        def counted_blocks(*args, **kwargs):
            nonlocal flushes
            if kwargs.get("force"):
                flushes += 1
                if flushes > 5:
                    raise AssertionError("the failed read is awaited again and again")
            return real_blocks(*args, **kwargs)

        monkeypatch.setattr(tempo_module, "_blocks", counted_blocks)

        async def source() -> AsyncIterator[bytes]:
            raise TimeoutError("provider read timed out")
            yield b""  # pragma: no cover

        async def consume() -> None:
            async for _ in change_tempo(source(), sample_rate=16_000, tempo=1.1):
                pass

        with pytest.raises(TimeoutError, match="provider read"):
            await asyncio.wait_for(consume(), 2.0)


class TestNumbersInTranscripts:
    @pytest.mark.parametrize(
        ("first", "second"),
        [("order 150 meters", "order 1.50 meters"), ("record value -20", "record value 20")],
    )
    def test_a_changed_number_is_a_different_transcript(self, first: str, second: str) -> None:
        from timbal.voice.session import _same_spoken_words

        assert not transcripts_equivalent(first, second)
        assert not _same_spoken_words(first, second)

    def test_punctuation_outside_numbers_still_matches(self) -> None:
        from timbal.voice.session import _same_spoken_words

        assert transcripts_equivalent("Order 150, please.", "order 150 please")
        assert _same_spoken_words("Record value 20.", "record value 20")

    async def test_a_sign_change_does_not_confirm_the_eager_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _flux_session(monkeypatch, [])
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="record value -20", confidence=0.9)
        )
        assert session._eager_unconfirmed == "record value -20"
        await session._handle_provider_commit("record value 20")
        assert [e.text for e in session.transcript if e.role == "user"] == ["record value 20"]


class _UnplayedTracker:
    """A client that has queued every byte and played none of it yet."""

    def __init__(self) -> None:
        self.emitted = 0
        self.interrupted = False

    def on_audio_emitted(self, num_bytes: int) -> None:
        self.emitted += num_bytes

    def on_interrupted(self) -> None:
        self.interrupted = True

    def on_playback_ack(self, _played_ms: float) -> None:
        return None

    ack_received = False
    played_bytes = 0

    @property
    def is_playing(self) -> bool:
        return self.emitted > 0 and not self.interrupted


class TestRepeatedRequestWhileProtected:
    """A confirmed request deferred behind a protected tool, then the same words again as an eager event."""

    async def test_turn_resumed_keeps_the_confirmed_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        session = _flux_session(monkeypatch, ran)
        _protect(session)
        await session._handle_provider_commit("Check the booking.")
        assert session._deferred_user_text == "Check the booking."
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Check the booking.", confidence=0.9)
        )
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        assert session._deferred_user_text == "Check the booking."
        session._turn_tool_batch_inflight = 0  # the protected tool ends: the request runs, its tool too
        await _until(lambda: ran)
        assert ran == ["friday"]

    async def test_the_confirmed_request_does_not_wait_on_the_repeat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        session = _flux_session(monkeypatch, ran)
        _protect(session)
        await session._handle_provider_commit("Check the booking.")
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Check the booking.", confidence=0.9)
        )
        session._turn_tool_batch_inflight = 0  # the deferred turn starts before TurnResumed
        await _until(lambda: ran)
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        assert ran == ["friday"]
        assert [e.text for e in session.transcript if e.role == "user"] == ["Check the booking."]

    async def test_two_eager_events_before_turn_resumed_are_both_taken_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = _flux_session(monkeypatch, [])
        _protect(session)
        await session._handle_provider_prediction(TranscriptEvent(type="eager", text="Check the", confidence=0.9))
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Check the booking", confidence=0.9)
        )
        assert session._deferred_user_text == "Check the booking"
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        assert session._deferred_user_text == ""

    async def test_eager_words_whose_hold_expired_behind_the_tool_are_taken_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[str] = []
        session = _flux_session(monkeypatch, ran, entity_patience=True, entity_patience_secs=0.1)
        _protect(session)
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="My booking code is 415 555", confidence=0.9)
        )
        assert session._held_user_text == "My booking code is 415 555"  # an open number: held
        await _until(lambda: session._deferred_user_text == "My booking code is 415 555")  # expired, deferred
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        assert session._deferred_user_text == ""
        session._turn_tool_batch_inflight = 0  # the protected tool ends
        await _until(lambda: session._deferred_turn_task is None or session._deferred_turn_task.done())
        assert session._current_turn_task is None
        assert ran == []

    async def test_a_refinement_that_starts_the_queued_turn_gives_the_request_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[str] = []
        session = _flux_session(monkeypatch, ran)
        session.hold_timeout_secs = 0.1
        _protect(session)
        await session._handle_provider_commit("Check the booking.")
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Check the booking today.", confidence=0.9)
        )
        assert session._deferred_user_text == "Check the booking today."
        session._turn_tool_batch_inflight = 0  # the queued turn starts on the refinement, before TurnResumed
        await _until(lambda: session._eager_turn_task is not None)
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        # The refinement is retracted; the confirmed request is held again and runs when the hold expires.
        await _until(lambda: ran)
        assert ran == ["friday"]
        assert [e.text for e in session.transcript if e.role == "user"][-1] == "Check the booking."

    async def test_new_words_after_the_request_are_still_taken_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _flux_session(monkeypatch, [])
        _protect(session)
        await session._handle_provider_commit("Check the booking.")
        await session._handle_provider_prediction(TranscriptEvent(type="eager", text="and cancel it", confidence=0.9))
        assert session._deferred_user_text == "Check the booking. and cancel it"
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        assert session._deferred_user_text == "Check the booking."


class TestRetractedReplyStillQueued:
    async def test_unplayed_audio_is_cut_and_the_reply_leaves_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _flux_session(monkeypatch, [])
        tracker = _UnplayedTracker()
        session.playback = tracker  # type: ignore[assignment]
        await session._handle_provider_prediction(TranscriptEvent(type="eager", text="Hello there.", confidence=0.9))
        turn = session._current_turn_task
        assert turn is not None
        await asyncio.wait_for(turn, timeout=30)  # generation is over; its 8 bytes wait unplayed
        assert tracker.emitted > 0 and tracker.is_playing
        assert [e.role for e in session.transcript] == ["user", "assistant"]
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        assert tracker.interrupted
        assert session.transcript == []


class TestEagerCancellation:
    async def test_turn_resumed_clears_an_unfinished_hold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        session = _flux_session(monkeypatch, ran, hold_unfinished_secs=0.1)
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Cancel my booking and", confidence=0.9)
        )
        assert session._held_user_text == "Cancel my booking and"
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        # The hold and its timer are gone: nothing can start a turn on the retracted words.
        assert session._held_user_text is None
        assert session._hold_task is None
        assert session._current_turn_task is None
        assert ran == []

    async def test_deferred_eager_words_do_not_interrupt_the_protected_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = _flux_session(monkeypatch, [])
        release = asyncio.Event()
        lookup = asyncio.create_task(release.wait())  # the confirmed turn whose protected tool runs
        session._current_turn_task = lookup
        _protect(session)
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="cancel my booking.", confidence=0.9)
        )
        assert session._deferred_user_text == "cancel my booking."
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        assert session._deferred_user_text == ""
        assert not lookup.done()
        release.set()
        await asyncio.wait_for(lookup, timeout=30)

    async def test_a_merged_eager_turn_gives_its_prefix_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        session = _flux_session(monkeypatch, ran)
        session.turn_detector.hold_unfinished_secs = 5.0
        await session._handle_provider_commit("I want to")
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="cancel my booking.", confidence=0.9)
        )
        assert [e.text for e in session.transcript if e.role == "user"] == ["I want to cancel my booking."]
        await _until(lambda: any(key.startswith("start:") for key in session._turn_tool_call_ids))
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        # The merged row is retracted and its prefix is held again for the final transcript.
        assert [e.text for e in session.transcript if e.role == "user"] == []
        assert session._held_user_text == "I want to"
        await session._handle_provider_commit("keep my room, thanks.")
        assert [e.text for e in session.transcript if e.role == "user"] == ["I want to keep my room, thanks."]
        assert ran == []


class TestSpeculativeDiscard:
    async def test_a_discard_without_a_live_attempt_leaves_the_confirmed_turn(self) -> None:
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        release = asyncio.Event()
        confirmed = asyncio.create_task(release.wait())
        session._current_turn_task = confirmed
        assert session._speculative_attempt is None
        await session._discard_speculative(reason="turn_resumed")
        assert not confirmed.done()
        release.set()
        await asyncio.wait_for(confirmed, timeout=30)

    async def test_a_mismatched_attempt_is_discarded_before_speech_is_deferred(self) -> None:
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        waiting = asyncio.create_task(asyncio.Event().wait())  # its protected tool waits on the gate
        attempt = SpeculativeAttempt(
            text="reserve a table",
            source="eager_eot",
            gate=SpeculativeGate(),
            hold=AudioHold(),
            prior_context=None,
        )
        attempt.task = waiting
        session._speculative_attempt = attempt
        session._current_turn_task = waiting
        _protect(session)
        await session._handle_committed("reserve a table for four")
        assert waiting.cancelled()
        assert session._speculative_attempt is None


class TestHeldAudioOrder:
    async def test_chunks_synthesized_during_the_flush_play_after_the_buffered_ones(self) -> None:
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        hold = AudioHold()
        session._audio_hold = hold
        record = ["", 0]
        await session._deliver_pcm(b"AA", record, turnless=False)  # before confirmation: held
        played: list[bytes] = []
        original = session._emit_pcm

        async def emit(chunk: bytes, rec: list, *, turnless: bool) -> None:
            played.append(chunk)
            if chunk == b"AA":
                await session._deliver_pcm(b"BB", rec, turnless=False)  # synthesis goes on meanwhile
            await original(chunk, rec, turnless=turnless)

        session._emit_pcm = emit  # type: ignore[method-assign]
        hold.release()
        await session._flush_audio_hold()
        assert played == [b"AA", b"BB"]
        assert hold.drained


class TestHangupGoodbye:
    async def test_the_turn_itself_is_not_goodbye_audio(self) -> None:
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(
            agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic", hangup_grace_secs=0.2
        )
        session._is_speaking = True  # the hangup tool runs inside the turn
        session._last_commit_at = time.monotonic()
        assert not session._hangup_audio_pending()
        started = time.monotonic()
        assert await asyncio.wait_for(session.commit_hangup(), 5.0) is True
        assert time.monotonic() - started < 2.0  # not the 20 s cap
        # Nothing had been said: the session spoke the farewell itself, so its window already passed.
        assert session._hangup_farewell_text
        assert session._end_call_grace_secs == 0
        session._stop_end_call()


class TestToolModesAndDispatch:
    async def test_each_call_keeps_its_own_interruption_modes(self) -> None:
        from timbal.core.tool import Tool
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        agent = Agent(
            name="shared",
            model=TestModel(responses=["ok"]),
            tools=[Tool(name="reserve", handler=lambda: "ok", interruption_mode="disable_during_tool")],
        )
        first = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        second = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        token = CURRENT_VOICE_SESSION.set(first)
        try:
            await agent._resolve_tools(0)
        finally:
            CURRENT_VOICE_SESSION.reset(token)
        agent.tools[0].interruption_mode = "allow"  # the second call's catalog differs
        token = CURRENT_VOICE_SESSION.set(second)
        try:
            await agent._resolve_tools(0)
        finally:
            CURRENT_VOICE_SESSION.reset(token)
        assert first._tool_interruption_mode("reserve") == "disable_during_tool"
        assert second._tool_interruption_modes == {}

    def test_a_dispatched_call_is_counted_once_under_its_target(self) -> None:
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        agent = Agent(name="voice_test", model=TestModel(responses=["ok"]), tools=[], tool_search="dispatch")
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        session._tool_interruption_modes = {"reserve": "disable_during_tool"}
        session._note_tool_call("call_tool", "c1", {"name": "reserve", "arguments": {}})
        session._note_tool_start("voice_test.reserve", "c1")
        assert session._turn_open_tools == {"reserve": 1}
        assert session._interruptions_suppressed()
        session._note_tool_output("voice_test.reserve", paused=False)
        assert session._turn_tool_batch_inflight == 0
        assert not session._interruptions_suppressed()

    async def test_an_agent_tool_named_call_tool_is_not_a_dispatch(self) -> None:
        """The application's own ``call_tool`` takes the name; the dispatcher gets another and is not called."""
        from timbal import Tool
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        ran: list[str] = []
        suppressed_after: list[bool] = []
        session: VoiceSession | None = None

        def call_tool(name: str) -> str:
            """The application's own tool."""
            ran.append(f"call_tool:{name}")
            return "ok"

        def reserve() -> str:
            """Reserve a table."""
            ran.append("reserve")
            return "reserved"

        def reply(messages: list) -> Message | str:
            if messages[-1].role == "user":
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="c1", name="call_tool", input={"name": "reserve"})],
                    stop_reason="tool_use",
                )
            assert session is not None
            suppressed_after.append(session._interruptions_suppressed())  # call_tool has returned
            return "Done."

        agent = Agent(
            name="voice_test",
            model=TestModel(handler=reply),
            tools=[call_tool, Tool(handler=reserve, defer_loading=True, interruption_mode="disable_during_tool")],
            tool_search="dispatch",
        )
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        await session._begin_user_turn("reserve it", replace_user_entry=False)
        await asyncio.wait_for(session._current_turn_task, timeout=30)
        assert ran == ["call_tool:reserve"]
        assert suppressed_after == [False]
        assert "reserve" not in session._turn_open_tools


    @pytest.mark.parametrize(
        "call_input",
        [
            {"name": "reserve", "arguments": "oops"},
            {"name": "reserve", "arguments": [1, 2]},
            {"name": "no_such_tool", "arguments": {}},
        ],
    )
    async def test_a_dispatch_the_agent_does_not_run_is_not_counted(self, call_input: dict) -> None:
        """The dispatcher answers a malformed call itself: its target never runs and must not stay counted."""
        from timbal import Tool
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        ran: list[str] = []
        suppressed_after: list[bool] = []
        open_after: list[int] = []
        session: VoiceSession | None = None

        def reserve() -> str:
            """Reserve a table."""
            ran.append("reserve")
            return "reserved"

        def reply(messages: list) -> Message | str:
            if messages[-1].role == "user":
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="c1", name="call_tool", input=call_input)],
                    stop_reason="tool_use",
                )
            assert session is not None
            suppressed_after.append(session._interruptions_suppressed())  # the dispatcher has answered
            open_after.append(session._turn_tool_batch_inflight)
            return "Done."

        agent = Agent(
            name="voice_test",
            model=TestModel(handler=reply),
            tools=[Tool(handler=reserve, defer_loading=True, interruption_mode="disable_during_tool")],
            tool_search="dispatch",
        )
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        await session._begin_user_turn("reserve it", replace_user_entry=False)
        await asyncio.wait_for(session._current_turn_task, timeout=30)
        assert ran == []
        assert suppressed_after == [False]
        assert open_after == [0]  # nothing left counted as running
        assert "reserve" not in session._turn_open_tools

    async def test_a_valid_dispatch_is_still_counted_under_its_target(self) -> None:
        from timbal import Tool
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        suppressed_during: list[bool] = []
        session: VoiceSession | None = None

        def reserve() -> str:
            """Reserve a table."""
            assert session is not None
            suppressed_during.append(session._interruptions_suppressed())
            return "reserved"

        def reply(messages: list) -> Message | str:
            if messages[-1].role == "user":
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="c1", name="call_tool", input={"name": "reserve", "arguments": {}})],
                    stop_reason="tool_use",
                )
            return "Done."

        agent = Agent(
            name="voice_test",
            model=TestModel(handler=reply),
            tools=[Tool(handler=reserve, defer_loading=True, interruption_mode="disable_during_tool")],
            tool_search="dispatch",
        )
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        await session._begin_user_turn("reserve it", replace_user_entry=False)
        await asyncio.wait_for(session._current_turn_task, timeout=30)
        assert suppressed_during == [True]
        assert session._turn_tool_batch_inflight == 0


class TestAnthropicSystemOverride:
    async def test_provider_params_system_wins(self) -> None:
        captured: dict = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)

            async def empty():
                return
                yield  # pragma: no cover

            return empty()

        client = MagicMock()
        client.messages.create = fake_create
        override = [{"type": "text", "text": "Provider-supplied rules."}]
        create_stream, _ = prepare_messages_request(
            client=client,
            model_name="claude-haiku-5-5",
            request_headers={},
            system_prompt="Agent prompt.",
            messages=[],
            tools=None,
            max_tokens=100,
            temperature=None,
            output_model=None,
            provider_params={"system": override},
        )
        with contextlib.suppress(Exception):
            async for _ in create_stream():
                pass
        assert captured["system"] == override


class TestFluxRejectedRefresh:
    async def test_a_rejected_refresh_keeps_the_accepted_patience(self) -> None:
        stt = DeepgramFluxSTT()
        stt._ws = _FakeWs()
        await stt.configure_turn(eot_threshold=1.0, eot_timeout_ms=2500, patient=True)
        await stt.configure_turn(eot_threshold=1.0, eot_timeout_ms=3000, patient=True)
        await stt._handle_message({"type": "ConfigureSuccess"})
        await stt._handle_message({"type": "ConfigureFailure", "description": "nope"})
        assert stt.patient_thresholds_live is True


class TestConfirmedSpeculativeAudio:
    async def test_confirmation_plays_the_held_audio_without_waiting_for_the_turn_loop(self) -> None:
        """A tool the confirmation releases (a hangup awaiting ``commit_hangup``) runs before the turn loop
        sees another event: the held goodbye must go out anyway, so the hangup does not wait out its cap."""
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        hold = AudioHold()
        attempt = SpeculativeAttempt(
            text="bye", source="eager_eot", gate=SpeculativeGate(), hold=hold, prior_context=None
        )
        session._speculative_attempt = attempt
        session._audio_hold = hold
        await session._deliver_pcm(b"GOODBYE", ["", 0], turnless=False)
        assert session._hangup_audio_pending()  # held, not played
        await session._confirm_speculative("bye", vad_endpointed=False)
        await _until(lambda: hold.drained)
        assert not hold.chunks


class TestThousandsSeparators:
    def test_a_separator_that_could_change_the_quantity_is_a_different_value(self) -> None:
        from timbal.voice.session import _same_spoken_words

        # "1,000" is a thousand in English and one in Spanish: never confirmed as "1000".
        for first, second in [("pay 1,000 euros", "pay 1000 euros"), ("record 1.250", "record 1250")]:
            assert not transcripts_equivalent(first, second)
            assert not _same_spoken_words(first, second)
        # Unambiguous forms are one value.
        for first, second in [
            ("pay 1,000,000", "pay 1000000"),
            ("pay 1.000.000", "pay 1000000"),
            ("pay 1,000.50", "pay 1000.50"),
            ("pay 1,50", "pay 1.50"),
        ]:
            assert transcripts_equivalent(first, second), (first, second)
            assert _same_spoken_words(first, second), (first, second)
        # A decimal point is not a thousands separator.
        assert not transcripts_equivalent("order 150", "order 1.50")

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            # A currency sign belongs to one number.
            ("buy 3 $5 tickets", "buy 3$ 5 tickets"),
            ("add 2$ 50 items", "add 2 $50 items"),
            # A unit after a no-break or narrow no-break space (French typography).
            ("set discount to 12\u00a0€", "set discount to 12\u00a0%"),
            ("set discount to 12\u202f€", "set discount to 12\u202f%"),
            ("set discount to 12\u00a0€", "set discount to 12"),
            # How two numbers combine.
            ("what is 5 + 3", "what is 5 - 3"),
            ("what is 5 * 3", "what is 5 / 3"),
            ("what is 5+3", "what is 5-3"),
        ],
    )
    def test_a_unit_or_operator_change_is_a_different_value(self, first: str, second: str) -> None:
        from timbal.voice.session import _same_spoken_words

        assert not transcripts_equivalent(first, second)
        assert not _same_spoken_words(first, second)

    @pytest.mark.parametrize(
        ("first", "second"),
        [("what is 5 - 3", "What is 5-3?"), ("what is 5 -3", "what is 5 - 3"), ("pay 12\u00a0€", "pay 12 €")],
    )
    def test_spacing_around_a_unit_or_operator_is_not_a_change(self, first: str, second: str) -> None:
        from timbal.voice.session import _same_spoken_words

        assert _same_spoken_words(first, second)

    def test_the_same_spoken_words_are_found_whatever_their_grouping(self) -> None:
        from timbal.voice.session import _contains_spoken_words, _without_spoken_words

        assert _contains_spoken_words("I want room 1,000 now", "room 1000")
        assert _without_spoken_words("I want room 1,000 now", "room 1000") == "I want now"

    @pytest.mark.parametrize(
        ("text", "words", "left"),
        [
            # The caller's numbers come back as they said them.
            ("The total is 2.500 euros and my card ends 4", "my card ends 4", "The total is 2.500 euros and"),
            ("Pay 1,000 euros. Cancel it", "cancel it", "Pay 1,000 euros."),
            ("nothing 1,000 here", "cancel", "nothing 1,000 here"),
            # A number goes whole, sign and unit included.
            ("-20", "-20", ""),
            ("Charge $12.", "charge $12", ""),
            ("Set it to 12% please and then", "and then", "Set it to 12% please"),
            ("Set it to 5 -3 degrees", "-3 degrees", "Set it to 5"),
        ],
    )
    def test_removing_words_keeps_the_rest_as_written(self, text: str, words: str, left: str) -> None:
        from timbal.voice.session import _without_spoken_words

        assert _without_spoken_words(text, words) == left


class TestChangedQuantityRegeneratesTheCall:
    @pytest.mark.parametrize(("eager", "final"), _CHANGED_QUANTITIES)
    async def test_flux_eager_turn(self, monkeypatch: pytest.MonkeyPatch, eager: str, final: str) -> None:
        from timbal.server.voice import build_voice_session

        got: list[str] = []
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(_amount_agent(got), VoiceConfig(predict_during_speech=True), {})
        _stub_tts(session)
        await session._handle_provider_prediction(TranscriptEvent(type="eager", text=eager, confidence=0.9))
        assert session._eager_unconfirmed == eager
        await session._handle_provider_commit(final)
        await _until(lambda: got)
        await asyncio.wait_for(session._current_turn_task, timeout=30)
        assert got == [final.split()[-1]]
        assert [e.text for e in session.transcript if e.role == "user"] == [final]

    @pytest.mark.parametrize(("eager", "final"), _CHANGED_QUANTITIES)
    async def test_speculative_attempt(self, eager: str, final: str) -> None:
        from timbal.voice import VoiceSession

        from .test_speculative import ScriptedSTT, ScriptedTTS, _Drive

        got: list[str] = []
        stt = ScriptedSTT()
        session = VoiceSession(
            agent=_amount_agent(got),
            stt=stt,
            tts=ScriptedTTS(),
            turn_detector="heuristic",
            speculative_generation=True,
        )
        async with _Drive(session, stt):
            await stt.inject(TranscriptEvent(type="eager", text=eager))
            await _until(lambda: session._speculative_attempt is not None)
            await stt.inject(TranscriptEvent(type="committed", text=final))
            await _until(lambda: got)
            await _until(lambda: session._current_turn_task is None or session._current_turn_task.done())
        assert got == [final.split()[-1]]


class TestRetractionAndHoldEnd:
    async def test_a_heard_reply_keeps_its_row_and_its_prefix_is_not_held_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Part of the reply was heard: it stays with its request, and the words before the eager ones are in
        that row already, so holding them again would repeat them in the next turn."""
        from timbal.voice import TranscriptEntry

        ran: list[str] = []
        session = _flux_session(monkeypatch, ran)
        session.turn_detector.hold_unfinished_secs = 5.0
        await session._handle_provider_commit("I want to")
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="cancel my booking.", confidence=0.9)
        )
        await _until(lambda: any(key.startswith("start:") for key in session._turn_tool_call_ids))
        # What the caller heard of the reply before its tool (the interrupt's truncation writes the same).
        session._transcript.append(TranscriptEntry(role="assistant", text="Let me check that."))
        await session._handle_provider_prediction(TranscriptEvent(type="resumed", text="", confidence=0.1))
        assert [e.text for e in session.transcript if e.role == "user"] == ["I want to cancel my booking."]
        assert session._held_user_text is None
        assert ran == []

    async def test_the_hold_stays_until_a_running_flush_has_drained(self) -> None:
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        hold = AudioHold()
        session._audio_hold = hold
        await session._deliver_pcm(b"AA", ["", 0], turnless=False)
        gate = asyncio.Event()
        played: list[bytes] = []
        original = session._emit_pcm

        async def slow_emit(chunk: bytes, rec: list, *, turnless: bool) -> None:
            played.append(chunk)
            await gate.wait()  # the confirmation's flush is still draining
            await original(chunk, rec, turnless=turnless)

        session._emit_pcm = slow_emit  # type: ignore[method-assign]
        hold.release()
        flush = asyncio.create_task(session._flush_audio_hold())
        await _until(lambda: hold.flushing)
        end = asyncio.create_task(session._settle_audio_hold_at_turn_end())
        await asyncio.sleep(0)
        assert session._audio_hold is hold  # not dropped while the flush drains
        await session._deliver_pcm(b"BB", ["", 0], turnless=False)  # synthesis goes on: it queues behind
        gate.set()
        await asyncio.wait_for(asyncio.gather(flush, end), 3.0)
        assert played == [b"AA", b"BB"]
        assert session._audio_hold is None

    async def test_settling_never_flushes_a_newer_attempts_hold(self) -> None:
        from timbal.voice import VoiceSession

        from .test_session import MockSTT, MockTTS

        agent = Agent(name="t", model=TestModel(responses=["ok"]), tools=[])
        session = VoiceSession(agent=agent, stt=MockSTT(script=[]), tts=MockTTS(), turn_detector="heuristic")
        old = AudioHold()
        session._audio_hold = old
        await session._deliver_pcm(b"OLD", ["", 0], turnless=False)
        gate = asyncio.Event()
        played: list[bytes] = []
        original = session._emit_pcm

        async def slow_emit(chunk: bytes, rec: list, *, turnless: bool) -> None:
            played.append(chunk)
            await gate.wait()
            await original(chunk, rec, turnless=turnless)

        session._emit_pcm = slow_emit  # type: ignore[method-assign]
        old.release()
        flush = asyncio.create_task(session._flush_audio_hold())  # the confirmation's flush, of ``old``
        await _until(lambda: old.flushing)
        end = asyncio.create_task(session._settle_audio_hold_at_turn_end())
        await asyncio.sleep(0)
        # A new speculative attempt starts meanwhile, with audio of its own held until it is confirmed.
        new = AudioHold()
        session._audio_hold = new
        await session._deliver_pcm(b"NEW", ["", 0], turnless=False)
        gate.set()
        await asyncio.wait_for(asyncio.gather(flush, end), 3.0)
        assert played == [b"OLD"]
        assert not new.released and new.chunks
        assert session._audio_hold is new
