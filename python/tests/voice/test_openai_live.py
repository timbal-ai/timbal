"""GPT-Live (full-duplex) session: wire events → VoiceSessionEvents, delegation → Agent → commentary.

Driven by a scripted FakeLiveTransport replaying the exact event shapes observed
against ``wss://api.openai.com/v1/live/sessions`` (2026-09-14). No network.
"""

import asyncio
import base64
import re

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice import (
    AgentStatus,
    AgentTextDelta,
    AgentTextDone,
    AudioOutput,
    DelegationCreated,
    DelegationResult,
    LiveSession,
    LiveSessionSummary,
    LiveTransport,
    SessionEnded,
    SessionError,
    SessionStarted,
    TranscriptCommitted,
    TranscriptEntry,
    TranscriptPartial,
    TurnMetricsEvent,
)
from timbal.voice.openai_live import OpenAILiveClient, split_for_append

SILENCE = b"\x00" * 4800
SPEECH = b"\x01\x02" * 2400


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def audio(data: bytes) -> dict:
    return {"type": "session.output_audio.delta", "delta": b64(data)}


def user(delta: str, start: int, end: int) -> dict:
    return {"type": "session.input_transcript.delta", "delta": delta, "start_ms": start, "end_ms": end, "event_id": "e"}


def agent_tx(delta: str, start: int, end: int) -> dict:
    return {
        "type": "session.output_transcript.delta",
        "delta": delta,
        "start_ms": start,
        "end_ms": end,
        "event_id": "e",
    }


def delegation(did: str, offset: int) -> dict:
    return {
        "type": "session.delegation.created",
        "offset_ms": offset,
        "delegation": {"id": did, "type": "delegation", "target": "client"},
        "event_id": "e",
    }


STARTED = {
    "type": "session.started",
    "event_id": "e",
    "client_event_id": "session_start",
    "session": {"id": "live_test", "model": "gpt-live-1", "status": "active"},
}
CLOSED = {"type": "session.closed", "event_id": "e", "reason": "close_requested", "usage": {"seconds": 20.0}}


class FakeLiveTransport(LiveTransport):
    """Replays a script. Markers: ``"<wait:commentary>"`` blocks until a commentary
    append arrived, ``"<wait:append>"`` until any commentary/thinking append,
    ``"<hold>"`` until :meth:`release`, a float sleeps. Ending a script without
    ``CLOSED`` simulates a socket drop; ``reconnect()`` moves on to the next
    script in ``more_scripts``."""

    def __init__(self, script: list, *, end_after: bool = True, more_scripts: list[list] | None = None):
        self.scripts = [script, *(more_scripts or [])]
        self.gen = 0
        self.end_after = end_after
        self.sent: list[dict] = []
        self.audio: list[bytes] = []
        self.connected = False
        self.connect_count = 0
        self.closed = False
        self.reconnect_inputs: list[list[dict]] = []
        self.fail_reconnects = 0
        self._finalized = False
        self._commentary = asyncio.Event()
        self._append = asyncio.Event()
        self._release = asyncio.Event()

    async def connect(self):
        self.connect_count += 1
        self.connected = True
        return STARTED

    async def reconnect(self, *, input=None):
        if self.fail_reconnects > 0:
            self.fail_reconnects -= 1
            raise ConnectionError("still down")
        self.reconnect_inputs.append(list(input or []))
        self.gen += 1
        if self.gen >= len(self.scripts):
            raise ConnectionError("no more scripts")
        return {"type": "session.started", "session": {"id": f"live_test_{self.gen}"}}

    @property
    def finalized(self) -> bool:
        return self._finalized

    async def send_audio(self, chunk: bytes) -> None:
        self.audio.append(chunk)

    async def send(self, event: dict) -> None:
        self.sent.append(event)
        if event["type"] == "session.commentary.append":
            self._commentary.set()
        if event["type"] in ("session.commentary.append", "session.thinking.append"):
            self._append.set()

    async def events(self):
        for ev in self.scripts[self.gen]:
            if ev == "<wait:commentary>":
                await asyncio.wait_for(self._commentary.wait(), 5)
                continue
            if ev == "<wait:append>":
                await asyncio.wait_for(self._append.wait(), 5)
                continue
            if ev == "<hold>":
                await self._release.wait()
                continue
            if isinstance(ev, float):
                await asyncio.sleep(ev)
                continue
            if ev.get("type") == "session.closed":
                self._finalized = True
            yield ev
        if self.end_after:
            return
        await asyncio.sleep(3600)

    def release(self) -> None:
        self._release.set()

    async def close(self, timeout: float = 15.0):  # noqa: ARG002
        self.closed = True
        self.close_reason = "close_requested"
        return {"seconds": 20.0}

    def commands(self, typ: str) -> list[dict]:
        return [e for e in self.sent if e["type"] == typ]


async def _mic():
    for _ in range(3):
        yield SILENCE
    await asyncio.sleep(3600)


async def _collect(session: LiveSession, timeout: float = 5.0):
    events = []

    async def go():
        async for ev in session.run(_mic()):
            events.append(ev)

    await asyncio.wait_for(go(), timeout)
    return events


def _types(events):
    return [e.type for e in events]


# ---------------------------------------------------------------------------


class TestAudioAndTranscripts:
    async def test_continuous_audio_forwarded_and_silence_optionally_dropped(self):
        script = [audio(SILENCE), audio(SPEECH), audio(SILENCE), CLOSED]
        t = FakeLiveTransport(script)
        events = await _collect(LiveSession(t, agent=None, on_delegation=None))
        outs = [e for e in events if isinstance(e, AudioOutput)]
        assert [len(o.data) for o in outs] == [4800, 4800, 4800]

        t2 = FakeLiveTransport(script)
        s2 = LiveSession(t2, drop_silence=True)
        events = await _collect(s2)
        outs = [e for e in events if isinstance(e, AudioOutput)]
        assert len(outs) == 1 and outs[0].data == SPEECH
        assert s2.speech_output_bytes == 4800
        assert events[0].type == "session_started" and events[-1].type == "session_ended"

    async def test_mic_audio_forwarded_to_transport(self):
        t = FakeLiveTransport([0.05, CLOSED])
        await _collect(LiveSession(t))
        assert t.audio and all(c == SILENCE for c in t.audio)
        assert t.connected and t.closed

    async def test_prepare_connects_once_and_run_does_not_reconnect(self):
        t = FakeLiveTransport([CLOSED])
        s = LiveSession(t)
        assert s.closed is False
        await s.prepare()
        assert t.connect_count == 1 and s.live_session_id == "live_test"
        assert re.fullmatch(r"[0-9a-f]{32}", s.session_id)  # ours, minted before connect
        events = await _collect(s)
        assert t.connect_count == 1
        assert events[0].type == "session_started" and events[-1].type == "session_ended"
        await s.close()
        assert s.closed is True

    async def test_client_close_takes_reason_from_transport(self):
        # Client-initiated close: the dispatcher never sees session.closed.
        t = FakeLiveTransport(["<hold>"], end_after=False)
        s = LiveSession(t)

        async def drive():
            async for ev in s.run(_mic()):
                if isinstance(ev, SessionStarted):
                    await s.close()

        await asyncio.wait_for(drive(), 5)
        assert s.close_reason == "close_requested" and s.usage_seconds == 20.0

    async def test_user_fragments_group_into_a_row(self):
        script = [user(" Hi", 1600, 1800), user(" there", 1800, 2000), user(".", 2000, 2200), CLOSED]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=1000)
        events = await _collect(s)
        partials = [e.text for e in events if isinstance(e, TranscriptPartial)]
        assert partials == ["Hi", "Hi there", "Hi there."]
        committed = [e.text for e in events if isinstance(e, TranscriptCommitted)]
        assert committed == ["Hi there."]
        assert [(e.role, e.text) for e in s.transcript] == [("user", "Hi there.")]

    async def test_timeline_gap_splits_rows(self):
        script = [user(" One", 1000, 1200), user(" two", 5000, 5200), CLOSED]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=1000)
        events = await _collect(s)
        assert [e.text for e in events if isinstance(e, TranscriptCommitted)] == ["One", "two"]

    async def test_wallclock_gap_closes_row_before_session_end(self):
        # Fragment, then wall-clock silence longer than row_gap_ms: the timer
        # finalizes the row so the UI gets a bubble without waiting for the
        # session to end. The next fragment is contiguous on the session clock
        # (100 ms), i.e. late delivery, not a new turn — it extends that row.
        script = [user(" First", 1000, 1200), 0.25, user(" Second", 1300, 1500), CLOSED]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=100, row_close_slack_ms=50)
        events = await _collect(s)
        committed = [(e.text, e.replace) for e in events if isinstance(e, TranscriptCommitted)]
        assert committed == [("First", False), ("First Second", True)]
        assert [e.text for e in s.transcript] == ["First Second"]

    async def test_delivery_jitter_does_not_split_a_row(self):
        # 400 ms on the session clock, but the second fragment arrives 0.9 s late (wall clock).
        # Timeline decides splits; the wall-clock timer (gap + slack) only finalizes.
        script = [user(" I live in", 1000, 1400), 0.9, user(" Barcelona", 1800, 2200), CLOSED]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=800, row_close_slack_ms=700)
        events = await _collect(s)
        assert [e.text for e in events if isinstance(e, TranscriptCommitted)] == ["I live in Barcelona"]

    async def test_assistant_row_emits_delta_done_and_timeline_metrics(self):
        script = [
            user(" Weather", 1000, 1200),
            user(" please", 1200, 1400),
            audio(SILENCE),
            agent_tx(" Checking", 2200, 2400),
            audio(SPEECH),
            agent_tx(".", 2400, 2600),
            CLOSED,
        ]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=1000)
        events = await _collect(s)
        assert [e.text for e in events if isinstance(e, AgentTextDelta)] == [" Checking", "."]
        done = [e for e in events if isinstance(e, AgentTextDone)]
        assert len(done) == 1 and done[0].text == "Checking." and done[0].run_id is None
        m = [e.metrics for e in events if isinstance(e, TurnMetricsEvent)]
        assert len(m) == 1
        # Session-timeline: assistant start 2200 − user end 1400.
        assert m[0].eou_to_first_audio_ms == 800.0
        assert m[0].turn_total_ms == 400.0
        assert m[0].audio_bytes == 4800  # only the non-silent frame during the row
        assert m[0].user_text_chars == len("Weather please")
        assert [(e.role, e.text) for e in s.transcript] == [("user", "Weather please"), ("assistant", "Checking.")]

    async def test_reply_start_commits_user_row_first(self):
        # Observed in the playground: the user row's wall-clock timer fired *after*
        # the reply's first fragment, so the reply came out before the ask and was
        # labelled an opener. The assistant starting after the user's last speech
        # is the turn boundary — commit, then delta.
        script = [user(" How are", 1000, 1200), user(" you?", 1200, 1500), agent_tx(" Great", 1900, 2100), CLOSED]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=1000, row_close_slack_ms=5000)
        events = await _collect(s)
        kinds = [e.type for e in events if e.type in ("transcript_committed", "agent_text_delta")]
        assert kinds == ["transcript_committed", "agent_text_delta"]

    async def test_late_user_fragment_extends_committed_row(self):
        # Tail of the user row lands after the reply already committed it.
        script = [
            user(" I live in", 1000, 1400),
            agent_tx(" Nice", 2000, 2200),  # commits "I live in"
            user(" Barcelona", 1400, 1800),  # contiguous with the closed row
            0.3,
            CLOSED,
        ]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=800, row_close_slack_ms=100)
        events = await _collect(s)
        committed = [(e.text, e.replace) for e in events if isinstance(e, TranscriptCommitted)]
        assert committed == [("I live in", False), ("I live in Barcelona", True)]
        assert [(e.role, e.text) for e in s.transcript if e.role == "user"] == [("user", "I live in Barcelona")]

    async def test_late_assistant_fragment_continues_done_row(self):
        # "How are you" ... [row closed by the timer] ... "?" — one bubble, not two.
        script = [
            agent_tx(" How are you", 1000, 1600),
            0.3,  # timer closes the row
            agent_tx("?", 1600, 1700),
            0.3,
            CLOSED,
        ]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=100, row_close_slack_ms=100)
        events = await _collect(s)
        deltas = [(e.text, e.continues) for e in events if isinstance(e, AgentTextDelta)]
        assert deltas == [(" How are you", False), ("?", True)]
        assert [e.text for e in events if isinstance(e, AgentTextDone)] == ["How are you", "How are you?"]
        assert len([e for e in events if isinstance(e, TurnMetricsEvent)]) == 1
        assert [(e.role, e.text) for e in s.transcript] == [("assistant", "How are you?")]
        # Seed for a replacement session carries the merged row once.
        assert [i["content"][0]["text"] for i in s.history_items()] == ["How are you?"]

    async def test_delegation_view_tidies_fragment_punctuation(self):
        # Real fragments: " Um", " .", "I tried..." → prompt line "Um. I tried" not "Um .I tried".
        script = [user(" Um", 1000, 1200), user(" .", 1200, 1250), user("I tried", 1300, 1700), CLOSED]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=800)
        await _collect(s)
        assert [f.delta for f in s.transcript_since_last_delegation()] == ["Um. I tried"]

    async def test_new_row_drops_leading_punctuation(self):
        script = [user(" Fine", 1000, 1200), user(". Actually", 3000, 3400), CLOSED]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=800)
        events = await _collect(s)
        assert [e.text for e in events if isinstance(e, TranscriptCommitted)] == ["Fine", "Actually"]

    async def test_overlapping_speakers_keep_independent_rows(self):
        script = [
            agent_tx(" The", 1000, 1200),
            user(" wait", 1100, 1300),
            agent_tx(" answer", 1200, 1400),
            user(" stop", 1300, 1500),
            CLOSED,
        ]
        s = LiveSession(FakeLiveTransport(script), row_gap_ms=1000)
        await _collect(s)
        roles = [(e.role, e.text) for e in s.transcript]
        assert ("assistant", "The answer") in roles and ("user", "wait stop") in roles

    async def test_error_event_is_nonfatal(self):
        script = [
            {"type": "error", "error": {"type": "invalid_request", "message": "bad append", "client_event_id": "x"}},
            user(" ok", 1000, 1200),
            CLOSED,
        ]
        events = await _collect(LiveSession(FakeLiveTransport(script)))
        errs = [e for e in events if isinstance(e, SessionError)]
        assert len(errs) == 1 and "bad append" in errs[0].message
        assert any(isinstance(e, TranscriptCommitted) for e in events)
        assert isinstance(events[-1], SessionEnded)

    async def test_usage_tracked_from_snapshots(self):
        script = [
            {"type": "session.usage.updated", "usage": {"seconds": 13.0}, "context_window": {"usage_ratio": 0.01}},
            CLOSED,
        ]
        s = LiveSession(FakeLiveTransport(script))
        await _collect(s)
        assert s.usage_seconds == 20.0 and s.close_reason == "close_requested"
        summary = LiveSessionSummary.from_session(s)
        assert summary.est_usd == round(20 / 60 * 0.05, 4)
        assert summary.live_session_id == "live_test" and summary.session_id == s.session_id
        assert summary.started_at is not None and summary.ended_at >= summary.started_at


class TestDelegation:
    async def test_agent_answers_delegation_via_commentary(self):
        script = [
            user(" Hi", 1600, 1800),
            user(" there", 1800, 2000),
            user(". What's the", 2400, 2800),
            user(" weather in Barcelona", 3000, 3600),
            delegation("item_1", 3600),
            agent_tx(" Checking.", 3800, 4000),
            "<wait:commentary>",
            agent_tx(" Barcelona is sunny.", 7600, 8400),
            CLOSED,
        ]
        t = FakeLiveTransport(script)
        model = TestModel(responses=["Barcelona: 24°C, sunny, light sea breeze."])
        agent = Agent(name="backend", model=model, tools=[])
        s = LiveSession(t, agent, row_gap_ms=1000)
        events = await _collect(s)

        created = [e for e in events if isinstance(e, DelegationCreated)]
        assert len(created) == 1 and created[0].delegation_id == "item_1"
        # Prompt is rebuilt from the transcript (wire event carries no text).
        assert "user: Hi there. What's the weather in Barcelona" in created[0].prompt

        comm = t.commands("session.commentary.append")
        assert len(comm) == 1
        assert comm[0]["delegation_id"] == "item_1"
        assert comm[0]["content"] == "Barcelona: 24°C, sunny, light sea breeze."

        result = [e for e in events if isinstance(e, DelegationResult)][0]
        assert result.text.startswith("Barcelona: 24") and result.run_id and result.error is None
        assert model.call_count == 1
        assert s.delegations["item_1"]["status"] == "done"
        # The agent's text is NOT an AgentTextDone — only voiced transcript is.
        assert [e.text for e in events if isinstance(e, AgentTextDone)] == ["Checking.", "Barcelona is sunny."]

    async def test_tool_calls_surface_as_status_and_quiet_thinking(self):
        from timbal.types.content import ToolUseContent
        from timbal.types.message import Message

        def lookup(city: str) -> str:
            """Look up weather."""
            return f"sunny in {city}"

        tool_turn = Message(
            role="assistant",
            content=[ToolUseContent(id="c1", name="lookup", input={"city": "BCN"})],
            stop_reason="tool_use",
        )
        model = TestModel(responses=[tool_turn, "It's sunny in Barcelona."])
        agent = Agent(name="backend", model=model, tools=[lookup])
        script = [user(" weather", 1000, 1400), delegation("item_2", 1400), "<wait:commentary>", CLOSED]
        t = FakeLiveTransport(script)
        events = await _collect(LiveSession(t, agent, row_gap_ms=1000))

        statuses = [e.text for e in events if isinstance(e, AgentStatus)]
        assert any("lookup" in s for s in statuses)
        thinking = t.commands("session.thinking.append")
        assert any("lookup" in th["content"] and th["delegation_id"] == "item_2" for th in thinking)
        comm = t.commands("session.commentary.append")
        assert comm and comm[-1]["content"] == "It's sunny in Barcelona."
        assert model.call_count == 2

    async def test_memory_chains_across_delegations(self):
        seen: list[int] = []

        def handler(messages):
            seen.append(len(messages))
            return f"reply {len(seen)}"

        agent = Agent(name="backend", model=TestModel(handler=handler), tools=[])
        script = [
            user(" first", 1000, 1200),
            delegation("d1", 1200),
            "<wait:commentary>",
            user(" second", 5000, 5200),
            delegation("d2", 5200),
            "<wait:commentary>",
            CLOSED,
        ]
        t = FakeLiveTransport(script)
        s = LiveSession(t, agent, row_gap_ms=1000)
        events = await _collect(s)
        results = [e for e in events if isinstance(e, DelegationResult)]
        assert [r.text for r in results] == ["reply 1", "reply 2"]
        # Second run sees the first exchange in memory (user, assistant, user).
        assert seen == [1, 3]
        # Second prompt only carries transcript since the first delegation.
        prompts = [e.prompt for e in events if isinstance(e, DelegationCreated)]
        assert "first" in prompts[0] and "first" not in prompts[1] and "second" in prompts[1]

    async def test_custom_on_delegation_handler_and_chunking(self):
        long_text = " ".join(f"Sentence number {i} is here." for i in range(120))

        async def handler(_session, _did, _prompt):
            return long_text

        script = [user(" go", 1000, 1200), delegation("d", 1200), "<wait:commentary>", 0.05, CLOSED]
        t = FakeLiveTransport(script)
        events = await _collect(LiveSession(t, on_delegation=handler, row_gap_ms=1000))
        comm = t.commands("session.commentary.append")
        assert len(comm) > 1 and all(len(c["content"]) <= 1400 for c in comm)
        assert " ".join(c["content"] for c in comm) == long_text
        assert [e for e in events if isinstance(e, DelegationResult)][0].text == long_text

    async def test_no_backend_still_answers(self):
        script = [delegation("d", 100), "<wait:commentary>", CLOSED]
        t = FakeLiveTransport(script)
        await _collect(LiveSession(t))
        comm = t.commands("session.commentary.append")
        assert comm and "No backend" in comm[0]["content"]

    async def test_agent_failure_becomes_spoken_apology_and_error(self):
        async def boom(_session, _did, _prompt):
            raise RuntimeError("db down")

        script = [delegation("d", 100), "<wait:commentary>", CLOSED]
        t = FakeLiveTransport(script)
        events = await _collect(LiveSession(t, on_delegation=boom))
        res = [e for e in events if isinstance(e, DelegationResult)][0]
        assert res.error == "db down" and "Apologize" in res.text
        assert t.commands("session.commentary.append")[0]["delegation_id"] == "d"

    async def test_responses_delegation_is_surfaced_not_run(self):
        script = [
            {
                "type": "session.delegation.created",
                "offset_ms": 100,
                "delegation": {"id": "r1", "type": "delegation", "target": "responses", "response_id": "resp_1"},
            },
            {
                "type": "response.event",
                "delegation_id": "r1",
                "event": {"type": "response.output_text.delta", "delta": "The forecast is"},
            },
            CLOSED,
        ]
        t = FakeLiveTransport(script)
        events = await _collect(LiveSession(t, on_delegation=None))
        assert [e.delegation_id for e in events if isinstance(e, DelegationCreated)] == ["r1"]
        assert not t.commands("session.commentary.append")
        assert any(isinstance(e, AgentStatus) and "forecast" in e.text for e in events)

    async def test_control_helpers_send_expected_commands(self):
        t = FakeLiveTransport(["<hold>", CLOSED])
        s = LiveSession(t)

        async def drive():
            async for ev in s.run(_mic()):
                if isinstance(ev, SessionStarted):
                    await s.greet("Greet in Catalan, then listen.")
                    await s.context("Caller is on the checkout page.")
                    t.release()

        await asyncio.wait_for(drive(), 5)
        types = [e["type"] for e in t.sent]
        assert types[:3] == ["session.instructions.append", "session.commentary.append", "session.thinking.append"]
        assert all(e["delegation_id"] is None for e in t.sent[:3])


class TestStaleResults:
    async def test_substantive_caller_speech_routes_result_to_thinking(self):
        gate = asyncio.Event()

        async def slow(_s, _d, _p):
            await gate.wait()
            return "Barcelona: 24C and sunny."

        script = [
            user(" weather in Barcelona", 1000, 1600),
            delegation("d1", 1600),
            # Caller changes their mind while the backend works.
            user(" actually forget that, book me a table for two tonight", 3000, 5000),
            0.05,
            "<release>",
            "<wait:append>",
            CLOSED,
        ]
        t = _ReleasingTransport(script, gate)
        events = await _collect(LiveSession(t, on_delegation=slow))
        assert not t.commands("session.commentary.append")
        th = t.commands("session.thinking.append")
        assert len(th) == 1 and th[0]["delegation_id"] == "d1"
        assert "Late result" in th[0]["content"] and "book me a table" in th[0]["content"]
        assert th[0]["content"].endswith("Barcelona: 24C and sunny.")
        res = [e for e in events if isinstance(e, DelegationResult)][0]
        assert res.stale is True and res.spoken is False and res.text == "Barcelona: 24C and sunny."

    async def test_backchannel_does_not_make_result_stale(self):
        gate = asyncio.Event()

        async def slow(_s, _d, _p):
            await gate.wait()
            return "Sunny."

        script = [
            user(" weather", 1000, 1400),
            delegation("d1", 1400),
            user(" okay,", 3000, 3200),
            user(" thanks", 3200, 3400),
            0.05,
            "<release>",
            "<wait:commentary>",
            CLOSED,
        ]
        t = _ReleasingTransport(script, gate)
        events = await _collect(LiveSession(t, on_delegation=slow))
        assert [c["content"] for c in t.commands("session.commentary.append")] == ["Sunny."]
        res = [e for e in events if isinstance(e, DelegationResult)][0]
        assert res.stale is False and res.spoken is True

    async def test_drop_policy_discards_stale_result(self):
        gate = asyncio.Event()

        async def slow(_s, _d, _p):
            await gate.wait()
            return "Sunny."

        script = [
            delegation("d1", 1000),
            user(" never mind, different question entirely", 2000, 3500),
            0.05,
            "<release>",
            0.05,
            CLOSED,
        ]
        t = _ReleasingTransport(script, gate)
        events = await _collect(LiveSession(t, on_delegation=slow, stale_policy="drop"))
        assert not t.commands("session.commentary.append") and not t.commands("session.thinking.append")
        res = [e for e in events if isinstance(e, DelegationResult)][0]
        assert res.stale is True and res.spoken is False

    async def test_speak_policy_ignores_staleness(self):
        gate = asyncio.Event()

        async def slow(_s, _d, _p):
            await gate.wait()
            return "Sunny."

        script = [
            delegation("d1", 1000),
            user(" never mind, different question entirely", 2000, 3500),
            0.05,
            "<release>",
            "<wait:commentary>",
            CLOSED,
        ]
        t = _ReleasingTransport(script, gate)
        events = await _collect(LiveSession(t, on_delegation=slow, stale_policy="speak"))
        assert [c["content"] for c in t.commands("session.commentary.append")] == ["Sunny."]
        assert [e for e in events if isinstance(e, DelegationResult)][0].stale is False

    async def test_newer_delegation_makes_earlier_result_quiet(self):
        async def handler(_s, did, _p):
            await asyncio.sleep(0.02)
            return f"answer for {did}"

        script = [delegation("d1", 1000), delegation("d2", 1100), 0.3, CLOSED]
        t = FakeLiveTransport(script)
        events = await _collect(LiveSession(t, on_delegation=handler))
        th = t.commands("session.thinking.append")
        comm = t.commands("session.commentary.append")
        # d1 (superseded) → thinking; d2 (latest) → spoken. The "queued" note for d2 is thinking too.
        assert [c["delegation_id"] for c in comm] == ["d2"]
        assert any(c["delegation_id"] == "d1" and "answer for d1" in c["content"] for c in th)
        results = {e.delegation_id: e for e in events if isinstance(e, DelegationResult)}
        assert results["d1"].stale and not results["d1"].spoken
        assert not results["d2"].stale and results["d2"].spoken


class _ReleasingTransport(FakeLiveTransport):
    """``"<release>"`` marker sets the given gate (lets a blocked backend finish mid-script)."""

    def __init__(self, script, gate: asyncio.Event, **kw):
        super().__init__(script, **kw)
        self._gate = gate

    async def events(self):
        for ev in self.scripts[self.gen]:
            if ev == "<release>":
                self._gate.set()
                await asyncio.sleep(0)
                continue
            if ev == "<wait:commentary>":
                await asyncio.wait_for(self._commentary.wait(), 5)
                continue
            if ev == "<wait:append>":
                await asyncio.wait_for(self._append.wait(), 5)
                continue
            if isinstance(ev, float):
                await asyncio.sleep(ev)
                continue
            if ev.get("type") == "session.closed":
                self._finalized = True
            yield ev


class TestReconnect:
    async def test_drop_reconnects_with_transcript_seed_and_continues(self):
        first = [
            user(" Hi", 1000, 1200),
            user(" there", 1200, 1400),
            agent_tx(" Hello!", 2000, 2400),
            0.05,
            # socket drops here (no CLOSED)
        ]
        second = [user(" still here?", 500, 900), CLOSED]
        t = FakeLiveTransport(first, more_scripts=[second])
        s = LiveSession(t, reconnect_backoff_secs=(0.0,))
        events = await _collect(s)

        statuses = [e.text for e in events if isinstance(e, AgentStatus)]
        assert statuses == ["Reconnecting…", "Reconnected"]
        assert not [e for e in events if isinstance(e, SessionError)]
        assert s.reconnects == 1 and s.live_session_id == "live_test_1"
        seed = t.reconnect_inputs[0]
        assert [(i["role"], i["content"][0]["type"], i["content"][0]["text"]) for i in seed] == [
            ("user", "input_text", "Hi there"),
            ("assistant", "output_text", "Hello!"),
        ]
        # Rows straddling the drop were closed; the new session's fragments start fresh rows.
        assert [e.text for e in events if isinstance(e, TranscriptCommitted)] == ["Hi there", "still here?"]
        assert [(e.role, e.text) for e in s.transcript] == [
            ("user", "Hi there"),
            ("assistant", "Hello!"),
            ("user", "still here?"),
        ]
        assert isinstance(events[-1], SessionEnded)

    async def test_open_row_is_seeded_and_mic_keeps_flowing(self):
        first = [user(" unfinished", 1000, 1400)]  # drop with an open user row
        second = [CLOSED]
        t = FakeLiveTransport(first, more_scripts=[second])
        s = LiveSession(t, reconnect_backoff_secs=(0.0,), row_gap_ms=10_000)
        await _collect(s)
        assert t.reconnect_inputs[0][0]["content"][0]["text"] == "unfinished"
        assert len(t.audio) >= 3  # uplink was never torn down

    async def test_reconnect_retries_with_backoff_then_succeeds(self):
        t = FakeLiveTransport([0.01], more_scripts=[[CLOSED]])
        t.fail_reconnects = 2
        s = LiveSession(t, reconnect_attempts=3, reconnect_backoff_secs=(0.0, 0.01, 0.01))
        events = await _collect(s)
        assert s.reconnects == 1
        assert not [e for e in events if isinstance(e, SessionError)]

    async def test_reconnect_gives_up_after_attempts(self):
        t = FakeLiveTransport([0.01], more_scripts=[[CLOSED]])
        t.fail_reconnects = 5
        s = LiveSession(t, reconnect_attempts=2, reconnect_backoff_secs=(0.0,))
        events = await _collect(s)
        errs = [e.message for e in events if isinstance(e, SessionError)]
        assert len(errs) == 1 and "reconnect failed" in errs[0]
        assert s.reconnects == 0 and isinstance(events[-1], SessionEnded)

    async def test_reconnect_disabled_reports_lost_connection(self):
        t = FakeLiveTransport([0.01])
        events = await _collect(LiveSession(t, reconnect_attempts=0))
        assert [e.message for e in events if isinstance(e, SessionError)] == ["GPT-Live connection lost"]
        assert not t.reconnect_inputs

    async def test_transport_without_reconnect_support(self):
        class Plain(FakeLiveTransport):
            async def reconnect(self, *, input=None):
                raise NotImplementedError

        events = await _collect(LiveSession(Plain([0.01]), reconnect_backoff_secs=(0.0,)))
        errs = [e.message for e in events if isinstance(e, SessionError)]
        assert len(errs) == 1 and "cannot reconnect" in errs[0]

    async def test_server_initiated_close_is_final(self):
        closed = {**CLOSED, "reason": "safety_violation"}
        t = FakeLiveTransport([user(" hi", 0, 200), closed], more_scripts=[[CLOSED]])
        s = LiveSession(t, reconnect_backoff_secs=(0.0,))
        events = await _collect(s)
        errs = [e.message for e in events if isinstance(e, SessionError)]
        assert errs == ["GPT-Live session closed by server: safety_violation"]
        assert s.reconnects == 0 and not t.reconnect_inputs

    async def test_delegation_in_flight_answers_into_new_session(self):
        gate = asyncio.Event()

        async def slow(_s, _d, _p):
            await gate.wait()
            return "late but valid"

        first = [user(" order status", 1000, 1600), delegation("d1", 1600), 0.02]  # drop
        second = [0.02, "<release>", "<wait:commentary>", CLOSED]
        t = _ReleasingTransport(first, gate, more_scripts=[second])
        s = LiveSession(t, on_delegation=slow, reconnect_backoff_secs=(0.0,))
        events = await _collect(s)
        assert s.reconnects == 1
        assert [c["content"] for c in t.commands("session.commentary.append")] == ["late but valid"]
        assert [e for e in events if isinstance(e, DelegationResult)][0].spoken is True

    @staticmethod
    def _dropping_transport(gate: asyncio.Event, *, lag: float, reconnect_delay: float = 0.2):
        """Socket dies after the first script; ``lag`` is how long the reader
        takes to notice (events() returning) after sends already fail."""

        class _DroppingTransport(FakeLiveTransport):
            down = False

            async def events(self):
                async for ev in super().events():
                    yield ev
                if not self._finalized:
                    self.down = True
                    gate.set()
                    if lag:
                        await asyncio.sleep(lag)

            async def reconnect(self, *, input=None):
                await asyncio.sleep(reconnect_delay)
                started = await super().reconnect(input=input)
                self.down = False
                return started

            async def send(self, event):
                if self.down:
                    raise ConnectionError("socket closed")
                await super().send(event)

        return _DroppingTransport

    @pytest.mark.parametrize("lag", [0.0, 0.1, 0.6])
    async def test_result_finishing_while_socket_is_down_lands_on_new_session(self, lag):
        # Bugbot: a delegation that completes during the outage used to hit the
        # dead transport, swallow the error and never speak. lag=0: the reader
        # already noticed the drop (delivery waits for the reconnect). lag>0:
        # the socket is dead but the reader has not returned yet — the first
        # send raises and the retry waits for the *next generation*, however
        # long the reader takes (0.6 s beat the old fixed 0.25 s grace).
        gate = asyncio.Event()

        async def slow(_s, _d, _p):
            await gate.wait()
            return "late but valid"

        first = [user(" order status", 1000, 1600), delegation("d1", 1600)]  # drop after
        second = ["<wait:commentary>", CLOSED]
        t = self._dropping_transport(gate, lag=lag)(first, more_scripts=[second])
        s = LiveSession(t, on_delegation=slow, reconnect_backoff_secs=(0.0,))
        events = await _collect(s)
        assert s.reconnects == 1
        assert [c["content"] for c in t.commands("session.commentary.append")] == ["late but valid"]
        assert [e for e in events if isinstance(e, DelegationResult)][0].spoken is True

    async def test_result_retry_gives_up_promptly_when_reconnect_fails(self):
        # Send fails, reader notices late, reconnect never succeeds: the retry
        # must drop when the session ends — not sit out the reconnect budget.
        gate = asyncio.Event()

        async def slow(_s, _d, _p):
            await gate.wait()
            return "never spoken"

        first = [user(" hi", 1000, 1200), delegation("d1", 1200)]
        t = self._dropping_transport(gate, lag=0.1)(first)  # no more scripts → reconnect raises
        t.fail_reconnects = 0
        s = LiveSession(t, on_delegation=slow, reconnect_attempts=1, reconnect_backoff_secs=(0.0,))
        events = await asyncio.wait_for(_collect(s, timeout=5.0), 6.0)
        assert not t.commands("session.commentary.append")
        result = [e for e in events if isinstance(e, DelegationResult)]
        assert result and result[0].spoken is False

    async def test_result_after_session_over_is_dropped_not_hung(self):
        gate = asyncio.Event()

        async def slow(_s, _d, _p):
            await gate.wait()
            return "too late"

        t = FakeLiveTransport([user(" hi", 1000, 1200), delegation("d1", 1200), 0.02])  # drop, no reconnect
        s = LiveSession(t, on_delegation=slow, reconnect_attempts=0)

        async def go():
            async for _ in s.run(_mic()):
                pass

        await asyncio.wait_for(go(), 5)
        gate.set()
        await asyncio.sleep(0.05)
        assert not t.commands("session.commentary.append")

    async def test_usage_accumulates_across_generations_and_is_unconfirmed(self):
        first = [
            {"type": "session.usage.updated", "usage": {"seconds": 12.0}, "context_window": {"usage_ratio": 0.01}},
            0.01,  # drop
        ]
        second = [CLOSED]  # usage 20.0
        t = FakeLiveTransport(first, more_scripts=[second])
        s = LiveSession(t, reconnect_backoff_secs=(0.0,))
        await _collect(s)
        assert s.usage_seconds == 32.0 and s.usage_confirmed is False
        summary = LiveSessionSummary.from_session(s)
        assert summary.reconnects == 1 and summary.usage_confirmed is False
        assert summary.est_usd == round(32 / 60 * 0.05, 4)

    def test_history_items_are_bounded_from_the_tail(self):
        s = LiveSession(FakeLiveTransport([]), reconnect_attempts=0)
        for i in range(100):
            s._transcript.append(TranscriptEntry(role="user" if i % 2 == 0 else "assistant", text=f"line {i}"))
        items = s.history_items(max_items=10)
        assert len(items) == 10 and items[-1]["content"][0]["text"] == "line 99"
        items = s.history_items(max_items=10, max_chars=10)
        assert [i["content"][0]["text"] for i in items] == ["line 99"]

    def test_default_row_gap(self):
        assert LiveSession(FakeLiveTransport([]), reconnect_attempts=0).row_gap_ms == 800


class FakeRecorder:
    """Stand-in for CallRecorder: records the feed, captures the manifest."""

    def __init__(self, path: str = "/tmp/rec/abc.mp3"):
        from pathlib import Path

        self.audio_path = Path(path)
        self.layout = "mixed"
        self.sample_rate = 24_000
        self.bitrate_kbps = 32
        self.duration_secs = 1.5
        self.meta = {"org_id": "o1"}
        self.mic = b""
        self.agent = b""
        self.manifest: dict | None = None
        self.saved: list = []
        self.retargeted: list = []
        self.on_saved = self._on_saved

    def add_mic(self, chunk: bytes) -> None:
        self.mic += chunk

    def add_agent(self, chunk: bytes) -> None:
        self.agent += chunk

    def retarget(self, path) -> None:
        self.retargeted.append(path)
        self.audio_path = path

    def close(self, manifest=None):
        from timbal.voice.recording import RecordingResult

        self.manifest = manifest
        return RecordingResult(audio_path=self.audio_path, manifest_path=None, duration_secs=1.5)

    async def _on_saved(self, result) -> None:
        self.saved.append(result)


class TestRecordingAndSessionRun:
    async def test_recorder_fed_both_directions_and_manifest_written(self):
        script = [audio(SILENCE), audio(SPEECH), agent_tx(" Hello", 1000, 1400), CLOSED]
        rec = FakeRecorder()
        s = LiveSession(FakeLiveTransport(script), recorder=rec, session_id="abc", session_run=False)
        await _collect(s)
        # Mic drives the clock (3 silence frames from _mic); agent gets *every*
        # frame, zeros included — the output stream is the timeline.
        assert rec.mic == SILENCE * 3
        assert rec.agent == SILENCE + SPEECH
        m = rec.manifest
        assert m is not None and m["session_id"] == "abc"
        assert m["meta"]["pipeline"] == "live" and m["meta"]["live_session_id"] == "live_test"
        assert m["meta"]["usage_seconds"] == 20.0 and m["meta"]["est_usd"] == round(20 / 60 * 0.05, 4)
        assert [(e["role"], e["text"]) for e in m["transcript"]] == [("assistant", "Hello")]
        # Session-clock start of the row, not the wall clock of its close.
        assert m["transcript"][0]["offset_ms"] == 1000
        assert len(rec.saved) == 1 and rec.saved[0].audio_path.name == "abc.mp3"

    def test_session_id_setter_retargets_recorder(self):
        rec = FakeRecorder()
        s = LiveSession(FakeLiveTransport([]), recorder=rec, session_id="abc", reconnect_attempts=0)
        s.session_id = "call-42"
        assert s.session_id == "call-42" and rec.audio_path.name == "call-42.mp3"
        with pytest.raises(ValueError):
            s.session_id = "bad id!"

    async def test_session_run_carries_voice_seconds_and_summary(self):
        from timbal.state.tracing.providers import InMemoryTracingProvider

        script = [
            user(" Hi", 1000, 1200),
            {"type": "session.usage.updated", "usage": {"seconds": 13.0}},
            CLOSED,
        ]
        rec = FakeRecorder()
        s = LiveSession(FakeLiveTransport(script), recorder=rec, parent_run_id=None)
        await _collect(s)
        assert s.session_run_id is not None
        trace = InMemoryTracingProvider._storage[s.session_run_id]
        root = trace.get(trace._root_call_id)
        assert root.path == "voice_live_session"
        assert root.usage["openai/gpt-live-1:seconds"] == 20
        summary = root.metadata["voice_live_session"]
        assert summary["session_id"] == s.session_id and summary["live_session_id"] == "live_test"
        assert summary["usage_seconds"] == 20.0 and summary["recording"]["file"] == rec.audio_path.name
        assert [e["text"] for e in summary["transcript"]] == ["Hi"]
        # The run spans the call, not the millisecond it took to persist it.
        assert root.t0 == int(s.started_at * 1000) and root.t1 >= root.t0
        # Manifest names the run; the run names the recording.
        assert rec.manifest["meta"]["session_run_id"] == s.session_run_id

    async def test_session_run_chains_behind_last_delegation(self):
        from timbal.state.tracing.providers import InMemoryTracingProvider

        script = [
            user(" Book it", 1000, 1400),
            delegation("d1", 1400),
            "<wait:commentary>",
            CLOSED,
        ]
        agent = Agent(name="backend", model=TestModel(responses=["Booked."]), tools=[])
        s = LiveSession(FakeLiveTransport(script), agent, parent_run_id="run_prev")
        await _collect(s)
        deleg_run = s.delegations["d1"]["run_id"]
        assert deleg_run and s.last_run_id == deleg_run
        # Both the delegation run and the session run are persisted; the
        # delegation run chains to the call's parent, the session run to it.
        assert deleg_run in InMemoryTracingProvider._storage
        trace = InMemoryTracingProvider._storage[s.session_run_id]
        root = trace.get(trace._root_call_id)
        assert root.path == "voice_live_session"
        summary = root.metadata["voice_live_session"]
        assert summary["parent_run_id"] == "run_prev" and summary["last_run_id"] == deleg_run
        assert summary["delegations"]["d1"]["run_id"] == deleg_run

    async def test_session_run_can_be_disabled(self):
        s = LiveSession(FakeLiveTransport([CLOSED]), session_run=False)
        await _collect(s)
        assert s.session_run_id is None


class TestClientConfig:
    def test_session_config_pcm_default(self):
        c = OpenAILiveClient(api_key="k", instructions="Be terse.")
        cfg = c.session_config()
        assert cfg == {
            "model": "gpt-live-1",
            "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}},
            "instructions": "Be terse.",
            "delegation": {"type": "client"},
        }

    def test_session_config_telephony_and_responses(self):
        c = OpenAILiveClient(
            api_key="k",
            encoding="mulaw",
            sample_rate=8000,
            voice="vesper",
            delegation={"type": "responses", "responses": {"model": "gpt-5.6-luna", "tools": [{"type": "web_search"}]}},
            input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            extra={"store": True},
        )
        cfg = c.session_config()
        assert cfg["audio"]["format"] == {"type": "audio/pcmu", "rate": 8000}
        assert cfg["delegation"]["type"] == "responses"
        assert cfg["input"][0]["role"] == "user" and cfg["store"] is True

    def test_bad_pcm_rate_rejected(self):
        with pytest.raises(ValueError):
            OpenAILiveClient(api_key="k", sample_rate=44100).session_config()

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from timbal.voice.openai_live import _resolve_api_key

        with pytest.raises(ValueError):
            _resolve_api_key(None)

    async def test_connect_401_names_the_api_key(self, monkeypatch):
        from unittest.mock import AsyncMock, patch

        from websockets.datastructures import Headers
        from websockets.exceptions import InvalidStatus
        from websockets.http11 import Response

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        rejection = InvalidStatus(Response(401, "Unauthorized", Headers()))
        with patch("websockets.asyncio.client.connect", AsyncMock(side_effect=rejection)):
            with pytest.raises(ConnectionError, match="OPENAI_API_KEY"):
                await OpenAILiveClient().connect()

    def test_split_for_append(self):
        assert split_for_append("") == []
        assert split_for_append("short") == ["short"]
        chunks = split_for_append("A. " * 800, max_chars=100)
        assert all(len(c) <= 100 for c in chunks) and "".join(c.replace(" ", "") for c in chunks) == "A." * 800
        assert split_for_append("x" * 250, max_chars=100) == ["x" * 100, "x" * 100, "x" * 50]
