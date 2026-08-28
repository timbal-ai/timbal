"""Suspensions on the voice wire.

A voice agent that parks on an approval gate or an ``ask_user`` is, from the
caller's side, a voice agent that went quiet. These tests pin that the
suspension becomes a session event — the one event class that requires a client
response — and that it carries the ``run_id`` a client needs to resume the run
over HTTP.
"""
# ruff: noqa: ARG002

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import aclosing

from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.server.voice import event_to_payloads
from timbal.state import suspend
from timbal.types.content import ToolUseContent
from timbal.types.message import Message
from timbal.voice import (
    AgentApproval,
    AgentInteraction,
    AgentTextDone,
    AudioInputConfig,
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
    VoiceSession,
    VoiceSessionEvent,
)

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _STT(SpeechToText):
    """Replays a single committed utterance, then closes the session."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def connect(self, config: AudioInputConfig) -> None:
        pass

    async def push_audio(self, chunk: bytes) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        yield TranscriptEvent(type="committed", text=self._text)

    async def close(self) -> None:
        pass


class _TTS(TextToSpeech):
    def __init__(self) -> None:
        self.synthesized: list[str] = []

    async def connect(self, config: AudioOutputConfig) -> None:
        pass

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.synthesized.append(text)
        yield b"\x00\x01" * 16

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call(name: str, **input_: object) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=f"call_{name}", name=name, input=input_)],
        stop_reason="tool_use",
    )


async def _run(agent: Agent) -> tuple[VoiceSession, list[VoiceSessionEvent]]:
    """Drive one turn and return the session plus every event it emitted."""

    async def _no_audio() -> AsyncIterator[bytes]:
        return
        yield

    session = VoiceSession(agent=agent, stt=_STT("go ahead"), tts=_TTS(), turn_detector="heuristic")
    events: list[VoiceSessionEvent] = []
    async with aclosing(session.run(_no_audio())) as stream:
        async for ev in stream:
            events.append(ev)
    return session, events


def _ask_name() -> str:
    """Suspend asking for a name."""
    return suspend({"question": "what name should I use?"}, kind="ask_user")


def _wire_money(amount: int) -> str:
    """Wire money."""
    return f"wired {amount}"


def _interaction_agent() -> Agent:
    return Agent(
        name="voice_hitl",
        model=TestModel(responses=[_tool_call("_ask_name"), "Thanks!"]),
        tools=[_ask_name],
    )


def _approval_agent() -> Agent:
    return Agent(
        name="voice_approval",
        model=TestModel(responses=[_tool_call("wire_money", amount=500), "done"]),
        tools=[
            Tool(
                name="wire_money",
                handler=_wire_money,
                requires_approval=True,
                approval_kind="payment",
                approval_prompt="Wire $500?",
                approval_ui={"title": "Wire transfer", "severity": "high"},
            )
        ],
    )


# ---------------------------------------------------------------------------
# Session events
# ---------------------------------------------------------------------------


class TestSuspensionsBecomeSessionEvents:
    async def test_interaction_is_emitted_with_ids_and_payload(self) -> None:
        session, events = await _run(_interaction_agent())

        interactions = [e for e in events if isinstance(e, AgentInteraction)]
        assert len(interactions) == 1, "a suspended run must not be silent on the wire"
        ev = interactions[0]
        assert ev.kind == "ask_user"
        assert ev.payload == {"question": "what name should I use?"}
        assert ev.interaction_id, "the resume key must be carried"
        assert ev.tool_call_id == "call__ask_name"

        # run_id is the only identifier a voice client gets for the run behind
        # the call, and resuming needs it as parent_id.
        assert ev.run_id == str(session._last_run_context.id)

        # The run really is parked — the event is not cosmetic.
        assert session._last_run_context.root_span().status.reason == "input_required"

    async def test_approval_is_emitted_with_its_card(self) -> None:
        _, events = await _run(_approval_agent())

        approvals = [e for e in events if isinstance(e, AgentApproval)]
        assert len(approvals) == 1
        ev = approvals[0]
        assert ev.approval_id
        assert ev.kind == "payment"
        assert ev.prompt == "Wire $500?"
        assert ev.ui == {"title": "Wire transfer", "severity": "high"}
        assert ev.input == {"amount": 500}
        assert ev.input_schema is not None, "input + schema is what makes a generic card renderable"

    async def test_emitted_at_the_event_not_at_end_of_turn(self) -> None:
        """Deferring to the end of the turn would put the suspension behind the
        span close-out, the terminal OutputEvent and a trace save."""
        _, events = await _run(_approval_agent())

        types = [type(e) for e in events]
        assert types.index(AgentApproval) < types.index(AgentTextDone)

    async def test_a_turn_that_does_not_suspend_is_unchanged(self) -> None:
        agent = Agent(name="plain", model=TestModel(responses=["All good."]), tools=[])
        _, events = await _run(agent)

        assert not [e for e in events if isinstance(e, AgentInteraction | AgentApproval)]


# ---------------------------------------------------------------------------
# Wire mapping
# ---------------------------------------------------------------------------


class TestSuspensionsReachTheWire:
    """``event_to_payloads`` is shared by every transport, so WS, WebRTC and
    LiveKit are covered together."""

    def test_interaction_payload(self) -> None:
        ev = AgentInteraction(
            run_id="run-1",
            interaction_id="int-1",
            kind="ask_user",
            payload={"question": "which one?"},
            response_schema={"type": "string"},
            tool_call_id="toolu_1",
        )
        assert event_to_payloads(ev, session=None, meta={}) == [
            {
                "type": "agent_interaction",
                "run_id": "run-1",
                "interaction_id": "int-1",
                "kind": "ask_user",
                "payload": {"question": "which one?"},
                "response_schema": {"type": "string"},
                "tool_call_id": "toolu_1",
            }
        ]

    def test_approval_payload(self) -> None:
        ev = AgentApproval(
            run_id="run-1",
            approval_id="app-1",
            kind="payment",
            prompt="Wire $500?",
            ui={"title": "Wire transfer"},
            input={"amount": 500},
        )
        (payload,) = event_to_payloads(ev, session=None, meta={})
        assert payload["type"] == "agent_approval"
        assert payload["run_id"] == "run-1"
        assert payload["approval_id"] == "app-1"
        assert payload["ui"] == {"title": "Wire transfer"}
        assert payload["input"] == {"amount": 500}

    def test_unserializable_input_is_stringified_not_dropped(self) -> None:
        """A tool's validated input is arbitrary. A transport that raised while
        encoding would look exactly like not emitting approvals at all."""

        class _Opaque:
            def __repr__(self) -> str:
                return "<opaque>"

        ev = AgentApproval(run_id="r", approval_id="a", input={"file": _Opaque()})
        (payload,) = event_to_payloads(ev, session=None, meta={})
        assert payload["input"] == {"file": "<opaque>"}
        assert json.dumps(payload), "the payload must survive the transport encoder"

    def test_session_events_it_does_not_know_are_still_ignored(self) -> None:
        class _Unknown(VoiceSessionEvent):
            type: str = "unknown"

        assert event_to_payloads(_Unknown(), session=None, meta={}) == []


class TestSuspensionsOverAFullTurn:
    async def test_wire_sequence_is_purely_additive(self) -> None:
        """A client dispatching on ``type`` sees its existing sequence with one
        new frame spliced in. Drop the frame it does not know and every message
        it does know is still there, in the order it already expects."""
        session, events = await _run(_approval_agent())
        types = [p["type"] for e in events for p in event_to_payloads(e, session, meta={})]

        assert "agent_approval" in types
        assert types.index("agent_approval") < types.index("agent_text_done")

        legacy_view = [t for t in types if t != "agent_approval"]
        assert legacy_view[0] == "session_started"
        assert legacy_view[-2:] == ["session_transcript", "session_ended"]
        assert set(legacy_view) <= {
            "session_started",
            "transcript_committed",
            "agent_status",
            "agent_text_delta",
            "agent_text_done",
            "audio",
            "metrics",
            "session_transcript",
            "session_ended",
        }
