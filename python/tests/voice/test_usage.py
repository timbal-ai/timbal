"""Usage delivery must survive turn cancellation and session teardown."""

import pytest
from timbal.server.voice import event_to_payloads
from timbal.voice import VoiceUsageEvent
from timbal.voice.providers import VoiceUsageEmitter

from .test_session import _collect_events, _make_session


def usage_event(usage_id="req", status="complete"):
    return VoiceUsageEvent(
        usage_id=usage_id,
        provider="openai",
        operation="stt",
        model="whisper-1",
        status=status,
        usage={"type": "duration", "seconds": 1.25} if status == "complete" else None,
    )


def test_listeners_are_isolated_idempotent_and_removable():
    first, second = VoiceUsageEmitter(), VoiceUsageEmitter()
    received = []

    def broken(event):
        event.usage["seconds"] = 999
        raise RuntimeError("consumer failed")

    first.add_usage_listener(broken)
    first.add_usage_listener(received.append)
    first.add_usage_listener(received.append)
    second._emit_usage(usage_event())
    assert received == []
    first._emit_usage(usage_event())
    assert len(received) == 1 and received[0].usage["seconds"] == 1.25
    first.remove_usage_listener(received.append)
    first.remove_usage_listener(received.append)
    first._emit_usage(usage_event())
    assert len(received) == 1


async def test_session_delivers_shutdown_usage_before_ended(monkeypatch):
    session, stt, tts = _make_session()
    received = []
    stt.add_usage_listener(received.append)
    original_close = stt.close

    async def close_with_usage():
        stt._emit_usage(usage_event("pending", "incomplete"))
        await original_close()

    monkeypatch.setattr(stt, "close", close_with_usage)
    events = await _collect_events(session)
    assert [e.type for e in events] == ["session_started", "voice_usage", "session_ended"]
    assert events[-2].status == "incomplete"
    assert received == [events[-2]]
    assert stt._usage_listeners == [received.append]
    assert tts._usage_listeners == []


async def test_session_forwards_complete_usage_and_does_not_drop_it_on_interrupt():
    session, stt, _ = _make_session()
    await session.prepare()
    try:
        stt._emit_usage(usage_event())
        session._drop_queued_audio_output()
        assert session._event_queue.get_nowait() == usage_event()
    finally:
        await session.close()
    assert stt._usage_listeners == []


async def test_prepare_failure_removes_session_listeners(monkeypatch):
    session, stt, tts = _make_session()

    async def fail(_config):
        raise RuntimeError("invalid configuration")

    monkeypatch.setattr(tts, "connect", fail)
    with pytest.raises(RuntimeError, match="invalid configuration"):
        await session.prepare()
    assert stt._usage_listeners == tts._usage_listeners == []


def test_usage_wire_payload_preserves_provider_quantities_and_ids():
    event = usage_event()
    payloads = event_to_payloads(event, None, {})
    assert payloads == [event.model_dump(mode="json")]
    assert VoiceUsageEvent.model_validate(payloads[0]) == event
