"""What happens to a call when a provider socket dies mid-conversation.

Neither STT adapter reconnects, so a dropped socket ends the call. Two
invariants make that ending honest instead of silent, and both are pinned here
for Deepgram and ElevenLabs alike:

* The receive loop must enqueue its end-of-stream sentinel on *every* exit
  path. ``VoiceSession._process_stt_events`` closes the session when
  ``stt.events()`` completes, and nothing else notices a dead STT. Drop the
  sentinel and the caller keeps talking to a session that can no longer hear
  them, indefinitely, with no error.

* An *unrequested* close — any code, 1000/1001 included — must surface an
  error event. The provider hanging up mid-call is not the user hanging up;
  before this was pinned, a normal-code close ended the session with a
  debug-level log as its only trace. Requested closes (``close()`` sets
  ``_stop`` before touching the socket) stay silent.

A dead-but-open TCP connection is the websockets library's problem, not ours:
both adapters keep the library's default ping keepalive (20s interval, 20s
timeout), so a peer that stops answering pings surfaces in the receive loop as
``ConnectionClosedError`` 1011 and flows through the same paths pinned here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from timbal.voice.deepgram import _DeepgramSTTBase
from timbal.voice.elevenlabs import ElevenLabsRealtimeSTT
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close


class _ClosingWs:
    """A socket that yields nothing and reports itself closed on first use."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.sent: list[Any] = []

    def __aiter__(self) -> _ClosingWs:
        return self

    async def __anext__(self) -> str:
        raise self._error

    async def send(self, _data: Any) -> None:
        # Real sockets raise here too once closed; the adapters swallow it.
        raise self._error


class _StubDeepgramSTT(_DeepgramSTTBase):
    """Concrete subclass: the base leaves ``_handle_message`` abstract."""

    def _build_uri(self, _config: Any) -> str:
        return "wss://example.invalid/listen"

    async def _handle_message(self, _msg: dict[str, Any]) -> None:
        return None

    async def commit(self) -> None:
        await self._flush_audio()
        await self._send_json({"type": "Finalize"})


def _deepgram(error: BaseException) -> _StubDeepgramSTT:
    stt = _StubDeepgramSTT(api_key="test-key")
    stt._ws = _ClosingWs(error)
    return stt


def _elevenlabs(error: BaseException) -> ElevenLabsRealtimeSTT:
    stt = ElevenLabsRealtimeSTT(api_key="test-key")
    stt._ws = _ClosingWs(error)
    return stt


PROVIDERS = (
    pytest.param(_deepgram, id="deepgram"),
    pytest.param(_elevenlabs, id="elevenlabs"),
)
CLOSES = (
    pytest.param(ConnectionClosedOK(Close(1000, ""), None), id="1000-normal"),
    pytest.param(ConnectionClosedOK(Close(1001, "going away"), None), id="1001-going-away"),
    pytest.param(ConnectionClosedError(Close(1011, "internal error"), None), id="1011-internal"),
    pytest.param(ConnectionClosedError(Close(1006, "abnormal"), None), id="1006-abnormal"),
)


class TestSttStreamAlwaysTerminates:
    """``events()`` must complete when the socket dies, whatever the close code."""

    @pytest.mark.parametrize("make_stt", PROVIDERS)
    @pytest.mark.parametrize("error", CLOSES)
    async def test_receive_loop_enqueues_the_sentinel(self, make_stt, error: BaseException) -> None:
        stt = make_stt(error)
        await stt._receive_loop()

        drained = _drain_queue(stt)
        missing = (
            f"no end-of-stream sentinel after {type(error).__name__}: events() would await "
            f"forever and the session would run on with a dead STT (queued: {drained})"
        )
        assert drained and drained[-1] is None, missing

    @pytest.mark.parametrize("make_stt", PROVIDERS)
    @pytest.mark.parametrize("error", CLOSES)
    async def test_events_completes_and_raises_on_unrequested_close(
        self, make_stt, error: BaseException
    ) -> None:
        """Every unrequested close — normal codes included — is an error, not a shrug."""
        stt = make_stt(error)
        receiver = asyncio.create_task(stt._receive_loop())

        async def _drain() -> None:
            with pytest.raises(RuntimeError, match="STT connection closed"):
                async for _ in stt.events():
                    pass

        # A hang here is the bug this file exists to catch, so bound the wait.
        await asyncio.wait_for(_drain(), timeout=2.0)
        await asyncio.wait_for(receiver, timeout=2.0)


class TestRequestedCloseStaysSilent:
    """Our own ``close()`` must not manufacture an error out of its side effects.

    ``close()`` sets ``_stop`` before touching the socket; the receive loop
    reads that flag to tell a teardown we asked for apart from a provider
    hanging up on us. An error event here races session close and can surface
    a spurious SessionError on a perfectly normal hangup.
    """

    @pytest.mark.parametrize("make_stt", PROVIDERS)
    async def test_no_error_event_when_stop_was_requested(self, make_stt) -> None:
        stt = make_stt(ConnectionClosedOK(Close(1000, ""), None))
        stt._stop.set()
        await stt._receive_loop()
        drained = _drain_queue(stt)
        assert [e for e in drained if e is not None] == []
        assert drained[-1] is None  # the sentinel still terminates events()


class TestSendsSurviveADeadSocket:
    """Audio/control writes to a closed socket must not raise into their callers.

    Silent dropping is only acceptable because the receive loop's sentinel is
    already tearing the session down. The callers that must not see a raise:
    the session's audio forwarder (``push_audio``), the adapters' periodic
    flushers, and the VAD endpointing fast path (``commit()``).
    """

    @pytest.mark.parametrize("make_stt", PROVIDERS)
    async def test_push_audio_and_flush(self, make_stt) -> None:
        stt = make_stt(ConnectionClosedOK(Close(1000, ""), None))
        # Enough PCM to cross Deepgram's eager-send threshold (2560 bytes).
        await stt.push_audio(b"\x00\x01" * 4096)
        await stt._flush_audio()

    @pytest.mark.parametrize("make_stt", PROVIDERS)
    async def test_commit(self, make_stt) -> None:
        stt = make_stt(ConnectionClosedOK(Close(1000, ""), None))
        await stt.push_audio(b"\x00\x01" * 64)
        await stt.commit()


def _drain_queue(stt: Any) -> list[Any]:
    out = []
    while not stt._queue.empty():
        out.append(stt._queue.get_nowait())
    return out
