"""What happens to a call when a provider socket dies mid-conversation.

There is no reconnection in the Deepgram adapter, so a dropped socket ends the
call. That is a deliberate-enough outcome, but it rests entirely on one
guarantee: the receive loop must enqueue its end-of-stream sentinel on *every*
exit path. ``VoiceSession._process_stt_events`` closes the session when
``stt.events()`` completes, and nothing else notices a dead STT — there is no
heartbeat and no "connected but silent" watchdog. Drop the sentinel and the
failure mode changes from "the call ends" to "the caller keeps talking to a
session that can no longer hear them, indefinitely, with no error".

These tests pin that sentinel, for the normal-close case as well as the abnormal
one, because the normal case is the one that emits no error and so leaves no
other trace.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from timbal.voice.deepgram import _DeepgramSTTBase
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close


class _ClosingWs:
    """A socket that yields nothing and reports itself closed on first read."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.sent: list[Any] = []

    def __aiter__(self) -> _ClosingWs:
        return self

    async def __anext__(self) -> str:
        raise self._error

    async def send(self, _data: Any) -> None:
        # Real sockets raise here too once closed; the adapter swallows it.
        raise self._error


class _StubSTT(_DeepgramSTTBase):
    """Concrete subclass: the base leaves ``_handle_message`` abstract."""

    def _build_uri(self, _config: Any) -> str:
        return "wss://example.invalid/listen"

    async def _handle_message(self, _msg: dict[str, Any]) -> None:
        return None

    async def commit(self) -> None:
        return None


def _stt(error: BaseException) -> _StubSTT:
    stt = _StubSTT(api_key="test-key")
    stt._ws = _ClosingWs(error)
    return stt


NORMAL_CLOSES = (
    pytest.param(ConnectionClosedOK(Close(1000, ""), None), id="1000-normal"),
    pytest.param(ConnectionClosedOK(Close(1001, "going away"), None), id="1001-going-away"),
)
ABNORMAL_CLOSES = (
    pytest.param(ConnectionClosedError(Close(1011, "internal error"), None), id="1011-internal"),
    pytest.param(ConnectionClosedError(Close(1006, "abnormal"), None), id="1006-abnormal"),
)


class TestSttStreamAlwaysTerminates:
    """``events()`` must complete when the socket dies, whatever the close code."""

    @pytest.mark.parametrize("error", [*NORMAL_CLOSES, *ABNORMAL_CLOSES])
    async def test_receive_loop_enqueues_the_sentinel(self, error: BaseException) -> None:
        stt = _stt(error)
        await stt._receive_loop()

        drained = _drain_queue(stt)
        missing = (
            f"no end-of-stream sentinel after {type(error).__name__}: events() would await "
            f"forever and the session would run on with a dead STT (queued: {drained})"
        )
        assert drained and drained[-1] is None, missing

    @pytest.mark.parametrize("error", [*NORMAL_CLOSES, *ABNORMAL_CLOSES])
    async def test_events_completes_rather_than_hanging(self, error: BaseException) -> None:
        stt = _stt(error)
        receiver = asyncio.create_task(stt._receive_loop())

        async def _drain() -> list[str]:
            texts = []
            with pytest.raises(RuntimeError) if _is_abnormal(error) else _no_raise():
                async for event in stt.events():
                    texts.append(event.text)
            return texts

        # A hang here is the bug this file exists to catch, so bound the wait.
        await asyncio.wait_for(_drain(), timeout=2.0)
        await asyncio.wait_for(receiver, timeout=2.0)

    async def test_abnormal_close_is_reported_and_normal_close_is_not(self) -> None:
        """Only an abnormal code produces an error event.

        The consequence is worth stating plainly: on a 1000/1001 close mid-call the
        session simply ends, indistinguishable from the user hanging up, and the
        only trace is a debug-level ``dg_stt_ws_closed``.
        """
        abnormal = _stt(ConnectionClosedError(Close(1011, "internal error"), None))
        await abnormal._receive_loop()
        assert [e for e in _drain_queue(abnormal) if e is not None and e.type == "error"]

        normal = _stt(ConnectionClosedOK(Close(1000, ""), None))
        await normal._receive_loop()
        assert [e for e in _drain_queue(normal) if e is not None] == []

    async def test_push_audio_survives_a_dead_socket(self) -> None:
        """Audio written to a closed socket must not raise into the audio task.

        It is silently dropped, which is only acceptable because the receive loop's
        sentinel is already tearing the session down.
        """
        stt = _stt(ConnectionClosedOK(Close(1000, ""), None))
        await stt.push_audio(b"\x00\x01" * 4096)
        await stt._flush_audio()


class _no_raise:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


def _is_abnormal(error: BaseException) -> bool:
    return isinstance(error, ConnectionClosedError) and error.rcvd is not None and error.rcvd.code not in (1000, 1001)


def _drain_queue(stt: _StubSTT) -> list[Any]:
    out = []
    while not stt._queue.empty():
        out.append(stt._queue.get_nowait())
    return out
