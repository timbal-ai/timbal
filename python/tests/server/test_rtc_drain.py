"""``_drain_data_channel``: a session that ends on its own still gets its last messages out."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from types import SimpleNamespace

from timbal.server.rtc import _drain_data_channel


def _pc(**queues: deque) -> SimpleNamespace:
    sctp = SimpleNamespace(_data_channel_queue=deque(), _outbound_queue=deque(), _sent_queue=deque())
    for name, queue in queues.items():
        setattr(sctp, name, queue)
    return SimpleNamespace(sctp=sctp)


def _channel(state: str = "open", buffered: int = 0) -> SimpleNamespace:
    return SimpleNamespace(readyState=state, bufferedAmount=buffered)


def _returns_without_waiting(drain) -> bool:
    """Whether the call finishes without ever suspending: stepped by hand, so no clock is involved (a
    loaded runner can take most of a second to get round to a coroutine that waits for nothing)."""
    try:
        drain.send(None)
    except StopIteration:
        return True
    drain.close()
    return False


async def test_nothing_waiting_returns_at_once() -> None:
    assert _returns_without_waiting(_drain_data_channel(_pc(), _channel()))


async def test_waits_for_the_last_messages_to_be_acknowledged() -> None:
    in_flight: deque = deque(["session_transcript", "session_ended"])
    pc = _pc(_sent_queue=in_flight)

    async def acknowledge() -> None:
        await asyncio.sleep(0.15)
        in_flight.clear()

    acked = asyncio.create_task(acknowledge())
    started = time.monotonic()
    # A limit far away: returning before it is returning because the messages were acknowledged.
    await asyncio.wait_for(_drain_data_channel(pc, _channel(), timeout=60), 30)

    assert acked.done() and not in_flight
    assert time.monotonic() - started >= 0.1


async def test_gives_up_on_a_peer_that_never_acknowledges() -> None:
    started = time.monotonic()
    # Nothing else ends this wait: that it returns at all is the limit being kept.
    await asyncio.wait_for(
        _drain_data_channel(_pc(_outbound_queue=deque(["x"])), _channel(buffered=10), timeout=0.2), 30
    )
    assert time.monotonic() - started >= 0.2


async def test_a_client_that_left_is_not_waited_for() -> None:
    assert _returns_without_waiting(_drain_data_channel(_pc(_sent_queue=deque(["x"])), _channel(state="closed")))
    assert _returns_without_waiting(_drain_data_channel(_pc(), None))
