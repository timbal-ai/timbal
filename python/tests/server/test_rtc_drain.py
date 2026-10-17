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


async def test_nothing_waiting_returns_at_once() -> None:
    started = time.monotonic()
    await _drain_data_channel(_pc(), _channel())
    assert time.monotonic() - started < 0.05


async def test_waits_for_the_last_messages_to_be_acknowledged() -> None:
    in_flight: deque = deque(["session_transcript", "session_ended"])
    pc = _pc(_sent_queue=in_flight)

    async def acknowledge() -> None:
        await asyncio.sleep(0.15)
        in_flight.clear()

    acked = asyncio.create_task(acknowledge())
    started = time.monotonic()
    await _drain_data_channel(pc, _channel())
    await acked

    assert 0.1 <= time.monotonic() - started < 1.0


async def test_gives_up_on_a_peer_that_never_acknowledges() -> None:
    started = time.monotonic()
    await _drain_data_channel(_pc(_outbound_queue=deque(["x"])), _channel(buffered=10), timeout=0.2)
    assert 0.2 <= time.monotonic() - started < 0.6


async def test_a_client_that_left_is_not_waited_for() -> None:
    started = time.monotonic()
    await _drain_data_channel(_pc(_sent_queue=deque(["x"])), _channel(state="closed"))
    await _drain_data_channel(_pc(), None)
    assert time.monotonic() - started < 0.05
