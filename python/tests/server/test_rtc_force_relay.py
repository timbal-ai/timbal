"""Tests for ``TIMBAL_VOICE_RTC_FORCE_RELAY`` — relay-only ICE for private-subnet boxes.

Pure SDP/env logic; only the ``_ice_servers`` test needs aiortc.
"""

from __future__ import annotations

import pytest
from timbal.server.rtc import _force_relay, _ice_servers, _strip_non_relay_candidates

from .voice_env import VOICE_ENV_KEYS

_HOST = "a=candidate:6815297761 1 udp 2130706431 10.0.1.5 40000 typ host"
_SRFLX = "a=candidate:2932157868 1 udp 1694498815 34.1.2.3 40001 typ srflx raddr 10.0.1.5 rport 40000"
_RELAY = "a=candidate:3456 1 udp 25108223 52.4.5.6 50000 typ relay raddr 34.1.2.3 rport 40001"


def _sdp(*candidates: str) -> str:
    return "\r\n".join(
        [
            "v=0",
            "o=- 1 1 IN IP4 0.0.0.0",
            "s=-",
            "t=0 0",
            "m=audio 40000 UDP/TLS/RTP/SAVPF 96",
            "c=IN IP4 10.0.1.5",
            *candidates,
            "a=sendrecv",
            "",
        ]
    )


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in VOICE_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("TIMBAL_STUN_URL", raising=False)


class TestStripNonRelayCandidates:
    def test_keeps_only_relay_candidates(self) -> None:
        out = _strip_non_relay_candidates(_sdp(_HOST, _SRFLX, _RELAY))
        assert _RELAY in out
        assert "typ host" not in out
        assert "typ srflx" not in out
        # Everything else survives untouched, CRLF included.
        assert out == _sdp(_RELAY)

    def test_degrades_to_original_when_no_relay_candidate(self) -> None:
        """A failed TURN allocation must not produce an unconnectable answer."""
        sdp = _sdp(_HOST, _SRFLX)
        assert _strip_non_relay_candidates(sdp) == sdp

    def test_noop_on_relay_only_sdp(self) -> None:
        sdp = _sdp(_RELAY)
        assert _strip_non_relay_candidates(sdp) == sdp


class TestForceRelayFlag:
    def test_off_by_default(self) -> None:
        assert not _force_relay()

    def test_requires_turn_to_be_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_RTC_FORCE_RELAY", "1")
        assert not _force_relay()  # degrades loudly instead of answering with nothing

        monkeypatch.setenv("TIMBAL_TURN_URL", "turn:turn.timbal.ai:3478")
        assert _force_relay()


class TestIceServers:
    def test_relay_only_returns_just_the_turn_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("aiortc", reason="timbal[voice] extra (aiortc) not installed")
        monkeypatch.setenv("TIMBAL_TURN_URL", "turn:turn.timbal.ai:3478")
        monkeypatch.setenv("TIMBAL_TURN_USERNAME", "user")
        monkeypatch.setenv("TIMBAL_TURN_PASSWORD", "secret")

        servers = _ice_servers(relay_only=True)
        assert len(servers) == 1
        assert servers[0].urls == "turn:turn.timbal.ai:3478"
        assert servers[0].username == "user"
        assert servers[0].credential == "secret"

        # Without relay_only the default STUN server is included too.
        assert len(_ice_servers()) == 2
