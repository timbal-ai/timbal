"""Unit tests for LiveKit SIP helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from timbal.server.livekit_sip import (
    CallerDisconnectAction,
    caller_disconnect_action,
    dtmf_code,
    find_eligible_caller,
    is_eligible_caller,
    phone_tuned_voice_config,
    sip_call_context,
    sip_recording_meta,
)


def _p(
    identity: str,
    *,
    kind: str | int = "PARTICIPANT_KIND_STANDARD",
    reason: str | None = None,
    attributes: dict[str, str] | None = None,
) -> SimpleNamespace:
    disc = SimpleNamespace(name=reason) if reason else None
    return SimpleNamespace(
        identity=identity,
        kind=kind,
        disconnect_reason=disc,
        attributes=attributes or {},
    )


class TestEligibleCaller:
    def test_standard_remote_is_eligible(self) -> None:
        assert is_eligible_caller(_p("caller-abc"), local_identity="agent-1")

    def test_sip_remote_is_eligible(self) -> None:
        assert is_eligible_caller(
            _p("+34600111222", kind="PARTICIPANT_KIND_SIP"),
            local_identity="agent-1",
        )

    def test_self_is_not_eligible(self) -> None:
        assert not is_eligible_caller(_p("agent-1"), local_identity="agent-1")

    def test_egress_ingress_excluded(self) -> None:
        assert not is_eligible_caller(
            _p("recorder", kind="PARTICIPANT_KIND_EGRESS"), local_identity="agent-1"
        )
        assert not is_eligible_caller(
            _p("ingress", kind="PARTICIPANT_KIND_INGRESS"), local_identity="agent-1"
        )

    def test_caller_hint_does_not_gate_match(self) -> None:
        assert is_eligible_caller(
            _p("not-a-prefix"),
            local_identity="agent-1",
            caller_hint="caller-",
        )

    def test_proto_int_kind_is_eligible(self) -> None:
        """The Python FFI exposes STANDARD=0 / SIP=3, not the enum name."""
        assert is_eligible_caller(_p("caller-abc", kind=0), local_identity="agent-1")
        assert is_eligible_caller(_p("+34111", kind=3), local_identity="agent-1")

    def test_proto_int_service_and_agent_are_excluded(self) -> None:
        assert not is_eligible_caller(_p("rec", kind=2), local_identity="agent-1")  # EGRESS
        assert not is_eligible_caller(_p("ing", kind=1), local_identity="agent-1")  # INGRESS
        assert not is_eligible_caller(_p("bot", kind=4), local_identity="agent-1")  # AGENT

    def test_first_eligible_wins(self) -> None:
        remotes = [
            _p("agent-1"),
            _p("recorder", kind="PARTICIPANT_KIND_EGRESS"),
            _p("human", kind="PARTICIPANT_KIND_SIP"),
        ]
        found = find_eligible_caller(remotes, local_identity="agent-1")
        assert found is not None and found.identity == "human"


class TestSipMetadata:
    def test_call_context_maps_sip_attributes(self) -> None:
        ctx = sip_call_context(
            {
                "sip.callID": "abc",
                "sip.phoneNumber": "+34111",
                "sip.trunkPhoneNumber": "+34999",
                "sip.callStatus": "active",
            }
        )
        assert ctx == {
            "call_id": "abc",
            "from": "+34111",
            "to": "+34999",
            "sip_call_status": "active",
        }

    def test_recording_meta(self) -> None:
        meta = sip_recording_meta({"sip.callID": "x", "sip.phoneNumber": "+1"})
        assert meta["transport_detail"] == "livekit_sip"
        assert meta["sip_call_id"] == "x"

    def test_recording_meta_ignores_non_sip_attributes(self) -> None:
        assert sip_recording_meta({"lk.theme": "dark", "user.id": "u1"}) == {}
        assert sip_recording_meta({}) == {}
        assert sip_recording_meta(None) == {}


class TestPhoneTunedConfig:
    def test_explicit_stt_extra_wins(self) -> None:
        out = phone_tuned_voice_config({"stt_extra": {"vad_threshold": 0.9}})
        assert out["stt_extra"]["vad_threshold"] == 0.9
        assert out["stt_extra"]["min_speech_duration_ms"] == 150

    def test_overlay_fills_missing_keys(self) -> None:
        out = phone_tuned_voice_config({})
        assert out["stt_extra"]["vad_threshold"] == 0.55


class TestDisconnectPolicy:
    def test_browser_uses_abandon(self) -> None:
        assert caller_disconnect_action(_p("u")) is CallerDisconnectAction.ABANDON

    def test_sip_bye_closes_immediately(self) -> None:
        assert (
            caller_disconnect_action(
                _p("sip", kind="PARTICIPANT_KIND_SIP", reason="CLIENT_INITIATED")
            )
            is CallerDisconnectAction.CLOSE
        )

    def test_sip_media_blip_short_abandon(self) -> None:
        assert (
            caller_disconnect_action(
                _p("sip", kind="PARTICIPANT_KIND_SIP", reason="STATE_MISMATCH")
            )
            is CallerDisconnectAction.SHORT_ABANDON
        )


class TestDtmfCode:
    def test_digits(self) -> None:
        assert dtmf_code("5") == 5
        assert dtmf_code("#") == 11

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            dtmf_code("Q")
