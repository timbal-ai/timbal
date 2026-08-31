"""SIP / PSTN helpers for :mod:`timbal.server.livekit_session`.

Participant resolution, phone-tuned config overlays, disconnect policy, and
SIP attribute extraction. Kept separate from the driver so the rules are
testable without the FFI extra.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Literal

import structlog

logger = structlog.get_logger("timbal.server.livekit_sip")

# LiveKit ``ParticipantKind`` wire names (Python SDK enum suffixes).
KIND_STANDARD = "PARTICIPANT_KIND_STANDARD"
KIND_SIP = "PARTICIPANT_KIND_SIP"
KIND_EGRESS = "PARTICIPANT_KIND_EGRESS"
KIND_INGRESS = "PARTICIPANT_KIND_INGRESS"

SERVICE_KINDS = frozenset({KIND_EGRESS, KIND_INGRESS, "EGRESS", "INGRESS"})
CALLER_KINDS = frozenset({KIND_STANDARD, KIND_SIP, "STANDARD", "SIP"})

# §7.1 — definitive inbound BYE / room teardown vs media blip.
DEFINITIVE_DISCONNECT = frozenset(
    {
        "CLIENT_INITIATED",
        "ROOM_DELETED",
        "DisconnectReason.CLIENT_INITIATED",
        "DisconnectReason.ROOM_DELETED",
    }
)

SIP_ATTR_CALL_ID = "sip.callID"
SIP_ATTR_PHONE = "sip.phoneNumber"
SIP_ATTR_TRUNK_PHONE = "sip.trunkPhoneNumber"
SIP_ATTR_CALL_STATUS = "sip.callStatus"

# DTMF RFC 2833 event codes (0-9, *, #, A-D).
_DTMF_CODES: dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "*": 10,
    "#": 11,
    "A": 12,
    "B": 13,
    "C": 14,
    "D": 15,
}

# Overlay only — explicit client / env config wins (§3.1 corollary).
_PHONE_TUNED_STT_EXTRA: dict[str, Any] = {
    "vad_threshold": 0.55,
    "vad_silence_threshold_secs": 1.5,
    "min_speech_duration_ms": 150,
}

_SIP_ABANDON_SECS = 10.0


class CallerDisconnectAction(str, Enum):
    """What to do when the human participant leaves."""

    ABANDON = "abandon"  # browser blip — full abandon window
    SHORT_ABANDON = "short_abandon"  # SIP media failure — bounded wait
    CLOSE = "close"  # definitive hangup — end immediately


def _kind_label(kind: Any) -> str:
    if kind is None:
        return ""
    name = getattr(kind, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(kind)


def participant_kind(participant: Any) -> str:
    return _kind_label(getattr(participant, "kind", None))


def is_sip_participant(participant: Any) -> bool:
    return participant_kind(participant) in {KIND_SIP, "SIP"}


def disconnect_reason_label(participant: Any) -> str | None:
    reason = getattr(participant, "disconnect_reason", None)
    if reason is None:
        return None
    name = getattr(reason, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(reason)
    if text.endswith(".UNKNOWN_REASON") or text == "UNKNOWN_REASON":
        return None
    return text.split(".")[-1] if "." in text else text


def is_eligible_caller(
    participant: Any,
    *,
    local_identity: str,
    caller_hint: str = "",
) -> bool:
    """First-remote-participant rule (§3.1): not self, not service, STANDARD/SIP.

    ``caller_hint`` (``caller-`` prefix from the monolith) is for logging /
    dev assertions only — never the match condition.
    """
    identity = getattr(participant, "identity", "") or ""
    if not identity or identity == local_identity:
        return False
    kind = participant_kind(participant)
    if kind in SERVICE_KINDS:
        return False
    if kind not in CALLER_KINDS:
        return False
    if caller_hint and identity.startswith(caller_hint):
        logger.debug("livekit_caller_hint_match", identity=identity, hint=caller_hint)
    return True


def find_eligible_caller(
    remote_participants: Any,
    *,
    local_identity: str,
    caller_hint: str = "",
) -> Any | None:
    """Return the first eligible remote participant, or ``None``."""
    for participant in remote_participants:
        if is_eligible_caller(participant, local_identity=local_identity, caller_hint=caller_hint):
            return participant
    return None


def sip_call_context(attributes: dict[str, str] | None) -> dict[str, str]:
    """Map ``sip.*`` attributes → ``session.call_context`` keys."""
    if not attributes:
        return {}
    ctx: dict[str, str] = {}
    mapping = {
        SIP_ATTR_CALL_ID: "call_id",
        SIP_ATTR_PHONE: "from",
        SIP_ATTR_TRUNK_PHONE: "to",
        SIP_ATTR_CALL_STATUS: "sip_call_status",
    }
    for attr, key in mapping.items():
        val = attributes.get(attr)
        if isinstance(val, str) and val:
            ctx[key] = val
    return ctx


def sip_recording_meta(attributes: dict[str, str] | None) -> dict[str, str]:
    """Extra manifest ``meta`` keys from SIP attributes.

    Returns empty unless at least one ``sip.*`` attribute is present — a
    browser participant with unrelated attributes must not be tagged SIP.
    """
    if not attributes:
        return {}
    out: dict[str, str] = {}
    for attr, key in (
        (SIP_ATTR_CALL_ID, "sip_call_id"),
        (SIP_ATTR_PHONE, "sip_phone_number"),
        (SIP_ATTR_TRUNK_PHONE, "sip_trunk_phone_number"),
    ):
        val = attributes.get(attr)
        if isinstance(val, str) and val:
            out[key] = val
    if out:
        out["transport_detail"] = "livekit_sip"
    return out


def phone_tuned_voice_config(config: dict[str, Any]) -> dict[str, Any]:
    """Overlay PSTN-friendly STT/VAD defaults; explicit keys in ``config`` win."""
    out = dict(config)
    base_extra = out.get("stt_extra")
    merged_extra = dict(_PHONE_TUNED_STT_EXTRA)
    if isinstance(base_extra, dict):
        merged_extra = {**merged_extra, **base_extra}
    out["stt_extra"] = merged_extra
    return out


def caller_disconnect_action(participant: Any) -> CallerDisconnectAction:
    """§7.1 — SIP hangup vs blip vs browser abandon."""
    if not is_sip_participant(participant):
        return CallerDisconnectAction.ABANDON
    reason = disconnect_reason_label(participant)
    if reason is not None and reason in DEFINITIVE_DISCONNECT:
        return CallerDisconnectAction.CLOSE
    if reason is not None:
        return CallerDisconnectAction.SHORT_ABANDON
    return CallerDisconnectAction.SHORT_ABANDON


def sip_abandon_secs() -> float:
    raw = os.environ.get("TIMBAL_VOICE_SIP_ABANDON_SECS", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("livekit_bad_sip_abandon_secs", value=raw)
    return _SIP_ABANDON_SECS


def dtmf_code(digit: str) -> int:
    """RFC 2833 event code for a single DTMF digit."""
    key = digit.strip().upper()
    if key not in _DTMF_CODES:
        raise ValueError(f"invalid DTMF digit {digit!r}")
    return _DTMF_CODES[key]


def dtmf_event_payload(*, digit: str, code: int, identity: str = "") -> dict[str, Any]:
    """Wire payload for an inbound DTMF event (``timbal.events`` topic)."""
    payload: dict[str, Any] = {"type": "dtmf", "digit": digit, "code": code}
    if identity:
        payload["identity"] = identity
    return payload


def call_id_from_env() -> str:
    return os.environ.get("TIMBAL_VOICE_CALL_ID", "").strip()
