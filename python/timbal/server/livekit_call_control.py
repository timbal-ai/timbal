"""Call-scoped LiveKit SIP controls (DTMF + transfer).

``send_dtmf`` uses the in-room ``publish_dtmf`` API when the driver holds a
``room`` reference. ``transfer_call`` delegates to the platform monolith —
the box must not carry LiveKit API secrets (same rule as recording upload).

Monolith contract (to be implemented server-side):

    POST {host}/orgs/{org_id}/projects/{project_id}/voice/calls/{call_id}/transfer
    Authorization: Bearer {token}
    Content-Type: application/json
    {
      "room": "<room name>",
      "participant_identity": "<SIP participant identity>",
      "transfer_to": "tel:+14155550100",   // or sip:user@host
      "play_dialtone": false
    }

    → 204 No Content on success; 4xx/5xx with JSON ``{"error": "..."}`` on failure.

Optional env overrides:

* ``TIMBAL_VOICE_CALL_ID`` — call id for the path segment (falls back to
  ``sip.callID`` attribute or room tail).
* ``TIMBAL_VOICE_CALL_CONTROL_URL`` — call-scoped base URL ending in
  ``/voice/calls/{call_id}``.
* ``TIMBAL_VOICE_CALL_CONTROL_TOKEN`` — call-scoped JWT sent only as
  ``X-Timbal-Sip-Call-Token``.
"""

from __future__ import annotations

from typing import Any

import structlog

from ..core.tool import Tool
from ..platform.utils import _request
from .livekit_sip import call_id_from_env, dtmf_code

logger = structlog.get_logger("timbal.server.livekit_call_control")


class LivekitCallControl:
    """Per-call handles wired by :mod:`timbal.server.livekit_session`."""

    def __init__(
        self,
        *,
        room: Any,
        room_name: str,
        caller_identity: str,
        call_id: str = "",
        is_sip: bool = False,
        call_control_url: str = "",
        call_control_token: str = "",
    ) -> None:
        self._room = room
        self.room_name = room_name
        self.caller_identity = caller_identity
        self.call_id = call_id or call_id_from_env()
        self.is_sip = is_sip
        self._call_control_url = call_control_url or _call_control_url()
        self._call_control_token = call_control_token or _call_control_token()

    async def send_dtmf(self, digit: str) -> None:
        """Send one DTMF tone into the room (SIP leg receives it)."""
        code = dtmf_code(digit)
        local = self._room.local_participant
        publish = getattr(local, "publish_dtmf", None)
        if publish is None:
            raise RuntimeError("livekit publish_dtmf is unavailable in this SDK build")
        await publish(code=code, digit=digit.strip())

    async def transfer_call(
        self,
        transfer_to: str,
        *,
        play_dialtone: bool = False,
    ) -> None:
        """Cold-transfer the SIP caller via the platform (SIP REFER)."""
        if not self.is_sip:
            raise RuntimeError("transfer_call is only available on SIP calls")
        if not self.caller_identity:
            raise RuntimeError("caller identity is not resolved yet")
        url = _control_endpoint(self._call_control_url, self.call_id, "transfer")
        if not url:
            raise RuntimeError("transfer_call requires TIMBAL_VOICE_CALL_CONTROL_URL and a call id")
        if not self._call_control_token:
            raise RuntimeError("transfer_call requires TIMBAL_VOICE_CALL_CONTROL_TOKEN")
        body = {
            "room": self.room_name,
            "participant_identity": self.caller_identity,
            "transfer_to": transfer_to,
            "play_dialtone": play_dialtone,
        }
        await _call_control_request(url, self._call_control_token, body)
        logger.info(
            "livekit_transfer_requested",
            room=self.room_name,
            caller=self.caller_identity,
            transfer_to=transfer_to,
        )


def _call_control_url() -> str:
    import os

    return os.environ.get("TIMBAL_VOICE_CALL_CONTROL_URL", "").strip()


def _call_control_token() -> str:
    import os

    return os.environ.get("TIMBAL_VOICE_CALL_CONTROL_TOKEN", "").strip()


def _control_endpoint(base: str, call_id: str, operation: str) -> str | None:
    if not base or not str(call_id).strip():
        return None
    expanded = base.format(call_id=call_id) if "{call_id}" in base else base
    return f"{expanded.rstrip('/')}/{operation}"


async def _call_control_request(url: str, token: str, body: dict[str, Any]) -> None:
    headers = {"X-Timbal-Sip-Call-Token": token}
    if url.startswith(("http://", "https://")):
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
        return
    await _request("POST", url, headers=headers, json=body)


def livekit_call_control_tools(control: LivekitCallControl | None) -> list[Tool]:
    """Optional agent tools — attach when ``control`` is SIP-capable."""
    if control is None or not control.is_sip:
        return []

    async def _send_dtmf(digit: str) -> str:
        await control.send_dtmf(digit)
        return f"sent DTMF {digit!r}"

    async def _transfer_call(transfer_to: str, play_dialtone: bool = False) -> str:
        await control.transfer_call(transfer_to, play_dialtone=play_dialtone)
        return f"transfer to {transfer_to!r} requested"

    return [
        Tool(
            name="send_dtmf",
            description="Send one DTMF keypad digit to the phone caller (0-9, *, #).",
            handler=_send_dtmf,
        ),
        Tool(
            name="transfer_call",
            description=("Cold-transfer the phone caller to another number or SIP URI (e.g. tel:+14155550100)."),
            handler=_transfer_call,
        ),
    ]


def with_call_tools(agent: Any, tools: list[Tool]) -> Any:
    """Return a per-call Agent copy with call-scoped tools appended.

    The server's runnable is process-wide on deployed workers. Mutating it
    would leak the first call's room and participant handles into later or
    concurrent calls.
    """
    if not tools:
        return agent
    existing = list(getattr(agent, "tools", None) or [])
    names = {getattr(t, "name", None) for t in existing}
    for tool in tools:
        if tool.name not in names:
            existing.append(tool)
    model_copy = getattr(agent, "model_copy", None)
    if not callable(model_copy):
        raise TypeError("call-scoped tools require a copyable Agent")
    return model_copy(update={"tools": existing})
