"""LiveKit call control client + tools."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from timbal.server.livekit_call_control import (
    LivekitCallControl,
    livekit_call_control_tools,
    with_call_tools,
)


@pytest.mark.asyncio
async def test_send_dtmf_publishes_via_local_participant() -> None:
    published: list[tuple[int, str]] = []

    async def _pub(*, code: int, digit: str) -> None:
        published.append((code, digit))

    room = SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=_pub))
    ctrl = LivekitCallControl(
        room=room,
        room_name="r1",
        caller_identity="sip-user",
        is_sip=True,
    )
    await ctrl.send_dtmf("3")
    assert published == [(3, "3")]


@pytest.mark.asyncio
async def test_transfer_posts_to_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: dict = {}

    async def _req(url: str, token: str, body: dict) -> None:
        posted["url"] = url
        posted["token"] = token
        posted["json"] = body

    monkeypatch.setenv("TIMBAL_VOICE_CALL_CONTROL_URL", "https://api.test/voice/calls/{call_id}")
    monkeypatch.setenv("TIMBAL_VOICE_CALL_CONTROL_TOKEN", "jwt-call-9")
    monkeypatch.setattr("timbal.server.livekit_call_control._call_control_request", _req)

    ctrl = LivekitCallControl(
        room=SimpleNamespace(local_participant=SimpleNamespace()),
        room_name="room-1",
        caller_identity="sip-1",
        call_id="call-9",
        is_sip=True,
    )
    await ctrl.transfer_call("tel:+15550100")
    assert posted["url"] == "https://api.test/voice/calls/call-9/transfer"
    assert posted["token"] == "jwt-call-9"
    assert posted["json"]["transfer_to"] == "tel:+15550100"
    assert posted["json"]["participant_identity"] == "sip-1"


@pytest.mark.asyncio
async def test_transfer_requires_a_call_id(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: dict = {}

    async def _req(url: str, _token: str, _body: dict) -> None:
        posted["url"] = url

    monkeypatch.setenv("TIMBAL_VOICE_CALL_CONTROL_URL", "https://api.test/voice/calls/{call_id}")
    monkeypatch.setenv("TIMBAL_VOICE_CALL_CONTROL_TOKEN", "jwt-call-9")
    monkeypatch.delenv("TIMBAL_VOICE_CALL_ID", raising=False)
    monkeypatch.setattr("timbal.server.livekit_call_control._call_control_request", _req)

    ctrl = LivekitCallControl(
        room=SimpleNamespace(local_participant=SimpleNamespace()),
        room_name="room-1",
        caller_identity="sip-1",
        call_id="",
        is_sip=True,
    )
    with pytest.raises(RuntimeError, match="call id"):
        await ctrl.transfer_call("tel:+15550100")
    assert posted == {}


def test_tools_only_for_sip() -> None:
    assert livekit_call_control_tools(None) == []
    browser = LivekitCallControl(
        room=SimpleNamespace(local_participant=SimpleNamespace()),
        room_name="r",
        caller_identity="u",
        is_sip=False,
    )
    assert livekit_call_control_tools(browser) == []
    sip = LivekitCallControl(
        room=SimpleNamespace(local_participant=SimpleNamespace()),
        room_name="r",
        caller_identity="sip",
        is_sip=True,
    )
    names = {t.name for t in livekit_call_control_tools(sip)}
    assert names == {"send_dtmf", "transfer_call"}


def test_call_tools_copy_process_wide_agent() -> None:
    original_tool = SimpleNamespace(name="existing")

    class CopyableAgent:
        def __init__(self) -> None:
            self.tools = [original_tool]

        def model_copy(self, *, update: dict) -> SimpleNamespace:
            return SimpleNamespace(tools=update["tools"])

    agent = CopyableAgent()
    call_tool = SimpleNamespace(name="transfer_call")
    copied = with_call_tools(agent, [call_tool])

    assert agent.tools == [original_tool]
    assert copied.tools == [original_tool, call_tool]
