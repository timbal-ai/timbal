"""The local Norte Market demo: simulated shop, real voice config shape."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("voice_support_demo", ROOT / "examples" / "voice_support_demo.py")
assert _SPEC and _SPEC.loader
demo = importlib.util.module_from_spec(_SPEC)
sys.modules["voice_support_demo"] = demo
_SPEC.loader.exec_module(demo)


def _use(session_id: str):
    from timbal.voice.session import _current_voice_session

    class _Session:
        pass

    holder = _Session()
    holder.session_id = session_id
    return _current_voice_session.set(holder)


@pytest.fixture(autouse=True)
def _clean_store():
    demo.reset_demo_store()
    yield
    demo.reset_demo_store()


def test_config_is_spanish_hint_and_follow_reply() -> None:
    cfg = demo.demo_voice_config("openai/gpt-5.4-mini")
    assert cfg["language"] == "es"
    assert cfg["reply_language"] == "follow"
    assert cfg["entity_patience"] is True
    assert cfg["hangup_grace_secs"] == 1.2
    assert cfg["filler"]["message"] == "Un momento."
    assert cfg["speculative_generation"]["enabled"] is True
    assert cfg["speculative_generation"]["preemptive_tts"] is True
    other = demo.demo_voice_config("openai/gpt-5-mini")
    assert other["speculative_generation"]["enabled"] is False
    assert "voice" not in cfg
    names = [tool.name for tool in demo.agent.tools]
    assert names == [
        "lookup_order",
        "lookup_customer",
        "track_shipment",
        "stage_delivery_change",
        "stage_support_ticket",
        "apply_pending",
        "cancel_pending",
        "end_call",
    ]
    for tool in demo.agent.tools:
        assert (tool.handler.__doc__ or "").strip()
        schema = tool.params_model.model_json_schema()
        assert schema["type"] == "object"
    order_schema = demo.agent.tools[0].params_model.model_json_schema()
    assert "order_id" in order_schema["properties"]
    assert "order_id" in order_schema["required"]


async def test_unknown_slow_and_retry_stay_inside_one_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo, "SLOW_SECS", 0)
    monkeypatch.setattr(demo, "FAST_SECS", 0)
    token = _use("call-a")
    try:
        assert "No simulated order 9999" in await demo.lookup_order("9999")
        assert "Ana Ruiz" in await demo.lookup_order("1001")
        failed = await demo.lookup_order("4040")
        assert "failed" in failed
        recovered = await demo.lookup_order("4040")
        assert "Pol Serra" in recovered
        assert "NM1001" in await demo.track_shipment("1001")
        assert "Ana Ruiz" in await demo.lookup_customer("ana ruiz")
        assert "No simulated customer" in await demo.lookup_customer("nobody")
    finally:
        from timbal.voice.session import _current_voice_session

        _current_voice_session.reset(token)


async def test_writes_wait_for_apply_and_do_not_cross_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo, "FAST_SECS", 0)
    from timbal.voice.session import _current_voice_session

    first = _use("call-a")
    try:
        assert "Nothing is staged" in await demo.apply_pending()
        staged = await demo.stage_delivery_change("1001", "Ring twice")
        assert "not saved" in staged
        assert "Leave it with the neighbour" in await demo.lookup_order("1001")
        saved = await demo.apply_pending()
        assert "Saved in this call only" in saved
        assert "Ring twice" in await demo.lookup_order("1001")
        ticket = await demo.stage_support_ticket("1003", "Box arrived open")
        assert "not opened" in ticket
        await demo.cancel_pending()
        assert demo._bag()["tickets"] == []
    finally:
        _current_voice_session.reset(first)

    second = _use("call-b")
    try:
        assert "Leave it with the neighbour" in await demo.lookup_order("1001")
        assert "Nothing is staged" in await demo.apply_pending()
    finally:
        _current_voice_session.reset(second)
