"""Blank voice selectors inherit defaults rather than preventing startup."""
from types import SimpleNamespace

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.voice import build_voice_session, merge_client_voice_overrides, merge_voice_config
from timbal.voice.config import DEFAULT_VOICE_ID, VoiceConfig


@pytest.mark.parametrize("voice", [None, "", "  ", False, 42, {}])
@pytest.mark.parametrize("provider,configured", [("elevenlabs", "custom-elevenlabs"), ("munsit", "custom-munsit")])
def test_empty_client_voice_preserves_configured_provider_voice(voice, provider, configured):
    defaults = VoiceConfig(tts_provider=provider, voice=configured)
    merged = merge_client_voice_overrides(defaults, {"voice": voice})
    assert merged.voice == configured
    assert merged.tts_provider == provider


def test_explicit_voice_is_trimmed_and_preserved():
    merged = merge_client_voice_overrides(VoiceConfig(voice="old"), {"voice": " chosen "})
    assert merged.voice == "chosen"


@pytest.mark.parametrize("voice", ["", "  "])
def test_empty_agent_voice_inherits_operator_voice(monkeypatch, voice):
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "operator-voice")
    monkeypatch.delenv("TIMBAL_VOICE_ID", raising=False)
    assert merge_voice_config(SimpleNamespace(voice_config={"voice": voice})).voice == "operator-voice"


@pytest.mark.parametrize("voice", [None, "", "  ", False])
def test_empty_platform_voice_builds_a_session_with_elevenlabs_default(voice):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]), tools=[])
    defaults = VoiceConfig(turn_detector="heuristic").model_copy(update={"voice": voice})
    session, _ = build_voice_session(agent, defaults, {"voice": ""})
    assert session.audio_output.voice == DEFAULT_VOICE_ID


def test_explicit_client_voice_wins_over_empty_platform_voice():
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]), tools=[])
    defaults = VoiceConfig(turn_detector="heuristic").model_copy(update={"voice": ""})
    session, _ = build_voice_session(agent, defaults, {"voice": "chosen"})
    assert session.audio_output.voice == "chosen"
