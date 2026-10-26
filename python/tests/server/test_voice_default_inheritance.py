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


@pytest.mark.parametrize("beat", [None, 0, 0.25])
def test_tool_beat_reaches_voice_session(beat):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    session, _ = build_voice_session(agent, VoiceConfig(turn_detector="heuristic"), {"tool_beat_secs": beat})
    assert session.tool_beat_secs == beat


@pytest.mark.parametrize("beat", [-1, float("nan"), float("inf")])
def test_invalid_tool_beat_is_rejected(beat):
    with pytest.raises(ValueError):
        VoiceConfig(tool_beat_secs=beat)


@pytest.mark.parametrize("beat", ["0.6", "0", " 0.25 "])
def test_numeric_client_tool_beat_is_coerced(beat):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    defaults = VoiceConfig(turn_detector="heuristic", tool_beat_secs=0.4)
    merged = merge_client_voice_overrides(defaults, {"tool_beat_secs": beat})
    assert merged.tool_beat_secs == float(beat)
    session, _ = build_voice_session(agent, defaults, {"tool_beat_secs": beat})
    assert session.tool_beat_secs == float(beat)


@pytest.mark.parametrize(
    "beat", [-1, "-1", "oops", "", "nan", "inf", float("nan"), float("inf"), [], {}, True]
)
def test_invalid_client_tool_beat_preserves_server_default(beat):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    defaults = VoiceConfig(turn_detector="heuristic", tool_beat_secs=0.4)
    session, _ = build_voice_session(agent, defaults, {"tool_beat_secs": beat})
    assert session.tool_beat_secs == 0.4


def test_resumed_window_is_server_only_and_reaches_native_provider(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-not-a-real-key")
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    defaults = VoiceConfig(merge_resumed_secs=2)
    session, _ = build_voice_session(agent, defaults, {"merge_resumed_secs": 30})
    assert session.turn_detector.merge_resumed_secs == 2
    assert session._vad_endpointing is False


def test_resumed_window_does_not_mutate_shared_detector():
    from timbal.voice.turn_detection import ProviderTurnDetector

    detector = ProviderTurnDetector()
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    session, _ = build_voice_session(agent, VoiceConfig(turn_detector=detector, merge_resumed_secs=2), {})
    assert session.turn_detector.merge_resumed_secs == 2
    assert detector.merge_resumed_secs is None


def test_resumed_window_is_ignored_when_resolved_detector_is_not_provider(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    # Fallback STT (ElevenLabs Scribe) with default turn_detector=None resolves to local/lexical.
    session_fallback, _ = build_voice_session(agent, VoiceConfig(merge_resumed_secs=2), {})
    assert getattr(session_fallback.turn_detector, "merge_resumed_secs", None) is None
    # Client playground detector override ("heuristic") on a server config with merge_resumed_secs.
    defaults = VoiceConfig(turn_detector="provider", merge_resumed_secs=2)
    session_override, _ = build_voice_session(agent, defaults, {"turn_detector": "heuristic"})
    assert getattr(session_override.turn_detector, "merge_resumed_secs", None) is None

