"""Blank voice selectors inherit defaults rather than preventing startup."""

from types import SimpleNamespace

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.voice import build_voice_session, merge_client_voice_overrides, merge_voice_config
from timbal.voice.config import DEFAULT_VOICE_ID, VoiceConfig
from timbal.voice.session import VoiceSession


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


@pytest.mark.parametrize("beat", [-1, "-1", "oops", "", "nan", "inf", float("nan"), float("inf"), [], {}, True])
def test_invalid_client_tool_beat_preserves_server_default(beat):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    defaults = VoiceConfig(turn_detector="heuristic", tool_beat_secs=0.4)
    session, _ = build_voice_session(agent, defaults, {"tool_beat_secs": beat})
    assert session.tool_beat_secs == 0.4


def test_tts_pace_defaults_are_off():
    cfg = VoiceConfig()
    assert cfg.max_tts_pause_secs is None
    assert cfg.tts_quiet_dbfs == -45.0
    assert cfg.tts_tempo is None


def test_tts_pace_reaches_the_session_from_agent_config():
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    agent.voice_config = {
        "max_tts_pause_secs": 0.2,
        "tts_quiet_dbfs": -40,
        "tts_tempo": 1.1,
        "turn_detector": "heuristic",
    }
    merged = merge_voice_config(agent)
    assert merged.max_tts_pause_secs == 0.2
    assert merged.tts_quiet_dbfs == -40
    assert merged.tts_tempo == 1.1
    session, _ = build_voice_session(agent, merged, {})
    assert session.max_tts_pause_secs == 0.2
    assert session.tts_quiet_dbfs == -40.0
    assert session.tts_tempo == 1.1


@pytest.mark.parametrize(
    "field,raw,expected",
    [
        ("max_tts_pause_secs", "0.15", 0.15),
        ("tts_quiet_dbfs", "-42", -42.0),
        ("tts_tempo", "1.25", 1.25),
    ],
)
def test_numeric_client_tts_pace_is_coerced(field, raw, expected):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    defaults = VoiceConfig(turn_detector="heuristic", max_tts_pause_secs=0.4, tts_tempo=1.1, tts_quiet_dbfs=-45)
    session, _ = build_voice_session(agent, defaults, {field: raw})
    assert getattr(session, field) == expected


@pytest.mark.parametrize("field", ["max_tts_pause_secs", "tts_quiet_dbfs", "tts_tempo"])
@pytest.mark.parametrize("raw", ["nope", "", "nan", True, -1])
def test_invalid_client_tts_pace_keeps_the_server_value(field, raw):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    defaults = VoiceConfig(turn_detector="heuristic", max_tts_pause_secs=0.4, tts_tempo=1.1, tts_quiet_dbfs=-40)
    # -1 is legal for tts_quiet_dbfs (a level, not a duration). Skip that pair.
    if field == "tts_quiet_dbfs" and raw == -1:
        session, _ = build_voice_session(agent, defaults, {field: raw})
        assert session.tts_quiet_dbfs == -1.0
        return
    if field == "tts_tempo" and raw == -1:
        session, _ = build_voice_session(agent, defaults, {field: raw})
        assert session.tts_tempo == 1.1
        return
    session, _ = build_voice_session(agent, defaults, {field: raw})
    assert getattr(session, field) == getattr(defaults, field)


def test_voice_session_rejects_bad_pace_values():
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))

    def _make(**kwargs):
        return VoiceSession(
            agent=agent,
            stt=object(),
            tts=object(),
            turn_detector="heuristic",
            **kwargs,
        )

    with pytest.raises(ValueError):
        _make(max_tts_pause_secs=-0.1)
    with pytest.raises(ValueError):
        _make(tts_tempo=0)
    with pytest.raises(ValueError):
        _make(tts_quiet_dbfs=float("nan"))
    session = _make()
    assert session.max_tts_pause_secs is None
    assert session.tts_tempo is None
    assert session.tts_quiet_dbfs == -45.0


def test_hangup_grace_defaults_off():
    assert VoiceConfig().hangup_grace_secs is None
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    session, _ = build_voice_session(agent, VoiceConfig(turn_detector="heuristic"), {})
    assert session.hangup_grace_secs == 0


@pytest.mark.parametrize("grace", [0, 1.5])
def test_hangup_grace_reaches_voice_session(grace):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    session, _ = build_voice_session(
        agent,
        VoiceConfig(turn_detector="heuristic", hangup_grace_secs=grace),
        {},
    )
    assert session.hangup_grace_secs == grace


@pytest.mark.parametrize("grace", [-1, 31, float("nan"), float("inf"), True])
def test_invalid_hangup_grace_is_rejected(grace):
    with pytest.raises(ValueError):
        VoiceConfig(hangup_grace_secs=grace)


@pytest.mark.parametrize("grace", ["1.5", "0", " 0.25 "])
def test_numeric_client_hangup_grace_is_coerced(grace):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    defaults = VoiceConfig(turn_detector="heuristic", hangup_grace_secs=0.4)
    merged = merge_client_voice_overrides(defaults, {"hangup_grace_secs": grace})
    assert merged.hangup_grace_secs == float(grace)
    session, _ = build_voice_session(agent, defaults, {"hangup_grace_secs": grace})
    assert session.hangup_grace_secs == float(grace)


@pytest.mark.parametrize(
    "grace", [-1, "-1", "oops", "", "nan", "inf", "31", float("nan"), float("inf"), [], {}, True]
)
def test_invalid_client_hangup_grace_preserves_server_default(grace):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    defaults = VoiceConfig(turn_detector="heuristic", hangup_grace_secs=0.4)
    session, _ = build_voice_session(agent, defaults, {"hangup_grace_secs": grace})
    assert session.hangup_grace_secs == 0.4


def _reply_session(hello: dict):
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    defaults = VoiceConfig(turn_detector="heuristic", language="en", reply_language="ca")
    return build_voice_session(agent, defaults, hello)[0]


@pytest.mark.parametrize("bad", ["english", "en-", "say it in english", True, 1, "", "   ", None])
def test_invalid_client_reply_language_keeps_the_server_pin(bad):
    session = _reply_session({} if bad is None else {"reply_language": bad})
    assert session.reply_language == "ca"
    assert session.audio_input.language == "en"
    assert session.effective_reply_language() == "ca"


def test_client_follow_and_tag_apply_without_moving_each_other():
    followed = _reply_session({"reply_language": " FOLLOW "})
    assert followed.reply_language == "follow"
    assert followed.audio_input.language == "en"
    assert followed.effective_reply_language() is None

    pinned = _reply_session({"reply_language": "es-ES", "language": "en"})
    assert pinned.reply_language == "es-ES"
    assert pinned.audio_input.language == "en"
    assert pinned.effective_reply_language() == "es-ES"


async def test_set_language_does_not_unlock_an_explicit_reply_policy():
    session = _reply_session({"reply_language": "follow"})
    await session.set_language("es")
    assert session.audio_input.language == "es"
    assert session.reply_language == "follow"
    assert session.effective_reply_language() is None


def test_client_hello_cannot_enable_speculative_generation():
    agent = Agent(name="voice_test", model=TestModel(responses=["hi"]))
    session, _ = build_voice_session(
        agent,
        VoiceConfig(turn_detector="heuristic"),
        {"speculative_generation": {"enabled": True, "preemptive_tts": True}},
    )
    assert session.speculative_generation.enabled is False
