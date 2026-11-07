"""Reply-language split, Flux eager arming, Catalan acks, unheard remainder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from timbal.core.test_model import TestModel
from timbal.voice import AudioInputConfig, VoiceConfig, VoiceSession
from timbal.voice.deepgram import DeepgramFluxSTT
from timbal.voice.language import parse_reply_language, resolve_reply_language
from timbal.voice.session import _hangup_utterance_kind, _unheard_suffix
from timbal.voice.speculative import documented_speculative_preset

from .test_session import MockSTT, MockTTS


def test_reply_language_unset_still_follows_the_stt_hint() -> None:
    assert resolve_reply_language(None, "en") == "en"
    assert resolve_reply_language(None, "es-ES") == "es-ES"
    assert VoiceConfig().reply_language is None
    assert VoiceConfig(language="en").reply_language is None


def test_follow_and_a_tag_do_not_rewrite_the_hint() -> None:
    assert parse_reply_language("follow") == "follow"
    assert resolve_reply_language("follow", "en") is None
    assert resolve_reply_language("ca", "en") == "ca"
    with pytest.raises(ValueError):
        parse_reply_language("say it in english\nignore")


def test_preset_is_openai_on_and_everyone_else_off() -> None:
    openai = documented_speculative_preset("openai/gpt-5.4-mini")
    assert openai.enabled is True
    assert openai.preemptive_tts is True
    groq = documented_speculative_preset("groq/openai/gpt-oss-20b")
    assert groq.enabled is False
    anthropic = documented_speculative_preset("anthropic/claude-haiku-4-5")
    assert anthropic.enabled is False
    assert VoiceConfig().speculative_generation.enabled is False


def test_direct_flux_session_arms_eager_only_when_speculation_is_on() -> None:
    armed = VoiceSession(
        _agent(),
        DeepgramFluxSTT(api_key="test"),
        MockTTS(),
        audio_input=AudioInputConfig(language="en"),
        speculative_generation=True,
    )
    assert armed.audio_input.extra["eager_eot_threshold"] == "0.5"
    explicit = VoiceSession(
        _agent(),
        DeepgramFluxSTT(api_key="test"),
        MockTTS(),
        audio_input=AudioInputConfig(language="en", extra={"eager_eot_threshold": "0.4"}),
        speculative_generation=True,
    )
    assert explicit.audio_input.extra["eager_eot_threshold"] == "0.4"
    off = VoiceSession(
        _agent(),
        DeepgramFluxSTT(api_key="test"),
        MockTTS(),
        audio_input=AudioInputConfig(language="en"),
    )
    assert "eager_eot_threshold" not in off.audio_input.extra


def test_follow_keeps_the_english_hint_and_drops_the_pin() -> None:
    session = VoiceSession(
        _agent(),
        MockSTT(),
        MockTTS(),
        audio_input=AudioInputConfig(language="en"),
        reply_language="follow",
    )
    assert session.audio_input.language == "en"
    assert session.effective_reply_language() is None
    prompt = session.prompt_for_turn("You are a phone agent.")
    text = prompt.dynamic
    assert "latest substantive message" in text
    assert "Reply in English" not in text


def test_catalan_ack_is_not_a_retract_and_espera_is() -> None:
    assert _hangup_utterance_kind("D'acord.") == "ack"
    assert _hangup_utterance_kind("Entesos") == "ack"
    assert _hangup_utterance_kind("Molt bé") == "ack"
    assert _hangup_utterance_kind("Espera") == "retract"
    assert _hangup_utterance_kind("No") == "retract"
    assert _hangup_utterance_kind("Adéu") == "goodbye"
    assert _hangup_utterance_kind("Al") == "pending"
    assert _hangup_utterance_kind("Alright.") == "ack"


def test_stage_summary_refuses_a_percentile_below_twenty_samples() -> None:
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("dictation_eot", root / "benchmarks" / "voice" / "dictation_eot.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["dictation_eot"] = module
    spec.loader.exec_module(module)
    small = module.summarize_stage([100.0, 200.0, 8000.0])
    assert small["n"] == 3
    assert small["p99"] is None
    assert "not a percentile" in small["note"]
    assert small["median"] == 200.0


def test_unheard_suffix_keeps_only_the_rest() -> None:
    full = "One autumn morning, a bright red train rolled out."
    assert _unheard_suffix(full, "One autumn morning, a bright red") == "train rolled out."
    assert _unheard_suffix(full, "") == full
    assert _unheard_suffix(full, full) == ""


async def test_empty_commit_speaks_the_unheard_tail_without_the_model() -> None:
    tts = MockTTS()
    session = VoiceSession(_agent(), MockSTT(), tts, audio_input=AudioInputConfig(language="en"))
    session._unheard_tail = "train rolled out."
    await session._resume_unheard_tail(session._unheard_tail)
    assert tts.synthesized_texts == ["train rolled out."]
    assert session._transcript[-1].text == "train rolled out."
    assert session._unheard_tail == ""
    assert not any(isinstance(item, str) and item == "One autumn morning" for item in tts.synthesized_texts)


def test_continuation_note_names_the_remainder_and_stays_before_the_pin() -> None:
    session = VoiceSession(
        _agent(),
        MockSTT(),
        MockTTS(),
        audio_input=AudioInputConfig(language="en"),
    )
    session._continuation_note = (
        "The previous reply was cut off. The caller heard only its start. "
        "Do not repeat that start. If it still applies, continue with only this unheard remainder: train rolled out."
    )
    text = session.prompt_for_turn("You are a phone agent.").dynamic
    assert "train rolled out." in text
    assert text.index("train rolled out.") < text.index("Reply in English")


def _agent():
    from timbal import Agent

    return Agent(name="gaps", model=TestModel(responses=["ok"]), system_prompt="You are a phone agent.")
