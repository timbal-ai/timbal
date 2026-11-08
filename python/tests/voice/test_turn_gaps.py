"""Reply-language split, Flux eager arming, Catalan acks, unheard remainder."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path

import pytest
from timbal.core.test_model import TestModel
from timbal.voice import AudioInputConfig, SessionStarted, TranscriptEvent, VoiceConfig, VoiceSession
from timbal.voice.deepgram import DeepgramFluxSTT
from timbal.voice.language import parse_reply_language, resolve_reply_language
from timbal.voice.session import _hangup_utterance_kind, _unheard_suffix, false_interrupt_followup
from timbal.voice.speculative import documented_speculative_preset

from .test_session import DelayedMockSTT, MockSTT, MockTTS


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


def test_preset_is_only_the_measured_openai_model() -> None:
    measured = documented_speculative_preset("openai/gpt-5.4-mini")
    assert measured.enabled is True
    assert measured.preemptive_tts is True
    other_openai = documented_speculative_preset("openai/gpt-5-mini")
    assert other_openai.enabled is False
    groq = documented_speculative_preset("groq/openai/gpt-oss-20b")
    assert groq.enabled is False
    anthropic = documented_speculative_preset("anthropic/claude-haiku-4-5")
    assert anthropic.enabled is False
    assert VoiceConfig().speculative_generation.enabled is False


def test_reply_language_rejects_malformed_values() -> None:
    assert parse_reply_language("  FOLLOW ") == "follow"
    assert parse_reply_language("en-US") == "en-US"
    assert parse_reply_language("es_ES") == "es-ES"
    assert parse_reply_language("zh-Hans") == "zh-Hans"
    assert parse_reply_language("es-419") == "es-419"
    assert parse_reply_language("") is None
    assert parse_reply_language("   ") is None
    for bad in ("english", "en-", "e", "123", "en-US-x-private", "say it in english", "follow-up"):
        with pytest.raises(ValueError):
            parse_reply_language(bad)
    for bad in (True, 1):
        with pytest.raises(TypeError):
            parse_reply_language(bad)
    with pytest.raises(ValueError):
        VoiceConfig(reply_language="english")
    with pytest.raises(TypeError):
        VoiceConfig(reply_language=True)
    assert VoiceConfig(reply_language="").reply_language is None
    assert VoiceConfig(language="en", reply_language="ca").language == "en"


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


def test_reply_language_judge_reads_the_words_not_the_prompt() -> None:
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("validate_263", root / "benchmarks" / "voice" / "validate_263.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_263"] = module
    spec.loader.exec_module(module)
    assert module.judge_reply_language("Ara mateix són les 10:42 UTC.") == "ca"
    assert module.judge_reply_language("I can’t check the live weather today.") == "en"
    assert module.judge_reply_language("No puedo ver el tiempo de hoy.") == "es"
    assert module.recall_digits("four one five five five five oh one nine nine") == 1.0


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


def test_truncation_marks_only_a_finished_reply_resumable() -> None:
    session = VoiceSession(_agent(), MockSTT(), MockTTS(), audio_input=AudioInputConfig(language="en"))
    full = "Close! But C knows. specific event or maybe a mix"
    session._turn_assistant_text = full
    session._turn_tts_segment_records = [(full, 1000)]
    session._turn_heard_bytes = 400
    session._turn_played_baseline = 0
    session._turn_finalized_ok = False
    session._apply_interruption_truncation(None)
    assert session._unheard_tail
    assert session._unheard_tail_resumable is False
    session._turn_finalized_ok = True
    session._apply_interruption_truncation(None)
    assert session._unheard_tail_resumable is True


def test_unheard_suffix_keeps_only_the_rest() -> None:
    full = "One autumn morning, a bright red train rolled out."
    assert _unheard_suffix(full, "One autumn morning, a bright red") == "train rolled out."
    assert _unheard_suffix(full, "") == full
    assert _unheard_suffix(full, full) == ""


def test_a_pure_ack_is_not_a_new_request() -> None:
    assert false_interrupt_followup("")
    assert false_interrupt_followup("Yeah.")
    assert false_interrupt_followup("Yeah. Okay.")
    assert false_interrupt_followup("d'acord")
    assert not false_interrupt_followup("What is the weather?")
    assert not false_interrupt_followup("No, I meant how antibiotics work.")
    assert not false_interrupt_followup("Okay. Let's go.")


def test_a_second_ack_does_not_cut_audio_and_a_request_does() -> None:
    session = VoiceSession(_agent(), MockSTT(), MockTTS(), audio_input=AudioInputConfig(language="en"))
    session._is_speaking = True
    assert session._pure_ack_keeps_the_reply("Yeah. Okay.") is True
    assert session._pure_ack_keeps_the_reply("No, I meant how antibiotics work.") is False
    session._is_speaking = False
    session._last_assistant_active_at = time.monotonic() - 10
    assert session._pure_ack_keeps_the_reply("Yeah.") is False
    session._last_assistant_active_at = time.monotonic()
    assert session._pure_ack_keeps_the_reply("Yeah. Okay.") is True
    session._unheard_tail = "train rolled out."
    session._unheard_tail_resumable = True
    assert session._pure_ack_keeps_the_reply("Yeah.") is False


async def test_backchannel_resumes_and_a_real_request_does_not() -> None:
    tts = MockTTS()
    session = VoiceSession(_agent(), MockSTT(), tts, audio_input=AudioInputConfig(language="en"))
    session._unheard_tail = "train rolled out."
    session._unheard_tail_resumable = True
    await session._begin_user_turn("Yeah.", replace_user_entry=False)
    assert tts.synthesized_texts == ["train rolled out."]
    assert [entry.text for entry in session._transcript] == ["train rolled out."]
    assert session._current_turn_task is None or session._current_turn_task.done()

    tts.synthesized_texts.clear()
    session._unheard_tail = "SECRET-TAIL"
    session._unheard_tail_resumable = True
    await session._begin_user_turn("What is the weather?", replace_user_entry=False)
    task = session._current_turn_task
    assert task is not None
    await task
    assert "SECRET-TAIL" not in "".join(tts.synthesized_texts)
    assert session._continuation_note == ""
    prompt = session.prompt_for_turn("You are a phone agent.").dynamic
    assert "SECRET-TAIL" not in prompt
    assert any(entry.role == "user" and entry.text == "What is the weather?" for entry in session._transcript)


def test_a_request_that_starts_with_an_ignored_ack_is_not_a_replay() -> None:
    session = VoiceSession(_agent(), MockSTT(), MockTTS(), audio_input=AudioInputConfig(language="en"))
    session._settle_consumed_commit("Okay.")
    assert session._is_replay_of_consumed("Okay.")
    assert session._partial_extends_consumed("Okay")
    assert not session._is_replay_of_consumed("Okay. What about order 1002?")
    assert not session._partial_extends_consumed("Okay. What about order")


async def test_a_patience_update_that_was_not_sent_is_tried_again() -> None:
    session = VoiceSession(_agent(), MockSTT(), MockTTS(), audio_input=AudioInputConfig(language="en"))
    calls: list[bool] = []

    async def configure_turn(**kwargs) -> bool:
        calls.append(kwargs["patient"])
        return len(calls) > 1

    session.stt.configure_turn = configure_turn  # type: ignore[attr-defined]
    assert await session._send_flux_thresholds(eot_threshold=1.0, eot_timeout_ms=2500, patient=True) is False
    assert await session._send_flux_thresholds(eot_threshold=1.0, eot_timeout_ms=2500, patient=True) is True
    assert calls == [True, True]


def test_the_ack_window_opens_when_the_reply_stops_playing() -> None:
    # Nobody spoke during the reply, so no caller event refreshed the clock.
    session = VoiceSession(_agent(), MockSTT(), MockTTS(), audio_input=AudioInputConfig(language="en"))
    session._note_turn_audio(session.audio_output.sample_rate)  # half a second of PCM16
    assert not session._assistant_active
    assert session._pure_ack_keeps_the_reply("Yeah. Okay.")
    session._last_assistant_active_at = time.monotonic() - 2.0
    assert not session._pure_ack_keeps_the_reply("Yeah. Okay.")


async def test_empty_commit_speaks_the_unheard_tail_without_the_model() -> None:
    tts = MockTTS()
    session = VoiceSession(_agent(), MockSTT(), tts, audio_input=AudioInputConfig(language="en"))
    session._unheard_tail = "train rolled out."
    await session._resume_unheard_tail(session._unheard_tail)
    assert tts.synthesized_texts == ["train rolled out."]
    assert session._transcript[-1].text == "train rolled out."
    assert session._unheard_tail == ""
    assert not any(isinstance(item, str) and item == "One autumn morning" for item in tts.synthesized_texts)


async def test_logged_requests_leave_the_tail_out_of_the_prompt() -> None:
    from timbal.voice.session import TranscriptEntry

    heard = "The Western Empire lasted from around 27 BCE"
    cases = (
        ("Okay. Let's go. Let's go.", "really shapes the next topic."),
        ("Oh, physical.", "specific event or maybe a mix"),
        ("no", "really shapes the next topic."),
        ("stop", "really shapes the next topic."),
    )
    for text, remainder in cases:
        tts = MockTTS()
        session = VoiceSession(_agent(), MockSTT(), tts, audio_input=AudioInputConfig(language="en"))
        session._transcript.append(TranscriptEntry(role="assistant", text=heard))
        session._unheard_tail = remainder
        session._unheard_tail_resumable = True
        await session._begin_user_turn(text, replace_user_entry=False)
        task = session._current_turn_task
        assert task is not None
        await task
        prompt = session.prompt_for_turn("You are a phone agent.").dynamic
        assert remainder not in prompt
        assert session._continuation_note == ""
        assert remainder not in "".join(tts.synthesized_texts)
        assert session._transcript[0].text == heard
        assert any(entry.role == "user" and entry.text == text for entry in session._transcript)


async def test_a_cancelled_generation_is_not_spoken_on_a_backchannel() -> None:
    """A model stop that never finished is not a remainder to recite."""
    tts = MockTTS()
    session = VoiceSession(_agent(), MockSTT(), tts, audio_input=AudioInputConfig(language="en"))
    session._unheard_tail = "specific event or maybe a mix"
    session._unheard_tail_resumable = False
    await session._begin_user_turn("Yeah.", replace_user_entry=False)
    assert "specific event or maybe a mix" not in "".join(tts.synthesized_texts)
    assert session._continuation_note == ""
    task = session._current_turn_task
    assert task is not None
    await task


async def test_a_second_interruption_still_does_not_script_the_tail() -> None:
    tts = MockTTS()
    session = VoiceSession(_agent(), MockSTT(), tts, audio_input=AudioInputConfig(language="en"))
    session._unheard_tail = "really shapes the next topic."
    session._unheard_tail_resumable = True
    await session._begin_user_turn("Okay. Let's go. Let's go.", replace_user_entry=False)
    task = session._current_turn_task
    assert task is not None
    await task
    session._unheard_tail = "specific event or maybe a mix"
    session._unheard_tail_resumable = True
    await session._begin_user_turn("Oh, physical.", replace_user_entry=False)
    task = session._current_turn_task
    assert task is not None
    await task
    prompt = session.prompt_for_turn("You are a phone agent.").dynamic
    assert "specific event or maybe a mix" not in prompt
    assert "really shapes" not in prompt


class _EmptyCommitSTT(DelayedMockSTT):
    """Yield an empty ``committed`` event. The stock test STT drops blank text."""

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item


async def _until(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.01)


async def test_empty_stt_commit_skips_a_cancelled_fragment_and_clears_it() -> None:
    fragment = "specific event or maybe a mix"
    stt = _EmptyCommitSTT()
    tts = MockTTS()
    session = VoiceSession(_agent(), stt, tts, audio_input=AudioInputConfig(language="en"))
    seen: list[object] = []

    async def _no_audio() -> AsyncIterator[bytes]:
        return
        yield  # noqa: RET504

    async def _drive() -> None:
        await _until(lambda: any(isinstance(event, SessionStarted) for event in seen))
        session._unheard_tail = fragment
        session._unheard_tail_resumable = False
        await stt.inject(TranscriptEvent(type="committed", text=""))
        await _until(lambda: session._unheard_tail == "")
        assert fragment not in tts.synthesized_texts
        assert session._unheard_tail_resumable is False
        assert session._continuation_note == ""
        await stt.inject(TranscriptEvent(type="committed", text="What is the weather?"))
        await _until(
            lambda: any(entry.role == "user" and entry.text == "What is the weather?" for entry in session._transcript)
        )
        assert fragment not in session.prompt_for_turn("You are a phone agent.").dynamic
        assert fragment not in "".join(tts.synthesized_texts)
        await stt.finish()

    async def _go() -> None:
        driver = asyncio.create_task(_drive())
        async with aclosing(session.run(_no_audio())) as stream:
            async for event in stream:
                seen.append(event)
        await driver

    await asyncio.wait_for(_go(), timeout=8)


async def test_empty_stt_commit_speaks_a_finished_tail_once() -> None:
    tail = "train rolled out."
    stt = _EmptyCommitSTT()
    tts = MockTTS()
    session = VoiceSession(_agent(), stt, tts, audio_input=AudioInputConfig(language="en"))

    async def _no_audio() -> AsyncIterator[bytes]:
        return
        yield  # noqa: RET504

    seen: list[object] = []

    async def _drive() -> None:
        await _until(lambda: any(isinstance(event, SessionStarted) for event in seen))
        session._unheard_tail = tail
        session._unheard_tail_resumable = True
        await stt.inject(TranscriptEvent(type="committed", text=""))
        await _until(lambda: tts.synthesized_texts == [tail] and session._unheard_tail == "")
        await stt.inject(TranscriptEvent(type="committed", text=""))
        await asyncio.sleep(0.05)
        assert tts.synthesized_texts == [tail]
        await stt.inject(TranscriptEvent(type="committed", text="Okay. Let's go. Let's go."))
        await _until(
            lambda: any(
                entry.role == "user" and entry.text == "Okay. Let's go. Let's go." for entry in session._transcript
            )
        )
        assert tail not in session.prompt_for_turn("You are a phone agent.").dynamic
        await stt.finish()

    async def _go() -> None:
        driver = asyncio.create_task(_drive())
        async with aclosing(session.run(_no_audio())) as stream:
            async for event in stream:
                seen.append(event)
        await driver

    await asyncio.wait_for(_go(), timeout=5)


def _agent():
    from timbal import Agent

    return Agent(name="gaps", model=TestModel(responses=["ok"]), system_prompt="You are a phone agent.")
