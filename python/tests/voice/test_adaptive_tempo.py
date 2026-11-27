"""A reply slowed down for a slow or upset caller (``VoiceConfig.adaptive_tempo``)."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.voice import CLIENT_SETTABLE_VOICE_FIELDS, build_voice_session
from timbal.voice import AudioInputConfig, TranscriptEvent, VoiceConfig, VoiceSession
from timbal.voice.adaptive_tempo import DEFAULT_RATES, MIN_SYLLABLES, CallerPace, syllables

from .test_mic_activity import silence, talk
from .test_session import DelayedMockSTT, MockTTS
from .test_tts_pace import SR, _emitted, _PcmTTS, _session, _sine

SLOW = "Hola, buenas tardes, quería saber cuánto tengo que pagar este mes"  # 22 syllables
QUICK = "Sí mira te llamo porque me ha llegado una factura que no entiendo"  # 22 syllables

# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,count",
    [
        ("hola", 2),
        ("Quería saber cuánto", 7),
        ("sí", 1),
        ("pst", 1),
        ("the weather is nice", 6),
        ("día", 2),
        ("país", 2),
        ("cuánto", 2),
        ("baúl", 2),
        ("", 0),
        ("600 123 456", 0),
        ("¿Y la factura?", 5),
    ],
)
def test_syllables(text, count):
    assert syllables(text) == count


@pytest.mark.parametrize(
    "text,secs",
    [
        ("sí, claro", 2.0),  # too short to say anything
        ("mi teléfono es el 600 123 456 y vivo en Madrid capital", 6.0),  # digits: nobody knows how it was said
        (SLOW, 0.5),  # under a second of speech
        (SLOW, 1.2),  # faster than anyone talks: the seconds were not all measured
        (SLOW, 30.0),  # slower than speech: noise was counted
        (SLOW, None),  # no meter
    ],
)
def test_an_utterance_that_says_nothing_about_pace(text, secs):
    assert CallerPace.measure(text, secs) is None


def test_the_rate_is_syllables_over_seconds_of_speech():
    assert syllables(SLOW) >= MIN_SYLLABLES
    assert CallerPace.measure(SLOW, 5.0) == pytest.approx(syllables(SLOW) / 5.0)


# ---------------------------------------------------------------------------
# The factor
# ---------------------------------------------------------------------------


def test_a_slow_caller_gets_the_slowest_and_a_quick_one_the_full_speed():
    slow, normal = DEFAULT_RATES["es"]
    assert CallerPace().heard(SLOW, syllables(SLOW) / (slow - 0.5), "es") == 0.9
    assert CallerPace().heard(QUICK, syllables(QUICK) / (normal + 0.5), "es") == 1.0


def test_in_between_it_is_a_straight_line():
    slow, normal = DEFAULT_RATES["es"]
    middle = (slow + normal) / 2
    assert CallerPace().heard(SLOW, syllables(SLOW) / middle, "es") == 0.95
    assert CallerPace(slowest=0.8).heard(SLOW, syllables(SLOW) / middle, "es") == 0.9


def test_nothing_heard_is_full_speed():
    pace = CallerPace()
    assert pace.factor("es") == 1.0
    assert pace.heard("sí", 0.4, "es") == 1.0
    assert pace.rate is None


def test_english_is_measured_against_its_own_rates():
    text = "I was wondering if you could tell me what my balance is today"
    rate = 4.2
    assert CallerPace().heard(text, syllables(text) / rate, "en") > 0.9
    assert CallerPace().heard(text, syllables(text) / rate, "es") == 0.9


def test_the_rates_can_be_set():
    pace = CallerPace(slow_rate=3.0, normal_rate=4.0)
    assert pace.rates("es") == (3.0, 4.0)
    assert pace.heard(SLOW, syllables(SLOW) / 4.2, "es") == 1.0


def test_the_pace_is_smoothed_over_utterances():
    pace = CallerPace()
    assert pace.heard(SLOW, syllables(SLOW) / 4.0, "es") == 0.9
    pace.heard(QUICK, syllables(QUICK) / 7.0, "es")
    assert pace.rate == pytest.approx(5.5)
    assert 0.9 < pace.factor("es") < 1.0


def test_an_utterance_that_says_nothing_keeps_what_was_known():
    pace = CallerPace()
    pace.heard(SLOW, syllables(SLOW) / 4.0, "es")
    assert pace.heard("sí, eso es", 0.8, "es") == 0.9
    assert pace.last_measured is None


@pytest.mark.parametrize(
    "text,language",
    [
        ("Pues estoy harto de llamar y que nadie me diga nada", "es"),
        ("Esto es una VERGÜENZA.", "es"),
        ("es que me están tomando el pelo", "es"),
        ("I'm fed up with this", "en"),
        ("This is ridiculous!", "en"),
        ("Ja n'hi ha prou, estic farta", "ca"),
        ("esto es inaceptable", None),
        ("this is unacceptable", "de"),
    ],
)
def test_an_upset_caller_is_recognised_by_their_words(text, language):
    assert CallerPace().is_upset(text, language)


@pytest.mark.parametrize(
    "text,language",
    [
        ("Hola, quería cambiar la cuenta del banco", "es"),
        ("me harté de reír", "es"),
        ("el timón del barco", "es"),
        ("an angrier bird", "en"),
        ("", "es"),
    ],
)
def test_other_words_are_not(text, language):
    assert not CallerPace().is_upset(text, language)


def test_an_upset_caller_is_answered_slower_for_a_few_replies():
    pace = CallerPace()
    assert pace.heard("Estoy harto de esperar", 1.2, "es") == 0.93
    assert pace.heard("sí", 0.3, "es") == 0.93
    assert pace.heard("vale", 0.3, "es") == 0.93
    assert pace.heard("gracias", 0.4, "es") == 1.0


def test_being_upset_again_starts_the_count_again():
    pace = CallerPace(upset_turns=2)
    pace.heard("esto es inaceptable", 1.2, "es")
    pace.heard("ya está bien", 0.9, "es")
    assert pace.upset_left == 2
    assert pace.heard("vale", 0.3, "es") == 0.93
    assert pace.heard("vale", 0.3, "es") == 1.0


def test_slow_and_upset_is_the_slower_of_the_two():
    pace = CallerPace()
    assert pace.heard(SLOW + " y estoy harto", syllables(SLOW) / 4.0 + 1.0, "es") == 0.9
    assert CallerPace(slowest=0.97).heard(SLOW + " y estoy harto", syllables(SLOW) / 4.0 + 1.0, "es") == 0.93


def test_mood_can_be_left_out_and_so_can_pace():
    assert CallerPace(upset_tempo=None).heard("Estoy harto de esperar", 1.2, "es") == 1.0
    only_mood = CallerPace(slowest=1.0)
    assert only_mood.heard(SLOW, syllables(SLOW) / 4.0, "es") == 1.0
    assert only_mood.heard("esto es una estafa", 1.3, "es") == 0.93


def test_the_words_can_be_replaced():
    pace = CallerPace(upset_words=["qué lata"])
    assert pace.is_upset("¡Qué lata me dais!", "es")
    assert not pace.is_upset("estoy harto", "es")
    by_language = CallerPace(upset_words={"es": ["qué lata"], "*": ["ugh"]})
    assert by_language.is_upset("qué lata", "es")
    assert by_language.is_upset("ugh, again", "en")
    assert not by_language.is_upset("ugh", "es")


def test_a_transcript_that_grew_is_one_utterance():
    pace = CallerPace()
    pace.heard("Quiero cambiar la cuenta del banco", 2.0, "es")
    first = pace.rate
    longer = "Quiero cambiar la cuenta del banco donde me cobráis la luz"
    pace.heard(longer, 2.0, "es")
    assert pace.rate == pytest.approx(syllables(longer) / 4.0)
    assert pace.rate != first


def test_a_grown_transcript_does_not_count_an_upset_reply_twice():
    pace = CallerPace(upset_turns=2)
    pace.heard("esto es inaceptable", 1.2, "es")
    pace.heard("vale", 0.3, "es")
    pace.heard("vale pues nada", 0.5, "es")
    assert pace.upset_left == 1


def test_a_preview_keeps_nothing():
    pace = CallerPace()
    assert pace.preview("Estoy harto de esperar", 1.5, "es") == 0.93
    assert pace.preview(SLOW, syllables(SLOW) / 4.0, "es") == 0.9
    assert pace.rate is None
    assert pace.upset_left == 0
    assert pace.factor("es") == 1.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_off_by_default_and_true_is_the_defaults():
    assert VoiceConfig().adaptive_tempo is None
    assert VoiceConfig(adaptive_tempo=False).adaptive_tempo is None
    assert VoiceConfig(adaptive_tempo={}).adaptive_tempo is None
    block = VoiceConfig(adaptive_tempo=True).adaptive_tempo
    assert (block.slowest, block.upset_tempo, block.upset_turns) == (0.9, 0.93, 3)
    assert block.enabled
    assert not VoiceConfig(adaptive_tempo={"slowest": 1.0, "upset_tempo": None}).adaptive_tempo.enabled


def test_upset_words_are_cleaned_and_keyed_by_language():
    block = VoiceConfig(adaptive_tempo={"upset_words": {"ES-es": [" qué  lata ", "qué lata"], "*": ["ugh"]}})
    assert block.adaptive_tempo.upset_words == {"es": ["qué lata"], "*": ["ugh"]}
    assert VoiceConfig(adaptive_tempo={"upset_words": ["ugh", ""]}).adaptive_tempo.upset_words == ["ugh"]


@pytest.mark.parametrize(
    "value",
    [
        {"slowest": 0.5},
        {"slowest": 1.1},
        {"slowest": True},
        {"upset_tempo": 1.2},
        {"upset_turns": 0},
        {"slow_rate": 5.0, "normal_rate": 5.0},
        {"slow_rate": 0},
        {"upset_words": "harto"},
        {"upset_words": [1]},
        {"upset_words": {"": ["x"]}},
        {"unknown": 1},
        "slow",
    ],
)
def test_bad_values_are_rejected(value):
    with pytest.raises(ValueError):
        VoiceConfig(adaptive_tempo=value)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def make_session(tmp_path: Path, raw: bytes, **kwargs) -> VoiceSession:
    session, recorder = _session(_PcmTTS(raw), tmp_path, audio_input=AudioInputConfig(language="es"), **kwargs)
    recorder.close()
    session._start_mic_activity()
    return session


def caller_says(session: VoiceSession, text: str, speech_secs: float) -> None:
    """The mic hears ``speech_secs`` of talking, then the transcriber commits ``text``."""
    session._mic_activity.push(silence(0.3) + talk(speech_secs) + silence(0.6))
    session._note_caller_utterance(text)


async def test_without_the_option_nothing_is_built(tmp_path: Path):
    raw = _sine(int(0.5 * SR))
    session = make_session(tmp_path, raw)
    assert session._mic_activity is None
    await session._run_turn(SLOW)
    assert _emitted(session) == raw
    assert session.metrics[-1].tts_tempo is None
    assert session.metrics[-1].caller_syllables_per_sec is None


async def test_a_slow_caller_is_answered_slower(tmp_path: Path):
    raw = _sine(int(1.0 * SR))
    session = make_session(tmp_path, raw, adaptive_tempo=True)
    caller_says(session, SLOW, syllables(SLOW) / 4.0)
    await session._run_turn(SLOW)
    assert session._tempo_factor == 0.9
    assert len(_emitted(session)) == pytest.approx(len(raw) / 0.9, rel=0.03)
    metrics = session.metrics[-1]
    assert metrics.tts_tempo == 0.9
    assert metrics.caller_syllables_per_sec == pytest.approx(4.0, abs=0.15)


async def test_a_quick_caller_is_answered_at_the_configured_speed(tmp_path: Path):
    raw = _sine(int(1.0 * SR))
    session = make_session(tmp_path, raw, adaptive_tempo=True)
    caller_says(session, QUICK, syllables(QUICK) / 6.8)
    await session._run_turn(QUICK)
    assert _emitted(session) == raw
    assert session.metrics[-1].tts_tempo == 1.0


async def test_it_multiplies_the_configured_tempo(tmp_path: Path):
    raw = _sine(int(1.0 * SR))
    session = make_session(tmp_path, raw, adaptive_tempo=True, tts_tempo=1.1)
    caller_says(session, SLOW, syllables(SLOW) / 4.0)
    await session._run_turn(SLOW)
    assert session._effective_tts_tempo() == pytest.approx(0.99)
    assert session.metrics[-1].tts_tempo == 0.99
    assert len(_emitted(session)) == pytest.approx(len(raw) / 0.99, rel=0.03)


async def test_each_utterance_is_timed_from_the_one_before(tmp_path: Path):
    session = make_session(tmp_path, _sine(int(0.2 * SR)), adaptive_tempo=True)
    caller_says(session, SLOW, syllables(SLOW) / 4.0)
    caller_says(session, QUICK, syllables(QUICK) / 7.0)
    assert session._caller_pace.last_measured == pytest.approx(7.0, abs=0.3)


async def test_an_upset_caller_is_answered_slower_and_it_wears_off(tmp_path: Path):
    raw = _sine(int(1.0 * SR))
    session = make_session(tmp_path, raw, adaptive_tempo={"upset_turns": 1})
    caller_says(session, "Estoy harto de esperar", 1.2)
    await session._run_turn("Estoy harto de esperar")
    assert session.metrics[-1].tts_tempo == 0.93
    assert len(_emitted(session)) == pytest.approx(len(raw) / 0.93, rel=0.03)
    caller_says(session, "vale", 0.3)
    await session._run_turn("vale")
    assert session.metrics[-1].tts_tempo == 1.0
    assert _emitted(session) == raw


async def test_a_reply_started_before_its_transcript_is_confirmed_reads_it_too(tmp_path: Path):
    session = make_session(tmp_path, _sine(int(0.2 * SR)), adaptive_tempo=True)
    session._mic_activity.push(silence(0.3) + talk(syllables(SLOW) / 4.0) + silence(0.6))
    session._set_turn_tempo(SLOW, speculative=True)
    assert session._tempo_factor == 0.9
    assert session._caller_pace.rate is None  # nothing kept until it is confirmed
    session._note_caller_utterance(SLOW)
    assert session._caller_pace.factor("es") == 0.9


async def test_mic_audio_in_another_encoding_keeps_the_mood_only(tmp_path: Path):
    session, recorder = _session(
        _PcmTTS(_sine(int(0.2 * SR))),
        tmp_path,
        adaptive_tempo=True,
        audio_input=AudioInputConfig(language="es", encoding="mulaw"),
    )
    recorder.close()
    session._start_mic_activity()
    assert session._mic_activity is None
    session._note_caller_utterance(SLOW)
    assert session._caller_pace.factor("es") == 1.0
    session._note_caller_utterance("esto es una vergüenza")
    assert session._caller_pace.factor("es") == 0.93


async def test_the_call_feeds_the_meter_and_reads_the_commit():
    """Through ``run()``: mic audio reaches the meter and a committed transcript sets the pace."""
    stt = DelayedMockSTT()
    session = VoiceSession(
        agent=Agent(name="t", model=TestModel(responses=["Claro, ahora te lo miro."]), tools=[]),
        stt=stt,
        tts=MockTTS(),
        turn_detector="heuristic",
        adaptive_tempo=True,
        audio_input=AudioInputConfig(language="es"),
    )
    speech_secs = syllables(SLOW) / 4.0
    sent = asyncio.Event()

    async def mic() -> AsyncIterator[bytes]:
        audio = silence(0.3) + talk(speech_secs) + silence(0.6)
        for start in range(0, len(audio), 3200):
            yield audio[start : start + 3200]
        sent.set()
        await asyncio.Event().wait()

    async def drive() -> None:
        await sent.wait()
        await asyncio.sleep(0.05)
        # A long commit with no partial before it is read as a transcriber hallucination.
        await stt.inject(TranscriptEvent(type="partial", text=SLOW[:20]))
        await stt.inject(TranscriptEvent(type="committed", text=SLOW + "."))
        while not session.metrics:
            await asyncio.sleep(0.01)
        await stt.finish()

    driver = asyncio.create_task(drive())
    async with aclosing(session.run(mic())) as stream:
        async for _ in stream:
            pass
    await driver
    assert session._mic_activity.speech_secs == pytest.approx(speech_secs, abs=0.1)
    assert session.metrics[-1].tts_tempo == 0.9
    assert session.metrics[-1].caller_syllables_per_sec == pytest.approx(4.0, abs=0.15)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def test_the_agent_sets_it_and_a_caller_hello_cannot():
    assert "adaptive_tempo" not in CLIENT_SETTABLE_VOICE_FIELDS
    agent = Agent(name="desk", model=TestModel(responses=["hi"]), tools=[])
    session, _ = build_voice_session(
        agent, VoiceConfig(stt_provider="elevenlabs", adaptive_tempo={"slowest": 0.85}), {}
    )
    assert session.adaptive_tempo.slowest == 0.85
    assert session._caller_pace.slowest == 0.85
    session, _ = build_voice_session(agent, VoiceConfig(stt_provider="elevenlabs"), {"adaptive_tempo": True})
    assert session.adaptive_tempo is None
