"""The caller's talking and pausing, read from the mic level (``timbal.voice.mic_activity``)."""

import math
import random
from array import array

import pytest
from timbal.voice.mic_activity import MicActivity

SR = 16_000


def tone(secs: float, *, dbfs: float = -20.0, rate: int = SR) -> bytes:
    amp = 32768 * 10 ** (dbfs / 20) * math.sqrt(2)
    return array("h", (int(amp * math.sin(2 * math.pi * 220 * i / rate)) for i in range(int(secs * rate)))).tobytes()


def talk(secs: float, *, dbfs: float = -20.0, rate: int = SR) -> bytes:
    """Something shaped like speech: five syllables a second, a short gap before each."""
    syllable = silence(0.04, rate=rate) + tone(0.16, dbfs=dbfs, rate=rate)
    total = int(secs * rate) * 2
    return (syllable * math.ceil(secs / 0.2))[-total:] if total else b""


def noise(secs: float, *, dbfs: float, rate: int = SR, seed: int = 1) -> bytes:
    rng = random.Random(seed)
    amp = 32768 * 10 ** (dbfs / 20) * math.sqrt(3)
    return array("h", (int(rng.uniform(-amp, amp)) for _ in range(int(secs * rate)))).tobytes()


def silence(secs: float, *, rate: int = SR) -> bytes:
    return bytes(int(secs * rate) * 2)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def meter(rate: int = SR) -> tuple[MicActivity, Clock]:
    clock = Clock()
    return MicActivity(rate, clock=clock), clock


def test_a_silent_line_has_heard_nobody():
    mic, _ = meter()
    mic.push(silence(3.0))
    assert mic.speech_secs == 0.0
    assert mic.quiet_secs() is None
    assert not mic.speaking


def test_talking_is_timed():
    mic, _ = meter()
    mic.push(silence(0.5) + talk(2.0))
    assert mic.speech_secs == pytest.approx(2.0, abs=0.06)
    assert mic.speaking
    assert mic.quiet_secs() == 0.0


def test_the_gap_between_two_words_is_part_of_the_talking():
    mic, _ = meter()
    mic.push(talk(1.0) + silence(0.12) + talk(1.0))
    assert mic.speech_secs == pytest.approx(2.12, abs=0.08)


def test_a_pause_is_not():
    mic, _ = meter()
    mic.push(talk(1.0) + silence(1.0) + talk(1.0))
    assert mic.speech_secs == pytest.approx(1.96, abs=0.06)


def test_how_long_the_caller_has_been_quiet():
    mic, _ = meter()
    mic.push(talk(1.5))
    mic.push(silence(0.6))
    assert mic.quiet_secs() == pytest.approx(0.6, abs=0.03)
    assert not mic.speaking
    mic.push(talk(0.2))
    assert mic.quiet_secs() == 0.0


def test_a_click_is_not_a_voice():
    mic, _ = meter()
    mic.push(silence(1.0) + tone(0.02) + silence(1.0))
    assert mic.speech_secs == 0.0
    assert mic.quiet_secs() is None


def test_audio_under_the_agent_is_followed_but_not_timed():
    mic, _ = meter()
    mic.push(talk(1.0), count=False)
    assert mic.speech_secs == 0.0
    assert mic.speaking
    mic.push(silence(0.5), count=False)
    assert mic.quiet_secs() == pytest.approx(0.5, abs=0.03)
    mic.push(talk(1.0))
    assert mic.speech_secs == pytest.approx(1.0, abs=0.06)


def test_line_noise_is_not_speech_and_a_voice_over_it_is():
    mic, _ = meter()
    mic.push(noise(4.0, dbfs=-52.0))
    assert mic.speech_secs == 0.0
    mic.push(talk(1.0, dbfs=-22.0))
    assert mic.speech_secs == pytest.approx(1.0, abs=0.06)
    mic.push(noise(1.0, dbfs=-52.0, seed=2))
    assert mic.quiet_secs() == pytest.approx(1.0, abs=0.05)


def test_a_line_that_got_noisier_is_speech_for_three_seconds_at_most():
    mic, _ = meter()
    mic.push(silence(1.0))
    mic.push(noise(4.0, dbfs=-38.0))
    assert mic.speech_secs <= 3.1
    counted = mic.speech_secs
    mic.push(noise(20.0, dbfs=-38.0, seed=3))
    assert mic.speech_secs == counted
    assert not mic.speaking


def test_a_voice_over_that_noise_is_still_heard():
    mic, _ = meter()
    mic.push(noise(5.0, dbfs=-38.0))
    spoken = array("h", talk(2.0, dbfs=-18.0))
    hiss = array("h", noise(2.0, dbfs=-38.0, seed=4))
    mic.push(array("h", (a + b for a, b in zip(spoken, hiss, strict=True))).tobytes())
    assert mic.speech_secs == pytest.approx(2.0, abs=0.1)


def test_a_steady_tone_is_not_a_voice_for_long():
    mic, _ = meter()
    mic.push(silence(0.5) + tone(20.0))
    assert mic.speech_secs <= 3.1
    assert not mic.speaking


def test_a_long_monologue_is_speech_to_its_end():
    mic, _ = meter()
    mic.push(silence(0.5))
    for _ in range(30):
        mic.push(talk(1.0))
    assert mic.speech_secs == pytest.approx(30.0, abs=0.1)


def test_chunk_sizes_do_not_matter():
    audio = silence(0.3) + talk(1.2) + silence(0.5) + talk(0.7) + silence(0.4)
    whole, _ = meter()
    whole.push(audio)
    pieces, _ = meter()
    for start in range(0, len(audio), 701):
        pieces.push(audio[start : start + 701])
    assert pieces.speech_secs == pytest.approx(whole.speech_secs, abs=1e-9)
    assert pieces.quiet_secs() == pytest.approx(whole.quiet_secs(), abs=0.021)


def test_telephone_audio_works_the_same():
    mic, _ = meter(8_000)
    mic.push(silence(0.5, rate=8_000) + talk(2.0, rate=8_000) + silence(0.5, rate=8_000))
    assert mic.speech_secs == pytest.approx(2.0, abs=0.06)
    assert mic.quiet_secs() == pytest.approx(0.5, abs=0.03)


def test_a_mic_that_stopped_sending_has_no_opinion():
    mic, clock = meter()
    mic.push(talk(1.0) + silence(0.5))
    assert mic.fresh
    clock.now += 1.5
    assert not mic.fresh
    assert mic.quiet_secs() is None
    assert not mic.speaking


def test_an_empty_chunk_and_a_bad_rate():
    mic, _ = meter()
    mic.push(b"")
    assert not mic.fresh
    with pytest.raises(ValueError):
        MicActivity(0)
