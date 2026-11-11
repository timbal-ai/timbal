"""Pause rhythm: which silences are joined, kept or lengthened, and that speech is never held."""

from __future__ import annotations

import ast
import asyncio
import inspect

import numpy as np
import pytest
from structlog.testing import capture_logs
from timbal.voice import rhythm
from timbal.voice.config import NATURAL_SPEECH_RHYTHM, SpeechRhythm, VoiceConfig
from timbal.voice.rhythm import (
    FRAME_SECS,
    JOINED_SECS,
    LENGTHENED_MAX_SECS,
    MAX_HOLD_SECS,
    MIN_SILENCE_SECS,
    PROMOTED_SECS,
    STRETCH_FACTOR,
    PauseShaper,
    shape_pauses,
)

SR = 16_000


def test_voice_extra_is_installed() -> None:
    """CI installs timbal[voice] via --extra all. Missing numpy must fail, not skip."""
    assert np.__version__


def _speech(secs: float, amp: int = 8000, freq: float = 220.0, sr: int = SR) -> bytes:
    t = np.arange(int(round(secs * sr)))
    return (amp * np.sin(2 * np.pi * freq * t / sr)).astype("<i2").tobytes()


def _quiet(secs: float, amp: int = 0, sr: int = SR) -> bytes:
    return np.full(int(round(secs * sr)), amp, dtype="<i2").tobytes()


def _room_tone(secs: float, seed: int = 3, amp: float = 30.0) -> bytes:
    """Low-level noise, the way a recording's silence actually looks (about -60 dBFS)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(round(secs * SR))) * amp).astype("<i2").tobytes()


def _soft(secs: float, dbfs: float = -44.0, seed: int = 5) -> bytes:
    """Noise at ``dbfs`` RMS: a final "s" or a breath. Under the -42 frame level, with peaks well over it."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(round(secs * SR))) * 32768 * 10 ** (dbfs / 20)).astype("<i2").tobytes()


def _shape(raw: bytes, **kwargs) -> tuple[bytes, dict]:
    """One pass through the shaper itself: no clock, so the result depends on the audio alone."""
    shaper = PauseShaper(kwargs.pop("sample_rate", SR), **kwargs)
    return shaper.push(raw) + shaper.finish(), shaper.stats


def _between(out: bytes, first: bytes, second: bytes) -> np.ndarray:
    """What sits between an opening ``first`` and a closing ``second``."""
    assert out.startswith(first) and out.endswith(second)
    return np.frombuffer(out[len(first) : len(out) - len(second)], dtype="<i2")


async def _chunks(parts: list[bytes]):
    for part in parts:
        yield part


async def _run_source(source, **kwargs) -> bytes:
    kwargs.setdefault("sample_rate", SR)
    out: list[bytes] = []
    async for chunk in shape_pauses(source, **kwargs):
        out.append(chunk)
    return b"".join(out)


async def _run(parts: list[bytes], **kwargs) -> bytes:
    return await _run_source(_chunks(parts), **kwargs)


def _split(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)] or [b""]


def _silences(pcm: bytes, sr: int = SR, floor: int = 200) -> list[float]:
    """Seconds of each run of samples under ``floor``, edges included, between two stretches of speech."""
    loud = np.abs(np.frombuffer(pcm, dtype="<i2").astype(np.int32)) >= floor
    # A sine crosses zero: call a sample quiet only when the next 2 ms are quiet too.
    span = int(0.002 * sr)
    padded = np.concatenate([loud, np.zeros(span, dtype=bool)])
    active = np.array([padded[i : i + span].any() for i in range(len(loud))])
    runs: list[float] = []
    start = None
    for index, on in enumerate(active):
        if not on and start is None:
            start = index
        elif on and start is not None:
            runs.append((index - start) / sr)
            start = None
    if start is not None:
        runs.append((len(active) - start) / sr)
    return runs


def _internal(pcm: bytes) -> list[float]:
    """The silences that sit between two stretches of speech (not the lead or the tail)."""
    samples = np.abs(np.frombuffer(pcm, dtype="<i2").astype(np.int32)) >= 200
    runs = _silences(pcm)
    if runs and not samples[:8].any():
        runs = runs[1:]
    if runs and not samples[-8:].any():
        runs = runs[:-1]
    return runs


class TestOffByDefault:
    async def test_no_share_set_yields_every_chunk_unchanged(self) -> None:
        raw = _quiet(0.3) + _speech(0.2) + _quiet(0.16) + _speech(0.2) + _quiet(0.5) + _speech(0.2)
        parts = _split(raw, 701)
        out: list[bytes] = []
        async for chunk in shape_pauses(_chunks(parts), sample_rate=SR):
            out.append(chunk)
        assert out == parts

    def test_disabled_shaper_reports_it(self) -> None:
        assert PauseShaper(SR).enabled is False
        assert PauseShaper(SR, join_gaps=0.5).enabled is True
        assert PauseShaper(SR, stretch_pauses=0.5).enabled is True
        assert PauseShaper(SR, promote_gaps=0.5).enabled is True


class TestWordGaps:
    async def test_a_gap_between_two_words_is_closed(self) -> None:
        word = _speech(0.2)
        out = await _run([word + _quiet(0.16) + word], join_gaps=1.0, seed=1)
        (gap,) = _internal(out)
        assert JOINED_SECS[0] - 2 * FRAME_SECS <= gap <= JOINED_SECS[1] + 2 * FRAME_SECS
        assert len(out) < len(word) * 2 + len(_quiet(0.16))

    @pytest.mark.parametrize("sr", [8_000, 16_000, 24_000, 44_100])
    @pytest.mark.parametrize("secs", [0.023, 0.05, 0.073, 0.1])
    async def test_a_closed_gap_is_exactly_as_long_as_asked(self, sr: int, secs: float) -> None:
        """The crossfade plays both ends of the cut at once. That overlap must not come off the gap."""
        word = _speech(0.2, sr=sr)
        shaper = PauseShaper(sr, join_gaps=1.0, joined_secs=(secs, secs))
        out = shaper.push(word + _quiet(0.16, sr=sr) + word) + shaper.finish()
        assert shaper.stats["joined"] == 1
        assert (len(out) - 2 * len(word)) // 2 == int(secs * sr)

    @pytest.mark.parametrize("seed", range(1, 9))
    async def test_a_closed_gap_stays_inside_the_documented_span(self, seed: int) -> None:
        word = _speech(0.2)
        out = await _run([word + _quiet(0.16) + word], join_gaps=1.0, seed=seed)
        left = (len(out) - 2 * len(word)) / 2 / SR
        assert JOINED_SECS[0] - 1 / SR <= left <= JOINED_SECS[1]

    async def test_a_cut_of_a_few_samples_still_gives_the_asked_length(self) -> None:
        """Less to remove than the crossfade is long: the fade shrinks, the length does not."""
        word = _speech(0.2)
        shaper = PauseShaper(SR, join_gaps=1.0, joined_secs=(0.155, 0.155))
        out = shaper.push(word + _room_tone(0.16) + word) + shaper.finish()
        assert (len(out) - 2 * len(word)) // 2 == int(0.155 * SR)

    async def test_join_share_zero_leaves_the_gap_alone(self) -> None:
        raw = _speech(0.2) + _quiet(0.16) + _speech(0.2)
        assert await _run([raw], stretch_pauses=1.0, seed=1) == raw

    async def test_a_consonant_closure_is_never_touched(self) -> None:
        """Under ``gap_min_secs`` it is the stop of a p/t/k. Shortening it eats the consonant."""
        raw = _speech(0.2) + _quiet(0.08) + _speech(0.2)
        assert await _run([raw], join_gaps=1.0, promote_gaps=0.0, stretch_pauses=1.0, seed=1) == raw

    async def test_share_decides_how_many_gaps_are_closed(self) -> None:
        word, gap = _speech(0.1), _quiet(0.16)
        raw = word + (gap + word) * 200
        shaper = PauseShaper(SR, join_gaps=0.5, seed=7)
        shaper.push(raw)
        shaper.finish()
        assert shaper.stats["gaps"] == 200
        assert 70 <= shaper.stats["joined"] <= 130
        assert shaper.stats["joined"] + shaper.stats["kept"] == 200

    async def test_a_promoted_gap_becomes_a_real_pause(self) -> None:
        word = _speech(0.2)
        out = await _run([word + _quiet(0.16) + word], promote_gaps=1.0, seed=1)
        (gap,) = _internal(out)
        assert PROMOTED_SECS[0] - 2 * FRAME_SECS <= gap <= PROMOTED_SECS[1] + 2 * FRAME_SECS

    async def test_join_and_promote_draw_from_the_same_gaps(self) -> None:
        word, gap = _speech(0.1), _quiet(0.16)
        shaper = PauseShaper(SR, join_gaps=0.6, promote_gaps=0.4, seed=11)
        shaper.push(word + (gap + word) * 300)
        assert shaper.stats["joined"] + shaper.stats["promoted"] == 300
        assert shaper.stats["kept"] == 0
        assert 90 <= shaper.stats["promoted"] <= 150

    async def test_gap_bounds_move_the_line_between_gap_and_pause(self) -> None:
        """A 0.22 s comma pause is kept at the default and joined when ``gap_max_secs`` is 0.26."""
        raw = _speech(0.2) + _quiet(0.22) + _speech(0.2)
        assert await _run([raw], join_gaps=1.0, seed=1) == raw
        out = await _run([raw], join_gaps=1.0, gap_max_secs=0.26, seed=1)
        assert len(out) < len(raw)


class TestSoftSound:
    """A silence found by frame level still holds the end of a word, a final "s", a breath."""

    @pytest.mark.parametrize("seed", range(1, 7))
    async def test_a_soft_consonant_before_the_gap_is_kept(self, seed: int) -> None:
        word, final_s = _speech(0.2), _soft(0.06)
        raw = word + final_s + _quiet(0.12) + word
        out, stats = _shape(raw, join_gaps=1.0, seed=seed)
        assert stats["joined"] == 1
        # All 60 ms of it, bit for bit. Only the silence after it got shorter.
        left = _between(out, word + final_s, word)
        assert int(MIN_SILENCE_SECS * SR) <= len(left) < int(0.12 * SR)
        assert not left.any()

    async def test_a_soft_onset_after_the_gap_is_kept(self) -> None:
        word, onset = _speech(0.2), _soft(0.05)
        out, stats = _shape(word + _quiet(0.12) + onset + word, join_gaps=1.0, seed=1)
        assert stats["joined"] == 1
        assert not _between(out, word, onset + word).any()

    async def test_a_joined_gap_always_keeps_some_true_silence(self) -> None:
        word = _speech(0.2)
        # 130 ms of soft sound and 30 ms of silence: closing to 23 ms would eat the sound.
        raw = word + _soft(0.13) + _quiet(0.03) + word
        out, _stats = _shape(raw, join_gaps=1.0, joined_secs=(0.023, 0.023), seed=1)
        assert len(_between(out, word + _soft(0.13), word)) == int(MIN_SILENCE_SECS * SR)

    async def test_a_gap_with_no_true_silence_is_left_alone(self) -> None:
        raw = _speech(0.2) + _soft(0.16) + _speech(0.2)
        for kwargs in ({"join_gaps": 1.0}, {"promote_gaps": 1.0}):
            out, stats = _shape(raw, seed=1, **kwargs)
            assert out == raw
            assert stats["skipped"] == 1 and stats["joined"] == stats["promoted"] == 0

    @pytest.mark.parametrize("seed", range(1, 7))
    async def test_a_breath_in_a_pause_is_not_repeated(self, seed: int) -> None:
        word, breath = _speech(0.2), _soft(0.10)
        raw = word + _quiet(0.10) + breath + _quiet(0.10) + word
        out, stats = _shape(raw, stretch_pauses=1.0, seed=seed)
        assert stats["stretched"] == 1 and len(out) > len(raw)
        assert out.count(breath) == 1
        # What was added is silence: nothing in the pause but the breath is over the level.
        pause = _between(out, word, word)
        assert (np.abs(pause.astype(np.int32)) > 260).sum() <= (
            np.abs(np.frombuffer(breath, dtype="<i2").astype(np.int32)) > 260
        ).sum()

    async def test_a_pause_that_is_all_breath_is_not_lengthened(self) -> None:
        raw = _speech(0.2) + _soft(0.30) + _speech(0.2)
        out, stats = _shape(raw, stretch_pauses=1.0, seed=1)
        assert out == raw and stats["skipped"] == 1


class TestRealPauses:
    async def test_a_sentence_pause_is_kept_when_only_joining(self) -> None:
        raw = _speech(0.2) + _quiet(0.45) + _speech(0.2)
        assert await _run([raw], join_gaps=1.0, seed=1) == raw

    @pytest.mark.parametrize("seed", range(1, 7))
    async def test_a_stretched_pause_grows_within_the_factor(self, seed: int) -> None:
        word = _speech(0.2)
        out, stats = _shape(word + _quiet(0.22) + word, stretch_pauses=1.0, seed=seed)
        pause = len(_between(out, word, word)) / SR
        assert stats["stretched"] == 1
        assert 0.22 * STRETCH_FACTOR[0] - 1 / SR <= pause <= 0.22 * STRETCH_FACTOR[1]

    @pytest.mark.parametrize("seed", range(1, 7))
    @pytest.mark.parametrize("secs", [0.30, 0.45, 0.54])
    async def test_a_lengthened_pause_never_exceeds_the_bound(self, secs: float, seed: int) -> None:
        """0.45 s times 2.3 is over a second: mid-reply that reads as the agent having finished."""
        word = _speech(0.2)
        out, _stats = _shape(word + _quiet(secs) + word, stretch_pauses=1.0, seed=seed)
        assert secs <= len(_between(out, word, word)) / SR <= LENGTHENED_MAX_SECS

    @pytest.mark.parametrize("secs", [0.56, 0.60, 0.85])
    async def test_a_pause_already_over_the_bound_is_left_alone(self, secs: float) -> None:
        raw = _speech(0.2) + _quiet(secs) + _speech(0.2)
        out, stats = _shape(raw, stretch_pauses=1.0, seed=1)
        assert out == raw
        assert stats["stretched"] == 0 and stats["skipped"] == 1

    async def test_the_bound_can_be_lowered(self) -> None:
        word = _speech(0.2)
        out, _stats = _shape(word + _quiet(0.30) + word, stretch_pauses=1.0, max_pause_secs=0.35, seed=1)
        assert len(_between(out, word, word)) == int(0.35 * SR)
        out, _stats = _shape(word + _quiet(0.16) + word, promote_gaps=1.0, max_pause_secs=0.25, seed=1)
        assert len(_between(out, word, word)) == int(0.25 * SR)
        # A bound under the silence itself never shortens it.
        raw = word + _quiet(0.16) + word
        assert _shape(raw, promote_gaps=1.0, max_pause_secs=0.10, seed=1)[0] == raw

    @pytest.mark.parametrize("dbfs", [-77.0, -58.0, -52.0, -47.0])
    async def test_room_tone_at_any_level_stays_within_the_bound(self, dbfs: float) -> None:
        """The pause limiter does not see room tone over its own level, so rhythm must not rely on it."""
        word = _speech(0.2)
        for seed in range(1, 6):
            out, _stats = _shape(
                word + _soft(0.30, dbfs=dbfs) + word, stretch_pauses=1.0, max_pause_secs=0.35, seed=seed
            )
            assert len(_between(out, word, word)) <= int(0.35 * SR)

    async def test_a_lengthened_pause_is_made_of_its_own_room_tone(self) -> None:
        word = _speech(0.2)
        tone = _room_tone(0.30)
        out, stats = _shape(word + tone + word, stretch_pauses=1.0, seed=1)
        pause = _between(out, word, word).astype(np.float64)
        before = np.frombuffer(tone, dtype="<i2").astype(np.float64)
        assert stats["stretched"] == 1 and len(pause) > len(before)
        # Same level as the original, no stretch of digital zeros, nothing louder than a crossfade can make it.
        assert np.sqrt((pause**2).mean()) == pytest.approx(np.sqrt((before**2).mean()), rel=0.25)
        frames = pause[: len(pause) // 160 * 160].reshape(-1, 160)
        assert np.sqrt((frames**2).mean(axis=1)).min() > 0.5 * np.sqrt((before**2).mean())
        assert np.abs(pause).max() <= np.abs(before).max() * 1.5

    @pytest.mark.parametrize("seed", range(1, 13))
    async def test_added_room_tone_does_not_loop(self, seed: int) -> None:
        """One 75 ms slice repeated end to end is periodic: over 0.4 s of it the added audio
        correlates with itself at 0.8 one slice later, and room tone that does that flutters."""
        word = _speech(0.2)
        out, stats = _shape(word + _room_tone(0.15) + word, promote_gaps=1.0, promoted_secs=(0.55, 0.55), seed=seed)
        assert stats["promoted"] == 1
        pause = _between(out, word, word).astype(np.float64)
        assert len(pause) == int(0.55 * SR)
        # The middle of the pause is what was added: the original 0.15 s is its two ends.
        half = int(0.15 * SR) // 2
        added = pause[half : len(pause) - half]
        added -= added.mean()
        full = np.correlate(added, added, mode="full")[len(added) - 1 :]
        # From 5 ms on: any shorter lag is the tone's own texture.
        lags = full[int(0.005 * SR) : len(added) // 2] / full[0]
        assert np.abs(lags).max() < 0.5

    @pytest.mark.parametrize("seed", range(1, 7))
    async def test_added_room_tone_is_joined_without_a_step(self, seed: int) -> None:
        """A hum under the level: a piece spliced in without a crossfade would jump mid-cycle."""
        word = _speech(0.2)
        t = np.arange(int(0.30 * SR))
        hum = (200 * np.sin(2 * np.pi * 50 * t / SR)).astype("<i2")
        out, stats = _shape(word + hum.tobytes() + word, stretch_pauses=1.0, seed=seed)
        assert stats["stretched"] == 1
        pause = _between(out, word, word).astype(np.int32)
        # A 50 Hz hum of amplitude 200 moves 4 per sample. A bare splice can jump 400.
        assert np.abs(np.diff(pause)).max() <= 3 * np.abs(np.diff(hum.astype(np.int32))).max()

    async def test_share_decides_how_many_pauses_are_stretched(self) -> None:
        word, pause = _speech(0.1), _quiet(0.30)
        shaper = PauseShaper(SR, stretch_pauses=0.25, seed=5)
        shaper.push(word + (pause + word) * 200)
        assert 25 <= shaper.stats["stretched"] <= 75


class TestEdges:
    @pytest.mark.parametrize("lead", [0.04, 0.30, 1.2])
    async def test_leading_silence_passes_unchanged(self, lead: float) -> None:
        raw = _quiet(lead) + _speech(0.2)
        assert await _run([raw], join_gaps=1.0, stretch_pauses=1.0, seed=1) == raw

    async def test_a_soft_opening_is_not_cut(self) -> None:
        """A reply can open under the frame level: an "s", an "h", a breath in."""
        raw = _soft(0.12) + _speech(0.35)
        assert await _run([raw], join_gaps=1.0, stretch_pauses=1.0, seed=1) == raw

    async def test_leading_silence_is_not_held(self) -> None:
        shaper = PauseShaper(SR, join_gaps=1.0, seed=1)
        assert shaper.push(_quiet(0.3)) == _quiet(0.3)
        assert shaper.holding is False

    async def test_sentences_synthesized_one_by_one_keep_the_pause_between_them(self) -> None:
        """Without a text stream each sentence has its own shaper, and the pause between two is the
        tail of one plus the lead of the next. Neither is internal, so neither is changed."""
        sentence = _quiet(0.25) + _speech(0.3) + _quiet(0.16) + _speech(0.3) + _quiet(0.20)
        first, _stats = _shape(sentence, join_gaps=1.0, stretch_pauses=1.0, seed=1)
        second, _stats = _shape(sentence, join_gaps=1.0, stretch_pauses=1.0, seed=2)
        assert first.startswith(_quiet(0.25) + _speech(0.3)) and first.endswith(_speech(0.3) + _quiet(0.20))
        assert second.startswith(_quiet(0.25) + _speech(0.3))
        assert len(first) < len(sentence)

    async def test_trailing_silence_leaves_unchanged(self) -> None:
        raw = _speech(0.2) + _quiet(0.35)
        assert await _run([raw], join_gaps=1.0, stretch_pauses=1.0, seed=1) == raw

    async def test_a_long_silence_is_released_without_waiting_for_speech(self) -> None:
        """A tool wait is not a pause. It must not sit in the filter until the next word."""
        shaper = PauseShaper(SR, join_gaps=1.0, stretch_pauses=1.0, seed=1)
        word = _speech(0.2)
        assert shaper.push(word) == word
        held = shaper.push(_quiet(MAX_HOLD_SECS - 0.1))
        assert held == b""
        released = shaper.push(_quiet(0.1))
        assert released == _quiet(MAX_HOLD_SECS)
        # Past that point silence passes straight through, and is not reshaped.
        assert shaper.push(_quiet(0.5)) == _quiet(0.5)
        assert shaper.push(word) == word
        assert shaper.stats["stretched"] == 0

    async def test_an_odd_trailing_byte_is_dropped(self) -> None:
        raw = _speech(0.2)
        out = await _run([raw + b"\x01"], join_gaps=1.0, seed=1)
        assert out == raw

    async def test_a_partial_last_frame_is_not_lost(self) -> None:
        raw = _speech(0.2) + _quiet(0.16) + _speech(0.203)
        out = await _run(_split(raw, 333), join_gaps=0.0, stretch_pauses=1.0, seed=1)
        assert out == raw


class TestStreaming:
    @pytest.mark.parametrize("size", [1, 2, 3, 160, 320, 701, 4096])
    async def test_chunk_size_does_not_change_the_result(self, size: int) -> None:
        raw = (
            _quiet(0.2)
            + _speech(0.25)
            + _quiet(0.16)
            + _speech(0.2)
            + _quiet(0.07)
            + _speech(0.15)
            + _quiet(0.4)
            + _speech(0.3)
            + _quiet(0.2)
        )
        kwargs = {"join_gaps": 0.9, "stretch_pauses": 0.8, "promote_gaps": 0.08, "seed": 4}
        whole, _stats = _shape(raw, **kwargs)
        shaper = PauseShaper(SR, **kwargs)
        assert b"".join([shaper.push(part) for part in _split(raw, size)]) + shaper.finish() == whole
        assert len(whole) != len(raw)

    @pytest.mark.parametrize("size", [320, 701, 4096])
    async def test_the_stream_gives_what_the_shaper_gives(self, size: int) -> None:
        raw = _speech(0.25) + _quiet(0.16) + _speech(0.2) + _quiet(0.4) + _speech(0.3) + _quiet(0.2)
        kwargs = {"join_gaps": 1.0, "stretch_pauses": 1.0, "seed": 4}
        assert await _run(_split(raw, size), **kwargs) == _shape(raw, **kwargs)[0]

    async def test_speech_is_yielded_before_the_next_chunk_is_read(self) -> None:
        """Only silence is held. A word must not wait for the audio after it."""
        word = _speech(0.2)
        pulled: list[int] = []
        seen: list[int] = []

        async def source():
            for index, part in enumerate([word, _quiet(0.16), word]):
                pulled.append(index)
                yield part

        async for _chunk in shape_pauses(source(), sample_rate=SR, join_gaps=1.0, seed=1):
            seen.append(len(pulled))
        # The first yield happened with one chunk read, not after the gap arrived.
        assert seen[0] == 1

    async def test_speech_samples_pass_through_bit_exact(self) -> None:
        first, second = _speech(0.2, freq=180.0), _speech(0.3, freq=310.0)
        out = await _run([first + _quiet(0.16) + second], join_gaps=1.0, seed=1)
        assert out.startswith(first)
        assert out.endswith(second)

    async def test_same_seed_gives_the_same_audio(self) -> None:
        raw = (_speech(0.1) + _quiet(0.16)) * 30 + _speech(0.1)
        kwargs = {"join_gaps": 0.5, "promote_gaps": 0.2, "seed": 9}
        assert await _run([raw], **kwargs) == await _run([raw], **kwargs)
        assert await _run([raw], **{**kwargs, "seed": 10}) != await _run([raw], **kwargs)

    async def test_other_sample_rates_keep_the_same_durations(self) -> None:
        for sr in (8_000, 24_000, 44_100):
            word = _speech(0.2, sr=sr)
            raw = word + _quiet(0.16, sr=sr) + word
            out = await _run([raw], sample_rate=sr, join_gaps=1.0, seed=1)
            removed = (len(raw) - len(out)) / 2 / sr
            assert 0.16 - JOINED_SECS[1] <= removed <= 0.16 - JOINED_SECS[0] + 1 / sr


class TestStalls:
    """A held silence costs nothing while the caller is still hearing earlier audio, and must not outlast that."""

    @staticmethod
    async def _timed(source, **kwargs) -> tuple[list[tuple[float, bytes]], float]:
        loop = asyncio.get_running_loop()
        start = loop.time()
        seen: list[tuple[float, bytes]] = []
        async for chunk in shape_pauses(source, sample_rate=SR, **kwargs):
            seen.append((loop.time() - start, chunk))
        return seen, start

    async def test_a_stalled_provider_does_not_keep_the_silence(self) -> None:
        """0.1 s of speech is heard, then nothing arrives for 0.5 s: the caller is waiting, not the filter."""
        word, resumed = _speech(0.1), []

        async def source():
            yield word + _quiet(0.2)
            await asyncio.sleep(0.5)
            resumed.append(asyncio.get_running_loop().time())
            yield _quiet(0.1) + word

        seen, start = await self._timed(source(), stretch_pauses=1.0, seed=1)
        raw = word + _quiet(0.2) + _quiet(0.1) + word
        # Nothing was lengthened on top of the wait, and the held silence left before the provider came back.
        assert b"".join(chunk for _at, chunk in seen) == raw
        before = b"".join(chunk for at, chunk in seen if at < resumed[0] - start)
        assert before == word + _quiet(0.2)
        assert [at for at, chunk in seen if chunk == _quiet(0.2)][0] < 0.35

    async def test_a_provider_ahead_of_playback_is_still_shaped(self) -> None:
        """0.6 s of speech is still playing when the next chunk arrives 0.2 s later: the pause is internal."""
        word = _speech(0.6)

        async def source():
            yield word + _quiet(0.2)
            await asyncio.sleep(0.2)
            yield _quiet(0.1) + word

        seen, _start = await self._timed(source(), stretch_pauses=1.0, seed=1)
        out = b"".join(chunk for _at, chunk in seen)
        assert 0.30 < len(_between(out, word, word)) / SR <= LENGTHENED_MAX_SECS

    async def test_a_faster_playback_rate_ends_the_wait_sooner(self) -> None:
        """At tempo 2 the 0.6 s of speech is heard in 0.3 s, so a chunk 0.45 s later comes after a stall."""
        word = _speech(0.6)

        async def source():
            yield word + _quiet(0.2)
            await asyncio.sleep(0.45)
            yield _quiet(0.1) + word

        raw = word + _quiet(0.3) + word
        seen, _start = await self._timed(source(), stretch_pauses=1.0, seed=1, playback_rate=2.0)
        assert b"".join(chunk for _at, chunk in seen) == raw
        seen, _start = await self._timed(source(), stretch_pauses=1.0, seed=1)
        assert len(b"".join(chunk for _at, chunk in seen)) > len(raw)

    async def test_silence_after_a_release_is_not_shaped(self) -> None:
        shaper = PauseShaper(SR, stretch_pauses=1.0, join_gaps=1.0, seed=1)
        word = _speech(0.2)
        assert shaper.push(word + _quiet(0.2)) == word
        assert shaper.holding is True
        assert shaper.release() == _quiet(0.2)
        assert shaper.holding is False and shaper.release() == b""
        # The rest of that silence passes as it comes. The next internal silence is shaped again.
        assert shaper.push(_quiet(0.3) + word) == _quiet(0.3) + word
        assert len(shaper.push(_quiet(0.16) + word)) < len(_quiet(0.16) + word)
        assert shaper.stats["released"] == 1 and shaper.stats["stretched"] == 0 and shaper.stats["joined"] == 1

    async def test_a_timeout_raised_by_the_provider_is_not_taken_for_a_stall(self) -> None:
        async def source():
            yield _speech(0.2) + _quiet(0.2)
            raise TimeoutError("provider read timed out")

        with pytest.raises(TimeoutError, match="provider read timed out"):
            await asyncio.wait_for(_run_source(source(), stretch_pauses=1.0, seed=1), timeout=2.0)

    async def test_closing_the_stream_closes_the_provider(self) -> None:
        closed = asyncio.Event()

        async def source():
            try:
                yield _speech(0.2) + _quiet(0.2)
                await asyncio.sleep(30)
            finally:
                closed.set()

        stream = shape_pauses(source(), sample_rate=SR, stretch_pauses=1.0, seed=1)
        assert await anext(stream) == _speech(0.2)
        await stream.aclose()
        assert closed.is_set()

    async def test_what_was_done_is_logged_once_per_stream(self) -> None:
        raw = _speech(0.2) + _quiet(0.16) + _speech(0.2) + _quiet(0.30) + _speech(0.2) + _quiet(0.05) + _speech(0.2)
        with capture_logs() as logs:
            await _run([raw], join_gaps=1.0, stretch_pauses=1.0, seed=1)
        (event,) = [entry for entry in logs if entry["event"] == "tts_rhythm_shaped"]
        assert event["log_level"] == "info"
        assert (event["gaps"], event["joined"], event["stretched"], event["released"]) == (3, 1, 1, 0)

    async def test_a_stream_with_no_internal_silence_logs_nothing(self) -> None:
        with capture_logs() as logs:
            await _run([_quiet(0.2) + _speech(0.4) + _quiet(0.2)], join_gaps=1.0, seed=1)
        assert not [entry for entry in logs if entry["event"] == "tts_rhythm_shaped"]


class TestQuietLevel:
    async def test_room_tone_under_the_level_counts_as_silence(self) -> None:
        word = _speech(0.2)
        raw = word + _room_tone(0.16) + word
        assert len(await _run([raw], join_gaps=1.0, seed=1)) < len(raw)

    async def test_a_breath_over_the_level_is_speech(self) -> None:
        """-30 dBFS is over the default -42: it is audio, so the gap is not a gap."""
        breath = np.full(int(0.16 * SR), int(32768 * 10 ** (-30 / 20)), dtype="<i2").tobytes()
        raw = _speech(0.2) + breath + _speech(0.2)
        assert await _run([raw], join_gaps=1.0, seed=1) == raw
        assert len(await _run([raw], join_gaps=1.0, quiet_dbfs=-25.0, seed=1)) < len(raw)


class TestValidation:
    @pytest.mark.parametrize("field", ["join_gaps", "stretch_pauses", "promote_gaps"])
    @pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.5", float("nan")])
    def test_shares_are_numbers_from_zero_to_one(self, field: str, value) -> None:
        with pytest.raises(ValueError):
            PauseShaper(SR, **{field: value})

    def test_join_and_promote_cannot_exceed_every_gap(self) -> None:
        with pytest.raises(ValueError, match="together at most 1"):
            PauseShaper(SR, join_gaps=0.8, promote_gaps=0.3)

    def test_gap_bounds_must_be_ordered(self) -> None:
        with pytest.raises(ValueError):
            PauseShaper(SR, join_gaps=0.5, gap_min_secs=0.3, gap_max_secs=0.2)

    @pytest.mark.parametrize("rate", [0, -16000, True, 16000.0])
    def test_sample_rate_is_a_positive_integer(self, rate) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            PauseShaper(rate, join_gaps=0.5)

    def test_stretch_factor_never_shortens(self) -> None:
        with pytest.raises(ValueError):
            PauseShaper(SR, stretch_pauses=0.5, stretch_factor=(0.5, 1.2))

    @pytest.mark.parametrize("level", [0.0, -10.0, -19.9, -90.1, float("nan"), float("-inf"), True, "-42"])
    def test_quiet_level_is_a_level_silence_can_have(self, level) -> None:
        """At 0 dBFS every frame is silence: the whole reply would be held and reshaped."""
        with pytest.raises(ValueError, match="quiet_dbfs"):
            PauseShaper(SR, join_gaps=0.5, quiet_dbfs=level)

    def test_a_gap_cannot_be_longer_than_what_is_held(self) -> None:
        """A silence of ``MAX_HOLD_SECS`` is passed on before its end is known, so it can never be sorted."""
        with pytest.raises(ValueError, match="gap_max_secs"):
            PauseShaper(SR, join_gaps=0.5, gap_max_secs=MAX_HOLD_SECS + 0.1)
        assert PauseShaper(SR, join_gaps=0.5, gap_max_secs=MAX_HOLD_SECS).enabled

    @pytest.mark.parametrize("bound", [-0.1, float("nan"), True, "0.5"])
    def test_the_pause_bound_is_a_non_negative_number(self, bound) -> None:
        with pytest.raises(ValueError, match="max_pause_secs"):
            PauseShaper(SR, stretch_pauses=0.5, max_pause_secs=bound)

    @pytest.mark.parametrize("rate", [0, -1.0, float("nan"), True])
    async def test_playback_rate_is_positive(self, rate) -> None:
        with pytest.raises(ValueError, match="playback_rate"):
            await _run([_speech(0.2)], join_gaps=1.0, playback_rate=rate)


class TestConfig:
    def test_unset_and_empty_are_off(self) -> None:
        assert VoiceConfig().tts_rhythm is None
        assert VoiceConfig(tts_rhythm={}).tts_rhythm is None
        assert VoiceConfig(tts_rhythm=False).tts_rhythm is None

    def test_true_selects_the_measured_preset(self) -> None:
        picked = VoiceConfig(tts_rhythm=True).tts_rhythm
        assert picked is not None and picked.enabled
        assert picked.model_dump(include=set(NATURAL_SPEECH_RHYTHM)) == NATURAL_SPEECH_RHYTHM

    def test_a_block_with_no_share_is_not_enabled(self) -> None:
        assert SpeechRhythm(gap_max_secs=0.26).enabled is False

    def test_every_block_the_config_accepts_builds_a_shaper(self) -> None:
        """The limits are written twice, in the model and in the shaper. They must agree at the edges."""
        for block in (
            {"join_gaps": 1.0, "gap_min_secs": 0.0, "gap_max_secs": 0.0, "quiet_dbfs": -90.0},
            {"stretch_pauses": 1.0, "gap_min_secs": 0.9, "gap_max_secs": 0.9, "quiet_dbfs": -20.0},
        ):
            picked = SpeechRhythm(**block)
            assert PauseShaper(SR, **picked.model_dump(exclude={"seed"})).enabled

    @pytest.mark.parametrize(
        "block",
        [
            {"join_gaps": 1.5},
            {"join_gaps": True},
            {"join_gaps": 0.8, "promote_gaps": 0.3},
            {"gap_min_secs": 0.3, "gap_max_secs": 0.2},
            {"join_gaps": 0.5, "gap_max_secs": 1.5},
            {"join_gaps": 0.5, "quiet_dbfs": 0},
            {"join_gaps": 0.5, "quiet_dbfs": -10},
            {"join_gaps": 0.5, "quiet_dbfs": -120},
            {"unknown": 1},
        ],
    )
    def test_bad_blocks_fail_when_the_agent_is_built(self, block: dict) -> None:
        with pytest.raises(ValueError):
            VoiceConfig(tts_rhythm=block)


def test_module_imports_without_numpy_at_import_time() -> None:
    """``timbal.voice`` must stay importable without the voice extra: numpy loads on first use."""
    tree = ast.parse(inspect.getsource(rhythm))
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name for node in top_level for alias in node.names} | {
        node.module for node in top_level if isinstance(node, ast.ImportFrom)
    }
    assert "numpy" not in names
