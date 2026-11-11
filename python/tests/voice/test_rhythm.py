"""Pause rhythm: which silences are joined, kept or lengthened, and that speech is never held."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
from timbal.voice import rhythm
from timbal.voice.config import NATURAL_SPEECH_RHYTHM, SpeechRhythm, VoiceConfig
from timbal.voice.rhythm import (
    FRAME_SECS,
    JOINED_SECS,
    LEAD_SECS,
    MAX_HOLD_SECS,
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


async def _chunks(parts: list[bytes]):
    for part in parts:
        yield part


async def _run(parts: list[bytes], **kwargs) -> bytes:
    kwargs.setdefault("sample_rate", SR)
    out: list[bytes] = []
    async for chunk in shape_pauses(_chunks(parts), **kwargs):
        out.append(chunk)
    return b"".join(out)


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


class TestRealPauses:
    async def test_a_sentence_pause_is_kept_when_only_joining(self) -> None:
        raw = _speech(0.2) + _quiet(0.45) + _speech(0.2)
        assert await _run([raw], join_gaps=1.0, seed=1) == raw

    async def test_a_stretched_pause_grows_within_the_factor(self) -> None:
        word = _speech(0.2)
        out = await _run([word + _quiet(0.30) + word], stretch_pauses=1.0, seed=1)
        (pause,) = _internal(out)
        assert 0.30 * STRETCH_FACTOR[0] - 2 * FRAME_SECS <= pause <= 0.30 * STRETCH_FACTOR[1] + 2 * FRAME_SECS

    async def test_stretch_repeats_the_room_tone_and_adds_no_zeros(self) -> None:
        word = _speech(0.2)
        tone = _room_tone(0.30)
        out = await _run([word + tone + word], stretch_pauses=1.0, seed=1)
        samples = np.frombuffer(out, dtype="<i2")
        start = len(word) // 2
        pause = samples[start : len(samples) - len(word) // 2]
        assert len(pause) > len(tone) // 2
        # Every sample of the longer pause comes from the original one.
        assert set(pause.tolist()) <= set(np.frombuffer(tone, dtype="<i2").tolist())

    async def test_share_decides_how_many_pauses_are_stretched(self) -> None:
        word, pause = _speech(0.1), _quiet(0.30)
        shaper = PauseShaper(SR, stretch_pauses=0.25, seed=5)
        shaper.push(word + (pause + word) * 200)
        assert 25 <= shaper.stats["stretched"] <= 75


class TestEdges:
    async def test_leading_air_is_cut_to_the_lead(self) -> None:
        word = _speech(0.2)
        out = await _run([_quiet(0.30) + word], join_gaps=1.0, seed=1)
        assert len(out) == len(_quiet(LEAD_SECS)) + len(word)
        assert out.endswith(word)

    async def test_a_short_lead_is_kept_whole(self) -> None:
        raw = _quiet(0.04) + _speech(0.2)
        assert await _run([raw], join_gaps=1.0, seed=1) == raw

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
        whole = await _run([raw], **kwargs)
        assert await _run(_split(raw, size), **kwargs) == whole

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

    @pytest.mark.parametrize(
        "block",
        [
            {"join_gaps": 1.5},
            {"join_gaps": True},
            {"join_gaps": 0.8, "promote_gaps": 0.3},
            {"gap_min_secs": 0.3, "gap_max_secs": 0.2},
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
