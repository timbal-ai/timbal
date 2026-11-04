"""Pause trimmer: thresholds, dBFS, chunk boundaries, transients, first-byte latency."""

from __future__ import annotations

import ast
import inspect

import pytest

np = pytest.importorskip("numpy", reason="timbal[voice] extra (numpy) not installed")

from timbal.voice.pauses import ANALYSIS_WINDOW_SECS, limit_pauses  # noqa: E402

SR = 16_000


def _const(samples: int, amp: int) -> bytes:
    return np.full(samples, amp, dtype="<i2").tobytes()


def _sine(samples: int, freq: float = 440.0, amp: int = 8000, sr: int = SR) -> bytes:
    t = np.arange(samples)
    wave = amp * np.sin(2 * np.pi * freq * t / sr)
    return wave.astype("<i2").tobytes()


def _dbfs(db: float) -> int:
    return int(round(32768 * 10 ** (db / 20)))


async def _chunks(parts: list[bytes]):
    for part in parts:
        yield part


async def _run(parts: list[bytes], **kwargs) -> bytes:
    kwargs.setdefault("sample_rate", SR)
    out: list[bytes] = []
    async for chunk in limit_pauses(_chunks(parts), **kwargs):
        out.append(chunk)
    return b"".join(out)


def _split(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)] or [b""]


class TestThresholds:
    async def test_pause_longer_than_the_cap_is_cut_to_the_cap(self) -> None:
        speech = _const(int(0.05 * SR), 10_000)
        gap = _const(int(0.50 * SR), 0)
        out = await _run([speech, gap, speech], max_pause_secs=0.20)
        kept = _const(int(round(0.20 * SR)), 0)
        assert out == speech + kept + speech

    async def test_pause_shorter_than_the_cap_is_kept(self) -> None:
        speech = _const(int(0.05 * SR), 10_000)
        gap = _const(int(0.10 * SR), 0)
        raw = speech + gap + speech
        assert await _run([raw], max_pause_secs=0.20) == raw

    async def test_each_internal_pause_is_capped_on_its_own(self) -> None:
        speech = _const(int(0.04 * SR), 12_000)
        gap = _const(int(0.40 * SR), 0)
        kept = _const(int(round(0.12 * SR)), 0)
        out = await _run([speech, gap, speech, gap, speech], max_pause_secs=0.12)
        assert out == speech + kept + speech + kept + speech

    async def test_sentence_gap_of_about_half_a_second_loses_the_excess(self) -> None:
        """The gap this targets is ~0.52 s after a period. Capping at 0.12 s removes 0.40 s."""
        speech = _const(int(0.05 * SR), 10_000)
        gap = _const(int(round(0.52 * SR)), 0)
        out = await _run([speech, gap, speech], max_pause_secs=0.12)
        removed = (len(speech) * 2 + len(gap) - len(out)) / 2 / SR
        assert removed == pytest.approx(0.40, abs=0.001)

    async def test_zero_cap_still_keeps_the_window_that_overlaps_speech(self) -> None:
        speech = _const(int(0.05 * SR), 10_000)
        gap = _const(int(0.30 * SR), 0)
        out = await _run([speech, gap, speech], max_pause_secs=0)
        guard = max(0, int(round(SR * ANALYSIS_WINDOW_SECS)) - 1)
        assert out == speech + _const(guard, 0) + speech


class TestDbfs:
    async def test_breath_level_is_not_silence(self) -> None:
        # -40 dBFS is above the -45 default: a breath, not a gap.
        breath = _const(int(0.5 * SR), _dbfs(-40))
        assert await _run([breath], max_pause_secs=0.05) == breath

    async def test_level_below_the_threshold_is_trimmed(self) -> None:
        quiet = _const(int(0.5 * SR), _dbfs(-55))
        out = await _run([quiet], max_pause_secs=0.10)
        assert len(out) // 2 == int(round(0.10 * SR))

    async def test_custom_threshold_keeps_a_softer_breath(self) -> None:
        # -50 dBFS is silence at the default -45, and a breath at -60.
        soft = _const(int(0.4 * SR), _dbfs(-50))
        assert await _run([soft], max_pause_secs=0.05, quiet_dbfs=-45.0) != soft
        assert await _run([soft], max_pause_secs=0.05, quiet_dbfs=-60.0) == soft

    async def test_breath_inside_a_long_gap_survives(self) -> None:
        low = _const(int(0.40 * SR), _dbfs(-70))
        breath = _const(int(0.15 * SR), _dbfs(-40))
        out = await _run([low, breath, low], max_pause_secs=0.05)
        assert breath in out


class TestChunkBoundaries:
    async def test_splits_match_a_single_buffer(self) -> None:
        speech = _const(int(0.05 * SR), 9_000)
        gap = _const(int(0.35 * SR), 0)
        blob = speech + gap + speech
        whole = await _run([blob], max_pause_secs=0.08)
        for size in (1, 2, 3, 7, 64, 1000, 4096):
            assert await _run(_split(blob, size), max_pause_secs=0.08) == whole

    async def test_empty_and_odd_byte_inputs(self) -> None:
        speech = _const(80, 8_000)
        assert await _run([b"", speech, b""], max_pause_secs=0.1) == speech
        assert await _run([b"\x00"], max_pause_secs=0.1) == b""


class TestTransients:
    async def test_sine_wave_is_not_chopped_at_zero_crossings(self) -> None:
        sine = _sine(int(0.30 * SR), freq=440, amp=8_000)
        assert await _run([sine], max_pause_secs=0.05) == sine

    async def test_low_voice_sine_is_not_chopped(self) -> None:
        sine = _sine(int(0.30 * SR), freq=85, amp=8_000)
        assert await _run([sine], max_pause_secs=0.05) == sine

    async def test_onset_after_a_long_gap_is_intact(self) -> None:
        # Cosine, so the first sample is the peak. A clipped attack would drop it.
        t = np.arange(int(0.20 * SR))
        tone = (8_000 * np.cos(2 * np.pi * 440 * t / SR)).astype("<i2").tobytes()
        gap = _const(int(0.80 * SR), 0)
        out = await _run([gap, tone], max_pause_secs=0.10)
        assert out.endswith(tone)
        assert len(out) == int(round(0.10 * SR)) * 2 + len(tone)

    async def test_impulse_after_silence_is_kept(self) -> None:
        gap = _const(int(0.50 * SR), 0)
        impulse = _const(1, 20_000)
        tail = _const(int(0.05 * SR), 0)
        out = await _run([gap, impulse, tail], max_pause_secs=0.10)
        samples = np.frombuffer(out, dtype="<i2")
        assert int(np.count_nonzero(samples == 20_000)) == 1


class TestFirstAudio:
    async def test_speech_is_yielded_before_the_following_pause_is_read(self) -> None:
        speech = _const(int(0.05 * SR), 10_000)
        gap = _const(int(0.60 * SR), 0)
        pulled: list[str] = []

        async def chunks():
            pulled.append("speech")
            yield speech
            pulled.append("pause")
            yield gap

        yielded = False
        async for chunk in limit_pauses(chunks(), sample_rate=SR, max_pause_secs=0.10):
            assert chunk == speech
            assert pulled == ["speech"]
            yielded = True
            break
        assert yielded


class TestLazyImport:
    def test_numpy_is_not_imported_at_module_level(self) -> None:
        import timbal.voice.pauses as pauses

        tree = ast.parse(inspect.getsource(pauses))
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name == "numpy" or name.startswith("numpy.") for name in names)

    async def test_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError):
            await _run([b"\x00\x00"], sample_rate=0, max_pause_secs=0.1)
        with pytest.raises(ValueError):
            await _run([b"\x00\x00"], max_pause_secs=-0.1)
        with pytest.raises(ValueError):
            await _run([b"\x00\x00"], max_pause_secs=float("nan"))
