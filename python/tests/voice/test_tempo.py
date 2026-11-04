"""Pitch-preserving tempo: speed ratio, 20 ms framing, 0.25 s idle flush."""

from __future__ import annotations

import ast
import asyncio
import inspect
import math

import pytest

np = pytest.importorskip("numpy", reason="timbal[voice] extra (numpy) not installed")
pytest.importorskip("av", reason="timbal[voice] extra (av) not installed")

from timbal.voice.tempo import (  # noqa: E402
    FRAME_MS,
    IDLE_FLUSH_SECS,
    MAX_BUFFER_MS,
    atempo_stages,
    change_tempo,
    drain_frames,
    frame_bytes,
)

SR = 16_000


def _sine(samples: int, freq: float = 440.0, amp: int = 8000, sr: int = SR) -> bytes:
    t = np.arange(samples)
    wave = amp * np.sin(2 * np.pi * freq * t / sr)
    return wave.astype("<i2").tobytes()


def _peak_hz(pcm: bytes, sr: int = SR) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    # Skip the filter's fade-in and use a steady slice.
    start = min(len(samples) // 5, sr // 20)
    window = samples[start : start + sr // 5]
    if len(window) < 64:
        window = samples
    spec = np.abs(np.fft.rfft(window * np.hanning(len(window))))
    spec[0] = 0
    freqs = np.fft.rfftfreq(len(window), 1 / sr)
    return float(freqs[int(np.argmax(spec))])


async def _chunks(parts: list[bytes]):
    for part in parts:
        yield part


async def _run(parts: list[bytes], tempo: float, sr: int = SR) -> bytes:
    out: list[bytes] = []
    async for chunk in change_tempo(_chunks(parts), sample_rate=sr, tempo=tempo):
        out.append(chunk)
    return b"".join(out)


class TestFraming:
    def test_twenty_ms_blocks_and_fifty_ms_cap(self) -> None:
        assert FRAME_MS == 20
        assert MAX_BUFFER_MS == 50
        frame = frame_bytes(SR, FRAME_MS)
        cap = frame_bytes(SR, MAX_BUFFER_MS)
        assert frame == 640  # 320 samples, 20 ms at 16 kHz
        assert cap == 1600

        buf = bytearray(b"\x01\x00" * (frame // 2 - 16))  # 19 ms
        assert drain_frames(buf, frame_bytes=frame, max_buffer_bytes=cap, force=False) == []
        assert len(buf) == frame - 32

        buf.extend(b"\x01\x00" * 16)
        frames = drain_frames(buf, frame_bytes=frame, max_buffer_bytes=cap, force=False)
        assert frames == [b"\x01\x00" * (frame // 2)]
        assert buf == bytearray()

        # 45 ms → two 20 ms blocks, 5 ms held. Not one blob, and not 50 ms.
        buf = bytearray(b"\x02\x00" * int(0.045 * SR))
        frames = drain_frames(buf, frame_bytes=frame, max_buffer_bytes=cap, force=False)
        assert [len(frame_i) for frame_i in frames] == [frame, frame]
        assert 0 < len(buf) < frame

        tail = drain_frames(buf, frame_bytes=frame, max_buffer_bytes=cap, force=True)
        assert len(tail) == 1 and len(tail[0]) == int(0.045 * SR) * 2 - 2 * frame
        assert buf == bytearray()

    def test_overflow_past_fifty_ms_is_not_held_for_a_full_block(self) -> None:
        cap = frame_bytes(SR, MAX_BUFFER_MS)
        buf = bytearray(b"\x03\x00" * (cap // 2))
        frames = drain_frames(buf, frame_bytes=cap + 100, max_buffer_bytes=cap, force=False)
        assert [len(frame) for frame in frames] == [cap]
        assert buf == bytearray()

    def test_eighty_ms_is_four_blocks(self) -> None:
        frame = frame_bytes(SR, FRAME_MS)
        cap = frame_bytes(SR, MAX_BUFFER_MS)
        buf = bytearray(b"\x04\x00" * int(0.08 * SR))
        frames = drain_frames(buf, frame_bytes=frame, max_buffer_bytes=cap, force=False)
        assert [len(frame_i) for frame_i in frames] == [frame, frame, frame, frame]
        assert buf == bytearray()

    def test_stages_multiply_back_to_the_tempo(self) -> None:
        for tempo in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0):
            stages = atempo_stages(tempo)
            assert stages
            assert all(0.5 - 1e-9 <= stage <= 2.0 + 1e-9 for stage in stages)
            product = math.prod(stages)
            assert product == pytest.approx(tempo)


class TestSpeedAndPitch:
    @pytest.mark.parametrize("tempo", [0.5, 1.5, 2.0])
    async def test_duration_scales_and_pitch_stays(self, tempo: float) -> None:
        samples = int(0.6 * SR)
        raw = _sine(samples)
        out = await _run([raw], tempo)
        got = len(out) // 2
        expect = samples / tempo
        assert got == pytest.approx(expect, rel=0.08)
        peak = _peak_hz(out)
        assert peak == pytest.approx(440.0, abs=25)
        # A naive resample would move the peak to 440*tempo.
        assert abs(peak - 440.0) < abs(peak - 440.0 * tempo)

    async def test_tempo_one_is_byte_exact(self) -> None:
        raw = _sine(int(0.2 * SR)) + b"\x00"  # odd trailing byte is preserved, not filtered
        out = await _run([raw[:100], raw[100:]], 1.0)
        assert out == raw

    async def test_rejects_bad_tempo(self) -> None:
        with pytest.raises(ValueError):
            await _run([_sine(160)], 0)
        with pytest.raises(ValueError):
            await _run([_sine(160)], -1.0)
        with pytest.raises(ValueError):
            async for _ in change_tempo(_chunks([b"\x00\x00"]), sample_rate=0, tempo=1.5):
                pass


class TestIdleFlush:
    def test_idle_flush_is_a_quarter_second(self) -> None:
        assert IDLE_FLUSH_SECS == 0.25

    async def test_stall_flushes_the_phrase_before_the_next_chunk(self) -> None:
        first = _sine(int(0.40 * SR))
        second = _sine(int(0.40 * SR), freq=660)
        release = asyncio.Event()
        second_started = asyncio.Event()

        async def source():
            yield first
            await release.wait()
            second_started.set()
            yield second

        got = bytearray()

        async def consume() -> None:
            async for chunk in change_tempo(source(), sample_rate=SR, tempo=2.0):
                got.extend(chunk)
                if not release.is_set() and len(got) // 2 >= int(0.40 * SR / 2 * 0.85):
                    # Tail of the first phrase arrived while the source was stalled.
                    assert not second_started.is_set()
                    release.set()

        await asyncio.wait_for(consume(), timeout=2.0)
        assert release.is_set()
        assert second_started.is_set()
        expect = (len(first) + len(second)) / 2 / 2.0
        assert len(got) // 2 == pytest.approx(expect, rel=0.08)

    async def test_flush_does_not_wait_when_the_stream_ends(self) -> None:
        raw = _sine(int(0.30 * SR))
        started = asyncio.get_running_loop().time()
        out = await _run([raw], 2.0)
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 0.2
        assert len(out) // 2 == pytest.approx(len(raw) / 2 / 2.0, rel=0.08)


class TestLazyImport:
    def test_av_is_not_imported_at_module_level(self) -> None:
        import timbal.voice.tempo as tempo

        tree = ast.parse(inspect.getsource(tempo))
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name == "av" or name.startswith("av.") for name in names)
