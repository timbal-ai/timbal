"""Tests for the Silero VAD ONNX streaming wrapper (``timbal[voice]`` extra)."""

import os

import pytest

np = pytest.importorskip("numpy", reason="timbal[voice] extra not installed")
pytest.importorskip("onnxruntime", reason="timbal[voice] extra not installed")

from timbal.voice.vad import (  # noqa: E402
    _CONTEXT_SAMPLES,
    _VAD_SAMPLE_RATE,
    FRAME_SAMPLES,
    SileroVad,
)


class _FakeVadOrtSession:
    """Stands in for the Silero ort session; records inputs, returns fixed probs."""

    def __init__(self, probability: float = 0.7):
        self.probability = probability
        self.calls: list[dict] = []

    def run(self, output_names, feeds):
        assert output_names is None
        assert feeds["input"].shape == (1, _CONTEXT_SAMPLES + FRAME_SAMPLES)
        assert feeds["state"].shape == (2, 1, 128)
        assert feeds["sr"].dtype == np.int64
        self.calls.append({k: v.copy() for k, v in feeds.items()})
        return [
            np.array([[self.probability]], dtype=np.float32),
            np.ones((2, 1, 128), dtype=np.float32) * len(self.calls),
        ]


def _vad_with_fake(probability: float = 0.7, sample_rate: int = _VAD_SAMPLE_RATE) -> tuple[SileroVad, _FakeVadOrtSession]:
    vad = SileroVad()
    fake = _FakeVadOrtSession(probability)
    vad._session = fake
    vad._sample_rate = sample_rate
    return vad, fake


def _pcm(n_samples: int, value: int = 1000) -> bytes:
    return np.full(n_samples, value, dtype=np.int16).tobytes()


class TestFraming:
    def test_exact_frame_yields_one_prob(self):
        vad, fake = _vad_with_fake(0.9)
        probs = vad.process(_pcm(FRAME_SAMPLES))
        assert probs == [pytest.approx(0.9)]
        assert len(fake.calls) == 1

    def test_partial_chunks_buffer_until_full_frame(self):
        vad, fake = _vad_with_fake()
        assert vad.process(_pcm(FRAME_SAMPLES // 2)) == []
        assert len(fake.calls) == 0
        probs = vad.process(_pcm(FRAME_SAMPLES // 2))
        assert len(probs) == 1
        assert len(fake.calls) == 1

    def test_large_chunk_yields_multiple_probs(self):
        vad, fake = _vad_with_fake()
        probs = vad.process(_pcm(FRAME_SAMPLES * 3 + 10))
        assert len(probs) == 3
        assert len(fake.calls) == 3
        # The 10 leftover samples stay buffered.
        assert vad._buf.size == 10

    def test_empty_and_odd_byte_input(self):
        vad, _ = _vad_with_fake()
        assert vad.process(b"") == []
        assert vad.process(b"\x01") == []  # odd trailing byte dropped

    def test_process_before_start_raises(self):
        vad = SileroVad()
        with pytest.raises(RuntimeError):
            vad.process(_pcm(FRAME_SAMPLES))


class TestContextAndState:
    def test_first_frame_has_zero_context(self):
        vad, fake = _vad_with_fake()
        vad.process(_pcm(FRAME_SAMPLES, value=1000))
        first_input = fake.calls[0]["input"][0]
        assert not first_input[:_CONTEXT_SAMPLES].any()
        assert first_input[_CONTEXT_SAMPLES:].all()

    def test_context_carries_previous_frame_tail(self):
        vad, fake = _vad_with_fake()
        vad.process(_pcm(FRAME_SAMPLES, value=1000))
        vad.process(_pcm(FRAME_SAMPLES, value=2000))
        second_input = fake.calls[1]["input"][0]
        expected_ctx = 1000 / 32768.0
        assert second_input[:_CONTEXT_SAMPLES] == pytest.approx(expected_ctx, abs=1e-6)

    def test_recurrent_state_threads_between_calls(self):
        vad, fake = _vad_with_fake()
        vad.process(_pcm(FRAME_SAMPLES))
        vad.process(_pcm(FRAME_SAMPLES))
        # Second call must receive the state returned by the first (all ones).
        assert fake.calls[1]["state"].mean() == pytest.approx(1.0)

    def test_reset_clears_state_context_and_buffer(self):
        vad, fake = _vad_with_fake()
        vad.process(_pcm(FRAME_SAMPLES + 100))
        vad.reset()
        assert vad._buf.size == 0
        vad.process(_pcm(FRAME_SAMPLES))
        assert not fake.calls[-1]["input"][0][:_CONTEXT_SAMPLES].any()
        assert not fake.calls[-1]["state"].any()


class TestResampling:
    def test_48k_stride_down_to_16k(self):
        vad, fake = _vad_with_fake(sample_rate=48_000)
        # 3x the frame at 48k → exactly one 512-sample frame at 16k.
        probs = vad.process(_pcm(FRAME_SAMPLES * 3))
        assert len(probs) == 1
        assert len(fake.calls) == 1

    def test_44100_no_drift_over_many_chunks(self):
        vad, _ = _vad_with_fake(sample_rate=44_100)
        chunk = _pcm(4410)  # 100ms at 44.1k → 1600 samples at 16k
        total_probs = 0
        for _ in range(20):
            total_probs += len(vad.process(chunk))
        # 2 seconds of audio → 32000 samples at 16k → 62 full frames (+256 buffered).
        produced = total_probs * FRAME_SAMPLES + vad._buf.size
        assert produced == 32_000

    def test_16k_passthrough_no_counters(self):
        vad, _ = _vad_with_fake()
        vad.process(_pcm(FRAME_SAMPLES))
        assert vad._in_total == 0 and vad._out_total == 0


@pytest.mark.skipif(
    not os.environ.get("TIMBAL_SMART_TURN_INTEGRATION"),
    reason="set TIMBAL_SMART_TURN_INTEGRATION=1 to download and run the real checkpoint",
)
class TestRealSileroCheckpoint:
    async def test_silence_scores_low(self):
        vad = SileroVad()
        await vad.start(sample_rate=_VAD_SAMPLE_RATE)
        probs = vad.process(b"\x00\x00" * (FRAME_SAMPLES * 10))
        assert len(probs) == 10
        assert max(probs) < 0.3
