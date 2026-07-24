"""Tests for the Smart Turn v3 ONNX EOU backend (``timbal[voice]`` extra)."""

import os
import sys

import pytest

np = pytest.importorskip("numpy", reason="timbal[voice] extra not installed")
pytest.importorskip("onnxruntime", reason="timbal[voice] extra not installed")

from timbal.voice import LocalAudioTurnDetector, resolve_turn_detector  # noqa: E402
from timbal.voice import turn_detection as turn_detection_module  # noqa: E402
from timbal.voice._whisper_features import compute_whisper_log_mel_features  # noqa: E402
from timbal.voice.smart_turn import (  # noqa: E402
    _MODEL_SAMPLE_RATE,
    _MODEL_SAMPLES,
    _TAIL_SILENCE_SECS,
    DEFAULT_FILENAME,
    QUANTIZED_FILENAME,
    SmartTurnEouModel,
)


class _FakeOrtSession:
    """Stands in for ort.InferenceSession; records inputs, returns a fixed probability."""

    def __init__(self, probability: float = 0.9):
        self.probability = probability
        self.calls: list[dict] = []

    def run(self, output_names, feeds):
        assert output_names is None
        self.calls.append(feeds)
        return [np.array([[self.probability]], dtype=np.float32)]


def _pcm_silence(seconds: float, sample_rate: int = _MODEL_SAMPLE_RATE) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


def _pcm_tone(seconds: float, sample_rate: int = _MODEL_SAMPLE_RATE, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    samples = (np.sin(2 * np.pi * freq * t) * 12_000).astype(np.int16)
    return samples.tobytes()


class TestPrepareAudio:
    def test_short_audio_left_padded(self):
        pcm = _pcm_tone(1.0)
        audio = SmartTurnEouModel._prepare_audio(pcm, _MODEL_SAMPLE_RATE)
        assert audio.shape == (_MODEL_SAMPLES,)
        assert audio.dtype == np.float32
        # Padding zeros at the beginning, signal at the end.
        pad = _MODEL_SAMPLES - int(1.0 * _MODEL_SAMPLE_RATE)
        assert not audio[:pad].any()
        assert audio[pad:].any()

    def test_long_audio_keeps_tail(self):
        # 10s ramp: the tail has the highest values, so after truncation the
        # first retained sample must correspond to t = 2s.
        n = 10 * _MODEL_SAMPLE_RATE
        samples = np.linspace(0, 32_000, n).astype(np.int16)
        audio = SmartTurnEouModel._prepare_audio(samples.tobytes(), _MODEL_SAMPLE_RATE)
        assert audio.shape == (_MODEL_SAMPLES,)
        expected_first = samples[n - _MODEL_SAMPLES] / 32768.0
        assert audio[0] == pytest.approx(expected_first, abs=1e-4)

    def test_resamples_non_16k(self):
        pcm = _pcm_tone(2.0, sample_rate=8_000)
        audio = SmartTurnEouModel._prepare_audio(pcm, 8_000)
        assert audio.shape == (_MODEL_SAMPLES,)
        # 2s at 8k → 2s at 16k = 32k samples of signal at the end.
        pad = _MODEL_SAMPLES - 2 * _MODEL_SAMPLE_RATE
        assert not audio[:pad].any()
        assert audio[pad:].any()

    def test_odd_byte_count_tolerated(self):
        audio = SmartTurnEouModel._prepare_audio(b"\x00\x01\x02", _MODEL_SAMPLE_RATE)
        assert audio.shape == (_MODEL_SAMPLES,)

    def test_trailing_silence_trimmed(self):
        # 2s tone + 3s silence: the model must see the tone ending ~0.2s before
        # the window end, not 3s of dead air (STT commit debounce).
        pcm = _pcm_tone(2.0) + _pcm_silence(3.0)
        audio = SmartTurnEouModel._prepare_audio(pcm, _MODEL_SAMPLE_RATE)
        assert audio.shape == (_MODEL_SAMPLES,)
        tail_keep = int(_TAIL_SILENCE_SECS * _MODEL_SAMPLE_RATE)
        # Signal ends right before the kept tail — the last 0.2s is silence,
        # anything before that within the tone region is loud.
        assert not audio[-tail_keep:].any()
        assert np.abs(audio[-tail_keep - 100 : -tail_keep]).max() > 0.01

    def test_trailing_silence_trim_restores_window(self):
        # 9s tone + 3s silence: without the trim the 8s window would contain
        # 3s of silence and only 5s of speech; with it we keep ~7.8s of speech.
        pcm = _pcm_tone(9.0) + _pcm_silence(3.0)
        audio = SmartTurnEouModel._prepare_audio(pcm, _MODEL_SAMPLE_RATE)
        assert audio.shape == (_MODEL_SAMPLES,)
        tail_keep = int(_TAIL_SILENCE_SECS * _MODEL_SAMPLE_RATE)
        assert np.abs(audio[: _MODEL_SAMPLES - tail_keep - 100]).max() > 0.01

    def test_all_silence_untouched(self):
        audio = SmartTurnEouModel._prepare_audio(_pcm_silence(2.0), _MODEL_SAMPLE_RATE)
        assert audio.shape == (_MODEL_SAMPLES,)
        assert not audio.any()


class TestCheckpointSelection:
    def test_default_is_fp32(self, monkeypatch):
        monkeypatch.delenv("TIMBAL_SMART_TURN_CHECKPOINT", raising=False)
        assert SmartTurnEouModel().checkpoint == DEFAULT_FILENAME

    def test_aliases(self):
        assert SmartTurnEouModel(checkpoint="int8").checkpoint == QUANTIZED_FILENAME
        assert SmartTurnEouModel(checkpoint="fp32").checkpoint == DEFAULT_FILENAME
        assert SmartTurnEouModel(checkpoint="quantized").checkpoint == QUANTIZED_FILENAME

    def test_explicit_filename_passthrough(self):
        assert SmartTurnEouModel(checkpoint="smart-turn-v3.1-cpu.onnx").checkpoint == "smart-turn-v3.1-cpu.onnx"

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_SMART_TURN_CHECKPOINT", "int8")
        assert SmartTurnEouModel().checkpoint == QUANTIZED_FILENAME

    def test_constructor_beats_env(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_SMART_TURN_CHECKPOINT", "int8")
        assert SmartTurnEouModel(checkpoint="fp32").checkpoint == DEFAULT_FILENAME


class TestWhisperFeatures:
    def test_shape_and_dtype(self):
        audio = np.zeros(_MODEL_SAMPLES, dtype=np.float32)
        feats = compute_whisper_log_mel_features(audio)
        assert feats.shape == (80, 800)
        assert feats.dtype == np.float32
        assert np.isfinite(feats).all()

    def test_signal_differs_from_silence(self):
        silence = compute_whisper_log_mel_features(np.zeros(_MODEL_SAMPLES, dtype=np.float32))
        t = np.arange(_MODEL_SAMPLES) / _MODEL_SAMPLE_RATE
        tone = compute_whisper_log_mel_features(np.sin(2 * np.pi * 220.0 * t).astype(np.float32))
        assert not np.allclose(silence, tone)

    def test_rejects_multidim(self):
        with pytest.raises(ValueError):
            compute_whisper_log_mel_features(np.zeros((2, 100), dtype=np.float32))


class TestSmartTurnEouModel:
    async def test_predict_with_fake_session(self):
        model = SmartTurnEouModel()
        fake = _FakeOrtSession(probability=0.83)
        model._session = fake

        p = await model.predict_complete(_pcm_tone(1.5), sample_rate=_MODEL_SAMPLE_RATE)

        assert p == pytest.approx(0.83, abs=1e-6)
        assert len(fake.calls) == 1
        feats = fake.calls[0]["input_features"]
        assert feats.shape == (1, 80, 800)
        assert feats.dtype == np.float32

    async def test_close_keeps_session(self):
        # Shared default instance: one session closing must not unload the model.
        model = SmartTurnEouModel()
        model._session = _FakeOrtSession()
        await model.close()
        assert model._session is not None

    async def test_works_inside_local_detector(self):
        from timbal.voice import AudioInputConfig, CommitAction, TurnState

        model = SmartTurnEouModel()
        model._session = _FakeOrtSession(probability=0.1)  # incomplete
        detector = LocalAudioTurnDetector(audio_eou=model)
        await detector.start(AudioInputConfig())
        detector.push_audio(_pcm_tone(2.0))

        state = TurnState(
            assistant_active=False,
            audio_playing=False,
            assistant_text="",
            active_user_text="",
            seconds_since_turn_start=99.0,
            seconds_since_last_commit=99.0,
            partials_since_last_commit=3,
        )
        decision = await detector.on_committed("so I was wondering", state)
        assert decision.action is CommitAction.HOLD
        assert decision.reason == "audio_hold"

        model._session.probability = 0.95  # complete
        decision = await detector.on_committed("what's the weather like today?", state)
        assert decision.action is CommitAction.NEW_TURN


class TestResolveLocalMode:
    def _reset_default_cache(self):
        turn_detection_module._DEFAULT_AUDIO_EOU = turn_detection_module._AUDIO_EOU_UNSET
        turn_detection_module._DEFAULT_TEXT_EOU = turn_detection_module._TEXT_EOU_UNSET

    def test_local_gets_smart_turn_model(self):
        self._reset_default_cache()
        try:
            detector = resolve_turn_detector("local")
            assert isinstance(detector, LocalAudioTurnDetector)
            assert isinstance(detector.audio_eou, SmartTurnEouModel)
        finally:
            self._reset_default_cache()

    def test_local_shares_one_model_instance(self):
        self._reset_default_cache()
        try:
            a = resolve_turn_detector("local")
            b = resolve_turn_detector("local")
            assert a.audio_eou is b.audio_eou
            assert a.clone().audio_eou is a.audio_eou
        finally:
            self._reset_default_cache()

    def test_local_degrades_without_extra(self, monkeypatch):
        self._reset_default_cache()
        # Setting the sys.modules entry to None makes `from .smart_turn import ...`
        # raise ImportError — the same failure mode as a missing extra.
        monkeypatch.setitem(sys.modules, "timbal.voice.smart_turn", None)
        try:
            detector = resolve_turn_detector("local")
            assert isinstance(detector, LocalAudioTurnDetector)
            assert detector.audio_eou is None
        finally:
            self._reset_default_cache()


@pytest.mark.skipif(
    not os.environ.get("TIMBAL_SMART_TURN_INTEGRATION"),
    reason="set TIMBAL_SMART_TURN_INTEGRATION=1 to download and run the real checkpoint",
)
class TestRealCheckpoint:
    async def test_real_model_scores_audio(self):
        model = SmartTurnEouModel()
        await model.start(sample_rate=_MODEL_SAMPLE_RATE)
        p = await model.predict_complete(_pcm_silence(2.0), sample_rate=_MODEL_SAMPLE_RATE)
        assert 0.0 <= p <= 1.0
