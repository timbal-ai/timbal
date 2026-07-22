"""Smart Turn v3 ONNX backend for :class:`~timbal.voice.AudioEouModel`.

Runs Pipecat's open `Smart Turn v3 <https://huggingface.co/pipecat-ai/smart-turn-v3>`_
checkpoint locally on CPU (~8M params, Whisper-Tiny base + linear classifier,
~50-100ms inference incl. feature extraction). Scores the user's recent audio
and returns ``P(complete)`` — whether the user has finished their turn or is
mid-thought.

Requires the ``timbal[voice]`` extra (``onnxruntime`` + ``huggingface_hub``).
This module import fails without it; :func:`~timbal.voice.resolve_turn_detector`
catches that and degrades ``"local"`` mode to the heuristic detector.

Usage::

    from timbal.voice import LocalAudioTurnDetector
    from timbal.voice.smart_turn import SmartTurnEouModel

    detector = LocalAudioTurnDetector(audio_eou=SmartTurnEouModel())
    session = VoiceSession(..., turn_detector=detector)

Or simply ``turn_detector="local"`` — the resolver constructs the default
model automatically when the extra is installed.

Input contract (per the Smart Turn model card): 16 kHz mono PCM, last 8
seconds of the user's turn, left-padded with zeros when shorter. Non-16kHz
session rates are linearly resampled — fine for speech EOU scoring; if you
need higher-fidelity resampling, resample upstream and run the session at
16 kHz.

The model was trained on audio that ends ~0.2s after the user stops speaking
(Pipecat scores right at VAD stop). Our window instead ends at the STT commit,
which arrives after the provider's own silence timeout (~1.2s+), so
``_prepare_audio`` trims trailing silence down to :data:`_TAIL_SILENCE_SECS`
before scoring — long silence tails measurably corrupt the probabilities in
both directions.
"""

from __future__ import annotations

import asyncio
import os
from functools import partial

import numpy as np
import onnxruntime as ort
import structlog

from ._whisper_features import compute_whisper_log_mel_features
from .eou import AudioEouModel

logger = structlog.get_logger("timbal.voice.smart_turn")

_MODEL_SAMPLE_RATE = 16_000
_AUDIO_SECONDS = 8
_MODEL_SAMPLES = _MODEL_SAMPLE_RATE * _AUDIO_SECONDS  # 128_000

DEFAULT_REPO_ID = "pipecat-ai/smart-turn-v3"
# fp32 checkpoint ("-gpu" is just the unquantized graph; it runs fine on CPU).
# It benchmarks ~1pp better overall than the int8 "-cpu" one (93.7% vs 92.6%,
# FPR 3.5% vs 4.7%) and is far more decisive on real audio; the latency cost
# (~90ms vs ~45ms single-threaded) is noise next to the STT commit debounce.
DEFAULT_FILENAME = "smart-turn-v3.2-gpu.onnx"
QUANTIZED_FILENAME = "smart-turn-v3.2-cpu.onnx"

# The checkpoint expects audio ending shortly after end of speech; keep at most
# this much trailing silence when trimming (matches Pipecat's VAD stop timing).
_TAIL_SILENCE_SECS = 0.2
# Amplitude below this (|sample| of int16-normalized float) counts as silence.
_SILENCE_AMPLITUDE = 0.006


class SmartTurnEouModel(AudioEouModel):
    """Local end-of-turn scoring with the Smart Turn v3 ONNX checkpoint.

    Parameters
    ----------
    model_path:
        Path to a local ``.onnx`` file. When ``None`` (default), the checkpoint
        is downloaded once via ``huggingface_hub`` (cached under the standard
        HF cache directory) from :data:`DEFAULT_REPO_ID`.
    checkpoint:
        Which file to download from the HF repo when ``model_path`` is not
        set. Accepts a full filename or the shorthands ``"fp32"`` (default,
        :data:`DEFAULT_FILENAME`) and ``"int8"`` (:data:`QUANTIZED_FILENAME`,
        ~2x faster inference for ~1pp accuracy). Also settable via the
        ``TIMBAL_SMART_TURN_CHECKPOINT`` env var (constructor wins).
    cpu_count:
        ``intra_op_num_threads`` for the ONNX session. Smart Turn v3 is small
        enough that 1 thread keeps inference well under 100ms without stealing
        cores from the event loop.

    ``start()`` loads the model off the event loop; ``predict_complete()``
    offloads feature extraction + inference to the default executor. Instances
    are safe to share across sessions (``ort.InferenceSession.run`` is
    thread-safe; there is no per-call mutable state).
    """

    _CHECKPOINT_ALIASES = {
        "fp32": DEFAULT_FILENAME,
        "gpu": DEFAULT_FILENAME,
        "int8": QUANTIZED_FILENAME,
        "cpu": QUANTIZED_FILENAME,
        "quantized": QUANTIZED_FILENAME,
    }

    def __init__(
        self,
        model_path: str | None = None,
        *,
        checkpoint: str | None = None,
        cpu_count: int = 1,
    ) -> None:
        self.model_path = model_path
        raw = checkpoint or os.environ.get("TIMBAL_SMART_TURN_CHECKPOINT") or DEFAULT_FILENAME
        self.checkpoint = self._CHECKPOINT_ALIASES.get(raw.strip().lower(), raw)
        self.cpu_count = cpu_count
        self._session: ort.InferenceSession | None = None
        self._sample_rate = _MODEL_SAMPLE_RATE
        self._load_lock = asyncio.Lock()

    async def start(self, *, sample_rate: int) -> None:
        self._sample_rate = int(sample_rate) or _MODEL_SAMPLE_RATE
        async with self._load_lock:
            if self._session is not None:
                return
            loop = asyncio.get_running_loop()
            self._session = await loop.run_in_executor(None, self._load_session)

    async def close(self) -> None:
        # Shared across sessions; the ort session holds no per-run state and
        # model reload is expensive, so keep it alive for the process.
        pass

    def _load_session(self) -> ort.InferenceSession:
        path = self.model_path
        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=DEFAULT_REPO_ID, filename=self.checkpoint)
        logger.debug("smart_turn_loading_model", path=str(path))
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = self.cpu_count
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), sess_options=so)
        # INFO on purpose: "is the audio EOU model actually engaged?" is the
        # first question when debugging turn-taking, and this fires once.
        logger.info("smart_turn_model_loaded", path=str(path))
        return session

    async def predict_complete(self, pcm: bytes, *, sample_rate: int) -> float:
        if self._session is None:
            await self.start(sample_rate=sample_rate)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._predict_sync, pcm, sample_rate))

    def _predict_sync(self, pcm: bytes, sample_rate: int) -> float:
        audio = self._prepare_audio(pcm, sample_rate)
        features = compute_whisper_log_mel_features(audio, do_normalize=True)
        input_features = np.expand_dims(features, axis=0)  # (1, 80, 800)
        outputs = self._session.run(None, {"input_features": input_features})
        # The model returns sigmoid probabilities.
        return float(outputs[0][0].item())

    @staticmethod
    def _trim_trailing_silence(audio: np.ndarray) -> np.ndarray:
        """Drop the silence tail so the window ends ~0.2s after end of speech.

        The scored window ends at the STT commit, i.e. after the provider's
        silence timeout (1.2s+ of dead air). Smart Turn was trained on audio
        ending right at VAD stop; feeding it long tails shifts probabilities
        badly in both directions (measured: an incomplete "so I was thinking
        about" jumps from p≈0.1 to p≈0.5 with a 0.6s tail).
        """
        if not audio.size:
            return audio
        loud = np.flatnonzero(np.abs(audio) > _SILENCE_AMPLITUDE)
        if not loud.size:
            return audio
        keep = int(loud[-1]) + 1 + int(_TAIL_SILENCE_SECS * _MODEL_SAMPLE_RATE)
        return audio[: min(keep, audio.size)]

    @staticmethod
    def _prepare_audio(pcm: bytes, sample_rate: int) -> np.ndarray:
        """PCM16LE bytes → float32 mono at 16 kHz, silence tail trimmed, last 8s, left-padded."""
        if len(pcm) % 2:
            pcm = pcm[:-1]
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != _MODEL_SAMPLE_RATE and audio.size:
            n_out = int(round(audio.size * _MODEL_SAMPLE_RATE / sample_rate))
            audio = np.interp(
                np.linspace(0.0, audio.size - 1, max(n_out, 1)),
                np.arange(audio.size),
                audio,
            ).astype(np.float32)
        audio = SmartTurnEouModel._trim_trailing_silence(audio)
        if audio.size > _MODEL_SAMPLES:
            audio = audio[-_MODEL_SAMPLES:]
        elif audio.size < _MODEL_SAMPLES:
            # Left-pad per the model card: audio at the end, zeros at the start.
            audio = np.pad(audio, (_MODEL_SAMPLES - audio.size, 0), mode="constant")
        return audio
