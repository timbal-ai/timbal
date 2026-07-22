"""Silero VAD (ONNX) — local streaming voice-activity detection.

Runs the open `Silero VAD v5 <https://huggingface.co/onnx-community/silero-vad>`_
checkpoint (MIT, ~2MB, <1ms per frame on CPU) on raw mic PCM. Used by
:class:`~timbal.voice.endpointing.VadEndpointer` to detect end of speech
locally (~0.2s after the user stops) instead of waiting for the STT
provider's commit debounce (~1.2s+).

Requires the ``timbal[voice]`` extra (``onnxruntime`` + ``huggingface_hub``,
same as Smart Turn). This module import fails without it; callers degrade
gracefully (endpointing simply stays off).

The model is stateful and streaming: it consumes fixed 512-sample frames at
16 kHz (32ms hop) plus 64 samples of context carried between frames, and
returns ``P(speech)`` per frame. :class:`SileroVad` handles the framing,
context, resampling, and recurrent state so callers just push arbitrary PCM
chunks and receive per-frame probabilities.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import numpy as np
import onnxruntime as ort
import structlog

logger = structlog.get_logger("timbal.voice.vad")

DEFAULT_REPO_ID = "onnx-community/silero-vad"
DEFAULT_FILENAME = "onnx/model.onnx"

_VAD_SAMPLE_RATE = 16_000
# Fixed by the model: 512 new samples (32ms) per inference at 16 kHz...
FRAME_SAMPLES = 512
FRAME_SECS = FRAME_SAMPLES / _VAD_SAMPLE_RATE  # 0.032
# ...preceded by 64 samples of context from the previous frame (the official
# wrapper concatenates them; the recurrent state alone is not enough).
_CONTEXT_SAMPLES = 64


@lru_cache(maxsize=4)
def _load_ort_session(path: str) -> ort.InferenceSession:
    """One ort session per checkpoint path, shared across all SileroVad instances.

    ``InferenceSession.run`` is thread-safe and holds no per-call state; the
    recurrent state lives in :class:`SileroVad`, so sharing is safe.
    """
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(path, sess_options=so)
    # Warm the graph so the first mic frame doesn't pay one-time allocations.
    session.run(
        None,
        {
            "input": np.zeros((1, _CONTEXT_SAMPLES + FRAME_SAMPLES), dtype=np.float32),
            "state": np.zeros((2, 1, 128), dtype=np.float32),
            "sr": np.array(_VAD_SAMPLE_RATE, dtype=np.int64),
        },
    )
    logger.info("silero_vad_model_loaded", path=path)
    return session


class SileroVad:
    """Streaming Silero VAD: push PCM16LE chunks, get per-frame ``P(speech)``.

    Parameters
    ----------
    model_path:
        Path to a local ``.onnx`` file. When ``None`` (default), the checkpoint
        is downloaded once via ``huggingface_hub`` (standard HF cache) from
        :data:`DEFAULT_REPO_ID`.

    Instances hold per-stream mutable state (recurrent state, context, sample
    residual) — use one per audio stream / session. The underlying ort session
    is cached at module level and shared.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._session: ort.InferenceSession | None = None
        self._sample_rate = _VAD_SAMPLE_RATE
        self._sr_input = np.array(_VAD_SAMPLE_RATE, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        # Drift-free resampling counters (native samples in, 16k samples out).
        self._in_total = 0
        self._out_total = 0

    async def start(self, *, sample_rate: int) -> None:
        """Resolve the checkpoint (downloading off the event loop) and reset state."""
        self._sample_rate = int(sample_rate) or _VAD_SAMPLE_RATE
        if self._session is None:
            loop = asyncio.get_running_loop()
            self._session = await loop.run_in_executor(None, self._resolve_session)
        self.reset()

    def _resolve_session(self) -> ort.InferenceSession:
        path = self.model_path
        if path is None:
            from .smart_turn import _hf_download_cached_first

            path = _hf_download_cached_first(DEFAULT_REPO_ID, DEFAULT_FILENAME)
        return _load_ort_session(str(path))

    def reset(self) -> None:
        """Clear recurrent state, context, and buffered samples."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._in_total = 0
        self._out_total = 0

    def process(self, pcm: bytes) -> list[float]:
        """Consume a PCM16LE mono chunk; return ``P(speech)`` per complete 32ms frame.

        Chunks of any size are fine — leftover samples are buffered until the
        next call. Non-16 kHz input is linearly resampled per chunk with
        drift-free sample accounting (VAD probabilities are insensitive to the
        minor chunk-edge interpolation artifacts this introduces).
        """
        if self._session is None:
            raise RuntimeError("Call start() before process().")
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if not pcm:
            return []
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self._sample_rate != _VAD_SAMPLE_RATE:
            self._in_total += x.size
            target_total = int(self._in_total * _VAD_SAMPLE_RATE / self._sample_rate)
            n_out = target_total - self._out_total
            if n_out <= 0:
                return []
            x = np.interp(
                np.linspace(0.0, x.size - 1, n_out),
                np.arange(x.size),
                x,
            ).astype(np.float32)
            self._out_total = target_total
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x
        probs: list[float] = []
        while self._buf.size >= FRAME_SAMPLES:
            frame = self._buf[:FRAME_SAMPLES]
            self._buf = self._buf[FRAME_SAMPLES:]
            model_input = np.concatenate([self._context, frame])[None, :]
            out, self._state = self._session.run(
                None,
                {"input": model_input, "state": self._state, "sr": self._sr_input},
            )
            self._context = frame[-_CONTEXT_SAMPLES:]
            probs.append(float(out[0, 0]))
        return probs
