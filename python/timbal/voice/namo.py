"""Namo Turn Detector v1 — DistilBERT text EOU behind :class:`TextEouPredictor`.

Runs VideoSDK's open
`Namo-Turn-Detector-v1-English <https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-English>`_
(Apache-2.0, DistilBERT, quantized ONNX, ~135MB, <11ms CPU) on the STT
transcript and returns ``P(complete)``. Complements Smart Turn (audio) with a
semantic text signal — the missing half for mid-thought hedges like
"Uh, I don't know." that audio alone over-scores as complete.

The multilingual mmBERT checkpoint is available via ``repo_id`` /
``TIMBAL_NAMO_REPO_ID``, but measured English polarity is mushy on that
graph (card examples like "so that's all I have for today" score incomplete);
prefer the English specialist until VideoSDK ships a sharper multilingual.

Requires the ``timbal[voice]`` extra (``onnxruntime`` + ``huggingface_hub`` +
``transformers``). Import fails without it;
:func:`~timbal.voice.resolve_turn_detector` falls back to
:class:`~timbal.voice.PunctuationEouPredictor`.

Usage::

    from timbal.voice import LocalAudioTurnDetector
    from timbal.voice.namo import NamoTextEouPredictor

    detector = LocalAudioTurnDetector(
        audio_eou=...,
        fallback_text_eou=NamoTextEouPredictor(),
    )

Or ``turn_detector="local"`` — the resolver injects Namo automatically when
the extra is installed.

Label contract (per the model card): class ``1`` = end of turn, ``0`` = not
end of turn. We expose ``softmax(logits)[1]`` as ``P(complete)``.
"""

from __future__ import annotations

import asyncio
import os
import re
from functools import lru_cache, partial

import numpy as np
import onnxruntime as ort
import structlog
from transformers import AutoTokenizer

from .eou import TextEouPredictor
from .smart_turn import _hf_download_cached_first

logger = structlog.get_logger("timbal.voice.namo")

# Default: English DistilBERT specialist — decisive on measured hedges
# ("Uh, I don't know." → ~0.0 complete). Multilingual mmBERT is opt-in.
DEFAULT_REPO_ID = "videosdk-live/Namo-Turn-Detector-v1-English"
ENGLISH_REPO_ID = DEFAULT_REPO_ID
MULTILINGUAL_REPO_ID = "videosdk-live/Namo-Turn-Detector-v1-Multilingual"
DEFAULT_FILENAME = "model_quant.onnx"
# English DistilBERT uses 512; multilingual card uses 8192. Cap per-repo.
_MAX_LENGTH_BY_REPO = {
    DEFAULT_REPO_ID: 512,
    MULTILINGUAL_REPO_ID: 8192,
}
_DEFAULT_MAX_LENGTH = 512
# Class index for "end of turn" / complete (model card).
_COMPLETE_LABEL = 1

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_TRAILING_TERMINAL_RE = re.compile(r"[.!？。…]+$")
# Short backchannels Namo (and STT punctuation) mis-score as incomplete —
# measured: "Yeah."→0.04, "Yeah"→0.92, "Okay."/"Okay"→~0.00. Force complete.
_FORCE_COMPLETE_ACKS = frozenset(
    {
        "yeah",
        "yep",
        "yup",
        "yes",
        "ok",
        "okay",
        "sure",
        "bye",
        "goodbye",
        "thanks",
        "thank you",
        "hi",
        "hello",
        "hey",
        "no",
        "nah",
        "nope",
        "alright",
        "right",
        "cool",
        "fine",
        "got it",
        "mhmm",
        "mmhmm",
        "mhm",
    }
)
_ACK_FORCE_P = 0.95


def _prepare_text_for_namo(text: str) -> tuple[str, float | None]:
    """Normalize short STT acks; optionally force ``P(complete)``.

    Returns ``(text_for_model, override_or_None)``. Empty → complete.
    """
    stripped = text.strip()
    if not stripped:
        return "", 1.0
    words = _WORD_RE.findall(stripped)
    if len(words) <= 3:
        stripped = _TRAILING_TERMINAL_RE.sub("", stripped).strip()
        key = " ".join(w.lower() for w in _WORD_RE.findall(stripped))
        if key in _FORCE_COMPLETE_ACKS:
            return stripped, _ACK_FORCE_P
    return stripped, None


@lru_cache(maxsize=4)
def _load_bundle(
    repo_id: str, filename: str, cpu_count: int, max_length: int
) -> tuple[ort.InferenceSession, object]:
    """One (ort session, tokenizer) per repo — shared process-wide."""
    path = _hf_download_cached_first(repo_id, filename)
    logger.debug("namo_loading_model", path=path, repo_id=repo_id)
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = cpu_count
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(path, sess_options=so)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    # Warm the graph + tokenizer so the first live score isn't the cold path.
    inputs = tokenizer(
        "warmup",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )
    logger.info("namo_text_eou_loaded", path=path, repo_id=repo_id)
    return session, tokenizer


class NamoTextEouPredictor(TextEouPredictor):
    """Local text end-of-turn scoring with the Namo DistilBERT ONNX checkpoint.

    Parameters
    ----------
    repo_id:
        Hugging Face repo. Defaults to the English DistilBERT specialist
        (:data:`DEFAULT_REPO_ID`); pass :data:`MULTILINGUAL_REPO_ID` (or set
        ``TIMBAL_NAMO_REPO_ID``) for mmBERT multilingual weights. Constructor
        wins over the env var.
    model_path:
        Optional local ``.onnx`` path (skips the Hub download for the graph;
        the tokenizer still loads from ``repo_id``).
    cpu_count:
        ``intra_op_num_threads`` for the ONNX session (1 is plenty).

    ``start()`` loads off the event loop; ``predict_eou()`` runs tokenize +
    inference on the default executor. Instances are safe to share across
    sessions.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        *,
        model_path: str | None = None,
        cpu_count: int = 1,
    ) -> None:
        self.repo_id = (
            repo_id
            or os.environ.get("TIMBAL_NAMO_REPO_ID")
            or DEFAULT_REPO_ID
        )
        self.model_path = model_path
        self.cpu_count = cpu_count
        self.max_length = _MAX_LENGTH_BY_REPO.get(self.repo_id, _DEFAULT_MAX_LENGTH)
        self._session: ort.InferenceSession | None = None
        self._tokenizer: object | None = None
        self._load_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._load_lock:
            if self._session is not None:
                return
            loop = asyncio.get_running_loop()
            self._session, self._tokenizer = await loop.run_in_executor(None, self._load)

    async def close(self) -> None:
        # Shared process-wide; keep alive.
        pass

    def _load(self) -> tuple[ort.InferenceSession, object]:
        if self.model_path is not None:
            # Custom graph path: still need the matching tokenizer from the repo.
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.cpu_count
            session = ort.InferenceSession(self.model_path, sess_options=so)
            tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            return session, tokenizer
        return _load_bundle(self.repo_id, DEFAULT_FILENAME, self.cpu_count, self.max_length)

    async def predict_eou(self, text: str) -> float:
        if self._session is None:
            await self.start()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._predict_sync, text))

    def _predict_sync(self, text: str) -> float:
        stripped, override = _prepare_text_for_namo(text)
        if override is not None:
            return override
        if not stripped:
            return 1.0
        inputs = self._tokenizer(
            stripped,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        outputs = self._session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )
        logits = np.asarray(outputs[0][0], dtype=np.float64)
        # Stable softmax → P(complete) = P(label=1).
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp)
        if probs.shape[0] <= _COMPLETE_LABEL:
            return float(probs[-1])
        return float(probs[_COMPLETE_LABEL])
