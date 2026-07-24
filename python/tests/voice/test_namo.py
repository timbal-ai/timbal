"""Tests for the Namo text EOU backend (``timbal[voice]`` extra)."""

from __future__ import annotations

import os
import sys
import time

import pytest

np = pytest.importorskip("numpy", reason="timbal[voice] extra not installed")
pytest.importorskip("onnxruntime", reason="timbal[voice] extra not installed")
pytest.importorskip("transformers", reason="timbal[voice] extra not installed")

from timbal.voice import LocalAudioTurnDetector, resolve_turn_detector  # noqa: E402
from timbal.voice import turn_detection as turn_detection_module  # noqa: E402
from timbal.voice.eou import PunctuationEouPredictor  # noqa: E402
from timbal.voice.namo import (  # noqa: E402
    DEFAULT_REPO_ID,
    MULTILINGUAL_REPO_ID,
    NamoTextEouPredictor,
    _ACK_FORCE_P,
    _prepare_text_for_namo,
)


class _FakeTokenizer:
    def __call__(self, text, truncation=True, max_length=512, return_tensors="np"):
        n = min(len(text.split()) + 2, max_length)
        return {
            "input_ids": np.ones((1, n), dtype=np.int64),
            "attention_mask": np.ones((1, n), dtype=np.int64),
        }


class _FakeOrtSession:
    """Returns fixed 2-class logits: [incomplete, complete]."""

    def __init__(self, p_complete: float = 0.9) -> None:
        self.p_complete = p_complete
        self.calls: list[dict] = []

    def run(self, output_names, feeds):  # noqa: ARG002
        self.calls.append(feeds)
        # Inverse-softmax-ish logits so softmax → [1-p, p].
        p = min(1.0 - 1e-6, max(1e-6, self.p_complete))
        logit_complete = float(np.log(p / (1.0 - p)))
        logits = np.array([[0.0, logit_complete]], dtype=np.float32)
        return [logits]


def _reset_caches() -> None:
    turn_detection_module._DEFAULT_AUDIO_EOU = turn_detection_module._AUDIO_EOU_UNSET
    turn_detection_module._DEFAULT_TEXT_EOU = turn_detection_module._TEXT_EOU_UNSET


class TestNamoPredict:
    async def test_empty_is_complete(self) -> None:
        model = NamoTextEouPredictor()
        model._session = _FakeOrtSession(0.1)
        model._tokenizer = _FakeTokenizer()
        assert await model.predict_eou("") == 1.0
        assert await model.predict_eou("   ") == 1.0
        assert model._session.calls == []

    async def test_softmax_complete_label(self) -> None:
        model = NamoTextEouPredictor()
        model._session = _FakeOrtSession(0.87)
        model._tokenizer = _FakeTokenizer()
        p = await model.predict_eou("Tell me a story.")
        assert p == pytest.approx(0.87, abs=1e-3)
        assert len(model._session.calls) == 1

    async def test_softmax_incomplete(self) -> None:
        model = NamoTextEouPredictor()
        model._session = _FakeOrtSession(0.15)
        model._tokenizer = _FakeTokenizer()
        p = await model.predict_eou("Uh, I don't know.")
        assert p == pytest.approx(0.15, abs=1e-3)

    async def test_short_ack_with_period_forced_complete(self) -> None:
        """STT loves ``Yeah.`` / ``Okay.`` — Namo scores those incomplete."""
        model = NamoTextEouPredictor()
        model._session = _FakeOrtSession(0.05)  # would be incomplete if consulted
        model._tokenizer = _FakeTokenizer()
        for text in ("Yeah.", "Okay.", "ok!", "Bye.", "Thank you."):
            assert await model.predict_eou(text) == _ACK_FORCE_P, text
        assert model._session.calls == []

    async def test_ack_prep_strips_terminal_punct(self) -> None:
        text, override = _prepare_text_for_namo("Yeah.")
        assert text == "Yeah"
        assert override == _ACK_FORCE_P
        # Longer hedges must still hit the model (no force).
        text, override = _prepare_text_for_namo("Uh, I don't know.")
        assert override is None
        assert "know" in text.lower()

    def test_default_repo_is_english_specialist(self) -> None:
        model = NamoTextEouPredictor()
        assert model.repo_id == DEFAULT_REPO_ID
        assert "English" in model.repo_id
        assert model.max_length == 512

    def test_multilingual_repo_longer_context(self) -> None:
        model = NamoTextEouPredictor(repo_id=MULTILINGUAL_REPO_ID)
        assert model.max_length == 8192


class TestResolveLocalInjectsNamo:
    def test_local_gets_namo_text_eou(self) -> None:
        _reset_caches()
        try:
            detector = resolve_turn_detector("local")
            assert isinstance(detector, LocalAudioTurnDetector)
            assert isinstance(detector.fallback_text_eou, NamoTextEouPredictor)
            assert "English" in detector.fallback_text_eou.repo_id
        finally:
            _reset_caches()

    def test_local_shares_one_text_eou_instance(self) -> None:
        _reset_caches()
        try:
            a = resolve_turn_detector("local")
            b = resolve_turn_detector("local")
            assert a.fallback_text_eou is b.fallback_text_eou
            assert a.clone().fallback_text_eou is a.fallback_text_eou
        finally:
            _reset_caches()

    def test_local_falls_back_without_namo(self, monkeypatch) -> None:
        _reset_caches()
        monkeypatch.setitem(sys.modules, "timbal.voice.namo", None)
        try:
            detector = resolve_turn_detector("local")
            assert isinstance(detector, LocalAudioTurnDetector)
            assert isinstance(detector.fallback_text_eou, PunctuationEouPredictor)
        finally:
            _reset_caches()


@pytest.mark.skipif(
    not os.environ.get("TIMBAL_NAMO_INTEGRATION"),
    reason="set TIMBAL_NAMO_INTEGRATION=1 to download and run the real checkpoint",
)
class TestRealCheckpoint:
    async def test_real_model_scores_text(self) -> None:
        model = NamoTextEouPredictor()
        await model.start()
        p_complete = await model.predict_eou("Tell me a story.")
        p_hedge = await model.predict_eou("Uh, I don't know.")
        p_dangling = await model.predict_eou("I was thinking about")
        assert 0.0 <= p_complete <= 1.0
        assert 0.0 <= p_hedge <= 1.0
        # English specialist is decisive on the live failure mode.
        assert p_complete > 0.9
        assert p_hedge < 0.1
        assert p_dangling < 0.1

    async def test_inference_latency_under_50ms(self) -> None:
        """Warm inference should stay well under a STT commit debounce (~1.2s).

        Model card claims <11ms; we allow 50ms headroom for CI noise / cold CPU.
        """
        model = NamoTextEouPredictor()
        await model.start()
        # Discard one more warm call after start()'s own warmup.
        await model.predict_eou("warmup again")
        samples = (
            "Tell me a story.",
            "Uh, I don't know.",
            "Quite incredible.",
            "Yeah, sure. Anything else?",
            "I was thinking about",
        )
        times: list[float] = []
        for text in samples:
            t0 = time.perf_counter()
            await model.predict_eou(text)
            times.append(time.perf_counter() - t0)
        p50 = sorted(times)[len(times) // 2]
        assert p50 < 0.050, f"p50 inference {p50 * 1000:.1f}ms exceeded 50ms budget"
