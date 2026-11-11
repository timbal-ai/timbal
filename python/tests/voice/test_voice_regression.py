"""Synthetic voice regression: p95 latency and false cuts, no provider keys."""

from __future__ import annotations

import json
from pathlib import Path

from timbal.voice.metrics import TurnMetrics
from timbal.voice.regression import ReplayObservation, gate_voice_regression, replay_cases

_CORPUS = Path(__file__).parent / "fixtures" / "voice_regression.json"


def _load() -> dict:
    return json.loads(_CORPUS.read_text(encoding="utf-8"))


def test_gate_fails_closed_on_a_false_cut_and_on_p95() -> None:
    case = {
        "id": "scripted",
        "expected_user_turns": 1,
        "expect_interrupted": False,
        "user_text_contains": ["tracking number"],
    }
    slow = TurnMetrics(
        turn_index=1,
        user_text_chars=10,
        speech_end_to_transcript_ms=400,
        speech_end_source="vad",
        eou_to_first_audio_ms=900,
        turn_total_ms=1400,
        interrupted=True,
        tts_segments=1,
        audio_bytes=100,
    )
    observation = ReplayObservation(
        case_id="scripted",
        user_turns=["nope", "still nope"],
        interrupted=True,
        false_cuts=["false end of turn: audio started before the utterance finished"],
        metrics=[slow],
    )
    failures = gate_voice_regression(
        [observation],
        [case],
        p95_eou_to_first_audio_ms=500,
        p95_silence_to_first_audio_ms=500,
        min_eou_samples=1,
    )
    text = "\n".join(failures)
    assert "false end of turn" in text
    assert "false cut: the reply was interrupted" in text
    assert "expected 1 user turn(s), got 2" in text
    assert "tracking number" in text
    assert "end of utterance to first audio p95" in text
    assert "silence to first audio p95" in text


async def test_corpus_replay_holds_latency_and_does_not_false_cut() -> None:
    corpus = _load()
    observations = await replay_cases(corpus["cases"])
    failures = gate_voice_regression(
        observations,
        corpus["cases"],
        p95_eou_to_first_audio_ms=corpus["p95_eou_to_first_audio_ms"],
        p95_silence_to_first_audio_ms=corpus["p95_silence_to_first_audio_ms"],
        min_eou_samples=corpus["min_eou_samples"],
    )
    details = "\n".join(
        f"{item.case_id}: turns={item.user_turns} interrupted={item.interrupted} "
        f"cuts={item.false_cuts} error={item.error} eou={item.eou_ms}"
        for item in observations
    )
    assert failures == [], details
