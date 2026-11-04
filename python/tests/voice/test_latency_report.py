"""Silence-to-first-audio p50/p95 from TurnMetrics and traces."""

from __future__ import annotations

import json

from timbal.voice.latency import (
    gate_silence_to_first_audio,
    percentile,
    silence_to_first_audio_ms,
    summarize_traces,
    summarize_turn_metrics,
)
from timbal.voice.metrics import TurnMetrics


def _turn(**overrides) -> TurnMetrics:
    fields = {
        "turn_index": 1,
        "user_text_chars": 12,
        "speech_end_to_transcript_ms": 80.0,
        "speech_end_source": "vad",
        "eou_to_first_audio_ms": 420.0,
        "turn_total_ms": 600.0,
        "interrupted": False,
        "tts_segments": 1,
        "audio_bytes": 3200,
    }
    fields.update(overrides)
    return TurnMetrics(**fields)


def test_percentile_matches_type7_endpoints() -> None:
    assert percentile([], 50) is None
    assert percentile([12.0], 95) == 12.0
    # n=4, p50 sits halfway between the 2nd and 3rd sorted samples.
    assert percentile([100, 400, 200, 300], 50) == 250.0
    assert percentile([100, 200, 300, 400], 95) == 385.0


def test_headline_is_the_per_turn_sum_not_the_sum_of_tails() -> None:
    """LiveKit's warning: do not add a p95 of transcription to a p95 of first audio."""
    turns = [
        _turn(turn_index=1, speech_end_to_transcript_ms=100, eou_to_first_audio_ms=100),
        _turn(turn_index=2, speech_end_to_transcript_ms=100, eou_to_first_audio_ms=500),
        _turn(turn_index=3, speech_end_to_transcript_ms=400, eou_to_first_audio_ms=100),
    ]
    report = summarize_turn_metrics(turns)
    assert [silence_to_first_audio_ms(turn) for turn in turns] == [200.0, 600.0, 500.0]
    assert report.p50_ms == 500.0
    assert report.p95_ms == 590.0
    assert report.speech_end_to_transcript.p95_ms == 370.0
    assert report.eou_to_first_audio.p95_ms == 460.0
    assert report.p95_ms != report.speech_end_to_transcript.p95_ms + report.eou_to_first_audio.p95_ms


def test_missing_anchor_event_turn_and_filler() -> None:
    """Pipecat emits nothing without a stop; LiveKit's 0.0 sentinel is not used."""
    measured = _turn(turn_index=1, run_id="run-a")
    no_speech = _turn(turn_index=2, run_id="run-b", speech_end_to_transcript_ms=None, speech_end_source=None)
    no_audio = _turn(turn_index=3, run_id="run-c", eou_to_first_audio_ms=None)
    event = _turn(turn_index=4, run_id="run-d", event=True, speech_end_to_transcript_ms=5, eou_to_first_audio_ms=5)
    filler = _turn(
        turn_index=5,
        run_id="run-e",
        speech_end_to_transcript_ms=40,
        eou_to_first_audio_ms=60,
        filler_spoken=True,
        speech_end_source="partial",
    )
    assert silence_to_first_audio_ms(no_speech) is None
    assert silence_to_first_audio_ms(event) is None
    report = summarize_turn_metrics([measured, no_speech, no_audio, event, filler])
    assert report.samples == 2
    assert report.filler_samples == 1
    assert report.excluded_counts == {"no_speech_end": 1, "no_audio": 1, "event_turn": 1}
    assert set(report.by_speech_end_source) == {"vad", "partial"}
    assert gate_silence_to_first_audio(report, p95_ms=10_000) == []
    assert gate_silence_to_first_audio(summarize_turn_metrics([]), p95_ms=1000) == [
        "silence to first audio has no samples"
    ]


def test_trace_jsonl_dedupes_the_latest_snapshot(tmp_path) -> None:
    first = _turn(turn_index=1, run_id="run-1", eou_to_first_audio_ms=100)
    revised = _turn(turn_index=1, run_id="run-1", eou_to_first_audio_ms=250)
    other = _turn(turn_index=2, run_id="run-1", speech_end_to_transcript_ms=50, eou_to_first_audio_ms=50)
    path = tmp_path / "traces.jsonl"

    def line(turn: TurnMetrics) -> str:
        record = {
            "run_id": turn.run_id,
            "spans": [{"metadata": {"voice_turn_metrics": turn.model_dump()}, "nested": {"ignored": True}}],
        }
        return json.dumps(record)

    path.write_text(line(first) + "\n" + line(revised) + "\n" + line(other) + "\n", encoding="utf-8")
    # An unknown future field must not reject the rest of the trace.
    extra = json.loads(line(other))
    extra["spans"][0]["metadata"]["voice_turn_metrics"]["future_field"] = True
    extra["run_id"] = "run-2"
    extra["spans"][0]["metadata"]["voice_turn_metrics"]["run_id"] = "run-2"
    extra["spans"][0]["metadata"]["voice_turn_metrics"]["turn_index"] = 1
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(extra) + "\n", encoding="utf-8")

    report = summarize_traces([path])
    assert report.samples == 3
    # 80+250, 50+50, and the extra row's 50+50. The first snapshot of run-1/1 is gone.
    assert silence_to_first_audio_ms(revised) == 330.0
    assert report.silence_to_first_audio.samples == 3


def test_directory_of_jsonl_and_gate_message(tmp_path) -> None:
    slow = _turn(speech_end_to_transcript_ms=200, eou_to_first_audio_ms=900)
    folder = tmp_path / "traces"
    folder.mkdir()
    (folder / "a.jsonl").write_text(
        json.dumps({"spans": [{"metadata": {"voice_turn_metrics": slow.model_dump()}}]}) + "\n",
        encoding="utf-8",
    )
    report = summarize_traces([folder])
    assert report.p95_ms == 1100.0
    assert gate_silence_to_first_audio(report, p95_ms=1000) == ["silence to first audio p95 1100.0ms is above 1000.0ms"]


def test_cli_reports_and_gates(tmp_path, capsys) -> None:
    from timbal.voice.latency import main

    turn = _turn(speech_end_to_transcript_ms=80, eou_to_first_audio_ms=420)
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps({"spans": [{"metadata": {"voice_turn_metrics": turn.model_dump()}}]}) + "\n",
        encoding="utf-8",
    )
    assert main([str(path)]) == 0
    text = capsys.readouterr().out
    assert "p50=500.0ms" in text
    assert "p95=500.0ms" in text
    assert main([str(path), "--p95-ms", "100"]) == 1
    assert main([str(tmp_path / "missing.jsonl")]) == 2
