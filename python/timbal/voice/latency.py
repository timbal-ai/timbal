"""Silence-to-first-audio, per turn, as p50 and p95.

The sample is one turn's ``speech_end_to_transcript_ms + eou_to_first_audio_ms``.
Those clocks are adjacent: the caller going quiet, then the committed transcript,
then the first speech bytes emitted. The percentile is taken on that per-turn
sum. A p95 of transcription plus a p95 of first audio describes a turn where
both stages were slow together, which is a different and larger number.

This is the same interval other cascaded pipelines already publish:

* Pipecat's ``UserBotLatencyObserver`` (BSD-2-Clause) records when the VAD says
  the user stopped, back-dated by the silence the VAD had to hear
  (``timestamp - stop_secs``), and emits when ``BotStartedSpeakingFrame``
  arrives. It does not emit at all when that stop was never observed.
  https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/observers/user_bot_latency_observer.py
* LiveKit Agents (Apache-2.0) stores ``e2e_latency`` on the assistant message:
  back-dated VAD end of speech until the first audio frame. The same docs give
  the stage sum ``end_of_utterance_delay + llm.ttft + tts.ttfb`` and warn that
  you percentile the per-turn total, not the sum of the stage percentiles.
  ``transcription_delay`` is already inside ``end_of_utterance_delay``, so it is
  not added again. LiveKit writes ``0.0`` when the speech anchor is missing;
  this module leaves the turn out instead, matching Pipecat and the existing
  ``None`` on :class:`~timbal.voice.metrics.TurnMetrics`. A zero would pull the
  median down on turns that were never measured.
  https://docs.livekit.io/agents/build/metrics/

``speech_end_source="vad"`` is the back-dated last speech frame, comparable to
both of those. ``"partial"`` is the last interim transcript, which already
includes some provider delay, so that sample reads low. Event turns (a greeting
or a session event, with no caller silence) are excluded; Pipecat reports that
connect-to-first-speech number on a separate callback.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, computed_field

from .metrics import TurnMetrics

ExclusionReason = Literal["event_turn", "no_speech_end", "no_audio", "invalid"]


def percentile(values: Iterable[float], q: float) -> float | None:
    """Linear percentile. ``q`` is in ``[0, 100]``.

    One sample returns that sample for every ``q``. Empty input returns
    ``None``. The interpolation is Hyndman–Fan type 7 (``numpy.percentile``
    default, ``statistics.quantiles(..., method="inclusive")``): position
    ``(q / 100) * (n - 1)`` between the sorted samples. Rounded to 0.1 ms,
    the same precision :class:`~timbal.voice.metrics.TurnMetrics` stores.
    """
    if q < 0 or q > 100:
        raise ValueError(f"percentile q must be between 0 and 100, got {q}")
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 1)
    pos = (q / 100) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * frac, 1)


def silence_to_first_audio_ms(metrics: TurnMetrics | Mapping[str, Any]) -> float | None:
    """Caller went quiet → first emitted speech, or ``None`` when either clock is missing.

    A filler counts: the clock stops when speech starts, which is what the
    caller hears. ``filler_spoken`` on the turn says the bytes were the filler.
    """
    speech = _field(metrics, "speech_end_to_transcript_ms")
    audio = _field(metrics, "eou_to_first_audio_ms")
    if speech is None or audio is None:
        return None
    if bool(_field(metrics, "event")):
        return None
    if speech < 0 or audio < 0:
        return None
    return round(float(speech) + float(audio), 1)


class LatencyDistribution(BaseModel):
    """p50 and p95 of one duration, in milliseconds."""

    model_config = ConfigDict(extra="forbid")

    samples: int
    p50_ms: float | None = None
    p95_ms: float | None = None


class SilenceToFirstAudioReport(BaseModel):
    """p50/p95 of silence-to-first-audio, plus the two clocks it is built from.

    ``speech_end_to_transcript`` and ``eou_to_first_audio`` describe the same
    turns as the headline (both clocks present, not an event turn). Their p95
    values are not meant to be added.
    """

    model_config = ConfigDict(extra="forbid")

    silence_to_first_audio: LatencyDistribution
    speech_end_to_transcript: LatencyDistribution
    eou_to_first_audio: LatencyDistribution
    by_speech_end_source: dict[str, LatencyDistribution]
    filler_samples: int = 0
    excluded_counts: dict[str, int]

    @computed_field
    @property
    def samples(self) -> int:
        return self.silence_to_first_audio.samples

    @computed_field
    @property
    def p50_ms(self) -> float | None:
        return self.silence_to_first_audio.p50_ms

    @computed_field
    @property
    def p95_ms(self) -> float | None:
        return self.silence_to_first_audio.p95_ms


def summarize_turn_metrics(metrics: Iterable[TurnMetrics | Mapping[str, Any]]) -> SilenceToFirstAudioReport:
    """Percentiles of the per-turn sum. Later duplicates of ``(run_id, turn_index)`` win."""
    turns = _dedupe(_coerce(metrics))
    included: list[TurnMetrics] = []
    excluded: dict[str, int] = {}
    for turn in turns:
        reason = _exclude_reason(turn)
        if reason is None:
            included.append(turn)
        else:
            excluded[reason] = excluded.get(reason, 0) + 1
    sums = [silence_to_first_audio_ms(turn) for turn in included]
    # _exclude_reason already dropped the Nones. The cast keeps the type checker honest.
    sums_f = [value for value in sums if value is not None]
    by_source: dict[str, LatencyDistribution] = {}
    for source in ("vad", "partial"):
        group = [turn for turn in included if turn.speech_end_source == source]
        if group:
            by_source[source] = _distribution(
                [silence_to_first_audio_ms(turn) for turn in group if silence_to_first_audio_ms(turn) is not None]
            )
    return SilenceToFirstAudioReport(
        silence_to_first_audio=_distribution(sums_f),
        speech_end_to_transcript=_distribution(
            [turn.speech_end_to_transcript_ms for turn in included if turn.speech_end_to_transcript_ms is not None]
        ),
        eou_to_first_audio=_distribution(
            [turn.eou_to_first_audio_ms for turn in included if turn.eou_to_first_audio_ms is not None]
        ),
        by_speech_end_source=by_source,
        filler_samples=sum(1 for turn in included if turn.filler_spoken),
        excluded_counts=excluded,
    )


def summarize_traces(paths: Iterable[str | Path]) -> SilenceToFirstAudioReport:
    """Read ``voice_turn_metrics`` out of JSONL traces (or one JSON document) and summarize."""
    return summarize_turn_metrics(_iter_trace_metrics(paths))


def gate_silence_to_first_audio(report: SilenceToFirstAudioReport, *, p95_ms: float) -> list[str]:
    """Empty when the headline p95 is within ``p95_ms``. No samples is a failure.

    Opt-in. Nothing calls this unless you pass a ceiling. A trace with no
    measured caller turns must not look like a pass.
    """
    if report.samples == 0:
        return ["silence to first audio has no samples"]
    if report.p95_ms is not None and report.p95_ms > p95_ms:
        return [f"silence to first audio p95 {report.p95_ms:.1f}ms is above {p95_ms:.1f}ms"]
    return []


def format_report(report: SilenceToFirstAudioReport) -> str:
    """Plain-text reading of a :class:`SilenceToFirstAudioReport`."""
    lines = [
        "silence to first audio  " + _dist_text(report.silence_to_first_audio),
        "  speech end to transcript  " + _dist_text(report.speech_end_to_transcript),
        "  transcript to first audio  " + _dist_text(report.eou_to_first_audio),
    ]
    for source, dist in report.by_speech_end_source.items():
        lines.append(f"  {source}  " + _dist_text(dist))
    if "partial" in report.by_speech_end_source:
        lines.append("  partial anchors read low: the interim transcript already includes some transcriber delay")
    if report.filler_samples:
        lines.append(
            f"  filler turns in the sample: {report.filler_samples} (first audio is the filler, not the answer)"
        )
    if report.excluded_counts:
        parts = [f"{key}={count}" for key, count in report.excluded_counts.items()]
        lines.append("  excluded  " + "  ".join(parts))
    return "\n".join(lines)


def _dist_text(dist: LatencyDistribution) -> str:
    if dist.samples == 0:
        return "n=0"
    return f"n={dist.samples}  p50={_ms(dist.p50_ms)}  p95={_ms(dist.p95_ms)}"


def _ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}ms"


def _distribution(values: Iterable[float]) -> LatencyDistribution:
    samples = list(values)
    return LatencyDistribution(
        samples=len(samples),
        p50_ms=percentile(samples, 50),
        p95_ms=percentile(samples, 95),
    )


def _field(metrics: TurnMetrics | Mapping[str, Any], name: str) -> Any:
    if isinstance(metrics, TurnMetrics):
        return getattr(metrics, name)
    return metrics.get(name)


def _coerce(metrics: Iterable[TurnMetrics | Mapping[str, Any]]) -> list[TurnMetrics]:
    out: list[TurnMetrics] = []
    for item in metrics:
        if isinstance(item, TurnMetrics):
            out.append(item)
        else:
            out.append(TurnMetrics.model_validate(_known_fields(item)))
    return out


def _known_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Drop keys this version of :class:`TurnMetrics` does not know, so a newer trace still loads."""
    known = set(TurnMetrics.model_fields)
    return {key: value for key, value in raw.items() if key in known}


def _dedupe(turns: list[TurnMetrics]) -> list[TurnMetrics]:
    ordered: list[TurnMetrics] = []
    index: dict[tuple[str, int], int] = {}
    for turn in turns:
        if not turn.run_id:
            ordered.append(turn)
            continue
        key = (turn.run_id, turn.turn_index)
        slot = index.get(key)
        if slot is None:
            index[key] = len(ordered)
            ordered.append(turn)
        else:
            ordered[slot] = turn
    return ordered


def _exclude_reason(turn: TurnMetrics) -> ExclusionReason | None:
    if turn.event:
        return "event_turn"
    speech = turn.speech_end_to_transcript_ms
    audio = turn.eou_to_first_audio_ms
    if speech is None:
        return "no_speech_end"
    if audio is None:
        return "no_audio"
    if speech < 0 or audio < 0:
        return "invalid"
    return None


def _iter_trace_metrics(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    for path in _expand_paths(paths):
        text = path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if not stripped:
            continue
        if stripped[0] in "[{":
            # A JSON document (array or object). JSONL also starts with "{", so
            # try one document first and fall back to lines when it does not parse.
            try:
                yield from _walk_metrics(json.loads(text))
                continue
            except json.JSONDecodeError:
                pass
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            yield from _walk_metrics(record)


def _expand_paths(paths: Iterable[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.glob("*.jsonl")))
        else:
            found.append(path)
    return found


def _walk_metrics(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        metrics = node.get("voice_turn_metrics")
        if isinstance(metrics, dict) and "turn_index" in metrics:
            yield metrics
        for value in node.values():
            if value is metrics:
                continue
            yield from _walk_metrics(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_metrics(item)


__all__ = [
    "LatencyDistribution",
    "SilenceToFirstAudioReport",
    "format_report",
    "gate_silence_to_first_audio",
    "percentile",
    "silence_to_first_audio_ms",
    "summarize_traces",
    "summarize_turn_metrics",
]


def main(argv: list[str] | None = None) -> int:
    """Print p50/p95 for trace files. ``--p95-ms`` fails when the headline p95 is over the ceiling."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Report silence-to-first-audio p50/p95 from voice_turn_metrics in JSONL traces."
    )
    parser.add_argument("paths", nargs="+", help="trace .jsonl files, a JSON document, or a directory of .jsonl")
    parser.add_argument(
        "--p95-ms",
        type=float,
        default=None,
        help="exit 1 when silence-to-first-audio p95 is above this many milliseconds",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    missing = [path for path in args.paths if not Path(path).exists()]
    if missing:
        print("not found: " + ", ".join(missing), file=sys.stderr)  # noqa: T201
        return 2
    report = summarize_traces(args.paths)
    if args.json:
        print(report.model_dump_json(indent=2))  # noqa: T201
    else:
        print(format_report(report))  # noqa: T201
    if args.p95_ms is None:
        return 0
    failures = gate_silence_to_first_audio(report, p95_ms=args.p95_ms)
    for failure in failures:
        print(failure, file=sys.stderr)  # noqa: T201
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
