"""Offline detector snapshot replay for #228; no speech/model/network calls.

Run from the repository root with PYTHONPATH=python. Expectations describe
caller intent, not fixes. See benchmarks/voice/voice-turn-taking-lab.md for interpretation.
"""

# ruff: noqa: T201
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import socket
import sys
import unittest
from contextlib import ExitStack, contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, model_validator
from timbal.voice.turn_detection import ProviderTurnDetector, TurnState

DEFAULT_CASES = Path(__file__).with_name("turn_taking_cases.json")
CATEGORIES = ("acknowledgment", "answer", "interruption", "incomplete", "resumed_request", "merge_guard")


class FixtureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Expected(FixtureModel):
    action: Literal["ignore", "barge_in", "new_turn", "continue", "hold"]
    text: str | None = None
    hold_timeout_secs: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class Event(FixtureModel):
    at_ms: int = Field(ge=0)
    kind: Literal["partial", "committed"]
    text: str = Field(min_length=1)
    state: dict
    expected: Expected

    @model_validator(mode="after")
    def valid_expectation(self):
        if not self.text.strip():
            raise ValueError("transcript must contain speech")
        if self.kind == "partial":
            if self.expected.action not in {"ignore", "barge_in"}:
                raise ValueError("partial expects ignore or barge_in")
            if self.expected.text is not None or self.expected.hold_timeout_secs is not None:
                raise ValueError("partial decisions have no text or timeout")
        elif self.expected.action == "barge_in" or self.expected.text is None:
            raise ValueError("commits require expected text and a commit action")
        if self.expected.hold_timeout_secs is not None and self.expected.action != "hold":
            raise ValueError("only HOLD may expect a timeout")
        return self


class Case(FixtureModel):
    id: str = Field(min_length=1)
    language: Literal["ca", "es", "en"]
    category: Literal["acknowledgment", "answer", "interruption", "incomplete", "resumed_request", "merge_guard"]
    rationale: str = Field(min_length=1)
    events: list[Event] = Field(min_length=1)

    @model_validator(mode="after")
    def chronological(self):
        times = [event.at_ms for event in self.events]
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("event timestamps must increase strictly within a case")
        return self


class Corpus(FixtureModel):
    schema_version: Literal[1]
    state_defaults: dict
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_states(self):
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        TurnState.model_validate(self.state_defaults, strict=True)
        for case in self.cases:
            for event in case.events:
                state = TurnState.model_validate(self.state_defaults | event.state, strict=True)
                if state.audio_playing and not state.assistant_active:
                    raise ValueError(f"{case.id}: playback requires an active assistant")
                if state.holding and state.assistant_active:
                    raise ValueError(f"{case.id}: HOLD is not active assistant generation")
                for value in (state.seconds_since_turn_start, state.seconds_since_last_commit):
                    if not 0 <= value < float("inf"):
                        raise ValueError(f"{case.id}: elapsed times must be finite and nonnegative")
        return self


def load_cases(path: Path) -> Corpus:
    return Corpus.model_validate(json.loads(path.read_text(encoding="utf-8")))


@contextmanager
def no_network():
    """Fail closed on Python socket connects/DNS, including accidental model calls."""
    with ExitStack() as stack:
        for target in (
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.create_connection",
            "socket.getaddrinfo",
        ):
            stack.enter_context(patch(target, side_effect=RuntimeError("network is disabled in turn-taking replay")))
        yield


def differences(expected: Expected, observed: dict) -> list[str]:
    """Compare actions AND committed text, so losing a resumed prefix cannot pass."""
    return [field for field, value in expected.model_dump(exclude_none=True).items() if observed.get(field) != value]


async def replay(corpus: Corpus, *, languages=(), categories=(), merge_resumed_secs: float | None = None) -> dict:
    results = []
    for case in corpus.cases:
        if languages and case.language not in languages:
            continue
        if categories and case.category not in categories:
            continue
        # A fresh detector per conversation avoids echo-latch leakage between cases.
        detector = (
            ProviderTurnDetector()
            if merge_resumed_secs is None
            else ProviderTurnDetector(merge_resumed_secs=merge_resumed_secs)
        )
        events = []
        try:
            for event in case.events:
                state = TurnState.model_validate(corpus.state_defaults | event.state, strict=True)
                # Times and states are fixture inputs; never sleep or infer playback.
                # Language, rationale and desired decisions are not policy inputs.
                if event.kind == "partial":
                    decision = await detector.on_partial(event.text, state)
                    observed = {"action": decision.value}
                else:
                    decision = await detector.on_committed(event.text, state)
                    observed = decision.model_dump(mode="json")
                mismatch = differences(event.expected, observed)
                events.append(
                    {
                        "at_ms": event.at_ms,
                        "kind": event.kind,
                        "input_text": event.text,
                        "state": state.model_dump(mode="json"),
                        "expected": event.expected.model_dump(exclude_none=True),
                        "observed": observed,
                        "mismatched_fields": mismatch,
                        "status": (
                            "observed_baseline_failure" if merge_resumed_secs is None else "observed_enabled_failure"
                        )
                        if mismatch
                        else "matches_expectation",
                    }
                )
        finally:
            await detector.close()
        results.append(
            {
                "id": case.id,
                "language": case.language,
                "category": case.category,
                "rationale": case.rationale,
                "events": events,
                "matches_expectation": all(not event["mismatched_fields"] for event in events),
            }
        )
    if not results:
        raise ValueError("filters selected no cases")
    source = Path(sys.modules[ProviderTurnDetector.__module__].__file__)
    return {
        "schema_version": 1,
        "detector": "ProviderTurnDetector",
        "mode": "baseline" if merge_resumed_secs is None else "merge_enabled",
        "merge_resumed_secs": merge_resumed_secs,
        "detector_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus.model_dump_json().encode()).hexdigest(),
        "scope": "synthetic transcript + supplied state snapshots; not a VoiceSession/audio replay",
        "summary": {
            "cases": len(results),
            "matching_cases": sum(case["matches_expectation"] for case in results),
            "mismatch_cases": sum(not case["matches_expectation"] for case in results),
            "events": sum(len(case["events"]) for case in results),
            "mismatch_events": sum(bool(e["mismatched_fields"]) for c in results for e in c["events"]),
        },
        "cases": results,
    }


def selfcheck(corpus: Corpus) -> bool:
    """Harness tests are separate from quality expectations, which may still fail."""

    class ReplayTests(unittest.TestCase):
        def test_multilingual_coverage(self):
            for language in ("ca", "es", "en"):
                actual = {c.category for c in corpus.cases if c.language == language}
                self.assertTrue(set(CATEGORIES) - {"merge_guard"} <= actual)

        def test_invalid_fixtures_are_rejected(self):
            base = corpus.model_dump()
            mutations = [
                lambda d: d["cases"].append(copy.deepcopy(d["cases"][0])),
                lambda d: d["cases"][0].update(language="xx"),
                lambda d: d["cases"][0].update(events=[]),
                lambda d: d["cases"][0]["events"][0].update(at_ms=-1),
                lambda d: d["cases"][0]["events"][0].update(text=" "),
                lambda d: d["cases"][0]["events"][0]["state"].update(audio_playng=True),
                lambda d: d["cases"][0]["events"][0]["state"].update(audio_playing="true"),
                lambda d: d["cases"][0]["events"][0]["state"].update(seconds_since_last_commit=float("nan")),
                lambda d: d["cases"][0]["events"][0]["expected"].update(action="hold"),
                lambda d: d["cases"][0]["events"][0]["expected"].update(hold_timeout_secs=1.2),
                lambda d: d["cases"][0]["events"][1].update(at_ms=0),
                lambda d: d["cases"][0]["events"][1]["expected"].update(text=None),
            ]
            for index, mutate in enumerate(mutations):
                with self.subTest(mutation=index):
                    data = copy.deepcopy(base)
                    mutate(data)
                    with self.assertRaises(ValueError):
                        Corpus.model_validate(data)

        def test_score_checks_merged_text(self):
            desired = Expected(action="continue", text="Pump three, I think. The one next to the cooling unit.")
            self.assertEqual(
                differences(desired, {"action": "continue", "text": "The one next to the cooling unit."}), ["text"]
            )
            self.assertEqual(differences(desired, desired.model_dump()), [])

        def test_score_checks_hold_timeout(self):
            desired = Expected(action="hold", text="because", hold_timeout_secs=1.2)
            self.assertEqual(
                differences(desired, {"action": "hold", "text": "because", "hold_timeout_secs": 0.1}),
                ["hold_timeout_secs"],
            )

        def test_no_network_guard(self):
            with self.assertRaisesRegex(RuntimeError, "network is disabled"):
                socket.create_connection(("example.invalid", 443))
            with self.assertRaisesRegex(RuntimeError, "network is disabled"):
                socket.getaddrinfo("example.invalid", 443)

        def test_determinism_and_fixture_immutability(self):
            before = corpus.model_dump_json()
            first = asyncio.run(replay(corpus))
            self.assertEqual(first, asyncio.run(replay(corpus)))
            self.assertEqual(before, corpus.model_dump_json())
            # Case order must not leak a previous detector's echo state.
            reverse = corpus.model_copy(update={"cases": list(reversed(corpus.cases))})
            second = asyncio.run(replay(reverse))
            self.assertEqual(first["cases"], list(reversed(second["cases"])))

        def test_answers_survive_and_failures_are_counted(self):
            report = asyncio.run(replay(corpus))
            answers = [c for c in report["cases"] if c["category"] == "answer"]
            self.assertTrue(answers)
            self.assertTrue(all(c["matches_expectation"] for c in answers))
            failed = sum(bool(e["mismatched_fields"]) for c in report["cases"] for e in c["events"])
            self.assertEqual(failed, report["summary"]["mismatch_events"])

        def test_merge_opt_in_and_guard_cases(self):
            categories = ("resumed_request", "merge_guard")
            baseline = asyncio.run(replay(corpus, categories=categories))
            self.assertTrue(all(e["observed"]["action"] != "continue" for c in baseline["cases"] for e in c["events"]))
            enabled = asyncio.run(replay(corpus, categories=categories, merge_resumed_secs=2.0))
            self.assertEqual(enabled["summary"]["mismatch_events"], 0)
            self.assertEqual(enabled, asyncio.run(replay(corpus, categories=categories, merge_resumed_secs=2.0)))
            self.assertTrue(any(c["category"] == "merge_guard" for c in enabled["cases"]))

        def test_filter_and_empty_selection(self):
            report = asyncio.run(replay(corpus, languages=("ca",), categories=("answer",)))
            self.assertTrue(all(c["language"] == "ca" and c["category"] == "answer" for c in report["cases"]))
            with self.assertRaisesRegex(ValueError, "no cases"):
                asyncio.run(replay(corpus, languages=("missing",)))

    with no_network():
        return (
            unittest.TextTestRunner(verbosity=2)
            .run(unittest.defaultTestLoader.loadTestsFromTestCase(ReplayTests))
            .wasSuccessful()
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--language", action="append", choices=("ca", "es", "en"), default=[])
    parser.add_argument("--category", action="append", choices=CATEGORIES, default=[])
    parser.add_argument("--merge-resumed-secs", type=float, help="opt in to provider merging; default is disabled")
    parser.add_argument("--json", action="store_true", help="emit deterministic JSON on stdout")
    parser.add_argument("--strict", action="store_true", help="exit 1 for any desired-behavior mismatch")
    parser.add_argument("--selfcheck", action="store_true", help="test fixtures/harness offline; not a quality pass")
    args = parser.parse_args()
    try:
        if args.merge_resumed_secs is not None:
            ProviderTurnDetector(merge_resumed_secs=args.merge_resumed_secs)  # Validate even before filtering.
        corpus = load_cases(args.cases)
        with no_network():
            if args.selfcheck:
                return 0 if selfcheck(corpus) else 1
            report = asyncio.run(
                replay(
                    corpus,
                    languages=args.language,
                    categories=args.category,
                    merge_resumed_secs=args.merge_resumed_secs,
                )
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(report["scope"])
        print(f"Mode: {report['mode']}; merge_resumed_secs={report['merge_resumed_secs']}")
        for case in report["cases"]:
            print(f"{'MATCH' if case['matches_expectation'] else report['mode'].upper() + ' FAILURE'} {case['id']}")
            for event in case["events"]:
                if event["mismatched_fields"]:
                    print(f"  {event['kind']}: expected {event['expected']}; observed {event['observed']}")
        print(json.dumps(report["summary"], sort_keys=True))
        print("Matches are fixture agreement, not production validation or closure of #228.")
    return int(args.strict and report["summary"]["mismatch_events"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
