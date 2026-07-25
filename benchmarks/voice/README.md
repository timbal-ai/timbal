# Voice replay harness

Scripted conversations synthesized to audio with ElevenLabs and replayed into a
real `VoiceSession` at real-time pace, as if spoken into a mic.

> **This directory breaks the rules of `benchmarks/`.** Everything else here
> measures pure framework overhead and needs no API keys. This is a *behavioral
> eval* of the voice stack: it needs `ELEVENLABS_API_KEY` and `DEEPGRAM_API_KEY`,
> it makes real network calls, and it runs in real time. Never wire it into the
> default test path.

Full design: `planning/VOICE_REPLAY_HARNESS_PLAN.md`.

## Why

Every voice test in `python/tests/voice/` injects `TranscriptEvent`s through a
mock STT. That covers the session state machine and nothing below it — no audio,
no real STT connection, no VAD, no endpointing. The bugs that actually reach
production live in that gap.

## Status: 21 scenarios across 5 domains, scoring and gating in place

| File | Role |
|---|---|
| `synth.py` | ElevenLabs → cached PCM16 16k, fluent-utterance slicing, PCM/frame/WAV helpers |
| `scenario.py` | script primitives, declarative expectations, scenario library |
| `harness.py` | fake browser: paced feeder, reactive driver, playback ack pump |
| `score.py` | JSONL records, scorecard aggregation, baseline diffing |
| `cli.py` | runner, reporting, regression gate |

Still to come: the config matrix with parallelism, and AEC-leak simulation.

## What it found

**The biggest thing it found was a bug in the harness, not in Timbal.**
Synthesizing each `Say()` on its own gives every fragment sentence-final
intonation, because ElevenLabs renders a fragment as a complete sentence. A
speaker who pauses mid-sentence does no such thing. Measured with Smart Turn v3 on
identical words ending at the same boundary, "I want to return an item I bought"
scores **0.986 (finished)** rendered standalone and **0.019 (mid-thought)** cut out
of the fluent whole-sentence render — a swing of 0.967 from intonation alone.

Three of the known failures were that artifact. `fluent()` now renders such
sentences in one TTS call and slices them at the character timestamps ElevenLabs
returns alongside the audio; `food_list_pause`, `support_pause` and
`banking_digits_pause` went green across every detector, and median latency
improved ~10% because turns stopped being split and re-run.

The lesson generalizes beyond this harness: a synthetic voice fixture tests TTS
phrasing unless you deliberately build it to test turn-taking. Anything previously
concluded about prosody from per-fragment audio is worth re-checking.

**It also invalidated a fix that had looked good.** Before fluent synthesis,
`local` failed `medical_hesitant_pause` because Smart Turn scored the fragment
0.395 — correctly mid-thought — armed a HOLD, and a text-confidence tier then cut
that hold from 3.0s to 0.35s against a 1.5s pause, because Flux's auto-appended
period made the transcript read finished. Gating the shrink on a *confidently*
wrong audio score fixed it and held 3/3: under-scored closers sat near the floor
("That's all." → 0.115) and real fragments just under the threshold. On faithful
audio that same fragment scores **0.054** — below the closer — so the bands invert
and the threshold turns out to have been fitting the artifact. Reverted rather
than shipped.

**Trailing-modifier continuations are still unsolved, but only one case is left,
and it is a genuine tradeoff rather than a bug.** `medical_hesitant_pause` splits
on every detector. Namo scores "The pain started" 1.00 complete with or without the
trailing period, so this was never about provider-added punctuation; the audio EOU
hears the continuation clearly (0.054) but the text-complete tier cuts its hold to
0.35s, under the 1.5s pause. `local` merges it about one run in five.

Disabling that tier makes it merge 5/5 with no other scenario regressing — but the
suite had nothing covering what the tier protects, so that result was measured
blind. `support_closer` now covers it: Smart Turn under-scores "That's all." at
0.115 while the text reads finished, and removing the tier costs **2.7s of dead
air** there (speech end to reply: 0.91s with the tier, 3.59s without). Thirteen of
fourteen closers measured score above 0.5 and never reach the tier at all, so the
population it protects is small — but 2.7s is far too expensive to trade for the
fragment case, so the shipped behavior stands.

Gating the tier on the audio score does not resolve it either. At the tier, closers
score {0.115} and fragments {0.054, 0.376}: the fragments straddle the closer, so
no threshold separates them.

**Careful with `eou→audio`: it is measured from the *accepted* commit, so it does
not include hold time.** A detector that holds for three seconds and then answers
in 200ms reports 200ms. Cross-detector latency comparisons in this README are only
meaningful where holds do not fire; where they do, read the event timeline instead.

**Flux's `end_of_turn_confidence` does not rescue it.** Across all 28 commits in a
suite run, mid-thought fragments scored 0.690–0.920 and genuine turn ends
0.701–0.904 — overlapping almost completely, and the most confident "end of turn"
in the suite was a fragment mid-account-number. The best available threshold
catches 4 of 5 fragments while wrongly holding 35% of real turn ends, roughly
1.5s of dead air on a third of all turns. It measures "has speech stopped", not
"has the thought finished". Measured and dropped rather than built.

**The punctuation rescue in `effective_text_eou` is load-bearing in both
directions.** It promotes a Namo-incomplete verdict whenever the text ends in
terminal punctuation, which wrongly rescues `support_pause`'s fragment (Namo 0.00,
correct) and rightly rescues "Pepperoni pizza, please." (Namo 0.00, wrong).
Restricting it to question marks fixes the first and breaks the second, one for
one — both are Namo 0.00 plus a provider-added period, so nothing separates them.
Left alone.

`banking_confirmation` intermittently splits "Yes, that's correct." in two with a
self-barge-in, and `banking_digits_pause` under `provider` still emits the one
genuine ghost turn the suite reproduces. Both are recorded as known failures with
written reasons, scoped to the detectors where they actually occur, so they report
on every run without blocking the gate.

Three cells are baselined so far:

| cell | gated | p50 | p95 | known failures |
|---|---|---:|---:|---:|
| `deepgram-flux/local` | 19/19 | 228ms | 300ms | 2 |
| `deepgram-flux/lexical` | 19/19 | 230ms | 301ms | 2 |
| `deepgram-flux/provider` | 16/16 | 222ms | 326ms | 3 |

The three are indistinguishable at the median now. `provider` carries the extra
known failure — it is the only configuration that still splits a digit string
across a pause and emits the one ghost turn the suite reproduces — and it has the
longest tail, so it still looks like the wrong default on Flux. `local` and
`lexical` are otherwise equivalent here, which means the audio EOU is not currently
buying anything Namo does not already provide, at least on Flux with clean audio.
Three cells is not a matrix; worth confirming across STT backends before changing
any default.

## Running

```bash
export ELEVENLABS_API_KEY=...
export DEEPGRAM_API_KEY=...

uv run python benchmarks/voice/cli.py                        # all scenarios
uv run python benchmarks/voice/cli.py --list                 # what exists
uv run python benchmarks/voice/cli.py -s barge_in -s pause   # a subset
uv run python benchmarks/voice/cli.py --detector lexical     # Timbal's HOLD path
uv run python benchmarks/voice/cli.py --stt elevenlabs
uv run python benchmarks/voice/cli.py --quick                # ~30s subset, one per domain
uv run python benchmarks/voice/cli.py --repeat 3              # variance + flaky detection
uv run python benchmarks/voice/cli.py --quiet                # results only
uv run python benchmarks/voice/cli.py --dump                 # WAVs to results/dumps/
uv run python benchmarks/voice/cli.py --update-baseline       # accept current behavior
```

Exit code is non-zero when any expectation fails or a regression is gated.

## Results and the regression gate

Every run appends one JSON line per scenario to `results/run-<timestamp>.jsonl`,
so a run stays analyzable after the fact rather than only pass/fail in the moment.

Gating compares against `baseline.json` (committed — deliberately not under the
gitignored `results/`), one entry per `stt/detector` label so a single file covers
the whole matrix. It fails on **movement**, never on absolute thresholds, since
STT providers drift under us and yesterday's ceiling is tomorrow's false alarm:

| Gated | Not gated |
|---|---|
| a scenario's pass rate drops | speedups |
| ghost turns increase | brand-new scenarios (nothing to compare) |
| median latency worsens >15% | latency with fewer than 20 samples |

**Latency gates on p50, not p95.** With ~8 latency samples, p95 is the maximum in
disguise: one slow turn moved it 394ms → 477ms on an unchanged tree while all five
scenarios passed. The scorecard prints `n=` and says "ungated: too few samples"
outright, so nobody reads stability into a number that has none.

Measured on this suite, p50 is worth gating and the tail is not — two independent
`--repeat 3` runs on an unchanged tree:

| Statistic | Run A | Run B | Drift |
|---|---:|---:|---:|
| p50 | 298ms | 299ms | 0.3% |
| p95 | 383ms | 375ms | 2% |
| max | 462ms | 378ms | 18% |

At 20 scenarios a single pass produces 28 latency samples, so `--repeat 1` clears
the floor on its own. Repeats still buy flaky detection: any scenario that passes
some repeats and fails others is reported by name as **FLAKY** — usually a real
race rather than a bad test.

## Known failures

A scenario can declare `known_failure={detector_or_star: reason}`. It still runs
and still reports, but it does not gate, and its *desired* expectations stay in
the file rather than being rewritten to match broken behavior. Known failures are
excluded from `per_scenario`, so marking one cannot quietly bake a defect into the
baseline. If one starts passing, that's reported as an unexpected pass telling you
to drop the marker — unless it's also `intermittent=True`, where passing sometimes
means nothing.

`--update-baseline` refuses to save while any ungated run is failing. Otherwise
the first bad run silently becomes the accepted status quo. Fix it, or name it a
known failure with a written reason.

Audio tooling, standalone:

```bash
uv run python benchmarks/voice/synth.py "I'd like to order a large"   # listen to a clip
uv run python benchmarks/voice/synth.py --verify                      # reproducibility check
```

## Writing a scenario

A scenario is a script plus expectations. Silence is explicit and exact, because
gap duration is what turn detection actually keys on:

```python
Scenario(
    id="pause",
    domain="food_ordering",
    replies=["Sure — one large pepperoni. Anything else?"],
    script=[
        *fluent("I'd like to order a large", 1.6, "pepperoni pizza please."),
        Silence(3.0),
    ],
    expect=[Merged("I'd like to order a large pepperoni pizza please."), NoErrors()],
)
```

**Use `fluent()` for any pause *inside* a sentence.** It renders the whole sentence
in one TTS call and slices it at character timestamps, so each fragment keeps the
intonation it actually has mid-sentence. Writing that example as three separate
steps — `Say(...)`, `Silence(1.6)`, `Say(...)` — is the trap described above: it
hands the detector a fragment that sounds finished and tests TTS phrasing instead
of turn-taking. Plain `Say` is for genuinely separate utterances, like the second
half of a barge-in.

`AwaitAssistantAudio(offset_ms=600)` is the reactive step that makes barge-in
testable — the interruption lands at a known offset into the reply rather than
wherever a fixed sleep happens to fall.

Expectations: `UserTurns`, `Merged`, `NoGhostTurns`, `Interrupted`, `HeardPrefix`,
`NoAgentReply`, `MaxLatency`, `NoErrors`. Text comparisons use normalized
similarity — **never assert verbatim transcripts**, STT wobbles between runs.

## Per-detector expectations

The same observable behavior can be correct for different reasons, so scenarios
can override expectations per detector (`expect_by_detector`) or declare which
detectors they're meaningful for at all (`detectors`).

This matters: with `--stt deepgram-flux --detector provider`, Flux holds through
the `pause` scenario's 1.6s gap and merges inside its own turn machine, so that
run exercises none of Timbal's HOLD path. The same scenario under `--detector
lexical` does. `hesitation` is skipped entirely for `provider`, where trusting
the provider's commit means a bare "Um..." legitimately starts a turn.

## What the current library measures

`pause` (1.6s gap, merges) and `long_pause` (3.0s gap, splits) bracket Deepgram
Flux's turn window between those two durations. `barge_in` and `barge_in_late`
interrupt the same reply at 600ms and 2500ms, yielding heard prefixes of ~39 and
~77 characters — which is how you check the truncation path scales with real
playback position.

## Voices

Two distinct ElevenLabs voices — replaying the user in the assistant's voice
would trip the session's echo heuristics on legitimate speech.

| Env var | Default | Role |
|---|---|---|
| `ELEVENLABS_VOICE_ID` | `1SM7GgM6IMuvQlz2BwM3` | assistant (TTS) |
| `TIMBAL_BENCH_USER_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | user (replayed) |

Both must exist on your account; cloned and custom voices are account-specific,
so override if the defaults 404.

## Cache

Synthesized clips land in `cache/`, keyed by a hash of
`(text, voice_id, tts_model)` — nothing is generated twice unless the script text
changes. Both `cache/` and `results/` are gitignored: audio does not belong in
git, and the generation script plus a content-addressed cache is reproducible
enough.

## Reading the output

Real-time pacing is mandatory, so a run takes as long as the conversation.

```
=== barge_in  (deepgram-flux/provider) ===
  [+  0.00s] session_started
  [+  0.00s] say               "Tell me about your return policy."
  [+  1.63s] partial           "Tell me about your return policy."
  [+  1.99s] await_audio       600ms into reply
  [+  2.46s] committed         "Tell me about your return policy."
  [+  3.08s] metrics           eou→audio 373.3ms  segments 1  acks True  vad_eou False
  [+  3.45s] say               "Actually, cancel that."
  [+  4.92s] interrupted       heard: 'Our return policy allows returns within'
```

`acks True` confirms the harness is exercising the client-truth playback path
rather than the wall-clock estimate fallback — production behavior, not a
degraded fallback.
