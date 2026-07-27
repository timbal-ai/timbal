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

## Status: 38 scenarios across 5 domains, matrix, scoring and gating in place

| File | Role |
|---|---|
| `synth.py` | ElevenLabs → cached PCM16 16k, fluent-utterance slicing, PCM/frame/WAV helpers |
| `scenario.py` | script primitives, declarative expectations, scenario library |
| `harness.py` | fake browser: paced feeder, reactive driver, playback ack pump |
| `score.py` | JSONL records, scorecard aggregation, baseline diffing |
| `cli.py` | runner, reporting, regression gate |

### The echo suppressor fails on echo it cannot read

`--aec-leak GAIN` mixes the assistant's own output back into the mic, standing in
for an echo canceller that does not fully cancel. Until it existed, all ~3,000
runs fed clean user-only audio, so `_likely_stt_echo` — the guard that stops the
assistant interrupting itself on speaker bleed — had never once been exercised.
`coding_barge_in_echo` only ever proved it does not fire on genuine speech.

Measured on `deepgram-flux/local` with `support_echo_silence`, where the user
says nothing at all for the whole reply:

| leak gain | outcome |
|---|---|
| 0.0 (every prior run) | clean |
| 0.15 | clean |
| 0.20 | 1 failure in 3 |
| 0.30 | **4 failures in 6** |

Failure means the assistant cut itself off mid-reply and committed its own words
as a user turn. Real barge-in still passes at 0.30, so this is under-suppression,
not over-suppression.

**The mechanism is the interesting part.** The committed ghosts were `'Our
retailers.'`, `'Arise retailers.'` and `'has aroused retailers.'` — all manglings
of the same phrase from the reply's tail, "...authorized retailers." The
suppressor is a text-similarity check against what the assistant just said, so
echo the STT transcribes *badly enough* stops resembling its source and passes
the filter built to catch it. Clean echo gets suppressed; garbled echo does not,
which inverts the intuition that louder bleed is the dangerous case.

That also explains why the boundary is probabilistic rather than a threshold: it
depends on how the STT happens to mis-hear a given phrase, not on the gain alone.

**The "clean at 0.15" row above is an artifact of measuring one scenario.** The
leak axis had only ever been run against `support_echo_silence`; running the whole
suite at 0.15 on `deepgram-flux/local` gives 57/78 with **12 ghost turns and 9
scenarios failing**, every one of them `interrupted=True` with a fragment of the
assistant's own reply committed as a user turn — `'Add memory access.'` from "a bad
memory access", `'exactly once.'`, `'retailers. Stop.'`, `'Hopefully, you should see
a doctor.'`. `support_simple` — a single question, one reply, no pauses — has the
assistant interrupting itself on 3 of 4 repeats. So a merely mediocre echo canceller
does not degrade turn-taking at the margins; it breaks the ordinary case, and the
suite's clean bill of health came entirely from never having fed it echo.

**Fixed by comparing against a length-matched window.** The cap above is entirely an
artifact of scoring a short commit against a long tail, so the repair is to score it
against a slice of the tail its own length, aligned on the longest run the two share:

```python
block = SequenceMatcher(None, c, tail).find_longest_match(0, len(c), 0, len(tail))
start = max(0, min(len(tail) - len(c), block.b - block.a))
return SequenceMatcher(None, c, tail[start : start + len(c)]).ratio() >= 0.65
```

Anchoring beats sliding every window, which is the obvious alternative and measurably
worse: a weak anchor yields a badly-matched window, so coincidental resemblance
elsewhere in the reply cannot inflate the score. On the adversarial case it is the
difference between 0.75 and 0.58. The threshold is calibrated on transcripts from leak
runs rather than picked — garbled echo of a known reply scores 0.68–1.00 and genuine
barge-ins against the same replies score 0.24–0.58, two classes that do not overlap —
and 0.65 sits in that gap nearer the echo side deliberately, since suppressing real
speech makes the assistant uninterruptible, the worse of the two failures.

| | before | after |
|---|---|---|
| leak 0.15, gated | 44/48 and 55/60 (~92%) | **40/40 (100%)** |
| leak 0.15, ghost turns | 12 per 78 runs | **6** |
| clean, 4 baselined cells | 100% | **100%, no regressions** (126 runs) |

Those numbers stood for a few hours. The clean-run check below missed the case that
undoes them — see "the suppressor ate a real turn on a clean run".

The clean-run check is the one that matters for shipping it: a suppressor that catches
more echo can start eating genuine interruptions. All seven baselined cells were run,
and the decisive one is `deepgram-flux/provider`, where `_likely_stt_echo` is the *only*
semantic filter between a partial and cancelling the reply — no VAD corroboration, no
EOU, nothing to catch an over-suppression. It scores 53/54 with all ten barge-in
scenarios passing every repeat, `coding_barge_in_echo` included, while
`support_echo_silence` stays uninterrupted. Its one flake is `food_long_pause` with
`interrupted=False`, so the assistant never spoke over it and the echo check was never
consulted.

The ElevenLabs cells look worse (17/24 and 21/24 on the barge-in subset) and are not:
the genuine barge-in is heard, and the extra turn is `'Yes.'` or `'Yeah.'` — the
provider's documented hallucination on trailing silence, the same family as the `'Bye.'`
above. 17/24 is also up from the 15/24 measured there before this change.

Reading that comparison surfaced a gate defect worth knowing about. Per-scenario rates
are matched by name and safe on a filtered run, but ghost turns, latency and dead air
are aggregates over whatever subset ran, so comparing them against a full-suite
baseline compares populations. The barge-in subset alone carries most of the suite's
ghost turns and duly reported "ghost turns 1 → 3" while being clean; a subset that
*excludes* them would have reported an improvement just as falsely. `compare()` now
takes `partial` and skips aggregates for filtered runs, which is the same reasoning
that already makes `--update-baseline` refuse them.

**The residual had two causes, and the second was self-inflicted.** Both call sites
gated the check behind `state.assistant_active`, so echo committing just *after*
playback ended was never tested. `coding_followup_after_reply` was the clean
demonstration: it commits `'memory access.'`, an exact substring of its own reply, and
nothing looks. `TurnState` now carries `seconds_since_assistant_active` and the check
runs for 2s past playback — bounded, because `assistant_text` outlives its turn, so an
unbounded window would let a user utterance resembling the last reply be dropped, and
dropping it means no new turn starts to clear that text.

That alone fixed nothing, which is the interesting part. Tracing the scenario showed the
suppressor working perfectly and the rescue path undoing it:

```
stt_committed_received   'memory access.'   audio_playing=True
stt_commit_ignored       reason=echo               <- suppressed correctly
stt_stale_partial_commit mic_quiet=True stale_secs=1.9
stt_stale_partial_synthesized 'memory access.'     <- resurrected 3s later
stt_committed_accepted   action=new_turn           <- ghost turn
```

An IGNOREd commit does not bump `_last_commit_at`, so suppressed echo still looks like a
stranded partial to the stale-partial watchdog, which synthesizes it back once the grace
window has closed. The watchdog exists to rescue user speech that an over-eager AEC
ducked below the provider's commit threshold — which is the one thing echo is not — so
it now runs the same echo check before rescuing. Together: `coding_followup_after_reply`
0/3 → 3/3, and ghost turns across the whole 78-run leak suite 12 → 2.

Worth noting the shape of this bug, since the codebase invites more of it: two fixes
that are individually correct, landed a day apart, in different files, that cancel each
other. Neither test suite could catch it because each mechanism is right in isolation.

| leak 0.15, deepgram-flux/local | ghosts (78 runs) | `coding_followup_after_reply` |
| --- | --- | --- |
| before the suppressor fix | 12 | 0/2 |
| window-anchored suppressor | 6 | 0/3 |
| \+ grace window | 6 | 0/3 |
| \+ rescue guard | **2** | **3/3** |

Eleven of the thirteen leak markers were retired by this. One of the two that remained is
different in kind and survives everything below: `support_barge_in_one_word` gets the echo
tail and the user's word in a *single* commit (`'retailers. Stop.'`), so suppressing it
loses the barge-in the scenario tests — that needs the echo span located inside the commit,
not a verdict on the whole string.

**Then the suppressor ate a real turn on a clean run, which reframes everything above.**
`food_long_pause` on `elevenlabs/lexical` went 4/4 → 0/4. The user says "Pepperoni pizza,
please." after a reply of "Got it — a large pepperoni pizza.", and the check drops it —
on a run with no leak injected at all, suppressing echo that could not exist. It scores
0.67, inside the 0.68–1.00 band the threshold was calibrated on.

The calibration was not wrong about its samples; it was drawn only from barge-ins, where
the user is *changing* the subject and shares little vocabulary with the reply. But
confirmations, disambiguation and digit readback all put the user's next words into the
reply by design, and there the populations interleave — genuine speech at 0.67 and 0.78
against real echo at 0.76 and 0.79. No threshold splits that, so this is not a tuning
problem, and a per-utterance "is this echo?" has no reliable text-only answer.

**So resemblance was latched behind proof.** The check is now two branches with different
standing: a verbatim substring of what is playing is evidence and always counts, while
garbled resemblance is a guess, armed only once the same session has produced verbatim
echo. The argument was that base rate decides it — when AEC works there is no echo in the
signal at all, so every fuzzy suppression is a false positive by construction, and when it
leaks the verbatim branch will see it.

**It never arms, and the reason invalidates the design rather than its tuning.** Every
remaining ghost under leak is garbled and not one is verbatim: `'surprised retailers.'`,
`'Veroni, anything else?'`, `'Nash exactly once.'`, `'Not anything else?'`. Leak severity
determines transcription quality. A leak quiet enough to be garbled is garbled
*consistently* — the STT is transcribing faint, distorted audio, so of course every copy
comes back wrong — while verbatim echo belongs to a louder, cleaner leak. The two are
near-disjoint regimes rather than sequential stages, so "verbatim echo seen" does not
predict "garbled echo present", and a 20-second scenario never earns its proof.

| leak 0.15, deepgram-flux/local | gated pass | ghost turns |
| --- | --- | --- |
| unconditional resemblance | ~100% (13 markers) | 2 |
| latched behind proof | 85% (2 markers) | 10 |

**Kept regardless, because it changes which error you get.** Unconditional, a *working*
AEC drops genuine user speech and the assistant answers something the user never said.
Latched, a *leaking* AEC produces ghost turns. A ghost turn on a broken microphone is
plainly the lesser harm, so the latch dominates on worst case even though it gives the
leak numbers back. The real fix is timing, which is the only signal that survives both
regimes: echo of a word arrives while that word is playing, and a user repeating it does
so afterwards. Text similarity cannot see that and never will.

One narrowing worth recording, since the earlier entry over-claimed: the latch restored
`food_long_pause` on `elevenlabs/lexical` only. It still fails on `deepgram-flux/provider`
and `deepgram-nova/lexical`, so those are a different cause and folding them into the echo
story was wrong.

**The axis baselines now, and the old refusal rule was what blocked it.** At `--repeat 4`
it scores 85% with per-scenario rates recorded, and only `banking_digits` and
`banking_digits_pause` fail every repeat. The previous rule — refuse if anything fails —
demanded that a probabilistic axis be perfect before it could gate at all, so the axis
never gated and its markers were fitted to whichever run was in front of me. Refusal now
triggers only on a scenario that fails *every* repeat; flaky ones are recorded at their
measured rate. The repeat count matters more than expected: at `--repeat 2` five scenarios
looked hard and three of those five turned out to be intermittent at 4.

Digit readback is the honest residual, and it is the worst case from both directions at
once. Digits never produce verbatim echo — a run of digits garbles into other digits — so
the filter never arms. And the obvious repair, trusting resemblance here because a readback
plainly echoes, is exactly backwards: the reply *is* the user's digits, so a genuine
correction ("Four four eight." against "…account four four seven.") scores 0.75. It is
where resemblance is least informative and most tempting.

Still open on this axis: 4 session errors per 120 runs under leak, untriaged.

**The latch then armed itself on a clean run, and the mechanism refutes the claim above.**
`food_long_pause` went 100% → 0% on `deepgram-flux/provider` and `deepgram-nova/lexical`,
and the trace shows `'pepperoni pizza, please.'` suppressed as echo while the reply is
still playing — which only the *resemblance* branch can do, so the latch was armed on a
run with no leak. What armed it was the user's own partials. The STT emits `Pepperoni`,
then `Pepperoni pizza`, and `'pepperoni pizza'` is a verbatim substring of "Got it — a
large pepperoni pizza."

So "a verbatim substring cannot also be produced by a user saying the same thing" is
false, and confirmation flows are exactly where it is false: the reply restates the
request, so the user's words are a substring of it by construction. Length does not
separate the cases either — the echo this exists to catch is `'memory access'` at 13
characters and the false proof is `'pepperoni pizza'` at 15.

What does separate them is *what kind of event* the evidence is. A partial is text the STT
is still revising, and reading it as evidence about the microphone is a category error;
a commit is a claim the provider has settled on. Only commits arm the latch now.
Suppressing an echo-shaped partial still happens and predates all of this — that was never
the problem, arming on it was. `food_long_pause` returns to 100% on both cells, and
`banking_correction` and `support_barge_in`, which regressed the same way, clear with it.

**Full matrix, 12 cells × 39 scenarios × 3 repeats: zero ghost turns.** Not one in 1386
runs, against a suite where they were routine this morning. That is the clearest signal
that the day's filters — resemblance latched to commits, the unvoiced-commit guard, the
watchdog's echo check — are subtracting spurious turns without eating real ones.

It also prices the hold path for the first time, because five cells had never been
baselined and four of them are holdless. Raw rates, before the pause family was marked
on those cells — they read 100% now, which is the point of marking, so these numbers
only exist here:

| detector | ElevenLabs | Flux | Nova |
|---|---|---|---|
| `heuristic` (no hold) | 69% | **90%** | 67% |
| `provider` (no hold) | 68% | 96% | 65% |
| `lexical` | 98% | 100% | 96% |
| `local` | 100% | 100% | 99% |

The holdless cells fail one family — pause merging — and fail it identically: `nova/heuristic`
and `nova/provider` miss the same 12 scenarios, both ElevenLabs cells the same 11. With no
hold, every fragment the STT commits is a turn boundary and nothing downstream can merge
it, so roughly 30 points of the suite is what holding buys.

`deepgram-flux/heuristic` at 90% is the exception that makes it an STT × detector fact
rather than a detector one: Flux holds the fragment provider-side before any detector sees
it, so the same holdless detector merges nine of the scenarios Nova's cannot. That is why
the markers are written per cell — a blanket "holdless cannot merge" rule would mark those
nine as expected failures on Flux and hide any future regression in them.

**All twelve cells now gate.** Previously seven did, and two of those seven compared against
`pass=1.0` entries recorded before the day's changes, so they would have passed a gate they
should have failed. What is left is flaky rather than broken and recorded at measured rates:
`coding_double_pause` on three cells, `medical_barge_in` and `banking_digits_pause` on
`elevenlabs/lexical`, `food_long_pause` and `coding_pause` on `deepgram-nova/local`. One
ghost turn survives, on `deepgram-flux/provider` in `medical_simple` — the only one in the
matrix, and untriaged.

Two things this run does *not* establish, worth stating so the baseline is not over-read.
Latency and dead air were measured at `--jobs 6`, so they gate only against future runs at
the same concurrency. And the leak axis has its own baseline at 85%, which is the number to
watch for echo work — the clean matrix says nothing about it.

Both mechanisms that consume mic energy were ruled out as causes by disabling them:
the hold's VAD extension (`HOLD_VAD_MAX_EXTENSION_SECS = 0`) and the stale-partial
watchdog's mic-quiet anchor, separately and together, fail at the same rate. That
matters because both look like plausible suspects — echo carries energy, so in
principle it can defer a hold and can go quiet in a way that triggers a rescue — and
neither is responsible. The under-suppression is the whole story.

**The fuzzy branch of that check is unreachable, provably.** `_likely_stt_echo`
falls back to `SequenceMatcher(c, tail).ratio() >= 0.68`, where `tail` is
`max(3*len(c), 100)` characters. Since `ratio()` is `2M/(len(c)+len(tail))` with
`M <= len(c)`, sizing the tail at three times the commit caps the score at
exactly **0.50** — a *perfect* match scores well under the threshold:

| commit chars | tail chars | best possible ratio |
|---:|---:|---:|
| 10 | 100 | 0.182 |
| 33 | 100 | 0.496 |
| 60 | 180 | 0.500 |
| 200 | 400 | 0.667 |

It can only fire while the assistant's whole reply is still shorter than the
~100-char window. Echo is guarded during the opening of a reply and unguarded for
the rest of it, with nothing but exact-substring matching left — which is why
every ghost came from the tail of a long reply and every one was a near-miss.

**A window-matching fix was tried and reverted.** Replacing the whole-string
ratio with a best-window comparison separates the recorded ghosts (0.615–0.963)
from clean user phrases (0.340–0.500) with a clear gap, and at a 0.58 threshold
it takes `support_echo_silence` at 0.30 leak from 4-in-6 failing to 6/6 passing.
It also breaks barge-in outright: `support_barge_in`, which passes at 0.30 leak
today, fails every run, and the assistant becomes uninterruptible — the worse
failure of the two.

The flaw was in how the threshold was set. Scoring *clean* user phrases against
the reply is the wrong sample: under leak the STT transcribes a **blend** of user
speech and echo, and the blend lands between the two distributions rather than
inside the genuine one. No threshold on that comparison separates them, because
the thing being measured is not two populations but a continuum.

Which leaves the audio reference. Real AEC correlates the mic against recently
played output, and the harness already has both signals. Note that Silero is
*not* the answer despite being available: `_vad_vetoes_barge_in` says plainly
that speaker echo carries energy, so VAD cannot tell it from speech.

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

**Two timings, and confusing them will mislead you.** `eou→audio` is measured
from the *accepted* commit, so it does not include hold time: a detector that
holds three seconds and then answers in 200ms reports 200ms. `dead air` is
speech end → turn accepted, and is the only metric that can see a hold. Both are
gated per cell against the baseline. A change can move one without the other —
removing the text-complete hold tier leaves `eou→audio` flat while adding 2.6s of
dead air to six barge-in cells.

Use `MaxDeadAir` on any scenario where the speaker has genuinely finished, so
that buying a merge with silence has to be declared rather than slipping through
as a free win.

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
self-barge-in, and `banking_digits_pause` under `provider` emits the one genuine
ghost turn the suite reproduces. Both are recorded as known failures with written
reasons, scoped to the detectors where they actually occur, so they report on
every run without blocking the gate.

The digit-string one is a good argument for `--repeat`. A single run showed it
passing and the marker looked stale; at `--repeat 6` it merged 5 times and split
once, emitting a ghost `'I'`. A rate like that is invisible to any single run in
either direction — it would read as fixed five times out of six and as reliably
broken the sixth. It is marked `intermittent` so passing no longer reports as an
unexpected pass, since passing genuinely proves nothing here.

**Adding a second STT backend broke seven cells that were working correctly.**
`banking_digits` failed in seven of twelve cells while committing exactly one turn
in all twelve: Flux writes "four four seven two nine one" and Nova writes "447291"
for the same audio. Perfect turn-taking, reported as a turn-taking defect. The
suite had already been bitten by this once and fixed it by rewriting the script to
match Flux's spelling, which is precisely the fix that does not survive a second
provider. `normalize()` now spells digit runs out, so the comparison is
representation-independent.

**A scenario was quietly grading providers against each other's habits.**
`food_long_pause` asserted a merge under `lexical` and `local` because that had
been *observed* on Flux, so Nova failed it for splitting a 3.0s pause — which is
perfectly defensible behavior, and which the trace showed Flux never even gave a
detector the chance to get wrong. At that gap neither answer is knowably right:
holding costs 3s of dead air, splitting sends a fragment to the LLM. It now asserts
`ContentPreserved` — everything spoken arrived, across however many turns — which
still fails ElevenLabs for dropping half the order while letting both turn-taking
policies pass. Any expectation that encodes an observation rather than a
requirement will do this as soon as the matrix widens.

Three cells are baselined so far:

| cell | gated | p50 | p95 | known failures |
|---|---|---:|---:|---:|
| `deepgram-flux/local` | 19/19 | 214ms | 283ms | 2 |
| `deepgram-flux/lexical` | 19/19 | 235ms | 328ms | 2 |
| `deepgram-flux/provider` | 16/16 | 234ms | 324ms | 3 |

The three are indistinguishable at the median: the 21ms spread is inside the ±6%
drift two identical serial runs show (see below), so nobody should read a winner
out of this column.

### Measuring only Flux was measuring Flux

The Flux column cannot tell detectors apart, because Flux does turn detection
itself. Every detector scores 95–100% on it, and the reasonable-looking conclusion
from that column alone — "the audio EOU buys nothing Namo doesn't already provide"
— was an artifact of the choice of backend. Running the same 21 scenarios against
Nova and ElevenLabs, which return transcripts without deciding turns, separates
them immediately (2 repeats, 456 runs):

| detector | Flux | Nova | ElevenLabs |
|---|---:|---:|---:|
| `heuristic` | 95% | 79% | 79% |
| `local` | **100%** | **89%** | **95%** |
| `lexical` | 100% | 76% | 87% |
| `provider` | 97% | 75% | 75% |

`local` wins every column. Restricting to the mid-sentence pause merges, the only
family where detectors differ at all, makes the mechanism plain:

| detector | Flux | Nova | ElevenLabs |
|---|---:|---:|---:|
| `heuristic` | 80% | 20% | 20% |
| `local` | 100% | **60%** | **80%** |
| `lexical` | 100% | 20% | 70% |
| `provider` | 100% | **0%** | **0%** |

`provider` goes from perfect to zero. That is the expected result stated plainly:
it defers to the STT, and off Flux there is nothing to defer to. Every other
scenario family is flat — barge-in is 90–100% in all twelve cells — so pause
merging carries the entire discriminating signal in the suite.

The real finding is that **Smart Turn earns its keep precisely where the STT does
not endpoint for you**, which is invisible on Flux. `local` beats `lexical` by 40
points on Nova and 10 on ElevenLabs: hearing that the speaker's pitch is still
mid-phrase survives a transcript that reads like a finished sentence, and text
alone does not. If Timbal is ever pointed at a plain streaming ASR, `local` is the
default and `provider` is close to unusable for pauses.

One caveat: ElevenLabs drops content on hard inputs independently of turn-taking
(it committed "I'd like to order a large" and silently lost the pizza), which
`ContentPreserved` now catches but which is an STT-quality difference rather than
a turn-taking one.

### Seven cells gate, five do not, and that is deliberate

A census of the nine ungated cells at `--repeat 3` (1,041 runs) turned up 77
failing scenario-cells. Marking all 77 would have been the wrong answer, because
they are not 77 findings:

| detector | unmarked failing cells |
|---|---:|
| `heuristic` | 29 |
| `provider` | 24 |
| `lexical` | 14 |
| `local` | 10 |

`heuristic` has no EOU model at all and `provider` delegates to the STT, so
neither can hold a mid-sentence pause on a backend that does not endpoint for
them. Their 53 failures are one architectural fact recorded 53 times, and
marking them would turn `known_failure` from a record of defects into wallpaper.

So the four cells where the detector does real work — `deepgram-nova` and
`elevenlabs` × `local` and `lexical` — are now baselined, taking the gate from 3
cells to **7**. That cost 24 cell-scoped markers, each citing a measured rate.
The remaining five (`deepgram-flux/heuristic`, and `heuristic`/`provider` on both
other backends) are measured on every run and reported, but not gated: their
pause-family failures are a property of choosing a detector with nothing to hold
with, which is what the detector table above is for.

The markers cluster into three named reasons rather than 24 bespoke ones —
`_TEXT_ONLY_PAUSE_SHORTFALL` (Namo alone cannot hold the pause off Flux),
`_AUDIO_PAUSE_SHORTFALL` (Smart Turn hears it but the hold does not survive), and
`_EL_OVERMERGE` — so a fix to any one of them shows up as a block of unexpected
passes rather than a scatter.

The ranking survived the suite growing to 38 (numbers are not comparable to the
table above — the added scenarios are deliberately harder): `local` 92/81/92,
`lexical` 94/71/89, `heuristic` 89/65/76, `provider` 91/63/75. `local` still wins
both non-Flux columns by 10 points or more; `lexical` edges it by 2 on Flux, where
the backend is doing the work anyway.

### Harder cases: 21 → 38 scenarios

Barge-in scored 90–100% in all twelve cells, which meant five scenarios were
spending real time confirming something already known. Seventeen were added to
attack the edges instead: barge-in at 150ms instead of a comfortable 600, a
one-word "Stop.", two interruptions in one session, an interruption made of the
assistant's own words, a backchannel that must *not* interrupt, three-part
sentences, self-corrections, a 12-second run-on, two finished sentences 0.9s
apart that must not merge, and four seconds of pure silence.

**Barge-in survived all of it.** Instant, one-word, double and echoing barge-ins
pass in all twelve cells, as do pure silence, a bare "No." and a follow-up taken
after the assistant stops. The one-word case is the surprising pass:
`MIN_BARGE_IN_PARTIAL_WORDS` drops single-word *partials* as mic blips, so "Stop."
should have been ignored — the commit still arrives and still cuts the assistant
off. The gate delays a one-word barge-in rather than losing it.

Every new failure is a stop mid-thought, and the detector rankings barely move
(17 new scenarios, 12 cells):

| detector | Flux | Nova | ElevenLabs |
|---|---:|---:|---:|
| `heuristic` | 82% | 47% | 70% |
| `local` | 82% | **70%** | **88%** |
| `lexical` | 88% | 64% | 88% |
| `provider` | 82% | 47% | 70% |

**The metric was hiding failures.** `coding_double_pause` passed on ElevenLabs
having committed "The build fails when I import the module" — "inside a test"
simply gone. Whole-string similarity scored that 0.84 against the full sentence,
over the 0.8 bar, so a third of the utterance vanished and the suite called it a
clean merge. The same bug was passing `banking_digits_pause` on ElevenLabs at 0.85
while "291" was dropped from an account number. `Merged` and `ContentPreserved`
now check each spoken fragment against its best-matching window of the transcript,
where a missing part matches nothing and scores 0.40. Both of those are now
correctly red. A similarity threshold over a long string cannot see a dropped
tail, and the longer the utterance the blinder it gets.

Three findings are the STT's, identical under all four detectors:

- **Nova commits backchannels.** "Mm-hmm" over the assistant is transcribed and
  committed, and every detector then treats it as a barge-in and stops the reply.
  Flux and ElevenLabs swallow the sound and pass. Timbal has no notion of a
  backchannel, so on any STT that transcribes one, an acknowledgement silences the
  assistant.
- **Nova endpoints at sentence boundaries.** The 12-second run-on comes back as
  three turns, split at clause boundaries inside continuous speech.
- **ElevenLabs merges finished sentences.** "I'll have a coffee." and "Actually,
  make it two." 0.9s apart come back as one turn in all four cells. Its endpointer
  waits longer than Flux's, which helps on pauses and hurts here — the suite had
  no scenario punishing an over-merge before this, so a detector could have scored
  well by holding everything.

Two findings sharpen older ones. The spurious single-token `'I'` turn under Flux +
`provider` had been attributed to digit strings for as long as
`banking_digits_pause` was the only scenario producing it; `medical_filler_midway`
now produces it verbatim on ordinary words, so it is the path, not the content.
And running the baselined cells at `--repeat 3` caught `coding_pause` splitting
once in three under `provider` — a scenario baselined as a clean pass, unchanged,
that had only ever been run once per cell.

The trailing-modifier failure now has four instances rather than one, and they
agree: `medical_hesitant_pause`, `medical_self_correction`, `banking_correction`
and `coding_double_pause` all commit a fragment that is a complete sentence in
isolation, and no text signal argues for holding. `banking_correction` is the one
with teeth — split, the agent transfers to account 447 when the caller said 448 —
and it is also the clearest win for Namo anywhere in the suite: `lexical` merges it
3/3 on Flux where `local` and `provider` never do.

Known failures went from 3 to 10 of 38, which is worth watching. Eight of the ten
are scoped to specific cells rather than blanket-marked, and eight are
`intermittent` — but a suite that marks everything interesting gates nothing, and
this is the point at which that stops being a theoretical concern.

### The suite could not see the cost of a hold

`eou→audio` starts counting at the *accepted* commit, so every second a hold
spends deciding costs nothing measurable. Pass/fail only ever saw whether a turn
eventually merged. Between them, the two metrics were blind to turn-taking
timing — and that blindness nearly shipped a bad change.

The text-complete hold tier shortens a hold to 0.35s when the audio model says
"incomplete" but the transcript reads finished. Disabling it looks like an
unambiguous win: **11 scenario-cells fixed, zero regressions**, every `local`
cell converging on 95% (Flux 89→95, Nova 79→95, ElevenLabs 87→95), including
`medical_hesitant_pause` on all three backends.

Timing speech end to commit tells the other half. The tier fires on **21% of
scenario-cells**, and the cost lands almost entirely on cases that were already
correct:

| scenario-cell | cost of removing the tier | was it failing before? |
|---|---:|---|
| `medical_barge_in`, `medical_barge_in_twice` (6 cells) | +2.6s | no — pure loss |
| `support_closer` (3 cells) | +2.6s | no — pure loss |
| `support_pause` (3 cells) | +2.4s | no — pure loss |
| `medical_long_utterance` (Nova) | **+8.2s** | counted as a *fix* |

Six barge-in cells taking 2.6s longer to answer an ordinary question is not
visible anywhere in pass/fail, and an 8.2s "fix" is not a fix. So the tier stays
— but **0.35 was the wrong value, and only a sweep could show that**, because
both endpoints look bad from where we were standing.

`MaxDeadAir` and a per-cell dead-air gate now exist, measuring speech end → turn
accepted, and `sweep.py` can vary a detector parameter without editing product
constants. Sweeping the tier over 0.35 / 0.8 / 1.2 / 2.0 / 3.0 found the shape
nobody had looked for:

| tier timeout | pause merges | dead air p50 | p95 |
|---|---:|---:|---:|
| 0.35 (was) | 58% | 852ms | 2270ms |
| **1.2** (now) | 81% | 889ms | 1944ms |
| 3.0 | 90% | 844ms | 3617ms |

**Median dead air is flat across the whole range.** The tier only fires when
audio says incomplete and text says finished — a minority of turns — so the
median cannot move and the cost is entirely a tail effect. That is exactly why
0.35 read as free and went eight months unexamined: the metric that would have
priced it did not exist, and the one that did could not see it.

Confirmed on the full suite at `--repeat 3`, 684 runs across all three backends
under `local`: correctness **84% → 92%**, with every scenario the tier exists to
protect — `support_closer`, all four barge-ins, `banking_short_reject`,
`coding_followup_after_reply` — unchanged at 100%. `coding_double_pause` goes
33→100%, `medical_long_utterance` 67→100%, `medical_hesitant_pause` 0→67%. The
cost is p95 dead air 1969 → 2683ms. 3.0 scores higher on merges still and is not
worth it: p95 3617ms and `support_pause_short` starts failing.

The default is now 1.2. The gated Flux cells stayed at 100% and their dead-air p50
improved, so they were re-baselined serially against the new default rather than
left holding pre-retune numbers the gate would have had slack against
(725→668, 819→656, 550→489ms).

What the sweep also settled is where this knob *stops*. `medical_self_correction`
tops out at 33% and `banking_correction` at 22% — at every value tried. Those are
the two trailing-modifier cases with real consequences, and no hold duration
fixes them, because their problem is not how long the hold runs but that nothing
arms one when both signals read finished. A fixed timeout is the wrong shape for
that, and a threshold on the audio score cannot discriminate either: the fragment
scores 0.054 and the closer it must not catch scores 0.115.

**Dead air immediately found something unrelated.** On
`deepgram-nova/lexical` a spoken account number sits at **7.5s** — reproducible,
and against 0.4–2.1s in all eleven other cells. Root cause:
`_maybe_start_endpointer` arms the VAD endpointing fast path only when the
detector exposes an audio EOU model, so `LexicalTurnDetector` (text-only) never
arms it, nothing ever sends Deepgram a `Finalize`, and the commit waits on Nova's
own endpointing. `local` commits the same audio in 750ms. Flux hides it entirely
by endpointing itself. The fast path's value — telling the STT to stop — is
independent of *how* EOU was decided, so the one configuration that needs it
most is the one architecturally forbidden from having it.

Getting there killed three hypotheses, each of which looked right:

- **Namo mis-scoring digit strings.** It scores them 0.987–1.000, complete.
- **The stale-partial watchdog rescuing a stranded turn.** The committed text is
  numeral-formatted (`"447291"`) while every partial was word-formatted, so the
  commit is Nova's own `smart_format` final, not a synthesized rescue.
- **`_endpoint_text_score` routing past Namo.** Real bug — the chain checks
  `effective_text_eou` then `fallback_text_eou`, and `LexicalTurnDetector` names
  its predictor `text_eou`, so the endpointer scored partials with the
  punctuation baseline. Fixing it changed dead air by 0ms, because that method is
  only ever called by an armed endpointer, which under `lexical` never exists.
  Dead code for the detector it names.

None of the three was visible in pass/fail or in `eou→audio`.

**Parallelism turned out to be nearly free, which was not the expectation.**
`--jobs` was built assuming concurrent sessions would contend for Silero and Smart
Turn inference and inflate `eou→audio`, so the gate refuses to compare runs
measured at different concurrency. Then the full suite was run four times, twice
each way — p50 per cell, in run order:

| cell | serial | serial | `--jobs 4` | `--jobs 4` |
|---|---:|---:|---:|---:|
| `deepgram-flux/local` | 228ms | 214ms | 214ms | 217ms |
| `deepgram-flux/lexical` | 230ms | 235ms | 213ms | 214ms |
| `deepgram-flux/provider` | 222ms | 234ms | 222ms | 215ms |

Two serial runs of the same cell differ by up to 6%, which is as large as any
serial-to-parallel gap — and the parallel runs are, if anything, slightly faster
and tighter (213–222ms, against 214–235ms serial). There is no separation to find
here.

Real-time pacing is why: a replayed session spends nearly all of its wall clock
asleep between 20ms frames, so four of them interleave in the gaps rather than
competing for CPU. The same 54 runs take 426s serially and 114s at `--jobs 4`, a
3.7× speedup that costs nothing measurable.

The guard stays anyway. It costs one serial run before committing a baseline, and
the headroom above is specific to this machine, this suite size and `--jobs 4` —
none of which the gate can verify at compare time. It fails toward reporting a
number rather than gating on one.

The same table is a warning about the latency figures generally: a 15% gate on a
statistic that drifts 6% between identical runs has less headroom than it looks
like, and any cross-detector comparison under ~20ms is noise.

### ElevenLabs was doing the merging, and that hid a defect in ours

ElevenLabs' dead air sits at p50 ~1.5s against Deepgram's 0.5–0.7s, which read as
a provider property until someone looked: `default_voice_config_from_env` ships
`vad_silence_threshold_secs=1.2` while Nova ships `endpointing=300`. A 4x
asymmetry, in *our* config, with a written justification on neither. The obvious
move was to cut it toward Nova's and collect the second.

Sweeping it says the opposite, and says it flatly:

| `vad_silence_threshold_secs` | pause merges | dead air p50 | p95 |
|---|---:|---:|---:|
| 0.3 | 47% | 728ms | 2388ms |
| 0.5 | 44% | 757ms | 2722ms |
| 0.8 | 59% | 1024ms | 3125ms |
| 1.2 (ships) | **86%** | 1409ms | 2721ms |

Every millisecond of that 1.5s is buying a merge: 39 points of them for 681ms.
And the per-scenario column says who was earning it — `medical_self_correction`
and `banking_correction`, the two cases the hold-tier sweep above could never fix
at any value, score 100% here and 0% at 0.3. **ElevenLabs' VAD was holding them,
not our detector.** Take its 1.2s away and Timbal's own merge path is exposed at
47%, which is also roughly where `deepgram-nova/lexical` has been sitting all
along at `endpointing=300`. Two findings filed separately were one defect, with a
provider's patience masking it on one backend and not the other.

That also killed a day of planned work before it started. Switching ElevenLabs to
`commit_strategy="manual"` — Timbal owning endpointing end to end, which sounds
strictly better — is the 0.3 column with extra steps, plus `rate_limited` being a
*fatal* STT message type and the endpointer silently dropping any commit it skips
for `session_gate` or `commit_interval`.

The defect itself is one line of the `coding_double_pause` trace at 0.3:

```
17:19:00  commit "The build fails when I-"   audio p=0.007  → HOLD 3.0s
17:19:02  commit "I import the module."      audio p=0.035  → HOLD 1.2s
17:19:03  [user starts speaking "inside a test."]
17:19:03  stt_hold_expired                                  ← mid-speech
17:19:04  commit "Inside a test."                            → 2 turns, FAIL
```

`_arm_hold` already refuses to fire mid-utterance — but it keys that on a new STT
*partial* since the commit, and lets Silero only corroborate one, never trigger.
When the provider commits on a short silence the next fragment's audio is in
flight for over a second before any partial exists, so the guard has nothing to
hold on and the timer wins. Silero saw that speech the whole time; the log line
`vad_speech_stop utterance_secs=0.96` sits four lines above the expiry. The
fastest signal in the pipeline was subordinated to the slowest.

Holds now also extend on mic energy alone (`_vad_hears_speech_now`), capped at
3.0s, re-checked every 500ms:

| at `vad_silence_threshold_secs=0.3` | pause merges | dead air p50 | p95 |
|---|---:|---:|---:|
| before | 47% | 728ms | 2388ms |
| after | **77%** | 742ms | 2230ms |

Thirty points for fourteen milliseconds — the tell that those merges were never
being bought with time. They were being lost to a timer firing while the user was
still talking, and waiting for the *user* rather than for a *duration* costs
nothing when the user has in fact stopped. At 1.2 the change is inert (86% → 83%,
inside the noise at n=78), exactly as it should be: ElevenLabs isn't splitting
utterances there, so there is nothing to catch.

The cap is not decoration. Echo surviving an imperfect canceller carries energy —
see the echo-suppressor section — so without it the assistant's own playback could
hold a turn open for as long as it speaks.

**What this changes is the shape of the config question, not just a number.** The
gap between an aggressive threshold and the shipped one has gone from 39 points to
six:

| | pause merges | dead air p50 |
|---|---:|---:|
| 0.3 | 77% | 742ms |
| 1.2 | 83% | 1417ms |

Six points for 675ms on every turn is a trade worth arguing about; 39 points for
the same 675ms was not. Unresolved deliberately: n=78 on one backend, and the
pause family excludes every barge-in by construction, which is precisely where
holding longer would cost.

**Confirming the fix on the full matrix caught the harness scoring it wrong.**
The gate reported `ghost turns 1 → 3` (or 4) against baseline in four cells — and
those are exactly the four where merging improved. `count_ghost_turns` asked
whether a committed turn resembled *one* script entry at ≥0.6 similarity, so a
correct three-fragment merge, being about 2.5x longer than any fragment it was
compared against, resembled none of them:

```
deepgram-nova/local   coding_double_pause   turns=1 ghosts=1 passed=True failures=[]
  committed: ['The build fails when I import the module. Inside a test.']
  script:    ['The build fails when I', 'import the module', 'inside a test.']
```

One turn, zero failures, and the metric calls it invented. Three repeats × the one
scenario in the suite with three fragments, plus one pre-existing real ghost,
accounts for every unit of that "regression" — and it gates. It mis-scored the
inverse too: `medical_long_utterance` split into three commits counted two ghosts,
because a piece of a run-on does not resemble the whole.

Turns are now matched against a *window* of the whole script (`best_window`, which
already existed for `HeardPrefix`) rather than any single entry, since turn
boundaries are the thing under test and are not expected to line up with script
boundaries. Turns under three words keep the per-entry rule: a one-word needle
finds a passable window in almost any script, and a spurious lone `"I"` turn —
which Flux really does emit — has to stay countable. Across all 38 speaking
scenarios, replayed both as one perfect merge and as perfect per-entry turns,
neither shape now registers a ghost.

**Chasing two apparent flakes found the harness had been mis-measuring
ElevenLabs on the whole pause family.** 69 failures in the matrix read
`fragments never heard` — the second half of an utterance absent entirely, the
worst failure class here. 68 were ElevenLabs, across 13 scenarios, concentrated
in `provider` (31) and `heuristic` (29): exactly the two detectors with no
Timbal-side endpointing, so exactly the two that wait on the provider's own
commit.

`AwaitCommit` waited for `len(committed) > base`, with `base` sampled when the
last `Say` *started*. ElevenLabs commits ~1.6s after speech ends, so the
*previous* fragment's commit landed after that sample and satisfied the wait
meant for the fragment under test. What remained was the 0.8s tail plus a 0.5s
drain, against a commit needing ~1.0s. The turn was lost to teardown:

```
[+ 3.19s] committed  "The pain is, um..."     <- fragment 1, 1.7s after it was spoken
[+ 4.05s] await_commit                        <- satisfied instantly by the line above
[+ 4.67s] partial    "Mostly at night."       <- fragment 2 arrives
                                              <- teardown ~4.85s. Never committed.
```

Nova commits in ~600ms, before the next fragment is even spoken, so its
watermark stays honest — which is why Nova showed real splits while ElevenLabs
showed phantom content loss. The bias tracked provider commit latency, so the
slower provider was penalised for being slow *twice*.

A count cannot distinguish a commit for the speech just fed from a late one for
earlier speech, so waits are now timed. Anything arriving after the last `Say`'s
audio finished feeding is the answer, because mic audio is paced at realtime and
the provider cannot have been sent a clip it has not received yet.

That alone is not enough, and the comment it replaced said why: "a fast commit
can land while the clip is still draining". Two different events look like that.
One is this bug — an earlier utterance's commit, arriving late — but the other is
real: the provider can have all the audio it needs before the harness finishes
handing the clip over, and `food_simple` commits `'…a large coffee and a
croissant'` without the final `?` for exactly that reason. Then nothing ever
arrives "after", and a strict timed wait hangs for its full timeout. Requiring it
cost 4 runs in a 111-run cell, `medical_barge_in_twice` taking 46s to fail on
three turns it had already committed correctly.

So a wait resolves on either: something arriving after the speech, or the session
going quiet for 1.0s with something already in hand. The fallback covers the
early commit, and also covers a final `Say` that correctly produces nothing —
which previously only avoided a timeout by accident, because a stale event
satisfied it. `food_backchannel`'s workaround of dropping the step entirely is no
longer load-bearing, though it is left in place.

Under the fix `medical_filler_midway` on `elevenlabs/provider` reports the same
thing 3/3: a split, no errors, nothing lost. `food_long_pause` on
`elevenlabs/lexical` goes 1/3 to 4/4 with content intact, and its known failure —
which blamed the provider for dropping audio — is gone. Two things worth
knowing: latency percentiles for affected cells were computed on a biased subset,
since a turn that never commits also never reports `eou→audio` (the repro went
from 3 samples to 6), so the ElevenLabs latency figures recorded before this are
drawn from a biased subset of turns.

**The re-baseline then caught ElevenLabs dropping barge-in commits outright.**
`medical_barge_in`, `support_barge_in`, `support_barge_in_instant` and
`support_barge_in_late` under `elevenlabs/lexical` went from 12/12 passing to
5/12 in the space of four hours, on cells whose only code change in between was
the harness. The interrupting turn interrupts — `interrupted=True`, heard text
present — and then never commits:

```
[+ 4.84s] say       "Sorry, one more thing."
[+ 5.02s] partial   "I don't know."          <- invented; this is what barges in
[+ 7.02s] partial   "Sorry, one more thing."  <- the real text, as a partial
[+ 8.23s] partial   "Yeah."                   <- invented
[+ 9.24s] partial   "Yeah."                   <- no commit, ever
```

Not the harness, and not the hold change: forcing the waits back to resolving
immediately reproduces it identically, 0/3, and the same scenario passes 3/3 on
`deepgram-nova/lexical` while ElevenLabs passes 4/4 on non-barge-in scenarios. The
mechanism is visible in the trace — ElevenLabs hallucinates on the trailing
silence, and with `commit_strategy="vad"` each invented partial restarts the 1.2s
silence timer that would otherwise commit, so the real utterance is held open
indefinitely. That is the same threshold the merge rates depend on, which makes it
a poor thing to be load-bearing twice.

**But calling it provider-side was letting Timbal off too easily.** The session
already has a watchdog for precisely this — a partial the provider never commits,
whose docstring describes the words hanging "as a '…' caption forever" and which is
meant to fire once "the provider clearly won't". It measured staleness from the
*last partial arrival*, so partials landing 1.0–1.2s apart against its 2.5s
threshold kept it permanently disarmed: the churn that stops ElevenLabs committing
is the same churn that stops the safety net noticing. Logging the anchor shows it
firing at `stale_secs` of 0.0, 0.4 and 0.9 once mic silence counts too — three
rescues that could never have happened on transcript staleness alone.

Anchoring on mic silence instead (`_mic_quiet_for`: Silero heard ≤0.1s of speech in
the window, so the user demonstrably stopped, whatever the provider is still
emitting) takes the four barge-in scenarios from 4/12 to **15/24**, no session
errors, with the three remaining failures flaky rather than hard and
`support_barge_in_instant` passing outright. It is deliberately not gated on
`_vad_evidence`: that flag guards inferences which can *suppress* something the
user did, and this one only rescues speech that would otherwise be lost. Confirmed
free of collateral damage across 156 runs on the four baselined Deepgram cells — no
unexpected failures, no errors, and the single ghost is `banking_confirmation` on
`deepgram-flux/local`, which produced the identical ghost in 1 of 3 repeats before
the change.

**Most of the remainder turned out to be this harness, not the provider.** Two further
changes took the same four scenarios from 15/24 to **24/24** with zero ghost turns, and
the second is the one that mattered.

The first keeps the churn from destroying what the watchdog exists to rescue. A
hallucinated `"Yeah."` overwrote `_latest_partial_text` and refreshed the staleness
anchor, so the real utterance was both gone and un-rescuable; a partial is now ignored
for those two purposes when a stranded partial is already waiting, the new text is not
that one refined, and Silero heard nothing recent. All three conditions are needed —
silence alone would break the watchdog's founding case, where the user speaks quietly
under playback and neither the provider VAD nor Silero registers it. Worth 19/24 alone.

The second: `settle_secs` was a flat 1.0s, and the shortest HOLD tier is 1.2s. A hold
emits nothing while it debounces — that is the entire point of it — so a quiescence rule
shorter than the hold cannot distinguish "finished" from "deliberately waiting", and the
harness tore down mid-hold. The trace is unambiguous: `'Sorry, one more thing.'` commits
as a 1.5s `lexical_hold` and the run ends before it can fire, scoring as the product
losing a turn. It is now derived from the detector's own hold timeouts (`local` 3.5s,
`lexical` 2.0s, the holdless detectors 1.0s) rather than chosen, because the failure
mode of getting it wrong is invisible and looks like someone else's bug.

That correction is worth stating plainly: the five failures written off above as
provider variance were substantially mine. The A/B that appeared to exonerate the wait
changes only compared *whether* a wait resolved, not whether it resolved early, and a
settle that fires mid-hold resolves perfectly happily. Full suite after both, previously
unbaselinable: `elevenlabs/local` 61/62, `elevenlabs/lexical` 63/66, no session errors on
either. Deepgram is unmoved — `flux/lexical` 34/34, `flux/provider` at its usual 26/27,
and the pause family that stood to lose most from a longer settle is 16/16 across
`support_pause`, `support_pause_short`, `food_long_pause` and `coding_double_pause`.

Two fixes that looked obvious and measured worse, both worth not re-trying:

| Hypothesis | Result |
|---|---|
| `commit_strategy="manual"`, so Timbal drives commits instead of ElevenLabs' VAD | **0/12**, and every run also errored `timed out waiting for assistant audio`. The first turn still commits via the endpointer, but the reply never arrives in time. Only viable at all where the endpointer arms — under `heuristic` or `provider` nothing would ever commit. |
| Dropping the synthesis fallback's requirement that the partial text hold still across its 400ms grace, since a hallucination landing inside that window skips the rescue | **4/12** and it synthesized a hallucinated `"Yeah."` as a turn of its own. The stability check is not merely guarding against a racing commit: it is the only thing separating real speech from churn. A stranded turn beats an invented one. |

Consequence for the baseline: 5 of 12 cells carry a current one, all Deepgram, all
at `--jobs 6`. The two ElevenLabs entries are stale — `--jobs 1`, ~31 runs from a
`--quick` subset — and still cannot be refreshed, since a cell with unmarked
failures is refused and the barge-ins remain flaky at 62%. Latency there is
consequently ungated on a concurrency mismatch as well.

Two incidental findings from the same session. `sweep.py` never configured
structlog, so every sweep — including the tier retune above — buried its own
result table under a few hundred thousand DEBUG lines, far enough to be dropped
outright by anything that caps captured output. And `python/tests/voice/` had four
red tests: one asserting the pre-retune `0.35`, and three that set
`session._endpointer` without `_vad_evidence`, which the endpointer-arming fix
split into separate flags — so the tests covering VAD veto behaviour were all
silently exercising the no-evidence path. Both are fixed. The tier assertion also
turned out to be unsatisfiable rather than merely stale: the retune moved the
complete tier to 1.2, which is what `TEXT_INCOMPLETE_HOLD_TIMEOUT_SECS` already
was, so the two confidence tiers now differ only in the logged reason and the
inverse tier has never been re-swept against its new counterpart.

Baselines may also now be taken in parallel. `--update-baseline` used to refuse
`--jobs > 1`, on the grounds that contention would bake the machine's load into
the latency the gate compares against. Measured on `deepgram-nova/local`,
`--quick`, 3 repeats, it does not: p50 278ms and p95 407ms at `--jobs 6` against
280ms and 453ms serial. `eou→audio` is mostly spent waiting on STT and TTS
sockets rather than competing for CPU, so the sessions overlap their waiting; the
only outlier in either run is a 962ms first-run model load, in the *serial* one.
The refusal bought nothing a busy machine at `--jobs 1` would not also spoil, and
it priced a full baseline at roughly six times what it needs to cost, which is
the kind of price that stops anyone taking one. What makes it sound is unchanged
and unrelated: a baseline records its own concurrency, and latency comparison is
declined across a mismatch.

### The provider's patience was hiding a sentence we could have read

The 1.2s conclusion above was measured against a harness that tore down mid-hold,
which penalises exactly the configurations relying on our holds rather than the
provider's. Re-swept once that was fixed, the aggressive end is nowhere near as bad
as it looked — 0.3 goes from 47% to 77% — and at `--repeat 3` the shipped value
looked barely defensible at all:

| `vad_silence_threshold_secs` | pause merges | dead air p50 | p95 |
|---|---:|---:|---:|
| 0.3 | 77% | 775ms | 2602ms |
| 0.5 | 83% | 765ms | **2388ms** |
| 0.8 | 73% | 962ms | 2924ms |
| 1.2 (ships) | 85% | 1409ms | 2882ms |

Two points for 642ms is a trade anyone would take. It is also not real: 0.8 scoring
below both neighbours is the signature of an underpowered measurement, n=78 makes
two points about 1.5 runs, and at `--repeat 6` on `local` alone the gap reopens to
94% against 83%. **1.2 stays.** Worth stating plainly because the tempting version
of this exercise — sweep, take the winner, ship it — would have shipped a 6% merge
regression on the strength of noise, twice, in opposite directions.

What the tighter sweep bought instead was the *shape* of the disagreement. Nothing
here is a curve; it is two families pulling opposite ways, each pinned at 0% or 100%
across all six repeats:

| scenario | 0.5 | 0.6 | 0.8 | 1.2 |
|---|---:|---:|---:|---:|
| `medical_self_correction` | 0% | 0% | 0% | 100% |
| `banking_correction` | 0% | 0% | 0% | 100% |
| `food_list_pause` | 100% | 100% | 100% | 17% |

The two that collapse are the two the hold-tier sweep could never fix at any value,
and the trace says why — with `holding=False`, no hold is armed at all, so there was
never a tier to blame:

```
commit 'Send it to account 447. Sorry.'   holding=False  → new_turn
commit '448.'                                            → new_turn, 2 turns, FAIL
```

Both EOU signals score that finished and both are right: it *is* a complete
sentence. What makes it incomplete is neither acoustic nor syntactic but discourse
— "Sorry." and "No, wait." announce a retraction, the way a hedge announces a pause.
So `local` now holds on a finished sentence plus a trailing correction marker
regardless of either score (`trailing_correction_marker`, short tier, since the
correction follows within ~300ms by nature and a false positive should not cost 3s).
The sentence-then-marker *shape* is the entire safety margin: a bare "Sorry." is
somebody's whole turn, and "I'm sorry." is an apology.

Measured at `--repeat 6`, at the shipped 1.2, against cells where both scenarios
were marked known failures:

| | `flux/local` | `nova/local` | `elevenlabs/local` |
|---|---:|---:|---:|
| `banking_correction` | **6/6** | 4/6 | 5/6 |
| `medical_self_correction` | 0/6 | **6/6** | 5/6 |

Two markers retired outright and three downgraded to a race the hold either wins or
loses. Both Deepgram cells gate at 100%; ElevenLabs holds its 96.8% baseline, and
the three scenarios that dipped to 2/3 there never trigger the rule once across 9
runs, so they are the provider variance documented above. It reaches only `local` —
`LexicalTurnDetector` is a sibling of `LocalAudioTurnDetector`, not a subclass — so
the improvements visible on `lexical` cells in the same runs are not this change,
which is its own hint that some markers there are stale.

The threshold question is now a different one. ElevenLabs' 1.2s is not buying
merges in general, it is buying *these* merges, by absorbing corrections before any
detector sees them — which is why the defect surfaced on Flux and Nova instead. Each
such case read at the detector is 630ms of dead air per turn that stops needing to
be bought.

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

# the matrix: --stt and --detector are crossed, one scorecard per cell
uv run python benchmarks/voice/cli.py --detector local,lexical,provider --jobs 4

# reproduce one cell of a sweep with the event stream visible — the step between
# "this value scores worse" and knowing why
uv run python benchmarks/voice/cli.py -s coding_double_pause --stt elevenlabs \
    --detector local --stt-param vad_silence_threshold_secs=0.3 --verbose
```

`--stt-param` and `--detector-param` take `KEY=VALUE`. STT keys are checked against
a per-backend allowlist (`harness.SWEEPABLE_STT_KEYS`) because providers ignore
unknown query params in silence: a typo would otherwise sweep one value under four
labels and every number in the table would agree with every other.

Exit code is non-zero when any expectation fails or a regression is gated.

## The matrix

`--stt` and `--detector` take comma-separated lists and are crossed, producing one
scorecard, one baseline entry and one gate per cell, plus a scenario × cell grid at
the end. The grid is the point: a row tells you whether a scenario fails everywhere
(Timbal's problem) or in one column (that provider's problem), which no amount of
staring at a single cell will tell you.

```
                            1    2    3
  banking_digits_pause      ✓    ✓    x
  banking_hesitation        ✓    ✓    ·
  medical_hesitant_pause    x    x    x

    1  deepgram-flux/local            19/19  p50   214ms
    2  deepgram-flux/lexical          19/19  p50   213ms
    3  deepgram-flux/provider         16/16  p50   222ms

  ✓ pass   ✗ FAIL   x known failure   ! XPASS   ~ flaky   · not run
```

`x` and `✗` are deliberately different glyphs: a row of `x` is a documented
limitation, a single `✗` is a regression. `·` means the scenario opted out of that
detector (see *Per-detector expectations*).

`known_failure` and `expect_by_detector` keys are matched most-specific-first:
`"deepgram-nova/lexical"`, then `"deepgram-nova/*"`, then `"lexical"`, then `"*"`.
Scope findings to a cell whenever they came from one — a bare detector key silently
applies a Flux observation to every other backend, which is how `food_long_pause`
came to fail Nova for behaving reasonably. The `"deepgram-nova/*"` rung is for
failures that are the STT's alone and land identically under all four detectors,
like Nova committing a "mm-hmm"; writing those per cell states one fact four times
and hides that it is one fact.

`--jobs N` overlaps runs across the whole queue, cells included. One rule keeps it
from corrupting the thing it accelerates: latency is never compared across differing
concurrency — it is printed with a note instead. Baselines may be taken in parallel
(see the measurement above), so what a parallel run records is gated against later
runs at the same `--jobs`. Pass rates and ghost turns gate regardless.

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
| median `eou→audio` worsens >15% | either timing with fewer than 20 samples |
| median dead air worsens >15% | either timing measured at a different `--jobs` |

Both timings gate through the same `_compare_timing` helper. That is deliberate
rather than tidiness: the latency block used to `return` early on a `--jobs`
mismatch, and a dead-air check bolted on after it would have been skipped
silently on exactly the parallel runs people actually use — the same shape of
blind spot the dead-air metric exists to close.

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

At 38 scenarios a single pass produces well over 40 latency samples, so
`--repeat 1` clears the floor on its own. Repeats still buy flaky detection: any
scenario that passes some repeats and fails others is reported by name as
**FLAKY** — usually a real race rather than a bad test. This is not optional
rigour: `coding_pause` sat in the baseline as a clean pass until `--repeat 3`
caught it splitting once in three under `provider`.

Flaky detection deliberately looks *within* a cell, not across cells. Two detectors
disagreeing is the matrix working as designed; the same detector disagreeing with
itself is a race. It also covers known failures, which the per-cell scorecard
cannot see — those are excluded from `per_scenario` by design, which is how a
marked scenario passing 5 runs in 6 stays invisible until you ask for repeats.

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

Scenarios come in matched sets, because a single data point about turn-taking is
rarely interpretable on its own.

Pause length is swept at 0.6s, 1.2–1.5s and 3.0s (`support_pause_short`, the
`*_pause` family, `food_long_pause`), which brackets each backend's turn window
and separates "merges anything" from "merges what a speaker would". Barge-in is
swept by offset at 150ms, 600ms and 2500ms against the same reply, yielding heard
prefixes from nothing to ~77 characters — which is how you check the truncation
path scales with real playback position — and by shape: one word, twice in a
session, echoing the assistant's own words, and a backchannel that must not
interrupt at all.

Each family also carries its own inverse, so a detector cannot score well by
always guessing the same way. Against the pause merges sit `food_rapid_fire` (two
finished sentences 0.9s apart, must split) and `medical_long_utterance` (12
unbroken seconds, must not). Against the barge-ins sits `food_backchannel`.
Against every commit sits `support_silence_only`, which says an open mic with no
speech produces no turn at all.

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
