# #242 voice integration: we had this, now we have this

Measured 4 Nov 2026 on this branch against `origin/main` (`fd24b412`), same harness, same clips, real providers. Caller audio is **synthetic** OpenAI `gpt-4o-mini-tts` (`coral`), resampled to 16 kHz PCM and paced in real time. STT is Deepgram Flux (`flux-general-multi`, key present). TTS is the default `eleven_v4_turbo`. The text model is `openai/gpt-5.4-mini` unless a row says Groq. Headline cells are 5 reps. Ablations and Groq are 3. No numbers below are filled in by hand from a missing run.

Reproduce:

```bash
uv sync --dev --extra voice
uv run python benchmarks/voice/bench_242.py all --reps 5
```

Raw rows: `benchmarks/voice/242-results/runs.jsonl`. Aggregates: `summary.json`, `summary.csv`.

The caller-facing clock is wall time from the last paced caller frame to the first emitted agent byte. Flux's `TurnMetrics` sum (`speech_end_to_transcript_ms + eou_to_first_audio_ms`) uses the last partial as the speech anchor, and that partial lands near `EndOfTurn`, so it hides the silence the caller already sat through. Both clocks are in the JSON. The table uses the wall clock.

## Before → after

`before` is main, default `VoiceConfig` (Flux, provider turn detector, VAD endpointing off, nothing from #243–#250). `after` is this branch with every opt-in on: backchannels ignored, unfinished hold 1.2 s, floors 0.30/0.20, first flush at 4 words, predict-during-speech, speculative generation with preemptive TTS, pause trim 0.2 s. Tempo stayed off.

Silence to first audio, milliseconds, `gpt-5.4-mini`, Flux.

| Turn | Before p50 | After p50 | Before p90 | After p90 |
| --- | ---: | ---: | ---: | ---: |
| English statement | 1144.1 | 768.0 | 1462.3 | 881.1 |
| Spanish statement | 1766.2 | 1403.1 | 1958.3 | 1547.9 |
| English question | 2104.4 | 1668.4 | 2215.2 | 1718.1 |
| Spanish question | 2461.1 | 1801.0 | 2708.2 | 2131.6 |
| English unfinished ("…for" / "Thursday") | 1657.8 | 1244.2 | 2043.8 | 1337.9 |
| Spanish unfinished ("…para" / "el jueves") | 1633.3 | 793.4 | 1752.8 | 1460.9 |
| Order lookup | 1908.6 | 1476.6 | 2288.6 | 1936.6 |
| 3-turn call, ~7k-token prompt | 940.2 | 343.2 | 1213.7 | 715.0 |
| **All Flux turns** | **1858.3** | **1345.7** | **2801.5** | **1697.5** |

Commit-to-first-audio (`eou_to_first_audio_ms`), all Flux turns: before p50 1180.9 / p90 1982.2, after p50 546.5 / p90 1448.6. A lot of that drop is speculative audio already buffered at confirm, so the clock reads near zero. It is not a faster model.

| Quality | Before | After |
| --- | ---: | ---: |
| False interruptions from backchannels | 10/15 | 0/15 |
| Missed real barge-ins | 0/10 | 0/10 |
| English unfinished split into two turns | 5/5 | 5/5 |
| Spanish unfinished split | 0/5 | 0/5 |
| `end_call` on a turn that was later retracted | 0 | 0 |
| Speculative attempts started / discarded | 0 / 0 | 113 / 24 |
| Wasted tokens on discarded attempts | — | not on the trace |

`mhm` never interrupted on either side: Flux did not commit it. `okay got it` and `vale` interrupted 5/5 on main and 0/5 with `ignore_backchannels`.

## Ablation

One knob off the candidate above, except tempo, which was turned on. Flux, `gpt-5.4-mini`, 3 reps, unless noted.

| Knob | What changed |
| --- | --- |
| `ignore_backchannels` off | `okay got it` and `vale` interrupt again, 3/3 each. `mhm` still does not. This is the whole backchannel win. |
| `hold_unfinished_secs` off | English unfinished still splits 3/3. Spanish still does not. The 1.2 s hold did not merge the English cut on Flux. |
| Floors back to 0.45 / 0.20 | Same English split, 3/3. On Flux the local endpointer is forced off, so these floors do not move the headline. |
| `predict_during_speech` off | Same English split, 3/3. With speculative generation on, Flux eager events are owned by speculation either way. |
| `first_flush_min_words` off | Cache-call silence p50 342.7 vs 343.2 with it on. Median did not move. The comma flush did fire in the logs. |
| Pause trim off | Internal agent silence on the cache call, p90 0.5 s vs 0.3 s with `max_tts_pause_secs=0.2`. p50 0.1 s either way. First audio is a different clock. |
| Tempo 1.05 | Same pause p50 0.1 s, p90 0.2 s. No listener score. Leave it off. |
| Speculative, TTS held until confirm | Only run on the hang-up script. Not enough to separate from preemptive TTS on the latency table. The latency drop on statements lines up with eager start plus preemptive TTS (English statement 1144 → 768). |
| Cache bust (new nonce every turn) | Span TTFT 307.9, 434.5, 643.3 ms with `input_text_tokens` ~12.8k and **no** cached-token field. A warm cached turn on the best profile reported span TTFT 308.1–459.6 ms with 12544 cached + ~200–300 uncached. The bands overlap. This run does not show a TTFT win from the cache. |
| Scribe, default floors vs 0.30 + predict | Statement silence p50 1957.2 vs 2094.0 (n=3). English unfinished merged 0/3 at 0.45 and 2/3 at 0.30. Spanish unfinished split 3/3 both ways. Not a reason to lower the default floor. |

Groq `openai/gpt-oss-20b`, same Flux path, 3 reps, statements + Spanish question + the 3-turn call mixed: silence p50 2288.0 → 2262.3, p90 **3603.2 → 4935.1**. Median flat, tail worse. Do not ship this stack on that model on the strength of this run.

## Trade-offs

Speculation is the latency win and the new cost. 24 of 113 attempts on the best profile were discarded (5 unfinished, 6 barge-in, 13 on the long-prompt call, none on plain statements). Discarded runs did not carry a token usage chunk on the trace, so wasted tokens are **unmeasured**, not free. `end_call` still ran on 4/5 hang-up-then-retract calls on both sides. Flux confirmed "Please hang up the call." as its own turn before "No wait…" arrived (0.6 s gap, default `eot_timeout_ms=2000`). `defer_until_eot` does not help once the turn is confirmed. That count did not get worse, and it did not get better.

The English unfinished phrase still splits 5/5. The inserted 0.8 s gap is two OpenAI renders, not one breath inside a sentence. Smart Turn would score that cut differently. Spanish did not split. `hold_unfinished_secs` is not earning its place on default Flux.

Prompt cache hit rate on turns that reported tokens is about 12544 / 12744 ≈ 98% after the prefix is hot. The before side was already at 12288 cached tokens on its first cache turn, because those calls ran after the after-side had warmed the same prefix at OpenAI. There is no cold number in this file. Span TTFT with the prefix busted sits in the same range as span TTFT with it cached.

`mhm` is not evidence for the backchannel filter. Flux never committed it.

## Recommended production config

For the default Flux path:

```python
voice_config = {
    "ignore_backchannels": True,
    "speculative_generation": {"enabled": True, "preemptive_tts": True},
    "max_tts_pause_secs": 0.2,
}
```

Leave the rest at the defaults: `hold_unfinished_secs` unset, floors 0.45 / 0.20, `first_flush_min_words` unset, `predict_during_speech` false, `tts_tempo` unset. Prompt caching is not a flag. The session already puts the static prefix first and binds a per-call cache key.

Do not copy the speculative block onto `groq/openai/gpt-oss-20b` until the p90 regression is understood.

## Fixes that belong in individual PRs

**#249 prompt caching.** `python/tests/server/test_voice_greeting.py` fails because the greeting note moved out of the `system_prompt` kwarg into `VoiceSession.prompt_for_turn` (dynamic suffix). `TestGreetingMemory` and `TestGreetingPromptNoteAfterBargeIn` spy on `Agent.__call__` and see `None`. The note is applied when the session is current (`session.run` sets that). The barge-in test calls `_run_turn` directly, so it also has to clear `_cancel_turn` (interrupt leaves it set; `_begin_user_turn` clears it on a real next turn) and set `_current_voice_session`. The integration branch updates those tests to read `system_prompt_text` of `_voice_system_prompt`. That change should land in #249 or the greeting tests stay red when it merges.

No other integration failure showed up in the suite. Full `pytest` (voice extra, not integration) was green except those greeting tests before the spy update, and the greeting file is green after it. `python/tests/tools/test_response_collector.py::TestCustomToolCallEvents::test_output_item_done_tracks_usage` was in the first last-failed cache and passed on a rerun.

## Merge conflicts, for whoever rebases second

Merged in PR order onto `origin/main`. Clean merges: #244 into #243, #246 into #245. Everything else conflicted.

**#245 into #243.** `docs/voice/configuration.mdx` server-only list and `docs/voice/turn-taking.mdx` both added a section in the same place. Kept both: backchannels/holds and endpointing floors, union of the server-only fields (`ignore_backchannels`, `hold_unfinished_secs`, `min_delay_secs`, `stop_silence_secs`, `first_flush_min_words`). Python auto-merged and kept both constructor args.

**#247 into the stack.** Same docs pattern, plus `VoiceSession.__init__`, `VoiceConfig`, and `build_voice_session` kwargs. Kept endpointing/flush fields and `predict_during_speech` / `eager_eot_threshold`. `predict_during_speech` stays in `CLIENT_SETTABLE_VOICE_FIELDS` and in the client-override sentence.

**#248 into the stack.** Client-override sentence and the `session_kwargs` block. Union: `predict_during_speech` stays client-settable, and so do `max_tts_pause_secs`, `tts_quiet_dbfs`, `tts_tempo`. Both kwargs blocks are forwarded.

**#249 into the stack.** `python/timbal/voice/__init__.py` import block. Kept `SpeechEouPredictor` and `voice_prompt`.

**#250 into #247 (the one to rebase carefully).**

- `python/timbal/voice/predictive_eou.py` was add/add. #247 is the real module. #250's file was only `flux_eager_threshold`. The function body is the same. One copy lives in #247's module. Docstring now says both features call it. There is no 0.3 floor. Omitted value is `min(0.5, eot_threshold - 0.1)`. An explicit `eager_eot_threshold` is lowered the same way when it is not already strictly below `eot_threshold`.
- `build_voice_session` sets that value **once** when `predict_during_speech` is on (native EOU, then `isinstance` Flux) **or** `speculative_generation.enabled` on `deepgram-flux`. Not twice.
- `docs/voice/turn-taking.mdx` and `docs/voice/configuration.mdx`: one section, "Flux eager threshold". `predict_during_speech` stays in the client-settable list. `speculative_generation` stays server-only. The two old paragraphs that each described the threshold are gone.
- `VadEndpointer.push`: on resume, call `predictor.on_speech_resumed()` **and** `on_speech_resume` when a pending endpoint was cancelled. Do not drop either.
- `VadEndpointer._run_endpoint`: call `on_score` only when `driver == "audio"`. Do **not** recompute `remaining = delay - elapsed` after the predictive branch. #250's patch did that and would overwrite the short predictive wait.
- `TranscriptEvent.text` defaults to `""` so an empty `TurnResumed` is delivered. Deepgram yields `resumed` even with no text.
- `_process_stt_events`: on a partial, both `note_partial` and the speculative discard-if-text-diverged check run. Barge-in keeps #246's interruption suppression. Flux `eager`/`resumed`: if speculative generation is on it owns those events (hold the reply); otherwise `predict_during_speech` commits via `_handle_provider_prediction`. Confirmed transcripts still go through `_handle_provider_commit`, which calls `_handle_committed`, so a matching speculative attempt still confirms.
- `python/timbal/voice/__init__.py` exports both `SpeechEouPredictor` and `SpeculativeGeneration`.

If #250 merges first, #247 must not replace `predictive_eou.py` with the stub, must not drop `predict_during_speech` from the client list, and must not let a second `eager_eot_threshold` assignment overwrite the first.

## Wave 2 — recommended config, the rest of the calls

Same branch against the same `origin/main`. `before` is main with default `VoiceConfig`. `after` is **only** the recommended block from the section above (`ignore_backchannels`, speculative generation with preemptive TTS, `max_tts_pause_secs=0.2`). Not the everything-on candidate. Each run gets its own `Call id <uuid>` prefix on the ~7k-token prompt, and even reps run before-then-after while odd reps run after-then-before, so the two sides do not warm each other.

Flux × `openai/gpt-5.4-mini` is n=10 (51 scenarios × 2 sides = 1020). ElevenLabs Scribe on the same 51 scenarios, and Groq `openai/gpt-oss-120b` on 8 of them, are n=5. A full n=10 on Scribe as well is another ~510 realtime calls on top of a matrix that already took 5.6 hours; n=5 is the floor the request allowed, and it is what those two cells are. 1610 rows in `benchmarks/voice/242-results/runs-wave2.jsonl`. Reproduce with `uv run python benchmarks/voice/bench_242.py wave2 --flux-reps 10 --other-reps 5` (resumes rows already in that file). Two runs timed out and are excluded from the percentiles below: `long_10_en` after rep 7, and `en_after_es_greet` after Flux rep 0.

Caller audio is still synthetic `gpt-4o-mini-tts`. Telephony rows go through `benchmarks/voice/degrade.py`: G.711 μ-law, 300–3400 Hz, 8 kHz, then back up. Street is pink noise at 8 dB SNR, office pink at 18 dB, TV white at 12 dB. The clock in the tables is the same wall clock as wave 1 (last paced caller frame → first emitted agent byte), in milliseconds.

The first cut of `en_after_es_greet` treated the Spanish greeting as the agent's answer and closed 0.8 s after the caller. Those 30 rows were deleted and rerun after the harness started waiting for playback to finish and for a reply that arrives after the greeting. Everything else is from the original pass.

### 1. Telephony

Flux, `gpt-5.4-mini`, n=10. Quality column is replies whose language does not match the caller.

| Scenario | Before p50 | After p50 | Before p90 | After p90 | Before | After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **all five** | 1465 | 1086 | 2181 | 1805 | lang 11/50 | lang 14/50 |
| phone, English | 1038 | 711 | 1397 | 1023 | 6/10 | 7/10 |
| phone, Spanish | 1587 | 1074 | 1773 | 1308 | 0/10 | 0/10 |
| street 8 dB, English | 1244 | 1768 | 1318 | 1949 | 3/10 | 2/10 |
| office 18 dB, Spanish | 1563 | 968 | 2095 | 1147 | 0/10 | 0/10 |
| TV 12 dB, English | 2148 | 1536 | 3126 | 1761 | 2/10 | 5/10 |

Clean phone and office get faster. Street gets slower: after started 20 speculative attempts and discarded 10, and the silence samples spread from a tight 1073–1338 ms to 953–2205 ms. The language misses are English audio, committed in English, answered in Spanish. Spanish telephony was 0/10 both sides. Detail under "Language" below.

Scribe, n=5, same five: before p50 2104 / p90 2762, after p50 1910 / p90 2963. Slower than Flux, and the p90 does not improve. Language misses 9/25 → 5/25, same English-answered-in-Spanish shape.

### 2. Hesitations

Gap 1.5–1.8 s inside the sentence. Quality is premature cut-off: agent audio before the continuation was fed.

| Scenario | Before p50 | After p50 | Before p90 | After p90 | Premature |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Flux, all three** | 1892 | 2030 | 2704 | 3082 | 30/30 → 30/30 |
| "um", English | 1780 | 2372 | 2114 | 2906 | 10/10 → 10/10 |
| "eh", Spanish | 2584 | 1935 | 3141 | 2466 | 10/10 → 10/10 |
| "o sea", Spanish | 1680 | 2170 | 2267 | 3202 | 10/10 → 10/10 |

The recommended config does not hold the turn. Flux ends it during the pause on both sides, then the continuation is a second turn. After is slower on "um" and "o sea" because that second turn follows a discarded speculation. Scribe holds the Spanish ones (0/5 premature both sides) and still splits "um" 5/5. Scribe overall p50 2773 → 2391.

### 3. Dictation

Pauses 0.45–0.8 s between groups. Quality is premature cut-off, then whether any reply contains the read-back needle (`4821`, `655` / "seis cinco cinco", "alex" and "example", "jones" / "jota").

| Scenario | Before p50 | After p50 | Before p90 | After p90 | Before | After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Flux, all four** | 1275 | 1190 | 2318 | 2180 | prem 40/40, read-back 34/40 | prem 40/40, read-back 31/40 |
| order id, English | 190 | 281 | 2330 | 2604 | 10/10, 6/10 | 10/10, 4/10 |
| phone, Spanish | 1107 | 976 | 2341 | 1688 | 10/10, 10/10 | 10/10, 10/10 |
| email, English | 1398 | 1198 | 1859 | 2076 | 10/10, 8/10 | 10/10, 7/10 |
| spelled name, Spanish | 1275 | 1228 | 1658 | 2240 | 10/10, 10/10 | 10/10, 10/10 |

Flux splits every dictation into one turn per group, both sides. A later reply sometimes contains the digits (order id read-back fails 6/10 → 4/10), the phone number and the spelled name never come back whole. One after Flux order-id run committed "four eight." / "two one." / "nine zero." as three turns and the second reply asked the caller to repeat "4821", which counts as a read-back hit and is still a broken dictation. Scribe merged the English order id 5/5 before and only 2/5 after (premature 0/5 → 3/5). That is a regression on the production STT.

### 4. Long monologue

Four sentences, 0.55 s gaps, about 20 s of speech.

| Scenario | Before p50 | After p50 | Before p90 | After p90 | Premature |
| --- | ---: | ---: | ---: | ---: | ---: |
| English, Flux | 1420 | 805 | 2026 | 1670 | 10/10 → 10/10 |
| Spanish, Flux | 2493 | 2310 | 2638 | 2550 | 0/10 → 0/10 |

Same split as wave 1's unfinished phrases: English breaks, Spanish does not, and the recommended config does not change it. The English p50 drop is the early reply, not a held monologue. Scribe held both languages, 0/5 both sides (p50 2539 → 2376).

### 5. Fast and slow

`ash` at 1.2× English, `sage` at 0.85× Spanish.

| Scenario | Before p50 | After p50 | Before p90 | After p90 | Lang misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast English | 1555 | 1045 | 1755 | 1288 | 9/10 → 10/10 |
| slow Spanish | 1099 | 754 | 1356 | 932 | 0/10 → 0/10 |

Latency moves the right way. The fast English caller was transcribed in English and answered in Spanish on 19 of 20 Flux runs. Scribe fast-English language misses 2/5 → 4/5, and slow-Spanish p50 got worse (1231 → 1874).

### 6. Backchannels

Spoken over the agent after 400 ms of playback. Quality is a false cut (`SessionInterrupted` before the scenario closed).

| Utterance | Before Flux | After Flux | Before Scribe | After Scribe |
| --- | ---: | ---: | ---: | ---: |
| sí | 10/10 | 0/10 | 5/5 | 0/5 |
| ajá | 0/10 | 0/10 | 5/5 | 0/5 |
| claro | 10/10 | 0/10 | 5/5 | 0/5 |
| vale vale | 2/10 | 0/10 | 5/5 | 0/5 |
| uh-huh | 5/10 | 0/10 | 5/5 | 0/5 |
| right | 10/10 | 0/10 | 5/5 | 0/5 |
| yeah | 10/10 | 0/10 | 5/5 | 0/5 |
| mm | 9/10 | 10/10 | 4/5 | 5/5 |
| **all** | **56/80** | **10/80** | **39/40** | **5/40** |

Silence p50 on Flux for the whole group: 2632 → 1657, p90 2904 → 1842. That drop is the agent not restarting after the cut. `ajá` never interrupted on Flux to begin with (it was not committed as its own turn). Every remaining false cut is `mm`. Flux commits it as `M.`, and `_BACKCHANNEL_WORDS` has `mhmm` / `mm hmm` but not `mm` or `m`.

### 7. Barge-in

Miss rate is an interrupt that never fired. Stop-audio is interrupt time minus the moment the barge clip started.

| | Misses | Stop p50 | Stop p90 |
| --- | ---: | ---: | ---: |
| Flux before | 0/80 | 2319 | 3307 |
| Flux after | 0/80 | 644 | 868 |
| Scribe before | 0/40 | 2499 | 3452 |
| Scribe after | 0/40 | 1448 | 1724 |

Short ones included, Flux n=10: "no" stop p50 1674 → 657, "stop" 2005 → 644, "wait" 1944 → 752, "espera" 2489 → 780, start / mid / end of the reply all 0 misses. Nothing in this set got worse. Overlap (caller starts as the agent starts) is the same miss rate, 0/20 Flux and 0/10 Scribe, both sides. Flux overlap stop p50 1343 → 1104. Scribe overlap stop stays about 2.2 s.

### 8. Self-corrections and hang-up retractions

Correction gap is 0.4 s. Quality is premature cut-off, and whether a reply contains "Wednesday" / "miércoles".

| Scenario | Before p50 | After p50 | Before p90 | After p90 | Premature | Read-back miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English, Flux | 1074 | 380 | 1462 | 777 | 10/10 → 10/10 | 7/10 → 8/10 |
| Spanish, Flux | 1353 | 582 | 1565 | 770 | 10/10 → 10/10 | 5/10 → 1/10 |

The p50 collapse is the premature reply getting faster under speculation. The turn still ends before "no, Wednesday". Spanish read-back is the one quality win in this group. Scribe held the Spanish correction (premature 0/5 both sides) and split the English one 5/5.

Hang-up then retract. The caller says "please hang up" and takes it back after the gap. Quality is `end_call` invocations. Fewer is better. Every one of these is a retraction, so an `end_call` is a wrong tool execution. `destructive_on_retracted` in the row stays 0: that flag requires an empty replacement commit or a discarded speculation on the same row, and a turn Flux already confirmed does not look like either.

| Gap | Flux before | Flux after | Scribe before | Scribe after |
| --- | ---: | ---: | ---: | ---: |
| English 0.3 s | 10/10 | 6/10 | 0/5 | 0/5 |
| English 0.6 s | 8/10 | 6/10 | 4/5 | 1/5 |
| English 1.0 s | 10/10 | 7/10 | 5/5 | 2/5 |
| Spanish 0.3 s | 0/10 | 0/10 | 0/5 | 0/5 |
| Spanish 0.6 s | 10/10 | 9/10 | 0/5 | 0/5 |
| Spanish 1.0 s | 10/10 | 8/10 | 0/5 | 0/5 |

Spanish 0.3 s is one Flux commit ("cuelga la llamada. No, espera…") and the model does not hang up. At 0.6 s and 1.0 s Flux has already closed the turn and `end_call` runs before the retraction is heard. After hangs up a bit less often. It does not fix it. Scribe keeps the Spanish retraction inside one turn out to 1.0 s (0 hang-ups) and is the better English result too, especially after.

### 9. Language

Three different things were getting called "the agent answered English in Spanish."

**Ordinary English turns.** `VoiceConfig.language` is `en`. Flux's commit is English. `gpt-5.4-mini` still replies in Spanish on 90/280 of those turns before and 101/279 after. Fast English is 9/10 → 10/10. The warranty tool is 9/10 and 10/10. The system prompt says "Reply in the caller's language" and never pins English. Groq `gpt-oss-120b` on the clean English phone line was 0/5 both sides, same Flux transcript. This is the model, not STT and not the recommended config.

**Code-switch.** "Quiero el refund del order cuatro ocho dos uno, please." Expect Spanish. Flux language misses 0/10 both sides. Read-back of the order id missed 4/10 → 5/10. Silence p50 3209 → 2123. Scribe language misses 1/5 → 0/5, but silence p50 3694 → 4929. After is slower there.

**English caller after a Spanish greeting.** The harness sets `VoiceConfig.language` from the scenario language, and this scenario is tagged `es` because the greeting is Spanish, so STT is hinted Spanish. That is the config leg.

Flux committed only the tail, `in English, please.`, on 19/20 runs (the 20th after-run timed out with no commit). The model then answered in English, 19/19. It did not answer in Spanish. The full sentence "I need the return window in English please" was not in the commit. Scribe committed the full sentence, often with a hallucinated leading `Sí.` (language hint `es`), and answered in English 4/5 before and 4/5 after. The two misses are real model slips on a correct English transcript: one before reply is entirely Spanish, one after reply starts "La ventana de devolución…" and finishes in English.

So: the systematic "English in, Spanish out" on normal turns is `gpt-5.4-mini` ignoring an English transcript. The greeting case is not that bug. Under `language=es`, Flux drops the front of the English sentence, and Scribe invents a `Sí`. Neither side's recommended config causes it. Pinning the prompt ("if the transcript is English, answer in English") would be the fix for the model half. Not hinting STT as `es` on a call that may switch would be the fix for the truncation and the phantom `Sí`.

### 10. Long call, cold cache

Ten turns, Flux, unique prefix per run. `long_10_en` after rep 7 timed out, so after latency is n=9.

| | Silence p50 | Silence p90 | Cold hit | Cold TTFT p50 | Warm hit | Warm TTFT p50 | Warm TTFT p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 608 | 1178 | 0.0 (n=10, ~12700 uncached) | 757 | 0.968 (n=91) | 730 | 930 |
| after | 583 | 1492 | 0.0 (n=9, ~12790 uncached) | 1324 | 0.958 (n=82) | 532 | 1528 |

Turn 1 is actually cold this time. Wave 1's "before was already warm" result was the shared prefix. Hit rate on later turns is ~96% on OpenAI either side. Warm TTFT median improves (730 → 532). The tails do not: silence p90 1178 → 1492, warm TTFT p90 930 → 1528, and cold TTFT is worse after because turn 1's speculation is discarded and the confirmed call pays for it. Per-turn silence medians bounce between ~100 ms and ~1400 ms with no upward drift across the 10 turns.

Groq on the same call, n=5: silence p50 1186 → 912, p90 1733 → 1404. Cold hit 0 both sides (TTFT p50 1648 → 1173). Warm hit only 0.404 → 0.493. Groq is not caching this prefix the way OpenAI is.

### 11. Tools

Dead air is tool-start → next emitted audio. `filler_spoken` was 0 on every Flux run, both sides. An `agent_status` event (`Calling slow_check…`) is emitted at tool start. It is not audio. The 4 s `slow_check` is uncovered unless the model already spoke.

| Scenario | Before p50 | After p50 | Before p90 | After p90 | Dead-air p50 | What happened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| two tools | 3470 | 2301 | 3973 | 3356 | 1330 → 873 | both tools called 9/10 → 5/10 |
| slow, 4 s | 3157 | 4197 | 3796 | 6571 | 970 → 2796 | after has a cluster at 4.8–5.7 s |
| error | 3444 | 2553 | 3900 | 2764 | 983 → 928 | tool raised, model apologised, 10/10 both |

`tools_two` asking for a lookup and then the delivery window: before called both tools 9/10, after 5/10 (the other 5 called only `lookup_order`). That is a real regression. `tools_slow` dead air is the other one: before sits around 1 s, after splits into a fast cluster (~0.4–0.8 s) and a cluster that covers the whole tool. Scribe shows the same slow-tool dead-air jump (980 → 2883 ms). No filler audio on either STT.

### 12. Silence after a question

8 s of caller silence after the agent answers. `user_idle` is unset, which is the default, so the session does not re-prompt. Phantom turns (new reply or commit during the tail): 0 on Flux (20/20) and 0 on Scribe (10/10), both sides. Nobody got a reprompt because nothing is configured to speak one. The 8 s tail is exactly `UserIdleConfig.timeout_secs`'s default, so this does not measure the prompt text. It measures that the default leaves the line quiet.

Silence-to-first-audio for the answer itself, Flux: English 1210 → 1044 (p90 1324 → 1294), Spanish 1286 → 2303 (p90 1334 → 2712). The Spanish one is worse after. Scribe English 1810 → 1668, Spanish 3172 → 3319.

### 13. Overlap

Covered with barge-in. Miss rate 0 everywhere. Flux stop p50 1343 → 1104. Scribe does not get the fast stop (2209 → 2144).

## Scribe vs Flux

Scribe is the production STT and it is slower on the clean turns (telephony p50 ~2.1 s vs Flux ~1.5 s before, and after only shaves it to ~1.9 s). It is the one that holds a turn. Spanish hesitations, both monologues, the Spanish self-correction, and Spanish hang-up retractions out to 1.0 s stay one turn on Scribe and split on Flux. The recommended config undoes one of those: the English order-id dictation was premature 0/5 on Scribe before and 3/5 after. Backchannel suppression works on Scribe the same way it works on Flux, including the `mm` hole (5/5 after). Barge-in is caught on both; the stop-audio win is smaller on Scribe (2.5 s → 1.4 s, vs Flux 2.3 s → 0.64 s).

## Groq vs OpenAI

Subset, Flux, n=5: `tel_phone_en`, `hesit_um_en`, `dict_order_en`, `bc_yeah`, `barge_mid_en`, `lang_en_control`, `long_10_en`, `tools_slow`.

| Scenario | Before p50 | After p50 | Before p90 | After p90 |
| --- | ---: | ---: | ---: | ---: |
| phone, English | 1733 | 1462 | 1944 | 2881 |
| "um" | 2099 | 2035 | 2679 | 2238 |
| order id | 501 | 182 | 1355 | 2520 |
| "yeah" | 2925 | 2152 | 3337 | 2598 |
| barge mid | 2866 | 1666 | 3712 | 1856 |
| English control | 876 | 115 | 1533 | 426 |
| 10-turn | 1186 | 912 | 1733 | 1404 |
| slow tool | 3982 | 3384 | 6775 | 3702 |

Medians move the same direction as mini, and `bc_yeah` false cuts go 5/5 → 0/5. The phone p90 goes the wrong way (1944 → 2881), and the order-id p90 does too (1355 → 2520). Language misses on the phone line: 0/5 both sides, which mini does not manage. Warm cache hit on the long call is ~40–49%, not the 96% OpenAI gets. Wave 1's warning was `gpt-oss-20b` p90 3603 → 4935. This is a different model and a short subset. The tail is still the reason not to copy the speculative block onto Groq blindly. `tools_slow` dead-air p50 1446 → 1340, which is the one place Groq did not pick up mini's dead-air regression.

## Where after is worse

Plainly:

- Street noise, Flux, silence p50 1244 → 1768. Half the speculative attempts were discarded.
- "um" and "o sea" hesitations are slower, and still cut 10/10.
- Scribe English order-id dictation starts getting cut (0/5 → 3/5 premature).
- `tools_two` calls both tools 9/10 → 5/10 on Flux.
- `tools_slow` dead-air p50 970 → 2796 on Flux, 980 → 2883 on Scribe. No filler audio.
- Spanish silence-after-question p50 1286 → 2303.
- Long-call p90 1178 → 1492, cold TTFT 757 → 1324, one after run timed out.
- Code-switch on Scribe p50 3694 → 4929.
- Groq phone p90 1944 → 2881.
- `mm` is not fixed. 9/10 → 10/10 false cuts on Flux.
- Fast English answered in Spanish 9/10 → 10/10. Both sides do it; after does not help.
- English hang-up retractions still fire `end_call` most of the time. The count dropped (48/60 → 36/60 on Flux across the six gaps) and that is not a fix.

## Revised recommended config

Keep it for Flux + `gpt-5.4-mini`:

```python
voice_config = {
    "ignore_backchannels": True,
    "speculative_generation": {"enabled": True, "preemptive_tts": True},
    "max_tts_pause_secs": 0.2,
}
```

That is the backchannel result (56/80 → 10/80, everything except `mm`) and the barge stop-audio result (2.3 s → 0.64 s) and the clean-phone latency result (1038 → 711). It is not a turn-holding config. `hold_unfinished_secs` was left unset on purpose, and this run says the pauses in dictation, hesitation, and self-correction still end the turn. Turning it on was already tested in wave 1 and did not merge the English unfinished cut on Flux. Scribe is what holds those turns, and speculation made Scribe's order-id dictation worse, so do not assume the block is free on the production STT.

Do not expect filler audio during a multi-second tool. Nothing in this config speaks into that gap. `filler` is a separate switch and it was left at whatever the server default already is; `filler_spoken` was 0.

Leave Groq off this block until someone looks at the phone p90.

## Bugs, by owning PR

Not fixed here.

**#243 backchannels.** `mm` is outside `_BACKCHANNEL_WORDS`. Flux transcribes it as `M.` and it cuts the agent 10/10 with the filter on (9/10 with it off). Scribe 5/5 with the filter on. `sí`, `claro`, `vale vale`, `uh-huh`, `right`, `yeah` drop to 0. `ajá` was already 0 on Flux. The word list is the hole.

**#244 latency suite.** No product bug. The wall clock reads ~50 ms p50 on `lang_en_control` before because the first audio byte often lands before the feeder arms the timer. `eou_to_first_audio_ms` is in the row for that case. Both clocks are in the JSON.

**#245 floors and first flush.** Not in this config. English monologue and English unfinished-style pauses still split 10/10 on Flux, both sides. Spanish monologue splits 0/10. Same result as wave 1. The floors are not what moves Flux.

**#246 interruption modes.** Barge miss stayed 0/80 including "no", "stop", "wait", "espera". No new miss. The faster stop is real and was not isolated to this PR.

**#247 predict during speech.** Left off. Nothing new to attribute.

**#248 pause trim.** On, at 0.2 s, and not separated from speculation in this wave. No listener score, same as wave 1.

**#249 prompt cache.** Cold is real when the prefix is unique: hit rate 0 on turn 1, ~96% after that on OpenAI, ~40–49% on Groq. After's cold TTFT on the long call is worse (757 → 1324), not better. One `long_10_en` after run timed out waiting for a reply.

**#250 speculative generation.** 1026 attempts started, 213 discarded, on Flux × mini × after. Discarded runs still do not carry output tokens on the trace (3 rows recorded 37632 cached input tokens and 0 output). The regressions that show up only after, and that line up with a discarded attempt, are street-noise latency, `tools_slow` dead air, and `tools_two` dropping the second tool (9/10 → 5/10). Scribe's order-id dictation going from 0/5 premature to 3/5 is the production-STT version of the same early commit. `end_call` on a retraction got somewhat better and is still wrong at 0.6 s and 1.0 s, because Flux confirms the hang-up turn before the retraction arrives. `defer_until_eot` cannot help a turn that has already ended.

## Wave 3 — borrows against the wave-2 failures

Same clips, same Flux × `gpt-5.4-mini`, n=10. Both sides run on this branch. `before` is the wave-2 ship block (`ignore_backchannels`, speculative generation with preemptive TTS, `max_tts_pause_secs=0.2`). `after` is that block plus `hangup_grace_secs=1.2`, `entity_patience=True` (default 2.5 s), and `filler={"enabled": true, "soft_timeout_secs": 0.8, "message": "One moment."}`. Tool hold is not a separate knob: with speculation on, #255 holds every tool until the turn is confirmed, so it is on for both sides. Latest #243 (`mm` / `M.`) is on the branch for both sides too. 280 rows in `benchmarks/voice/242-results/runs-wave3.jsonl`. Four Deepgram 503s are dropped from the percentiles (`dict_phone_es` before, `dict_name_es` both, `dict_order_en` after). Reproduce with `uv run python benchmarks/voice/bench_242.py wave3 --flux-reps 10`.

### 1. Slow tool

We had ~5 s of dead air and `filler_spoken` 0. When the model actually calls `slow_check` (4 s), that is now a spoken line.

| | Runs | Called `slow_check` | `filler_spoken` | Dead-air p50 | Dead-air p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 10 | 4 | 0 | 3156 | 5244 |
| after | 10 | 2 | 4 | 785 | 941 |
| before, `slow_check` only | 4 | 4 | 0 | 5007 | 5440 |
| after, `slow_check` only | 2 | 2 | 2 | 762 | 780 |

The overall p50 mixes in instant `lookup_order` calls. The model calls `slow_check` on 4/10 before and 2/10 after. On those calls the filler is 0/4 → 2/2 and the gap falls from about 5 s to about 0.8 s, which is the soft-timeout window. `filler_spoken` is 4/10 after and 0/10 before. Speculative starts/discards: 10/0 both sides.

### 2. Two tools

We had both tools on 9/10 before the ship block and 5/10 after it. Tool hold is on both sides here. Discards are 0.

| | Both tools | Dead-air p50 | Silence p50 |
| --- | ---: | ---: | ---: |
| before | 7/10 | 1244 | 2935 |
| after | 6/10 | 902 | 2171 |

Not fixed. The misses are the model answering "arrives Thursday" from the lookup string and never calling `delivery_window`. Nothing in the trace shows a second tool started and then discarded.

### 3. Spanish hang-up, then retract

Gap after "Cuelga la llamada", before "No, espera…". `end_call` is the tool running. `hung_up` is the session actually closing before the harness does.

| Gap | Tool before | Tool after | Session closed |
| --- | ---: | ---: | ---: |
| 0.3 s | 0/10 | 0/10 | 0 / 0 |
| 0.6 s | 8/10 | 10/10 | 0 / 0 |
| 1.0 s | 10/10 | 9/10 | 0 / 0 |

Not fixed. The tool still runs at 0.6 s and 1.0 s. Grace does not stop it, and 0.6 s got worse (8/10 → 10/10). The line stayed up on both sides: a later commit cancels teardown even when `hangup_grace_secs` is unset, which is why `hung_up` is 0 on the ship block too. The sample replies are "Hasta luego" / "Adiós, cuelgo" and then "Claro, dime en qué te ayudo." The destructive call still happened.

### 4. Dictation

`entity_patience` holds digits, single letters, emails, and mixed codes. These clips are spoken words: "seis cinco cinco", "four eight" / "two one", "jota" / "o" / "ene". Flux commits them as words. Patience never sees an open entity.

| Scenario | Premature | Read-back miss |
| --- | ---: | ---: |
| phone, Spanish | 9/9 → 10/10 | 9/9 → 10/10 |
| spelled name | 9/9 → 9/9 | 9/9 → 9/9 |
| order id | 10/10 → 9/9 | 4/10 → 6/9 |

Not fixed. Same splits as wave 2. One before order-id commit is still "four eight." / "two one." / "nine zero."

### 5. Ship-config sanity

`mm` is fixed on this branch, both sides. Wave 2's after side cut on `mm` 10/10. False cuts here:

| | sí | yeah | mm |
| --- | ---: | ---: | ---: |
| before | 0/10 | 0/10 | 0/10 |
| after | 0/10 | 0/10 | 0/10 |

Clean English phone, silence p50 926 → 827, p90 1204 → 1121. Wave 2's ship cell was 711. This run is slower and still in the same band. Borrows do not make it worse.

Barge misses 0/10. Stop-audio p50: mid-reply 425 → 460, "stop" 683 → 629. The ship win (wave 2 after, about 0.6 s) is intact.

## What to ship after wave 3

Keep the wave-2 block. Add the spoken soft timeout when a tool can run for seconds:

```python
voice_config = {
    "ignore_backchannels": True,
    "speculative_generation": {"enabled": True, "preemptive_tts": True},
    "max_tts_pause_secs": 0.2,
    "filler": {"enabled": True, "soft_timeout_secs": 0.8, "message": "One moment."},
}
```

Leave `hangup_grace_secs` unset. It did not stop `end_call` at 0.6 s or 1.0 s. Leave `entity_patience` unset for these dictation scripts. It does not match spoken number words or Spanish letter names. Tool hold stays on automatically with speculative generation. It did not bring the second tool back to 9/10.

## Wave 4 — the three gaps

Same Flux × `gpt-5.4-mini`. `before` is the wave-2 ship block plus the soft timeout (`soft_timeout_secs=0.8`, message "One moment."). `after` adds `hangup_grace_secs=1.2` and `entity_patience=True`. The lookup tool no longer returns the arrival day, on both sides. Hang-up tools await `commit_hangup()` and log `end_call` only when that returns true. n=10 except `tools_slow` n=5 and `dict_order_en` before n=9 (one Deepgram 503). 150 rows in `benchmarks/voice/242-results/runs-wave4.jsonl`.

### Hang-up then retract

We had the tool run at 0.6 s and 1.0 s. `commit_hangup` waits out the goodbye and the quiet window, and a new caller utterance cancels the wait before `end_call` is requested. LiveKit's EndCallTool waits on the speech handle the same way; Pipecat does not commit an irreversible tool on an early turn.

| Gap | Tool before | Tool after |
| --- | ---: | ---: |
| 0.3 s | 0/10 | 0/10 |
| 0.6 s | 10/10 | 0/10 |
| 1.0 s | 10/10 | 0/10 |

The session did not close on either side. The difference is the tool. With grace on, the handler is still inside the wait when the retraction commits, so the interruption cancels it and the log never happens.

### Spoken-word dictation

Number words in English and Spanish, letter names, and a trailing single letter after other words now count as an open entity. Flux often writes "jota" as "J". A one-word "a" or "o" still does not hold.

| Scenario | Premature | Read-back miss |
| --- | ---: | ---: |
| phone, Spanish | 10/10 → 0/10 | 10/10 → 10/10 |
| spelled name | 10/10 → 0/10 | 10/10 → 10/10 |
| order id | 9/9 → 0/10 | 6/9 → 8/10 |

The turn holds. One after phone commit is "seis cinco cinco cero uno dos tres cuatro." One after name commit is "me ha apellido J. O. N A A S A." The agent still does not read that string back as the number or the name. That miss is the reply, not the split.

### Two tools

Wave 3 was 7/10 and 6/10, with discards at 0. The lookup result said "arrives Thursday", so the model answered the window from that string. The result is now ship status only, and the prompt says to call each tool that owns a fact. Both sides of this run use that schema.

| | Both tools |
| --- | ---: |
| before | 10/10 |
| after | 10/10 |

Not a config knob. The tool return was the cause.

### Soft timeout still speaks

`tools_slow`, n=5. `filler_spoken` 5/5 before and 1/5 after. The one after run that called `slow_check` did speak (dead air 977 ms). The others returned without the slow tool. The filler path is unchanged.

## What to ship

```python
voice_config = {
    "ignore_backchannels": True,
    "speculative_generation": {"enabled": True, "preemptive_tts": True},
    "max_tts_pause_secs": 0.2,
    "filler": {"enabled": True, "soft_timeout_secs": 0.8, "message": "One moment."},
    "hangup_grace_secs": 1.2,
    "entity_patience": True,
}
```

The hang-up tool has to await `session.commit_hangup()` and treat a false result as "do not hang up". `session.end_call()` alone still only delays teardown after the tool has already run. No new dependencies.
