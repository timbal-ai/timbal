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
