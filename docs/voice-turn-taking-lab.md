# Turn-taking lab: first-PR baseline for #228

This lab supports the first PR for [#228](https://github.com/timbal-ai/timbal/issues/228):
state evidence and opt-in safe resumed-request merging on `ProviderTurnDetector`.
It does **not** close the issue. HOLD for unfinished speech and production
acknowledgment/interruption policy remain follow-up work.

There are two separate entry points:

- `benchmarks/voice/turn_taking_replay.py` replays synthetic CA/ES/EN transcripts
  against the real detector, with explicit state snapshots. It is deterministic,
  requires no API keys, and blocks Python socket connections and DNS.
- `examples/catalan_agent.py` is a Catalan microphone demo with a simulated bicycle
  workshop availability tool. Its STT, LLM, and TTS use live, billable services.
  The tool itself makes no network calls and cannot reserve an appointment.

## Offline replay

From the repository root, with the base Timbal dependencies already installed:

```bash
PYTHONPATH=python python benchmarks/voice/turn_taking_replay.py --selfcheck
PYTHONPATH=python python benchmarks/voice/turn_taking_replay.py
PYTHONPATH=python python benchmarks/voice/turn_taking_replay.py --json > /tmp/turn-taking-baseline.json
PYTHONPATH=python python benchmarks/voice/turn_taking_replay.py --merge-resumed-secs 2 --json > /tmp/turn-taking-enabled.json
```

Use your project's Python interpreter. For an already provisioned uv environment,
replace `python` with `uv run --offline --no-sync python`; this command will not
install dependencies. Neither the replay nor selfcheck downloads weights, starts
STT/TTS/LLM clients, reads `.env`, imports the demo agent, or uses audio fixtures.
The voice extra is not needed for replay. Missing base dependencies must be
installed separately before running it.

The default report exits zero when the replay executed successfully, even if
quality expectations fail. `--strict` exits 1 for any mismatch; invalid fixtures,
invalid merge windows, and empty selections exit 2. For example:

```bash
# Expected to fail: exposes baseline quality gaps, not a passing product test.
PYTHONPATH=python python benchmarks/voice/turn_taking_replay.py --strict

# First-PR scope: safe resumed requests plus guards, with merging enabled.
PYTHONPATH=python python benchmarks/voice/turn_taking_replay.py \
  --merge-resumed-secs 2 --category resumed_request --category merge_guard --strict

# Repeatable language-specific inspection (language/category flags can repeat).
PYTHONPATH=python python benchmarks/voice/turn_taking_replay.py --language ca --json
```

`--selfcheck` runs nine test methods covering multilingual coverage, malformed
fixtures (including wrong types, IDs, time order, unknown state keys and missing
expected text), deterministic replay, case-order isolation, fixture immutability,
network blocking, result accounting, filtering, exact merged-text scoring,
HOLD-timeout scoring, default-off behavior and enabled merge guards. A successful
selfcheck is a harness/runtime regression check; it does not mean the full corpus
meets its desired behavior.

## What the scenarios mean

`turn_taking_cases.json` separates inputs (`text`, `state`, logical `at_ms`) from
expectations (`action`, committed `text`, optional `hold_timeout_secs`). Labels,
language and rationale are never passed to the detector. The runner adds no
acknowledgment word list and loads no acoustic classifier.

CA, ES and EN each include a listener acknowledgment, a longer acknowledgment,
a delayed acknowledgment commit, the **same words as an answer to a question**,
a short negative answer, short and corrective interruptions, an unfinished
sentence, and a resumed request. English controls cover cumulative transcripts,
duplicates, filler-only playback, and cases where merging must remain off.

Each case gets a fresh detector. Every event uses `state_defaults` plus its
explicit state overrides; snapshots are supplied, not derived from prior
returned decisions. `at_ms` orders events but does not advance a real clock or
change a snapshot's elapsed-time fields. For example, the late-commit case
supplies playback at the partial and no playback 600 ms before the commit. It
checks decisions in those states, not the session's production onset tracking.
The report includes the effective state and expected/observed action and text
for each event, plus corpus and detector-source hashes for comparing runs.
No timings or accuracy percentages are claimed.

The runtime's new evidence fields are:

- `user_turn_active`: a live caller task, excluding greeting/event tasks.
- `reply_heard_bytes`: known non-filler reply playback; `None` means unknown.
- `tool_started`: explicit tool-work evidence; unknown is not tool-free.
- `caller_partial_started_during_playback`: playback at the **first observed STT
  partial**, not acoustic speech onset. Commit-only speech leaves it unknown.

`merge_resumed_secs` defaults to `None` (disabled). The enabled comparison uses
2 seconds. Eligible cases have a live user request, a nonempty active transcript,
elapsed time strictly below that window, known zero heard reply bytes, no tool
work and no HOLD. The expected `continue` text retains the entire original
request plus the continuation. Equality at the window boundary does not merge.
The guards include heard reply bytes, unknown playback, started/unknown tools,
expired timing, HOLD, no live caller task and empty request text. Non-filler
pre-tool speech is conservatively treated as reply audio. Filler-only playback
is explicitly supplied as zero non-filler reply bytes; the replay does not test
whether the session accounts for those bytes correctly.

Default provider behavior is expected to expose unwanted acknowledgment turns,
missed immediate one-word partial interruptions, premature turns for unfinished
speech, and separate turns for resumed requests. Enabling safe resumed merging
addresses only the eligible resumed cases. The report calls other mismatches
`observed_baseline_failure` or `observed_enabled_failure`; they are not silently
marked as passing or counted as fixes. Full-corpus strict mode should therefore
still fail with merging enabled. Expectations are frozen for the documented
2-second comparison; another window can intentionally change the result.

## Catalan microphone demo (optional, billable)

The demo preserves the prior choices: Catalan `ca`, `openai/gpt-6-luna` with low
reasoning effort, ElevenLabs `scribe_v2_realtime`, `eleven_v4_turbo`, and voice
`dNjJKg63Fr5AXwIdkATa`. It uses supported `provider` mode with local VAD endpointing
disabled; Scribe owns endpointing. The prior `local_shadow` mode and experimental
runtime imports are not reused. No acoustic classifier is enabled. Its dynamic
Madrid-date prompt, simulated availability, spoken Catalan/Spanish times,
Catalan greeting, fillers, idle handling and timeout response are preserved.

Provide `OPENAI_API_KEY` and `ELEVENLABS_API_KEY` through the normal server
environment or a private local `.env`. No separate checkout's credentials are
loaded by the example. With dependencies already installed, from the repo root:

```bash
uv run --offline --no-sync python -m timbal.server.playground \
  --import_spec examples/catalan_agent.py::agent
```

For initial dependency setup instead (may require network):

```bash
uv run --extra server python -m timbal.server.playground \
  --import_spec examples/catalan_agent.py::agent
```

Open http://localhost:7777,
select WebSocket, and click **Use agent settings** to clear stale browser
choices. Keep ElevenLabs Scribe, Catalan and provider mode. Local VAD endpointing
should be off. Stop and restart the child server after changes. WebSocket needs
no LiveKit credentials; optional LiveKit transport requires its own credentials
and `--extra voice-livekit` installation.

Environment overrides are `TIMBAL_CATALAN_MODEL`, `TIMBAL_CATALAN_VOICE`, and
`TIMBAL_CATALAN_MERGE_RESUMED_SECS`. Provider-specific reasoning parameters are
only applied to the original default model. Leave the merge variable unset for
the baseline; set it to a positive finite number for the opt-in comparison:

```bash
TIMBAL_CATALAN_MERGE_RESUMED_SECS=2 uv run --offline --no-sync python -m timbal.server.playground \
  --import_spec examples/catalan_agent.py::agent
```

Try “Voldria una revisió demà a la tarda” for the simulated tool. Compare
“D'acord” while listening with “D'acord” as an answer to a question; try
“Espera, vull canviar la data” as an interruption. To investigate resumed
requests, pause before adding a detail, and observe whether a tool or audible
reply already started: either condition should prevent merging. Live pauses,
provider commits and playback acknowledgments are not deterministic.

## Validation limits

The replay validates synthetic text/state fixtures and local code paths. It does
not validate recordings, natural CA/ES/EN overlap, acoustic classification,
provider endpoint timing, microphone quality, actual playback accounting,
VoiceSession cancellation/history reconstruction, or production behavior.
The supplied overlap flag cannot prove acoustic onset. No paid API calls or
recording sessions are part of the replay commands.
Live demo quality and account-specific model/voice availability remain
unvalidated. No recording or production deployment is enabled here.

## Local verification record (2026-10-26)

Against the shared worktree implementation, the 40 cases / 70 events produced
19 matching cases with default provider settings and 25 with merging enabled
at 2 seconds. All 16 resumed-request/merge-guard cases (22 events) matched with
merging enabled. The other 15 cases still expose acknowledgment, immediate
short-interruption and unfinished-sentence gaps; both full-corpus strict runs
correctly exited 1. These are synthetic fixture results, not measured language
accuracy. The JSON reports include source and corpus hashes so comparisons identify the
exact fixture and detector revisions.

All nine selfchecks passed. Separate CLI checks verified strict/filter/error
exit codes, invalid merge windows and byte-identical repeat output. Offline demo
checks validated VoiceConfig, default/override configuration, the date prompt,
and simulated availability for future/past/invalid dates, weekends, elapsed
same-day slots and unknown services. Ruff lint and format checks passed. No live
provider call was made by these demo/replay checks.

Separately, the voice regression tests exercise cancellation races, tool guards,
playback accounting, guardrail rewrites, and transcript/memory replacement,
including a JSONL persistence round trip. The broader voice/server suite passed
1,668 tests and skipped five; its live OpenAI speech smoke test failed API-key
authentication. Actual microphone and provider validation remains outstanding.
