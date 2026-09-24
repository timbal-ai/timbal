# OpenAI speech provider verification — 2026-09-24

This audit checks the supported API contracts and local audio-processing costs.
It does not establish production latency, transcription accuracy, or concurrent-call capacity.

## Reproduce

With the project environment installed and `OPENAI_API_KEY` set:

```bash
.venv/bin/python benchmarks/voice/openai_speech.py --output /tmp/openai-cpu.json
# Billable requests using only synthetic speech:
.venv/bin/python benchmarks/voice/openai_speech.py --live --output /tmp/openai-live.json
.venv/bin/pytest python/tests/voice python/tests/server/test_voice_config.py python/tests/server/test_voice_ws.py python/tests/tools/test_openai_audio.py --no-cov -q
.venv/bin/pytest python/tests/voice/test_openai.py python/tests/tools/test_openai_audio.py -m integration --no-cov -q
```

Raw measurements are saved locally in
`results/openai-speech-2026-09-24.json` (gitignored benchmark output).

## Coverage and results

- Regression suite: **733 passed, 4 skipped, 3 deselected**. One existing warning
  in `TestTimeoutFallbackRunId::test_fallback_done_still_names_the_hung_run`
  reports an unawaited `BaseCollector.__anext__`; it was also present before this audit.
- Live integration tests: **2 passed**. Streaming TTS/STT roundtrip at the default
  16 kHz session rate, manual and automatic turn commits, and standalone WAV TTS/file STT.
- Live model matrix: all six supported STT identifiers transcribed the synthetic
  phrase correctly using manual commit, server VAD, and file upload (18 checks).
  The identifiers are `gpt-transcribe`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`,
  `gpt-4o-mini-transcribe-2025-03-20`, `gpt-4o-mini-transcribe-2025-12-15`, and `whisper-1`.
- TTS matrix: **75/75 combinations returned audio**, covering all built-in
  voices on all seven supported identifiers:
  `gpt-4o-mini-tts` and its 2025-03-20 / 2025-12-15 snapshots (13 voices each),
  `tts-1`, `tts-1-1106`, `tts-1-hd`, and `tts-1-hd-1106` (9 voices each).
- Unit tests cover cancellation during setup/audio reads, reconnect, response cleanup,
  HTTP/WebSocket errors, transcript ordering, PCM byte alignment and duration,
  8/16/24/48 kHz sample rates, minimum-size commits, provider selection,
  credential plumbing, file upload validation, and model-specific language fields.
- A fresh-process regression verifies selecting existing providers does not import
  the OpenAI adapter. Their per-frame audio processing paths are unchanged.

The default STT model is now `gpt-transcribe`. Legacy STT models are supported
for compatibility, but OpenAI lists their deprecation in its
[deprecation schedule](https://developers.openai.com/api/docs/deprecations).
`gpt-live-transcribe`, `gpt-realtime-whisper`, and diarization models are explicitly
unsupported by this server-VAD adapter, rather than silently substituted.
The first two rejected server VAD in live API probes; see the
[live transcription protocol](https://developers.openai.com/api/docs/guides/realtime-transcription).
Voice restrictions follow the [Speech API documentation](https://developers.openai.com/api/docs/guides/text-to-speech).

## Performance observations

On this local macOS/arm64 machine with Python 3.11.11:

- STT resampling plus base64/JSON encoding: median **0.714 ms CPU per second of
  input audio** at 16 kHz; **0.245–0.859 ms** across 8/16/24/48 kHz inputs.
- TTS resampling from 24 kHz to 16 kHz: **0.450 ms CPU per second of audio**.
  Each result is the median of five runs of 60 seconds of audio in 20 ms frames.
  These microbenchmarks exclude sockets, TLS, event-loop scheduling, STT/TTS
  inference, the agent, turn detection, playback, and recording.
- First returned TTS audio on fresh HTTP clients for the same short phrase,
  across the voices on each alias: `gpt-4o-mini-tts` **399–858 ms** (median 568),
  `tts-1` **815–2357 ms** (median 1165), `tts-1-hd` **1472–2544 ms** (median 1806).
  These are single-request-per-voice observations, not percentiles or an SLA.
  Dated snapshots were also tested; the slowest observed first audio was
  3565 ms on `tts-1-hd-1106` / `sage`.
- `gpt-transcribe`: **785 ms** from audio sent/manual commit to final transcript;
  **766 ms** after sending the automatic-VAD turn's trailing silence. Audio was
  fed faster than real time. These measurements are not conversational EOU latency.

The Speech API receives a complete text segment per HTTP request. Timbal reuses
its HTTP client but schedules segments through the existing sequential synthesis
fallback. This can introduce inter-segment gaps and differs from ElevenLabs'
incremental text stream. Switching providers can change conversational latency
and prosody even when local framework CPU costs are small.

## Fixes found by the audit

- Added model-specific voice/instruction validation before TTS startup, and explicit
  rejection of unsupported OpenAI model IDs.
- Replaced the legacy STT default and mapped language hints to the current API schema.
- Kept the OpenAI adapter lazy and filtered its exclusive tuning keys when selecting
  existing providers.
- Preserved next-turn audio accounting when an earlier commit acknowledgement arrives late.
- Prevented later-turn partial transcripts from overwriting an earlier outstanding turn.
- Flushed resampler tails before enforcing the 100 ms minimum audio commit size.
- Moved standalone file reads off the async event loop.

Remaining evaluation: human microphone/PSTN audio, accents and languages, noise,
long sessions, sustained concurrent load, live interruption timing, listening
quality, and production deployment measurements. Custom/cloned voices and model
identifiers outside the explicit catalog are not covered.
