# Live tool-speech test

Run from the repository root, using a checkout containing the tool-speech changes:

```bash
uv sync --extra server --extra voice
export PYTHONPATH="$PWD/python"
export VOICE_DEMO_ENV="$PWD/.env"
```

The example loads that dotenv file with precedence over inherited shell variables,
so stale exported provider keys do not override its credentials.
You need `DEEPGRAM_API_KEY` and `ELEVENLABS_API_KEY`; live LLM mode also needs
`OPENAI_API_KEY`. Set `ELEVENLABS_VOICE_ID` if you want a particular voice.
No credentials are included in the example. Tools return simulated order data.

Start the server with the checkout’s environment:

```bash
VOICE_DEMO_MODE=live VOICE_DEMO_BEAT=0 \
  uv run --no-sync python -m timbal.server.http \
  --import_spec examples/voice_tool_speech.py::agent --port 4444
```

For PowerShell, use `$env:PYTHONPATH="$PWD/python"`,
`$env:VOICE_DEMO_ENV="$PWD/.env"`, `$env:VOICE_DEMO_MODE="live"`, and
`$env:VOICE_DEMO_BEAT="0"`, then run the `uv run --no-sync python` command on one line
without the Bash environment prefixes or backslashes. Change the beat variable
to `"0.6"` before the second run.

Open http://localhost:4444/voice/ in **one tab** and connect once. If another server
already uses port 4444, stop it first. Keep the microphone permission enabled;
headphones help avoid speaker audio being transcribed back into the call.

## Listening sequence

1. Say **“Fast test.”** Expect “Okay, let me check that order for you…” followed
   by “Order 101 has shipped and arrives tomorrow.” Both phrases should be heard
   once. The terminal should log `demo_tool_started` and `demo_tool_finished`
   before the acknowledgement has finished playing.
2. Say **“Slow test.”** The acknowledgement should play while the eight-second
   lookup runs, followed by the result. It should not wait eight seconds to say
   the acknowledgement.
3. Say **“Silent test.”** Expect only the result, with no pre-tool phrase.
4. Say **“Slow test”**, then **“Stop”** while waiting. The lookup should be
   cancelled and the old result should not resume later.

Disconnect, stop the server with Ctrl-C, then repeat with a 0.6-second beat:

```bash
VOICE_DEMO_MODE=live VOICE_DEMO_BEAT=0.6 \
  uv run --no-sync python -m timbal.server.http \
  --import_spec examples/voice_tool_speech.py::agent --port 4444
```

Refresh the page and reconnect after each restart. Avoid changing model or voice
settings in the playground during the comparison. The fast test should now have
at least 0.6 seconds of silence after the acknowledgement, plus synthesis/network
latency. The slow test should add no intentional pause after tool completion if
that beat has already elapsed. The silent test should add no intentional beat.
For an obvious cancellation check, temporarily use `VOICE_DEMO_BEAT=3`, say
“Fast test”, then say “Stop” during the silent gap; the old result must not play.

## Deterministic control

An LLM may choose to omit its pre-tool text despite the prompt. If that happens,
repeat with this control (real microphone and TTS, scripted non-streaming model):

```bash
VOICE_DEMO_MODE=scripted VOICE_DEMO_BEAT=0.6 \
  uv run --no-sync python -m timbal.server.http \
  --import_spec examples/voice_tool_speech.py::agent --port 4444
```

Use the same phrases. Compare with `VOICE_DEMO_BEAT=0`. Scripted mode does not
exercise streamed LLM deltas; live mode does. It also requires no OpenAI calls
for generating replies. Provider startup may still depend on server warmup behavior.

A good listening result verifies overlap, complete text, pause timing and
interruption. The ElevenLabs settings' exact wire placement is covered by the
mock-WebSocket regression tests; listening alone cannot prove that placement.
