# FDE test: #242 voice ship config

Do not merge this branch to main. It is a callable slice of the voice work, not the lab report. The agent already declares the ship `VoiceConfig`. Leave the playground on **Server default** and you are on that config.

```python
voice_config = {
    "language": "en",  # or "es"; omit to follow the latest substantive message
    "ignore_backchannels": True,
    "speculative_generation": {"enabled": True, "preemptive_tts": True},
    "max_tts_pause_secs": 0.2,
    "filler": {"enabled": True, "soft_timeout_secs": 0.8, "message": "One moment."},
    "hangup_grace_secs": 1.2,
    "entity_patience": True,
}
```

`ignore_backchannels` and `speculative_generation` are server-only. A browser hello cannot turn them off. `end_call` awaits `session.commit_hangup()` so speech inside the 1.2s grace window cancels the hangup. A bare `session.end_call()` does not.

The example file is not in the wheel (`/examples` is excluded from the sdist). Check out the branch and run from the repo root.

## Install

```bash
git clone --branch voice/fde-ship-config --depth 1 https://github.com/timbal-ai/timbal.git
cd timbal
uv sync --extra server --extra voice
```

Library only, into the current env (you still need `examples/voice_fde_ship.py` from this branch):

```bash
uv pip install "timbal[server,voice] @ git+https://github.com/timbal-ai/timbal.git@voice/fde-ship-config"
# or
pip install "timbal[server,voice] @ git+https://github.com/timbal-ai/timbal.git@voice/fde-ship-config"
```

## Keys

The server loads a `.env` from the working directory. Export these or put them there. No keys are committed.

| Key | Why |
| --- | --- |
| `OPENAI_API_KEY` | Default model `openai/gpt-5.4-mini`, the model this config was measured on. Override with `TIMBAL_VOICE_DEMO_MODEL`. Do not point this stack at `groq/openai/gpt-oss-20b`. |
| `DEEPGRAM_API_KEY` | Flux `flux-general-multi`, the measured STT. Without it the session falls back to ElevenLabs Scribe. |
| `ELEVENLABS_API_KEY` | TTS, default `eleven_v4_turbo`. Optional `ELEVENLABS_VOICE_ID`. |

Language, set before the process starts:

| `TIMBAL_VOICE_LANGUAGE` | Effect |
| --- | --- |
| absent | `"en"` |
| `es` (or another BCP-47 tag) | pin that language |
| empty, `unset`, `auto`, `none`, `off` | omit `language`; reply follows the latest substantive message |

The filler line stays `"One moment."` in every language. That string is the measured `filler.message`.

## Run

From the repo root, one server on the port:

```bash
uv run python -m timbal.server --import_spec examples/voice_fde_ship.py::agent --port 4444
```

Open http://127.0.0.1:4444/voice in one tab and connect once. Headphones, so the speaker is not transcribed back into the call. If 4444 is taken, stop the other process or pass another `--port`.

This example does not place a phone call. The playground microphone is the call. There is no LLM picker on the page; the model is the one in the agent. Do not flip Filler, language, or entity patience off the Server default if you want the ship block.

Print the resolved config without starting a call:

```bash
uv run python examples/voice_fde_ship.py
```

## What to try

- **Language pin.** With the default, replies stay English, including after a Spanish word that is not an explicit request to switch. Restart with `TIMBAL_VOICE_LANGUAGE=es` for a Spanish line. Restart with `TIMBAL_VOICE_LANGUAGE=unset` and change language mid-call; the reply should follow the latest full request, not a backchannel or a borrowed word.
- **Hangup retract within ~1s.** Say goodbye. While the agent is saying goodbye, within about a second say "wait, one more thing". The line stays up and the tool returns that you spoke again. Say goodbye and stay quiet to actually hang up.
- **Spoken phone number, no premature cut.** Say "my number is six five five, zero one two, three four" with short pauses between groups, or spell a name ("J, O, N, E, S"). The agent should hold the turn across the groups instead of answering after the first one.
- **Slow-tool filler.** Ask it to look up order 1001. `check_order` sleeps 4 seconds. After about 0.8s of silence you should hear "One moment." then the status.
- **Speculative barge-in.** Ask a question, then talk over the answer ("stop", "no, that's wrong"). The reply should cut. While it is speaking, "yeah", "okay got it", and "vale" should not cut it.

**Known gap.** Dictation read-back of numbers and names is still weak. Spoken-word dictation no longer ends the turn early, but the model often does not say the whole number or the whole spelled name back.
