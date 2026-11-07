# Norte Market voice demo

A local customer-support call. The microphone, Deepgram Flux, `openai/gpt-5.4-mini`, and ElevenLabs are live. The shop is not. Orders, names, tracking, notes, and tickets are synthetic data in this process. The tools do not send messages, take payment, refund, or call a carrier.

Leave the playground on **Server default**. The agent already sets the config.

```python
voice_config = {
    "language": "es",              # speech-to-text hint
    "reply_language": "follow",    # a full English or Catalan request is answered in that language
    "ignore_backchannels": True,
    "speculative_generation": {"enabled": True, "preemptive_tts": True},  # only for openai/gpt-5.4-mini
    "max_tts_pause_secs": 0.2,
    "filler": {"enabled": True, "soft_timeout_secs": 0.8, "message": "Un momento."},
    "hangup_grace_secs": 1.2,
    "entity_patience": True,
}
```

`ignore_backchannels` and `speculative_generation` are server-only. A browser hello cannot turn them off. Another `TIMBAL_VOICE_DEMO_MODEL` turns speculation off. The filler line stays Spanish because it follows the `es` hint, not the language of the latest reply.

Headphones. The page is a browser microphone, not a phone call.

## Clone

Use a new folder. Do not run this inside a checkout you are already using.

```bash
git clone --branch cursor/voice-turn-language-143e --single-branch https://github.com/timbal-ai/timbal.git timbal-voice-demo
cd timbal-voice-demo
git rev-parse HEAD
```

The commit this file belongs to is the one `git rev-parse HEAD` prints after that clone. Run every later command from `timbal-voice-demo`, the repository root. `uv sync` from `python/` will not see this `pyproject.toml`.

You need `git` and `uv` already on your `PATH`. This does not install them.

```bash
uv sync --extra server --extra voice
```

## Keys

The server calls `load_dotenv()` in the working directory. By default that does not replace variables already exported in the shell.

Print only missing names. This does not print values.

```bash
python -c 'import os; need=("OPENAI_API_KEY","DEEPGRAM_API_KEY","ELEVENLABS_API_KEY"); missing=[n for n in need if not os.environ.get(n)]; print("\n".join(missing) if missing else "none missing")'
```

If a name is printed, set that one in the shell. The value is not part of the command:

```bash
read -r OPENAI_API_KEY
export OPENAI_API_KEY
```

Repeat for each missing name. Do not write `KEY=...` into a file that already has a real value. If the name is already exported, leave it.

Voice id is optional. If `ELEVENLABS_VOICE_ID` or `TIMBAL_VOICE_ID` is already set, the server uses it. If neither is set, the server uses the library default voice. The example does not set a voice id. Set one of those variables only when ElevenLabs reports that the voice is missing from the account, using the same `read` / `export` pattern.

## Run

One server, from the clone root:

```bash
uv run python -m timbal.server --import_spec examples/voice_support_demo.py::agent --port 4444
```

To keep a local log as well:

```bash
uv run python -m timbal.server --import_spec examples/voice_support_demo.py::agent --port 4444 2>&1 | tee voice-demo.log
```

`voice-demo.log` will contain transcripts. It matches `*.log*` in `.gitignore`. Do not commit it or paste it into a ticket.

Open http://127.0.0.1:4444/voice and allow the microphone. Connect once. If 4444 is taken, pass another `--port` and use that port in the URL.

Print the config without a call:

```bash
uv run python examples/voice_support_demo.py
```

Check the example without providers:

```bash
uv run pytest python/tests/voice/test_voice_support_demo.py -q --no-cov -p no:cacheprovider
```

## What the shop remembers

Each browser connection gets its own copy of the seed orders. A note or ticket saved on this call is not visible on the next connection. Restarting the server drops all of them. There is no file and no external shop.

`stage_delivery_change` and `stage_support_ticket` only preview. `apply_pending` writes the preview into that call's copy, and only after it was staged. `cancel_pending` drops a preview. Nothing is saved if you never hear `apply_pending` say it was saved.

| Id | Record |
|---|---|
| 1001 | Ana Ruiz, Barcelona, packed, tracking NM1001. Fast lookup. |
| 1002 | Jordi Puig, Girona, processing. Lookup sleeps 4 seconds. |
| 1003 | Marta Soler, Valencia, delivered. |
| 4040 | Pol Serra. First lookup in the call fails. The retry returns the order. |
| 9999 | Unknown. |

## Tour

Say these. Expected results are what the tools return, not a timing guarantee.

1. **es-order.** "¿Qué sabes del pedido diez cero cero uno?" Status packed, Ana Ruiz, no invented delivery window.
2. **en-follow.** "What is the status of order one zero zero three?" The reply is English. Marta Soler, delivered.
3. **ca-follow.** "On és la comanda deu zero zero u?" The reply is Catalan and uses the simulated tracking for 1001.
4. **unknown.** "Pedido nueve nueve nueve nueve." It says there is no such order.
5. **slow-filler.** "Mira el pedido diez cero cero dos." After a short wait you hear "Un momento." Then Jordi Puig, processing.
6. **retry.** "Pedido cuatro cero cuatro cero." The first lookup fails. "Inténtalo otra vez." The second returns Pol Serra.
7. **both.** "Estado y seguimiento del diez cero cero uno." It calls lookup and tracking. Ana Ruiz and NM1001.
8. **number.** "Mi número es seis cinco cinco, cero uno dos, tres cuatro." It waits through the pauses and reads the whole number once.
9. **name.** "El nombre es J, O, N, E, S." It reads every letter.
10. **stage-yes.** "Cambia la nota del diez cero cero uno a llamar al llegar." It reads the preview and does not say it is saved. "Sí." Then it says it was saved in this call.
11. **stage-no.** Stage another note. "No." Nothing is saved.
12. **stop.** Ask for a long explanation, then "Stop" while it speaks. Playback cuts.
13. **okay.** While it speaks, "Okay" or "Vale". Playback continues.
14. **alright.** "Cuelga." You hear one goodbye, not "Un momento." During the quiet second, "Alright" or "Vale". The call still ends.
15. **wait.** "Cuelga." During the goodbye, "Espera, una cosa más." The call stays up. Say goodbye again and stay quiet. One goodbye, then the call ends.

A new connection starts from the seed again, so the note from step 10 is gone.
