# Norte Market voice demo

A local customer-support call on the same LiveKit path the playground uses in production.

The microphone, Deepgram Flux, `openai/gpt-5.4-mini`, and ElevenLabs are live. The room and the audio are a real LiveKit SFU. The shop is not. Orders, names, tracking, notes, and tickets are synthetic data in the agent process. The tools do not send messages, take payment, refund, or call a carrier. This is not a phone call: nothing here dials SIP or the public telephone network.

Two processes, both started for you:

1. `python -m timbal.server.playground` mints a room and the caller token.
2. On Start it spawns `timbal.server`, and that process joins the room with `livekit.rtc`. That spawned server is the worker. There is no separate LiveKit Agents process.

Leave every pipeline control on **Server default**. The agent already sets Spanish transcription, follow-the-caller replies, filler, entity patience, speculation for `openai/gpt-5.4-mini`, and hangup.

Headphones. Allow the microphone. Speaker audio can be heard by the microphone.

## Update an existing checkout

From the demo clone, with only `uv.lock` dirty. Do not stash, reset, or `checkout --force`.

```bash
cd /Users/dberges/Desktop/timbal-ai/timbal-voice-demo
git status --short
git fetch origin cursor/voice-turn-language-143e
git merge --ff-only origin/cursor/voice-turn-language-143e
git rev-parse HEAD
```

`git status --short` before the merge should show `uv.lock` and nothing else. `--ff-only` refuses the update if it is not a straight fast-forward. Stop if it refuses. The merge does not edit `uv.lock` when the incoming commit does not touch that file, so a dirty lock stays as it is.

A new clone, in a new folder:

```bash
git clone --branch cursor/voice-turn-language-143e --single-branch https://github.com/timbal-ai/timbal.git timbal-voice-demo
cd timbal-voice-demo
```

Every later command is from this directory, the repository root. `uv` from `python/` does not see this `pyproject.toml`. You need `git` and `uv` already on `PATH`.

## Install

`livekit` is the `voice-livekit` extra (`livekit==1.1.14` in `uv.lock`, including the macOS arm64 wheel). It is not installed by a default sync. `--frozen` installs from the lock and does not rewrite it.

```bash
uv sync --frozen --extra server --extra voice --extra voice-livekit
```

If that command fails, stop. Do not remove `--frozen` and do not run `uv lock`.

Do not set `PYTHONPATH`. Do not pass `--no-sync` on this install command. A later `uv run` without the extras can start a process that cannot import `livekit.rtc`; the launcher below pins `--frozen` and `--no-sync` only after this install.

## Keys

Put the keys in the root `.env` only. The launcher lets a nonempty root `.env` value replace the same name in the shell, so a stale exported `OPENAI_API_KEY` does not win. It does not print values.

Names the LiveKit path reads:

- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `ELEVENLABS_API_KEY`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `TIMBAL_LIVEKIT_URL`

`LIVEKIT_URL` is an alias and is optional when `TIMBAL_LIVEKIT_URL` is set. `LIVEKIT_INFERENCE_API_KEY` and `LIVEKIT_INFERENCE_API_SECRET` are not read. Leave `ELEVENLABS_VOICE_ID` and `TIMBAL_VOICE_ID` unset unless ElevenLabs reports the library default voice missing; the example does not set a voice id.

The browser needs a `wss://` URL. A `ws://` URL is accepted only for localhost. Port 7880 is not a browser address.

Do not paste a placeholder over a line that already has a value. To add one missing name:

```bash
read -r OPENAI_API_KEY
printf 'OPENAI_API_KEY=%s\n' "$OPENAI_API_KEY" >> .env
```

Use the missing name. `>>` appends. It does not replace an existing line. If the name is already in `.env`, do nothing.

## Run

Stop the old direct server first if it is still the process on port 4444. That page cannot mint a LiveKit dial. Ctrl-C in the terminal that started it. Do not open `http://127.0.0.1:4444/voice`.

```bash
uv run --frozen python examples/voice_support_demo_livekit.py
```

Check without opening the playground:

```bash
uv run --frozen python examples/voice_support_demo_livekit.py --check
```

A passing check prints `checks=ok`. It does not print `ready`, a token, or the SFU URL. `list_rooms` is a read. The join creates one empty room named `pg-` plus 12 hex characters, connects the agent, then deletes that room with a `roomCreate` grant. `roomAdmin` is not that permission: LiveKit answers it with HTTP 401. A 404 means the room is already gone. The check does not publish a microphone, so Flux and ElevenLabs are not opened by it, and it does not change any other room, dispatch rule, or SIP trunk. If the delete line is not `delete_room=ok`, the command stops and does not print `checks=ok`.

Then the same command without `--check` prints:

```text
playground=http://localhost:7777/?transport=livekit
```

Open that URL. The query selects **LiveKit (SFU)** before the browser's saved choice. A saved WebRTC choice is why a tab opened at `http://localhost:7777/` can show WebRTC while this command is healthy. Close that tab. Do not change the dropdown by hand. Pipeline controls, including the voice id, stay on **Server default**. Press **Start**.

If port 7777 is already taken, the command prints `playground_port=7777 in_use` and exits. Ctrl-C the playground that holds it, then run the same command again. Events show up only in the terminal that is running this process.

The child is started with `python -u`, so its logs are not held until exit. They are printed in this same terminal, with URLs and tokens redacted. Exceptions are printed too. `voice_livekit_joined` means the worker is in the room. `voice_livekit_session_ready` means the microphone was subscribed and the session was built. `dg_flux_session_started` means Flux accepted the stream. `turn_interruption_truncation` carries `heard_chars` and `generated_chars`. A missing interruption line is not proof that nobody barged in. Do not paste an unredacted log.

The Agent field is prefilled with `examples/voice_support_demo.py::agent`. That is the shop. `examples/voice_playground_agents.py::thinking` is a different English tutor with a slow `get_datetime` tool. Leave the shop spec in place for this test.

The child cwd is `examples/` because the agent file lives there. `uv` walks up to this repository. The version string on the page is package metadata. It does not, by itself, mean the source is old.

`Session ended` / `Click Start for a new session` is the page after Stop. It is not a server log, and it does not say whether Flux started. WebRTC is a different transport: its server line is `voice_rtc_connected`, and that path is not this demo. LiveKit does not fall through to WebRTC or WebSocket when the URL selected it.

`voice_livekit_missing_extra` means the sync command above was not the interpreter this process is using.

To keep a local log:

```bash
uv run --frozen python examples/voice_support_demo_livekit.py 2>&1 | tee voice-demo.log
```

`voice-demo.log` matches `*.log*` in `.gitignore`. It can contain the transcript and the SFU URL. Do not commit it or paste it.

## Cases

| Id | You say | Expect |
| --- | --- | --- |
| es-order | Quiero el estado del pedido 1001. | Ana Ruiz, Barcelona, packed. |
| en-follow | And the shipment for 1003? | English. Marta Soler, delivered. |
| ca-follow | I el 1002? | Catalan. Jordi Puig, processing. |
| unknown | Pedido 9999. | Unknown. No write. |
| slow-filler | Estado del 1002. | "Un momento." during about four seconds, then processing. |
| retry | Cliente del pedido 4040. | One failure, then Pol Serra. |
| both | Cliente y envío del 1001. | Both tools, one answer. |
| stage-yes | Cambia el 1001 a Calle Mallorca 10, Barcelona. Sí. | Saved only after the spoken yes. |
| stage-no | Cambia el 1001 a Calle Provença 5. No. | Unchanged. |

One regression call: a phone number with pauses; "Ana Ruiz" spelled with pauses; the slow order and its filler; Stop; "Okay" while the agent is speaking; Spanish then English; "Cuelga. Alright."; then "Espera." followed by "Adiós.", with one audible farewell.

A delivery or ticket change is staged first. It is stored only after a spoken yes. "No" drops it. There is no refund tool.

A new browser Start is a new room and a new copy of the shop. Restarting the playground clears every copy. Two calls do not share a bag. Nothing is written to disk or to a CRM.

The filler stays "Un momento." because its language follows the Spanish transcription hint.

## Still open

Hangup at `72240731` is not finished. One goodbye, a retract that rearms, Okay, and Bye bye have been confirmed. Two observations are still open and need another call: a repeated "Alright. Alright." that cancels the hangup, and another partial that cancels during the goodbye.

The tutor in `examples/voice_playground_agents.py::thinking` keeps `get_datetime` so a date or time question can show the slow tool and the filler. The prompt now says not to call it for anything else. A hats puzzle is not a date question. The line "Calling get_datetime…" is the tool-status text. "Hang on…" / "Hold on…" are the filler, generated by `filler.model`, which is `groq/qwen/qwen3.8-27b` even when `TIMBAL_VOICE_DEMO_MODEL` changes the tutor model.

## What this machine proved

The unit tests cover the redacted report, file-over-shell keys, the room-list grant, a delete that uses `roomCreate` (and the HTTP 401 for `roomAdmin`), a playground spawn whose JSON contains a URL and a token, and the page query that selects LiveKit ahead of a saved choice. They do not contact a LiveKit project. A cloud `--check` without the three LiveKit names printed the provider lines and `livekit_package=ok`, then `blocked=config_package_or_provider`. It did not join a room. Your `--check` is the run that talks to your project.
