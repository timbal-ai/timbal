# Timbal HTTP server

FastAPI app factory and CLI live in `http.py` (`python -m timbal.server`, or `run_server_cli`). The runnable is selected via `TIMBAL_RUNNABLE` (same object used for `/run`).

Two things are documented here: [running a runnable over HTTP](#runs-run-stream-and-reconnecting) (including how a dropped stream reconnects), and how a **custom frontend** integrates with the **voice agent** over WebSocket (not the bundled `voice.html` playground).

---

## Runs: `/run`, `/stream`, and reconnecting

| Route | Purpose |
|---|---|
| `POST /run` | Run to completion, return the final `OUTPUT` event as JSON. |
| `POST /stream` | Run and stream every event as SSE. |
| `GET /runs/{run_id}/events` | Replay a run's events after a cursor — the reconnect path. |
| `POST /cancel/{run_id}` | Cancel a running run. `404` if unknown or already finished. |

Set `context.id` on the request to choose the run id; otherwise one is generated and you can read it off the first event. Naming an id that a *running* run already holds is a `409` — the alternative is silently orphaning that run, leaving it executing with nothing able to read or cancel it.

Reusing the id of a run that has *finished* but is still inside its retention window is allowed, and rebinds the id: anything still polling `/runs/{run_id}/events` for the old run starts receiving the new one's events, with seqs counting from 1 again. Use fresh ids per run unless you mean that.

### A dropped connection does not stop the run

Runs execute on their own task, decoupled from the HTTP response. When a client disconnects, the run keeps going and keeps appending to its **event log** — an append-only list where every event has a 1-based monotonic `seq`. Readers hold a cursor into that log rather than consuming from a queue, so a disconnect costs you nothing but your place, several readers can watch one run at once, and coming back is just asking for everything after the last `seq` you saw.

Finished runs stay readable for a retention window (default 5 minutes, `JobStore(retention_secs=…)`) so a client that dropped right at the end can still collect the tail.

Only `/stream` keeps a replayable log. A `/run` is read to completion by the one request that started it and nothing can reconnect to it, so its events are dropped as they are consumed rather than held — otherwise every non-streaming request would pin a whole run's deltas in memory for five minutes after answering. `/runs/{run_id}/events` reports `expired: true` for a `/run`.

### Reconnecting

`GET /runs/{run_id}/events`:

| Query | Meaning |
|---|---|
| `after` | Last `seq` seen (exclusive). Default `0` = from the start. |
| `limit` | Max events per response, clamped to 1–2000. Default `500`. |
| `wait_ms` | Long-poll budget, clamped to 0–30000. Default `0` (return immediately). When `>0` and nothing is available past `after`, the request blocks until an event arrives, the run ends, or the timeout elapses. |

```json
{
  "run_id": "0198f3aa…",
  "events": [{ "seq": 1, "data": { "type": "START", … } }],
  "next_cursor": 1,
  "done": false,
  "expired": false
}
```

Feed `next_cursor` back as `after` on the next poll. It equals the last `seq` returned, or your `after` unchanged when the batch is empty.

**`done` means "you have everything", not "the run finished."** A terminal run whose events don't fit in one page reports `done: false` so you keep paging.

**`expired: true` is the field you cannot ignore.** It means the log is unavailable — reaped after retention, never held by this process, belonging to a `/run`, or trimmed past your cursor. The response then looks *identical* to a cleanly drained stream (`done: true`, no events, cursor unmoved), so a client that only checks `done` stops early and silently loses the tail. Treat `expired` as **terminal but possibly incomplete** and reconcile from wherever you persist runs.

`POST /stream` also emits the `seq` as the SSE `id:` field, so a client following the stream always knows the cursor to reconnect with without parsing the payload. It is *not* an `EventSource` resume: `/stream` is a POST, so `EventSource` cannot reach it, and nothing here reads `Last-Event-ID`. Reconnect through `/runs/{run_id}/events?after=`.

A stream the log outran — a reader stalled long enough for the ring to drop events it had not taken — is closed with a final frame rather than just ending, since ending is indistinguishable from a clean finish:

```
event: expired
data: {"run_id":"0198f3aa…","next_cursor":41,"done":true,"expired":true}
```

Same meaning as the field: terminal, possibly incomplete. A `/run` in that position answers `500` instead — its whole response is the run's last event, and the one it holds is not it.

### Limits

**Single process.** The log and the job registry live in the serving process's memory, and nothing routes a request to the worker that owns a given run. Run the server with one worker, or pin runs to a worker at the load balancer. With `--workers > 1` (the CLI warns) both `/cancel/{run_id}` and `/runs/{run_id}/events` are a coin flip: a request landing on a sibling worker gets `404` and `expired: true` respectively, for a run that is alive and fine. `expired` cannot distinguish "gone" from "not mine", so a client would go reconciling against durable storage for a run still eight minutes from finishing.

A run whose process dies has nothing to replay. A replayable log is a ring: default 50 000 events or 32 MiB, whichever hits first (`JobStore(max_events=…, max_bytes=…)`). Older events are dropped and a reconnect whose cursor is behind the floor reports `expired: true` rather than silently skipping to the new head. Durable, cross-process replay needs a backing store behind this same cursor contract.

**No authorization.** These routes carry none of their own, and CORS is wide open, so anything that can reach the port can replay any run whose id it can name — and ids are client-chosen, so they are guessable. Deploy behind something that authenticates and scopes by run.

---

### Bundled voice playground

- **Embedded** (served by a running agent): `GET /voice` on a `timbal.server` — auto-dials that agent; the page injects the runnable meta at serve time.
- **Standalone** (no agent required): `python -m timbal.server.playground` — serves the same HTML raw and opens a Target panel. Local target: enter an agent path (`path/to/agent.py::object`, optional fixed port) and press Start — the launcher spawns `uv run python -m timbal.server --import_spec …` from the agent file's directory, waits for the healthcheck, and the page dials it (changing agent/port respawns on the next Start). Platform target: a deployed workforce (`api.dev.timbal.ai` / `api.timbal.ai`) via ticket-authenticated WS / bearer-authenticated RTC. Fields persist in `localStorage`.

---

## Voice WebSocket: `/voice/ws`

Connect to the same host as the HTTP server:

- **URL:** `ws://<host>/voice/ws` or `wss://<host>/voice/ws`
- **Runnable:** The same object as `/run` (`TIMBAL_RUNNABLE`). **Voice requires that runnable to be a Timbal `Agent`.** If it is not, the server closes the WebSocket with code **1008** and reason `Voice requires an Agent runnable` (no JSON messages are sent). There is no separate voice-agent id in the protocol.

Server-side env and keys (e.g. ElevenLabs, model provider for the agent) are an operations concern; the client only speaks this socket. The server needs `timbal[server]` (includes `websockets`) and an ElevenLabs API key for STT/TTS.

### Connection order

1. Open the WebSocket.
2. **First message (within ~2 seconds):** either  
   - **one text frame** with JSON config (see [Config overrides](#config-overrides)), or  
   - **binary** PCM as the first frame (server uses defaults and treats that frame as audio), or  
   - nothing — the server waits up to 2s; if no frame arrives, it continues with defaults.  
   The config hello is the only client JSON **without** a `"type"` field; typed protocol frames (`"playback"` acks, `"audio"`, `"mic_change"`) that race ahead of it are consumed as protocol messages — any number of them — until the hello arrives or the 2s deadline passes.  
   Malformed early frames (invalid JSON, an `"audio"` frame with a missing/bad `data` payload) are logged and skipped — they do not end the handshake, so a valid hello sent afterward within the window still applies. If no hello arrives by the deadline, the server continues with **empty** client overrides (defaults still apply).
3. After that, stream **microphone audio** until the socket is closed.

If the client needs to set `sample_rate` or `language`, the **first** frame should be that JSON text message (unless they are fine with defaults and send binary first).

### Audio uplink (client → server)

- **Format:** mono **PCM**, **16-bit signed little-endian** (`pcm_s16le`), at **`sample_rate`** from the merged config (default **16000** Hz).
- **Transport (either):**
  - **Binary frames:** raw PCM bytes (chunk duration is up to the client; ~100 ms chunks are a reasonable default).
  - **Text JSON:** `{ "type": "audio", "data": "<base64-encoded PCM bytes>" }` on any later message.

If the **first** message is binary, it is queued as the first audio chunk (no separate config message).

### Playback acks (client → server)

Optional but strongly recommended: `{ "type": "playback", "played_ms": <number> }`. Send one every ~250 ms while audio is playing, plus a final one when playback stops. The server advertises this in `session_started` (`"playback_acks": "recommended"`).

**`played_ms` contract** (truncation correctness depends on it — treat as normative):

- **Cumulative per session:** total milliseconds of TTS audio actually played since the WebSocket opened. Never reset it — not per turn, not per audio segment.
- **Monotonic:** each ack must be `>=` the previous one. The server ignores acks that move backwards.
- **Interrupted audio is excluded, forever:** when you receive `interrupted`, stop playback and drop your buffer; the dropped audio must never be counted in a later ack. Only add milliseconds while audio is audibly playing.
- **Played, not received:** count what came out of the speaker (e.g. Web Audio scheduled time already elapsed), not bytes decoded or buffered.

The server uses these to know exactly what the user *heard* when they barge in, so it can truncate the assistant's transcript entry and conversation memory to the heard prefix (see `interrupted.heard_text` below). Without acks the server falls back to a wall-clock estimate of gapless playback, which is close but can't account for client-side buffering delays — the first estimate-only barge-in logs a `voice_ws_truncation_degraded` warning server-side. Malformed `playback` messages are ignored.

Whether acks were received is reported per turn in `metrics.playback_acks_received`, and interrupted turns carry `metrics.heard_bytes` (PCM bytes actually heard) so estimate-vs-ack drift is measurable from traces.

### Config overrides

Optional **first** text frame: a JSON object merged on top of `app.state.voice_config`, which is built at startup from environment defaults and optional `runnable.voice_config` on the loaded agent (`http` lifespan). Server-side `voice_config` (a dict, zero-arg callable, or `timbal.voice.VoiceConfig`) is validated strictly at startup — an unknown key fails server boot instead of being silently ignored.

Only send keys you need; omitted keys keep server defaults. Client keys are allowlist-filtered (`CLIENT_SETTABLE_VOICE_FIELDS`): the table below plus `model` (per-session LLM override, `"provider/model"`), `turn_timeout_secs`, and `turn_timeout_fallback` (`""` disables the spoken apology). Anything else — notably `recording` — is server policy and is ignored with a log line.

| Key           | Description |
|---------------|-------------|
| `stt_provider` | `"elevenlabs"` (default), `"deepgram-flux"`, or `"deepgram-nova"` (bare `"deepgram"` routes by `stt_model`, defaulting to Flux). Deepgram needs `DEEPGRAM_API_KEY` on the server. Flux (`/v2/listen`) does model-native end-of-turn detection: the session auto-selects the `provider` turn detector (explicit `turn_detector` still wins) and disables local VAD endpointing. Nova-3 (`/v1/listen`) is plain ASR — Timbal turn detection and VAD endpointing work exactly as with ElevenLabs. Env default: `TIMBAL_STT_PROVIDER`. |
| `stt_model`   | Speech-to-text model id (ElevenLabs realtime `scribe_*`, Deepgram `flux-general-en`/`flux-general-multi`/`nova-3*`). Model ids that don't belong to the selected provider are ignored (provider default used). |
| `tts_model`   | Text-to-speech model id. |
| `voice`       | ElevenLabs voice id string. |
| `language`    | e.g. `"es"`. Unset → provider auto-detect. |
| `sample_rate` | Hz; STT/TTS audio use this unless extended later. |
| `encoding`    | Default `"pcm_s16le"`. |
| `stt_extra`   | Object deep-merged over the server's STT options (e.g. VAD). Client keys are allow-listed to tuning knobs (`CLIENT_TUNING_STT_EXTRA`); `stt_host`, `callback` and any other key are dropped with a log line and the server value is kept — the provider host is server config only. |
| `tts_extra`   | Object deep-merged over the server's TTS options. Same rule: allow-listed tuning keys only (`CLIENT_TUNING_TTS_EXTRA`); `tts_host` is server config only. |
| `turn_detector` | A mode name: `"local"` (default — Smart Turn v3 + Namo DistilBERT + Silero VAD endpointing when `timbal[voice]` is installed; falls back to `"lexical"` without the extra), `"provider"` (trust STT/realtime endpointing; auto-forced for Deepgram Flux), `"lexical"` (punctuation HOLD), `"heuristic"` (no holds — opt-in), or `"raw"` (debug: no silence/noise/echo filtering — every STT commit becomes a turn, including the agent's own speech leaking through the mic; never use in production). Set `TIMBAL_SMART_TURN_CHECKPOINT=int8` to trade ~1pp accuracy for ~2x faster inference on Smart Turn. The client JSON may only send a mode name string (the playground page has a dropdown for this); it takes precedence over the server value for that session. Server-side `runnable.voice_config` may additionally supply a zero-arg factory returning a `TurnDetector`, or an instance — instances are `clone()`d per WebSocket session so concurrent clients never share buffers or lifecycle; custom detectors with per-session mutable state should override `TurnDetector.clone()`. |
| `vad_endpointing` | Bool. The **local VAD endpointing fast path**: Silero VAD (auto-downloaded with `timbal[voice]`, MIT) runs on the mic PCM; ~0.2s after you stop speaking, the audio EOU model (Smart Turn) scores the utterance and the EOU probability maps to a variable extra wait — confident-complete commits in ~0.3s of silence instead of waiting the STT provider's ~1.2s debounce; incomplete utterances compute a delay longer than the debounce so the provider commit + HOLD machinery win untouched. Default (key absent) is **auto**: active when the effective turn detector has an audio EOU model (i.e. `"local"` mode with the extra installed), off otherwise. Send `false` to force off, `true` to force on (logs a warning when unavailable). Turns committed by this path have `vad_endpointed: true` in [Turn metrics](#turn-metrics). |

Example — align server with the browser capture rate (only if that rate is supported end-to-end):

```json
{ "sample_rate": 48000 }
```

Default pipeline is tuned for **16 kHz** unless capture, resampling, and this field are aligned.

### Server messages (server → client)

All downlink messages are **text JSON** with a **`type`** field.

| `type`                  | Fields        | Meaning |
|-------------------------|---------------|--------|
| `session_started`       | `session_id`, `playback_acks`, `stt_provider`, `stt_model`, `model`, `turn_detector`, `vad_endpointing` | Voice session is live; safe to show “listening”. `session_id` keys the session's [call recording](#call-recording) files and any platform-side conversation record. `playback_acks: "recommended"` advertises the [playback ack](#playback-acks-client--server) protocol. `stt_provider` is the config id actually in effect (`elevenlabs` / `deepgram-flux` / `deepgram-nova`), not the Python class name. `turn_detector` is the class name of the detector actually in effect (e.g. `LocalAudioTurnDetector`). `vad_endpointing` is a bool: whether the local [VAD endpointing](#config-overrides) fast path actually armed for this session (not merely what was requested). |
| `transcript_partial`    | `text`        | Live STT (may change). |
| `transcript_committed`  | `text`        | Final user transcript for the utterance. |
| `agent_text_delta`      | `text`        | Streaming assistant text (captions / UI). |
| `agent_text_done`       | `text`        | Assistant text for the segment completed. |
| `audio`                 | `data` (base64) | TTS audio: PCM s16le at merged `sample_rate`. Decode and play via Web Audio or equivalent. |
| `metrics`               | `metrics`     | Per-turn latency metrics, sent once per turn after `agent_text_done` (also for interrupted turns). See [Turn metrics](#turn-metrics). |
| `interrupted`           | `heard_text`  | Interrupt / barge-in; stop playback, then send a final playback ack. `heard_text` is the assistant text the user actually heard (use it to fix displayed captions); `null` when unknown, `""` when nothing was played. |
| `error`                 | `message`     | Error description (STT, audio forward, turn, TTS, etc.). |
| `session_transcript`    | `entries`     | Full conversation transcript (sent right before `session_ended`). See [Session transcript](#session-transcript). |
| `session_ended`         | —             | Session ended on the server side. |

Implement handling with a `switch` on `msg.type` (or equivalent).

**After `error`:** The session still shuts down cleanly: you will normally receive **`session_transcript`** then **`session_ended`** (same order as a successful close), unless the TCP/WebSocket connection drops first.

### Session transcript

Right before the server sends `session_ended`, it sends a `session_transcript` message containing the full ordered conversation:

```json
{
  "type": "session_transcript",
  "started_at": 1713099998.500,
  "entries": [
    { "role": "user", "text": "Hola, ¿qué tal?", "timestamp": 1713100000.123, "offset_ms": 1623 },
    { "role": "assistant", "text": "¡Hola! Todo bien, ¿en qué puedo ayudarte?", "timestamp": 1713100002.456, "offset_ms": 3956 },
    { "role": "user", "text": "Cuéntame una historia", "timestamp": 1713100010.789, "offset_ms": 12289 },
    { "role": "assistant", "text": "Había una vez…", "timestamp": 1713100012.012, "offset_ms": 13512 }
  ]
}
```

`started_at` is the session's wall-clock start (Unix seconds). Each entry has:

| Field       | Type   | Description |
|-------------|--------|-------------|
| `role`      | string | `"user"` or `"assistant"`. |
| `text`      | string | Final committed text for that turn. |
| `timestamp` | float  | Unix timestamp (seconds) when the text was committed. |
| `offset_ms` | int    | Milliseconds since `started_at` — lines up with the call recording's playback position. |

This lets the frontend persist the conversation without having to accumulate `transcript_committed` / `agent_text_done` messages itself.

**Audio recording** is available server-side via the `VoiceSession` Python API (`record_audio=True`) but is not sent over the WebSocket (PCM dumps are too large for a single JSON frame). Use the `session.input_audio` / `session.output_audio` properties to access raw PCM bytes after the session closes for server-side storage, conversion, or upload. For a persistent, playable per-call file see [Call recording](#call-recording) below.

### Turn metrics

Once per turn — after `agent_text_done`, and also when the turn is interrupted — the server sends a `metrics` message:

```json
{
  "type": "metrics",
  "metrics": {
    "turn_index": 1,
    "user_text_chars": 24,
    "eou_to_llm_first_token_ms": 312.4,
    "eou_to_tts_first_byte_ms": 587.9,
    "eou_to_first_audio_ms": 587.9,
    "llm_total_ms": 1204.0,
    "tts_total_ms": 1890.2,
    "turn_total_ms": 2101.7,
    "interrupted": false,
    "tts_segments": 3,
    "audio_bytes": 96000,
    "playback_acks_received": true,
    "heard_bytes": null,
    "vad_endpointed": false
  }
}
```

`eou_to_first_audio_ms` is the headline voice latency: user end-of-utterance (committed transcript) to the first TTS byte emitted. Duration fields are `null` when the stage never happened (e.g. an interrupted turn with no audio). `playback_acks_received` says whether the heard position was client-truth (acks) or the wall-clock estimate; interrupted turns set `heard_bytes` to the PCM bytes the user actually heard (`null` on uninterrupted turns). `vad_endpointed` is `true` when the turn's transcript was force-committed by the local [VAD endpointing](#config-overrides) fast path instead of the provider's silence debounce. Server-side, the same metrics are available as `session.metrics` (Python API) and are attached to the run trace as `voice_turn_metrics` on the root span metadata.

### Frontend notes

- **Autoplay policies:** Many browsers require a user gesture before audio output; use a “Start” action to open the socket and resume `AudioContext` as needed.
- **Capture vs server rate:** `getUserMedia` / Web Audio often run at 44.1 kHz or 48 kHz. Resample to the negotiated `sample_rate` before sending, or set `sample_rate` in the first JSON to match what you send (must be supported by the pipeline).
- **Teardown:** Closing the WebSocket ends the session from the client.
- **Privacy:** Audio is sent to the server; use WSS on trusted origins.

### What this is not

- Not the bundled **`GET /voice/`** HTML demo — only the **`/voice/ws`** contract.
- Not REST/SSE for the live voice loop; real-time voice uses this WebSocket (or WebRTC, below).
- The LLM/agent runs on the server; the client only captures audio, plays TTS, and renders text events.

## Voice over WebRTC: `POST /voice/rtc`

The same voice session over WebRTC instead of a WebSocket. Requires the **`timbal[voice]`** extra (which ships aiortc) on the server — without it the route answers **501** with an install hint. The bundled playground has a transport dropdown that exercises this path.

**Why choose it over `/voice/ws`:** Opus instead of raw PCM (~10x less bandwidth), jitter buffering and packet-loss concealment on lossy/mobile networks, and **server-paced playback** — the server sends TTS at real time from its own clock, so it always knows exactly what the caller has heard. Barge-in truncation (`interrupted.heard_text`, memory truncation) is exact with **no playback acks**, and the unspoken tail of an interrupted reply is dropped server-side instead of asking the client to clear buffers.

### Signaling

One HTTP round trip, WHIP-style — no trickle ICE (wait for ICE gathering to complete before posting the offer):

```
POST /voice/rtc
{ "sdp": "<offer sdp>", "type": "offer", "config": { ...same keys as the WS config frame... } }

200 → { "sdp": "<answer sdp>", "type": "answer" }
400 → bad offer / no audio track / runnable is not an Agent
409 → single-session server already served (or is serving) its one session
501 → timbal[voice] extra (aiortc) not installed
```

The offer **must** contain:

- **One audio track** — the mic. Sent Opus-encoded; the server decodes and resamples to the session rate, so the client-side `sample_rate` config key is irrelevant on this transport. Keep `echoCancellation: true` in `getUserMedia` constraints.
- **A data channel** (any label) — created by the client so its SCTP m-line rides the offer. All non-audio session events arrive here as JSON: the exact payloads of the WS protocol, except there are **no `audio` messages** (TTS is a real audio track the browser plays natively) and **no `playback` acks** (`session_started.playback_acks` is `"native"`, and `transport` is `"webrtc"`).

The server answers with the TTS audio track on the same m-line. Client teardown = close the peer connection; server teardown (session end/error) closes the connection and fires `session_ended` on the data channel first.

### ICE configuration

| Env var | Meaning |
|---|---|
| `TIMBAL_STUN_URL` | STUN server; defaults to `stun:stun.l.google.com:19302`. Set to empty to disable (loopback/LAN). |
| `TIMBAL_TURN_URL` | Optional TURN server for clients behind symmetric NATs. |
| `TIMBAL_TURN_USERNAME` / `TIMBAL_TURN_PASSWORD` | TURN credentials. |
| `TIMBAL_VOICE_RTC_FORCE_RELAY` | `1` → relay-only ICE: allocate on the TURN server above, skip STUN, and strip host/srflx candidates from the answer SDP. For servers on private subnets (serverless boxes) where every non-relay candidate is unreachable dead weight that slows the browser's ICE convergence. Requires `TIMBAL_TURN_URL` — without it (or if the TURN allocation yields no relay candidate) the server logs an error and answers with all candidates instead of an unconnectable SDP. |

## Telephony: `/voice/twilio` and `/voice/telnyx`

Answer real phone calls with the same voice session. Twilio and Telnyx stream call audio (G.711 μ-law, 8kHz mono) over a WebSocket they open toward the server; a shared bridge decodes/resamples into the session and sends TTS back the same way. Requires **`timbal[voice]`** (av for resampling).

Routes per provider:

- `POST /voice/twilio/incoming` / `POST /voice/telnyx/incoming` — the phone number's voice webhook. Answers with TwiML/TeXML that connects the call to the media WebSocket below (URL derived from `X-Forwarded-Proto`/`Host`, so it works behind a TLS proxy or an ngrok tunnel).
- `WS /voice/twilio/stream` / `WS /voice/telnyx/stream` — the bidirectional media stream.

**Provider setup:**

- **Twilio:** point the number's *A call comes in* webhook (HTTP POST) at `https://<host>/voice/twilio/incoming`. Set `TWILIO_AUTH_TOKEN` so webhook signatures (`X-Twilio-Signature`, HMAC-SHA1) are enforced — without it requests are accepted with a warning.
- **Telnyx:** create a TeXML application whose voice webhook is `https://<host>/voice/telnyx/incoming` and assign the number to it. Set `TELNYX_PUBLIC_KEY` (portal → API keys → public key) to enforce Ed25519 webhook signatures. The returned TeXML pins `bidirectionalMode="rtp"` and PCMU both ways.

**Bridge semantics:**

- Caller audio: μ-law 8kHz → PCM16 → resampled to the session rate (default 16kHz). No client `sample_rate` config applies; the line format is fixed by the carrier.
- TTS audio: resampled to 8kHz, μ-law encoded, sent as `media` frames batched to ≥20ms (Telnyx's RTP minimum). A `mark` frame carrying the cumulative byte count follows each media frame; its echo is the playback ack, so barge-in truncation (`heard_text`, memory) works exactly like the browser transports.
- Barge-in sends the provider's `clear` message, dropping their buffered audio immediately. Marks echoed *by* the clear (audio that never played) are ignored.
- Custom `<Parameter>` values on the stream may override allowlisted session config (`turn_detector`, `stt_provider`, `stt_model`, `tts_model`, `model`, `language`, `voice`) — the webhook's TwiML already forwards `from`/`to`/`call_sid` so recordings are call-addressable.
- DTMF frames are logged and otherwise ignored (no keypad actions yet). Outbound calls are not implemented yet either — these routes cover inbound.

Single-session mode applies: a phone call counts as the process's one session.

## Single-session lifetime (serverless voice boxes)

`TIMBAL_VOICE_SINGLE_SESSION=1` makes the server process serve **exactly one voice session and then exit 0** — for deployments that spawn one process per call and reap the box on process exit (the platform cannot see a WebRTC call: media flows browser ↔ TURN ↔ box, so the process must own its lifetime). Applies to both transports; whichever session arrives first (WS or RTC) owns the process.

- **Exit on session end.** When the one session ends (peer connection `failed`/`closed`, WebSocket closed, or the session ends server-side), the process finalizes — including waiting for the [platform recording push](#call-recording) to finish, since the process is the only thing holding that data — and exits 0.
- **Exit if nobody ever connects.** If no media connection is established within `TIMBAL_VOICE_IDLE_EXIT_SECS` (default `60`) of server start, exit 0. The window runs boot → *media established*, so an offer whose ICE never completes also exits after the window.
- **Refuse a second session.** While a session is live or after one has been served: `POST /voice/rtc` → **409**, `/voice/ws` → close **1008**. An offer rejected with 400 (bad SDP, no audio track) never became a session and does not consume the slot.

All lifetime exits are code 0; env is read at server start (never at import time), so CRIU-restored processes with late-injected env behave identically.

## Ambient background audio

Looped background sound so calls don't happen in sterile silence (server-side only — clients cannot switch it on, off, or point it at a file):

```python
agent.voice_config = {
    "ambient": {
        "source": "office",   # preset name or path to an audio file
        "volume": 0.2,        # 0.0–1.0
    }
}
```

Or via env: `TIMBAL_VOICE_AMBIENT_SOURCE` / `TIMBAL_VOICE_AMBIENT_VOLUME`. Default is off.

Presets (`GET /voice/ambience` lists them): `office` (quiet room tone with typing), `call-center` (typing, printers, ringing phones), `cafe` (low-passed restaurant murmur), `city` (night traffic), `typing` (close keyboard typing). Preset tracks are not bundled — the server downloads them from the Timbal CDN on first use, verifies a pinned sha256, and caches them in `~/.cache/timbal/ambience/` (override the CDN with `TIMBAL_AMBIENCE_BASE_URL`). Custom tracks should be voiceless or unintelligible murmur — intelligible chatter that leaks back through the caller's mic gets transcribed — and loop-clean (the presets bake a crossfade into the seam; see `timbal/voice/ambience/ATTRIBUTIONS.md`).

**How it plays:** `session_started` carries the resolved config as `ambient: {source, volume} | null`; the client fetches the track from `GET /voice/ambience/current` and loops it locally (the playground page does this through WebAudio, with a local volume slider, and a picker for any preset or local override). Nothing is mixed server-side yet, which also means recordings don't contain ambience. Keep volume low (~0.2–0.3): browser echo cancellation handles page-played audio, but a phone speaker can still leak it into the mic.

## Tool-call fillers

Tools mean dead air: the caller asks something, the agent goes quiet for seconds while a lookup runs. Fillers mask it with a short spoken phrase ("One sec, let me check that…") generated by a one-shot LLM call — not hardcoded, so it's contextual and automatically in the caller's language (the generator sees the user's last utterance and the tool name):

```python
agent.voice_config = {
    "filler": {},  # enable with defaults
}
# or tuned:
agent.voice_config = {
    "filler": {
        "system_prompt": "…",              # replaces the built-in generator prompt
        "model": "openai/gpt-4o-mini",     # None → the session's LLM; pick something fast
        "delay_secs": 1.0,                 # tools that finish sooner get no filler
        "repeat_secs": 10.0,               # follow-up on prolonged silence; None (default) = one filler max
        "max_per_turn": 3,                 # hard cap when repeating
    }
}
```

Or via env: `TIMBAL_VOICE_FILLER=1` (defaults), `TIMBAL_VOICE_FILLER_SYSTEM_PROMPT`, `TIMBAL_VOICE_FILLER_MODEL`, `TIMBAL_VOICE_FILLER_DELAY_SECS`, `TIMBAL_VOICE_FILLER_REPEAT_SECS`. Default is off.

`filler` is also client-settable (session config frame): the client's dict is deep-merged over the server's, so `{"filler": {"delay_secs": 0.3}}` keeps a server-set custom prompt, and `{"filler": {"enabled": false}}` switches a server-default filler off for that session. The playground exposes this — a Filler speech on/off/server-default picker plus prompt and delay fields, applied on the next Start.

**Mechanics:** generation starts the moment a tool call is detected and overlaps the grace delay; the phrase is only spoken if the tool is still running after `delay_secs` and nothing else has been said this turn. One filler per turn regardless of how many tools run — unless `repeat_secs` is set, in which case a short follow-up ("still on it…") fires after that much continued silence, up to `max_per_turn`; the generator is told what it already said so follow-ups don't repeat themselves. Generation failures are silent — dead air is the status quo, never an error the caller hears. The phrase reaches clients as `{"type": "filler", "text": …}` (the playground renders it as a dimmed italic bubble), lands in the session transcript flagged `filler: true`, and is audible in call recordings — but it never enters the agent's memory or the reply text, and a barge-in during the filler counts as "the user heard nothing" of the reply. Turn metrics carry `filler_spoken` so you can tell filler-first audio latency from real reply latency.

The playground also has a local **thinking sound** toggle — a quiet keyboard loop played between "the agent started a tool" and the first reply audio. Browser-only, off by default, independent of the server-side filler.

## Call recording

Server-side, transport-agnostic (WS and WebRTC share the wiring): every session writes **one playable MP3** — the call as heard, on the call timeline — plus a **JSON manifest** with the timestamped transcript and per-turn latency metrics. This is the data an ElevenLabs-style conversation review UI needs: audio player, transcript entries at `offset_ms`, latency chips per turn.

**Enable** (server-side only — client config frames cannot switch recording on or off):

```python
# via the runnable, full control:
agent.voice_config = {
    "recording": {
        "dir": "recordings",           # required — files land here
        "layout": "mixed",             # "mixed" (default): mono, both voices summed
                                       # "split": stereo, caller left / agent right
        "bitrate_kbps": 32,            # MP3 bitrate (32 kbps mono ≈ 0.25 MB/min)
        "on_saved": my_async_hook,     # optional: async (RecordingResult) -> None
    },
}
```

or via env (the platform's config surface — all read **per session**, never cached at boot, so late-injected env e.g. after a CRIU restore still applies):

| Env var | Meaning |
|---|---|
| `TIMBAL_VOICE_RECORDING_DIR` | Enables recording; files land here. |
| `TIMBAL_VOICE_RECORDING_LAYOUT` | `mixed` (default) or `split`. |
| `TIMBAL_VOICE_RECORDING_BITRATE_KBPS` | MP3 bitrate, default `32`. |
| `TIMBAL_VOICE_RECORDING_UPLOAD` | `platform` → push files to the platform API after each call (see below). |

Keys in `voice_config["recording"]` win over env, per key. Requires the `timbal[voice]` extra (av + numpy); without it the server logs a warning and runs the call unrecorded.

When `TIMBAL_ORG_ID` / `TIMBAL_PROJECT_ID` / `TIMBAL_PROJECT_ENV_ID` / `TIMBAL_APP_ID` / `TIMBAL_PROJECT_REV` are present in env, they are stamped into the manifest `meta` (as `org_id`, `project_id`, ...) so the files are self-describing for ingest.

Per session, keyed by the `session_id` from `session_started`:

- **`{session_id}.mp3`** — mic and TTS mixed on the real call timeline. TTS synthesizes faster than real time, so the agent side is paced by the mic clock; on barge-in the **unheard tail is dropped** exactly like `interrupted.heard_text` truncates the text (exact on WebRTC, ack/estimate-based on WS). Encoded progressively — a crashed process still leaves a playable file.
- **`{session_id}.json`** — manifest: `session_id`, `started_at`/`ended_at`, resolved config (`transport`, `model`, `stt_provider`, ...), `transcript` entries with `offset_ms`, `turns` (the full `TurnMetrics` per turn), and the audio descriptor (`layout`, `sample_rate`, `bitrate_kbps`, `duration_secs`). Written **atomically** (tmp + rename) and always *after* the MP3 is finalized — "manifest exists" reliably means "recording complete"; an MP3 without a manifest is a crashed call, playable up to the crash.

`on_saved` fires after both files are written (e.g. upload to platform storage and delete the local copy); its failures are logged, never crash the session.

**Platform push** (`TIMBAL_VOICE_RECORDING_UPLOAD=platform`): after each call, a background task PUTs both files as multipart to `{host}/orgs/{org}/projects/{project}/sessions/{session_id}` — host, Bearer credential and org/project resolved by the standard `resolve_platform_config` (env / ~/.timbal), fresh per session. 2xx → local files deleted; 4xx → keep files, log, no retry; 5xx/429/network → exponential backoff (1s base, ×5, cap 5 min, ~1h budget). Uploads never block session teardown, and files are only deleted after a confirmed 2xx — a crash mid-upload leaves them intact for re-ingest. A user-provided `on_saved` in `voice_config` takes precedence over the platform push.
