# Timbal HTTP server

FastAPI app factory and CLI live in `http.py` (`python -m timbal.server`, or `run_server_cli`). The runnable is selected via `TIMBAL_RUNNABLE` (same object used for `/run`).

This README documents how a **custom frontend** integrates with the **voice agent** over WebSocket (not the bundled `voice.html` playground).

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
| `stt_extra`   | Object merged with default STT options (e.g. VAD). |
| `tts_extra`   | Object merged with default TTS options. |
| `turn_detector` | A mode name: `"heuristic"` (default), `"provider"` (trust STT/realtime endpointing), `"local"` (audio EOU — auto-loads the Smart Turn v3 ONNX model when `timbal[voice]` is installed, heuristic degradation otherwise; set `TIMBAL_SMART_TURN_CHECKPOINT=int8` to trade ~1pp accuracy for ~2x faster inference), `"lexical"` (punctuation HOLD), or `"raw"` (debug: no silence/noise/echo filtering — every STT commit becomes a turn, including the agent's own speech leaking through the mic; never use in production). The client JSON may only send a mode name string (the playground page has a dropdown for this); it takes precedence over the server value for that session. Server-side `runnable.voice_config` may additionally supply a zero-arg factory returning a `TurnDetector`, or an instance — instances are `clone()`d per WebSocket session so concurrent clients never share buffers or lifecycle; custom detectors with per-session mutable state should override `TurnDetector.clone()`. |
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
