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
   If the first text frame is **not valid JSON**, the server logs a warning and continues with **empty** client overrides (defaults still apply).
3. After that, stream **microphone audio** until the socket is closed.

If the client needs to set `sample_rate` or `language`, the **first** frame should be that JSON text message (unless they are fine with defaults and send binary first).

### Audio uplink (client → server)

- **Format:** mono **PCM**, **16-bit signed little-endian** (`pcm_s16le`), at **`sample_rate`** from the merged config (default **16000** Hz).
- **Transport (either):**
  - **Binary frames:** raw PCM bytes (chunk duration is up to the client; ~100 ms chunks are a reasonable default).
  - **Text JSON:** `{ "type": "audio", "data": "<base64-encoded PCM bytes>" }` on any later message.

If the **first** message is binary, it is queued as the first audio chunk (no separate config message).

### Config overrides

Optional **first** text frame: a JSON object merged on top of `app.state.voice_config`, which is built at startup from environment defaults and optional `runnable.voice_config` on the loaded agent (`http` lifespan).

Only send keys you need; omitted keys keep server defaults.

| Key           | Description |
|---------------|-------------|
| `stt_model`   | Speech-to-text model id (ElevenLabs realtime). |
| `tts_model`   | Text-to-speech model id. |
| `voice`       | ElevenLabs voice id string. |
| `language`    | e.g. `"es"`. |
| `sample_rate` | Hz; STT/TTS audio use this unless extended later. |
| `encoding`    | Default `"pcm_s16le"`. |
| `stt_extra`   | Object merged with default STT options (e.g. VAD). |
| `tts_extra`   | Object merged with default TTS options. |

Example — align server with the browser capture rate (only if that rate is supported end-to-end):

```json
{ "sample_rate": 48000 }
```

Default pipeline is tuned for **16 kHz** unless capture, resampling, and this field are aligned.

### Server messages (server → client)

All downlink messages are **text JSON** with a **`type`** field.

| `type`                  | Fields        | Meaning |
|-------------------------|---------------|--------|
| `session_started`       | —             | Voice session is live; safe to show “listening”. |
| `transcript_partial`    | `text`        | Live STT (may change). |
| `transcript_committed`  | `text`        | Final user transcript for the utterance. |
| `agent_text_delta`      | `text`        | Streaming assistant text (captions / UI). |
| `agent_text_done`       | `text`        | Assistant text for the segment completed. |
| `audio`                 | `data` (base64) | TTS audio: PCM s16le at merged `sample_rate`. Decode and play via Web Audio or equivalent. |
| `interrupted`           | —             | Interrupt / barge-in; stop playback and reset “current assistant” UI if needed. |
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
  "entries": [
    { "role": "user", "text": "Hola, ¿qué tal?", "timestamp": 1713100000.123 },
    { "role": "assistant", "text": "¡Hola! Todo bien, ¿en qué puedo ayudarte?", "timestamp": 1713100002.456 },
    { "role": "user", "text": "Cuéntame una historia", "timestamp": 1713100010.789 },
    { "role": "assistant", "text": "Había una vez…", "timestamp": 1713100012.012 }
  ]
}
```

Each entry has:

| Field       | Type   | Description |
|-------------|--------|-------------|
| `role`      | string | `"user"` or `"assistant"`. |
| `text`      | string | Final committed text for that turn. |
| `timestamp` | float  | Unix timestamp (seconds) when the text was committed. |

This lets the frontend persist the conversation without having to accumulate `transcript_committed` / `agent_text_done` messages itself.

**Audio recording** is available server-side via the `VoiceSession` Python API (`record_audio=True`) but is not sent over the WebSocket (PCM dumps are too large for a single JSON frame). Use the `session.input_audio` / `session.output_audio` properties to access raw PCM bytes after the session closes for server-side storage, conversion, or upload.

### Frontend notes

- **Autoplay policies:** Many browsers require a user gesture before audio output; use a “Start” action to open the socket and resume `AudioContext` as needed.
- **Capture vs server rate:** `getUserMedia` / Web Audio often run at 44.1 kHz or 48 kHz. Resample to the negotiated `sample_rate` before sending, or set `sample_rate` in the first JSON to match what you send (must be supported by the pipeline).
- **Teardown:** Closing the WebSocket ends the session from the client.
- **Privacy:** Audio is sent to the server; use WSS on trusted origins.

### What this is not

- Not the bundled **`GET /voice/`** HTML demo — only the **`/voice/ws`** contract.
- Not REST/SSE for the live voice loop; real-time voice uses this WebSocket.
- The LLM/agent runs on the server; the client only captures audio, plays TTS, and renders text events.
