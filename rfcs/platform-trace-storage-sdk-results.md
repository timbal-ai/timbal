# Trace saving (monolith#553): SDK status and test results

SDK: timbal `feat/bots` @ `43de91ad`. Not released yet: none of this reaches users until it ships and agents upgrade.
Tested against `api.dev.timbal.ai` (#563 with `TIMBAL_TRACE_DELTAS_ENABLED=1`, plus `01203842`) and `api.timbal.ai` (fallback only).
Org 1, app 5471 (`latex-assistant`). Every run used timbal's `TestModel`, so there were no model calls.

## What the SDK does on the wire now

**Format negotiation**
- **First PATCH:** a process's first PATCH to a host is a whole trace. The response headers decide the rest: `Accept-Patch` turns on deltas, `Accept-Encoding: gzip` turns on gzip.
- **Delta saves:** finished spans the platform hasn't acknowledged yet, plus the root.
- **The root:** sent whole until the platform acknowledges a whole root carrying `memory`, then as a header you merge into it. A process that dies mid-turn still leaves its prompt, history and session data stored.

**Snapshot ordering**
- **Every PATCH carries `X-Timbal-Snapshot`.**
- **Stale resend:** an `Applied: false` whose stored snapshot is newer and wasn't written by this process means another process wrote the run. The SDK resends once, numbered above the stored snapshot.

**Fallbacks** (each one turns the format off for that host)
- **Delta or gzip answered with `400`, `415` or `422`:** resent whole and uncompressed.
- **Delta answered with `204` and no echo:** handled the same way.

**Over the 50 MB limit (`413`)**
- **No format fallback, never resent as is.** From then on, that run's saves go whole with long strings cut: 64 KB, then 4 KB, then 512 characters, until the body is under 40 MB.
- **Marker:** the root gets `metadata.trace_truncated = {max_chars, strings_cut}`.
- **A second `413` is final.**

**Reads**
- **`?select=resume`** once the host advertises it.
- **The parent:** taken from `parent_idempotency_key` when present, else `parent_id`.

**Connections:** trace PATCHes and GETs now reuse one keep-alive connection pool per process event loop. You'll see far fewer TLS handshakes from the SDK.

**When saves happen**
- **`ChatSession` turns:** buffered, at most one background save per run every 0.5 s. Before running any tool not marked read-only, the run is stored and the SDK waits for that save.
- **Other runs:** unchanged (`sync`): every finished span is saved and awaited, about 2 PATCHes per tool step.

**Failure handling**
- **Shutdown:** flushes owed saves, bounded by 5 s.
- **A failed run end:** resent in the background for up to 10 minutes, on transient errors only (429, 5xx, timeouts).

## Results on dev (all passed)

| Scenario | Result |
|---|---|
| `scripts/trace_deltas_e2e.py` | Every check passed. Rows check skipped: no DB access on our side, but you confirmed rows are gone. |
| Multi-turn `ChatSession` | Whole first, then deltas. Every turn stored `success`, and the last turn's memory has all messages. |
| Approval resumed in another process | Resumed run stored complete and chained. The SDK reads the parent back via `parent_idempotency_key`. |
| Writer with its clock 1 h behind, same run | First save `Applied: false`, resent numbered above, landed. No span lost. |
| Cancel mid-turn | Terminal save landed: `cancelled`. |
| SIGTERM to `timbal.server` mid-turn | Shutdown flush landed: `cancelled`. |
| Nested agent parked on an approval | Projection has only the root and the nested agent, with `X-Timbal-Snapshot`. Resume completed. |
| Background child finishing after the end | Late output merged by `call_id`. No span lost, still `success`. |
| 60 MB tool output (`413`) | One 413, then cut-down whole saves (~1–2 KB gzipped). Stored `success`, all spans, marked `trace_truncated`. Before this change the run stayed `running`. |
| Tools of 33 and 40 min (past `RUN_TIMEOUT_MINUTES=30`) | Run read `timeout` once the 30 idle minutes passed; when the tool finished, the next save brought it back to `success` with the full trace (8/8). Seed-back path confirmed live. |
| Tool of 12 min | Complete (8/8 spans), `success`. |

## Fallback against prod (`api.timbal.ai`, all passed)

- **Default:** whole saves only. Prod advertises gzip, and whole saves echo `X-Timbal-Snapshot`.
- **Forced deltas, new run:** `400` (the body has no `trace`). Resent whole, deltas off, 8/8 spans stored.
- **Forced deltas, existing run:** `204` with no echo. Resent whole, deltas off, nothing lost.
- **Forced gzip:** works.

## Performance (dev, measured from a laptop over the internet)

20 tool steps of about 4 KB each, giving a 1.07 MB trace. No model calls, so the run time is the time spent waiting on saves.

| Setup | Uploaded per run | PATCH p50 / p95 | Run time |
|---|---|---|---|
| Whole, uncompressed (today's SDK) | 16.9 MB | 638 / 1,143 ms | 27.9 s |
| Whole, gzip | 3.5 MB | 428 / 677 ms | 18.8 s |
| Deltas + gzip, new connection per save | 0.30 MB | 287 / 371 ms | 11.8 s |
| Deltas + gzip, pooled connection | 0.30 MB | 85 / 173 ms | 4.5 s |
| Deltas + gzip, pooled, buffered (chat turns) | 0.29 MB | 89 / 216 ms | 2.4 s (21 PATCHes instead of 42) |

**Reads:** a full GET returns 1,074 KB in about 700 ms; `?select=resume` returns 86 KB in about 365–420 ms.

On a reused connection the server answers in roughly 75–110 ms. Most of the old ~290 ms per request was TLS setup on the client.

## Open points

1. **A live run reads `timeout` during long tools.** Once a tool idles past `RUN_TIMEOUT_MINUTES`, the run shows `timeout` until its next save. The data is safe, but the status is wrong in that window. The SDK could send a small keep-alive delta (just the root header) every few minutes while a run is in flight. That needs no backend change. Do you want it, or would you rather handle it server-side?
2. **Truncated traces:** `metadata.trace_truncated` on the root marks one. The UI may want to show it.
3. **Prod:** deltas are still behind the flag. Gzip and snapshot ordering are already live, and the SDK works with both.
4. **Release:** the SDK changes need a timbal release (or the trace-saving commits split out of `feat/bots`) before anyone benefits.

<details><summary>Run ids (app 5471, Nov 8)</summary>

- 12 min: `06af070fe1d277f580000f5f5ba4c721` (row `4270390`)
- 33 min: `06af084fddf17f798000e9f18ecec8e4` (row `4272962`)
- 40 min: `06af0859bd3e74048000a3d434ab6ca3` (row `4273088`)
- 413 runs before the SDK fix: rows `4270443`, `4270458`
- Everything else ran between 13:50 and 16:03 CET.
</details>
