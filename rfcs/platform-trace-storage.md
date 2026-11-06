# Platform trace storage: smaller saves, smaller traces

Status: proposal (v2). Owners: framework (timbal) — SDK side; platform API (monolith) — server side.
Reviewed against monolith `origin/master` `1d3d6b3e` (2026-11-06).

## Problem

Framework agents store runs through `PlatformTracingProvider`: a `PATCH /orgs/{org}/apps/{app}/runs/{run_id}`
with the **whole trace** every time a span finishes (`durability="sync"`, the default). Per request
`handlers/orgs/apps/runs/edit.rs` runs the auth gate, `parse_trace` over the whole trace (status, timing,
cost; a read-replica query), replaces the JSONB `trace` column of `OrgsAppsVersionsRuns`, and on the
terminal write runs chat-session ingest and `record_costs`. Composer writes the same shape to
`PATCH /orgs/{org}/traces/{run_id}` (`Traces`). Every turn's resume `GET`s the parent's whole trace.

Measured (one run, sync saves, offline model, 8 tools, `/tmp/e2e/bench_payloads.py`):

| Run | Saves | Final trace | LLM spans' input history | LLM spans' tool schemas | Uploaded per run | Gzipped |
|---|---|---|---|---|---|---|
| 40 steps, 2 000-char tool outputs | 82 | 2.3 MB | **82 %** | 7 % | 72.6 MB | 1.52 MB |
| 40 steps, 200-char tool outputs | 82 | 0.7 MB | **63 %** | 21 % | 25.3 MB | 0.64 MB |
| 100 steps, 2 000-char tool outputs | 202 | 12.8 MB | **92 %** | 3 % | 927 MB | 16.4 MB |

Two multiplicative costs:

1. **Every save resends the trace** (quadratic in steps): fixed by span deltas.
2. **Every LLM span repeats its whole input** (messages + tool schemas — quadratic again, and across turns:
   the next turn's first call resends the conversation): fixed by content-addressing messages and tool
   lists. This one also inflates storage and every resume `GET`, which deltas alone do not touch.

| Design (40 steps, 2 000-char outputs) | Uploaded | Gzipped | Largest body | Stored per run |
|---|---|---|---|---|
| Full snapshots (today) | 72.6 MB | 1.52 MB | 2.3 MB | 2.3 MB |
| Span deltas | 2.4 MB | 0.07 MB | 99 KB | 2.3 MB |
| Full snapshots + content-addressed inputs | 11.1 MB | 0.31 MB | 261 KB | ~0.3 MB |
| Span deltas + content-addressed inputs | 0.28 MB | 0.03 MB | 94 KB | ~0.3 MB |

## A — Request compression (hours)

- Server: `tower_http::decompression::RequestDecompressionLayer` (gzip, zstd) on the router; advertise it
  with `Accept-Encoding: gzip, zstd` on responses (RFC 7694 — how a server tells clients which request
  encodings it takes).
- SDK (done): `PlatformTracingProvider` compresses `PATCH` bodies once a response from that host
  advertised it (or `compress=True` / `TIMBAL_TRACE_GZIP=1`); a `415` turns it off for the host and the
  save is retried uncompressed.

Upload 48× smaller on today's snapshots. Server work unchanged.

## B — Content-addressed LLM inputs

An LLM span's `input.messages` and `input.tools` become lists of hashes; the bodies are stored once per
thread.

```json
"input": {"model": "...", "system_prompt": {"$ref": "sha256:..."},
          "messages": [{"$ref": "sha256:a1..."}, {"$ref": "sha256:b2..."}],
          "tools": {"$ref": "sha256:t9..."}, "max_tokens": 1024}
```

- Hash = sha256 of the canonical JSON (sorted keys, no whitespace) of the message dict as sent to the
  router. Not a pointer into agent memory: compaction rewrites memory in place, attachment limits swap in
  stand-ins per request, steering and tool search change messages and tools per step — the hash names
  exactly what that call saw.
- Blobs travel with the save that first uses them (`"blobs": {"sha256:a1...": {...}}`), deduplicated
  client-side per thread (the SDK remembers which hashes the server acknowledged) and server-side by
  primary key.
- Server: `TraceBlobs(org_id, session_id, hash, body jsonb, PRIMARY KEY (org_id, session_id, hash))`,
  keyed by thread (`session_id` = root run id, already on `Traces`; `group_id`'s root key on app runs), so
  turn N+1 re-references turn N's history instead of storing it again. Retention follows the thread's.
- Readers: the UI's LLM-call view and evals/exports resolve `$ref`s (`GET .../runs/{id}?resolve=refs` can
  inline them for old clients). The framework's own resume reads agent memory, not LLM inputs: unaffected.
- `format_version` bumps (`Traces.format_version` exists for exactly this); `parse_trace` reads only the
  root span and usage, unaffected.

## C — Span deltas

```text
PATCH /orgs/{org}/apps/{app}/runs/{run_id}        and   PATCH /orgs/{org}/traces/{run_id}
Content-Type: application/vnd.timbal.trace-delta+json
{"parent_id": ..., "rev": ..., "snapshot": 17, "ended": false,
 "root": {"call_id": ..., "path": ..., "t0": ..., "parent_call_id": null},
 "spans": [ <finished spans the server has not acknowledged> ],
 "blobs": { <B, once both ship> }}
```

```sql
CREATE TABLE public."TraceSpans" (
    org_id   bigint NOT NULL,
    run_id   text   NOT NULL,   -- Timbal context.id = Traces.run_id = OrgsAppsVersionsRuns.idempotency_key
    call_id  text   NOT NULL,
    seq      int    NOT NULL,   -- first-seen order (the trace's order)
    snapshot bigint NOT NULL,   -- SDK save counter: an older delta never overwrites a newer span
    span     jsonb  NOT NULL,
    PRIMARY KEY (org_id, run_id, call_id)
);
-- ON CONFLICT (org_id, run_id, call_id) DO UPDATE SET span = EXCLUDED.span, snapshot = EXCLUDED.snapshot
--   WHERE "TraceSpans".snapshot < EXCLUDED.snapshot
```

- Shared by both run tables (keyed by the producer run id), so it works on today's app-run path and
  survives app runs moving to `Traces`.
- `ended: false`: upsert spans; create the run row as `running` on the first delta, else bump
  `updated_at` (the liveness signal `services/traces.rs` already describes). No `parse_trace`, costs,
  metrics or ingest.
- `ended: true` (root finished — includes runs parked on approval): upsert, assemble `trace` / `data`
  from the rows in `seq` order, run today's finalize path unchanged (parse, metrics, chat ingest, costs,
  titles), delete the rows.
- `GET` of a run that never ended (its process died) assembles from the rows: what crash recovery reads.
- Mid-run saves carry finished spans only (immutable once finished) plus a root header without memory.
  The full root goes with `ended: true`. A span changed after it finished (a background child's real
  output after the root ended) is resent with a newer `snapshot`.
- Negotiation: responses carry `Accept-Patch: application/vnd.timbal.trace-delta+json` (RFC 5789); the SDK
  switches per host once seen, sends full traces otherwise. Old SDKs keep working unchanged.

Effect: a save becomes one or two spans; parsing and finalization happen once per run; `sync` becomes
affordable, so crash recovery sees every finished tool call.

## Chat session state (after C)

Durable `ChatSession` needs a small mutable record (queue, wakes, head run, lease). Not a new table: make
`ChatSession.id` the thread's root run id (= `Traces.session_id` = `Sessions.session_id`) and keep the
runtime state on the `Sessions` envelope (`runtime jsonb`, `runtime_version`, lease columns), or wherever
the platform prefers — it must not become a fourth "session" (see SESSIONS_PLAN "Naming collisions").

## Order

| Step | Who | Size (estimate) |
|---|---|---|
| A server: decompression + `Accept-Encoding` | platform | hours |
| A SDK: negotiated gzip | framework | done |
| C SDK: delta sender, fallback to full | framework | in progress |
| C server: `TraceSpans`, both PATCH paths, GET assembly | platform | 3–5 days |
| B SDK + server + UI/evals `$ref` resolution | both | 1–2 weeks |
| Chat session runtime state | platform decision | after C |

## Open questions

- Where do platform-hosted agents run, and is local disk kept across deploys?
- Should `durability="buffered"` be the default for every platform run, not only `ChatSession` turns?
- B: hash the provider-ready message (what was sent) or the Timbal `Message` dump? Proposal: the Timbal
  dump the LLM span already stores today, so the UI renders what it renders now.
