# Platform trace storage: smaller saves, smaller traces


Status: proposal (v3). Owners: framework (timbal) — SDK side; platform API (monolith) — server side.
Reviewed against monolith `origin/master` `046ce015` (2026-11-06).

**Changes since v2:** A is scoped to the two trace PATCH routes, and the SDK falls back on `400` as well as
`415`. C gains a run-level snapshot high-water mark and a row lock (closing the "span changes after the root
ended" gap), client-assigned `seq` (root at 0), `rev` / `metadata` in the envelope, per-span scrubbing, and a
storage marker for in-flight runs. B hashes RFC 8785 canonical JSON verified server-side and gets explicit
retention. Chat runtime state moves to a 1:1 side table. Two open questions are resolved.

## Problem

Framework agents store runs through `PlatformTracingProvider`: a `PATCH /orgs/{org}/apps/{app}/runs/{run_id}`
with the **whole trace** every time a span finishes (`durability="sync"`, the default). Per request
`handlers/orgs/apps/runs/edit.rs` runs the auth gate, `parse_trace` over the whole trace (status, timing,
cost; a read-replica query), replaces the JSONB `trace` column of `OrgsAppsVersionsRuns`, and on the
terminal write runs chat-session ingest and `record_costs`. Composer writes the same shape to
`PATCH /orgs/{org}/traces/{run_id}` (`Traces`). Every turn's resume `GET`s the parent's whole trace.

Measured (one run, sync saves, offline model, 8 tools; benchmark script available on request):

| Run | Saves | Final trace | LLM spans' input history | LLM spans' tool schemas | Uploaded per run | Gzipped |
|---|---|---|---|---|---|---|
| 40 steps, 2 000-char tool outputs | 82 | 2.3 MB | **82 %** | 7 % | 72.6 MB | 1.52 MB |
| 40 steps, 200-char tool outputs | 82 | 0.7 MB | **63 %** | 21 % | 25.3 MB | 0.64 MB |
| 100 steps, 2 000-char tool outputs | 202 | 12.8 MB | **92 %** | 3 % | 927 MB | 16.4 MB |

The upload is not the only cost. Postgres has no partial update of a TOASTed `jsonb` value, so every PATCH
rewrites the whole `trace` / `data` value: "uploaded per run" is also roughly WAL written per run (927 MB for
the 100-step run), plus TOAST churn and replica lag. Compression (A) does nothing for that; span deltas (C) do.
It also rules out merging deltas in place (`trace = trace || $delta`), which still rewrites the whole value.

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

- Server: `tower_http::decompression::RequestDecompressionLayer` (gzip, zstd) on the two trace PATCH routes
  only, with `Accept-Encoding: gzip, zstd` on their responses, including the `204` for `Prefer: return=minimal`
  (RFC 7694 — how a server tells clients which request encodings it takes).
  - **Not on the global router.** Git smart HTTP already decodes gzip itself (`smart_http.rs`) on routes with
    `DefaultBodyLimit::disable()`, and the wake proxy forwards raw bodies with no limit. A global layer would
    decompress those before the handler's own streaming and limits, with no bomb guard.
  - The global 50 MB `DefaultBodyLimit` is enforced by the extractor on the *decompressed* body, which is the
    bomb guard on these routes. A compressed body that expands past it gets `413`.

  ```toml
  tower-http = { version = "0.6.8", features = ["trace", "cors", "decompression-gzip", "decompression-zstd", "set-header"] }
  ```

  ```rust
  let trace_ingest = tower::ServiceBuilder::new()
      .layer(SetResponseHeaderLayer::overriding(
          header::ACCEPT_ENCODING,
          HeaderValue::from_static("gzip, zstd"),
      ))
      .layer(RequestDecompressionLayer::new());
  // .route("/orgs/{org_id}/apps/{app_id}/runs/{run_id}", patch(edit).layer(trace_ingest.clone()))
  // .route("/orgs/{org_id}/traces/{run_id}",             patch(edit).layer(trace_ingest.clone()))
  ```

- SDK (done): `PlatformTracingProvider` compresses `PATCH` bodies once a response from that host
  advertised it (or `compress=True` / `TIMBAL_TRACE_GZIP=1`). If the host rejects a compressed save, the SDK
  turns compression off for that host and retries uncompressed. The rejection is a `415` from a server with the
  layer, but a `400` from today's server: without the layer, axum's `Json` extractor fails to parse the gzip
  bytes. **The SDK must treat both as the signal.** Otherwise forced compression drops saves on old servers.

Upload 48× smaller on today's snapshots. Server-side database work is unchanged.

## B — Content-addressed LLM inputs

An LLM span's `input.messages` and `input.tools` become lists of hashes; the bodies are stored once per
thread.

```json
"input": {"model": "...", "system_prompt": {"$ref": "sha256:..."},
          "messages": [{"$ref": "sha256:a1..."}, {"$ref": "sha256:b2..."}],
          "tools": {"$ref": "sha256:t9..."}, "max_tokens": 1024}
```

- **What gets hashed:** the Timbal `Message` dump the LLM span already stores today, not the provider-ready
  message. The UI keeps rendering what it renders now. Provider-ready messages also carry `cache_control`
  breakpoints that move every step, which would defeat dedup.
- Not a pointer into agent memory: compaction rewrites memory in place, attachment limits swap in stand-ins per
  request, and steering and tool search change messages and tools per step. The hash names exactly what that
  call saw.
- **Hash = `sha256` of RFC 8785 (JCS) canonical JSON**, not Python
  `json.dumps(sort_keys=True, separators=(",", ":"))`. The two differ on non-ASCII escaping (`ensure_ascii`)
  and float formatting (`1e-05` vs `1e-5`). Use `jcs` / `canonicaljson` in the SDK and `serde_jcs` in the
  monolith, with a shared test-vector file in both repos.
- **The server verifies** `sha256(jcs(body)) == hash` on every blob it receives, and rejects the save on a
  mismatch, so a client can never store a body under a hash it does not match. Verification runs on the raw blob,
  *before* NUL scrubbing (`strip_pg_jsonb_nuls`) and composer's `redact_replay_secrets`. The stored body is the
  scrubbed one, so a hash names the content as sent, pre-scrub. That is deliberate; do not re-hash after
  scrubbing.
- Blobs travel with the save that first uses them (`"blobs": {"sha256:a1...": {...}}`), deduplicated
  client-side per thread (the SDK remembers which hashes the server acknowledged) and server-side by
  primary key. A `$ref` the server cannot resolve rejects the save (`422`, listing the missing hashes); the SDK
  forgets them and resends with bodies.
- Server: `TraceBlobs(org_id, session_id, hash, body jsonb, PRIMARY KEY (org_id, session_id, hash))`, keyed by
  thread so turn N+1 re-references turn N's history instead of storing it again.
  - `session_id` is the root run id: `Traces.session_id` directly. For app runs it is the root run's
    `idempotency_key`, since `group_id` is the root's bigint PK and is resolved on write.
- **Retention:** blobs belong to the thread, not the run. Soft-deleting one run frees nothing; hard-deleting
  the thread or the retention sweep deletes its blobs.
- Readers: the UI's LLM-call view and evals/exports resolve `$ref`s (`GET .../runs/{id}?resolve=refs` can
  inline them for old clients). The framework's own resume reads agent memory, not LLM inputs: unaffected.
  - `trace_view.rs` size chips (`json_size(span.get("input"))`) would show ref sizes. Resolve, or have the SDK
    put `input_bytes` on the span.
  - `deployments/evals.rs` and the exports still need an audit for reads of LLM inputs.
- Unaffected, verified: `parse_trace` (root span + usage) and `chat_ingest::decompose_chat_run` (root span's
  `memory`, `input.prompt`, `output`).
- `format_version` bumps (`Traces.format_version` exists for exactly this).

## C — Span deltas

```text
PATCH /orgs/{org}/apps/{app}/runs/{run_id}        and   PATCH /orgs/{org}/traces/{run_id}
Content-Type: application/vnd.timbal.trace-delta+json
{"parent_id": ..., "rev": ..., "metadata": {...}, "snapshot": 17, "ended": false,
 "root": {"call_id": ..., "path": ..., "t0": ..., "parent_call_id": null},
 "spans": [ {"seq": 3, "span": {...}}, ... ],   // finished spans the server has not acknowledged
 "blobs": { <B, once both ship> }}
```

```sql
CREATE TABLE public."TraceSpans" (
    org_id   bigint NOT NULL,
    run_id   text   NOT NULL,   -- Timbal context.id = Traces.run_id = OrgsAppsVersionsRuns.idempotency_key
    call_id  text   NOT NULL,
    seq      int    NOT NULL,   -- client-assigned span START index; root = 0
    snapshot bigint NOT NULL,   -- SDK save counter: an older delta never overwrites a newer span
    span     jsonb  NOT NULL,
    PRIMARY KEY (org_id, run_id, call_id)
) WITH (autovacuum_vacuum_scale_factor = 0.01, autovacuum_vacuum_cost_delay = 0);  -- insert-then-delete churn
-- ON CONFLICT (org_id, run_id, call_id) DO UPDATE SET span = EXCLUDED.span, snapshot = EXCLUDED.snapshot
--   WHERE "TraceSpans".snapshot < EXCLUDED.snapshot

-- On both "Traces" and "OrgsAppsVersionsRuns":
--   trace_snapshot bigint NOT NULL DEFAULT 0   -- highest delta snapshot applied to the run
```

- Shared by both run tables (keyed by the producer run id), so it works on today's app-run path and
  survives app runs moving to `Traces`.
- **Envelope.** `rev` and `metadata` ride on the first delta and on `ended: true`, because `UpdateRunReqBody`
  carries both today. `rev` keeps first-write-wins; composer metadata keeps merging.
- **Order.** `seq` is the span's start index assigned by the SDK, root = 0. Server first-seen order would be
  finish order with the root last, but `chat_ingest` takes `trace[0]` as the root and today's traces are in start
  order. Assembly is `ORDER BY seq`.
- **Dispatch.** Branch on `Content-Type` before body extraction: read `Bytes`, then parse as a delta or as
  `UpdateRunReqBody`. axum's `Json` accepts any `application/*+json`, so a delta would otherwise deserialize as
  a full PATCH with `trace: None`.
- **Scrubbing.** Per span, before storing in `TraceSpans`: `strip_pg_jsonb_nuls` on both paths, plus
  `redact_replay_secrets` on composer. Otherwise NULs make the upsert fail with a 500, and unredacted secrets
  sit in `TraceSpans` and get served by GET assembly for crashed runs.
- **Concurrency.** Every delta first locks the run row (`SELECT status, finished_at, trace_snapshot ... FOR
  UPDATE`, or an advisory lock on `(org_id, run_id)` while the row does not exist yet), then:

  | Case | Action |
  |---|---|
  | `snapshot <= trace_snapshot` | Stale or retried delta: ack, no-op. |
  | run not terminal, `ended: false` | Upsert spans. Create the run row as `running` on the first delta (`started_at` = `root.t0`; composer's `create_running` row already exists), else bump `updated_at` (the liveness signal `services/traces.rs` already describes). No `parse_trace`, costs, metrics or ingest. |
  | run not terminal, `ended: true` (root finished, including runs parked on approval) | Upsert, assemble `trace` / `data` `ORDER BY seq`, run today's finalize path unchanged (parse, metrics, chat ingest, costs, titles, SIP parked-run observe), delete the rows. |
  | run terminal, newer snapshot | A span that changed after the root ended (composer builders keep running after the parent turn: `services/traces.rs`). Merge the spans into the assembled trace by `call_id`, rerun parse, costs (`record_costs` reconciles) and chat ingest (idempotent replay). No `TraceSpans` rows are created. |

  Every applied delta sets `trace_snapshot = snapshot`. A delta never moves a terminal run back to `running`.
- **Late spans are not metered.** `observe_trace_metrics` skips already-terminal runs, so post-end spans never
  feed custom trace metrics. This already happens today; it is now explicit. Decide whether to accept it (open question).
- **Fallback mid-run.** A full-trace PATCH on a run that has `TraceSpans` rows (the SDK lost `Accept-Patch` for
  the host) is authoritative: replace `trace`, delete the rows, reset `trace_snapshot`.
- **Readers of in-flight runs.** `Traces.storage = 'spans'` while in flight, `'inline'` after finalize. Readers
  already must check `storage`. For `OrgsAppsVersionsRuns` (transitional, until app runs move to `Traces`):
  `trace IS NULL AND status = 'running'` means "assemble from `TraceSpans`". `GET` of a run that never ended
  (its process died) assembles from the rows; this is what crash recovery reads. List views and exports see no
  trace for in-flight runs, where today they see a partial one.
- Mid-run saves carry finished spans only (immutable once finished) plus a root header without memory.
  The full root goes with `ended: true`.
- Negotiation: responses carry `Accept-Patch: application/vnd.timbal.trace-delta+json` (RFC 5789); the SDK
  switches per host once seen, sends full traces otherwise. Old SDKs keep working unchanged.

Effect: a save becomes one or two spans, and parsing and finalization happen once per run. The database
writes small rows instead of rewriting a growing TOAST value, so `sync` becomes affordable and crash recovery
sees every finished tool call.

## Chat session state (after C)

Durable `ChatSession` needs a small mutable record (queue, wakes, head run, lease). Make `ChatSession.id` the
thread's root run id (= `Traces.session_id` = `Sessions.session_id`), so it is not a fourth "session" (see
SESSIONS_PLAN "Naming collisions").

Keep the runtime state in a 1:1 side table, not on the `Sessions` row:
`SessionRuntime(org_id, session_id, runtime jsonb, runtime_version, lease_owner, lease_until, PRIMARY KEY
(org_id, session_id), FOREIGN KEY → Sessions)`. `Sessions` is the list read surface. Lease renewals on it would
rewrite the envelope row and update its list indexes every few seconds. The side table is a column family, not
a new concept.

Caveat: `Sessions.project_id` is `NOT NULL`, but composer home threads have `Traces.project_id IS NULL`. Durable
`ChatSession` cannot cover home threads without relaxing that.

## Order

| Step | Who | Size (estimate) |
|---|---|---|
| A server: decompression + `Accept-Encoding` on the two trace PATCH routes | platform | hours |
| A SDK: negotiated gzip, `400`/`415` fallback | framework | done |
| C SDK: delta sender, client `seq`, `rev`/`metadata`, fallback to full | framework | in progress |
| C server on `Traces` first: `TraceSpans`, `trace_snapshot`, lock, `storage = 'spans'`, GET assembly | platform | 3–4 days |
| C server on app runs | platform | 1–2 days |
| B SDK + server (JCS, verification, retention) + UI/evals `$ref` resolution | both | 1–2 weeks |
| Chat session runtime state (`SessionRuntime`) | platform | after C |

`Traces` goes first because composer already pre-inserts the `running` row (`create_running`), and
`Traces.storage` already exists so readers can tell where the body lives.

## Open questions

- Where do platform-hosted agents run, and is local disk kept across deploys?
- C: should late post-end spans feed custom trace metrics? If yes, `observe_trace_metrics` needs a per-span
  guard instead of the already-terminal one.
- B: flat `$ref` lists stay quadratic in ref count (small: ~75 B per ref). Should we use a prefix-chained form
  (`"messages": {"$prefix": "<hash of previous call's list>", "append": [...]}`) to make it linear and make
  compaction visible as a broken prefix? It can come later with a `format_version` bump.
- B (speculative): the platform LLM proxy already sees every proxied request body and could content-address on
  the server with no SDK change. That would complement B; it would not replace it, because BYOK direct calls
  bypass the proxy.
- Chat session: relax `Sessions.project_id NOT NULL` for composer home threads, or keep them out of durable
  `ChatSession`?

### Resolved since v2

- `durability="buffered"` as the default for every platform run: **no.** With C, `sync` is cheap and keeps every
  finished span for crash recovery. Buffering without C only hides the cost.
- B: hash the provider-ready message or the Timbal dump: **the Timbal dump** (see B).
