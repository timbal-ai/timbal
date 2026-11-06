# Platform trace storage: smaller saves, smaller traces

Status: proposal (v5). Owners: framework (timbal) — SDK side; platform API (monolith) — server side.
Reviewed against monolith `origin/master` `046ce015` (2026-11-06).

**Changes since v4:** B is deferred until after 0, A and C ship, with listed measurements and a simpler
alternative (prefix deltas with keyframes). A new section lists the rules that keep a later move of finalized
trace bodies to object storage feasible; 0.4 and C follow them.

**Changes since v3:** new section "0 — Fix first", covering two ordering bugs in today's save path (one live, one
that would break C) and two cheap additions: SDK single-flight coalescing, and a resume projection that skips LLM and tool spans.
C's `snapshot` now uses the cross-process counter from 0.2. Order updated.

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

## 0 — Fix first (hours each, independent of A/B/C)

SDK code referenced below is `python/timbal/state/tracing/providers/platform.py` on timbal `docs/timbal-connect`
(`bcf11c3c`). `feat/bots` may already differ.

### 0.1 Bug (live today): an older save can overwrite a newer trace

The provider documents that saves of one run overlap and can finish in any order ("parallel tools, background
children"). The server ignores order:
- The app-run path (`edit.rs`, path A) replaces `trace` unconditionally.
- Composer's `upsert_trace` does the same with `data`.

So the last PATCH to *arrive* wins, not the newest snapshot. Worst case: the terminal save overtakes a slower
mid-run save. The late one lands, and its `parse_trace` sets `status = running` and `finished_at = NULL` on a run
that already went through chat ingest and `record_costs`.

Fix, with full snapshots and no deltas needed:
- The SDK sends its snapshot on every full PATCH: `X-Timbal-Snapshot: <n>`.
- Both run tables get `trace_snapshot bigint NOT NULL DEFAULT 0` (the same column C needs).
- Both write paths guard the update: `... WHERE trace_snapshot < $n` (and on the upsert's `DO UPDATE ... WHERE`).
  A losing write is acked (`204`), not an error, since the newer state is already stored.
- A PATCH without the header (old SDKs) keeps today's behaviour.

### 0.2 Bug (blocks C and 0.1): the snapshot counter restarts per process

`_snapshots = itertools.count()` is process-global and starts at 0. A run continued by another process (an
approval resumed elsewhere, a voice turn on another pod) would number its saves below what the server already
stored. Under 0.1 and C those saves would be silently dropped as stale.

Fix: per run, `snapshot = max(last_snapshot + 1, time.time_ns())`. This stays monotonic within a process and
moves forward across processes, as long as clocks are roughly in sync and a takeover happens after the previous
writer's last save, which approvals and turns guarantee. Optional hardening: the server returns
`X-Timbal-Snapshot: <stored>` on every PATCH response, and the SDK continues from the max of that and its own.

### 0.3 Cheap addition: single-flight, latest-wins saves (SDK)

At most one PATCH in flight per run. Spans that finish while one is in flight mark the run dirty. When it
returns, one save goes out, built from the *latest* state.
- Parallel tool calls and background children collapse into one save each round trip.
- Within a process, saves can no longer reorder. 0.1 is still needed for cross-process writes and retries.
- Works with today's full snapshots and with C's deltas unchanged: a delta then carries every span finished
  since the last ack.
- The terminal save (root ended) waits for any in-flight save, then goes out; it is never dropped as superseded.

### 0.4 Cheap addition: resume reads only the spans it uses

`PlatformTracingProvider.get()` `GET`s the parent's entire trace, and only an in-process cache (5 min, 64 MiB)
avoids that on the same pod. What resume actually reads (timbal `state/context.py`, `core/agent.py`,
`core/continuation.py`):
- the root span's `session` (`get_session`) and the trace's `parent_id`;
- the agent's own span (`_own_span`, possibly nested): `memory`, `status` (pause detection), `t1`, `metadata`
  (compaction), and `input` (the fallback walk when `memory` is not a list).

It never reads the LLM and tool child spans, which hold 63–92 % of the bytes (Problem table).

- At finalize, also write `resume_spans jsonb` on both run tables: the root span, plus every span whose `memory`
  is a list (agent spans), in trace order. Nothing else.
  A SQL JSONB-path projection over `trace` is not enough: Postgres still loads the whole TOASTed value to answer it.
- `GET .../runs/{id}?select=resume` returns `{"parent_id": ..., "trace": <resume_spans>}`. That is the same shape
  `get()` parses today, so `Trace(...)`, `_own_span` and `root_span()` work unchanged.
- GET responses advertise support with `X-Timbal-Select: resume`. The SDK switches per host once seen, else
  falls back to the full `GET`.
- Runs finalized before the column existed: the server computes it from `trace` on read (one full load, as today).
- In-flight runs (C): `resume_spans` comes from the `TraceSpans` rows with the same filter.

This removes most of the resume read cost without B, and covers the "inflates every resume `GET`" half of B's
motivation.

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

**Status: deferred pending measurements.** B is the only part here with an unsolved spec (canonical JSON across
Python and Rust), a new table with its own retention, and changes for every reader. 0.4 already removes the
resume-read half of its motivation; what remains is stored size and the bytes inside LLM spans. After 0, A and C
ship, measure:
- stored bytes per finalized run (p50 / p95 / p99), and the share in LLM span `input`;
- turns per thread (p50 / p95 / p99): cross-turn dedup is the only thing B does that prefix deltas below do not;
- how often LLM span inputs are read at all, by age (UI LLM-call view, evals, exports).

Then pick one: B as specified below, prefix deltas (below), or nothing. If old LLM inputs are rarely read, a
retention job that strips `input.messages` from LLM spans after N days may be enough (speculative).

### Alternative: prefix deltas with keyframes

Each LLM span references the previous LLM call of the same agent in the same run, instead of a blob store:

```json
"messages": {"$from": "<previous LLM call_id in this run>", "keep": 37, "append": [ ... ]},
"tools":    {"$same_as": "<previous LLM call_id>"}
```

- The SDK writes a keyframe (full `messages` / `tools`) on the first LLM call of each run, every 16 calls, and
  whenever the prefix breaks (compaction, steering, attachment stand-ins: `keep` would be shorter than the
  previous list, or an earlier message changed).
- Readers resolve by walking back at most 16 spans of the same trace. No new table, no hashing spec, no
  server-side verification, no retention coupling: everything stays inside one run's body.
- Costs one full history per turn (the keyframe), so storage stays quadratic in *turns* rather than *steps*. B
  only wins for long threads.
- Same `format_version` bump, the same reader work (UI, evals, exports, `trace_view` size chips), and the same
  non-impact on `parse_trace` / `chat_ingest` as B.

### B as specified (if the measurements justify it)

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
    snapshot bigint NOT NULL,   -- SDK save counter (0.2: cross-process monotonic): an older delta never overwrites a newer span
    span     jsonb  NOT NULL,
    PRIMARY KEY (org_id, run_id, call_id)
) WITH (autovacuum_vacuum_scale_factor = 0.01, autovacuum_vacuum_cost_delay = 0);  -- insert-then-delete churn
-- ON CONFLICT (org_id, run_id, call_id) DO UPDATE SET span = EXCLUDED.span, snapshot = EXCLUDED.snapshot
--   WHERE "TraceSpans".snapshot < EXCLUDED.snapshot

-- On both "Traces" and "OrgsAppsVersionsRuns":
--   trace_snapshot bigint NOT NULL DEFAULT 0   -- highest snapshot applied to the run (added by 0.1)
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
  the host) is authoritative when its `X-Timbal-Snapshot` is newer (0.1): replace `trace`, delete the rows, set
  `trace_snapshot` to it.
- **Readers of in-flight runs.** `Traces.storage = 'spans'` while in flight, `'inline'` after finalize. Readers
  already must check `storage`. For `OrgsAppsVersionsRuns`, prefer an explicit `trace_storage` column (see
  "Keeping object-storage offload open", rule 5). The fallback, transitional until app runs move to `Traces`, is
  that `trace IS NULL AND status = 'running'` means "assemble from `TraceSpans`". `GET` of a run that never ended
  (its process died) assembles from the rows; this is what crash recovery reads. List views and exports see no
  trace for in-flight runs, where today they see a partial one.
- Mid-run saves carry finished spans only (immutable once finished) plus a root header without memory.
  The full root goes with `ended: true`.
- Negotiation: responses carry `Accept-Patch: application/vnd.timbal.trace-delta+json` (RFC 5789); the SDK
  switches per host once seen, sends full traces otherwise. Old SDKs keep working unchanged.

Effect: a save becomes one or two spans, and parsing and finalization happen once per run. The database
writes small rows instead of rewriting a growing TOAST value, so `sync` becomes affordable and crash recovery
sees every finished tool call.

## Keeping object-storage offload open (later, not in this RFC)

Moving *finalized* trace bodies out of Postgres is a separate, later step. `Traces.storage` already reserves
`'s3'` / `'external'`, and `services/traces.rs` already tests that offloaded bodies are not walked. In-flight
spans (`TraceSpans`) stay in Postgres as the hot tier. Langfuse made the same move in v3: bodies in S3, metadata
in Postgres. What 0, A and C must do now so that move is a new backend plus a backfill, not a rewrite:

1. **One write path for the assembled body.** Every write of a finalized `trace` / `data` goes through one
   function per run table (`store_trace_body(run, body)`): C's `ended: true`, C's post-end merge, a full PATCH,
   and 0.1's guarded writes. Today it writes the inline JSONB value; later it writes an object and the pointer.
2. **One read path.** Every reader loads the body through one function that switches on `storage`
   (`'inline' | 'spans' | 's3' | 'external'`). Composer's `get_trace` already returns
   `StoredTrace { storage, data }`; app-run readers do not yet. Route GET single, the full resume fallback,
   exports, evals, `trace_view`, and the SIP parked-run observer through it as they are touched.
3. **No new SQL into the body.** No new queries with JSONB operators on `trace` / `data`
   (`->`, `jsonb_array_elements(trace)`, `@>`). Anything a list, filter, analytics or resume read needs is
   extracted at finalize into its own column, so list views and resume never fetch the body. That already covers
   status, timings, cost, title, `resume_spans` (0.4) and chat ingest. A new derived field gets a new column.
4. **Merges are read-modify-write in Rust, not `jsonb_set` in SQL.** C's post-end merge loads the body through
   rule 2, merges by `call_id`, and writes it back through rule 1, so it works unchanged when the body is an object.
5. **An explicit storage marker on app runs too.** When C lands on `OrgsAppsVersionsRuns`, add
   `trace_storage text NOT NULL DEFAULT 'inline'` instead of the implicit "`trace IS NULL AND status = 'running'`"
   rule. Or move app runs to `Traces` first.
6. **Immutable body versions.** When offload comes, key objects by snapshot
   (`traces/{org_id}/{run_id}/{trace_snapshot}.json.zst`), and flip the pointer and `trace_snapshot` in the same
   transaction. A post-end merge writes a new object, so readers never see a half-written body; old versions
   are garbage-collected. Nothing to do now except keep `trace_snapshot` (0.1) monotonic.
7. **Deletion goes through one function** that will also delete the objects (hard delete and the retention
   sweep; a soft delete keeps them). The same applies to `TraceSpans`, and to `TraceBlobs` if B ships.
8. **Store access goes through a store abstraction, not direct S3 calls in handlers.** That is the same seam
   as `KbStore` / customer-store credentials, so `'external'` (the tenant's data plane) is a second backend, not
   a fork. Our own bucket is a transitional vendor-S3 dependency and should be stated as such when it lands.

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
| 0.1 server: `trace_snapshot` + guarded writes on both PATCH paths | platform | hours |
| 0.1 + 0.2 SDK: `X-Timbal-Snapshot`, cross-process counter | framework | hours |
| 0.3 SDK: single-flight, latest-wins saves | framework | hours |
| 0.4 server: `resume_spans` at finalize + `?select=resume`; SDK: use it when advertised | both | ~1 day |
| A server: decompression + `Accept-Encoding` on the two trace PATCH routes | platform | hours |
| A SDK: negotiated gzip, `400`/`415` fallback | framework | done (add the `400` fallback) |
| C SDK: delta sender, client `seq`, `rev`/`metadata`, fallback to full | framework | in progress |
| C server on `Traces` first: `TraceSpans`, lock, `storage = 'spans'`, GET assembly | platform | 3–4 days |
| C server on app runs | platform | 1–2 days |
| Measure (see B), then B, prefix deltas, or nothing | both | decide after C; B 1–2 weeks, prefix deltas less |
| Chat session runtime state (`SessionRuntime`) | platform | after C |
| Offload finalized bodies to object storage | platform | later; separate RFC |

`Traces` goes first because composer already pre-inserts the `running` row (`create_running`), and
`Traces.storage` already exists so readers can tell where the body lives.

## Open questions

- Where do platform-hosted agents run, and is local disk kept across deploys?
- C: should late post-end spans feed custom trace metrics? If yes, `observe_trace_metrics` needs a per-span
  guard instead of the already-terminal one.
- B, if chosen: flat `$ref` lists stay quadratic in ref count (small: ~75 B per ref). Should we use a
  prefix-chained form (`"messages": {"$prefix": "<hash of previous call's list>", "append": [...]}`) to make it
  linear? It can come later with a `format_version` bump.
- B (speculative): the platform LLM proxy already sees every proxied request body and could content-address on
  the server with no SDK change. That would complement B; it would not replace it, because BYOK direct calls
  bypass the proxy.
- Chat session: relax `Sessions.project_id NOT NULL` for composer home threads, or keep them out of durable
  `ChatSession`?

### Resolved since v2

- `durability="buffered"` as the default for every platform run: **no.** With C, `sync` is cheap and keeps every
  finished span for crash recovery. Buffering without C only hides the cost.
- B: hash the provider-ready message or the Timbal dump: **the Timbal dump** (see B).
