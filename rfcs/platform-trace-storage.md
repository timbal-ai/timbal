# Platform trace storage and chat sessions

Status: proposal. Owners: framework (timbal), platform API (monolith).

## Problem

`PlatformTracingProvider` stores a run by sending its whole trace in a `PATCH
/orgs/{org}/apps/{app}/runs/{run_id}` every time a span finishes. Per request the API
(`handlers/orgs/apps/runs/edit.rs`, monolith at `03a03bcf`):

1. runs the auth gate query, which also looks the run up by id or idempotency key;
2. runs `parse_trace` over the whole trace (status, timing, cost; a read-replica query);
3. replaces the JSONB `trace` column of `OrgsAppsVersionsRuns`;
4. spawns `record_costs`.

A 40-step agent turn makes ~80 saves of a trace that grows to ~0.5 MB (every LLM span carries its
whole input history), so ~20 MB uploaded, parsed and rewritten in Postgres for one turn. With the
default `durability="sync"` the turn waits on every one. Request bodies are not compressed (the router
has no decompression layer).

Durable chat sessions (restart recovery for `ChatSession`) also need a small, mutable, per-session
record. The API has nowhere to put it: no sessions resource, and app runs cannot be listed or queried
by metadata.

## Framework changes (done, no API change)

- `ChatSession` turns on a platform provider default to `durability="buffered"`: saves during a turn
  are coalesced to one PATCH per flush interval (0.5 s), the turn's end still waits for its save. An
  explicit `ChatSession(durability=...)`, a provider that set `durability`, or
  `TIMBAL_TRACE_DURABILITY` wins.
- `PlatformTracingProvider.compress` / `TIMBAL_TRACE_GZIP=1`: gzip PATCH bodies
  (`Content-Encoding: gzip`). Off until the API decompresses requests (phase 1).

## Phase 1 — small API changes

### 1a. Request decompression

`tower_http::decompression::RequestDecompressionLayer` on the router (gzip, zstd). The SDK then turns
`compress` on by default once the deployed API advertises it (e.g. `Accept-Encoding` on
`OPTIONS /runs`, or a version check).

### 1b. Sessions resource

```text
GET   /orgs/{org}/apps/{app}/sessions/{session_id}        200 {snapshot, version, lease, updated_at} | 404
PUT   /orgs/{org}/apps/{app}/sessions/{session_id}        body {snapshot}; If-Match: <version> (absent = create)
                                                           200 {version} | 412 version mismatch | 409 leased by another owner
POST  /orgs/{org}/apps/{app}/sessions/{session_id}/lease  body {owner, ttl_s}  200 {lease} | 409 held
DELETE /orgs/{org}/apps/{app}/sessions/{session_id}       204
GET   /orgs/{org}/apps/{app}/sessions?pending=true        sessions with work pending (eager restore after a deploy)
```

```sql
CREATE TABLE public."OrgsAppsSessions" (
    id            BIGSERIAL PRIMARY KEY,
    org_app_id    BIGINT NOT NULL REFERENCES public."OrgsApps"(id),
    session_id    TEXT   NOT NULL,          -- caller-supplied, like a run idempotency key
    user_id       BIGINT NOT NULL REFERENCES public."Users"(id),
    snapshot      JSONB  NOT NULL,          -- a few KB: queue, wakes, head run, parked, in-flight run, log tail
    version       BIGINT NOT NULL DEFAULT 1,
    head_run_id   BIGINT REFERENCES public."OrgsAppsVersionsRuns"(id),
    pending       BOOLEAN NOT NULL DEFAULT false,  -- queued messages, wakes, or a turn in flight
    lease_owner   TEXT,
    lease_until   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (org_app_id, session_id)
);
```

Authorization mirrors run writes: `projects.runs.write` on the app's project, author-bound (only the
session's `user_id` reads or writes it; superadmin overrides), scoped credentials capped to the project.
`PUT` is a compare-and-swap on `version`; a held, unexpired lease by another owner rejects writes.
Writes are small and few (a send, a turn's start and end), so no batching is needed.

The SDK side is two classmethods on the provider, shared with SQLite:
`load_session(session_id) -> SessionSnapshot | None` and
`save_session(snapshot, expect_version=) -> version`.

## Phase 2 — span deltas

```text
PATCH /orgs/{org}/apps/{app}/runs/{run_id}
Content-Type: application/vnd.timbal.spans+json
{"parent_id": ..., "rev": ..., "spans": [<changed spans>], "ended": false}
```

- Store spans in `OrgsAppsVersionsRunSpans(run_pk, call_id, seq, span JSONB, PRIMARY KEY (run_pk, call_id))`,
  upserted per request: each save writes the spans that changed (finished since the last save, or still
  running), not the run.
- `parse_trace`, the `trace` column, status/cost and `record_costs` run when `ended: true` (and on the
  timeout sweep for runs that never end). While a run is live, status comes from its root span row.
- `GET .../runs/{run_id}` assembles the trace from span rows for live runs, the `trace` column for ended
  ones. The UI's live view reads the same.
- The SDK sends deltas when the API advertises the content type, whole traces otherwise. It already
  tracks which spans changed (`SqliteTracingProvider` does the same locally).

Effect: a save is one or two spans (KB) instead of the run (hundreds of KB), so `durability="sync"`
becomes affordable on the platform, and crash recovery sees every finished tool call.

## Phase 3 — LLM span history

Each `<agent>.llm` span stores the full message list it was called with, so a trace grows quadratically
with its steps. Store a reference instead (`{"memory_ref": <agent call_id>, "upto": n}`) and render the
input from the agent span's memory. Needs the UI (and evals/exports that read LLM span inputs) to
resolve the reference. Biggest payload cut; independent of phases 1–2.

## Order and sizing (estimates)

| Phase | Where | Size |
|---|---|---|
| Framework: buffered chat turns, gzip flag | timbal | done |
| 1a decompression | monolith | hours |
| 1b sessions resource | monolith (+ SDK provider methods) | 1–2 days |
| 2 span deltas | monolith + SDK + UI live view | 3–5 days |
| 3 LLM span references | SDK + UI + evals | 2–3 days |

## Open questions

- Is the monolith checkout used here (`03a03bcf`, Sept 23) what production runs?
- Where do platform-hosted agents run, and is local disk kept across deploys? (Decides whether a
  platform deployment could ever use the SQLite session store instead of 1b.)
- Should `ChatSession` be the only buffered default, or every platform run?
