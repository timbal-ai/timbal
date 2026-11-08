# Session runtime on the platform (monolith#567)

What a durable `ChatSession` needs from the platform so that any worker of an app can serve any of its
sessions. The SDK side is `PlatformChatSessionStore` (`python/timbal/state/platform_chat_store.py`), tested
against a fake implementing exactly this contract (`python/tests/state/test_platform_chat_store.py`). Opt in
on a Timbal server with `TIMBAL_CHAT_STORE=platform`.

## What is stored

One row per session: the snapshot (`SessionSnapshot`, opaque JSON to the platform), a version, a lease,
and an inbox. The conversation itself stays in the traces. The snapshot holds what is on top of them: the
head run, messages and wakes no turn has taken yet, a turn in flight, whether the session is parked, and
the ids of the inbox items it applied.

It is written ahead of every change. A message is stored before its `202` goes out, so the first write
happens **before the session's first run exists**.

## Endpoints

All app-scoped, under `orgs/{org}/apps/{app}/session_runtime`, authenticated like every other app route.

| Call | Request | Answer |
|---|---|---|
| Read | `GET /{sid}` | `{runtime, runtime_version, write_id, lease_owner, lease_until}`, or 404 |
| Write | `PUT /{sid}`, `If-Match: "<version>"` (`If-None-Match: *` to create), body `{runtime, has_work, closed, root_run_id, write_id}` | `{runtime_version}` (the old one + 1), or **412** when the stored version is not the one given |
| Delete | `DELETE /{sid}` | 204 (the inbox goes too) |
| List | `GET ?with_work=true` | `{session_ids}`: open sessions (`closed = false`). With `with_work`, only those with `has_work` or inbox items |
| Lease | `POST /{sid}/lease` `{owner, ttl_s}` | `{lease_owner, lease_until}`, or **409** while another owner's lease has not expired, or 404 for no row |
| Release | `DELETE /{sid}/lease?owner=` | 204; does nothing unless `owner` holds it |
| Queue | `POST /{sid}/inbox` `{kind, body}` (`kind`: `message`, `wake`, `agent_message`) | 201 `{id}`, or 404 for no session |
| Pull | `GET /{sid}/inbox?owner=&wait_ms=` | `{items: [{id, kind, body, posted_at}]}`, oldest first, until acked; **409** when `owner` does not hold the lease. Long poll: answer as soon as an item exists, else after `wait_ms` (the SDK asks for 20,000) |
| Ack | `POST /{sid}/inbox/ack` `{owner, ids}` | 204 |

`has_work`, `closed` and `root_run_id` sit next to the opaque snapshot so the platform can list and link
without parsing it. `write_id` is stored as given and returned by the read.

## Semantics

**Compare-and-swap, not last-write-wins.** Every write names the version it read. A worker that lost the
session (its lease expired and another took it) has its next write refused (412), and stops without
overwriting anything. A new owner writes as soon as it restores the session, so the old owner is fenced off
on its next write even if its clock is behind.

**An answerless write is not a takeover.** A `PUT` that commits and then times out leaves the SDK one
version behind. Its next write would be refused and look like a takeover. So every `PUT` carries a fresh
`write_id`, and the SDK remembers the id of a write whose answer it never got. On a 412 it reads the row:
if the stored `write_id` is that one, nobody wrote since, and it writes on top of the stored version.
`PUT`s are never retried blindly.

**Lease.** The TTL is 30 s, renewed every 10 s. After `lease_until`, any owner may take it. The owner is
`host:pid:uuid7`, unique across restarts. The platform only compares owners and checks expiry (no pid
checks).

**Inbox, at least once.**
- A worker that receives a request for a session another worker holds queues it here instead of answering
  409. The SDK's server does this for `POST .../messages`, `.../wake` and `.../agent_messages`, answering
  `202 {"queued": "inbox", "inbox_id"}`.
- The owner long-polls, applies each item, stores the session, and only then acks.
- The snapshot keeps the ids it applied (`inbox_seen`, the last 256). An item pulled again after a crash
  between storing and acking is skipped.
- An item that can't apply (a bad body) is recorded in the session log (`inbox_rejected`) and acked, so it
  never comes back.

## Answers to the open questions in #567

- **The wake API for routines.** `POST /{sid}/inbox {"kind": "wake", "body": {text, key, tool_policy,
  silent_ok, trigger}}`. The dispatcher doesn't need to know who holds the lease: the holder's long poll
  answers within moments. If you want it immediate across instances, a Postgres `LISTEN/NOTIFY` on the
  session id can wake the waiting poll. No direct call to the owner.
- **A session nobody holds**, with queued work: `GET ?with_work=true` lists it, and a worker restoring
  sessions at startup (or on demand) takes it and drains the inbox. Who restores sessions on
  platform-hosted agents is the platform's call.
- **Lease semantics.** See above. Fencing is the version CAS on every write, plus the lease check on
  every inbox pull.
- **Local disk.** With this store the SDK keeps no session state on local disk.

## Points for the platform to decide

- **The FK to `Sessions`.** The first `PUT` comes before the session's first run is stored, so a foreign
  key to `Sessions` would refuse it. Options: create the `Sessions` envelope on the first `PUT`, make the
  key deferrable, or link it later through `root_run_id`.
- **The session id.** An id the SDK generated is the thread's root run id (the first turn runs under it).
  An application can also pass its own key (`ChatSession(session_id="conv-42")`). Then the root run id
  differs, and comes in `root_run_id` once the first turn has started. If `SessionRuntime` must be keyed by
  the root run id, platform-hosted sessions should let the SDK generate their ids.
- **Composer home threads** (`Sessions.project_id NOT NULL`): the store does not care, the route is
  app-scoped.
