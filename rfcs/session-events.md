# RFC: event-triggered turns across chat and voice

Status: **draft; design plus a first implementation covering #214's use case**.
Reviewed against main at `49af46e9` on 2026-10-23 (after #216 and #223 merged).
Phase 1 (the text driver, session events, and the voice adapter) is implemented
on this branch as described below; "Implementation notes" records where
implementing it changed earlier revisions of this text. What remains of Phase 2
(durable inbox, authenticated session identity, cross-worker ownership,
idempotency keys, expiry) is a proposal, not a shipped interface.

## Decision in brief

A task finishing is an input to a conversation, not a caller utterance and not
an instruction to speak immediately. Separate three responsibilities:

1. Record a correlated event/result for the owning conversation.
2. Decide whether and when another agent turn may start.
3. Deliver its output through chat or voice, with channel-specific interruption
   and playback rules.

The shared unit is the **session store**: the existing background-task store
gains an event inbox, and both kinds of input (finished tasks, posted events)
are waited for, drained, and formatted by the same primitives
(`wait_for_completion`, `take_followup`, `format_session_notice`). Each channel
owns its turn loop on top of them:

- Text: `AgentContinuation` (with `Agent.stream_until_idle(...)` as sugar), a
  caller-driven stream of ordinary runs.
- Voice: `VoiceSession`, whose turns already run through `_run_turn` (TTS,
  fillers, barge-in, heard-prefix memory, metrics), adds an event loop gated by
  its quiet-line check.
- Any turn of the session, including a plain `agent(...)` call, receives what
  is pending ahead of its prompt, so nothing waits for a driver to exist.

It does not create a second task store, make `VoiceSession` the owner of
conversation state, promote client text to trusted runtime context, or change
the lifetime of existing agent calls. Event turns are opt-in: calling
`notify()`/`post_session_event()` is the opt-in for events; finished tasks start
a voice turn only with `announce_background_results`.

Not implemented: events surviving a process restart or reaching another worker,
wakeups for a text session with no driver running (its next turn still gets
them), and an interrupting (`now`) policy.

## Related work and migration

- [#214: voice notifications](https://github.com/timbal-ai/timbal/pull/214):
  **closed in favour of this RFC** (its closing comment links here). Its use
  case is real and is the first consumer of the driver; its mechanism is what
  this RFC decides against: a scheduler owned by `VoiceSession` beside the
  agent's own turn path, an overdue path that appends the note to the
  **previous** run's memory and re-serializes it (#216 caches the exact PATCH
  body that was stored and #223's end barrier has already uploaded the final
  snapshot, so that edit diverges from both), a `bool` return for several
  distinct outcomes, and client frames promoted to `source="runtime"` without
  input guardrails. Carried forward from its branch (`voice/session-notify`)
  into this PR:
  - the quiet-line check (no turn running, playback finished, no greeting or
    idle prompt, no held fragment, quiet window, VAD silent, last run not
    parked on the caller), unchanged in substance;
  - `current_voice_session()`, the session-context snapshot turns start from,
    `TurnMetrics.event`, and no fillers on event turns;
  - the LiveKit/WebRTC frame parsing and tests, with client `notify` frames
    opt-in through server-only config and screened by input guardrails;
  - key replacement and batching of waiting events.

  Changed on the way: an event the line never has room for rides the caller's
  **next** turn through the agent's normal injection (no `max_wait_secs`, no
  edit of the previous run); `notify()` returns a `SessionEvent` receipt at
  once instead of a `bool` after placement; the `now` policy is not carried.
- [#218: end call](https://github.com/timbal-ai/timbal/pull/218): was stacked on
  #214's branch. Its design is unaffected by this RFC: keep hang-up-after-playback
  as a voice policy, not a generic session close, and count partial speech/VAD
  as caller activity. `current_voice_session()` now lands with this PR, so it
  can retarget onto this branch (or `main` once this merges) without carrying
  it; #215, which it needed for `session_ended` on WebRTC, is merged.
- [#216: tracing and interrupted history](https://github.com/timbal-ai/timbal/pull/216)
  and [#223: buffered persistence](https://github.com/timbal-ai/timbal/pull/223):
  merged. Reuse read-after-save, interrupted-turn recovery, and the end
  barriers. Neither is an asynchronous/durable notification queue.
- [#179: external background polling](https://github.com/timbal-ai/timbal/pull/179):
  complementary read/cancel access, not a turn scheduler. Polling for a UI must
  not acknowledge delivery to the model.
- [#174: OpenAI live pipeline](https://github.com/timbal-ai/timbal/pull/174):
  a separate live-model adapter will need equivalent semantics without pretending
  its provider-owned loop is the ordinary text-agent loop.
- [#215: RTC drain](https://github.com/timbal-ai/timbal/pull/215) and
  [#217: interruption persistence](https://github.com/timbal-ai/timbal/pull/217):
  preserve their transport/playback and history fixes; neither is a session inbox.

#214 is the only PR this RFC closes. Nothing else is closed, rebased, or
modified by it.

## What exists already

`BackgroundTaskStore` is process-local and follows `RunContext.parent_id`. It
owns task state, append-only child event logs, and one-shot completion notices.
`Agent._inject_background_completions` attaches those notices to the receiving
turn. Model polling can acknowledge a result so the next turn does not repeat it.
Its docstring states the lifetime rule this RFC must keep: nothing is registered
in the process-local run-id map until the spawn path creates a store.

`JobStore` already decouples an HTTP run from its SSE reader. Its replay log is
per run, bounded, and process-local. Neither it nor a persisted trace schedules
another conversational turn. The existing server does not offer cross-worker
session ownership or authorization on its own.

Since #223 the end of a root run is a synchronous barrier in every durability
mode, so a follow-up's history read sees the stored final snapshot. A background
child finishing after the root triggers another barrier save of the parent run
(its real output over the placeholder); that does not change the parent's
memory, so it does not affect what a follow-up loads. #216's recovery rebuilds a
turn that stored no memory from its stored `input`, which is what makes a
notice carried as a run's prompt durable (see "Trace shape").

`VoiceSession` has additional state that a text loop does not: partial speech,
VAD, held fragments, greetings, fillers, and audio still playing after generation
has ended. Finishing a model run is therefore not sufficient voice admission.

## Evidence from other runtimes

Local experiments used scripted models and controlled asynchronous tools, not
paid model APIs. The real framework loops, tool execution, and local storage
were exercised. Nineteen behavioral checks passed in three trials each.

- Mastra core **1.79.0**, memory **1.39.0**, libsql **1.27.0**: normal streaming
  did not autonomously follow up; `untilIdle` did. Completion during a response
  waited for that response to end. Follow-ups reused history without another
  human message. Idle signals distinguished persist-only from wake. Aborting the
  wrapper left the task's abort signal unset and its result completed in storage.
  Continuation output belonged to the combined stream, not reliably to `.text`.
- LangGraph **1.2.16**: state updates did not wake an interrupted graph; explicit
  resume re-entered the node. Two library invocations on one thread overlapped.
  Agent Server **0.16.2**, local in-memory runtime **0.36.2**, with four execution
  slots serialized same-thread enqueued runs while different threads overlapped.
  A run with `on_disconnect="continue"` finished after its SSE reader disconnected.
- Google ADK **2.12.0**, google-genai **2.32.0**: the ordinary runner required
  explicit delivery of a correlated function response. The live dispatcher
  asynchronously persisted/enqueued results with `SILENT`, `WHEN_IDLE`, and
  `INTERRUPT` scheduling. Its live teardown routine cancelled background tools.

Boundaries: ADK's live dispatcher/cleanup were exercised without a Gemini
connection; actual speech timing was not tested. No production deployment,
distributed owner failover, or process-crash recovery was validated. These are
observations of the versions above, not guarantees for all framework versions.

Primary references:
[Mastra continuation](https://mastra.ai/docs/harness/background-tasks),
[Mastra signals](https://mastra.ai/docs/harness/signals),
[LangGraph Agent Server](https://docs.langchain.com/langsmith/agent-server),
[ADK long-running tools](https://adk.dev/tools-custom/function-tools/#long-run-tool),
[ADK live tools](https://adk.dev/live/tools/).

The useful commonality is not one universal Session class. It is the separation
of event receipt, execution ownership, and output policy.

## Phase 1: session events, text driver, voice adapter

### Session events

```python
from timbal.state import post_session_event

event = post_session_event(run_id, "The nightly export finished.", key="export")
event.state            # "pending" -> "delivered" (event.run_id) | "superseded"
await event.settled()  # observer; cancelling it does not withdraw the event
```

- The event goes to the `BackgroundTaskStore` of `run_id`'s session (one is
  registered under `run_id` when the session has none, for the turn chained on
  it). `BackgroundTaskStore.post_event` is the same thing given a store.
- Nonblocking: posting never waits for a turn. Text is capped at 4,000
  characters. A pending event with the same `key` is superseded (receipt state
  `superseded`); one a turn has already taken is not replaced.
- `source="app"` (server code) or `"client"` (a user's device). Client text is
  labelled for the model as information, never instructions; the voice
  transport additionally screens it with the agent's input guardrails and
  accepts it only when server config allows.
- Every root turn of the session drains pending events with the finished-task
  notices into one runtime message ahead of its prompt (`format_session_notice`;
  without events the notice is byte-identical to the existing one), records
  `metadata.session_events`, and settles the receipts with its run id. A run
  resuming an approval/suspension does not drain (the same rule as for
  completions), so nothing lands between a tool use and its result.
- A turn the session starts itself takes everything with `take_followup`. The
  receipts settle as delivered only once that run has started (its stored input
  then holds the notice); if it never starts, `Followup.release()` hands the
  work back, ahead of anything newer, superseding an event whose key a newer
  pending event now carries.
- The text is wrapped in `<session_event id source key>` blocks; a closing tag
  inside the text is neutralized so it cannot speak outside its block.

### Text driver API

```python
from contextlib import aclosing

from timbal.core.continuation import AgentContinuation, continuation_for_run

# Local, caller-driven: sugar over the driver with no admission hook.
stream = agent.stream_until_idle(
    prompt="Build the report in the background and tell me when it finishes.",
    max_idle_secs=300,
    max_continuations=10,
)
async with aclosing(stream):
    async for event in stream:
        await publish(event)
print(stream.stop_reason, stream.last_run_id, stream.continuations)

# Hosted: the host owns the consumer task and decides when a follow-up may start.
stream = AgentContinuation(
    agent,
    admit=wait_until_channel_free,  # awaited before each follow-up; may be None
    max_idle_secs=300,
    max_continuations=10,
    prompt=...,  # the remaining keywords are the agent call's
    parent_id=...,
)

# From a request handler that only has a run id.
driver = continuation_for_run(run_id)
if driver is not None:
    driver.stop()  # finish the in-flight run (if any), then end the stream
```

`publish` is application code. This does not add a server route. Callers
consume all runs, including multiple root `OUTPUT` events; each run retains a
fresh `run_id` and the previous turn as `parent_run_id`. There is intentionally
no single `.collect()` result that hides the extra turns.

- `admit` is an optional parameterless callable, sync or async. The driver calls
  it after a completion is pending and before it starts the follow-up; when it
  returns, the driver drains and starts immediately, so it *is* the start
  boundary. A preemption, `stop()`, or the idle timeout while `admit` is pending
  cancels it. The generic driver has no quiet windows or playback knowledge;
  those live in the channel's `admit`.
- `stop()` ends the stream at the next boundary: immediately while waiting,
  after the current run's `OUTPUT` while one is in flight. `aclose()` closes the
  in-flight run too. Neither cancels detached children.
- `continuation_for_run(run_id)` returns the driver owning that run's session
  store, or `None`. The store holds a weak reference, so a dropped driver does
  not pin itself.

### The driver

1. Starts the ordinary foreground agent run and marks the run context as owned
   by this driver. If the run spawns children, the store it creates inherits
   the owner; if a store already exists for the session (`parent_id` chain), the
   driver acquires it at the run's `START`, before parameter resolution, hooks,
   or the model run.
2. Streams the run to completion, including tracing finalization.
3. Stops on unsuccessful output, approval, or interaction suspension.
4. Waits on the store's signal, which a finished task or a posted event wakes,
   without polling or draining. Already finished tasks are reconciled before
   declaring quiescence, including the interval before their done callbacks
   have executed. A direct human turn on the session preempts this wait (see
   "Ownership and preemption").
5. Verifies readable parent history and no unresolved tool calls, then awaits
   `admit` if given.
6. Takes the pending notices and events (`take_followup`) and starts another
   ordinary run with the formatted notice as its `prompt`. The notice is a
   `role="user"` message with `metadata.source="runtime"`, so it is not a human
   utterance, is hidden from transcripts by `Message.is_runtime()`, and skips
   input guardrails like every runtime message. It opens with a line saying the
   turn was started by the updates, not by the user. No previous-trace edit
   occurs. If the driver is closed or cancelled before that run starts, the
   work goes back to the store.
7. Repeats until quiescent or a stop condition is reached.

Draining in the driver rather than in the receiving run means the notice is in
the follow-up's stored `input` from its `START`. An acknowledgement that arrives
after the driver has taken the notices does not withdraw them; one that arrives
before (model poll during the previous turn, external `ack_completion`) means
the inbox is empty and no follow-up starts.

### Ownership and preemption

Only one driver owns a store at a time. Any other **root turn** on an owned
session (a direct `agent(...)` call, or the first run of a new driver) is
handled by the owner's timing:

- Driver waiting between runs (completion signal or `admit`): the direct turn
  **preempts**. The driver ends with `stop_reason="preempted"` and releases the
  store; the new turn proceeds and any pending notices attach to it through the
  existing `_inject_background_completions`. A new driver then owns the store.
  One run, nothing lost, human first.
- Driver with a run in flight (from the drain that commits to a follow-up until
  that run's root `OUTPUT`): the new turn is **rejected** before parameter
  resolution, hooks, or the model run, with an error telling the caller to wait
  for that `OUTPUT`. Running two turns of one session concurrently is the one
  thing the owner guard exists to prevent.

This is the Phase 1 form of "human input takes priority": the common case (a
user types while the agent waits for a task) produces a normal turn, not an
error. Hosts that want queueing instead of rejection for the in-flight case
build it on `stop()` and `continuation_for_run`.

The check runs in the root path of `Runnable._stream`, right after `START` and
before parameter resolution. It cannot live in `Agent.handler`: `pre_hook` runs
in `Runnable` before the handler, so a refused turn would already have run its
hook. Sessions without a store pay one attribute read.

### Store lifetime

The driver must not allocate a `BackgroundTaskStore`, or register run ids in
the process-local map, for an agent that never spawns a child. The owner is
stashed on the `RunContext`; `ensure_background_store` copies it onto the store
it creates on the spawn path. After the first run, no store means `idle`.
Releasing the owner does not delete map entries; the existing retention/reaping
rules own that.

Known gap, unchanged from today: two root turns on one session that both start
before either spawns a child are not serialized by anything. The guard above
only exists once a store exists.

### Trace shape

- Initial run: `input.prompt` is the human prompt, as today.
- A turn the session starts itself (text follow-up or voice event turn):
  `input.prompt` is the notice (runtime user message; kind
  `background_task_completed`, or `session_event` when it carries events);
  `metadata.trigger = {"kind": "background_completion" | "session_event",
  "task_ids": [...], "event_ids": [...]}`, plus `continuation_index` from the
  text driver.
- A human turn that receives pending work has no `trigger`; the notice sits in
  its memory before its prompt, and `metadata.session_events` lists the event
  ids it received.
- A follow-up that fails before assembling memory still holds the notice in its
  stored `input`, so the next turn's rebuild (#216) re-attaches it. The notice
  is never only in memory.

### Stop conditions

Observable through `stop_reason`: `idle`, `idle_timeout`, `continuation_limit`,
`paused`, `run_stopped`, `history_unavailable`, `preempted`, `stopped`,
`closed`, `cancelled`, or `error`. `run_stopped` is accompanied by the run's
`status.code` (`last_status`), so an errored and a cancelled run are
distinguishable. `continuations` counts admitted follow-up runs.

- `max_idle_secs` is finite, positive, and applies between runs, including the
  history-read preflight and any `admit` wait. It is not a model/tool execution
  timeout. In a hosted setting it is also how long the host's consumer task
  lives with nothing to publish; pick it per channel.
- `max_continuations` is a non-negative integer; zero allows the initial turn only.
- No background work means immediate completion; this is not a permanent listener
  for hypothetical future tasks.
- Notices not yet taken by the driver remain pending on timeout, limit, stop,
  close, preemption, or missing history. A model that already polled a result
  does not get a duplicate.
- Closing/cancelling the driver cancels its foreground execution and further
  automatic responses, **not detached children**. Use existing explicit task
  cancellation to cancel those. Python event-loop/process shutdown is different:
  this does not make a local asyncio task survive shutdown.
- The stream is caller-driven. A client disconnect only leaves it running if the
  hosting application owns a separate consumer task. Existing `/run`, `/stream`,
  voice transports, and `JobStore` are not automatically switched to this API.
- Do not run two consumers or call `aclose()` concurrently with a pending
  `__anext__()`.

The first slice does not promise exactly-once consumption or durable delivery.
After a notice is taken, cancellation, provider failure, or a failed trace save
does not prove whether the model or user received it. Preserve that distinction
rather than automatically replaying a potentially side-effecting turn.

### Voice adapter

`VoiceSession` runs its own event loop over the same store, because its turns
must go through `_run_turn` (TTS, fillers, barge-in, heard-prefix memory,
metrics); hosting `AgentContinuation` would bypass all of that.

- `await session.notify(text, key=None, source="app") -> SessionEvent` posts to
  the call's conversation store and returns at once. The store is made on
  demand; one made before any turn has a run is adopted by the first turn's run.
- One watcher task per call, started by `notify()` and at the end of any turn
  whose store holds pending events (or, with `announce_background_results`,
  background work). Waiting for work uses the store's waiter; waiting for the
  line polls the quiet-line check every 50 ms, because "the caller's audio has
  finished playing" is an estimate with no event to wait on.
- Quiet line: session started and open, last run not parked on the caller, no
  turn running, agent not speaking, no greeting or idle prompt pending, playback
  finished, no held caller fragment, no uncommitted partial in the last 5 s,
  nothing from the caller in the last 0.6 s, local VAD silent. Event turns never
  interrupt; once one has started, the caller barges in as on any reply.
- The event turn starts under the same lock as caller turns, right after
  `take_followup`, as `_run_turn(followup=...)`: the notice is the prompt,
  fillers are off, `TurnMetrics.event` is true, the root span gets `trigger`,
  and the receipts settle at the run's first event. A task cancelled before its
  first step releases the work from a done callback, since its `finally` never
  runs.
- A caller who speaks first gets everything pending with their turn, through
  the agent's normal injection: #214's "overdue" case, without an edit of the
  previous run.
- Finished background tasks start an event turn only with
  `announce_background_results`; otherwise they wait for the caller's next turn
  as before. When an event turn happens anyway, finished tasks ride it.
- `source="client"` needs `client_events` and is screened by the agent's input
  guardrails (a block raises `PermissionError`). WebRTC data-channel frames and
  LiveKit frames from the caller are client events; LiveKit frames with no
  participant (the server API's `SendData`, i.e. the app's backend) are app
  events; other participants are ignored. `VoiceConfig.events`
  (`announce_background_results`, `client_notify`) is read from server config
  only, never from the hello.
- Events undelivered at hang-up stay pending in the conversation's store for
  its next turn (a text turn continuing the call), and are logged.
- `current_voice_session()` is set for the session's turns, the tools they
  call, and tasks they spawn; turns start from a snapshot of the session's
  context, so code outside the call (a `notify` caller, a LiveKit tap) does not
  leak its run context into them.

## Implementation notes

Earlier forms on this branch: an empty runtime wakeup prompt drained by the
receiving run, every competing root turn rejected, an opaque owner, a store
allocated for every driver run. They were replaced by the design above.
Implementing it changed these points of earlier revisions of this RFC:

- A new driver on an owned session is treated like a direct turn (preempt while
  waiting, reject while in flight) instead of always failing. A host that starts
  every user turn through a driver would otherwise have to find and stop the
  previous driver first.
- The ownership check stays in `Runnable._stream` rather than moving to
  `Agent` (see "Ownership and preemption" for why).
- `AgentContinuation` takes the agent call's keywords directly (`prompt=...`)
  rather than a `kwargs=` dict, matching `stream_until_idle`.
- Voice does not host `AgentContinuation` with its quiet-line check as `admit`,
  as an earlier revision said: the driver starts agent runs itself, and voice
  turns must start in `_run_turn`. The two share the store-level primitives
  (waiter, `take_followup`, notice formatting, receipts) instead.
- Taking work for a turn the session starts itself no longer settles it at
  once: receipts settle when the run starts, and `Followup.release()` hands the
  work back otherwise. Without this, a barge-in landing in the same loop tick as
  an event turn's start lost the events.

`Agent.handler` is unchanged from `main`; `_inject_background_completions` also
drains posted events. The rest is `core/continuation.py`, the event inbox on
`BackgroundTaskStore`, one `RunContext` attribute, the check call in
`Runnable._stream`, the voice event loop, and the transport frames.

## Phase 2: proposed shared session contract

Phase 1 implements part of it: event ids, nonblocking ingress with receipts and
an observer, key replacement only while pending, a bounded payload, an
app/client source with screening and labelling, human-first admission, and
release before attachment. The rest is a design target.

### Identity and ownership

Use an explicit authenticated conversation identity, separate from run IDs and
the parent trace chain. Scope it to the application/tenant/user ownership model.
A session driver is the sole authority admitting foreground turns; every user
message and event-triggered turn must go through it. Do not add a second voice
consumer beside a chat driver reading the same inbox.

The first hosted version may be single-process with explicit routing to its
owner. Cross-worker/restart support needs a persistent inbox, ownership leases
with fencing, and recovery semantics. A context variable alone provides none
of these. The deployed runtime's worker lifetime/routing must be checked before
claiming support for events arriving after an HTTP request or voice call ends.

### Event identity and admission

Still to add: an authenticated producer and session identity (today: whoever
can call `post_session_event` in-process, or the transport's caller check);
idempotency keys distinct from replacement keys (the same idempotency key with
a different payload is a conflict, not a silent overwrite); a bounded inbox with
explicit overflow; causal task/run ids on the event. Authenticated tool results
remain untrusted data, not instructions. Event validation, payload limits,
admission policy, and any result screening are independent of transcript
visibility.

### Scheduling versus delivery

- `context_only`: retain for the next eligible turn; never start one itself.
  Phase 1: a finished task in voice without `announce_background_results`.
- `when_idle`: start a turn once the owner/channel admits it; batch ready
  events. Phase 1: the text driver, and voice event turns.
- `interrupt`: explicit privileged policy, not a default or an expiry fallback.
  Voice must preserve its heard-prefix interruption behavior. Not implemented.

An event waiting too long does not silently become an interruption. Expiry
reports a reason and preserves whatever source/task result remains available;
not implemented: today an event waits until a turn takes it. Track separately:
accepted into local/durable storage, attached to model context, run
completed/failed/interrupted, and output delivered. Text generated, audio sent,
and audio played are distinct facts; Phase 1's receipt covers only the first
two. At-least-once transport and deduplicated admission still do not provide
exactly-once tool side effects or exactly-once audible announcements.

## Acceptance gates and rollout

Executable tests in the repository:

- `python/tests/core/test_continuation.py`, the text driver. Besides multi-run
  chaining, in-memory and JSONL history, no historical edits, no extra human
  message, completion batching, completion during generation, acknowledgement,
  same-agent session isolation, paused/failed runs, idle/continuation limits,
  missing history, close, and cancellation while waiting or generating:
  - A direct turn while the driver waits: driver ends `preempted`, exactly one
    run starts, its memory carries the pending notice before the human prompt,
    nothing is drained twice.
  - A direct turn while a follow-up is in flight: rejected before hooks (per-run
    hook record) and the follow-up finishes. The same two cases for a new driver.
  - `stop()` before iteration, while waiting, and during a follow-up (which
    reaches `OUTPUT`, then no more); `continuation_for_run` finds the driver and
    returns `None` after release.
  - Follow-up `input.prompt` is the notice and `is_runtime()`; `metadata.trigger`
    lists its task ids; a follow-up cut before memory assembly leaves the notice
    to the next turn through the #216 rebuild; an acknowledgement after the driver
    has taken a notice does not remove it from that run.
  - `admit` (sync or async) is called before each follow-up and cancelled by
    preemption.
  - An agent without background tools allocates no store and registers no run id.
- `python/tests/core/test_session_events.py`: posting (validation, key
  replacement only while pending, receipts and their observer), waking a waiter
  blocked on children, release ahead of newer work, the notice (byte-identical
  without events, labelled blocks, neutralized closing tag, client note), a turn
  receiving an event posted by run id, an event waking a waiting driver.
- `python/tests/voice/test_voice_events.py`: event turns spoken and never the
  caller's words, still in memory later, silence with no filler; waiting for
  playback and for caller speech; barge-in on an event turn; a caller who speaks
  first, or a busy line, gets the event with their turn (in-memory and JSONL);
  an event before any turn; key replacement and batching; announced versus
  left-for-the-caller background results; client events refused, screened and
  labelled; events before the call starts and at hang-up; an event turn cut
  before its run; `current_voice_session()`; the quiet-line check.
- `python/tests/server/test_voice_notify.py` and
  `test_livekit_session.py::TestNotifyFromTheRoom`: frame parsing, acks and
  rejections, the WebRTC data channel, LiveKit sender attribution, and
  server-only enabling.

These tests were checked against mutations of the preempt/reject decision, the
store's owner inheritance, the agent draining events, the quiet-line check,
`release()`, the event turn's done callback, store adoption, LiveKit sender
attribution, client screening, and closing-tag neutralization; each mutation
fails at least one of them. Existing background, runnable, tracing, guardrail,
and voice tests remain regression gates.

Not yet tested: two producers/retries with idempotency, a bounded inbox and
overflow; deployment owner routing; trace-save failure reporting; the quiet-line
windows on real LiveKit/STT/TTS timing (0.6 s quiet and 5 s stranded partial
are carried from #214, not measured).

Merge/readiness decisions: agree the driver's API and output contract, the
`SessionEvent` receipt, and the trust rules for client frames (server-only
opt-in, guardrails, labelling); then the durable/cross-worker inbox as a
separately reviewable change.
