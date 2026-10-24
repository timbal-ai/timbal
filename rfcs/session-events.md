# RFC: event-triggered turns across chat and voice

Status: **draft; design plus a first implementation covering #214's use case**.
Reviewed against main at `49af46e9` on 2026-10-23 (after #216 and #223 merged).
Phase 1 is implemented on this branch as described below:
- session events;
- per-producer announce policies (`when_idle`, `silent`, `interrupt`);
- progress from background tasks;
- one `SessionScheduler` shared by the text driver and the voice adapter.

"Implementation notes" records where implementing it changed earlier revisions
of this text. What remains of Phase 2 (durable inbox, authenticated session
identity, cross-worker ownership, idempotency keys, expiry) is a proposal, not
a shipped interface.

## Decision in brief

A task finishing is an input to a conversation, not a caller utterance and not
an instruction to speak immediately. Separate three responsibilities:

1. Record a correlated event/result for the owning conversation.
2. Decide whether and when another agent turn may start.
3. Deliver its output through chat or voice, with channel-specific interruption
   and playback rules.

The shared unit is the **session store**: the existing background-task store
gains an event inbox, and both kinds of input (finished tasks, posted events)
are drained and formatted by the same primitives (`take_followup`,
`format_session_notice`). The channel that owns the session holds a
**`SessionScheduler`** (`core/session_scheduler.py`) over that store. It
resolves each item's announce policy, decides when a turn of the session's own
is due, and admits every other root turn on the session. Each channel keeps its
own turn loop on top of it:

- Text: `AgentContinuation` (with `Agent.stream_until_idle(...)` as sugar), a
  caller-driven stream of ordinary runs.
- Voice: `VoiceSession`, whose turns already run through `_run_turn` (TTS,
  fillers, barge-in, heard-prefix memory, metrics), adds an event loop gated by
  its quiet-line check.
- Any turn of the session, including a plain `agent(...)` call, receives what
  is pending ahead of its prompt, so nothing waits for a driver to exist.

Each producer says how much attention its item deserves: a background tool
(`background_announce=`), a posted event (`announce=`), or a progress report.
`when_idle` starts a turn once the channel is free, `silent` waits for the next
turn, and `interrupt` cuts in where the channel can (voice). Unset, the
channel's default applies.

It does not create a second task store, make `VoiceSession` the owner of
conversation state, promote client text to trusted runtime context, or change
the lifetime of existing agent calls. Event turns are opt-in: calling
`notify()`/`post_session_event()` is the opt-in for events; finished tasks start
a voice turn only with `announce_background_results` or the tool's own
`background_announce`.

Not implemented: events surviving a process restart or reaching another worker,
wakeups for a text session with no driver running (its next turn still gets
them), interrupting a text run in flight, and an HTTP route for posting events.

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
  once instead of a `bool` after placement; its `now` policy returns as
  `announce="interrupt"`, which the app may use and the caller's device may not.
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

Checked against current documentation on 2026-10-23 (documentation, not
experiments):

- LiveKit Agents **1.6** async tools: the first `ctx.update()` hands control
  back to the LLM while the tool keeps running. Later updates are coalesced into
  a deferred reply, spoken once both the agent and the user are idle, and the
  final return arrives the same way. `report_progress` follows that model:
  keyed per task, so a newer report replaces an unread one, and delivered
  `when_idle`.
- Gemini Live `FunctionResponseScheduling`: `SILENT` (context only),
  `WHEN_IDLE` (default; no interruption), and `INTERRUPT` (cut in), chosen per
  function response. The announce policy has the same three values per
  producer. Here, though, `silent` is the voice default for finished tasks
  (today's behaviour) unless `announce_background_results` is set.
- LangGraph Agent Server "double texting": `enqueue` (the default), `reject`,
  `interrupt`, `rollback`, per run on a thread. `on_busy` offers `reject`
  and `enqueue`, defaulting to `reject`. Here the competing turn is an
  in-process `agent(...)` call, and queueing it blocks the caller for however
  long the owner's turn takes (a whole spoken reply), so that is opt-in. A
  text run is never cut mid-flight, and nothing rolls back.

The useful commonality is not one universal Session class. It is the separation
of event receipt, execution ownership, and output policy.

## Phase 1: session events, text driver, voice adapter

### Session events

```python
from timbal.state import post_session_event

event = post_session_event(run_id, "The nightly export finished.", key="export", announce="when_idle")
event.state            # "pending" -> "delivered" (event.run_id) | "superseded"
await event.settled()  # observer; cancelling it does not withdraw the event
```

- The event goes to the `BackgroundTaskStore` of `run_id`'s session; any run id
  of a session that has a store reaches it, and so does a run an owned session
  continued from before it had one (see "Store lifetime"). A session without one raises
  `LookupError` unless `create=True`, which registers a store under `run_id`.
  Only the turn chained directly on `run_id` finds that store, so a stale run
  id fails loudly instead of posting into a store no turn will read.
  `BackgroundTaskStore.post_event` is the same thing given a store.
- Nonblocking and thread-safe: posting never waits for a turn, and a post from
  another thread (a webhook handler, `offload_blocking`) hops onto the store's
  loop, as `emit()` does. Text is capped at 4,000 characters. A pending event
  with the same `key` is superseded (receipt state `superseded`); one a turn
  has already taken is not replaced.
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
- Announce policies decide whether an item *starts* a turn. Every turn still
  receives every pending item, `silent` ones included.
- The receiving run counts what it got as usage
  `session_events:delivered`. The guardrail records from screening a client
  event land on the receiving run as `metadata.guardrails`, the same report
  input rails write.
- A turn the session starts itself takes everything with `take_followup`. The
  receipts settle as delivered only once that run has started (its stored input
  then holds the notice); if it never starts, `Followup.release()` hands the
  work back, ahead of anything newer, superseding an event whose key a newer
  pending event now carries.
- The text is wrapped in `<session_event id source key>` blocks; a closing tag
  inside the text is neutralized so it cannot speak outside its block.

### Announce policy

```python
Tool(name="build", handler=..., background_mode="always", background_announce="when_idle")
post_session_event(run_id, "Deploy failed.", key="deploy", announce="interrupt")
await session.notify("Your table is ready.", announce="interrupt")
```

| Policy | Text driver | Voice |
| --- | --- | --- |
| `when_idle` | follow-up after the in-flight run | event turn once the line is quiet |
| `silent` | no follow-up; reaches the next turn | no event turn; reaches the caller's next turn |
| `interrupt` | same as `when_idle` | event turn now: cuts the reply in flight |

- Per item: `Runnable.background_announce` for a child's completion (copied
  onto its task record at spawn), `announce=` on `post_event` /
  `post_session_event` / `notify` / `report_progress`. `None` means the
  channel's default, set on its `SessionScheduler` (`completions=`, `events=`).
  Defaults: `when_idle` for both in the text driver. In voice, `when_idle` for
  events, and `silent` for completions unless `announce_background_results`.
- The scheduler keeps waiting only while something pending, or a running child
  that is not `silent`, could still ask for a turn. A session whose remaining
  children are all silent ends the driver `idle`, not `idle_timeout`.
- Voice `interrupt` skips the quiet-line wait and the caller's held fragment
  (disarmed as `submit_user_text` does), and calls `interrupt()` on the reply
  in flight, as a barge-in would. It is still blocked by a closed or unstarted
  call, a run parked on the caller (that run's prompt would be dropped on
  resume), and another channel's turn that is running. It goes ahead of other
  turns still queued behind the reply it cuts. Cutting that reply finishes the
  call's turn, which would let a queued turn in first, so the session is held
  (`SessionScheduler.holding()`) from the cut until the interrupt's own turn is
  busy. A turn it starts records `trigger.announce="interrupt"`.
- Client events cannot interrupt: `notify(source="client", announce="interrupt")`
  raises `PermissionError`, and the transports answer `not_allowed`.

### Progress from background tasks

```python
async def build(target: str):
    for phase in ("compile", "link", "test"):
        await run_phase(phase)
        report_progress(f"{phase} done")  # from timbal.state
    return "built"
```

- `report_progress(text, *, announce=None)` posts an event keyed
  `progress:{task_id}` to the task's session store, carrying `task_id`. It only
  works inside a background child: the spawn paths set the task id on the
  child's context. Outside one it raises `RuntimeError`.
- A newer report replaces one no turn has taken yet; the task's completion
  notice supersedes any report still pending, and `requeue` drops reports of
  tasks that have finished meanwhile. So the model never gets "50%" after
  "done".
- Rendered with `task_id="..."` on the block and a note that the task is still
  running and must not be presented as finished.

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
    on_busy="enqueue",  # other root turns wait for an in-flight follow-up instead of raising
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

One `SessionScheduler` owns a store at a time: a text driver's or a voice
call's. The store holds it by weak reference. Binding a store another
channel's scheduler owns raises. The owner's own runs carry a marker on their
`RunContext`, and a store a spawn creates inherits it. Any other **root turn**
on an owned session goes through `SessionScheduler.admit_foreign`: a direct
`agent(...)` call, the first run of a new driver, or a text turn on the
conversation of a live call. It is handled by the owner's timing:

- Owner idle (a driver waiting between runs or in `admit`, a call with no turn
  running): the turn **preempts**. A driver ends with `stop_reason="preempted"`
  and releases the store; the turn proceeds, and any pending notices attach to
  it through `_inject_background_completions`. One run, nothing lost, human
  first.
- Owner busy (from the take that commits to a turn of the owner's until that
  run's root `OUTPUT`, or while another admitted turn runs):
  - `on_busy="reject"` (default): the turn raises `SessionBusyError` before
    parameter resolution, hooks, or the model run, telling the caller to wait
    for that `OUTPUT`;
  - `on_busy="enqueue"`: the turn waits there instead, then is admitted (a
    waiting driver is preempted). If the owner let go while it waited, it asks
    whoever owns the store now.

  Running two turns of one session concurrently is the one thing the owner
  guard exists to prevent.

While an admitted turn runs, the owner starts nothing of its own. Its event
turns wait (`wait_ready`, `take`), and a voice turn for a caller who spoke waits
for running turns only, not queued ones, which would deadlock: they wait for
it. Queued turns also hold back the owner's event turns, except a voice
`interrupt` (see "Announce policy"). Admitted turns leave through a finish hook
(`RunContext._foreign_turn_of`, called from `Runnable._stream`'s `finally` after
the trace save), so a crashed or cancelled one cannot wedge the owner.

The session stays one line of turns. `SessionScheduler.head` is the session's
latest root run: the owner notes each of its runs, and an admitted turn
becomes the head. A turn may be admitted with an older run as its parent,
because it was queued while the owner's turn ran, or because its client had
not seen the latest turn yet. Such a turn is re-chained onto the head before
it reads anything (`RunContext._continue_from`: new `parent_id`, session data
reloaded from it, the requested parent kept as `metadata.requested_parent_id`).
Admission is after `START`, so that event still shows the requested parent;
`OUTPUT` and the stored trace show the adopted one. Without this, a chat
queued behind a voice turn would answer without that exchange, and the call,
chaining onto the chat, would lose it for good. A voice call stays the owner
after another channel's turn ran, and its next turn chains onto a newer head.
That includes turn one, whose seeded context is re-chained the same way,
because the reuse path ignores a `parent_id` keyword.

The check runs in the root path of `Runnable._stream`, right after `START` and
before parameter resolution, and it is async (enqueue awaits there). It cannot
live in `Agent.handler`: `pre_hook` runs in `Runnable` before the handler, so a
refused turn would already have run its hook. Sessions without an owned store
pay two attribute reads.

### Store lifetime

The driver must not allocate a `BackgroundTaskStore`, or register run ids in
the process-local map, for an agent that never spawns a child. The owner is
stashed on the `RunContext`; `ensure_background_store` copies it onto the store
it creates on the spawn path. After the first run, no store means `idle`.
Releasing the owner does not delete map entries by itself. When a text driver
or a voice call ends, `forget_idle_background_store` drops every run id
registered for its store if nothing in it can matter again (no children,
retained ones included, no pending notices or events, no owner). Otherwise the
existing retention/reaping rules own it. A voice call that only posted events
no longer leaves one map entry per turn behind.

A store also records the runs its session continued from before it existed
(`_joined_from`): the parent of the run that created or adopted it, when that
parent has no store of its own. Typically that's the run a voice call joined
(`parent_run_id`), or the run a driver was started on. A root turn chained
onto such a run inherits nothing, since no map entry leads from it. Yet a chat
tab that last saw that run is still part of the conversation. So while a
channel owns the store, admission finds it there (`store_joined_from`, a scan
of the few owned stores), and the turn is admitted, refused or queued like
any other. Once admitted it adopts the store and is re-chained onto the head.
`post_session_event` and `continuation_for_run` look there too, and so does a
new driver's first run. With no owner, nothing changes: a branch from that run
does not pick up the session's notices.

Known gap, unchanged from today: two root turns on one session that both start
before either spawns a child are not serialized by anything. The guard above
only exists once a store exists. Runs further back than the store's join
point (two or more turns before it existed) are not found either.

### Trace shape

- Initial run: `input.prompt` is the human prompt, as today.
- A turn the session starts itself (text follow-up or voice event turn):
  `input.prompt` is the notice (runtime user message; kind
  `background_task_completed`, or `session_event` when it carries events);
  `metadata.trigger = {"kind": "background_completion" | "session_event",
  "task_ids": [...], "event_ids": [...]}`, plus `continuation_index` from the
  text driver and `announce: "interrupt"` for a turn an interrupt started.
- A human turn that receives pending work has no `trigger`; the notice sits in
  its memory before its prompt, and `metadata.session_events` lists the event
  ids it received.
- Either way, the receiving run's usage has `session_events:delivered`, and a
  screened client event's guardrail records are in its `metadata.guardrails`.
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
  for hypothetical future tasks. Neither is work that can never ask for a turn:
  only `silent` items pending and only `silent` children running is `idle`.
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

- `await session.notify(text, key=None, source="app", announce=None) -> SessionEvent`
  posts to the call's conversation store and returns at once. The store is made
  on demand and bound to the call's scheduler; one made before any turn has a
  run is adopted by the first turn's run, or, if that run inherited its
  conversation's store, hands its pending events over to it.
- One watcher task per call, started by `notify()`, at the end of any turn
  after which something could still ask for a turn (`may_need_turn()`), and
  whenever the store gains work (the scheduler's `on_work`, called by the store
  for every finished task and event). That covers work that arrives between
  turns without `notify()`: a `report_progress` from a task that was silent
  when the turn ended, or a `post_session_event` from the app's backend. Waiting
  for work is the scheduler's `wait_ready`, event-driven on the store; waiting
  for the line polls the quiet-line check every 50 ms, because "the caller's
  audio has finished playing" is an estimate with no event to wait on.
- Quiet line: session started and open, the run the next turn chains onto not
  parked on its caller (the call's last run, or another channel's turn that ran
  since, which the scheduler records as `head_parked` when it ends), no
  other channel's turn running or queued, no turn running, agent not speaking,
  no greeting or idle prompt pending, playback finished, no held caller fragment,
  no uncommitted partial in the last 5 s, nothing from the caller in the last
  0.6 s, local VAD silent. Only an `interrupt` item skips the second half (see
  "Announce policy"); once an event turn has started, the caller barges in as on
  any reply.
- The event turn starts under the same lock as caller turns, right after the
  scheduler's `take()`, as `_run_turn(followup=...)`: the notice is the prompt,
  fillers are off, `TurnMetrics.event` is true, the root span gets `trigger`,
  and the receipts settle at the run's first event. A task cancelled before its
  first step releases the work from a done callback, since its `finally` never
  runs.
- The client gets `EventTurnStarted` (`{"type": "event_turn_started",
  "task_ids", "event_ids", "keys", "urgent"}`) when one starts, so a UI can tell
  an announcement from a reply to the caller before any text arrives.
- Another channel's turn on the call's conversation (a chat tab on the same
  `parent_id` chain) is admitted by the call's scheduler like any foreign turn.
  `VoiceConfig.events.on_busy` picks reject or enqueue while a call turn runs.
  A caller who speaks meanwhile waits for it, and the call's next turn chains
  onto it.
- A caller who speaks first gets everything pending with their turn, through
  the agent's normal injection: #214's "overdue" case, without an edit of the
  previous run.
- Finished background tasks start an event turn only with
  `announce_background_results` or the tool's `background_announce`;
  otherwise they wait for the caller's next turn as before. When an event turn
  happens anyway, finished tasks ride it.
- `source="client"` needs `client_events` and is screened by the agent's input
  guardrails. A block raises `PermissionError` and logs
  `voice_client_event_blocked`; the records of a rail that let it through
  (redacted, warned) travel with the event to the receiving run. WebRTC
  data-channel frames and LiveKit frames from the caller are client events;
  LiveKit frames with no participant (the server API's `SendData`, i.e. the
  app's backend) are app events; other participants are ignored. Frames carry
  `announce` (`policy: "when_idle"` from #214 is accepted as an alias).
  `VoiceConfig.events` (`announce_background_results`, `client_notify`,
  `on_busy`) is read from server config only, never from the hello.
- Events undelivered at hang-up stay pending in the conversation's store for
  its next turn (a text turn continuing the call), and are logged. A store with
  nothing left in it is forgotten (see "Store lifetime").
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
- The text driver and the voice watcher each reimplemented "wait, admit, take,
  start", and only the driver owned the store. A chat turn on a live call's
  conversation was not admitted by anyone. Both now sit on one
  `SessionScheduler`: ownership, policy resolution, waiting, taking, and
  foreign-turn admission live there, and the channels keep only how a turn
  starts (an agent run vs. `_run_turn`).
- Interrupting comes back as a per-item policy instead of #214's session-wide
  `now`. Rejecting busy turns becomes an `on_busy` choice, not the only answer.
- `post_session_event` with an unknown run id raised nothing and posted into a
  store no turn would read; it now raises unless `create=True`.
- Review of the scheduler found five gaps, now covered by tests:
  - **Stale parent.** An admitted turn kept the parent it was sent with, so a
    chat queued behind a voice turn forked the conversation. It is now
    re-chained onto the session head.
  - **Dropped head on turn one.** The call's first turn consumed and ignored a
    chat admitted before it. The scheduler now has a `head` instead of a
    consumed `take_head()`.
  - **Interrupt waited for queued turns.** An `interrupt` waited for turns
    queued behind the reply it should cut. It now goes ahead of them, holding
    the session across the cut.
  - **No watcher for work between turns.** Work posted between turns started
    no voice watcher unless it came through `notify()`. The store now tells
    its owner.
  - **Parked chat head.** The pause check looked only at the call's own last
    run. An admitted chat turn parked on an approval or `ask_user` let an
    event turn start onto it: the receipt said delivered, the resume dropped
    the prompt, and the event was gone. The check now follows the head, and
    the event waits until that pause is resolved.
  - **Parent from before the store.** A chat turn naming the run a call
    continued from, created before the call made its store, inherited
    nothing. So it was never admitted: it ran beside a call turn and was never
    re-chained. Stores now record that join point (see "Store lifetime").
    The same test exposed a separate bug: `notify()` before turn one of a
    seeded call (`parent_run_id` / `call_context`) hung the store on the seed,
    which the watcher never looked at, so the event waited for the caller.
- A second review (of `43b4c4aa`) found ten more, each with a test:
  - **Queued turns after the owner let go.** A turn queued with
    `on_busy="enqueue"` re-asked the store once its owner released it (a
    stopped driver, or one a turn just admitted preempted), found no owner, and
    ran unadmitted: not re-chained, and beside the turn admitted just before it.
    A channel's release now waits for the turns admitted or queued on its
    scheduler, so they still go one at a time onto the head, and a channel
    taking the session over meanwhile (a new driver, a call) waits behind them.
    Another channel's own run admitted there takes the session at once, and
    turns still queued then ask the new owner.
  - **Session data read too early.** `Runnable._stream` loaded session data
    before admission, so a turn queued behind the run it names read nothing
    (that run had not saved yet), including session-scoped approval grants. It
    now loads after admission, and a refused turn reads nothing.
  - **Inherited store unowned until turn one.** A call continuing a
    conversation that already had a store did not own or watch it until its
    first turn or `notify()`, so what that store held or finished waited for the
    caller, and chat turns on it were not admitted. The call takes it at start.
  - **Resume over HTTP invisible to the call.** A call turn parked on an
    approval or `ask_user` before the call had a store: its HTTP resume
    inherited nothing, so the call never admitted it. Events stayed blocked on
    the old pause, and the caller's next turn chained onto the resumed run.
    A turn that asks its caller now makes the store.
  - **Interrupt onto a parking turn.** An `interrupt` event arriving as a call
    turn asked its caller cut that turn and started onto it, which resumed it
    and swallowed the event. A turn counts as parked once it has asked, and the
    pause is checked again after the cut.
  - **Chat turns ahead of the caller.** Cutting a reply finished the call's
    turn before the caller's next one was marked busy, so a queued chat turn
    slipped in between; and a caller turn waiting out a running chat turn was
    not busy, so a chat turn queued behind that one went first. Every cut is
    now held until the caller's turn starts, which is busy from its commit.
  - **New driver on a call's session.** Its first run went through the call's
    admission (moving the call's head) and then failed to bind. It now fails
    before; only another driver gives way to it.
  - **Join point not searched** by `post_session_event`,
    `continuation_for_run`, a new driver's first run, or a call continuing from
    it (whose `notify()` then made a second store).
  - **Invalid `notify` frames** were dropped instead of answered `invalid`.
  - **Client event screened past hang-up** re-registered the forgotten store.

`Agent.handler` is unchanged from `main`; `_inject_background_completions` also
drains posted events and accounts for them. The rest is
`core/session_scheduler.py`, `core/continuation.py`, the event inbox and
announce/progress plumbing on `BackgroundTaskStore`, two `RunContext`
attributes, the admission call and finish hook in `Runnable._stream`,
`Runnable.background_announce`, the voice event loop, and the transport frames.

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

- `silent`: retain for the next eligible turn; never start one itself.
  Phase 1: `announce="silent"`, and the voice default for finished tasks
  without `announce_background_results`.
- `when_idle`: start a turn once the owner/channel admits it; batch ready
  events. Phase 1: the default for events, and for finished tasks in text.
- `interrupt`: explicit privileged policy, not a default or an expiry fallback.
  Phase 1: per item, app only (client events cannot use it). Voice cuts in
  through `interrupt()`, so the heard-prefix memory of the cut reply is kept.
  Text has no mid-run cut; there it acts as `when_idle`.

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
    hook record) and the follow-up finishes. The same two cases for a new driver,
    and a new driver started on the pre-store run the first one started from.
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
- `python/tests/core/test_session_events.py`:
  - posting: validation, key replacement only while pending, receipts and their
    observer, `create=` versus `LookupError`, a post from another thread landing
    on the loop's thread;
  - waking a waiter blocked on children; release ahead of newer work;
  - the notice: byte-identical without events, labelled blocks, neutralized
    closing tag, client note;
  - a turn receiving an event posted by run id (with the usage key); an event
    waking a waiting driver;
  - announce: a silent event riding a turn something else started, a silent
    task starting no follow-up, `interrupt` acting as `when_idle` for text,
    unknown policies refused;
  - progress: only inside a background task, newer report replaces, completion
    (and a report arriving after it) supersedes, the follow-up told the task is
    still running;
  - an enqueued turn waiting out a follow-up, then preempting, re-chained onto
    that follow-up (`parent_run_id`, `requested_parent_id`, what the model saw);
    the same when the driver stops during the follow-up, with a new driver
    waiting behind that turn; two enqueued turns run
    one after the other, each chained onto the one before; an enqueued turn
    naming the in-flight follow-up reading the session data it saved;
  - `RunContext._continue_from` reloading session data from the new parent;
  - an idle store forgotten.
- `python/tests/voice/test_voice_events.py`: event turns spoken and never the
  caller's words, still in memory later, silence with no filler; waiting for
  playback and for caller speech; barge-in on an event turn; a caller who speaks
  first, or a busy line, gets the event with their turn (in-memory and JSONL);
  an event before any turn; key replacement and batching; announced versus
  left-for-the-caller background results; client events refused, screened and
  labelled; events before the call starts and at hang-up; an event turn cut
  before its run; `current_voice_session()`; the quiet-line check. Also:
  - a tool announcing itself or staying silent; an `interrupt` event cutting a
    reply in flight; the caller's device refused `interrupt`; a silent event
    waiting for the caller;
  - progress narrated while the task runs;
  - another channel's turn refused while a call turn runs, queued, re-chained
    onto the call turn it waited for, and then chained onto by the call, with
    nothing dropped from either side's history; waited for by a caller's turn;
  - an `interrupt` going ahead of a queued chat turn, with a slow `interrupt()`
    standing in for the trace re-save that opens the gap after the cut;
  - a chat turn admitted before the call's first turn, which then continues it;
  - an event waiting while an admitted chat turn is parked on an approval, then
    delivered by a turn chained onto the resumed chat;
  - a chat turn naming the run the call continued from (before the call's
    store existed) queued behind a call turn, re-chained onto it, joining the
    store;
  - an event `notify()` posted before turn one of a call continuing a
    conversation, spoken without waiting for the caller;
  - progress reported after the reply by a silent task, and an event posted by
    run id from outside the call, each starting an event turn;
  - an event `notify()` queued before turn one reaching the store that turn
    inherits;
  - a continued call speaking up about an event already in the store it
    inherits; an event posted by the run the call joined before it had a store;
    a call on the run a driver's session joined sharing that session's store;
  - an event delivered after a parked call turn is resumed over HTTP; an
    `interrupt` event left pending when the turn it would cut asks its caller;
  - the caller's turn ahead of a queued chat turn, both on a barge-in and when
    the chat is queued behind another running one; a turn cut before its first
    step leaving the call idle;
  - a new driver refused on a call's session without moving the call's head; a
    client event screened past hang-up refused, its store left forgotten;
  - screening records on the receiving run; a quiet store forgotten at hang-up;
    `EventTurnStarted`.
- `python/tests/server/test_voice_notify.py` and
  `test_livekit_session.py::TestNotifyFromTheRoom`: frame parsing, acks and
  rejections (an invalid frame answered `invalid`), the WebRTC data channel,
  LiveKit sender attribution, and server-only enabling.

These tests were checked against mutations; each mutation fails at least one
of them. Mutated:
- the preempt/reject decision and admitting a busy owner's foreign turn;
- the store's owner inheritance and store adoption;
- the agent draining events, and delivery accounting;
- the quiet-line check, and `interrupt` waiting for a quiet line;
- `release()` and the event turn's done callback;
- LiveKit sender attribution, client screening, and closing-tag neutralization;
- ignoring an item's own policy;
- the progress key, and completion superseding progress;
- the thread hop onto the store's loop;
- the call adopting the other channel's head, and a call turn not waiting for
  a running other turn;
- keeping an idle store at hang-up;
- an admitted turn keeping its stale parent, and session data not reloaded on
  re-chaining;
- the first voice turn ignoring the head, and the seeded turn one not
  re-chained;
- an interrupt not going ahead of a queued turn, and no hold across the cut;
- no watcher started on new work;
- events queued before turn one orphaned;
- the pause check ignoring a parked head, and the scheduler never recording
  one;
- no lookup of the store a pre-store parent joined, the join point never
  recorded, and the admitted turn not adopting the store;
- the seeded context not searched for the call's store;
- a new driver's first run not looking up the join point, a call turn's
  task done callback, and the hold across a caller's cut (each of this
  round's ten findings also fails its test on the unfixed code). Marking the
  caller's turn busy before it waits survives on its own: the hold's wakeup
  already puts the caller ahead; with the hold removed too, it fails.

Existing background, runnable, tracing, guardrail, and voice tests remain
regression gates.

Not yet tested: two producers/retries with idempotency, a bounded inbox and
overflow; deployment owner routing; trace-save failure reporting; the quiet-line
windows on real LiveKit/STT/TTS timing (0.6 s quiet and 5 s stranded partial
are carried from #214, not measured).

Merge/readiness decisions: agree the driver's API and output contract, the
`SessionEvent` receipt, and the trust rules for client frames (server-only
opt-in, guardrails, labelling); then the durable/cross-worker inbox as a
separately reviewable change.
