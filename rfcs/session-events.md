# RFC: event-triggered turns across chat and voice

Status: **draft; design plus a deliberately limited first implementation**.
Reviewed against main at `49af46e9` on 2026-10-23 (after #216 and #223 merged).
Names for the later session API are proposals, not shipped interfaces. The
"Implementation status" section lists where the current branch still differs
from this text.

## Decision in brief

A task finishing is an input to a conversation, not a caller utterance and not
an instruction to speak immediately. Separate three responsibilities:

1. Record a correlated event/result for the owning conversation.
2. Decide whether and when another agent turn may start.
3. Deliver its output through chat or voice, with channel-specific interruption
   and playback rules.

The first implementation is a **hostable continuation driver** over the existing
background-completion inbox. `Agent.stream_until_idle(...)` is sugar over it for
the local, caller-driven case. The driver is the unit both channels share: the
HTTP job layer and `VoiceSession` already own long-lived loops, so the reusable
piece has to be something they can host, not an agent method that only lives
while a request iterates it.

It does not create a second task store, make `VoiceSession` the owner of
conversation state, expose untrusted event frames, or silently change the
lifetime of existing agent calls.

This PR is **not** a complete replacement for #214. External notification
ingress, arbitrary session wakeups after a driver has ended, and the voice
adapter remain unimplemented. Do not merge it as a claim that those product
requirements are finished.

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
  input guardrails. Carry forward from its branch (`voice/session-notify`) into
  a voice adapter PR once the driver lands:
  - the admission predicate (`when_idle` rules: no turn running, playback
    finished, no greeting/filler/held fragment, quiet window, VAD silent) as
    the driver's `admit`;
  - `current_voice_session()` and `TurnMetrics.event`;
  - the transport frame parsing and tests, with client `notify` frames opt-in
    (`VoiceConfig`) and routed through input guardrails as ordinary input;
  - an overdue note stays pending and attaches to the **next** run as a
    runtime message, the same way background completions do;
  - event turns tagged with the same `metadata.trigger` as below;
  - receipt states instead of `bool` (at minimum `own_turn`, `deferred`,
    `replaced`, `call_ended`), aligned with Phase 2.
- [#218: end call](https://github.com/timbal-ai/timbal/pull/218): was stacked on
  #214's branch. Its design is unaffected by this RFC: keep hang-up-after-playback
  as a voice policy, not a generic session close, and count partial speech/VAD
  as caller activity. Retarget it to `main` and bring `current_voice_session()`
  with it; #215, which it needed for `session_ended` on WebRTC, is merged.
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

## Phase 1: continuation driver

### API

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

# Hosted: the channel owns the loop and decides when a follow-up may start.
stream = AgentContinuation(
    agent,
    admit=session.wait_until_quiet,  # awaited before each follow-up; may be None
    max_idle_secs=300,
    max_continuations=10,
    kwargs={"prompt": ..., "parent_id": ...},
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

- `admit` is an optional parameterless awaitable. The driver awaits it after a
  completion is pending and before it starts the follow-up; when it returns, the
  driver starts immediately, so it *is* the start boundary. A preemption or
  `stop()` while `admit` is pending cancels that wait. The generic driver has no
  quiet windows or playback knowledge; those live in the channel's `admit`.
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
   driver acquires it before the run starts.
2. Streams the run to completion, including tracing finalization.
3. Stops on unsuccessful output, approval, or interaction suspension.
4. Waits on the store's completion signal without polling or draining. Already
   finished tasks are reconciled before declaring quiescence, including the
   interval before their done callbacks have executed. A direct human turn on
   the session preempts this wait (see "Ownership and preemption").
5. Verifies readable parent history and no unresolved tool calls, then awaits
   `admit` if given.
6. Drains the pending notices and starts another ordinary run with the formatted
   completion notice as its `prompt`. The notice is a `role="user"` message with
   `metadata.source="runtime"`, so it is not a human utterance, is hidden from
   transcripts by `Message.is_runtime()`, and skips input guardrails like every
   runtime message. No previous-trace edit occurs.
7. Repeats until quiescent or a stop condition is reached.

Draining in the driver rather than in the receiving run means the notice is in
the follow-up's stored `input` from its `START`. An acknowledgement that arrives
after the driver has taken the notices does not withdraw them; one that arrives
before (model poll during the previous turn, external `ack_completion`) means
the inbox is empty and no follow-up starts.

### Ownership and preemption

Only one driver may own a store. A competing driver on the same session fails
before parameter resolution, hooks, or model execution.

A **direct root turn** on an owned session is handled by its timing:

- Driver waiting between runs (completion signal or `admit`): the direct turn
  **preempts**. The driver ends with `stop_reason="preempted"`; the direct turn
  proceeds and any pending notices attach to it through the existing
  `_inject_background_completions`. One run, nothing lost, human first.
- Driver with a follow-up run in flight: the direct turn is **rejected** before
  hooks or model execution, with an error telling the caller to wait for the
  in-flight run or `stop()` the driver. Running two turns of one session
  concurrently is the one thing the owner guard exists to prevent.

This is the Phase 1 form of "human input takes priority": the common case (a
user types while the agent waits for a task) produces a normal turn, not an
error. Hosts that want queueing instead of rejection for the in-flight case
build it on `stop()` and `continuation_for_run`.

The ownership check runs on the orchestrator's root path (`Agent`), not in
`Runnable` for every runnable.

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
- Follow-up run: `input.prompt` is the completion notice (runtime user message,
  kind `background_task_completed`); `metadata.trigger = {"kind":
  "background_completion", "continuation_index": n, "task_ids": [...]}`.
- A preempting human turn has no `trigger`; the notices it carries sit in its
  memory before its prompt, as for any next turn today.
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

## Implementation status

The branch at the time of this revision implements an earlier form of Phase 1.
Deltas against this text, to be closed before merge:

- Follow-ups start with an empty-content runtime `prompt`
  (`kind="background_wakeup"`) and let the receiving run drain the inbox; this
  needs a `_background_continuation` flag and a `continuation_skipped` early
  return in `Agent.handler`, and persists an empty user message as the run's
  `input`. Target: drain in the driver, notice as `prompt`, both branches gone.
- A direct root turn is rejected whenever a driver owns the store. Target:
  preempt while waiting, reject only while a follow-up is in flight.
- `AgentContinuation` has no `admit`, no `stop()`, and the owner is an opaque
  `object()`, so nothing can find the driver from a run id. Target: `admit`,
  `stop()`, `continuation_for_run`.
- The driver calls `ensure_background_store` on every root run and deletes
  run-id map entries on release to compensate. Target: owner on `RunContext`,
  copied by the spawn path.
- `check_continuation_turn` is called from `Runnable._stream`. Target: the
  agent's root path.
- `run_stopped` does not carry the run's status code.

## Phase 2: proposed shared session contract

This section is a design target, not an API implemented by Phase 1.

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

An event records an immutable ID, authenticated producer, session identity,
kind, causal task/run IDs, bounded payload/reference, and admission time.
Idempotency keys suppress repeat delivery of the same event; replacement keys
coalesce pending updates. Do not conflate the two. Same idempotency key with a
different payload is a conflict, not a silent overwrite. Claimed/attached events
are not replaced behind a running model's back.

Ingress acknowledges acceptance promptly, returning an event ID and receipt
state. It must not await completion of a future turn: a tool awaiting that turn
would deadlock the currently running turn. Waiting for receipt changes is a
separate observer operation whose cancellation does not withdraw the event.

Start with server-generated task completions. A browser may request an action
or submit ordinary input; it must not assert `source="runtime"` or bypass input
guardrails by sending a notification frame. Authenticated tool results remain
untrusted data, not instructions. Event validation, payload limits, admission
policy, and any result screening are independent of transcript visibility.

### Scheduling versus delivery

Proposed scheduling choices:

- `context_only`: retain for the next eligible turn; never start one itself.
- `when_idle`: start a turn once the owner/channel admits it; batch ready events.
  Phase 1's driver with an `admit` hook is this policy for one event kind.
- `interrupt`: explicit privileged policy, not a default or an expiry fallback.
  Voice must preserve its heard-prefix interruption behavior.

An event waiting too long does not silently become an interruption. Expiry
reports a reason and preserves whatever source/task result remains available.
Events cannot bypass approvals or interactively suspended tool calls. Results
remain outside the pending tool-use/tool-result sequence until it is safe to
attach them. Human input preempts pending autonomous turns, as in Phase 1.

The inbox lifecycle is `pending -> taken -> attached`, or a terminal
`withdrawn / superseded / expired / closed` before being taken. Taking and turn
admission are atomic for one owner. Cancellation before the run's `START` is
stored releases the event back to pending; after that the event is in the run's
stored `input` and recovery goes through the trace, never by re-enqueueing.
Attachment records the receiving run ID and belongs to that run's memory, never
an older trace.

Track separately: accepted into local/durable storage, attached to model context,
run completed/failed/interrupted, and output delivered. Text generated, audio
sent, and audio played are distinct facts. No `True/False` receipt can represent
all of these. At-least-once transport and deduplicated admission still do not
provide exactly-once tool side effects or exactly-once audible announcements.

### Voice adapter

The adapter contributes `admit` (no foreground turn, outstanding playback,
greeting/filler, held fragment, or active/recent caller speech) and wakes the
driver when that changes. Quiet windows need measured defaults; they live in the
adapter, not the generic driver.

Both user and event turns enter the same voice owner. Barge-in stops the
foreground response, persists heard-prefix history, and leaves detached jobs
alone unless explicit application policy cancels them. Hang-up stops voice
delivery, not necessarily the underlying conversation's work. A separate
application policy decides whether the completion is later presented in chat.

## Acceptance gates and rollout

Phase 1's executable tests are in `python/tests/core/test_continuation.py` in
the repository. They cover multi-run chaining; in-memory and JSONL history; no
historical edits; no extra human message; completion batching; completion during
generation; acknowledgement; same-agent session isolation; owner conflicts
before hooks; paused/failed runs; idle/continuation limits; missing history;
close; and cancellation while waiting or generating. Existing background,
runnable, tracing, guardrail, and voice tests remain regression gates.

Required before merge, matching the "Implementation status" deltas:

- A direct turn while the driver waits: driver ends `preempted`, exactly one
  run starts, its memory carries the pending notices before the human prompt,
  nothing is drained twice.
- A direct turn while a follow-up is in flight: rejected before hooks (assert
  with a per-prompt hook counter, not a global one) and the follow-up finishes.
- `stop()` while waiting ends immediately; `stop()` during a follow-up lets it
  reach `OUTPUT`; `continuation_for_run` finds the driver and returns `None`
  after release.
- Follow-up `input.prompt` is the notice and `is_runtime()`; `metadata.trigger`
  lists its task ids; a follow-up failed before memory assembly leaves the
  notice recoverable by the next turn's rebuild; an acknowledgement after the
  driver has taken a notice does not remove it from that run.
- `admit` is awaited before each follow-up and cancelled by preemption/`stop()`.
- An agent without background tools allocates no store and registers no run id.

Before adding external notifications/voice, require deterministic tests for:

- Two producers/retries: stable idempotency, replacement only while pending,
  bounded inbox, and explicit overflow/expiry results.
- Cancellation at every take/attach boundary and close while waiting: no
  unresolvable receipt and no swallowed cancellation.
- Approval/input suspension: no inserted event between tool use and result and
  no fabricated resume value.
- Voice playback/partial speech/VAD/greetings: never interrupt under `when_idle`;
  normal barge-in and heard-prefix memory under authorized interruption.
- Disconnect versus hang-up versus stopping the driver versus cancelling jobs:
  separate outcomes and no assumption that the runtime survives deployment shutdown.
- Unauthorized producer/session, forged runtime metadata, and oversized payload:
  rejected before privileged admission; hidden transcript status is not trust.
- Deployment owner routing and trace-save failure: report uncertainty, never
  claim durable acceptance based solely on an in-memory append.

Merge/readiness decisions: close the Phase 1 deltas above and agree the driver's
API and output contract; then open the voice adapter PR on the `admit` hook,
carrying forward what "Related work" lists from #214; then implement the shared
inbox as a separately reviewable change. Do not expose client notification
frames before the trust boundary and receipt protocol are approved.
