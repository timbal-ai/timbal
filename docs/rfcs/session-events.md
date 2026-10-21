# RFC: event-triggered turns across chat and voice

Status: **draft; design plus a deliberately limited first implementation**.
Reviewed against main at `9202f863` on 2026-10-21. Names for the later session API
are proposals, not shipped interfaces.

## Decision in brief

A task finishing is an input to a conversation, not a caller utterance and not
an instruction to speak immediately. Separate three responsibilities:

1. Record a correlated event/result for the owning conversation.
2. Decide whether and when another agent turn may start.
3. Deliver its output through chat or voice, with channel-specific interruption
   and playback rules.

The first implementation is `Agent.stream_until_idle(...)`: an opt-in local
driver that reuses the existing background-completion inbox. It does not create
a second task store, make `VoiceSession` the owner of conversation state, expose
untrusted event frames, or silently change the lifetime of existing agent calls.

This PR is **not** a complete replacement for #214. External notification
ingress, arbitrary session wakeups after a driver has ended, and voice adapters
remain explicitly unimplemented. Do not merge it as a claim that those product
requirements are finished.

## Related work and migration

- [#214: voice notifications](https://github.com/timbal-ai/timbal/pull/214):
  retain the use cases, channel admission checks, context isolation work, and
  transport tests. Move the general event/turn contract out of `VoiceSession`.
  Do not carry forward prior-trace edits, boolean delivery acknowledgements, or
  client-provided text promoted directly to trusted runtime context.
- [#218: end call](https://github.com/timbal-ai/timbal/pull/218): currently stacked
  on #214. Keep hang-up-after-playback as a voice policy, not a generic session
  close operation. Rebase or extract it only after the voice adapter contract is
  agreed; caller activity must include partial speech/VAD, not only committed text.
- [#216: tracing and interrupted history](https://github.com/timbal-ai/timbal/pull/216):
  merged, included in this PR's base. Reuse read-after-save and interrupted-turn
  recovery. This is not an asynchronous/durable notification queue.
- [#179: external background polling](https://github.com/timbal-ai/timbal/pull/179):
  complementary read/cancel access, not a turn scheduler. Polling for a UI must
  not acknowledge delivery to the model.
- [#174: OpenAI live pipeline](https://github.com/timbal-ai/timbal/pull/174):
  a separate live-model adapter will need equivalent semantics without pretending
  its provider-owned loop is the ordinary text-agent loop.
- [#215: RTC drain](https://github.com/timbal-ai/timbal/pull/215) and
  [#217: interruption persistence](https://github.com/timbal-ai/timbal/pull/217):
  preserve their transport/playback and history fixes; neither is a session inbox.

No existing PR is automatically closed or declared wholly superseded.

## What exists already

`BackgroundTaskStore` is process-local and follows `RunContext.parent_id`. It
owns task state, append-only child event logs, and one-shot completion notices.
`Agent._inject_background_completions` attaches those notices to the receiving
turn. Model polling can acknowledge a result so the next turn does not repeat it.

`JobStore` already decouples an HTTP run from its SSE reader. Its replay log is
per run, bounded, and process-local. Neither it nor a persisted trace schedules
another conversational turn. The existing server does not offer cross-worker
session ownership or authorization on its own.

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

## Phase 1: implementation in this PR

```python
from contextlib import aclosing

stream = agent.stream_until_idle(
    prompt="Build the report in the background and tell me when it finishes.",
    max_idle_secs=300,
    max_continuations=10,
)
async with aclosing(stream):
    async for event in stream:
        await publish(event)

print(stream.stop_reason, stream.last_run_id)
```

`publish` is application code. This does not add a server route. Callers consume
all runs, including multiple root `OUTPUT` events; each run retains a fresh
`run_id` and the previous turn as `parent_run_id`. There is intentionally no
single `.collect()` result that hides the extra turns. Follow-up root outputs
carry `metadata.trigger.kind="background_completion"` and a continuation index.

The driver:

1. Starts the ordinary foreground agent run and reserves its background store.
2. Streams the run to completion, including tracing finalization.
3. Stops on unsuccessful output, approval, or interaction suspension.
4. Waits on a completion signal without polling or draining the inbox. Already
   finished tasks are reconciled before declaring quiescence, including the
   interval before their done callbacks have executed.
5. Verifies readable parent history and no unresolved tool calls, then starts
   another ordinary run. An empty runtime-tagged input satisfies the existing
   parameter contract; it is removed from model memory by normal empty-message
   sanitation. The actual input is the existing completion notice attached by
   the receiving run. No fabricated human utterance or previous-trace edit occurs.
6. Repeats until quiescent or a stop condition is reached.

Only one continuation driver may own a store. A competing root runnable is
rejected before parameter resolution, hooks, or model execution once the store
is reserved. The initial parent must be completed and readable; this API is not
an adoption mechanism for an already running foreground turn. Different stores
may use the same Agent concurrently.

Stop conditions are observable through `stop_reason`: `idle`, `idle_timeout`,
`continuation_limit`, `paused`, `run_stopped`, `history_unavailable`, `closed`,
`cancelled`, or `error`. `continuations` counts admitted follow-up runs.

- `max_idle_secs` is finite, positive, and applies between runs, including the
  history-read preflight. It is not a model/tool execution timeout.
- `max_continuations` is a non-negative integer; zero allows the initial turn only.
- No background work means immediate completion; this is not a permanent listener
  for hypothetical future tasks.
- Notices not yet attached to a turn remain pending on timeout, limit, close, or
  missing history. A model that already polled a result does not get a duplicate.
- Closing/cancelling the driver cancels its foreground execution and further
  automatic responses, **not detached children**. Use existing explicit task
  cancellation to cancel those. Python event-loop/process shutdown is different:
  this PR does not make a local asyncio task survive shutdown.
- The stream is caller-driven. A client disconnect only leaves it running if the
  hosting application owns a separate consumer task. Existing `/run`, `/stream`,
  voice transports, and `JobStore` are not automatically switched to this API.
- To send another human turn, cancel the task consuming the driver, await its
  teardown, then invoke with `parent_id=stream.last_run_id`. Do not run two
  consumers or call `aclose()` concurrently with a pending `__anext__()`.

The first slice does not promise exactly-once consumption or durable delivery.
After a notice is attached, cancellation, provider failure, or a failed trace
save does not prove whether the model or user received it. Preserve that
distinction rather than automatically replaying a potentially side-effecting turn.

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
- `interrupt`: explicit privileged policy, not Phase 1's default or an expiry
  fallback. Voice must preserve its heard-prefix interruption behavior.

An event waiting too long does not silently become an interruption. Expiry
reports a reason and preserves whatever source/task result remains available.
Events cannot bypass approvals or interactively suspended tool calls. Results
remain outside the pending tool-use/tool-result sequence until it is safe to
attach them. Human input takes priority over pending autonomous responses.

The inbox lifecycle is `pending -> reserved -> attached`, or a terminal
`withdrawn / superseded / expired / closed` before attachment. Reservation and
turn admission are atomic for one owner. Cancellation before attachment releases
the reservation. Attachment records the receiving run ID and belongs to that
run's memory, never an older trace. After attachment, record the run's outcome;
do not put the same event back into pending automatically.

Track separately: accepted into local/durable storage, attached to model context,
run completed/failed/interrupted, and output delivered. Text generated, audio
sent, and audio played are distinct facts. No `True/False` receipt can represent
all of these. At-least-once transport and deduplicated admission still do not
provide exactly-once tool side effects or exactly-once audible announcements.

### Voice adapter

The adapter contributes an admission predicate and wakes the scheduler when it
changes. `when_idle` requires no foreground turn, outstanding playback,
greeting/filler, held fragment, or active/recent caller speech. Re-check admission
at the actual start boundary. Quiet windows need measured defaults; they must
not be baked into a generic agent scheduler.

Do not wire Phase 1 directly into `_run_turn`: generation can end before audio
plays. Both user and event turns must enter the same voice owner. Barge-in stops
the foreground response, persists heard-prefix history, and leaves detached jobs
alone unless explicit application policy cancels them. Hang-up stops voice
delivery, not necessarily the underlying conversation's work. A separate
application policy decides whether the completion is later presented in chat.

## Acceptance gates and rollout

Phase 1's executable tests are in
[`test_continuation.py`](../../python/tests/core/test_continuation.py). They cover
multi-run chaining; in-memory and JSONL history; no historical edits; no extra
human message; completion batching; completion during generation; acknowledgement;
same-agent session isolation; owner conflicts before hooks; paused/failed runs;
idle/continuation limits; missing history; close; and cancellation while waiting
or generating. Existing background, runnable, tracing, guardrail, and voice tests
remain regression gates.

Before adding external notifications/voice, require deterministic tests for:

- Concurrent human input and task completion: one owner, no lost event, no
  double run, and documented priority.
- Two producers/retries: stable idempotency, replacement only while pending,
  bounded inbox, and explicit overflow/expiry results.
- Cancellation at every reservation/attachment boundary and close while waiting:
  no unresolvable receipt and no swallowed cancellation.
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

Merge/readiness decisions: agree the public Phase 1 method/lifetime and output
contract; review cancellation and ownership; then implement the shared inbox
and voice adapter as separately reviewable changes. Do not expose #214's client
notification frames before the trust boundary and receipt protocol are approved.
