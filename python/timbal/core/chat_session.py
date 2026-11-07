"""A text conversation that stays open: messages in at any time, one turn at a time, events out."""

import asyncio
import contextlib
import math
import os
import time
from collections.abc import AsyncIterator, Callable, Collection
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import structlog
from uuid_extensions import uuid7

from ..state import RunContext, set_call_id, set_parent_call_id, set_run_context
from ..state.background import (
    DEFAULT_BG_LOG_MAX_BYTES,
    DEFAULT_BG_LOG_MAX_EVENTS,
    BackgroundEventLog,
    BackgroundTaskStore,
    SessionEvent,
    adopt_background_store,
    forget_idle_background_store,
    store_for_run,
)
from ..state.chat_store import PROCESS_OWNER, ChatSessionStore, SessionLeased, SessionSnapshot, SnapshotConflict
from ..types.approval import Cancel
from ..types.content import TextContent
from ..types.events import OutputEvent
from ..types.message import RUNTIME_SOURCE, Message
from .session_scheduler import Followup, SessionScheduler, is_parked, user_batch

if TYPE_CHECKING:
    from .agent import Agent

__all__ = ["SEND_MODES", "SILENT_REPLY", "ChatSession", "OnDeadTurn", "SendMode", "ToolPolicy"]

logger = structlog.get_logger("timbal.core.chat_session")

SendMode = Literal["steer", "now", "followup", "collect", "interrupt"]
SEND_MODES: tuple[str, ...] = ("steer", "now", "followup", "collect", "interrupt")

ToolPolicy = Literal["read_only"] | Collection[str] | Callable[[Any], bool]
"""Which tools a turn offers its model: ``"read_only"`` (tools declared ``read_only``,
MCP tools the server marks ``readOnlyHint``, the agent's own readers), a collection of
tool names, or a callable deciding per tool."""

SILENT_REPLY = "NO_REPLY"
"""What a turn the runtime started answers to stay silent (``turn_finished.silent``)."""

WAKE_KIND = "wake"
RECOVERY_KIND = "recovery"

OnDeadTurn = Literal["continue", "notify", "drop"]
"""What a restored session does with a turn that died with its process (see :meth:`ChatSession.restore`)."""

_SILENCE_NOTE = f"If nothing here needs the user's attention, reply with exactly {SILENT_REPLY} and nothing else."
_WAKE_HEADER = "Scheduled wake-up (the user did not write this):"
# A turn that fails before it starts: retried after 0.5 s, then 1 s; past that, only when new work arrives.
_MAX_TURN_FAILURES = 3
_TURN_RETRY_DELAY_S = 0.5
# Between attempts to learn whether the run a session continues is parked (its trace store is down).
_RESTORE_RETRY_MAX_S = 30.0
_RECOVERY_HEADER = "Your previous turn was cut off by a server restart (the user did not write this)."
_RECOVERY_ADVICE = (
    "Continue where you left off. Do not repeat completed actions. Before redoing one that is not confirmed, "
    "check whether it already happened, or ask the user."
)
# A restored log continues this far past the saved cursor: further than any client of the old process read.
_LOG_RESTART_GAP = 1_000_000_000
_PERSISTED_KINDS = frozenset(
    {"message", "message_settled", "wake", "wake_superseded", "turn_started", "turn_finished", "interrupt", "closed"}
)
_READ_ONLY_NOTE = (
    "This turn can only read: tools that send, write, delete or buy are not available. If the task needs "
    "one, do not retry or look for a way around it; say what you would do and that it needs the user."
)
_SESSION_KWARGS = frozenset({"prompt", "messages", "parent_id", "resume"})


@dataclass(eq=False)
class _Wake:
    text: str
    key: str | None
    tool_policy: Any
    silent_ok: bool
    id: str = field(default_factory=lambda: uuid7(as_type="hex"))  # type: ignore[arg-type]
    kind: str = WAKE_KIND
    trigger: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "key": self.key,
            "tool_policy": _store_policy(self.tool_policy),
            "silent_ok": self.silent_ok,
            "id": self.id,
            "kind": self.kind,
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_Wake":
        return cls(
            text=data["text"],
            key=data.get("key"),
            tool_policy=_load_policy(data.get("tool_policy")),
            silent_ok=data.get("silent_ok", True),
            id=data["id"],
            kind=data.get("kind", WAKE_KIND),
            trigger=data.get("trigger"),
        )


@dataclass(frozen=True)
class _TurnOptions:
    """What a turn runs under; a resume of it runs under the same."""

    tool_policy: Any = None
    silence: bool = False
    # Takes the session's lane="next" messages between steps. A wake does not: a user message would be
    # answered under the wake's tool policy and silence, so it waits for a turn of its own.
    steer: bool = True


def _check_tool_policy(policy: Any) -> Any:
    if policy is None or policy == "read_only" or callable(policy):
        return policy
    if isinstance(policy, str) or not isinstance(policy, Collection):
        raise ValueError("tool_policy must be 'read_only', a collection of tool names, a callable, or None.")
    names = frozenset(policy)
    if not all(isinstance(name, str) for name in names):
        raise ValueError("tool_policy names must be strings.")
    return names


def _allowed_tool_paths(trace: Any) -> frozenset[str]:
    """What a callable tool policy let through in ``trace``, as last decided: each agent notes on its span what
    its latest tool resolution allowed (``tool_policy_allowed``), and an agent run more than once in the trace
    counts by its latest span, so a later decision (a permission taken back) wins over an earlier one."""
    latest: dict[str, Any] = {}
    for span in trace.values():
        if "tool_policy_allowed" not in (span.metadata or {}):
            continue
        seen = latest.get(span.path)
        if seen is None or (span.t0 or 0) >= (seen.t0 or 0):
            latest[span.path] = span
    return frozenset(path for span in latest.values() for path in span.metadata["tool_policy_allowed"])


@dataclass(frozen=True)
class _ToolPaths:
    """A tool policy restored from a trace: the tools with these paths (agent-scoped, unlike names)."""

    paths: frozenset[str]

    def __call__(self, tool: Any) -> bool:
        return getattr(tool, "_path", None) in self.paths


def _describe_policy(policy: Any) -> Any:
    if policy is None or policy == "read_only":
        return policy
    if callable(policy):
        return "custom"
    return sorted(policy)


def _with_note(prompt: Message, note: str) -> Message:
    return Message(
        role=prompt.role,
        content=[*prompt.content, TextContent(text=note)],
        metadata=dict(prompt.metadata) if prompt.metadata else None,
    )


def _store_policy(policy: Any) -> Any:
    """A tool policy as a snapshot holds it. A callable cannot be stored: it comes back ``"read_only"``."""
    if policy is None or policy == "read_only":
        return policy
    if callable(policy):
        logger.warning("chat_session_policy_not_durable", detail="a callable tool_policy restores as 'read_only'")
        return "read_only"
    return sorted(policy)


def _load_policy(stored: Any) -> Any:
    return frozenset(stored) if isinstance(stored, list) else stored


def _dump_prompt(prompt: Any) -> Any:
    from ..utils.serialization import _dump_sync, _NeedsAsync

    if prompt is None or isinstance(prompt, str):
        return prompt
    try:
        return _dump_sync(prompt)
    except _NeedsAsync:  # files: a resume only needs the turn to have a prompt, the agent drops it
        return {"role": prompt.role, "content": [{"type": "text", "text": prompt.collect_text()}]}


def _load_prompt(stored: Any) -> Any:
    return Message.validate(stored) if isinstance(stored, dict) else stored


def _dump_resume(values: dict[str, Any] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    return {key: {"$cancel": value.reason} if isinstance(value, Cancel) else value for key, value in values.items()}


def _load_resume(stored: dict[str, Any] | None) -> dict[str, Any] | None:
    if stored is None:
        return None
    return {
        key: Cancel(reason=value["$cancel"]) if isinstance(value, dict) and "$cancel" in value else value
        for key, value in stored.items()
    }


def _status_dict(status: Any) -> dict[str, Any] | None:
    if status is None:
        return None
    if isinstance(status, dict):
        return {k: status.get(k) for k in ("code", "reason", "message")}
    return {"code": status.code, "reason": status.reason, "message": status.message}


def _brief(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _blocks(message: Any, kind: str) -> list[dict[str, Any]]:
    content = message.get("content") if isinstance(message, dict) else None
    return [block for block in content or [] if isinstance(block, dict) and block.get("type") == kind]


def _outcome(span: Any) -> str:
    return _brief(span.error.get("message") if isinstance(span.error, dict) else span.output)


def _tool_calls(context: RunContext) -> tuple[list[str], list[str]]:
    """A dead turn's tool calls: completed (an outcome stored) and unconfirmed (asked for or started, no outcome).

    The model's requests come from its stored calls' outputs as well as the agent's memory: a save can land
    between a model call ending and its tool spans (or the memory) being stored.
    """
    root = context.root_span()
    llm_path = f"{root.path}.llm"
    children = [span for span in context._trace.as_records() if span.parent_call_id == root.call_id]
    memory = root.memory if isinstance(root.memory, list) else []
    requests: dict[str, tuple[str, Any]] = {}
    for message in [span.output for span in children if span.path == llm_path] + memory:
        for block in _blocks(message, "tool_use"):
            if block.get("id"):
                requests.setdefault(block["id"], (block.get("name") or "?", block.get("input")))
    results = {block.get("id"): block for message in memory for block in _blocks(message, "tool_result")}
    spans: dict[str, list[Any]] = {}
    for span in children:
        if span.path != llm_path:
            spans.setdefault(span.path.rsplit(".", 1)[-1], []).append(span)
    completed, unconfirmed = [], []
    for request_id, (name, args) in requests.items():
        call = f"{name}({_brief(args, 80)})"
        span = spans[name].pop(0) if spans.get(name) else None
        if request_id in results:
            text = " ".join(b.get("text", "") for b in _blocks(results[request_id], "text")) or "done"
            completed.append(f"{call} → {_brief(text)}")
        elif span is not None and span.t1 is not None:
            completed.append(f"{call} → {_outcome(span)}")
        else:
            unconfirmed.append(call)
    for name, rest in spans.items():  # started with no request stored
        for span in rest:
            call = f"{name}({_brief(span.input, 80)})"
            (completed if span.t1 is not None else unconfirmed).append(
                f"{call} → {_outcome(span)}" if span.t1 is not None else call
            )
    return completed, unconfirmed


def _recovery_notice(completed: list[str], unconfirmed: list[str]) -> str:
    lines = [_RECOVERY_HEADER]
    if completed:
        lines.append("Completed before the restart: " + "; ".join(completed) + ".")
    if unconfirmed:
        lines.append("Started but not confirmed (each may or may not have run): " + "; ".join(unconfirmed) + ".")
    if not completed and not unconfirmed:
        lines.append("No tool call had run yet.")
    lines.append(_RECOVERY_ADVICE)
    return "\n".join(lines)


class ChatSession:
    """One agent's conversation with a user, open for messages while its turns run.

    :meth:`send` takes a user message at any time. With no turn running it
    starts one. While a turn runs, ``mode`` decides what happens to it:

    - ``"steer"``: the running turn takes it after its current tool batch, as a
      user message, and does not end before answering it (``Agent.steering``,
      on for these runs unless the agent turns it off). Messages a turn ends
      without taking start the next turn together.
    - ``"now"``: like steer, but without waiting: the answer being streamed is
      cut (its text kept, marked interrupted) and running tools that allow it
      (``Tool.interruptible``, default: read-only ones) are cancelled with an
      interrupted result; the others finish first. Messages within a quarter
      second make one cut. Same turn — unlike ``"interrupt"``.
    - ``"followup"``: a turn of its own once the running one ends.
    - ``"collect"``: one turn for every collected message, once none has
      arrived for ``collect_window`` seconds.
    - ``"interrupt"``: cancels the running turn (it ends ``cancelled`` /
      ``interrupted``, keeping what it produced), then starts one with this
      message, ahead of anything else queued.

    ``busy_mode`` is the mode a message gets when none is given. Finished
    background tasks and posted session events start turns too (when their
    announce policy asks for one), as with :class:`AgentContinuation`, and
    :meth:`wake` starts one for the application (a routine, a heartbeat) with
    a narrowed tool set. Those turns may stay silent: told so in their prompt,
    one that answers exactly :data:`SILENT_REPLY` ends ``silent``
    (``allow_silence``). A turn that ends waiting on an approval or
    ``ask_user`` parks the session: new messages wait until :meth:`resume`
    answers it, or :meth:`interrupt` (a message sent with ``"interrupt"``)
    cancels what it waits for.

    :attr:`log` holds every event of every turn, in order, plus session records
    (``{"type": "SESSION", "kind": ...}``): ``message`` (a message was sent),
    ``message_settled`` (a turn took it, or it was superseded), ``wake`` /
    ``wake_superseded``, ``turn_started``, ``turn_finished`` (``status``,
    ``parked``, ``silent``, ``tool_search`` when it loaded deferred tools, ``error`` ``{type, message}``
    when it failed), ``interrupt``, ``closed``. Read it with :meth:`events`.

    Process-local, like the session store it owns. Call its methods on the loop
    it runs on.
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        parent_id: str | None = None,
        busy_mode: SendMode = "steer",
        collect_window: float = 1.0,
        allow_silence: bool = True,
        max_log_events: int | None = DEFAULT_BG_LOG_MAX_EVENTS,
        max_log_bytes: int | None = DEFAULT_BG_LOG_MAX_BYTES,
        durability: Literal["sync", "buffered", "end"] | None = None,
        session_id: str | None = None,
        session_store: ChatSessionStore | None = None,
        on_dead_turn: OnDeadTurn = "continue",
        **kwargs: Any,
    ) -> None:
        if durability not in (None, "sync", "buffered", "end"):
            raise ValueError(f"durability must be 'sync', 'buffered' or 'end', got {durability!r}")
        if on_dead_turn not in ("continue", "notify", "drop"):
            raise ValueError(f"on_dead_turn must be 'continue', 'notify' or 'drop', got {on_dead_turn!r}")
        if "." in agent._path or agent.background_mode != "never":
            raise RuntimeError("A chat session needs a foreground, top-level Agent.")
        if agent.tracing_provider is None:
            raise ValueError("A chat session needs a tracing provider: every turn reads the history of the last.")
        if busy_mode not in SEND_MODES:
            raise ValueError(f"busy_mode must be one of {list(SEND_MODES)}, got {busy_mode!r}")
        if not math.isfinite(collect_window) or collect_window < 0:
            raise ValueError("collect_window must be a finite, non-negative number of seconds.")
        passed = _SESSION_KWARGS.intersection(kwargs)
        if passed:
            raise TypeError(
                f"{sorted(passed)} are not turn arguments: send messages with send(), "
                "answer a parked turn with resume(), and continue a conversation with parent_id=."
            )
        self.id: str = session_id or uuid7(as_type="hex")  # type: ignore[assignment]
        self.busy_mode: SendMode = busy_mode
        self.on_dead_turn: OnDeadTurn = on_dead_turn
        self.collect_window = collect_window
        self._turn_failures = 0
        self._attempted: set[Any] = set()
        self.allow_silence = allow_silence
        self.log = BackgroundEventLog(max_events=max_log_events, max_bytes=max_log_bytes)
        self.closed = False
        self._parked = False
        self.turns = 0
        self.last_status: dict[str, Any] | None = None
        self.last_activity = time.monotonic()
        self._agent = agent
        self._kwargs = kwargs
        self._durability = durability
        self._providers: dict[tuple[type, str], type] = {}
        store = store_for_run(parent_id) if parent_id is not None else None
        self._store = store if store is not None else BackgroundTaskStore()
        self._scheduler = SessionScheduler(
            channel=self, on_busy="enqueue", can_interrupt=True, on_work=self._work_arrived, lends=True
        )
        self._scheduler.join(self._store)
        self._scheduler.head = parent_id
        self._work = asyncio.Event()
        self._resume: dict[str, Any] | None = None
        self._wakes: list[_Wake] = []
        self._last_prompt: Any = None
        self._last_options = _TurnOptions()
        self._worker: asyncio.Task | None = None
        self._turn: asyncio.Task | None = None
        self._turn_context: RunContext | None = None
        self._last_context: RunContext | None = None
        self._restore_from = parent_id
        self._restoring: asyncio.Task | None = None
        self._interrupt_on_restore = False
        # Durable sessions (session_store=): a snapshot written ahead of every change.
        self._session_store = session_store
        self._version: int | None = None
        self._dirty = False
        self._saver: asyncio.Task | None = None
        self._leased = False
        self._lease_task: asyncio.Task | None = None
        self._lost = False
        self._keep = False
        self._in_flight: dict[str, Any] | None = None

    # -- Public API -------------------------------------------------------------

    @property
    def head(self) -> str | None:
        """The session's latest run: where the next turn continues from."""
        return self._scheduler.head

    @property
    def busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    @property
    def run_id(self) -> str | None:
        """The running turn's run id, or None."""
        return self._turn_context.id if self.busy and self._turn_context is not None else None

    @property
    def store(self) -> BackgroundTaskStore:
        return self._store

    def send(self, text: str, *, mode: SendMode | None = None) -> SessionEvent:
        """Post a user message (see the class docstring for ``mode``); returns its receipt at once."""
        if self.closed:
            raise RuntimeError("The chat session is closed.")
        mode = mode or self.busy_mode
        if mode not in SEND_MODES:
            raise ValueError(f"mode must be one of {list(SEND_MODES)}, got {mode!r}")
        busy = self.busy
        if mode == "interrupt":
            event = self._store.post_event(text, source="user", mode="followup", announce="interrupt")
        else:
            # 'now' joins the running turn like steer, after cutting what is in flight.
            event = self._store.post_event(text, source="user", mode="steer" if mode == "now" else mode)
        self.last_activity = time.monotonic()
        self._record("message", event_id=event.id, text=event.text, mode=mode, busy=busy)
        self._watch(event)
        self._ensure_worker()
        if mode == "interrupt":
            self.interrupt()
        elif mode == "now" and busy and self._turn_context is not None and self._turn_context._cut is not None:
            self._turn_context._cut.set()
        return event

    def wake(
        self,
        text: str,
        *,
        key: str | None = None,
        tool_policy: ToolPolicy | None = "read_only",
        silent_ok: bool = True,
    ) -> str:
        """Start a turn for the application (a routine, a heartbeat): returns the wake's id.

        It runs once nothing the user sent or the session owes is waiting, as
        a turn of its own (never folded into a user's), with only the tools
        ``tool_policy`` allows (default ``"read_only"``: proactive work that
        cannot send, write or buy; ``None`` for all tools). With ``silent_ok``
        (and ``allow_silence``) it may answer :data:`SILENT_REPLY` to say
        nothing. A pending wake with the same ``key`` is replaced. The text is
        trusted application text: input guardrails do not screen it.
        """
        if self.closed:
            raise RuntimeError("The chat session is closed.")
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            raise ValueError("A wake needs non-empty text.")
        policy = _check_tool_policy(tool_policy)
        key = (key or "").strip() or None
        if key is not None:
            for old in [wake for wake in self._wakes if wake.key == key]:
                self._wakes.remove(old)
                self._record("wake_superseded", wake_id=old.id, key=key)
        wake = _Wake(text=text, key=key, tool_policy=policy, silent_ok=silent_ok)
        self._wakes.append(wake)
        self.last_activity = time.monotonic()
        self._record("wake", wake_id=wake.id, text=text, key=key, tool_policy=_describe_policy(policy))
        self._work.set()
        self._ensure_worker()
        return wake.id

    def message_task(self, task_id: str, text: str) -> SessionEvent:
        """Send one of the session's running background agents a message from the user.

        It reads it after its current step (:meth:`BackgroundTaskStore.message_task`
        raises for an unknown, finished or non-agent task). Background tasks are
        in-process: a durable session does not keep the message across a restart.
        """
        if self.closed:
            raise RuntimeError("The chat session is closed.")
        event = self._store.message_task(task_id, text, source="user")
        self.last_activity = time.monotonic()
        self._record("task_message", task_id=task_id, event_id=event.id, text=event.text)
        return event

    def interrupt(self) -> bool:
        """Cut what the session is doing; queued messages then start the next turn.

        Cancels the running turn, or, when the session is parked, every pending
        approval and question not answered yet (``Cancel``: the gated tools never
        run, the run ends ``cancelled``; answers already given stand). False when
        there is neither. Before :meth:`restore` has learned whether the session is
        parked, it is applied once it has (True).
        """
        if self._restore_pending():
            # Whether the session is parked is not known yet: restore() applies the interrupt.
            self._interrupt_on_restore = True
            self._ensure_worker()
            return True
        return self._interrupt_now()

    def _restore_pending(self) -> bool:
        return self._restore_from is not None or (self._restoring is not None and not self._restoring.done())

    def _interrupt_now(self) -> bool:
        turn = self._turn
        if turn is not None and not turn.done():
            self._record("interrupt", run_id=self.run_id)
            turn.cancel()
            return True
        if self.parked and self._last_context is not None:
            reason = "The user moved on before answering."
            answered = self._resume or {}
            pending = [a["approval_id"] for a in self._last_context.pending_approvals()]
            pending += [i["interaction_id"] for i in self._last_context.pending_interactions()]
            pending = [pending_id for pending_id in pending if pending_id not in answered]
            if not pending:
                return False
            self._record("interrupt", run_id=self._last_context.id, abandoned=pending)
            self.resume({pending_id: Cancel(reason=reason) for pending_id in pending})
            return True
        return False

    @property
    def parked(self) -> bool:
        """The last turn ended waiting on an approval or ``ask_user``: :meth:`resume` answers it.

        False again once another client's run of the session answers it or goes past it (the head moved).
        """
        context = self._last_context
        return self._parked and context is not None and self._scheduler.head == context.id

    def resume(self, values: dict[str, Any]) -> None:
        """Answer the parked turn: ``{approval_id: ...}`` / ``{interaction_id: value}``, as for ``resume=``."""
        if self.closed:
            raise RuntimeError("The chat session is closed.")
        turn = self._turn
        if turn is not None and not turn.done():
            # Its answers, if it ends waiting for some, are taken once it does.
            raise RuntimeError("A turn of this session is running; answer once it waits for an answer.")
        if not self.parked and not self._restore_pending():
            # Before restore() has finished, the answers wait for it: dropped if the run is not parked after all.
            raise RuntimeError("No turn of this session is waiting for an answer.")
        queued = self._resume or {}
        conflicts = sorted(key for key, value in values.items() if key in queued and queued[key] != value)
        if conflicts:
            raise ValueError(f"Already answered differently: {', '.join(conflicts)}.")
        # Answers sent before the worker takes them go together (two approvals of one parked turn).
        self._resume = {**queued, **values}
        self.last_activity = time.monotonic()
        self._persist()
        self._work.set()
        self._ensure_worker()

    def events(self, after: int = 0, *, numbered: bool = False) -> AsyncIterator[Any]:
        """Replay the log from cursor ``after``, then follow it until :meth:`close`.

        ``numbered``: ``(cursor, item)`` pairs instead, see :meth:`BackgroundEventLog.subscribe`.
        """
        return self.log.subscribe(after, numbered=numbered)

    def pending_messages(self) -> list[dict[str, Any]]:
        """Receipts of the user messages no turn has taken yet, oldest first."""
        return [
            {**event.receipt(), "text": event.text} for event in self._store.pending_events() if event.source == "user"
        ]

    def state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "session_id": self.id,
            "head": self.head,
            "busy": self.busy,
            "run_id": self.run_id,
            "parked": self.parked,
            "turns": self.turns,
            "busy_mode": self.busy_mode,
            "pending": self.pending_messages(),
            "wakes": [{"wake_id": wake.id, "text": wake.text, "key": wake.key} for wake in self._wakes],
            "last_status": self.last_status,
            "log_cursor": self.log.cursor_end,
            "durable": self._session_store is not None and not self._lost,
        }
        if self.parked and self._last_context is not None:
            state["pending_approvals"] = self._last_context.pending_approvals()
            state["pending_interactions"] = self._last_context.pending_interactions()
        return state

    async def close(self, *, keep: bool = False) -> None:
        """Stop the session: the running turn is cancelled, the store let go. Detached children keep running.

        A durable session's snapshot is deleted, unless ``keep``: this process is going away (shutdown, an idle
        session let go), and :meth:`restore` brings the session back. A turn this cuts is then recovered there
        like one that died with its process.
        """
        if self.closed:
            return
        cut = self._in_flight if keep and self.busy else None
        self.closed = True
        self._keep = keep
        worker, self._worker = self._worker, None
        if worker is not None and not worker.done():
            worker.cancel()
            with contextlib.suppress(BaseException):
                await worker
        if cut is not None:
            self._in_flight = {**cut, "cut": "shutdown"}
        if self._last_context is not None:
            self._scheduler.unmark(self._last_context)
        self._scheduler.release(self._store)
        forget_idle_background_store(self._store)
        self._record("closed", kept=keep)
        self.log.close()
        await self._settle_store()

    @classmethod
    async def restore(
        cls, agent: "Agent", session_store: ChatSessionStore, session_id: str, **kwargs: Any
    ) -> "ChatSession":
        """Bring back a durable session another process (or an earlier life of this one) left in ``session_store``.

        Queued messages and wakes come back with their ids; a parked turn can be answered with :meth:`resume`;
        clients reconnecting to :meth:`events` with an old cursor get a gap and continue from the new head
        (the log itself is not stored: the turns are, in the traces). A turn that was running when the old
        process died is settled by ``on_dead_turn``:

        - it had finished (the crash came between its trace and the snapshot): adopted as is;
        - it had saved nothing: its messages (or its wake) are queued again;
        - otherwise (cut mid-run) ``"continue"`` (default) runs a read-only recovery turn chained on it, told
          which tool calls completed and which were started but not confirmed, and to check before redoing
          those: nothing is re-run without the model deciding to. ``"notify"`` queues that notice for the
          next turn instead, ``"drop"`` forgets it.

        Raises LookupError (no such session), ValueError (another agent's) or
        :class:`~timbal.state.chat_store.SessionLeased` (another live process serves it).
        """
        snapshot = await session_store.load(session_id)
        if snapshot is None or snapshot.closed:
            raise LookupError(f"No stored chat session {session_id!r}.")
        if snapshot.agent != agent.name:
            raise ValueError(f"Chat session {session_id!r} belongs to agent {snapshot.agent!r}, not {agent.name!r}.")
        await session_store.acquire_lease(session_id, PROCESS_OWNER)
        config = {**snapshot.config, **kwargs}
        session = cls(agent, parent_id=snapshot.head, session_id=session_id, session_store=session_store, **config)
        session._version, session._leased = snapshot.version, True
        session.turns, session.last_status, session.parked = snapshot.turns, snapshot.last_status, snapshot.parked
        session._last_prompt = _load_prompt(snapshot.last_prompt)
        session._last_options = _TurnOptions(
            tool_policy=_load_policy(snapshot.last_options.get("tool_policy")),
            silence=bool(snapshot.last_options.get("silence")),
        )
        session.log.restart_at(snapshot.log_cursor + _LOG_RESTART_GAP)
        session._restore_events(snapshot.pending)
        session._wakes = [_Wake.from_dict(data) for data in snapshot.wakes]
        session._resume = _load_resume(snapshot.resume)
        recovered = None
        if snapshot.in_flight is not None:
            recovered = await session._recover(snapshot.in_flight)
        if session.parked and session._last_context is None:
            session._last_context = await session._load_run(session.head)
        session._record("restored", head=session.head, recovered=recovered, pending=len(snapshot.pending))
        session._persist()
        session._lease_task = asyncio.get_running_loop().create_task(session._renew_lease())
        if session._wakes or session._resume is not None or session._store.pending_events():
            session._work.set()
            session._ensure_worker()
        return session

    async def __aenter__(self) -> "ChatSession":
        await self.restore()
        return self

    async def restore(self) -> None:
        """Pick up the state of the run the session continues (``parent_id``): one parked on an approval or
        ``ask_user`` leaves the session parked, so messages wait and :meth:`resume` answers it (a turn chained
        onto it would resume it and drop its prompt). Once; ``async with`` and the first turn call it."""
        if self._restoring is None or (self._restoring.done() and self._restore_from is not None):
            self._restoring = asyncio.get_running_loop().create_task(self._restore_parked())
        await asyncio.shield(self._restoring)

    async def _restore_parked(self) -> None:
        try:
            await self._load_parked()
        finally:
            if self._interrupt_on_restore and self._restore_from is None:
                # interrupt() came before the session knew whether it was parked.
                self._interrupt_on_restore = False
                self._interrupt_now()

    async def _load_parked(self) -> None:
        run_id, self._restore_from = self._restore_from, None
        if run_id is None or self._last_context is not None:
            return
        # The context resolves the agent's setting (TRACING_UNSET: auto-detected) into the provider runs use.
        lookup = RunContext(parent_id=run_id, tracing_provider=self._agent.tracing_provider)
        provider = lookup._tracing_provider
        if provider is None:
            return
        try:
            # Not get() / _get_parent_trace: they read a failed lookup as "no parent", and an unparked session
            # would resume the parked run with the next message's turn and drop that message.
            trace = await provider.get_strict(lookup)
        except Exception as error:
            self._restore_from = run_id
            self._record("restore_failed", run_id=run_id, error=str(error))
            raise
        root = next((span for span in trace.values() if span.parent_call_id is None), None) if trace else None
        if root is None or not is_parked(root.status) or self._last_context is not None:
            return
        context = RunContext(id=run_id, tracing_provider=self._agent.tracing_provider)
        context.__dict__["_trace"] = trace
        prompt = (root.input or {}).get("prompt") or ""
        self._last_context, self._parked, self._last_prompt = context, True, prompt
        meta = prompt.get("metadata") if isinstance(prompt, dict) else getattr(prompt, "metadata", None)
        if meta and meta.get("kind") in (WAKE_KIND, RECOVERY_KIND):
            self._last_options = self._wake_options(meta, trace)

    def _wake_options(self, meta: dict[str, Any], trace: Any) -> _TurnOptions:
        """The options of a parked wake, from its prompt; a wake never resumes with more than it had.

        A callable policy is not stored: the agents of the run noted, by path, what it let through, and
        that becomes the policy (an agent that never resolved its tools in the run gets none). Wakes from
        before the policy was stored resume read-only.
        """
        policy = meta.get("tool_policy", "read_only")
        if policy == "custom":
            policy = _ToolPaths(_allowed_tool_paths(trace))
        elif isinstance(policy, list):
            policy = frozenset(policy)
        return _TurnOptions(tool_policy=policy, silence=self.allow_silence and bool(meta.get("silent")), steer=False)

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    # -- Turns ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._serve())

    def _work_arrived(self) -> None:
        self._work.set()
        # An event or a finished task can be the session's first work: nothing else started the worker.
        if not self.closed:
            with contextlib.suppress(RuntimeError):
                self._ensure_worker()

    async def _serve(self) -> None:
        try:
            await self._serve_turns()
        except Exception as error:
            # The next send() starts a new worker; the session itself stays usable.
            logger.error("chat_session_worker_failed", session_id=self.id, error=str(error), exc_info=True)
            raise

    async def _serve_turns(self) -> None:
        delay = _TURN_RETRY_DELAY_S
        while not self.closed:
            # No message is admitted until the state of the run the session continues is known.
            try:
                await self.restore()
                break
            except Exception as error:
                logger.warning("chat_session_restore_failed", session_id=self.id, error=str(error))
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RESTORE_RETRY_MAX_S)
        scheduler = self._scheduler
        while not self.closed:
            self._work.clear()
            if self._turn_failures >= _MAX_TURN_FAILURES:
                # The same work failed before it started, time after time (a broken model setup, a store
                # that is down): try again when new work arrives, not in a loop. The failed attempt handing
                # its work back sets _work too, so what counts is work it did not have.
                if self._resume is None and self._pending_work() <= self._attempted:
                    await self._work.wait()
                    continue
                self._turn_failures = _MAX_TURN_FAILURES - 1
            if self._turn_failures:
                await asyncio.sleep(_TURN_RETRY_DELAY_S * 2 ** (self._turn_failures - 1))
            if self._resume is not None:
                values, self._resume = self._resume, None
                if not self.parked:
                    # Another client's run answered or went past the parked turn first: these answers are for
                    # a run that no longer waits, and a resume now would re-send its prompt as a new message.
                    self._record("resume_dropped", run_id=self._last_context.id if self._last_context else None)
                    continue
                scheduler.turn_started()
                await self._run_turn(resume=values)
                continue
            if self._parked:
                if not self.parked:
                    # Another client's run answered the parked turn or went past it: go on from that run.
                    self._parked = False
                    self._record("unparked", run_id=self._last_context.id, head=scheduler.head)
                    continue
                await self._until_scheduler_or_work()
                continue
            if scheduler.head_parked:
                # Another client's run of this session waits for its caller (an approval, a question): a
                # turn now would chain onto it, and that run's resume would drop the turn's prompt. Its
                # resume, or any newer head, lets the session go on; messages stay pending until then.
                await self._until_scheduler_or_work()
                continue
            # A recovery turn goes before anything queued: it is about the turn the user saw cut, and the turns
            # after it must find it settled. Other wakes wait for the user.
            recovering = bool(self._wakes) and self._wakes[0].kind == RECOVERY_KIND
            if self._wakes and (recovering or scheduler.pending_turn() is None) and not scheduler.other_turn_first():
                wake = self._wakes.pop(0)
                scheduler.turn_started()
                await self._run_turn(wake=wake)
                continue
            if not await self._turn_due():
                continue
            # Take and busy in one step: a foreign root turn either goes before it or meets a busy owner.
            followup = scheduler.take()
            if followup is None:
                continue
            scheduler.turn_started()
            await self._run_turn(followup=followup)

    def _pending_work(self, followup: Followup | None = None, wake: _Wake | None = None) -> set[Any]:
        """Ids of the work waiting for a turn, plus what ``followup`` and ``wake`` took of it."""
        store = self._store
        events = store.pending_events() + (followup.events if followup is not None else [])
        completions = store.pending_completions() + (followup.completions if followup is not None else [])
        work: set[Any] = {event.id for event in events}
        work.update(("task", completion.get("task_id")) for completion in completions)
        work.update(("wake", item.id) for item in (self._wakes if wake is None else [*self._wakes, wake]))
        return work

    async def _until_scheduler_or_work(self) -> None:
        loop = asyncio.get_running_loop()
        changed = loop.create_task(self._scheduler._wait_change())
        work = loop.create_task(self._work.wait())
        try:
            await asyncio.wait({changed, work}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (changed, work):
                if not task.done():
                    task.cancel()

    async def _turn_due(self) -> bool:
        """True once the scheduler says a turn is due; False when other work arrived first (a wake, a resume)."""
        loop = asyncio.get_running_loop()
        ready = loop.create_task(self._scheduler.wait_ready(self._admit))
        work = loop.create_task(self._work.wait())
        try:
            await asyncio.wait({ready, work}, return_when=asyncio.FIRST_COMPLETED)
            if ready.done() and ready.result() is not None:
                return True
            # Nothing pending or running could ask for a turn: wait for new work.
            if not work.done():
                await work
            return False
        finally:
            for task in (ready, work):
                if not task.done():
                    task.cancel()

    async def _admit(self) -> None:
        """Hold a turn for collected messages until none has arrived for ``collect_window`` seconds, and while
        another client's run of the session waits for its caller (a turn would chain onto it, see _serve_turns)."""
        while True:
            if self._scheduler.head_parked:
                await self._scheduler._wait_change()
                continue
            collected = [
                event.posted_at for event in user_batch(self._store.pending_events()) if event.mode == "collect"
            ]
            if not collected:
                return
            delay = max(collected) + self.collect_window - time.time()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    def _turn_inputs(
        self, followup: Followup | None, resume: dict[str, Any] | None, wake: _Wake | None
    ) -> tuple[dict[str, Any], dict[str, Any], _TurnOptions]:
        """The agent kwargs, the log's trigger, and the options of the turn about to run."""
        kwargs = dict(self._kwargs)
        if resume is not None:
            # The run a resume chains onto drops its prompt; the agent still requires one.
            kwargs["prompt"] = self._last_prompt
            kwargs["resume"] = resume
            return kwargs, {"kind": "resume"}, self._last_options
        if wake is not None:
            options = _TurnOptions(
                tool_policy=wake.tool_policy, silence=self.allow_silence and wake.silent_ok, steer=False
            )
            # A recovery notice carries its own header (it is the runtime's, not a scheduled wake).
            text = wake.text if wake.kind == RECOVERY_KIND else f"{_WAKE_HEADER}\n\n{wake.text}"
            if wake.tool_policy == "read_only":
                text += f"\n\n{_READ_ONLY_NOTE}"
            prompt = Message(
                role="user",
                content=[TextContent(text=text)],
                metadata={
                    "source": RUNTIME_SOURCE,
                    "kind": wake.kind,
                    "wake_id": wake.id,
                    # Read back by restore(): a session continuing this run parked resumes it under the same.
                    "tool_policy": _describe_policy(wake.tool_policy),
                    "silent": options.silence,
                },
            )
            trigger = wake.trigger or {
                "kind": WAKE_KIND,
                "wake_id": wake.id,
                "key": wake.key,
                "tool_policy": _describe_policy(wake.tool_policy),
            }
        else:
            assert followup is not None
            trigger = followup.trigger
            options = _TurnOptions(silence=self.allow_silence and trigger["kind"] != "user_message")
            prompt = followup.prompt
        if options.silence:
            prompt = _with_note(prompt, _SILENCE_NOTE)
        kwargs["prompt"] = prompt
        return kwargs, trigger, options

    async def _run_turn(
        self,
        *,
        followup: Followup | None = None,
        resume: dict[str, Any] | None = None,
        wake: _Wake | None = None,
    ) -> None:
        """Run one turn the scheduler is already busy with (``turn_started``), and settle it."""
        scheduler = self._scheduler
        turn: asyncio.Task | None = None
        try:
            kwargs, trigger, options = self._turn_inputs(followup, resume, wake)
            context = RunContext(parent_id=scheduler.head, tracing_provider=self._agent.tracing_provider)
            context._tracing_provider = self._turn_provider(context._tracing_provider, context.platform_config)
            context._steering = options.steer
            context._tool_policy = options.tool_policy
            context._silence_token = SILENT_REPLY if options.silence else None
            context._cut = asyncio.Event()
            adopt_background_store(context, self._store)
            scheduler.mark_own(context)
            self._last_prompt, self._last_options = kwargs["prompt"], options
            self._attempted = self._pending_work(followup, wake)
            # What a restored session needs if this process dies mid-turn: stored from the turn's first event.
            in_flight = {
                "run_id": context.id,
                "parent_id": context.parent_id,
                "trigger": trigger,
                "events": [event.to_dict() for event in followup.events] if followup is not None else [],
                "wake": wake.to_dict() if wake is not None else None,
            }
            if resume is not None:
                in_flight["resume"] = _dump_resume(resume)
                in_flight["approved"] = self._approved_calls(resume)
            turn = asyncio.get_running_loop().create_task(self._pump(context, kwargs, followup, trigger, in_flight))
            self._turn, self._turn_context = turn, context
            try:
                await asyncio.wait({turn})
            except asyncio.CancelledError:
                turn.cancel()
                with contextlib.suppress(BaseException):
                    await turn
                raise
        finally:
            self._turn = None
            scheduler.turn_finished()
            if turn is None and followup is not None:
                followup.release()
        output = None
        if not turn.cancelled():
            error = turn.exception()
            if error is not None:
                logger.error("chat_session_turn_failed", session_id=self.id, run_id=context.id, error=str(error))
                self._turn_failures += 1
                self._record("turn_failed", run_id=context.id, error=str(error), attempt=self._turn_failures)
            else:
                output = turn.result()
        if output is not None:
            self._turn_failures = 0
        if output is None:
            # Never started (cut before its first event): the head stays on the last run that did.
            if wake is not None and not self.closed:
                self._wakes.insert(0, wake)
            if not self.closed:
                self._in_flight = None
                self._persist()
            return
        if self._last_context is not None and self._last_context is not context:
            scheduler.unmark(self._last_context)
        self._last_context = context
        status = output.status
        self._parked = is_parked(status)
        self.last_status = {"code": status.code, "reason": status.reason, "message": status.message}
        self.last_activity = time.monotonic()
        self._in_flight = None
        self._record(
            "turn_finished",
            run_id=context.id,
            status=self.last_status,
            parked=self.parked,
            silent=bool(output.metadata.get("silent")),
            **({"tool_search": output.metadata["tool_search"]} if output.metadata.get("tool_search") else {}),
            **({"error": {k: output.error.get(k) for k in ("type", "message")}} if output.error else {}),
        )

    async def _pump(
        self,
        context: RunContext,
        kwargs: dict[str, Any],
        followup: Followup | None,
        trigger: dict[str, Any],
        in_flight: dict[str, Any],
    ) -> OutputEvent | None:
        """Run one turn on its own task (so it can be cut) and copy its events into the log."""
        set_run_context(context)
        set_call_id(None)
        set_parent_call_id(None)
        output: OutputEvent | None = None
        started = False
        try:
            async with aclosing(self._agent(**kwargs)) as stream:
                async for event in stream:
                    if not started:
                        started = True
                        self._scheduler.note_run(context.id)
                        self.turns += 1
                        self.parked = False  # a resume took the answers: what is stored now is this turn
                        self._in_flight = in_flight
                        self._record("turn_started", run_id=context.id, parent_id=context.parent_id, trigger=trigger)
                        if followup is not None:
                            followup.started(context)
                        elif (root := context.root_span()) is not None:
                            # Wakes, recovery turns and resumes too: who started the turn is in its trace
                            # (an approval reviewer reads it: nobody is there to ask on a wake).
                            root.metadata["trigger"] = trigger
                    self.log.put_nowait(event)
                    if isinstance(event, OutputEvent) and event.run_id == context.id and event.parent_call_id is None:
                        output = event
        finally:
            if not started and followup is not None:
                followup.release()
        return output

    # -- Log ----------------------------------------------------------------------

    def _turn_provider(self, provider: Any, platform_config: Any = None) -> Any:
        """The provider a turn saves with: ``durability`` when given; else platform saves are buffered
        unless the host takes span deltas.

        A whole-trace save uploads the run's history again, so a turn that saved on every finished span
        would upload it dozens of times; buffered, it saves at most once per flush interval and waits only
        at the turn's end. With deltas a save is one or two spans, and ``sync`` keeps every finished span
        for crash recovery. A provider or ``TIMBAL_TRACE_DURABILITY`` that already chose a mode keeps it.
        """
        from ..state.tracing import uploads
        from ..state.tracing.providers import PlatformTracingProvider

        if not isinstance(provider, type):
            return provider
        want = self._durability
        if want is None:
            chosen = provider.durability is not None or os.environ.get(uploads.DURABILITY_ENV)
            if not issubclass(provider, PlatformTracingProvider) or chosen:
                return provider
            want = "sync" if provider.takes_deltas(platform_config) else "buffered"
        if uploads.durability(provider) == want:
            return provider
        configured = self._providers.get((provider, want))
        if configured is None:
            configured = self._providers[(provider, want)] = provider.configured(durability=want)
        return configured

    def _record(self, kind: str, **fields: Any) -> None:
        self.log.put_nowait({"type": "SESSION", "kind": kind, "t": int(time.time() * 1000), **fields})
        if kind in _PERSISTED_KINDS:
            self._persist()

    # -- Durability ---------------------------------------------------------------

    async def flush(self) -> None:
        """Wait until the session's latest state is stored (at once when it is not durable).

        The HTTP routes call it before answering: a message's ``202`` means it survives a restart.
        """
        while self._saver is not None and not self._saver.done():
            await asyncio.shield(self._saver)

    def _snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=self.id,
            agent=self._agent.name,
            config={
                "busy_mode": self.busy_mode,
                "collect_window": self.collect_window,
                "allow_silence": self.allow_silence,
                "durability": self._durability,
                "on_dead_turn": self.on_dead_turn,
            },
            head=self.head,
            turns=self.turns,
            last_status=self.last_status,
            parked=self.parked,
            last_prompt=_dump_prompt(self._last_prompt),
            last_options={
                "tool_policy": _store_policy(self._last_options.tool_policy),
                "silence": self._last_options.silence,
            },
            in_flight=self._in_flight,
            # Progress of a background task dies with the task (process-local): not worth restoring.
            pending=[event.to_dict() for event in self._store.pending_events() if event.task_id is None],
            wakes=[wake.to_dict() for wake in self._wakes],
            resume=_dump_resume(self._resume),
            log_cursor=self.log.cursor_end,
            closed=self.closed and not self._keep,
        )

    def persist(self) -> None:
        """Store a durable session now (it stores itself on every change; a new one has had none yet)."""
        self._persist()

    def _persist(self) -> None:
        """Queue a snapshot save: saves run one at a time, in order, a burst coalescing into one."""
        if self._session_store is None or self._lost:
            return
        self._dirty = True
        if self._saver is None or self._saver.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._saver = loop.create_task(self._save_loop())

    async def _save_loop(self) -> None:
        store = self._session_store
        assert store is not None
        while self._dirty and not self._lost:
            self._dirty = False
            snapshot = self._snapshot()
            try:
                self._version = await store.save(snapshot, expect_version=self._version)
            except SnapshotConflict as e:
                self._lose(str(e))
                return
            except Exception as e:
                # Kept dirty: the next change retries. A route's flush() returns: it does not hang on a store.
                self._dirty = True
                logger.error("chat_session_save_failed", session_id=self.id, error=str(e))
                return
            if not self._leased and not self.closed:
                try:
                    await store.acquire_lease(self.id, PROCESS_OWNER)
                except SessionLeased as e:
                    self._lose(str(e))
                    return
                self._leased = True
                self._lease_task = asyncio.get_running_loop().create_task(self._renew_lease())

    async def _renew_lease(self) -> None:
        store = self._session_store
        assert store is not None
        while not self.closed and not self._lost:
            await asyncio.sleep(store.lease_ttl_s / 3)
            if self.closed:
                return
            try:
                await store.acquire_lease(self.id, PROCESS_OWNER)
            except SessionLeased as e:
                self._lose(str(e))
                return
            except Exception as e:
                logger.warning("chat_session_lease_renewal_failed", session_id=self.id, error=str(e))

    def _lose(self, reason: str) -> None:
        """Another process serves this session now: stop, without writing over its state."""
        if self._lost:
            return
        self._lost = True
        logger.error("chat_session_ownership_lost", session_id=self.id, reason=reason)
        if not self.closed:
            asyncio.get_running_loop().create_task(self.close(keep=True))

    async def _settle_store(self) -> None:
        """After close: the last state stored (kept) or the row gone, and the lease let go."""
        store = self._session_store
        if store is None:
            return
        lease, self._lease_task = self._lease_task, None
        if lease is not None and not lease.done() and lease is not asyncio.current_task():
            lease.cancel()
        await self.flush()
        if self._lost:
            return
        try:
            if not self._keep:
                await store.delete(self.id)
            elif self._leased:
                await store.release_lease(self.id, PROCESS_OWNER)
        except Exception as e:
            logger.error("chat_session_close_store_failed", session_id=self.id, error=str(e))

    def _approved_calls(self, values: dict[str, Any]) -> list[str]:
        """The gated calls a resume lets run, as the recovery notice names them."""
        if self._last_context is None:
            return []
        calls = []
        for approval in self._last_context.pending_approvals():
            value = values.get(approval["approval_id"])
            approved = value is True or (isinstance(value, dict) and value.get("approved", True) is not False)
            if approved:
                name = approval.get("path", "?").rsplit(".", 1)[-1]
                calls.append(f"{name}({_brief(approval.get('input'), 80)})")
        return calls

    def _recover_resume(self, in_flight: dict[str, Any], run_id: str) -> str:
        """A resume that died before storing anything: the parked turn is answered again. The approvals it
        claimed stay claimed (``TracingProvider.claim_approval``), so the gated calls that may have started are
        not run twice; a recovery turn after it says they may have run. A provider without claims could
        run them again: the session parks instead, and the user answers again."""
        from ..state.tracing.providers.base import TracingProvider

        provider = self._agent.tracing_provider
        claims = isinstance(provider, type) and provider.claim_approval.__func__ is not (
            TracingProvider.claim_approval.__func__
        )
        self.parked = True
        if not claims:
            logger.warning("chat_session_resume_reparked", session_id=self.id, dead_run_id=run_id)
            return "reparked"
        self._resume = _load_resume(in_flight["resume"])
        approved = in_flight.get("approved") or []
        if approved and self.on_dead_turn != "drop":
            notice = _recovery_notice([], approved)
            self._wakes.insert(
                0,
                _Wake(
                    text=notice,
                    key=RECOVERY_KIND,
                    tool_policy="read_only",
                    silent_ok=True,
                    kind=RECOVERY_KIND,
                    trigger={"kind": RECOVERY_KIND, "dead_run_id": run_id, "cut": in_flight.get("cut") or "crash"},
                ),
            )
        return "resumed_again"

    def _restore_events(self, stored: list[dict[str, Any]]) -> None:
        # Restored in the same process (an idle session let go), the run's task store may still hold them.
        held = {event.id for event in self._store.pending_events()}
        for data in stored:
            if data["id"] in held:
                continue
            event = self._store.restore_event(data)
            if event.source == "user":
                self._watch(event)

    async def _load_run(self, run_id: str | None) -> RunContext | None:
        """A stored run of this session as a context (its trace loaded), or None when it was never stored."""
        if run_id is None:
            return None
        lookup = RunContext(parent_id=run_id, tracing_provider=self._agent.tracing_provider)
        trace = await lookup._get_parent_trace()
        if trace is None or not trace.as_records():
            return None
        loaded = RunContext(id=run_id, parent_id=trace.parent_id, tracing_provider=self._agent.tracing_provider)
        loaded._trace = trace
        return loaded

    async def _recover(self, in_flight: dict[str, Any]) -> str:
        """Settle the turn that was running when the session was last stored (see :meth:`restore`)."""
        run_id = in_flight["run_id"]
        self._in_flight = None
        context = await self._load_run(run_id)
        root = context.root_span() if context is not None else None
        cut = in_flight.get("cut")
        if root is not None and root.t1 is not None and cut is None:
            # It finished; the process died before the session stored that.
            self._scheduler.head = run_id
            self.parked = is_parked(root.status)
            self.last_status = _status_dict(root.status)
            self._last_context = context if self.parked else None
            return "finished"
        if root is None:
            # Died before its first save: nothing it did is known. Its input goes back in the queue.
            self._scheduler.head = in_flight.get("parent_id")
            self._restore_events(in_flight.get("events") or [])
            if in_flight.get("wake") is not None:
                self._wakes.insert(0, _Wake.from_dict(in_flight["wake"]))
            if in_flight.get("resume") is not None:
                return self._recover_resume(in_flight, run_id)
            return "requeued"
        self._scheduler.head = run_id
        self.parked = False
        if self.on_dead_turn == "drop":
            return "dropped"
        completed, unconfirmed = _tool_calls(context)
        notice = _recovery_notice(completed, unconfirmed)
        if self.on_dead_turn == "notify":
            # Rides with the next turn: it does not start one.
            self._store.post_event(notice, key=RECOVERY_KIND, announce="silent")
            return "notified"
        self._wakes.insert(
            0,
            _Wake(
                text=notice,
                key=RECOVERY_KIND,
                tool_policy="read_only",
                silent_ok=True,
                kind=RECOVERY_KIND,
                trigger={"kind": RECOVERY_KIND, "dead_run_id": run_id, "cut": cut or "crash"},
            ),
        )
        return "continued"

    def _watch(self, event: SessionEvent) -> None:
        if self.closed:
            return
        event.on_settle(
            lambda settled: self._record(
                "message_settled", event_id=settled.id, state=settled.state, run_id=settled.run_id
            )
        )
