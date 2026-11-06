"""A text conversation that stays open: messages in at any time, one turn at a time, events out."""

import asyncio
import contextlib
import math
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
from ..types.approval import Cancel
from ..types.content import TextContent
from ..types.events import OutputEvent
from ..types.message import RUNTIME_SOURCE, Message
from .session_scheduler import Followup, SessionScheduler, is_parked, user_batch

if TYPE_CHECKING:
    from .agent import Agent

__all__ = ["SEND_MODES", "SILENT_REPLY", "ChatSession", "SendMode", "ToolPolicy"]

logger = structlog.get_logger("timbal.core.chat_session")

SendMode = Literal["steer", "followup", "collect", "interrupt"]
SEND_MODES: tuple[str, ...] = ("steer", "followup", "collect", "interrupt")

ToolPolicy = Literal["read_only"] | Collection[str] | Callable[[Any], bool]
"""Which tools a turn offers its model: ``"read_only"`` (tools declared ``read_only``,
MCP tools the server marks ``readOnlyHint``, the agent's own readers), a collection of
tool names, or a callable deciding per tool."""

SILENT_REPLY = "NO_REPLY"
"""What a turn the runtime started answers to stay silent (``turn_finished.silent``)."""

WAKE_KIND = "wake"

_SILENCE_NOTE = f"If nothing here needs the user's attention, reply with exactly {SILENT_REPLY} and nothing else."
_WAKE_HEADER = "Scheduled wake-up (the user did not write this):"
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


@dataclass(frozen=True)
class _TurnOptions:
    """What a turn runs under; a resume of it runs under the same."""

    tool_policy: Any = None
    silence: bool = False


def _check_tool_policy(policy: Any) -> Any:
    if policy is None or policy == "read_only" or callable(policy):
        return policy
    if isinstance(policy, str) or not isinstance(policy, Collection):
        raise ValueError("tool_policy must be 'read_only', a collection of tool names, a callable, or None.")
    names = frozenset(policy)
    if not all(isinstance(name, str) for name in names):
        raise ValueError("tool_policy names must be strings.")
    return names


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


class ChatSession:
    """One agent's conversation with a user, open for messages while its turns run.

    :meth:`send` takes a user message at any time. With no turn running it
    starts one. While a turn runs, ``mode`` decides what happens to it:

    - ``"steer"``: the running turn takes it after its current tool batch, as a
      user message, and does not end before answering it (``Agent.steering``,
      on for these runs unless the agent turns it off). Messages a turn ends
      without taking start the next turn together.
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
        **kwargs: Any,
    ) -> None:
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
        self.id: str = uuid7(as_type="hex")  # type: ignore[assignment]
        self.busy_mode: SendMode = busy_mode
        self.collect_window = collect_window
        self.allow_silence = allow_silence
        self.log = BackgroundEventLog(max_events=max_log_events, max_bytes=max_log_bytes)
        self.closed = False
        self.parked = False
        """The last turn ended waiting on an approval or ``ask_user``: :meth:`resume` answers it."""
        self.turns = 0
        self.last_status: dict[str, Any] | None = None
        self.last_activity = time.monotonic()
        self._agent = agent
        self._kwargs = kwargs
        store = store_for_run(parent_id) if parent_id is not None else None
        self._store = store if store is not None else BackgroundTaskStore()
        self._scheduler = SessionScheduler(
            channel=self, on_busy="enqueue", can_interrupt=True, on_work=self._work_arrived
        )
        self._scheduler.bind(self._store)
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
            event = self._store.post_event(text, source="user", mode=mode)
        self.last_activity = time.monotonic()
        self._record("message", event_id=event.id, text=event.text, mode=mode, busy=busy)
        self._watch(event)
        self._ensure_worker()
        if mode == "interrupt":
            self.interrupt()
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

    def interrupt(self) -> bool:
        """Cut what the session is doing; queued messages then start the next turn.

        Cancels the running turn, or, when the session is parked, every pending
        approval and question (``Cancel``: the gated tools never run, the run
        ends ``cancelled``). False when there is neither.
        """
        turn = self._turn
        if turn is not None and not turn.done():
            self._record("interrupt", run_id=self.run_id)
            turn.cancel()
            return True
        if self.parked and self._resume is None and self._last_context is not None:
            reason = "The user moved on before answering."
            pending = [a["approval_id"] for a in self._last_context.pending_approvals()]
            pending += [i["interaction_id"] for i in self._last_context.pending_interactions()]
            if not pending:
                return False
            self._record("interrupt", run_id=self._last_context.id, abandoned=pending)
            self.resume({pending_id: Cancel(reason=reason) for pending_id in pending})
            return True
        return False

    def resume(self, values: dict[str, Any]) -> None:
        """Answer the parked turn: ``{approval_id: ...}`` / ``{interaction_id: value}``, as for ``resume=``."""
        if self.closed:
            raise RuntimeError("The chat session is closed.")
        if not self.parked:
            raise RuntimeError("No turn of this session is waiting for an answer.")
        self._resume = dict(values)
        self.last_activity = time.monotonic()
        self._work.set()
        self._ensure_worker()

    def events(self, after: int = 0) -> AsyncIterator[Any]:
        """Replay the log from cursor ``after``, then follow it until :meth:`close`."""
        return self.log.subscribe(after)

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
        }
        if self.parked and self._last_context is not None:
            state["pending_approvals"] = self._last_context.pending_approvals()
            state["pending_interactions"] = self._last_context.pending_interactions()
        return state

    async def close(self) -> None:
        """Stop the session: the running turn is cancelled, the store let go. Detached children keep running."""
        if self.closed:
            return
        self.closed = True
        worker, self._worker = self._worker, None
        if worker is not None and not worker.done():
            worker.cancel()
            with contextlib.suppress(BaseException):
                await worker
        if self._last_context is not None:
            self._scheduler.unmark(self._last_context)
        self._scheduler.release(self._store)
        forget_idle_background_store(self._store)
        self._record("closed")
        self.log.close()

    async def __aenter__(self) -> "ChatSession":
        return self

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
        scheduler = self._scheduler
        while not self.closed:
            self._work.clear()
            if self._resume is not None:
                values, self._resume = self._resume, None
                scheduler.turn_started()
                await self._run_turn(resume=values)
                continue
            if self.parked:
                await self._work.wait()
                continue
            if self._wakes and scheduler.pending_turn() is None and not scheduler.other_turn_first():
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
        """Hold a turn for collected messages until none has arrived for ``collect_window`` seconds."""
        while True:
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
            options = _TurnOptions(tool_policy=wake.tool_policy, silence=self.allow_silence and wake.silent_ok)
            text = f"{_WAKE_HEADER}\n\n{wake.text}"
            if wake.tool_policy == "read_only":
                text += f"\n\n{_READ_ONLY_NOTE}"
            prompt = Message(
                role="user",
                content=[TextContent(text=text)],
                metadata={"source": RUNTIME_SOURCE, "kind": WAKE_KIND, "wake_id": wake.id},
            )
            trigger = {
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
            context._steering = True
            context._tool_policy = options.tool_policy
            context._silence_token = SILENT_REPLY if options.silence else None
            adopt_background_store(context, self._store)
            scheduler.mark_own(context)
            self._last_prompt, self._last_options = kwargs["prompt"], options
            turn = asyncio.get_running_loop().create_task(self._pump(context, kwargs, followup, trigger))
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
            else:
                output = turn.result()
        if output is None:
            # Never started (cut before its first event): the head stays on the last run that did.
            if wake is not None and not self.closed:
                self._wakes.insert(0, wake)
            return
        if self._last_context is not None and self._last_context is not context:
            scheduler.unmark(self._last_context)
        self._last_context = context
        status = output.status
        self.parked = is_parked(status)
        self.last_status = {"code": status.code, "reason": status.reason, "message": status.message}
        self.last_activity = time.monotonic()
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
                        self._record("turn_started", run_id=context.id, parent_id=context.parent_id, trigger=trigger)
                        if followup is not None:
                            followup.started(context)
                    self.log.put_nowait(event)
                    if isinstance(event, OutputEvent) and event.run_id == context.id and event.parent_call_id is None:
                        output = event
        finally:
            if not started and followup is not None:
                followup.release()
        return output

    # -- Log ----------------------------------------------------------------------

    def _record(self, kind: str, **fields: Any) -> None:
        self.log.put_nowait({"type": "SESSION", "kind": kind, "t": int(time.time() * 1000), **fields})

    def _watch(self, event: SessionEvent) -> None:
        if self.closed:
            return
        event.on_settle(
            lambda settled: self._record(
                "message_settled", event_id=settled.id, state=settled.state, run_id=settled.run_id
            )
        )
