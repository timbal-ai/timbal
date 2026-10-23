"""Per-session admission: the turns a session starts on its own, and everyone else's.

A channel that owns a conversation (the text ``AgentContinuation`` driver, a
``VoiceSession``) holds one :class:`SessionScheduler`. It owns the session's
:class:`~timbal.state.background.BackgroundTaskStore`, decides when finished
tasks and posted events deserve a turn of their own (their announce policy),
and admits every other root turn on the session: refused or queued while one
of the owner's turns is in flight, otherwise let through.
"""

import asyncio
import contextlib
import inspect
import weakref
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..state.background import (
    Announce,
    BackgroundTaskStore,
    SessionEvent,
    _check_announce,
    adopt_background_store,
    format_session_notice,
)

if TYPE_CHECKING:
    from ..state.context import RunContext
    from ..types.message import Message

OnBusy = Literal["reject", "enqueue"]
Urgency = Literal["when_idle", "interrupt"]

_PARKED_REASONS = frozenset({"approval_required", "input_required"})


def is_parked(status: Any) -> bool:
    """A run status that leaves the run parked on its caller (an approval, ``ask_user``).

    The next run chained onto it resumes it, and memory resolution drops that
    run's prompt: a turn started for an event would be swallowed whole.
    """
    if isinstance(status, dict):
        code, reason = status.get("code"), status.get("reason")
    else:
        code, reason = getattr(status, "code", None), getattr(status, "reason", None)
    return code == "cancelled" and reason in _PARKED_REASONS


class SessionBusyError(RuntimeError):
    """Another turn of the session is in flight and its owner refuses to queue."""


@dataclass(eq=False)
class Followup:
    """A turn the session starts on its own: its prompt, and what it took from the store.

    Until :meth:`started` the work is only held: :meth:`release` hands it back,
    so a turn cancelled before its run exists loses nothing.
    """

    prompt: "Message"
    store: BackgroundTaskStore
    completions: list[dict[str, Any]]
    events: list[SessionEvent]
    urgent: bool = False
    is_started: bool = False

    @property
    def task_ids(self) -> list[str]:
        return [note["task_id"] for note in self.completions]

    @property
    def trigger(self) -> dict[str, Any]:
        """Root-span ``metadata.trigger`` for the receiving run."""
        trigger: dict[str, Any] = {
            "kind": "session_event" if self.events else "background_completion",
            "task_ids": self.task_ids,
        }
        if self.events:
            trigger["event_ids"] = [event.id for event in self.events]
        if self.urgent:
            trigger["announce"] = "interrupt"
        return trigger

    def started(self, run_context: "RunContext") -> None:
        """The receiving run exists: stamp it, and settle the events as delivered."""
        self.is_started = True
        root = run_context.root_span()
        if root is not None:
            root.metadata["trigger"] = self.trigger
        record_delivery(run_context, root, self.events)

    def release(self) -> None:
        """Hand everything back to the store if no run ever started with it."""
        if not self.is_started:
            self.is_started = True
            self.store.requeue(self.completions, self.events)


def record_delivery(run_context: Any, span: Any, events: list[SessionEvent]) -> None:
    """Settle ``events`` as delivered to ``run_context``'s run and account for them on ``span``."""
    for event in events:
        event._settle("delivered", run_context.id)
    if span is None or not events:
        return
    span.usage["session_events:delivered"] = span.usage.get("session_events:delivered", 0) + len(events)
    records = [record for event in events for record in event.guardrails]
    if records:
        from ..guardrails.apply import record_guardrails_metadata

        record_guardrails_metadata(span, records, run_context=run_context)


def take_followup(store: BackgroundTaskStore, *, urgent: bool = False) -> Followup | None:
    """Take every pending notice and event for a turn the session starts itself, or None.

    Taking is the commitment: start the turn before anything else can take a
    turn in the session, then call :meth:`Followup.started`, or
    :meth:`Followup.release` if it never runs.
    """
    completions = store.drain_completions()
    events = store.drain_events(settle=False)
    if not completions and not events:
        return None
    return Followup(
        prompt=format_session_notice(completions, events, own_turn=True),
        store=store,
        completions=completions,
        events=events,
        urgent=urgent,
    )


class SessionScheduler:
    """Admits the turns of one session for the channel that owns it.

    ``completions`` / ``events`` are the channel's default announce policies;
    a task (``Runnable.background_announce``) or an event (``announce=``) can
    override them. ``on_busy`` is what happens to another root turn while one
    of the owner's turns is in flight: ``"reject"`` (``SessionBusyError``) or
    ``"enqueue"`` (it waits for that turn to end). ``on_preempt`` runs when
    another turn is let in while the owner is idle. ``can_interrupt``: the
    channel cuts its own turn for an ``interrupt`` item (voice), which then
    goes ahead of turns still queued. ``on_work`` runs whenever the store
    gains something (a finished task, an event), for a channel that has no
    waiter running at the time.
    """

    def __init__(
        self,
        *,
        channel: Any = None,
        completions: Announce = "when_idle",
        events: Announce = "when_idle",
        on_busy: OnBusy = "reject",
        on_preempt: Callable[[], None] | None = None,
        can_interrupt: bool = False,
        on_work: Callable[[], None] | None = None,
    ) -> None:
        _check_announce(completions)
        _check_announce(events)
        if on_busy not in ("reject", "enqueue"):
            raise ValueError(f"on_busy must be 'reject' or 'enqueue', got {on_busy!r}")
        self._channel = weakref.ref(channel) if channel is not None else None
        self.completions: Announce = completions
        self.events: Announce = events
        self.on_busy: OnBusy = on_busy
        self.can_interrupt = can_interrupt
        self._on_preempt = on_preempt
        self._on_work = on_work
        self.store: BackgroundTaskStore | None = None
        # One of the owner's turns is committed to or in flight.
        self.busy = False
        # Other root turns admitted and still running, and those queued behind the owner.
        self._others: set[str] = set()
        self._others_waiting = 0
        # The owner is cutting its own turn to start another: queued turns keep waiting.
        self._holding = 0
        # The session's latest root run, the owner's or an admitted one: where the next turn chains.
        self.head: str | None = None
        # The head is another channel's run that ended parked on its caller (see is_parked).
        self.head_parked = False
        self._changed: asyncio.Future[None] | None = None

    @property
    def channel(self) -> Any:
        return self._channel() if self._channel is not None else None

    # -- Ownership ------------------------------------------------------------

    def bind(self, store: BackgroundTaskStore) -> None:
        """Own ``store`` (idempotent). Raises when another channel owns it."""
        if store.continuation_owner() is not self:
            try:
                store.acquire_continuation(self)
            except RuntimeError:
                raise RuntimeError(
                    "This session's turns are owned by another channel (a continuation stream or a voice call)."
                ) from None
        self.store = store

    def release(self, *stores: BackgroundTaskStore | None) -> None:
        """Stop owning the bound store and ``stores`` (a store a spawn created joins the owner unbound)."""
        for store in {self.store, *stores}:
            if store is not None:
                store.release_continuation(self)
        self._notify()

    def mark_own(self, run_context: Any) -> None:
        """Mark ``run_context`` as one of the owner's runs; a store it creates joins the owner."""
        run_context._continuation_owner = weakref.ref(self)

    def unmark(self, run_context: Any) -> None:
        marker = run_context._continuation_owner
        if marker is not None and marker() is self:
            run_context._continuation_owner = None

    # -- The owner's turns ------------------------------------------------------

    def turn_started(self) -> None:
        self.busy = True

    def turn_finished(self) -> None:
        self.busy = False
        self._notify()

    def note_run(self, run_id: str) -> None:
        """One of the owner's runs started: it is the session's head now."""
        self.head = run_id
        self.head_parked = False

    @contextlib.contextmanager
    def holding(self) -> Iterator[None]:
        """Keep queued turns waiting while the owner cuts its own turn to start another.

        Cutting a turn finishes it, which would let a queued turn in ahead of
        the one the cut was for; start that one (``turn_started``) inside.
        """
        self._holding += 1
        try:
            yield
        finally:
            self._holding -= 1
            self._notify()

    @property
    def other_turn_active(self) -> bool:
        """Another root turn runs in the session, or waits for the owner's to end."""
        return bool(self._others) or self._others_waiting > 0

    def other_turn_first(self, urgency: Urgency | None = None) -> bool:
        """Another root turn goes before the owner's next one.

        A running one always; a queued one unless the owner's turn is an
        ``interrupt`` on a channel that cuts its own turns for it.
        """
        if self._others:
            return True
        return self._others_waiting > 0 and not (self.can_interrupt and urgency == "interrupt")

    async def until_no_other_turn(self, *, include_queued: bool = True) -> None:
        """Wait out other turns in the session before starting one of the owner's.

        ``include_queued=False`` for a turn the owner has already committed to
        (a caller who spoke): turns queued behind the owner go after it, and
        waiting for them would wait for itself.
        """
        while self._others or (include_queued and self._others_waiting):
            await self._wait_change()

    def work_arrived(self) -> None:
        """The store gained a finished task or an event (called by the store, on its loop)."""
        self._notify()
        if self._on_work is not None:
            self._on_work()

    # -- Everyone else's turns ---------------------------------------------------

    async def admit_foreign(self, run_context: Any, store: BackgroundTaskStore) -> None:
        """Let another root turn into the session, queue it, or refuse it.

        An admitted turn continues from the session's head: one started with
        an older run as its parent (queued while the owner's turn ran, or sent
        by a client that had not seen the latest turn yet) is re-chained before
        anything reads its history or session data.
        """
        if self._occupied:
            if self.on_busy == "reject":
                raise SessionBusyError(
                    "A turn of this session is in flight; wait for its OUTPUT before starting another turn."
                )
            self._others_waiting += 1
            try:
                while self._occupied and store.continuation_owner() is self:
                    await self._wait_change()
            finally:
                self._others_waiting -= 1
            if store.continuation_owner() is not self:
                # The owner let go while this turn waited: ask whoever owns it now.
                await store.admit_root_turn(run_context)
                return
        self._others.add(run_context.id)
        run_context._foreign_turn_of = weakref.ref(self)
        if run_context._bg_store is None:
            # Chained onto a run from before the session had its store: it joins the session now.
            adopt_background_store(run_context, store)
        head, self.head = self.head, run_context.id
        self.head_parked = False
        if self._on_preempt is not None:
            self._on_preempt()
        self._notify()
        if head is not None and run_context.parent_id != head:
            await run_context._continue_from(head)

    @property
    def _occupied(self) -> bool:
        return self.busy or bool(self._others) or self._holding > 0

    def foreign_finished(self, run_id: str, status: Any = None) -> None:
        """An admitted turn ended with root ``status``; a parked head holds back turns chaining onto it."""
        self._others.discard(run_id)
        if run_id == self.head:
            self.head_parked = is_parked(status)
        self._notify()

    # -- Turns the session starts itself -----------------------------------------

    def announce_of(self, item: dict[str, Any] | SessionEvent) -> Announce:
        if isinstance(item, SessionEvent):
            return item.announce or self.events
        return item.get("announce") or self.completions

    def pending_turn(self) -> Urgency | None:
        """Whether pending work asks for a turn of its own, and how urgently."""
        store = self.store
        if store is None:
            return None
        store.reconcile_completions()
        policies = {self.announce_of(note) for note in store.pending_completions()}
        policies.update(self.announce_of(event) for event in store.pending_events())
        if "interrupt" in policies:
            return "interrupt"
        if "when_idle" in policies:
            return "when_idle"
        return None

    def may_need_turn(self) -> bool:
        """Something pending, or a running child that is not silent, could ask for a turn."""
        if self.pending_turn() is not None:
            return True
        store = self.store
        return store is not None and any(
            (record.announce or self.completions) != "silent" for record in store.running_tasks()
        )

    async def wait_ready(self, admit: Callable[[], Awaitable[Any] | Any] | None = None) -> Urgency | None:
        """Wait until a turn of the session's own is due and ``admit`` lets it start.

        Event-driven: wakes on finished tasks and posted events. ``None`` once
        nothing pending or running could ask for one. Other turns in the
        session go first. Call :meth:`take` right after, without awaiting.
        """
        store = self.store
        if store is None:
            return None
        while True:
            ready = await store.wait_for_completion(
                wanted=lambda _store: self.pending_turn() is not None,
                alive=lambda _store: self.may_need_turn(),
            )
            if not ready:
                return None
            if self.other_turn_first(self.pending_turn()):
                # Re-evaluated on every change: an interrupt arriving meanwhile may go ahead of queued turns.
                await self._wait_change()
                continue
            if admit is not None:
                result = admit()
                if inspect.isawaitable(result):
                    await result
            urgency = self.pending_turn()
            if urgency is not None and not self.other_turn_first(urgency):
                return urgency

    def take(self) -> Followup | None:
        """Take everything pending for a turn of the session's own, unless another turn comes first."""
        if self.store is None:
            return None
        urgency = self.pending_turn()
        if urgency is None or self.other_turn_first(urgency):
            return None
        return take_followup(self.store, urgent=urgency == "interrupt")

    # -- Internals ------------------------------------------------------------------

    def _notify(self) -> None:
        changed, self._changed = self._changed, None
        if changed is not None and not changed.done():
            changed.set_result(None)

    async def _wait_change(self) -> None:
        if self._changed is None:
            self._changed = asyncio.get_running_loop().create_future()
        # Shielded: one waiter being cancelled must not cancel the future the others share.
        await asyncio.shield(self._changed)
