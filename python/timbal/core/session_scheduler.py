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
    forget_idle_background_store,
    format_session_notice,
    format_user_message,
)

if TYPE_CHECKING:
    from ..state.context import RunContext
    from ..types.message import Message

OnBusy = Literal["reject", "enqueue"]
Urgency = Literal["when_idle", "interrupt"]

_OWNED_ELSEWHERE = "This session's turns are owned by another channel (a continuation stream or a voice call)."

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
        if any(event.source == "user" for event in self.events):
            kind = "user_message"
        else:
            kind = "session_event" if self.events else "background_completion"
        trigger: dict[str, Any] = {"kind": kind, "task_ids": self.task_ids}
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


def user_batch(events: list[SessionEvent]) -> list[SessionEvent]:
    """The user messages one turn starts with, out of ``events`` (oldest first).

    Messages that interrupted (``announce="interrupt"``) go first, together.
    Otherwise a ``followup`` message gets its turn alone, and the leading run
    of ``steer`` / ``collect`` messages shares one.

    Never a person's message together with one a running call's code posted
    (``sent_by``): they become one message, and the person's words in it would
    lose their standing with an approval reviewer (their latest say, a
    retraction, read as part of a tool's request). The leading ones that share
    a sender go; the rest wait for the next turn.
    """
    users = [event for event in events if event.source == "user"]
    urgent = [event for event in users if event.announce == "interrupt"]
    if urgent:
        return _one_sender(urgent)
    if not users or users[0].mode == "followup":
        return users[:1]
    batch: list[SessionEvent] = []
    for event in users:
        if event.mode == "followup":
            break
        batch.append(event)
    return _one_sender(batch)


def _one_sender(batch: list[SessionEvent]) -> list[SessionEvent]:
    """The leading messages of ``batch`` posted the same way as its first: by a person, or by a call's code."""
    by_code = batch[0].sent_by is not None
    for index, event in enumerate(batch):
        if (event.sent_by is not None) != by_code:
            return batch[:index]
    return batch


def take_followup(
    store: BackgroundTaskStore, *, urgent: bool = False, users: bool = True, others: bool = True
) -> Followup | None:
    """Take every pending notice and event for a turn the session starts itself, or None.

    User messages come first: the turn starts with :func:`user_batch` as its
    prompt, and leaves the rest pending for that turn's start to pick up (or
    for the turns after it). Taking is the commitment: start the turn before
    anything else can take a turn in the session, then call
    :meth:`Followup.started`, or :meth:`Followup.release` if it never runs.
    ``users`` / ``others``: whether user messages / everything else are this
    channel's to take (a call leaves typed messages to the chat it borrowed
    the session from, which answers nothing else meanwhile).
    """
    batch = user_batch(store.pending_events()) if users else []
    if batch:
        taken = {id(event) for event in batch}
        events = store.drain_events(settle=False, where=lambda event: id(event) in taken)
        return Followup(prompt=format_user_message(events), store=store, completions=[], events=events, urgent=urgent)
    if not others:
        return None
    completions = store.drain_completions()
    events = store.drain_events(settle=False, where=None if users else lambda event: event.source != "user")
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

    ``lends`` (a chat) / ``borrows`` (a call): a call joining a session a chat
    owns takes it over while it is live (see :meth:`take_over`), and gives it
    back when it ends; meanwhile the chat only answers user messages, its
    turns queue behind the call's, and the call (``user_messages=False``)
    answers everything but them.

    While a question waits (a run that ended on an approval or ``ask_user``,
    see :attr:`asking`) the session is its asker's: the answer is a turn that
    resumes that run, and any other turn started on it would be taken for the
    answer and lose its prompt. So a chat does not lend the session while its
    own question waits, a call hands it back when a turn of the chat's ends on
    one, and a turn of the other channel that brings no answer queues until
    the question is answered or another run goes past it (:meth:`_holds`).
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
        lends: bool = False,
        borrows: bool = False,
        user_messages: bool = True,
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
        # Stores the channel let go of while turns admitted or queued here were still on (see release).
        self._releasing: set[BackgroundTaskStore] = set()
        # Who takes each of those over once released (see take_over).
        self._successors: dict[BackgroundTaskStore, weakref.ref[SessionScheduler]] = {}
        # The owner is cutting its own turn to start another: queued turns keep waiting.
        self._holding = 0
        # The session's latest root run, the owner's or an admitted one: where the next turn chains.
        self.head: str | None = None
        # The head ended parked on its caller (see is_parked): another channel's run (foreign_finished), or
        # one of the owner's own when its channel says so (turn_finished, note_parked).
        self.head_parked = False
        # The channel's own latest run (note_run): the head is its own while they are the same.
        self._own_head: str | None = None
        self._changed: asyncio.Future[None] | None = None
        self.lends, self.borrows, self.user_messages = lends, borrows, user_messages
        # The call the session is lent to (while live), or waits to be lent to once this channel is idle.
        self._lent_to: weakref.ref[SessionScheduler] | None = None
        self._borrower: weakref.ref[SessionScheduler] | None = None

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
                raise RuntimeError(_OWNED_ELSEWHERE) from None
        self._releasing.discard(store)
        self._successors.pop(store, None)
        self._lent_to = self._borrower = None
        self.store = store

    def take_over(self, store: BackgroundTaskStore) -> None:
        """Own ``store``, or own it next if the channel owning it is letting go of it.

        Until that channel's admitted and queued turns end, it still decides
        for the session; then the store passes here, with its head, and
        :meth:`work_arrived` runs. A call (``borrows``) joining a channel that
        ``lends`` is lent the session as soon as that channel is idle (its
        turns until then are admitted there), and it goes back when the call
        releases it. Raises when another channel keeps it.
        """
        owner = store.continuation_owner()
        if isinstance(owner, SessionScheduler) and owner is not self:
            if owner.releasing(store):
                owner._successors[store] = weakref.ref(self)
                return
            if self.borrows and owner.lends:
                owner._borrower = weakref.ref(self)
                owner._maybe_lend()
                return
        self.bind(store)

    def join(self, store: BackgroundTaskStore) -> None:
        """Own ``store``; a channel that ``lends``, opened on a live call's session, starts lent to the call."""
        owner = store.continuation_owner()
        if self.lends and isinstance(owner, SessionScheduler) and owner is not self and owner.borrows:
            lender = owner.successor(store)
            if lender is not None and lender is not self and lender.lends:
                # Another chat already lent this session to the call: one chat per session, as when
                # the chat itself owns it. (The call hands it back to one channel.)
                raise RuntimeError(_OWNED_ELSEWHERE)
            self.store = store
            self._lent_to = weakref.ref(owner)
            owner._successors[store] = weakref.ref(self)
            return
        self.bind(store)

    @property
    def lent(self) -> bool:
        """The session is lent to a live call: only user messages are this channel's to answer."""
        store = self.store
        return self._lent_to is not None and store is not None and store.continuation_owner() is not self

    @property
    def asking(self) -> bool:
        """This channel's own question waits: its latest run is the head and ended on an approval or ``ask_user``."""
        return self.head_parked and self.head is not None and self.head == self._own_head

    def _deciding(self) -> "SessionScheduler":
        """Who decides for the session now: the call it is lent to, else this channel."""
        owner = self._lent_to() if self._lent_to is not None and self.lent else None
        return owner if owner is not None else self

    @property
    def session_head(self) -> str | None:
        """The session's latest root run, also while a call has the session (its head, not this channel's own)."""
        return self._deciding().head

    @property
    def session_parked(self) -> bool:
        """The session's head waits for an answer, whoever ran it and whoever has the session."""
        return self._deciding().head_parked

    def _maybe_lend(self) -> None:
        """Hand the session to the call waiting for it, once nothing of this channel's is in flight.

        Not while its own question waits: only this channel can answer it, and the call's next turn
        would chain onto that run in the answer's place.
        """
        borrower = self._borrower() if self._borrower is not None else None
        store = self.store
        if borrower is None or store is None or self._occupied or self._others_waiting or self.asking:
            return
        if store.continuation_owner() is not self:
            self._borrower = None
            return
        self._borrower = None
        store.release_continuation(self)
        borrower.bind(store)
        borrower._successors[store] = weakref.ref(self)
        if self.head is not None:
            borrower.head, borrower.head_parked = self.head, self.head_parked
        self._lent_to = weakref.ref(borrower)
        self._notify()
        borrower.work_arrived()

    def _lender_of(self, run_context: Any) -> bool:
        """``run_context`` is a run of the channel this one borrowed the session from."""
        marker = getattr(run_context, "_continuation_owner", None)
        owner = marker() if marker is not None else None
        return owner is not None and any(ref() is owner and owner.lends for ref in self._successors.values())

    def _holds(self, run_context: Any) -> bool:
        """``run_context`` is a turn of the channel sharing this session that must wait for a question.

        It brings no answer, so started now it would resume the run that asked: its prompt dropped,
        the gated tools asked again. A turn with answers is the answer and passes; so does any
        other client's (it decides for itself, as before).

        The chat that lent the session waits for any run parked ahead of it, whoever asked (the
        call, a run that answered the call's question and asked another, another client), as it
        does when the session is its own. A call's turn waits for the chat's own question, while
        the chat is there to answer it (also a call the chat has not heard of yet: one with no
        store of its own first reaches the chat with this turn). Once the chat lets go of the
        session the question is the call's (see :meth:`release`), and a run another client left
        parked it takes over as a call alone does.
        """
        if not self.head_parked or run_context._resume_values:
            return False
        if self._lender_of(run_context):
            return True
        if not self.lends or not self.asking or self.store in self._releasing:
            return False
        return self._caller(run_context) is not None

    def _caller(self, run_context: Any) -> "SessionScheduler | None":
        """The call ``run_context`` is a turn of (a channel that borrows the session), if it is one."""
        marker = getattr(run_context, "_continuation_owner", None)
        channel = marker() if marker is not None else None
        return channel if channel is not None and channel is not self and channel.borrows else None

    def _return_to_asker(self) -> None:
        """Give the session back to the chat that lent it, when a turn of the chat's ended on a question.

        The chat's answer is then a turn of its own, and this call's turns queue behind the
        question there (:meth:`_holds`) until the session is lent again.
        """
        store = self.store
        if not self.head_parked or store is None or store in self._releasing or store.continuation_owner() is not self:
            return
        ref = self._successors.get(store)
        lender = ref() if ref is not None else None
        if lender is None or not lender.lends or lender._own_head != self.head:
            return
        del self._successors[store]
        store.release_continuation(self)
        self.store = None
        lender._succeed(self, store)
        lender._borrower = weakref.ref(self)

    def _hand_back(self, store: BackgroundTaskStore) -> None:
        """Pass a released store to the channel waiting for it (the chat that lent it), with the head."""
        successor = self._successors.pop(store, None)
        successor = successor() if successor is not None else None
        if successor is not None and store.continuation_owner() is None:
            successor._succeed(self, store)
        else:
            # The channel's own forget ran while this still owned the store.
            forget_idle_background_store(store)

    def release(self, *stores: BackgroundTaskStore | None) -> None:
        """Stop owning the bound store and ``stores`` (a store a spawn created joins the owner unbound).

        Turns admitted or queued here keep the session until they end, so a
        channel taking it over meanwhile waits behind them like any turn;
        another channel's own run admitted here takes it at once.
        """
        targets = {store for store in {self.store, *stores} if store is not None}
        borrower, self._borrower = (self._borrower() if self._borrower is not None else None), None
        if borrower is not None and self.store is not None and self.store.continuation_owner() is self:
            # The call waiting to be lent the session takes it over as it stands: with its head, and the
            # question waiting on it, which is the call's from here (nobody is left to answer it here).
            self._successors.setdefault(self.store, weakref.ref(borrower))
        for store in targets:
            owner = store.continuation_owner()
            if isinstance(owner, SessionScheduler) and owner is not self:
                # Waiting to take it over from a channel letting go: not any more.
                successor = owner._successors.get(store)
                if successor is not None and successor() is self:
                    del owner._successors[store]
                # Nor to be lent it by a chat still busy (or asking): it would lend it to a call that ended.
                if owner._borrower is not None and owner._borrower() is self:
                    owner._borrower = None
                    owner._notify()
        if self._others or self._others_waiting:
            self._releasing |= targets
        else:
            for store in targets:
                owned = store.continuation_owner() is self
                store.release_continuation(self)
                if owned and store in self._successors:
                    self._hand_back(store)
        self._notify()

    def releasing(self, store: BackgroundTaskStore) -> bool:
        """The channel let go of ``store``; turns admitted or queued here still hold it."""
        return store in self._releasing

    def successor(self, store: BackgroundTaskStore) -> "SessionScheduler | None":
        """Who takes ``store`` over once released (a call that joined meanwhile), if anyone."""
        ref = self._successors.get(store)
        return ref() if ref is not None else None

    def _release_deferred(self) -> None:
        if self._others or self._others_waiting or not self._releasing:
            return
        released, self._releasing = self._releasing, set()
        for store in released:
            store.release_continuation(self)
            self._hand_back(store)

    def _succeed(self, previous: "SessionScheduler", store: BackgroundTaskStore) -> None:
        """Take ``store`` over from ``previous``, which decided for the session until now."""
        self.bind(store)
        if previous.head is not None:
            asked = previous.asking
            self.head, self.head_parked = previous.head, previous.head_parked
            if asked:
                # A question of the channel that let go: this one's to answer now (see :attr:`asking`).
                self._own_head = previous.head
        self.work_arrived()

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

    def turn_finished(self, parked: bool | None = None) -> None:
        """One of the owner's turns is over. ``parked``: whether its run ended waiting on an approval or
        a question; None for a turn that left no run (the head is as it was)."""
        self.busy = False
        if parked is not None and self.head is not None and self.head == self._own_head:
            self.head_parked = parked
        self._notify()
        self._maybe_lend()

    def note_run(self, run_id: str) -> None:
        """One of the owner's runs started: it is the session's head now."""
        self.head = self._own_head = run_id
        self.head_parked = False

    def note_parked(self, run_id: str) -> None:
        """The channel's own run ``run_id`` waits for an answer, and no turn of this process ended on it
        (a session restored parked from its stored trace)."""
        owner = self._deciding()
        if owner is not self and (owner.head != run_id or owner._own_head == run_id):
            # The call that has the session went past that run, or asked the question itself: the
            # session stays the call's (this channel can still answer: a resume is admitted there).
            return
        self._own_head = run_id
        if owner.head == run_id:
            owner.head_parked = True
            if owner is not self:
                # A call took the session before this channel knew of its question: it comes back.
                owner._return_to_asker()
            self._notify()

    def not_waiting(self, run_id: str) -> None:
        """The head ``run_id`` holds no question this channel can find (its stored trace is not parked,
        or holds nothing of it): turns may start on it, and a call may be lent the session."""
        if self.head == run_id and self.head_parked:
            self.head_parked = False
            self._notify()
            self._maybe_lend()

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
        ``interrupt`` on a channel that cuts its own turns for it. While a call
        has the session, the call's turns count: this channel's would only
        queue behind them at admission, started and with its work taken, where
        a question the turn ahead ends on can still hold it (:meth:`_holds`)
        but nothing can put its work back.
        """
        interrupts = self.can_interrupt and urgency == "interrupt"
        owner = self._deciding()
        if owner is not self and (owner._occupied or (owner._others_waiting > 0 and not interrupts)):
            return True
        if self._others:
            return True
        return self._others_waiting > 0 and not interrupts

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
        # The chat this call borrowed the session from still answers the user messages arriving in it.
        for ref in list(self._successors.values()):
            lender = ref()
            if lender is not None and lender.lent:
                lender._notify()
                if lender._on_work is not None:
                    lender._on_work()

    # -- Everyone else's turns ---------------------------------------------------

    async def admit_foreign(self, run_context: Any, store: BackgroundTaskStore) -> None:
        """Let another root turn into the session, queue it, or refuse it.

        An admitted turn continues from the session's head: one started with
        an older run as its parent (queued while the owner's turn ran, or sent
        by a client that had not seen the latest turn yet) is re-chained before
        anything reads its history or session data.
        """
        held = self._holds(run_context)
        if self._occupied or held:
            # The chat that lent the session queues its turns: they are what its user typed. So does a
            # turn held for this channel's question, whatever on_busy says about a turn in flight.
            if self.on_busy == "reject" and not held and not self._lender_of(run_context):
                raise SessionBusyError(
                    "A turn of this session is in flight; wait for its OUTPUT before starting another turn."
                )
            self._others_waiting += 1
            try:
                while (self._occupied or self._holds(run_context)) and not self._taken_over(store):
                    await self._wait_change()
            except BaseException:
                self._others_waiting -= 1
                self._release_deferred()
                self._notify()
                raise
            self._others_waiting -= 1
            if self._taken_over(store):
                # Another channel took the session over while this turn waited: it decides now.
                self._release_deferred()
                await store.admit_root_turn(run_context)
                return
        # Also when the channel let go while this turn waited (its driver stopped, a turn admitted
        # just before preempted it): it keeps the session until the turns queued here have gone.
        self._others.add(run_context.id)
        # Strong: a channel letting go of the session is collectable, and this turn's end is what releases it.
        run_context._foreign_turn_of = self
        if run_context._bg_store is None:
            # Chained onto a run from before the session had its store: it joins the session now.
            adopt_background_store(run_context, store)
        if self.lends and self._borrower is None and store not in self._releasing:
            caller = self._caller(run_context)
            if caller is not None and store.continuation_owner() is self:
                # A call that had no store of its own reaches the chat with this turn: it is lent the
                # session once the chat is idle, like a call that asked for it (see take_over). From
                # here, not while the turn waited: its run holds the store now, so a call that hangs
                # up lets go of it (a turn cut while it waited leaves nothing behind to lend to).
                self._borrower = weakref.ref(caller)
        if self.borrows and self.asking and run_context._resume_values:
            # The answer to the call's own question, brought by its client (a call's is answered over
            # HTTP, not by a turn of its own): one this run asks in turn is the call's question too.
            self._own_head = run_context.id
        head, self.head = self.head, run_context.id
        self.head_parked = False
        if self._on_preempt is not None and store.continuation_owner() is self and store not in self._releasing:
            self._on_preempt()
        marker = run_context._continuation_owner
        new_owner = marker() if marker is not None else None
        successor = self.successor(store)
        if (
            store in self._releasing
            and new_owner is not None
            and new_owner is not self
            and (successor is None or successor is new_owner)
        ):
            # Another channel's own run (a new driver's first, the successor's): the session is
            # theirs from here, and turns still queued here ask them. A call next in line keeps it.
            self._releasing.discard(store)
            self._successors.pop(store, None)
            store.release_continuation(self)
            if successor is not None:
                successor.bind(store)
        self._notify()
        if head is not None and run_context.parent_id != head:
            await run_context._continue_from(head)

    @property
    def _occupied(self) -> bool:
        return self.busy or bool(self._others) or self._holding > 0

    def _taken_over(self, store: BackgroundTaskStore) -> bool:
        owner = store.continuation_owner()
        return owner is not None and owner is not self

    def foreign_finished(self, run_id: str, status: Any = None) -> None:
        """An admitted turn ended with root ``status``; a parked head holds back turns chaining onto it."""
        self._others.discard(run_id)
        if run_id == self.head:
            self.head_parked = is_parked(status)
        self._release_deferred()
        self._notify()
        self._return_to_asker()
        self._maybe_lend()

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
        lent = self.lent
        policies = set() if lent else {self.announce_of(note) for note in store.pending_completions()}
        policies.update(self.announce_of(event) for event in store.pending_events() if self._answers(event, lent))
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
        return (
            store is not None
            and not self.lent
            and any((record.announce or self.completions) != "silent" for record in store.running_tasks())
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
        return take_followup(self.store, urgent=urgency == "interrupt", users=self.user_messages, others=not self.lent)

    def _answers(self, event: SessionEvent, lent: bool) -> bool:
        """Whether ``event`` is this channel's to answer: user messages unless it is a call, the rest unless lent."""
        return self.user_messages if event.source == "user" else not lent

    # -- Internals ------------------------------------------------------------------

    def _notify(self) -> None:
        changed, self._changed = self._changed, None
        if changed is not None and not changed.done():
            changed.set_result(None)
        # The chat this call borrowed the session from waits on what the call decides (its head, a
        # question answered): its own scheduler changes with this one.
        for ref in self._successors.values():
            lender = ref()
            if lender is not None and lender is not self and lender._lent_to is not None and lender.lent:
                lender._notify()

    async def _wait_change(self) -> None:
        if self._changed is None:
            self._changed = asyncio.get_running_loop().create_future()
        # Shielded: one waiter being cancelled must not cancel the future the others share.
        await asyncio.shield(self._changed)
