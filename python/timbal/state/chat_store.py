"""Durable chat sessions: what a :class:`~timbal.core.chat_session.ChatSession` needs to come back after a restart.

The conversation itself is in the traces. A snapshot holds the session layer on top of them: the head run,
the messages and wakes no turn has taken yet, a turn that was running, whether the session is parked. One
mutable row per session, written ahead of every change (a message is durable before its receipt goes out).

A lease (owner + expiry, renewed while the session is open) keeps two processes from serving one session,
and every save is a compare-and-swap on ``version`` that also checks the lease: an owner whose lease another
process took (it expired, or renewing it failed) cannot write, so the new owner is never the one to give up.

A store can also hold an **inbox** per session (``supports_inbox``): work another process takes for a session it
does not serve (a message, a wake, a message from another agent), for the lease owner to apply. See
:class:`ChatSessionStore`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog
from uuid_extensions import uuid7

logger = structlog.get_logger("timbal.state.chat_store")

__all__ = [  # ``PROCESS_OWNER`` is still readable (module __getattr__), for code written before process_owner()
    "process_owner",
    "ChatSessionStore",
    "INBOX_KEY_KEPT_S",
    "INBOX_KINDS",
    "InboxFull",
    "InMemoryChatSessionStore",
    "MAX_INBOX_ITEMS",
    "MAX_INBOX_KEY_CHARS",
    "SessionLeased",
    "SessionNotStored",
    "SessionSnapshot",
    "SnapshotConflict",
    "SqliteChatSessionStore",
    "storable",
]

_OWNER: tuple[int, str] | None = None


def process_owner() -> str:
    """This process, as a lease owner: ``host:pid:uuid``, unique across restarts of the same pid.

    Resolved when it is used, per pid: a server that imports this module and then forks its workers must
    not hand every child the parent's identity (each would renew, and take, the others' leases).
    """
    global _OWNER
    pid = os.getpid()
    if _OWNER is None or _OWNER[0] != pid:
        _OWNER = (pid, f"{socket.gethostname()}:{pid}:{uuid7(as_type='hex')}")
    return _OWNER[1]


def __getattr__(name: str) -> Any:
    # ``chat_store.PROCESS_OWNER`` still reads as this process's owner, at the time it is read. Importing the
    # name copies the value of that moment: after a fork it is the parent's. Call process_owner() instead.
    if name == "PROCESS_OWNER":
        return process_owner()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _holder_alive(owner: str | None) -> bool:
    """False for a lease held by a process of this host that no longer exists (``kill -9``): its lease would
    otherwise keep the session from the process restarted after it until it expires."""
    if not owner:
        return False
    host, _, rest = owner.partition(":")
    pid = rest.partition(":")[0]
    if host != socket.gethostname() or not pid.isdigit():
        return True  # another host's: only expiry tells
    return _pid_alive(int(pid))


def _pid_alive(pid: int) -> bool:
    """Whether a process of this host exists. Errs on "alive": a wrong "dead" lets two processes serve a session.

    Never ``os.kill(pid, 0)`` on Windows: there any signal but the console ones *terminates* the process.
    """
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # PermissionError: exists, another user's
    return True


def _windows_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    process_query_limited_information, still_active = 0x1000, 259
    error_invalid_parameter = 87  # what OpenProcess answers for a pid no process has
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return ctypes.get_last_error() != error_invalid_parameter  # access denied: exists
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        # An exited process whose handle someone still holds keeps its pid: its exit code tells.
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


class SessionLeased(RuntimeError):
    """Another live process holds the session."""


class SessionNotStored(RuntimeError):
    """The session's latest state could not be written (the store failed): a change made since the last write
    would not survive a restart."""


class SnapshotConflict(RuntimeError):
    """The stored snapshot is not the version this owner last wrote: another owner took the session."""


class InboxFull(RuntimeError):
    """The session's inbox already holds as many waiting items as its store lets wait (``max_inbox_items``):
    nothing was queued. Its owner applies them within moments while it runs; a full inbox means it does not."""


INBOX_KINDS = ("message", "wake", "agent_message")
"""What another process can queue for a session's owner (:meth:`ChatSessionStore.post_inbox`), as the platform's
session store names them: a user message ``{text, mode}``, a wake ``{text, key, tool_policy, silent_ok, trigger}``,
a message from another agent ``{from, text, message_id, reply_to, priority, hops}``."""

MAX_INBOX_ITEMS = 256
"""How many items may wait (posted, not acked) in one session's inbox: a post past it raises :class:`InboxFull`
(the platform's ``MAX_INBOX_ITEMS``). The one constant for both halves of the inbox's dedupe: a session that keeps
the ids it applied in its snapshot (``SessionSnapshot.inbox_seen``, for a store that acks apart from its saves)
keeps at least this many, since every item pulled again after a crash is one of those still waiting."""

INBOX_KEY_KEPT_S = 86_400.0
"""How long a post's ``idempotency_key`` keeps naming its item after the owner acked it (a day, as on the
platform): the sender may not have had the answer, and a repeat within that time queues nothing."""

MAX_INBOX_KEY_CHARS = 255
"""The longest ``idempotency_key`` (the platform's ``MAX_ID_CHARS``)."""


def _inbox_body(kind: str, body: Any, idempotency_key: str | None) -> str:
    """A post's checks, before anything is stored: the body as it is stored (JSON text)."""
    if kind not in INBOX_KINDS:
        raise ValueError(f"kind must be one of {list(INBOX_KINDS)}, got {kind!r}")
    if not isinstance(body, dict):
        raise TypeError("an inbox item's body must be a dict")
    if idempotency_key is not None and (
        not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= MAX_INBOX_KEY_CHARS
    ):
        raise ValueError(f"idempotency_key must be 1 to {MAX_INBOX_KEY_CHARS} characters")
    try:
        return _encode(body)
    except (TypeError, ValueError, RecursionError) as e:
        raise TypeError(f"an inbox item's body must be JSON a snapshot can hold: {e}") from None


def _inbox_item(item_id: str, kind: str, raw: str, posted_at: float) -> dict[str, Any]:
    """An item as a pull hands it out: its own copy of the body. A body that cannot be read (a row written by
    something else) comes out as None, which the owner refuses and acks, rather than failing every pull."""
    try:
        body = json.loads(raw)
    except (TypeError, ValueError, RecursionError):
        body = None
    return {"id": item_id, "kind": kind, "body": body, "posted_at": posted_at}


def _new_inbox_id() -> str:
    return f"in_{uuid7(as_type='hex')}"


def _wake_pull(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


class _InboxWaiters:
    """The pulls waiting for a session's next item, each on the loop it runs on: woken from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Future]]] = {}

    @contextlib.contextmanager
    def watching(self, session_id: str):
        """A future set once something is posted for ``session_id``. Registered before the pull's first look,
        so a post landing between that look and the wait still ends the wait."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        entry = (loop, future)
        with self._lock:
            self._waiting.setdefault(session_id, set()).add(entry)
        try:
            yield future
        finally:
            with self._lock:
                entries = self._waiting.get(session_id)
                if entries is not None:
                    entries.discard(entry)
                    if not entries:
                        del self._waiting[session_id]
            if not future.done():
                future.cancel()

    def sessions(self) -> list[str]:
        with self._lock:
            return list(self._waiting)

    def wake(self, session_id: str) -> None:
        with self._lock:
            entries = list(self._waiting.get(session_id, ()))
        for loop, future in entries:
            with contextlib.suppress(RuntimeError):  # its loop is closed: nobody waits there any more
                loop.call_soon_threadsafe(_wake_pull, future)


def _encode(value: Any) -> str:
    """The one JSON encoding a snapshot is written with. No ``default=`` (a value JSON cannot carry is not
    written as its ``str()``) and no NaN or Infinity (Python would write them; no other JSON reader takes them)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def storable(value: Any) -> bool:
    """Whether a snapshot can hold ``value`` as it is.

    The test a session applies to what its callers hand it (answers, turn arguments, a prompt) before it keeps
    it. It is the encoder :meth:`SessionSnapshot.to_json` writes with, so the two cannot disagree: a value this
    lets through never fails a save, and with it every save after it.
    """
    try:
        _encode(value)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


@dataclass
class SessionSnapshot:
    session_id: str
    agent: str
    """The agent's name: a session is only restored for the agent it belongs to."""
    config: dict[str, Any] = field(default_factory=dict)
    head: str | None = None
    turns: int = 0
    last_status: dict[str, Any] | None = None
    parked: bool = False
    last_prompt: Any = None
    """The dumped prompt of the last turn: a resume of a parked turn passes it again (the agent drops it)."""
    last_options: dict[str, Any] = field(default_factory=dict)
    in_flight: dict[str, Any] | None = None
    """The turn running at save time: ``run_id``, ``parent_id``, ``trigger``, ``prompt`` (dumped), ``options``,
    ``events`` (the messages it took). Cleared when it finishes; still set after a crash."""
    pending: list[dict[str, Any]] = field(default_factory=list)
    """Events no turn has taken yet (user messages, app/client events): ids and timestamps kept."""
    wakes: list[dict[str, Any]] = field(default_factory=list)
    resume: dict[str, Any] | None = None
    """Answers to a parked turn sent but not yet taken by a turn."""
    log_cursor: int = 0
    closed: bool = False
    version: int = 0
    root_run_id: str | None = None
    """The thread's root run, once known (:attr:`ChatSession.root_run_id`); absent from older snapshots. Last, so
    a snapshot built by position means what it did before the field."""
    answer_format: int = 1
    """How ``resume`` (and ``in_flight["resume"]``, which says so itself as ``in_flight["answer_format"]``) hold
    their answers. 1, what a snapshot without the field holds: as given, a ``Cancel`` as ``{"$cancel": reason}``.
    2: the same, and an answer that has a ``$cancel`` or ``$answer`` key of its own as ``{"$answer": answer}``, so
    that it is never read back as a ``Cancel``. Read by the rules of the format that wrote it. Last, as above."""
    inbox_seen: list[str] = field(default_factory=list)
    """Ids of the inbox items the session applied (the last :data:`MAX_INBOX_ITEMS` or more), with a store that
    acks them apart from its saves: an item pulled again after a crash between the save and the ack is skipped.
    Empty with a store that acks in the save itself (memory, SQLite). Last, as above."""

    def to_json(self) -> str:
        """Raises for a value JSON cannot carry (TypeError; ValueError for NaN and Infinity, which Python would
        write and no other JSON reader takes). Not written as its ``str()``: read back, that is a different
        value (an approval that is no longer one), and nothing would say so. The session checks what callers
        hand it before it takes it (answers, turn arguments), so a save does not get here with one; if one
        does, the save fails, is reported and retried, and what it held is never acknowledged as stored."""
        return _encode(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> SessionSnapshot:
        data = json.loads(raw)
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})

    def has_work(self) -> bool:
        """Something would run without a client asking: restore it at startup, not on first use."""
        return not self.closed and bool(self.in_flight or self.pending or self.wakes or self.resume)


class ChatSessionStore(ABC):
    """Where snapshots live. Implementations must make ``save`` and ``acquire_lease`` atomic.

    A store that ``supports_inbox`` also carries work other processes take for a session they do not serve (a
    server worker that is asked for a session another worker holds): :meth:`post_inbox` queues it, the lease
    owner long-polls :meth:`pull_inbox`, applies each item, stores the session, and the item is acked. At least
    once: an item stays until acked, and is pulled again after a crash. Two dedupes, for two failures:

    - a post repeated with its ``idempotency_key`` is the same item, for :data:`INBOX_KEY_KEPT_S` after its ack
      too (a sender that retries because it never got the answer);
    - the owner applies an item once, also when it is pulled again because the process died after storing the
      session and before the ack. A store that ``acks_with_save`` closes that window: the save that stores an
      item's effect acks it in the same write (``save(..., inbox_acked=)``). Otherwise the session acks with
      :meth:`ack_inbox` after the save, and its snapshot names what it applied (``inbox_seen``).

    At most ``max_inbox_items`` items wait per session (:class:`InboxFull` past it). Every write the owner makes
    names it (``owner``), and a pull or an ack by a process that does not hold the lease is refused
    (:class:`SessionLeased`). A session that is closed or gone takes no items (``LookupError``), and the write
    that closes or deletes it discards what still waited.
    """

    lease_ttl_s: float = 30.0
    supports_inbox: bool = False
    acks_with_save: bool = False
    max_inbox_items: int = MAX_INBOX_ITEMS

    @abstractmethod
    async def load(self, session_id: str) -> SessionSnapshot | None: ...

    @abstractmethod
    async def save(
        self,
        snapshot: SessionSnapshot,
        *,
        expect_version: int | None,
        owner: str | None = None,
        inbox_acked: Collection[str] = (),
    ) -> int:
        """Write ``snapshot`` if the stored version is ``expect_version`` (None: the row must not exist) and, with
        ``owner``, no other owner holds the lease (expired or not: one that took it may not have written yet).
        A new row is created with ``owner``'s lease, in the same write: a session never exists unowned.
        Returns the new version; raises :class:`SnapshotConflict` otherwise.

        ``inbox_acked`` (a store that ``acks_with_save``; needs ``owner``): inbox items the snapshot holds the
        effect of, acked in the same write, so neither lands without the other. A session passes it only to a
        store that says it ``acks_with_save`` and whose ``save`` takes it: other stores never see it."""

    @abstractmethod
    async def delete(self, session_id: str, *, expect_version: int | None = None, owner: str | None = None) -> None:
        """Remove the session. With ``expect_version`` and/or ``owner``: only while it is still at that version
        and no other owner holds its lease. Raises :class:`SnapshotConflict` otherwise (another process took
        the session and it is theirs now); a row already gone is not an error."""

    @abstractmethod
    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        """Open sessions (``with_work``: only those :meth:`SessionSnapshot.has_work` would restore eagerly)."""

    @abstractmethod
    async def acquire_lease(self, session_id: str, owner: str) -> None:
        """Take or renew the lease. Raises :class:`SessionLeased` while another owner's has not expired."""

    @abstractmethod
    async def release_lease(self, session_id: str, owner: str) -> None: ...

    async def post_inbox(
        self,
        session_id: str,
        kind: str,
        body: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        agent: str | None = None,
    ) -> str:
        """Queue ``kind`` (:data:`INBOX_KINDS`) with ``body`` for the session's owner; returns the item's id.

        With ``idempotency_key``, a repeat is the same item (its id comes back, nothing is queued), while it
        waits and for :data:`INBOX_KEY_KEPT_S` after its ack. ``agent``: only a session of that agent takes it
        (several agents can share a store, and a session id alone does not say whose it is). ``LookupError``:
        no such session, a closed one, or another agent's. :class:`InboxFull`: ``max_inbox_items`` wait already.
        ``ValueError`` / ``TypeError``: a kind, key or body the store cannot take."""
        raise NotImplementedError(f"{type(self).__name__} has no inbox")

    async def find_inbox_key(self, session_id: str, key: str) -> tuple[str, bool] | None:  # noqa: ARG002
        """The item ``key`` names in the session's inbox, while it waits and for :data:`INBOX_KEY_KEPT_S` after its
        ack (as a post with the key would find it): ``(id, acked)``, or None. Needs no lease: a worker that takes a
        request directly asks whether it was queued before. A store without it answers None."""
        return None

    async def inbox_waiting(self, session_id: str) -> int:
        """How many items wait in the session's inbox (no lease needed: a worker that queued for a session another
        serves looks, to restore it if that one lets it go with items waiting). The base class answers from
        :meth:`list_ids`, which counts a session's other work too: 1 when the session has any."""
        return int(session_id in await self.list_ids(with_work=True))

    async def pull_inbox(self, session_id: str, owner: str, *, wait_s: float = 0.0) -> list[dict[str, Any]]:
        """The items not acked yet, oldest first (``{id, kind, body, posted_at}``), waiting up to ``wait_s`` for a
        first one. Only for the holder of a lease that has not run out: :class:`SessionLeased` otherwise.
        ``LookupError``: no stored session."""
        raise NotImplementedError(f"{type(self).__name__} has no inbox")

    async def ack_inbox(self, session_id: str, owner: str, ids: Collection[str], *, forget: bool = False) -> None:
        """Remove items the owner applied and stored (one with a key stays as its key alone, for
        :data:`INBOX_KEY_KEPT_S`). ``forget``: their keys go too, for items the owner refused (a retry of one is
        screened again, not answered as the item that was refused). :class:`SessionLeased` when another owner holds
        the lease, and then nothing is removed. ``LookupError``: no stored session."""
        raise NotImplementedError(f"{type(self).__name__} has no inbox")


class _MemoryItem:
    __slots__ = ("id", "kind", "raw", "key", "posted_at", "acked_at")

    def __init__(self, item_id: str, kind: str, raw: str, key: str | None, posted_at: float) -> None:
        self.id, self.kind, self.raw, self.key = item_id, kind, raw, key
        self.posted_at, self.acked_at = posted_at, None


class InMemoryChatSessionStore(ChatSessionStore):
    """Process-local (the default): what sessions did before they were durable, through the same interface."""

    supports_inbox = True
    acks_with_save = True

    def __init__(self) -> None:
        self._rows: dict[str, tuple[SessionSnapshot, str | None, float]] = {}
        # Per session, oldest first: the waiting items and the keys kept after an ack.
        self._inbox: dict[str, list[_MemoryItem]] = {}
        self._inbox_waiters = _InboxWaiters()

    async def load(self, session_id: str) -> SessionSnapshot | None:
        row = self._rows.get(session_id)
        return SessionSnapshot.from_json(row[0].to_json()) if row is not None else None

    async def save(
        self,
        snapshot: SessionSnapshot,
        *,
        expect_version: int | None,
        owner: str | None = None,
        inbox_acked: Collection[str] = (),
    ) -> int:
        if inbox_acked and owner is None:
            raise ValueError("A save that acks inbox items names its owner.")
        row = self._rows.get(snapshot.session_id)
        current = row[0].version if row is not None else None
        if current != expect_version:
            raise SnapshotConflict(
                f"session {snapshot.session_id}: stored version {current}, expected {expect_version}"
            )
        if owner is not None and row is not None and row[1] not in (None, owner):
            raise SnapshotConflict(f"session {snapshot.session_id}: leased by {row[1]}")
        stored = SessionSnapshot.from_json(snapshot.to_json())
        stored.version = (current or 0) + 1
        if row is not None:
            holder, until = row[1], row[2]
        else:
            holder, until = (owner, time.time() + self.lease_ttl_s) if owner is not None else (None, 0.0)
        self._rows[snapshot.session_id] = (stored, holder, until)
        # In the same step as the snapshot (nothing awaits in between): neither is ever stored without the other.
        if stored.closed:
            self._inbox.pop(snapshot.session_id, None)  # nothing restores a closed session: what waited is dropped
        elif inbox_acked:
            self._ack(snapshot.session_id, inbox_acked)
        return stored.version

    async def delete(self, session_id: str, *, expect_version: int | None = None, owner: str | None = None) -> None:
        row = self._rows.get(session_id)
        if row is None:
            return
        if expect_version is not None and row[0].version != expect_version:
            raise SnapshotConflict(f"session {session_id}: stored version {row[0].version}, expected {expect_version}")
        if owner is not None and row[1] not in (None, owner):
            raise SnapshotConflict(f"session {session_id}: leased by {row[1]}")
        del self._rows[session_id]
        self._inbox.pop(session_id, None)

    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        return [
            sid
            for sid, (snap, _, _) in self._rows.items()
            if not snap.closed and (not with_work or snap.has_work() or self._waiting(sid))
        ]

    def _waiting(self, session_id: str) -> list[_MemoryItem]:
        return [item for item in self._inbox.get(session_id, ()) if item.acked_at is None]

    def _ack(self, session_id: str, ids: Collection[str], *, forget: bool = False) -> None:
        done, now = set(ids), time.time()
        kept = []
        for item in self._inbox.get(session_id, ()):
            if forget and item.id in done:
                continue  # the item and its key
            if item.acked_at is None and item.id in done:
                if item.key is None:
                    continue
                item.acked_at, item.raw = now, "{}"  # its key alone stays, so a repeated post still finds it
            elif item.acked_at is not None and item.acked_at < now - INBOX_KEY_KEPT_S:
                continue
            kept.append(item)
        if session_id in self._inbox:
            self._inbox[session_id] = kept

    def _holder(self, session_id: str, owner: str, *, running: bool) -> None:
        row = self._rows.get(session_id)
        if row is None:
            raise LookupError(f"No stored chat session {session_id!r}.")
        if row[1] != owner or (running and row[2] < time.time()):
            raise SessionLeased(f"session {session_id} is not leased to {owner}")

    async def post_inbox(
        self,
        session_id: str,
        kind: str,
        body: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        agent: str | None = None,
    ) -> str:
        raw = _inbox_body(kind, body, idempotency_key)
        row = self._rows.get(session_id)
        if row is None or row[0].closed or (agent is not None and row[0].agent != agent):
            raise LookupError(f"No stored chat session {session_id!r}" + (f" of agent {agent!r}." if agent else "."))
        items = self._inbox.setdefault(session_id, [])
        now = time.time()
        if idempotency_key is not None:
            # A key kept past its time is forgotten here, by the post that names it: the post is a new item.
            items[:] = [
                item
                for item in items
                if not (
                    item.key == idempotency_key and item.acked_at is not None and item.acked_at < now - INBOX_KEY_KEPT_S
                )
            ]
            for item in items:
                if item.key == idempotency_key:
                    return item.id
        if sum(1 for item in items if item.acked_at is None) >= self.max_inbox_items:
            raise InboxFull(f"Chat session {session_id!r} already has {self.max_inbox_items} inbox items waiting.")
        item = _MemoryItem(_new_inbox_id(), kind, raw, idempotency_key, now)
        items.append(item)
        self._inbox_waiters.wake(session_id)
        return item.id

    async def pull_inbox(self, session_id: str, owner: str, *, wait_s: float = 0.0) -> list[dict[str, Any]]:
        with self._inbox_waiters.watching(session_id) as posted:
            items = self._pull(session_id, owner)
            if items or wait_s <= 0:
                return items
            await asyncio.wait({posted}, timeout=wait_s)
        return self._pull(session_id, owner)

    def _keyed(self, session_id: str, key: str, now: float) -> _MemoryItem | None:
        for item in self._inbox.get(session_id, ()):
            if item.key == key and (item.acked_at is None or item.acked_at >= now - INBOX_KEY_KEPT_S):
                return item
        return None

    async def find_inbox_key(self, session_id: str, key: str) -> tuple[str, bool] | None:
        item = self._keyed(session_id, key, time.time())
        return (item.id, item.acked_at is not None) if item is not None else None

    async def inbox_waiting(self, session_id: str) -> int:
        return len(self._waiting(session_id))

    def _pull(self, session_id: str, owner: str) -> list[dict[str, Any]]:
        self._holder(session_id, owner, running=True)
        return [_inbox_item(item.id, item.kind, item.raw, item.posted_at) for item in self._waiting(session_id)]

    async def ack_inbox(self, session_id: str, owner: str, ids: Collection[str], *, forget: bool = False) -> None:
        self._holder(session_id, owner, running=False)
        self._ack(session_id, ids, forget=forget)

    async def acquire_lease(self, session_id: str, owner: str) -> None:
        row = self._rows.get(session_id)
        if row is None:
            return
        snap, holder, until = row
        if holder not in (None, owner) and until > time.time() and _holder_alive(holder):
            raise SessionLeased(f"session {session_id} is held by {holder}")
        self._rows[session_id] = (snap, owner, time.time() + self.lease_ttl_s)

    async def release_lease(self, session_id: str, owner: str) -> None:
        row = self._rows.get(session_id)
        if row is not None and row[1] == owner:
            self._rows[session_id] = (row[0], None, 0.0)


class _StoreThread:
    """The one thread a SQLite file's session-store calls run on.

    Not the event loop's shared pool (``asyncio.to_thread``). A store that hangs (a stalled mount) keeps
    whichever thread called it, and callers that stop waiting leave those threads behind: on the shared
    pool they add up until nothing else in the process gets a thread, and they hold the process open at
    exit. Here at most one thread is ever stuck per file, it is a daemon, and a call whose caller stopped
    waiting before it began is not run at all.

    Calls run in the order they were made. The thread ends after ``IDLE_S`` with nothing to do and is
    started again by the next call.
    """

    IDLE_S = 30.0

    def __init__(self, name: str) -> None:
        self._name = name
        self._jobs: deque[tuple[Any, asyncio.AbstractEventLoop, asyncio.Future, threading.Event]] = deque()
        self._wake = threading.Condition()
        self._running = False

    async def call(self, fn: Any) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        abandoned = threading.Event()
        # Set on the loop when the caller is cancelled (a timeout, a close), read by the thread before it starts.
        future.add_done_callback(lambda done: abandoned.set() if done.cancelled() else None)
        with self._wake:
            self._jobs.append((fn, loop, future, abandoned))
            if self._running:
                self._wake.notify()
            else:
                self._running = True
                threading.Thread(target=self._work, name=self._name, daemon=True).start()
        return await future

    def _work(self) -> None:
        while True:
            with self._wake:
                while not self._jobs:
                    if not self._wake.wait(self.IDLE_S) and not self._jobs:
                        self._running = False
                        return
                fn, loop, future, abandoned = self._jobs.popleft()
            if abandoned.is_set():
                continue
            try:
                outcome = (True, fn())
            except BaseException as error:
                outcome = (False, error)
            try:
                loop.call_soon_threadsafe(_deliver, future, outcome)
            except RuntimeError:
                pass  # the loop is closed: nobody is waiting for this


def _deliver(future: asyncio.Future, outcome: tuple[bool, Any]) -> None:
    if future.done():
        return  # its caller stopped waiting while it ran
    ok, value = outcome
    if ok:
        future.set_result(value)
    else:
        future.set_exception(value)


_STORE_THREADS: dict[str, _StoreThread] = {}
_STORE_THREADS_LOCK = threading.Lock()


def _store_thread(path: Path) -> _StoreThread:
    key = str(path.resolve())
    with _STORE_THREADS_LOCK:
        thread = _STORE_THREADS.get(key)
        if thread is None:
            thread = _STORE_THREADS[key] = _StoreThread(f"timbal-chat-store:{path.name}")
        return thread


class _InboxWatch:
    """The pulls waiting on one SQLite file's inbox, answered by one query per ``poll_s`` for all of them.

    Other processes post to the file and nothing tells this one, so a waiting pull has to look again. One
    thread per file does that for every session waited on in this process, while any pull waits, and wakes the
    pulls whose session has an item. A pull per session asking on its own would put a query per open session
    and per poll on the file's one store thread (:class:`_StoreThread`), ahead of the saves behind them. A post
    made in this process wakes its session's pulls at once.
    """

    def __init__(self, name: str) -> None:
        self.waiters = _InboxWaiters()
        self._name = name
        self._lock = threading.Lock()
        self._running = False

    @contextlib.contextmanager
    def watching(self, session_id: str, ready: Any, poll_s: float):
        """``ready(session_ids)`` answers which of those sessions have items waiting (on the watch's thread)."""
        with self.waiters.watching(session_id) as posted:
            with self._lock:
                start, self._running = not self._running, True
            if start:
                threading.Thread(target=self._poll, args=(ready, poll_s), name=self._name, daemon=True).start()
            yield posted

    def _poll(self, ready: Any, poll_s: float) -> None:
        while True:
            time.sleep(poll_s)
            with self._lock:
                watched = self.waiters.sessions()
                if not watched:
                    self._running = False  # under the lock a pull that starts now checks: it starts a new thread
                    return
            try:
                found = ready(watched)
            except Exception as e:  # the file is away: the pulls end on their own deadlines
                logger.warning("chat_session_inbox_poll_failed", error=str(e))
                continue
            for session_id in found:
                self.waiters.wake(session_id)


_INBOX_WATCHES: dict[str, _InboxWatch] = {}


def _inbox_watch(path: Path) -> _InboxWatch:
    key = str(path.resolve())
    with _STORE_THREADS_LOCK:
        watch = _INBOX_WATCHES.get(key)
        if watch is None:
            watch = _INBOX_WATCHES[key] = _InboxWatch(f"timbal-chat-inbox:{path.name}")
        return watch


def _forget_store_threads() -> None:
    """In a forked child: the parent's threads did not come along, and a lock one of them held stays held."""
    global _STORE_THREADS_LOCK
    _STORE_THREADS.clear()
    _INBOX_WATCHES.clear()
    _STORE_THREADS_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_store_threads)


# The sessions with work waiting, through the index over them (`idx_chat_sessions_work`): idle sessions stay
# in the table until deleted, and reading every one of them is what startup would otherwise spend its time on.
# The snapshot comes along only for a row from before the mark (NULL), which says nothing yet.
_WITH_WORK_SQL = (
    "SELECT rowid, session_id, CASE WHEN has_work IS NULL THEN snapshot END FROM chat_sessions WHERE has_work IS NOT 0"
)
# The sessions with inbox items waiting, found from the inbox's index of waiting items (a few rows at most, since
# their owners take them within moments). A row these come with is open: the write that closes a session drops
# its inbox, and a closed one takes no posts. The snapshot comes along for a row from before the mark, as above.
_WITH_INBOX_SQL = (
    "SELECT rowid, session_id, CASE WHEN has_work IS NULL THEN snapshot END FROM chat_sessions "
    "WHERE session_id IN (SELECT session_id FROM chat_inbox WHERE acked_at IS NULL)"
)
_WAITING_SESSIONS_SQL = "SELECT DISTINCT session_id FROM chat_inbox WHERE acked_at IS NULL"
# SQLite's smallest limit on the parameters of one statement is 999.
_IDS_PER_STATEMENT = 500


def _settled(conn: Any) -> Any:
    """The shared connection with no transaction left open, for a call that holds its lock. Every call on it ends
    its own (a trace save commits or rolls back inside the lock), so one still open was left by a statement that
    failed before its commit (``database is locked``): kept, it would hold a read of the file as it was then, and
    the inbox would not see what other processes posted since; a write would join it."""
    if conn.in_transaction:
        conn.rollback()
    return conn


def _transaction(conn: Any, fn: Any) -> Any:
    """``fn()`` in one write transaction: a check and the write that follows it cannot be split by another
    process's write, and a failure leaves nothing (the connection is shared: an open transaction would take the
    next caller's statements into it)."""
    _settled(conn).execute("BEGIN IMMEDIATE")
    try:
        result = fn()
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    return result


def _ack_rows(conn: Any, session_id: str, ids: Collection[str], now: float, *, forget: bool = False) -> None:
    """Ack inbox items (inside the caller's transaction): an item without a key goes, one with a key stays as its
    key alone (body emptied) for :data:`INBOX_KEY_KEPT_S`, unless ``forget`` (refused: key and all go); the
    session's keys kept longer than that go too."""
    ids = list(ids)
    for at in range(0, len(ids), _IDS_PER_STATEMENT):
        chunk = ids[at : at + _IDS_PER_STATEMENT]
        marks = ",".join("?" * len(chunk))
        if forget:
            conn.execute(f"DELETE FROM chat_inbox WHERE session_id = ? AND id IN ({marks})", (session_id, *chunk))
            continue
        conn.execute(
            f"DELETE FROM chat_inbox WHERE session_id = ? AND acked_at IS NULL AND idempotency_key IS NULL "
            f"AND id IN ({marks})",
            (session_id, *chunk),
        )
        conn.execute(
            f"UPDATE chat_inbox SET acked_at = ?, body = '{{}}' WHERE session_id = ? AND acked_at IS NULL "
            f"AND idempotency_key IS NOT NULL AND id IN ({marks})",
            (now, session_id, *chunk),
        )
    conn.execute("DELETE FROM chat_inbox WHERE session_id = ? AND acked_at < ?", (session_id, now - INBOX_KEY_KEPT_S))


def _closed(raw: str) -> bool:
    try:
        return json.loads(raw).get("closed") is True
    except (TypeError, ValueError, AttributeError, RecursionError):
        return False  # unreadable: not known to be closed (list_ids skips it; a restore would fail on it)


class SqliteChatSessionStore(ChatSessionStore):
    """Snapshots in the ``chat_sessions`` table of a SQLite file, next to
    :class:`~timbal.state.tracing.providers.SqliteTracingProvider`'s runs (same connection when it is the
    same file). The inbox is the ``chat_inbox`` table of the same file, acked in the save that stores what an
    item did (one transaction)."""

    supports_inbox = True
    acks_with_save = True
    inbox_poll_s: float = 0.25
    """How often waiting pulls look for items other processes posted (:class:`_InboxWatch`: one query per file)."""

    def __init__(self, path: Path | str, *, synchronous: str = "NORMAL") -> None:
        from .tracing.providers.sqlite import refuse_in_memory

        refuse_in_memory(path)  # a database in memory would be a new, empty one on every call
        self.path = Path(path)
        self.synchronous = synchronous

    def _db(self):
        from .tracing.providers.sqlite import database

        return database(self.path, self.synchronous)

    async def _run(self, fn):
        def locked():
            db = self._db()
            with db.lock:
                return fn(db.conn)

        # On the file's own thread (_StoreThread), never the loop's shared pool.
        return await _store_thread(self.path).call(locked)

    async def load(self, session_id: str) -> SessionSnapshot | None:
        def query(conn):
            return conn.execute(
                "SELECT snapshot, version FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

        row = await self._run(query)
        if row is None:
            return None
        snapshot = SessionSnapshot.from_json(row[0])
        snapshot.version = row[1]
        return snapshot

    async def save(
        self,
        snapshot: SessionSnapshot,
        *,
        expect_version: int | None,
        owner: str | None = None,
        inbox_acked: Collection[str] = (),
    ) -> int:
        if inbox_acked and owner is None:
            raise ValueError("A save that acks inbox items names its owner.")
        new_version = (expect_version or 0) + 1
        body = SessionSnapshot.from_json(snapshot.to_json())
        body.version = new_version
        raw, now = body.to_json(), int(time.time() * 1000)
        lease_until = time.time() + self.lease_ttl_s if owner is not None else None
        work = int(body.has_work())  # the row's mark for list_ids(with_work=True): never read from the snapshot
        acked = list(inbox_acked)

        def write(conn):
            try:
                return written(_settled(conn))
            except BaseException:
                conn.rollback()  # nothing of it stays open on the shared connection
                raise

        def written(conn):
            if expect_version is None:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO chat_sessions "
                    "(session_id, agent, snapshot, version, updated_at, lease_owner, lease_until, has_work) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (snapshot.session_id, snapshot.agent, raw, new_version, now, owner, lease_until, work),
                )
            elif owner is None:
                cursor = conn.execute(
                    "UPDATE chat_sessions SET snapshot = ?, version = ?, updated_at = ?, has_work = ? "
                    "WHERE session_id = ? AND version = ?",
                    (raw, new_version, now, work, snapshot.session_id, expect_version),
                )
            else:
                cursor = conn.execute(
                    "UPDATE chat_sessions SET snapshot = ?, version = ?, updated_at = ?, has_work = ? "
                    "WHERE session_id = ? AND version = ? AND (lease_owner IS NULL OR lease_owner = ?)",
                    (raw, new_version, now, work, snapshot.session_id, expect_version, owner),
                )
            if cursor.rowcount == 1 and expect_version is not None and (body.closed or acked):
                # In the transaction the statement above opened: the inbox changes with the snapshot or not at all.
                if body.closed:
                    # Nothing restores a closed session: what still waited cannot be delivered.
                    conn.execute("DELETE FROM chat_inbox WHERE session_id = ?", (snapshot.session_id,))
                else:
                    _ack_rows(conn, snapshot.session_id, acked, time.time())
            conn.commit()
            return cursor.rowcount

        if await self._run(write) != 1:
            raise SnapshotConflict(
                f"session {snapshot.session_id}: not at version {expect_version}"
                + (f", or leased by another owner than {owner}" if owner is not None else "")
            )
        return new_version

    async def delete(self, session_id: str, *, expect_version: int | None = None, owner: str | None = None) -> None:
        clauses, params = ["session_id = ?"], [session_id]
        if expect_version is not None:
            clauses.append("version = ?")
            params.append(expect_version)
        if owner is not None:
            clauses.append("(lease_owner IS NULL OR lease_owner = ?)")
            params.append(owner)

        def write(conn):
            # One statement: the check and the delete cannot be split by another process's write.
            cursor = conn.execute(f"DELETE FROM chat_sessions WHERE {' AND '.join(clauses)}", params)
            gone = cursor.rowcount == 1 or (
                conn.execute("SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone() is None
            )
            if gone:
                try:
                    # In the delete's transaction: no item outlives its session.
                    conn.execute("DELETE FROM chat_inbox WHERE session_id = ?", (session_id,))
                except BaseException:
                    conn.rollback()
                    raise
            conn.commit()
            return gone

        if not await self._run(write):
            raise SnapshotConflict(
                f"session {session_id}: not at version {expect_version}, or leased by another owner than {owner}"
            )

    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        def query(conn):
            if with_work:
                rows = {row[0]: (row[0], row[1], row[2], False) for row in conn.execute(_WITH_WORK_SQL).fetchall()}
                for row in conn.execute(_WITH_INBOX_SQL).fetchall():
                    rows[row[0]] = (row[0], row[1], row[2], True)  # items wait for it: it has work
                # Oldest first, as the table itself is read.
                return [(row[1], row[2], row[3]) for row in sorted(rows.values())]
            return [(row[0], row[1], False) for row in conn.execute("SELECT session_id, snapshot FROM chat_sessions")]

        ids = []
        for session_id, raw, queued in await self._run(query):
            if raw is None:
                ids.append(session_id)  # marked as having work by the save that wrote it, or items wait for it
                continue
            try:
                snapshot = SessionSnapshot.from_json(raw)
            except Exception as e:
                # One row that cannot be read (cut short, written by something else) must not hide every
                # session next to it, at startup and on each later look.
                logger.error("chat_session_snapshot_unreadable", session_id=session_id, error=str(e))
                continue
            if not snapshot.closed and (not with_work or queued or snapshot.has_work()):
                ids.append(session_id)
        return ids

    async def post_inbox(
        self,
        session_id: str,
        kind: str,
        body: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        agent: str | None = None,
    ) -> str:
        raw = _inbox_body(kind, body, idempotency_key)
        item_id, cap = _new_inbox_id(), self.max_inbox_items

        def write(conn):
            def post():
                now = time.time()
                row = conn.execute(
                    "SELECT snapshot, agent FROM chat_sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if row is None or _closed(row[0]) or (agent is not None and row[1] != agent):
                    raise LookupError(
                        f"No stored chat session {session_id!r}" + (f" of agent {agent!r}." if agent else ".")
                    )
                if idempotency_key is not None:
                    # A key kept past its time is forgotten here, by the post that names it: the post is a new item.
                    conn.execute(
                        "DELETE FROM chat_inbox WHERE session_id = ? AND idempotency_key = ? AND acked_at < ?",
                        (session_id, idempotency_key, now - INBOX_KEY_KEPT_S),
                    )
                    found = conn.execute(
                        "SELECT id FROM chat_inbox WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, idempotency_key),
                    ).fetchone()
                    if found is not None:
                        return found[0]
                waiting = conn.execute(
                    "SELECT count(*) FROM chat_inbox WHERE session_id = ? AND acked_at IS NULL", (session_id,)
                ).fetchone()[0]
                if waiting >= cap:
                    raise InboxFull(f"Chat session {session_id!r} already has {cap} inbox items waiting.")
                conn.execute(
                    "INSERT INTO chat_inbox (id, session_id, kind, body, idempotency_key, posted_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (item_id, session_id, kind, raw, idempotency_key, now),
                )
                return item_id

            # The session's row, the key, the count and the insert in one transaction: another process's post or
            # the write that closes the session comes before all of it or after.
            return _transaction(conn, post)

        posted = await self._run(write)
        _inbox_watch(self.path).waiters.wake(session_id)  # a pull of this process waiting for it: at once
        return posted

    async def pull_inbox(self, session_id: str, owner: str, *, wait_s: float = 0.0) -> list[dict[str, Any]]:
        def query(conn):
            held = (
                _settled(conn)
                .execute("SELECT lease_owner, lease_until FROM chat_sessions WHERE session_id = ?", (session_id,))
                .fetchone()
            )
            if held is None:
                raise LookupError(f"No stored chat session {session_id!r}.")
            if held[0] != owner or (held[1] or 0) < time.time():
                raise SessionLeased(f"session {session_id} is not leased to {owner}")
            rows = conn.execute(
                "SELECT id, kind, body, posted_at FROM chat_inbox WHERE session_id = ? AND acked_at IS NULL "
                "ORDER BY rowid",
                (session_id,),
            ).fetchall()
            return [_inbox_item(*row) for row in rows]

        if wait_s <= 0:
            return await self._run(query)
        with _inbox_watch(self.path).watching(session_id, self._sessions_with_items, self.inbox_poll_s) as posted:
            items = await self._run(query)
            if items:
                return items
            await asyncio.wait({posted}, timeout=wait_s)
        return await self._run(query)

    async def find_inbox_key(self, session_id: str, key: str) -> tuple[str, bool] | None:
        def query(conn):
            return (
                _settled(conn)
                .execute(
                    "SELECT id, acked_at FROM chat_inbox WHERE session_id = ? AND idempotency_key = ? "
                    "AND (acked_at IS NULL OR acked_at >= ?)",
                    (session_id, key, time.time() - INBOX_KEY_KEPT_S),
                )
                .fetchone()
            )

        row = await self._run(query)
        return (row[0], row[1] is not None) if row is not None else None

    async def inbox_waiting(self, session_id: str) -> int:
        def query(conn):
            return (
                _settled(conn)
                .execute("SELECT count(*) FROM chat_inbox WHERE session_id = ? AND acked_at IS NULL", (session_id,))
                .fetchone()[0]
            )

        return await self._run(query)

    def _sessions_with_items(self, session_ids: list[str]) -> set[str]:
        """For the watch's thread: which of ``session_ids`` have items waiting."""
        db = self._db()
        with db.lock:
            found = {row[0] for row in _settled(db.conn).execute(_WAITING_SESSIONS_SQL).fetchall()}
        return found.intersection(session_ids)

    async def ack_inbox(self, session_id: str, owner: str, ids: Collection[str], *, forget: bool = False) -> None:
        ids = list(ids)

        def write(conn):
            def ack():
                held = conn.execute(
                    "SELECT lease_owner FROM chat_sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if held is None:
                    raise LookupError(f"No stored chat session {session_id!r}.")
                if held[0] != owner:
                    # The items are the new owner's to apply now; what this one stored says it applied them.
                    raise SessionLeased(f"session {session_id} is not leased to {owner}")
                _ack_rows(conn, session_id, ids, time.time(), forget=forget)

            _transaction(conn, ack)

        await self._run(write)

    async def acquire_lease(self, session_id: str, owner: str) -> None:
        now, until = time.time(), time.time() + self.lease_ttl_s

        def write(conn):
            cursor = conn.execute(
                "UPDATE chat_sessions SET lease_owner = ?, lease_until = ? "
                "WHERE session_id = ? AND (lease_owner IS NULL OR lease_owner = ? OR lease_until < ?)",
                (owner, until, session_id, owner, now),
            )
            if cursor.rowcount == 1:
                conn.commit()
                return None
            held = conn.execute("SELECT lease_owner FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
            if held is not None and not _holder_alive(held[0]):
                # Held by a dead process of this host: take it, unless someone else took it meanwhile.
                cursor = conn.execute(
                    "UPDATE chat_sessions SET lease_owner = ?, lease_until = ? WHERE session_id = ? AND lease_owner = ?",
                    (owner, until, session_id, held[0]),
                )
                if cursor.rowcount == 1:
                    conn.commit()
                    return None
            conn.commit()
            return held

        held = await self._run(write)
        if held is not None:
            raise SessionLeased(f"session {session_id} is held by {held[0]}")

    async def release_lease(self, session_id: str, owner: str) -> None:
        def write(conn):
            conn.execute(
                "UPDATE chat_sessions SET lease_owner = NULL, lease_until = NULL "
                "WHERE session_id = ? AND lease_owner = ?",
                (session_id, owner),
            )
            conn.commit()

        await self._run(write)
