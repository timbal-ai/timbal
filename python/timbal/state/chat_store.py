"""Durable chat sessions: what a :class:`~timbal.core.chat_session.ChatSession` needs to come back after a restart.

The conversation itself is in the traces. A snapshot holds the session layer on top of them: the head run,
the messages and wakes no turn has taken yet, a turn that was running, whether the session is parked. One
mutable row per session, written ahead of every change (a message is durable before its receipt goes out).

A lease (owner + expiry, renewed while the session is open) keeps two processes from serving one session,
and every save is a compare-and-swap on ``version``: an owner that lost its lease cannot overwrite the new one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from uuid_extensions import uuid7

__all__ = [
    "PROCESS_OWNER",
    "ChatSessionStore",
    "InMemoryChatSessionStore",
    "SessionLeased",
    "SessionSnapshot",
    "SnapshotConflict",
    "SqliteChatSessionStore",
]

PROCESS_OWNER = f"{socket.gethostname()}:{os.getpid()}:{uuid7(as_type='hex')}"
"""This process, as a lease owner: unique across restarts of the same pid."""


def _holder_alive(owner: str | None) -> bool:
    """False for a lease held by a process of this host that no longer exists (``kill -9``): its lease would
    otherwise keep the session from the process restarted after it until it expires."""
    if not owner:
        return False
    host, _, rest = owner.partition(":")
    pid = rest.partition(":")[0]
    if host != socket.gethostname() or not pid.isdigit():
        return True  # another host's: only expiry tells
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, another user's
    return True


class SessionLeased(RuntimeError):
    """Another live process holds the session."""


class SnapshotConflict(RuntimeError):
    """The stored snapshot is not the version this owner last wrote: another owner took the session."""


@dataclass
class SessionSnapshot:
    session_id: str
    agent: str
    """The agent's name: a session is only restored for the agent it belongs to."""
    config: dict[str, Any] = field(default_factory=dict)
    head: str | None = None
    root_run_id: str | None = None
    """The thread's root run, once known (:attr:`ChatSession.root_run_id`); absent from older snapshots."""
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
    inbox_seen: list[str] = field(default_factory=list)
    """Ids of the inbox items this session applied (the last few): a pull repeated after a crash skips them."""
    version: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"), default=str)

    @classmethod
    def from_json(cls, raw: str) -> SessionSnapshot:
        data = json.loads(raw)
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})

    def has_work(self) -> bool:
        """Something would run without a client asking: restore it at startup, not on first use."""
        return not self.closed and bool(self.in_flight or self.pending or self.wakes or self.resume)


INBOX_KINDS = ("message", "wake", "agent_message")
"""What another process can queue for a session's owner (:meth:`ChatSessionStore.post_inbox`)."""
_INBOX_SEEN_KEPT = 256


def _inbox_item(session_id: str, kind: str, body: dict[str, Any]) -> dict[str, Any]:
    if kind not in INBOX_KINDS:
        raise ValueError(f"kind must be one of {list(INBOX_KINDS)}, got {kind!r}")
    if not isinstance(body, dict):
        raise TypeError("body must be a dict")
    json.dumps(body)  # must survive the store
    return {
        "id": f"in_{uuid7(as_type='hex')}",
        "session_id": session_id,
        "kind": kind,
        "body": body,
        "posted_at": time.time(),
    }


class ChatSessionStore(ABC):
    """Where snapshots live. Implementations must make ``save`` and ``acquire_lease`` atomic.

    A store that ``supports_inbox`` also carries work other processes queue for a session they do not serve
    (a message, a wake, an agent message): the lease owner long-polls :meth:`pull_inbox`, applies each item,
    stores the session, then :meth:`ack_inbox`. At least once: the snapshot remembers the items it applied
    (``inbox_seen``), so a pull repeated after a crash applies nothing twice.
    """

    lease_ttl_s: float = 30.0
    supports_inbox: bool = False

    async def post_inbox(self, session_id: str, kind: str, body: dict[str, Any]) -> str:
        """Queue ``kind`` (:data:`INBOX_KINDS`) for the session's owner; returns the item id. ``LookupError``:
        no such session."""
        raise NotImplementedError(f"{type(self).__name__} has no inbox")

    async def pull_inbox(self, session_id: str, owner: str, *, wait_s: float = 0.0) -> list[dict[str, Any]]:  # noqa: ARG002
        """The items not acked yet, oldest first (``{id, kind, body, posted_at}``), waiting up to ``wait_s`` for
        one. Only for the lease holder: :class:`SessionLeased` otherwise."""
        return []

    async def ack_inbox(self, session_id: str, owner: str, ids: list[str]) -> None:  # noqa: ARG002
        """Forget items the owner applied and stored."""
        return None

    @abstractmethod
    async def load(self, session_id: str) -> SessionSnapshot | None: ...

    @abstractmethod
    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None) -> int:
        """Write ``snapshot`` if the stored version is ``expect_version`` (None: the row must not exist).
        Returns the new version; raises :class:`SnapshotConflict` otherwise."""

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...

    @abstractmethod
    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        """Open sessions (``with_work``: only those :meth:`SessionSnapshot.has_work` would restore eagerly)."""

    @abstractmethod
    async def acquire_lease(self, session_id: str, owner: str) -> None:
        """Take or renew the lease. Raises :class:`SessionLeased` while another owner's has not expired."""

    @abstractmethod
    async def release_lease(self, session_id: str, owner: str) -> None: ...


class InMemoryChatSessionStore(ChatSessionStore):
    """Process-local (the default): what sessions did before they were durable, through the same interface."""

    supports_inbox = True

    def __init__(self) -> None:
        self._rows: dict[str, tuple[SessionSnapshot, str | None, float]] = {}
        self._inbox: dict[str, list[dict[str, Any]]] = {}
        self._inbox_changed: dict[str, asyncio.Event] = {}

    def _check_holder(self, session_id: str, owner: str) -> None:
        row = self._rows.get(session_id)
        if row is None:
            raise LookupError(f"No stored chat session {session_id!r}.")
        if row[1] != owner or row[2] < time.time():
            raise SessionLeased(f"session {session_id} is not leased to {owner}")

    async def post_inbox(self, session_id: str, kind: str, body: dict[str, Any]) -> str:
        if session_id not in self._rows:
            raise LookupError(f"No stored chat session {session_id!r}.")
        item = _inbox_item(session_id, kind, body)
        self._inbox.setdefault(session_id, []).append(item)
        changed = self._inbox_changed.get(session_id)
        if changed is not None:
            changed.set()
        return item["id"]

    async def pull_inbox(self, session_id: str, owner: str, *, wait_s: float = 0.0) -> list[dict[str, Any]]:
        self._check_holder(session_id, owner)
        if not self._inbox.get(session_id) and wait_s > 0:
            changed = self._inbox_changed.setdefault(session_id, asyncio.Event())
            changed.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(changed.wait(), wait_s)
            self._check_holder(session_id, owner)
        return [dict(item) for item in self._inbox.get(session_id, [])]

    async def ack_inbox(self, session_id: str, owner: str, ids: list[str]) -> None:  # noqa: ARG002
        done = set(ids)
        self._inbox[session_id] = [item for item in self._inbox.get(session_id, []) if item["id"] not in done]

    async def load(self, session_id: str) -> SessionSnapshot | None:
        row = self._rows.get(session_id)
        return SessionSnapshot.from_json(row[0].to_json()) if row is not None else None

    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None) -> int:
        row = self._rows.get(snapshot.session_id)
        current = row[0].version if row is not None else None
        if current != expect_version:
            raise SnapshotConflict(
                f"session {snapshot.session_id}: stored version {current}, expected {expect_version}"
            )
        stored = SessionSnapshot.from_json(snapshot.to_json())
        stored.version = (current or 0) + 1
        owner, until = (row[1], row[2]) if row is not None else (None, 0.0)
        self._rows[snapshot.session_id] = (stored, owner, until)
        return stored.version

    async def delete(self, session_id: str) -> None:
        self._rows.pop(session_id, None)
        self._inbox.pop(session_id, None)

    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        return [
            sid
            for sid, (snap, _, _) in self._rows.items()
            if not snap.closed and (not with_work or snap.has_work() or self._inbox.get(sid))
        ]

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


class SqliteChatSessionStore(ChatSessionStore):
    """Snapshots in the ``chat_sessions`` table of a SQLite file, next to
    :class:`~timbal.state.tracing.providers.SqliteTracingProvider`'s runs (same connection when it is the
    same file)."""

    supports_inbox = True
    # A pull waiting for an item polls the table this often (other processes write it; nothing notifies).
    inbox_poll_s: float = 0.25

    def __init__(self, path: Path | str, *, synchronous: str = "NORMAL") -> None:
        self.path = Path(path)
        self.synchronous = synchronous

    async def post_inbox(self, session_id: str, kind: str, body: dict[str, Any]) -> str:
        item = _inbox_item(session_id, kind, body)

        def write(conn):
            if conn.execute("SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
                return False
            conn.execute(
                "INSERT INTO chat_session_inbox (id, session_id, item, posted_at) VALUES (?, ?, ?, ?)",
                (item["id"], session_id, json.dumps(item), item["posted_at"]),
            )
            conn.commit()
            return True

        if not await self._run(write):
            raise LookupError(f"No stored chat session {session_id!r}.")
        return item["id"]

    async def pull_inbox(self, session_id: str, owner: str, *, wait_s: float = 0.0) -> list[dict[str, Any]]:
        def query(conn):
            held = conn.execute(
                "SELECT lease_owner, lease_until FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if held is None:
                raise LookupError(f"No stored chat session {session_id!r}.")
            if held[0] != owner or (held[1] or 0) < time.time():
                raise SessionLeased(f"session {session_id} is not leased to {owner}")
            rows = conn.execute(
                "SELECT item FROM chat_session_inbox WHERE session_id = ? ORDER BY posted_at, id", (session_id,)
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

        deadline = time.monotonic() + wait_s
        while True:
            items = await self._run(query)
            if items or time.monotonic() >= deadline:
                return items
            await asyncio.sleep(min(self.inbox_poll_s, max(0.0, deadline - time.monotonic())))

    async def ack_inbox(self, session_id: str, owner: str, ids: list[str]) -> None:  # noqa: ARG002
        if not ids:
            return

        def write(conn):
            conn.executemany(
                "DELETE FROM chat_session_inbox WHERE session_id = ? AND id = ?", [(session_id, i) for i in ids]
            )
            conn.commit()

        await self._run(write)

    def _db(self):
        from .tracing.providers.sqlite import database

        return database(self.path, self.synchronous)

    async def _run(self, fn):
        def locked():
            db = self._db()
            with db.lock:
                return fn(db.conn)

        return await asyncio.to_thread(locked)

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

    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None) -> int:
        new_version = (expect_version or 0) + 1
        body = SessionSnapshot.from_json(snapshot.to_json())
        body.version = new_version
        raw, now = body.to_json(), int(time.time() * 1000)

        def write(conn):
            if expect_version is None:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO chat_sessions (session_id, agent, snapshot, version, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (snapshot.session_id, snapshot.agent, raw, new_version, now),
                )
            else:
                cursor = conn.execute(
                    "UPDATE chat_sessions SET snapshot = ?, version = ?, updated_at = ? "
                    "WHERE session_id = ? AND version = ?",
                    (raw, new_version, now, snapshot.session_id, expect_version),
                )
            conn.commit()
            return cursor.rowcount

        if await self._run(write) != 1:
            raise SnapshotConflict(f"session {snapshot.session_id}: not at version {expect_version}")
        return new_version

    async def delete(self, session_id: str) -> None:
        def write(conn):
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_session_inbox WHERE session_id = ?", (session_id,))
            conn.commit()

        await self._run(write)

    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        def query(conn):
            rows = conn.execute("SELECT session_id, snapshot FROM chat_sessions").fetchall()
            queued = {row[0] for row in conn.execute("SELECT DISTINCT session_id FROM chat_session_inbox")}
            return rows, queued

        rows, queued = await self._run(query)
        ids = []
        for session_id, raw in rows:
            snapshot = SessionSnapshot.from_json(raw)
            if not snapshot.closed and (not with_work or snapshot.has_work() or session_id in queued):
                ids.append(session_id)
        return ids

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
