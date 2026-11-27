"""Durable chat sessions under faults placed everywhere, not at one hand-picked spot.

A scenario (a few user actions against a durable ``ChatSession``) is run once to count its writes to the
session store and to the trace store, then once more for every write: the process dies at that write (it and
everything after it never land, every task of the session stops), the session is restored the way a new
process would restore it, driven until it is idle, and checked:

- **at most once**: no tool with a side effect ran twice, across the dead process and the new one;
- **nothing acknowledged is lost**: a message, wake or answer whose ``flush()`` returned is in the
  conversation, or is named in a recovery notice;
- **nothing is answered twice**: no session event appears in the history more than once;
- **a stop holds**: once an interrupt is acknowledged against a question the session waits on (or is about
  to), the tool behind that question never runs, whatever is approved afterwards;
- **nothing said is forgotten**: while the process lives, every answer of a turn the session reported as
  finished is in the stored conversation (after a crash an unsaved answer is gone, and recovery says so);
- **it settles**: the restored session becomes idle (answering what it still waits for), with nothing left
  pending, and what it stores then matches what it holds;
- **the thread keeps its root**: ``root_run_id`` (the MCP session hint) is the run the thread started with,
  never a run that died and was run again under another id; restored from snapshots written before the field
  existed (``rooted_legacy``), it is that run or unknown.

Further sweeps: the process dies a second time, during the recovery; it shuts down in order at that write
instead of dying; it lives on while the session store, or the trace store, is away for a stretch of writes; it
stops running for longer than its lease and then runs on (its session finds the lease gone and stops).

The other crash tests in ``test_chat_session_durable.py`` kill a session after a last successful flush, which
is one point of this space; these find the others.
"""

import asyncio
import collections
import contextlib
import sys

import pytest
from timbal import Agent, Tool
from timbal.core.agent import _is_pause
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.state import background, set_call_id, set_parent_call_id, set_run_context, suspend
from timbal.state.chat_store import SessionNotStored, SqliteChatSessionStore
from timbal.state.tracing.providers import SqliteTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import OutputEvent
from timbal.types.message import Message

# A test that times out because nothing in its process ran, or because something blocked the event loop, says
# so, with the stacks (conftest.py `loop_watchdog`): on loaded CI runners a test process has stood still for
# 70-260 s, and the sessions it was waiting for looked stuck.
pytestmark = pytest.mark.usefixtures("loop_watchdog")

SIDE_EFFECTS = ("work", "ship", "charge", "doit", "dig")
RECOVERY = "cut off by a server restart"
WAKE_TEXT = "check the inbox"


@pytest.fixture(autouse=True)
def isolated_context(monkeypatch):
    import timbal.core.agent as agent_module

    monkeypatch.setattr(agent_module, "NOW_COALESCE_SECS", 0.005)  # a 'now' cut waits this long for more
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)


# The one SQLite file of a test (in its tmp_path): every run of a sweep stores its sessions and traces there.
DATABASE = "sessions.db"


@pytest.fixture(autouse=True)
def one_database(tmp_path):
    """One file for all the runs of a test, not one per run. Each run has a session and runs of its own (new ids),
    so no run reads another's rows, and a new process of a run reads the same file, as before. A file per run
    was most of a sweep's time on Windows: creating one (WAL, schema), then closing and deleting it, took
    40-180 ms there against 2-3 ms on Linux and macOS (measured on windows-latest, 2026-11)."""
    yield
    release(tmp_path / DATABASE)


@pytest.fixture(autouse=True)
def fine_timer():
    """Waits as long as the scenarios ask for, on Windows too.

    Windows ends a wait at a tick of its timer, every 15.6 ms by default, and with Python 3.13's precise clock a
    3 ms sleep often takes two ticks: each poll of :meth:`Client.settle`, each tool's pause and the window a
    second restore is watched in took 15-31 ms there instead of 3-10 ms (a third of a sweep's time on
    windows-latest / py3.13). This asks for a 1 ms timer for the test's length, as Linux and macOS have: a 3 ms
    sleep then takes 3.4-4 ms there. Clocks are left as they are: before Python 3.13 ``time.monotonic`` still moves
    in 15-16 ms steps (measured), so the coarse-clock cases of CLAUDE.md still run. Since Windows 10 2004 the
    request is the process's own: other workers keep theirs."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    winmm = ctypes.WinDLL("winmm")
    granted = winmm.timeBeginPeriod(1) == 0  # TIMERR_NOERROR
    try:
        yield
    finally:
        if granted:
            winmm.timeEndPeriod(1)


# -- The fault -------------------------------------------------------------------


class Fault:
    """Counts the writes of one session (snapshots, traces, approval claims) and breaks something at write
    number ``at + 1``:

    - ``"die"``: the process dies. That write and all later ones never land; every task of the session stops.
    - ``"close"``: the process shuts down in order from there (``close(keep=True)``); its writes land.
    - ``"store"`` / ``"trace"``: that store fails for ``length`` writes, and the process lives on.
    - ``"taken"``: another process takes the session over just before that write to the session store.
    - ``"stop"``: the user stops the session (``interrupt()``) just as that write is made, from inside it: a
      trace save is cut where it stands, which no amount of waiting in a scenario hits on purpose.
    - ``"stall"``: the process stops running for longer than its lease (a paused VM, a starved runner) just
      as that write is made: the clock jumps (:class:`Clock`), and the session finds its lease gone.

    ``then=(kind, after)``: an outage that the process does not outlive. It dies (``"die"``) or shuts down
    (``"close"``) ``after`` writes into it.
    """

    def __init__(self, at=None, kind="die", *, length=3, then=None):
        self.at, self.kind, self.length, self.then = at, kind, length, then
        self.writes = 0
        self.gone = asyncio.Event()  # the process died, or is shutting down: the caller stops
        self.dead = False
        self.closing: asyncio.Future | None = None
        self.taken: int | None = None  # the version the other process left the row at
        self.session = None
        self.stop = None  # what a "stop" fault calls: the client's stop()
        self.stalled = False

    def write(self) -> bool:
        """Count a write. False: the process is dead (the caller stops with its task)."""
        if self.dead:
            return False
        self.writes += 1
        if self.kind == "stop" and self.writes - 1 == self.at and self.stop is not None:
            self.stop()
        if self.at is not None and self.writes > self.at and not self.gone.is_set():
            kind = self.kind
            if self.then is not None and self.writes > self.at + self.then[1]:
                kind = self.then[0]
            if kind == "die":
                self.die()
                return False
            if kind == "close":
                self.gone.set()
                self.closing = asyncio.ensure_future(self.session.close(keep=True))
            if kind == "stall" and not self.stalled:
                self.stalled = True
                Clock.jump(STALL_S)
        return True

    def failing(self, kind) -> bool:
        """Whether the write counted just now is one the ``kind`` outage takes."""
        return self.kind == kind and self.at is not None and self.at < self.writes <= self.at + self.length

    def die(self) -> None:
        self.dead = True
        self.gone.set()
        session = self.session
        if session is None:
            return
        session._lost = True  # a dead process stores nothing more
        for task in (session._worker, session._turn, session._lease_task, session._saver):
            if task is not None and not task.done():
                task.cancel()


class Clock:
    """``time.monotonic`` as a stalled process sees it on waking: everything that measures time (the lease,
    the event loop's timers) finds ``jump`` seconds gone at once. Installed by the tests that stall."""

    shift = 0.0
    real = None

    @classmethod
    def install(cls, monkeypatch):
        import time

        cls.shift, cls.real = 0.0, time.monotonic
        monkeypatch.setattr(time, "monotonic", lambda: cls.real() + cls.shift)

    @classmethod
    def jump(cls, seconds):
        assert cls.real is not None, "Clock.install() first"
        cls.shift += seconds


# Past the 30 s lease and its margin, inside the scenario's 60 s hang guard.
STALL_S = 40.0


class FaultySqlite(SqliteTracingProvider):
    fault: Fault | None = None

    @classmethod
    async def _store(cls, run_context):
        if cls.fault is not None:
            if not cls.fault.write():
                raise asyncio.CancelledError
            if cls.fault.failing("trace"):
                raise OSError("the trace store is unreachable")
        await super()._store(run_context)

    @classmethod
    async def claim_approval(cls, parent_id, approval_id, run_id):
        if cls.fault is not None and not cls.fault.write():
            raise asyncio.CancelledError
        return await super().claim_approval(parent_id, approval_id, run_id)


def faulty_store(path, fault, *, legacy=False):
    """``legacy``: every snapshot loads without ``root_run_id`` (nor ``in_flight.starts_thread``), as one written
    before the fields existed."""
    store = SqliteChatSessionStore(path)
    real_save, real_acquire, real_release = store.save, store.acquire_lease, store.release_lease
    real_load = store.load

    async def load(session_id):
        snapshot = await real_load(session_id)
        if legacy and snapshot is not None:
            snapshot.root_run_id = None
            if snapshot.in_flight is not None:
                snapshot.in_flight.pop("starts_thread", None)
        return snapshot

    async def save(snapshot, *, expect_version, owner=None):
        if not fault.write():
            raise asyncio.CancelledError
        if fault.failing("store"):
            raise OSError("the session store is unreachable")
        if fault.kind == "taken" and fault.at is not None and fault.writes > fault.at and fault.taken is None:
            fault.taken = await take_over(store, snapshot.session_id)
        return await real_save(snapshot, expect_version=expect_version, owner=owner)

    async def acquire_lease(session_id, owner):
        if fault.dead:
            raise asyncio.CancelledError
        return await real_acquire(session_id, owner)

    async def release_lease(session_id, owner):
        if fault.dead:
            raise asyncio.CancelledError
        return await real_release(session_id, owner)

    store.save, store.acquire_lease, store.release_lease, store.load = save, acquire_lease, release_lease, load
    return store


TAKER = "another-host:1:x"


async def take_over(store, session_id) -> int | None:
    """Another process's takeover, as the store sees it: its lease on the row, and a write of its own."""

    def write(conn):
        conn.execute(
            "UPDATE chat_sessions SET lease_owner = ?, lease_until = ?, version = version + 1 WHERE session_id = ?",
            (TAKER, 4102444800.0, session_id),
        )
        row = conn.execute("SELECT version FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
        conn.commit()
        return row[0] if row is not None else None  # no row yet: nothing to take, the next write tries again

    return await store._run(write)


def release(path) -> None:
    """Close a test's database and delete it: the provider keeps one connection per file for the life of the
    process, and a worker runs hundreds of these tests."""
    from pathlib import Path

    from timbal.state.tracing.providers import sqlite

    resolved = str(Path(path).resolve())
    with sqlite._DATABASES_LOCK:
        for key in [key for key in sqlite._DATABASES if key[0] == resolved]:
            db = sqlite._DATABASES.pop(key)
            with db.lock:
                db.conn.close()
    for suffix in ("", "-wal", "-shm"):
        Path(resolved + suffix).unlink(missing_ok=True)


def forget(session) -> None:
    """Drop what only the old process's memory held, so the restore that follows starts as a new process does
    (in one test process the run registry would hand the new session the old one's task store)."""
    for run_id, store in list(background._STORES_BY_RUN_ID.items()):
        if store is session._store:
            del background._STORES_BY_RUN_ID[run_id]
    session._store._events.clear()


async def bury(session):
    """After :meth:`Fault.die`: wait for the dead process's tasks to stop."""
    tasks = [t for t in (session._worker, session._turn, session._lease_task, session._saver) if t is not None]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    session.closed = True
    session._scheduler.release(session._store)
    forget(session)


# -- The agent ---------------------------------------------------------------------

# First match wins: a wake's prompt carries runtime notes around its text, which must not pick the plan.
# First match wins: a wake's prompt carries runtime notes around its text, which must not pick the plan.
# Each step is the tools one model call asks for (two: a batch, run as one task per call).
PLANS = {
    "check": [("lookup",)],
    "two": [("lookup",), ("ship",)],
    "batch": [("ship", "lookup")],
    "delegate": [("child",)],
    "pay": [("charge",)],
    "ask": [("ask",), ("ship",)],
    "work": [("work",)],
}


def model(messages):
    """What a careful model does: a plan per request, one step at a time; told it was cut by a restart, it
    reports and stops (the recovery turn can only read)."""
    last_user = max(i for i, m in enumerate(messages) if m.role == "user")
    text = messages[last_user].collect_text()
    if RECOVERY in text:
        return "recovered"
    plan = next((steps for key, steps in PLANS.items() if key in text), [])
    done = sum(1 for m in messages[last_user:] if m.role == "assistant")
    if done < len(plan):
        return Message(
            role="assistant",
            content=[
                ToolUseContent(
                    id=f"{name}-{len(messages)}", name=name, input={"prompt": "dig"} if name == "child" else {}
                )
                for name in plan[done]
            ],
            stop_reason="tool_use",
        )
    return f"answer: {text.splitlines()[-1][:40]}"


def child_model(messages):
    if messages[-1].role == "tool":
        return "dug"
    return Message(
        role="assistant",
        content=[ToolUseContent(id=f"dig-{len(messages)}", name="dig", input={})],
        stop_reason="tool_use",
    )


class _Ledger:
    """Where the tools of the one agent count their runs: swapped for every run of a sweep."""

    calls: collections.Counter = collections.Counter()


def _build_agent():
    def counted(name, *, pause=0.003):
        async def handler() -> str:
            _Ledger.calls[name] += 1  # the side effect lands here, whatever happens to the process next
            await asyncio.sleep(pause)
            return f"{name} done"

        return handler

    def ask() -> str:
        _Ledger.calls["ask"] += 1
        return str(suspend({"question": "Ship it?"}))

    return Agent(
        name="agent",
        model=TestModel(handler=model),
        tools=[
            Tool(name="work", handler=counted("work")),
            Tool(name="ship", handler=counted("ship")),
            Tool(name="lookup", handler=counted("lookup"), read_only=True),
            Tool(name="charge", handler=counted("charge"), requires_approval=True),
            Tool(name="ask", handler=ask),
            Tool(name="doit", command="/doit", handler=counted("doit")),
            Agent(
                name="child",
                model=TestModel(handler=child_model),
                tools=[Tool(name="dig", handler=counted("dig"))],
                tracing_provider=FaultySqlite,
            ),
        ],
        tracing_provider=FaultySqlite,
    )


_AGENT: Agent | None = None


def make_agent(provider, calls):
    """The agent of every run. Built once (building its tools costs more than a whole run) and pointed at
    this run's trace store and ledger: the runs of a sweep come one after the other."""
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent()
    _Ledger.calls = calls
    _AGENT.tracing_provider = provider
    for tool in _AGENT.tools:
        if isinstance(tool, Agent):
            tool.tracing_provider = provider
    return _AGENT


# -- Driving a session ---------------------------------------------------------------


class Client:
    """What a caller does, and what it was told: ``acked`` holds what a ``flush()`` returned for."""

    def __init__(self, session, fault, calls):
        self.session, self.fault, self.calls = session, fault, calls
        self.acked: dict[str, str] = {}  # event id -> text
        self.acked_wakes: list[str] = []
        self.must_not_run: set[str] = set()  # tools an acknowledged stop has ruled out
        self.approved = False  # the client has said yes to what the session asked
        fault.stop = self.stop

    async def stored(self) -> bool:
        """``session.flush()`` as a route awaits it. False: the process died first, or the store refused."""
        flush = asyncio.ensure_future(self.session.flush())
        dead = asyncio.ensure_future(self.fault.gone.wait())
        await asyncio.wait({flush, dead}, return_when=asyncio.FIRST_COMPLETED)
        dead.cancel()
        if not flush.done():
            flush.cancel()
        await asyncio.gather(flush, dead, return_exceptions=True)
        return not flush.cancelled() and flush.exception() is None

    def _alive(self) -> bool:
        return not self.fault.gone.is_set() and not self.session.closed

    async def send(self, text, mode=None) -> bool:
        """Send a message as the route does. True: it was acknowledged."""
        if not self._alive():
            return False
        with contextlib.suppress(RuntimeError):  # closed under us
            event = self.session.send(text, mode=mode, defer_cut=True)
            mark = self.session.mark()
            if await self.stored() or (self._alive() and self.session.stored_since(mark)):
                self.acked[event.id] = text
                self.session.cut_for(event, mode)
                return True
            if self._alive():
                self.session.withdraw(event.id)  # as the route does before its 503
        return False

    async def agent(self, text, *, message_id, priority="normal") -> bool:
        """A message from another agent, as ``POST .../agent_messages`` takes it. True: it was acknowledged."""
        from timbal.core.peers import PeerRejected

        if not self._alive():
            return False
        with contextlib.suppress(PeerRejected):  # closed under us
            receipt = await self.session.receive_agent_message(
                text, sender="planner", message_id=message_id, priority=priority
            )
            mark = self.session.mark()
            if await self.stored() or (self._alive() and self.session.stored_since(mark)):
                self.acked[receipt["event_id"]] = text
                return True
            if self._alive() and not receipt["duplicate"]:
                self.session.withdraw(receipt["event_id"])  # as the route does before its 503
        return False

    def stop(self) -> None:
        """Stop the session as ``POST .../interrupt`` does. Once a turn has started (a message no turn has
        taken yet is still queued, and a stop leaves the queue alone), a stop that is taken rules out the
        gated tool, unless its user has already said yes to it: that call is on its way, and a stop that
        arrives while it is being started may find it started."""
        session = self.session
        if self._alive() and session.turns and self.calls["charge"] == 0 and session.interrupt():
            if not self.approved:
                self.must_not_run.add("charge")

    async def wake(self, text, key=None) -> None:
        if not self._alive():
            return
        with contextlib.suppress(RuntimeError):
            wake_id = self.session.wake(text, key=key)
            mark = self.session.mark()
            if await self.stored() or (self._alive() and self.session.stored_since(mark)):
                if key is not None:
                    self.acked_wakes.clear()  # it replaces the pending wake of its key
                self.acked_wakes.append(text)
            elif self._alive():
                self.session.withdraw(wake_id)

    # Hang guards, not timing checks: a scenario takes well under a second, but polling and store retries
    # stretch with a slow runner's timers (Windows before Python 3.13 sleeps in 15.6 ms ticks).
    async def until(self, condition, timeout=60.0) -> bool:
        """True once ``condition()`` holds; False if the process died first."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not condition():
            if not self._alive():
                return False
            assert asyncio.get_running_loop().time() < deadline, "the session never got there (a hang)"
            await asyncio.sleep(0.003)
        return True

    def at_rest(self) -> bool:
        session = self.session
        return (
            not session.busy
            and not session._wakes
            and session._resume is None
            and not session.pending_messages()
            and session._in_flight is None
            # An event that starts a turn (another agent's message) is still to be answered, as a message is.
            and not session.due
        )

    async def answer(self) -> bool:
        """Answer everything the parked turn waits for (approve, or "yes"). False when there is nothing."""
        state = self.session.state()
        values = {a["approval_id"]: True for a in state.get("pending_approvals", [])}
        values.update({i["interaction_id"]: "yes" for i in state.get("pending_interactions", [])})
        if not values or not self._alive():
            return False
        with contextlib.suppress(RuntimeError, ValueError):
            self.approved = True
            self.session.resume(values)
            mark = self.session.mark()
            if not (await self.stored() or not self._alive() or self.session.stored_since(mark)):
                self.session.withdraw_answers(values)
        return True

    async def settle(self, timeout=60.0) -> None:
        """Until the session has nothing left to do, answering whatever it parks on."""
        deadline = asyncio.get_running_loop().time() + timeout
        while self._alive():
            assert asyncio.get_running_loop().time() < deadline, f"never settled: {self.session.state()}"
            if self.session.parked and not self.session.busy and self.session._resume is None:
                await self.answer()
            elif self.at_rest():
                return
            await asyncio.sleep(0.003)


async def plain(c):
    await c.send("work")
    await c.settle()


async def two_steps(c):
    await c.send("two")
    await c.settle()


async def queued(c):
    await c.send("work")
    await c.send("later", mode="followup")
    await c.settle()


async def steered(c):
    await c.send("work")
    if await c.until(lambda: c.calls["work"] or c.at_rest()):
        await c.send("also", mode="steer")
    await c.settle()


async def approved(c):
    await c.send("pay")
    await c.settle()  # parks on the approval, answers it, runs the charge


async def questioned(c):
    await c.send("ask")
    await c.settle()  # parks on the question, answers it, goes on to ship


async def woken(c):
    await c.wake(WAKE_TEXT)
    await c.settle()


async def interrupted(c):
    await c.send("work")
    if await c.until(lambda: c.calls["work"] or c.at_rest()):
        await c.send("stop", mode="interrupt")
    await c.settle()


async def interrupted_as_it_ends(c):
    """The interrupt comes when the turn's tool is done and the turn is ending: with the trace store away it
    is still busy then, waiting for its trace."""
    await c.send("work")
    if await c.until(lambda: c.calls["work"] or c.at_rest()):
        await asyncio.sleep(0.012)
        await c.send("stop", mode="interrupt")
    await c.settle()


async def cut_now(c):
    await c.send("batch")
    if await c.until(lambda: c.calls["ship"] or c.at_rest()):
        await c.send("also", mode="now")
    await c.settle()


async def batch(c):
    await c.send("batch")
    await c.settle()


async def delegated(c):
    await c.send("delegate")
    await c.settle()


async def slash(c):
    await c.send("/doit")
    await c.settle()


async def collected(c):
    await c.send("work", mode="collect")
    await c.send("too", mode="collect")
    await c.settle()


async def queued_behind_an_approval(c):
    await c.send("pay")
    if await c.until(lambda: c.session.parked or c.at_rest()):
        await c.send("later")
    await c.settle()


async def approval_abandoned(c):
    await c.send("pay")
    if await c.until(lambda: c.session.parked or c.at_rest()):
        if c.calls["charge"] == 0 and await c.send("never mind", mode="interrupt"):
            c.must_not_run.add("charge")
    await c.settle()


async def approval_abandoned_as_it_parks(c):
    """The stop comes when the turn has just asked for approval and is ending: with the trace store away it
    is still busy then, waiting for its trace, and not parked yet."""
    await c.send("pay")
    if await c.until(lambda: c.session.turns or c.at_rest()):
        await asyncio.sleep(0.012)
        if c.calls["charge"] == 0 and await c.send("never mind", mode="interrupt"):
            c.must_not_run.add("charge")
    await c.settle()


async def wake_replaced(c):
    await c.send("work")
    await c.wake(WAKE_TEXT, key="k")
    await c.wake(WAKE_TEXT + " again", key="k")
    await c.settle()


async def rooted(c):
    """Two turns of one thread: the first starts it (its run is the root, under the session's id), the second
    continues it. As ``rooted_legacy``, every restore reads a snapshot that does not say what the root is."""
    await c.send("work")
    await c.settle()
    await c.send("later")
    await c.settle()


async def from_an_agent(c):
    """Another agent's message starts a turn of its own (core/peers.py), which runs a tool."""
    await c.agent("work on the release", message_id="am_1")
    await c.settle()


async def from_an_agent_urgently(c):
    """An urgent message from another agent joins the running turn after its tool batch."""
    await c.send("work")
    if await c.until(lambda: c.calls["work"] or c.at_rest()):
        await c.agent("the release is cancelled", message_id="am_2", priority="urgent")
    await c.settle()


# Scenarios whose snapshots load as ones written before ``root_run_id`` existed.
LEGACY_SNAPSHOTS = {"rooted_legacy"}

SCENARIOS = {
    "plain": plain,
    "two_steps": two_steps,
    "queued": queued,
    "steered": steered,
    "approved": approved,
    "questioned": questioned,
    "woken": woken,
    "interrupted": interrupted,
    "interrupted_as_it_ends": interrupted_as_it_ends,
    "cut_now": cut_now,
    "batch": batch,
    "delegated": delegated,
    "slash": slash,
    "collected": collected,
    "queued_behind_an_approval": queued_behind_an_approval,
    "approval_abandoned": approval_abandoned,
    "approval_abandoned_as_it_parks": approval_abandoned_as_it_parks,
    "wake_replaced": wake_replaced,
    "rooted": rooted,
    "rooted_legacy": rooted,
    "from_an_agent": from_an_agent,
    "from_an_agent_urgently": from_an_agent_urgently,
}


# -- One run, and what must hold after it ---------------------------------------------


async def history(session):
    """The conversation as the next turn will read it: the head run's memory, or, when that run was cut before
    it assembled one, the memory of the last run that did, continued with the prompts of the cut ones by the
    agent's own rule (``Agent._previous_memory``: a run that continues a paused one adds no prompt)."""
    cut, memory, status, run_id = [], [], None, session.head
    while run_id is not None:
        context = await session._load_run(run_id)
        root = context.root_span() if context is not None else None
        if root is None:
            break
        if isinstance(root.memory, list):
            memory, status = [m for m in root.memory if isinstance(m, dict)], root.status
            break
        cut.append(root)
        run_id = context.parent_id
    for root in reversed(cut):
        prompt = root.input.get("prompt") if isinstance(root.input, dict) else None
        if isinstance(prompt, dict) and not _is_pause(status):
            memory.append(prompt)
        status = root.status
    return memory


async def thread_root(session):
    """The run the session's thread started with: the head's parents, followed to the run that has none.
    None when the chain cannot be followed (a run in it is not stored)."""
    run_id = session.head
    while run_id is not None:
        context = await session._load_run(run_id)
        if context is None or context.root_span() is None:
            return None
        if context.parent_id is None:
            return run_id
        run_id = context.parent_id
    return None


def texts(message):
    return " ".join(b.get("text") or "" for b in message.get("content") or [] if isinstance(b, dict))


async def stalled(fault, session) -> None:
    """A session that lost its lease stopped without writing, for another process to restore: what a server
    does with it (it restores a handed-off session on its next request). After a ``"stall"`` fault it must
    have; it also happens unplanned (a CI runner that stops the test process for minutes, a save sent before
    a stall that lands after the next copy has loaded the row). Checked as if final, such a session would
    "lose" messages the store holds."""
    if fault.gone.is_set():
        return
    if fault.stalled:
        await asyncio.wait_for(session.wait_closed(), 10)
        assert session.lost, f"a session stalled past its lease went on: {session.state()}"
    if not session.closed:
        # Its writes landed, or one of them found the row moved on (the session is lost then).
        try:
            await session.flush()
        except asyncio.CancelledError:
            # The planned fault can strike at this flush's own write: a dead store cancels it. Anything else
            # (the test itself cancelled) goes on up.
            if not fault.dead or asyncio.current_task().cancelling():
                raise
        except Exception:
            pass
        if fault.gone.is_set():
            return  # it died or began shutting down during that flush: the caller restores it
    if session.lost:
        await asyncio.wait_for(session.wait_closed(), 10)
        fault.gone.set()
        fault.closing = asyncio.ensure_future(session.wait_closed())


async def run_once(tmp_path, scenario, durability, *faults):
    """Run ``scenario`` under the first fault; each time the process is gone, restore it under the next one
    (none left: nothing fails any more); settle; return what to check."""
    before = asyncio.all_tasks()
    calls = collections.Counter()
    path = tmp_path / DATABASE
    faults = list(faults) or [Fault()]
    fault = faults.pop(0)
    legacy = scenario in LEGACY_SNAPSHOTS
    provider = FaultySqlite.configured(_path=path, fault=fault, durability=durability)
    store = faulty_store(path, fault, legacy=legacy)
    session = ChatSession(make_agent(provider, calls), session_store=store, collect_window=0.0)
    fault.session = session
    client = Client(session, fault, calls)
    await SCENARIOS[scenario](client)
    await stalled(fault, session)
    writes = [fault.writes]
    while fault.gone.is_set():
        assert len(writes) <= 8, f"{scenario}: restored {len(writes)} times and never settled"
        if fault.dead:
            await bury(session)
        else:
            await fault.closing
            forget(session)
        acked, acked_wakes, must_not_run = client.acked, client.acked_wakes, client.must_not_run
        fault = faults.pop(0) if faults else Fault()
        provider = FaultySqlite.configured(_path=path, fault=fault, durability=durability)
        store = faulty_store(path, fault, legacy=legacy)
        try:
            session = await ChatSession.from_store(make_agent(provider, calls), store, session.id)
        except LookupError:
            # It died before the session itself was stored: nothing can have been acknowledged.
            assert not acked and not acked_wakes, "acknowledged, but the session was never stored"
            return {"writes": writes, "calls": calls, "session": None, "before": before, "path": path}
        fault.session = session
        client = Client(session, fault, calls)
        client.acked, client.acked_wakes, client.must_not_run = acked, acked_wakes, must_not_run
        await client.settle()
        await stalled(fault, session)
        writes.append(fault.writes)
    fault.at = None  # the scenario is over: nothing fails any more
    return {
        "writes": writes,
        "calls": calls,
        "session": session,
        "client": client,
        "store": store,
        "before": before,
        "path": path,
        "legacy": legacy,
    }


async def check(result, label):
    try:
        await _check(result, label)
    finally:
        session = result["session"]
        if session is not None and not session.closed:
            await session.close(keep=True)  # a failed check: nothing of it may leak into the next run


async def _check(result, label):
    calls, session = result["calls"], result["session"]
    for name in SIDE_EFFECTS:
        assert calls[name] <= 1, f"{label}: {name} ran {calls[name]} times"
    if session is None:
        return
    client, store = result["client"], result["store"]
    for name in client.must_not_run:
        assert calls[name] == 0, f"{label}: {name} ran after a stop was acknowledged"
    memory = await history(session)
    notices = " ".join(texts(m) for m in memory if RECOVERY in texts(m))
    taken = collections.Counter(
        event_id for m in memory for event_id in (m.get("metadata") or {}).get("event_ids") or []
    )
    for event_id, text in client.acked.items():
        assert taken[event_id] or f'"{text}"' in notices, f"{label}: acknowledged message {text!r} was lost"
    for text in client.acked_wakes:
        assert any(text in texts(m) for m in memory), f"{label}: acknowledged wake {text!r} was lost"
    twice = [event_id for event_id, count in taken.items() if count > 1]
    assert not twice, f"{label}: answered twice: {[client.acked.get(e, e) for e in twice]}"
    if len(result["writes"]) == 1:
        # One process, start to end: what it told the user it answered is what the next turn will read.
        said = [texts(m) for m in memory if m.get("role") == "assistant"]
        for event in session.log.peek(session.log.forgotten_through)[0]:
            if isinstance(event, OutputEvent) and event.parent_call_id is None and event.path == "agent":
                answer = event.output.collect_text() if isinstance(event.output, Message) else ""
                if event.status.code == "success" and answer:
                    assert answer in said, f"{label}: the answer {answer!r} is not in the stored conversation"
    assert client.at_rest() and not session.parked, f"{label}: not settled: {session.state()}"
    await session.flush()
    stored = await store.load(session.id)
    assert stored.in_flight is None and stored.pending == [] and stored.wakes == [], f"{label}: {stored}"
    assert stored.head == session.head and not stored.parked, f"{label}: stored head/parked differ"
    root = await thread_root(session)
    hint = session._last_context._session_id if session._last_context is not None else None
    if result["legacy"]:
        # Restored from snapshots that do not say: the root, or unknown (no hint), never another run.
        assert session.root_run_id in (None, root), f"{label}: root {session.root_run_id!r}, thread {root!r}"
    else:
        assert root is None or session.root_run_id == root, f"{label}: root {session.root_run_id!r}, thread {root!r}"
        assert stored.root_run_id == session.root_run_id, f"{label}: stored root differs"
    assert hint is None or hint == session.root_run_id, f"{label}: the last turn's hint {hint!r} is not the root"
    turns, head, root_run_id = session.turns, session.head, session.root_run_id
    await session.close(keep=True)
    forget(session)
    # Restoring it once more changes nothing: no turn starts, nothing comes back.
    again = await ChatSession.from_store(session._agent, store, session.id)
    await asyncio.sleep(0.01)
    assert (again.turns, again.head, again.busy) == (turns, head, False), f"{label}: a second restore ran something"
    assert result["legacy"] or again.root_run_id == root_run_id, f"{label}: a second restore lost the root"
    assert not again.pending_messages() and not again._wakes and not again.parked, f"{label}: {again.state()}"
    await again.close(keep=True)
    forget(again)
    for name in SIDE_EFFECTS:
        assert calls[name] <= 1, f"{label}: {name} ran {calls[name]} times"
    # Nothing of any of these sessions is left running.
    await asyncio.sleep(0)
    left = [t for t in asyncio.all_tasks() - result["before"] if not t.done() and t is not asyncio.current_task()]
    for task in left:
        task.cancel()
    assert not left, f"{label}: tasks left behind: {sorted({t.get_coro().__qualname__ for t in left})}"


async def sweep(tmp_path, scenario, durability, kind, **options):
    """One run per write of the scenario, the fault at that write. Returns the failures, one line each."""
    clean = await run_once(tmp_path, scenario, durability)
    await check(clean, f"{scenario}/{durability}/no fault")
    failures = []
    for at in range(clean["writes"][0] + 2):
        label = f"{scenario}/{durability}/{kind} at write {at + 1}"
        try:
            await check(await run_once(tmp_path, scenario, durability, Fault(at, kind, **options)), label)
        except (AssertionError, SessionNotStored) as error:
            failures.append(
                f"{label}: {str(error).splitlines()[0]}" if label not in str(error) else str(error).splitlines()[0]
            )
    return failures, clean["writes"][0] + 2


@pytest.mark.parametrize("durability", ["sync", "end"])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_a_crash_at_any_write_loses_and_repeats_nothing(tmp_path, scenario, durability):
    failures, points = await sweep(tmp_path, scenario, durability, "die")
    assert not failures, f"{len(failures)} of {points} crash points:\n" + "\n".join(failures)


@pytest.mark.parametrize("durability", ["sync", "end"])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_a_shutdown_at_any_write_loses_and_repeats_nothing(tmp_path, scenario, durability):
    failures, points = await sweep(tmp_path, scenario, durability, "close")
    assert not failures, f"{len(failures)} of {points} shutdown points:\n" + "\n".join(failures)


@pytest.mark.parametrize("kind", ["store", "trace"])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_an_outage_at_any_write_loses_and_repeats_nothing(tmp_path, scenario, kind, monkeypatch):
    import timbal.core.chat_session as chat_session_module

    monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.005)
    failures, points = await sweep(tmp_path, scenario, "sync", kind, length=3)
    assert not failures, f"{len(failures)} of {points} outage points:\n" + "\n".join(failures)


@pytest.mark.parametrize("durability", ["sync", "end"])
@pytest.mark.parametrize(
    "scenario", ["plain", "steered", "approved", "questioned", "batch", "queued_behind_an_approval"]
)
async def test_a_second_crash_during_the_recovery_loses_and_repeats_nothing(tmp_path, scenario, durability):
    """The process dies, and the one that restores the session dies too, at each of its first writes."""
    clean = await run_once(tmp_path, scenario, durability)
    await check(clean, f"{scenario}/{durability}/no fault")
    failures, points = [], 0
    for first in range(1, clean["writes"][0] + 1):
        for second in range(4):
            points += 1
            label = f"{scenario}/{durability}/dies at write {first + 1}, then at write {second + 1} of the restore"
            try:
                await check(await run_once(tmp_path, scenario, durability, Fault(first), Fault(second)), label)
            except AssertionError as error:
                failures.append(str(error).splitlines()[0])
    assert not failures, f"{len(failures)} of {points} double crash points:\n" + "\n".join(failures)


@pytest.mark.parametrize("durability", ["sync", "end"])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_a_stop_at_any_write_holds_and_loses_nothing(tmp_path, scenario, durability):
    """The user stops the session at each of its writes, from inside the write: the cut reaches a turn in the
    middle of a save (its last one, when it has just ended on a question), or one that is only starting."""
    failures, points = await sweep(tmp_path, scenario, durability, "stop")
    assert not failures, f"{len(failures)} of {points} stops:\n" + "\n".join(failures)


@pytest.mark.parametrize("durability", ["sync", "end"])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_a_stall_past_the_lease_at_any_write_loses_and_repeats_nothing(
    tmp_path, scenario, durability, monkeypatch
):
    """The process stops running at a write for longer than its lease (seen on loaded CI runners: 70-260 s),
    then runs on: the session finds its lease gone, stops without writing, and another process restores it."""
    Clock.install(monkeypatch)
    failures, points = await sweep(tmp_path, scenario, durability, "stall")
    assert not failures, f"{len(failures)} of {points} stalls:\n" + "\n".join(failures)


@pytest.mark.parametrize("scenario", ["plain", "collected", "queued", "approved", "questioned", "woken"])
async def test_a_stall_while_restoring_after_a_shutdown_loses_and_repeats_nothing(tmp_path, scenario, monkeypatch):
    """The process shuts down at a write, and the one restoring the session stands still past its lease at one
    of its first writes (CI: a shutdown point of `collected` whose restore stalled 263 s, then read as having
    lost 'work'). A save it had sent may land after the next copy loaded the row: that copy is lost too, and
    restored again. Nothing the store holds is lost."""
    Clock.install(monkeypatch)
    clean = await run_once(tmp_path, scenario, "end")
    await check(clean, f"{scenario}/no fault")
    failures, points = [], 0
    for first in range(clean["writes"][0] + 1):
        for second in range(6):
            points += 1
            label = f"{scenario}/close at write {first + 1}, the restore stalls at its write {second + 1}"
            try:
                await check(
                    await run_once(tmp_path, scenario, "end", Fault(first, "close"), Fault(second, "stall")), label
                )
            except (AssertionError, SessionNotStored) as error:
                failures.append(str(error).splitlines()[0])
    assert not failures, f"{len(failures)} of {points} points:\n" + "\n".join(failures)


@pytest.mark.parametrize("end", ["die", "close"])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_a_process_ending_during_a_trace_outage_loses_and_repeats_nothing(tmp_path, scenario, end, monkeypatch):
    """The trace store goes away at a write and the process does not outlive the outage: it dies, or shuts
    down, a few writes into it. A turn that ended meanwhile is held open for its trace all that time, which
    is when its user is most likely to send something, or to stop it."""
    import timbal.core.chat_session as chat_session_module

    monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.005)
    clean = await run_once(tmp_path, scenario, "sync")
    await check(clean, f"{scenario}/no fault")
    failures, points = [], 0
    for at in range(clean["writes"][0]):
        for after in (1, 3):
            points += 1
            label = f"{scenario}/trace away from write {at + 1}, {end} {after} writes later"
            fault = Fault(at, "trace", length=1000, then=(end, after))
            try:
                await check(await run_once(tmp_path, scenario, "sync", fault), label)
            except (AssertionError, SessionNotStored) as error:
                first = str(error).splitlines()[0]
                failures.append(first if label in first else f"{label}: {first}")
    assert not failures, f"{len(failures)} of {points} points:\n" + "\n".join(failures)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_a_takeover_at_any_write_is_never_written_over(tmp_path, scenario):
    """Another process takes the session (its lease, and a write of its own) just before one of this one's
    writes. From then on the row is the other's: this copy stops, and stores nothing more."""
    clean = await run_once(tmp_path, scenario, "sync")
    await check(clean, f"{scenario}/no fault")
    failures, points = [], 0
    for at in range(clean["writes"][0]):
        fault = Fault(at, "taken")
        calls = collections.Counter()
        path = tmp_path / DATABASE
        provider = FaultySqlite.configured(_path=path, fault=fault, durability="sync")
        store = faulty_store(path, fault)
        session = ChatSession(make_agent(provider, calls), session_store=store, collect_window=0.0)
        fault.session = session
        label = f"{scenario}/taken before write {at + 1}"
        try:
            await SCENARIOS[scenario](Client(session, fault, calls))
            if fault.taken is None:
                await session.close(keep=True)
                continue  # the scenario was over before that write: nothing was taken
            points += 1
            await asyncio.wait_for(session.wait_closed(), 5)
            assert session.lost and not session.busy, f"{label}: went on (lost={session.lost}, busy={session.busy})"

            def row(conn, session_id=session.id):
                return conn.execute(
                    "SELECT version, lease_owner FROM chat_sessions WHERE session_id = ?", (session_id,)
                ).fetchone()

            version, owner = await store._run(row)
            assert (version, owner) == (fault.taken, TAKER), f"{label}: wrote over it ({version}, {owner})"
        except (TimeoutError, AssertionError) as error:
            failures.append((str(error).splitlines() or [f"{label}: never closed"])[0])
        finally:
            if not session.closed:
                await session.close(keep=True)
            forget(session)
    assert not failures, f"{len(failures)} of {points} takeover points:\n" + "\n".join(failures)
