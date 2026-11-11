"""Durable chat sessions under faults placed everywhere, not at one hand-picked spot.

A scenario (a few user actions against a durable ``ChatSession``) is run once to count its writes to the
session store and to the trace store, then once more for every write: the process dies at that write (it and
everything after it never land, every task of the session stops), the session is restored the way a new
process would restore it, driven until it is idle, and checked:

- **at most once**: no tool with a side effect ran twice, across the dead process and the new one;
- **nothing acknowledged is lost**: a message, wake or answer whose ``flush()`` returned is in the
  conversation, or is named in a recovery notice;
- **nothing is answered twice**: no session event appears in the history more than once;
- **it settles**: the restored session becomes idle (answering what it still waits for), with nothing left
  pending, and what it stores then matches what it holds.

Further sweeps: the process dies a second time, during the recovery; it shuts down in order at that write
instead of dying; it lives on while the session store, or the trace store, is away for a stretch of writes.

The other crash tests in ``test_chat_session_durable.py`` kill a session after a last successful flush, which
is one point of this space; these find the others.
"""

import asyncio
import collections
import contextlib

import pytest
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.state import background, set_call_id, set_parent_call_id, set_run_context, suspend
from timbal.state.chat_store import SessionNotStored, SqliteChatSessionStore
from timbal.state.tracing.providers import SqliteTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

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


# -- The fault -------------------------------------------------------------------


class Fault:
    """Counts the writes of one session (snapshots, traces, approval claims) and breaks something at write
    number ``at + 1``:

    - ``"die"``: the process dies. That write and all later ones never land; every task of the session stops.
    - ``"close"``: the process shuts down in order from there (``close(keep=True)``); its writes land.
    - ``"store"`` / ``"trace"``: that store fails for ``length`` writes, and the process lives on.
    - ``"taken"``: another process takes the session over just before that write to the session store.
    """

    def __init__(self, at=None, kind="die", *, length=3):
        self.at, self.kind, self.length = at, kind, length
        self.writes = 0
        self.gone = asyncio.Event()  # the process died, or is shutting down: the caller stops
        self.dead = False
        self.closing: asyncio.Future | None = None
        self.taken: int | None = None  # the version the other process left the row at
        self.session = None

    def write(self) -> bool:
        """Count a write. False: the process is dead (the caller stops with its task)."""
        if self.dead:
            return False
        self.writes += 1
        if self.at is not None and self.writes > self.at and not self.gone.is_set():
            if self.kind == "die":
                self.die()
                return False
            if self.kind == "close":
                self.gone.set()
                self.closing = asyncio.ensure_future(self.session.close(keep=True))
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


def faulty_store(path, fault):
    store = SqliteChatSessionStore(path)
    real_save, real_acquire, real_release = store.save, store.acquire_lease, store.release_lease

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

    store.save, store.acquire_lease, store.release_lease = save, acquire_lease, release_lease
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
    """Close this run's database and delete it: the provider keeps one connection per file for the life of
    the process, and a sweep opens hundreds of files."""
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

    async def send(self, text, mode=None) -> None:
        if not self._alive():
            return
        with contextlib.suppress(RuntimeError):  # closed under us
            event = self.session.send(text, mode=mode, defer_cut=True)
            mark = self.session.mark()
            if await self.stored() or (self._alive() and self.session.stored_since(mark)):
                self.acked[event.id] = text
                self.session.cut_for(event, mode)
            elif self._alive():
                self.session.withdraw(event.id)  # as the route does before its 503

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

    async def until(self, condition, timeout=10.0) -> bool:
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
        )

    async def answer(self) -> bool:
        """Answer everything the parked turn waits for (approve, or "yes"). False when there is nothing."""
        state = self.session.state()
        values = {a["approval_id"]: True for a in state.get("pending_approvals", [])}
        values.update({i["interaction_id"]: "yes" for i in state.get("pending_interactions", [])})
        if not values or not self._alive():
            return False
        with contextlib.suppress(RuntimeError, ValueError):
            self.session.resume(values)
            mark = self.session.mark()
            if not (await self.stored() or not self._alive() or self.session.stored_since(mark)):
                self.session.withdraw_answers(values)
        return True

    async def settle(self, timeout=10.0) -> None:
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
        await c.send("never mind", mode="interrupt")
    await c.settle()


async def wake_replaced(c):
    await c.send("work")
    await c.wake(WAKE_TEXT, key="k")
    await c.wake(WAKE_TEXT + " again", key="k")
    await c.settle()


SCENARIOS = {
    "plain": plain,
    "two_steps": two_steps,
    "queued": queued,
    "steered": steered,
    "approved": approved,
    "questioned": questioned,
    "woken": woken,
    "interrupted": interrupted,
    "cut_now": cut_now,
    "batch": batch,
    "delegated": delegated,
    "slash": slash,
    "collected": collected,
    "queued_behind_an_approval": queued_behind_an_approval,
    "approval_abandoned": approval_abandoned,
    "wake_replaced": wake_replaced,
}


# -- One run, and what must hold after it ---------------------------------------------


async def history(session):
    context = await session._load_run(session.head)
    root = context.root_span() if context is not None else None
    return [
        m for m in (root.memory if root is not None and isinstance(root.memory, list) else []) if isinstance(m, dict)
    ]


def texts(message):
    return " ".join(b.get("text") or "" for b in message.get("content") or [] if isinstance(b, dict))


_RUNS = iter(range(10**9))


async def run_once(tmp_path, scenario, durability, *faults):
    """Run ``scenario`` under the first fault; each time the process is gone, restore it under the next one
    (none left: nothing fails any more); settle; return what to check."""
    before = asyncio.all_tasks()
    calls = collections.Counter()
    path = tmp_path / f"{scenario}-{durability}-{next(_RUNS)}.db"
    faults = list(faults) or [Fault()]
    fault = faults.pop(0)
    provider = FaultySqlite.configured(_path=path, fault=fault, durability=durability)
    store = faulty_store(path, fault)
    session = ChatSession(make_agent(provider, calls), session_store=store, collect_window=0.0)
    fault.session = session
    client = Client(session, fault, calls)
    await SCENARIOS[scenario](client)
    writes = [fault.writes]
    while fault.gone.is_set():
        if fault.dead:
            await bury(session)
        else:
            await fault.closing
            forget(session)
        acked, acked_wakes = client.acked, client.acked_wakes
        fault = faults.pop(0) if faults else Fault()
        provider = FaultySqlite.configured(_path=path, fault=fault, durability=durability)
        store = faulty_store(path, fault)
        try:
            session = await ChatSession.from_store(make_agent(provider, calls), store, session.id)
        except LookupError:
            # It died before the session itself was stored: nothing can have been acknowledged.
            assert not acked and not acked_wakes, "acknowledged, but the session was never stored"
            return {"writes": writes, "calls": calls, "session": None, "before": before, "path": path}
        fault.session = session
        client = Client(session, fault, calls)
        client.acked, client.acked_wakes = acked, acked_wakes
        await client.settle()
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
    }


async def check(result, label):
    try:
        await _check(result, label)
    finally:
        session = result["session"]
        if session is not None and not session.closed:
            await session.close(keep=True)  # a failed check: nothing of it may leak into the next run
        release(result["path"])


async def _check(result, label):
    calls, session = result["calls"], result["session"]
    for name in SIDE_EFFECTS:
        assert calls[name] <= 1, f"{label}: {name} ran {calls[name]} times"
    if session is None:
        return
    client, store = result["client"], result["store"]
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
    assert client.at_rest() and not session.parked, f"{label}: not settled: {session.state()}"
    await session.flush()
    stored = await store.load(session.id)
    assert stored.in_flight is None and stored.pending == [] and stored.wakes == [], f"{label}: {stored}"
    assert stored.head == session.head and not stored.parked, f"{label}: stored head/parked differ"
    turns, head = session.turns, session.head
    await session.close(keep=True)
    forget(session)
    # Restoring it once more changes nothing: no turn starts, nothing comes back.
    again = await ChatSession.from_store(session._agent, store, session.id)
    await asyncio.sleep(0.01)
    assert (again.turns, again.head, again.busy) == (turns, head, False), f"{label}: a second restore ran something"
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
        path = tmp_path / f"{scenario}-taken-{at}.db"
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
            release(path)
    assert not failures, f"{len(failures)} of {points} takeover points:\n" + "\n".join(failures)
