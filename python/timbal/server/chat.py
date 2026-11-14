"""Chat sessions over HTTP (:class:`~timbal.core.chat_session.ChatSession`), and the chat playground.

``GET /chat/`` serves ``chat.html``. Sessions are served by one process at a
time. They are durable when the agent stores its traces in SQLite (the
sessions go in the same file) or ``TIMBAL_CHAT_STORE`` names a store
(``sqlite:///path/to.db``; ``memory`` turns it off): a session this process
does not hold is restored on first use, those with work waiting at startup,
and shutdown keeps them for the next process. Otherwise a request for a
session another worker (or an earlier process) held is a 404.
"""

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..core.chat_session import SEND_MODES, ChatSession
from ..state.background import BACKGROUND_LOG_GAPPED
from ..state.chat_store import ChatSessionStore, SessionLeased, SessionNotStored, SqliteChatSessionStore

logger = structlog.get_logger("timbal.server.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

_HTML_PATH = Path(__file__).parent / "chat.html"

SESSION_TTL_ENV = "TIMBAL_CHAT_SESSION_TTL_S"
STORE_ENV = "TIMBAL_CHAT_STORE"
DEFAULT_SESSION_TTL_S = 3600.0
# An SSE comment this often keeps proxies from closing a quiet stream.
KEEPALIVE_S = 15.0
# How long a shutdown waits for a session restore already going before cancelling it.
_SHUTDOWN_RESTORE_S = 5.0
# How long startup waits for the store to list its sessions, and when it looks again after it could not.
_STARTUP_DISCOVERY_S = 5.0
_DISCOVERY_RETRY_S: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 30.0)


def _session_ttl() -> float:
    raw = os.environ.get(SESSION_TTL_ENV)
    try:
        return float(raw) if raw else DEFAULT_SESSION_TTL_S
    except ValueError:
        return DEFAULT_SESSION_TTL_S


def _sessions(request: Request) -> dict[str, ChatSession]:
    sessions = getattr(request.app.state, "chat_sessions", None)
    if sessions is None:
        sessions = request.app.state.chat_sessions = {}
    return sessions


def session_store_for(agent: Any) -> ChatSessionStore | None:
    """Where this server keeps its sessions: ``TIMBAL_CHAT_STORE``, else next to the agent's SQLite traces."""
    from ..state.tracing.providers import SqliteTracingProvider

    raw = (os.environ.get(STORE_ENV) or "").strip()
    if raw.lower() in ("memory", "none", "off"):
        return None
    if raw:
        # SQLAlchemy's form: sqlite:///relative.db, sqlite:////absolute/path.db
        if not raw.startswith("sqlite:///") or len(raw) == len("sqlite:///"):
            raise RuntimeError(f"{STORE_ENV} must be 'memory' or 'sqlite:///path/to.db', got {raw!r}")
        _require_durable_traces(agent)
        return SqliteChatSessionStore(raw[len("sqlite:///") :])
    provider = getattr(agent, "tracing_provider", None)
    if isinstance(provider, type) and issubclass(provider, SqliteTracingProvider) and provider._path is not None:
        return SqliteChatSessionStore(provider._path, synchronous=provider.synchronous)
    return None


def _require_durable_traces(agent: Any) -> None:
    """A session store that outlives the process is of no use with traces that do not.

    A restored session reads its conversation — and the approval a parked one waits on — from the trace of
    its last run. With process-local traces a restart leaves the snapshot pointing at runs nobody can read:
    the session comes back durable, with no history and a question that cannot be answered. Said here,
    before any work is accepted, instead of being found after the restart.
    """
    from ..state.context import RunContext

    # The agent's setting as runs resolve it (TRACING_UNSET: auto-detected from the platform config).
    provider = RunContext(tracing_provider=getattr(agent, "tracing_provider", None))._tracing_provider
    if provider is not None and getattr(provider, "durable", True):
        return
    name = provider.__name__ if provider is not None else "no tracing provider"
    raise RuntimeError(
        f"{STORE_ENV} keeps chat sessions across restarts, but this agent's traces do not ({name}): a "
        f"restored session would have no history and no pending approval to answer. Give the agent a durable "
        f"tracing_provider (SqliteTracingProvider, JsonlTracingProvider, PlatformTracingProvider), or set "
        f"{STORE_ENV}=memory."
    )


def _store(request: Request) -> ChatSessionStore | None:
    state = request.app.state
    if not hasattr(state, "chat_store"):
        runnable = getattr(state, "runnable", None)
        state.chat_store = session_store_for(runnable) if runnable is not None else None
    return state.chat_store


async def _session(request: Request, session_id: str, *, allow_closed: bool = False) -> ChatSession:
    sessions = _sessions(request)
    session = sessions.get(session_id)
    if session is not None and session.handed_off:
        # Lost its lease (another process serves it: 409) or let go with keep: the store says where it is now,
        # once this copy is done closing (until then its turn may run, its task store still be bound).
        await session.wait_closed()
        if sessions.get(session_id) is session:
            sessions.pop(session_id)
        session = None
    if session is None:
        session = await _restore(request, session_id)
    if session is None or (session.closed and not allow_closed):
        raise HTTPException(status_code=404, detail=f"No chat session {session_id!r}.")
    return session


async def _restore(request: Request, session_id: str) -> ChatSession | None:
    """A durable session this process does not hold, restored (once, however many requests ask at the same time)."""
    store = _store(request)
    if store is None:
        return None
    agent = _agent(request)
    try:
        return await _restore_once(request.app, agent, store, session_id)
    except LookupError:
        return None
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except SessionLeased as e:
        raise HTTPException(status_code=409, detail=f"Chat session {session_id!r} is served by another process.") from e
    except Exception as e:
        # The store or the run's trace could not be read: nothing was taken over, the stored state stays.
        logger.error("chat_session_restore_failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=503, detail=f"Chat session {session_id!r} could not be restored: {e}", headers=_RETRY
        ) from e


def _app_sessions(app: Any) -> dict[str, ChatSession]:
    sessions = getattr(app.state, "chat_sessions", None)
    if sessions is None:
        sessions = app.state.chat_sessions = {}
    return sessions


async def _restore_once(app: Any, agent: Any, store: ChatSessionStore, session_id: str) -> ChatSession:
    """The session restored from the store into this process's map, once per id: requests and the startup
    retry share one restore, and a session the process already serves is returned as is (this process can
    always renew its own lease, so a second ``from_store`` would make a second copy)."""
    state = app.state
    current = _app_sessions(app).get(session_id)
    if current is not None and not current.handed_off:
        return current
    if getattr(state, "chat_closing", False):
        raise RuntimeError("The server is shutting down.")
    restoring: dict[str, asyncio.Task] = getattr(state, "chat_restoring", None) or {}
    state.chat_restoring = restoring
    task = restoring.get(session_id)
    if task is None:
        task = restoring[session_id] = asyncio.ensure_future(_restore_into(app, agent, store, session_id))
        task.add_done_callback(lambda done: _restore_done(restoring, session_id, done))
    # Shielded: a caller that goes away (a dropped request, the retry cancelled at shutdown) does not stop it.
    return await asyncio.shield(task)


async def _restore_into(app: Any, agent: Any, store: ChatSessionStore, session_id: str) -> ChatSession:
    """One restore, whoever still waits for it: the session is in the map before this returns, so shutdown
    (which waits for the restores in flight) finds it and closes it with the others."""
    session = await ChatSession.from_store(agent, store, session_id)
    _app_sessions(app)[session_id] = session
    return session


def _restore_done(restoring: dict[str, asyncio.Task], session_id: str, task: asyncio.Task) -> None:
    if restoring.get(session_id) is task:
        del restoring[session_id]
    if not task.cancelled():
        task.exception()  # retrieved: every caller may have gone away


async def restore_chat_sessions(app: Any) -> list[str]:
    """Server startup: bring back the durable sessions with work waiting (a queued message, a wake, a turn
    that died with the last process), so they go on without waiting for a client. Returns their ids.

    Best effort, and never the reason a server does not start: the store is implied by the agent's SQLite
    traces, so a deployment that only serves ``/run`` and ``/stream`` gets here too. A store that cannot be
    read now (a mount that is not there, a file that is locked) is looked at again in the background until
    it answers; meanwhile every other route works, and chat requests fail safely on their own (``503``).
    """
    from ..core.agent import Agent

    app.state.chat_closing = False
    runnable = getattr(app.state, "runnable", None)
    if not isinstance(runnable, Agent):
        return []
    store = app.state.chat_store = session_store_for(runnable)
    if store is None:
        return []
    _app_sessions(app)  # the map exists from here on, whether or not the store can be read yet
    try:
        # Bounded: a store that hangs (a stalled network mount) must not hold the server's start either.
        session_ids = await asyncio.wait_for(store.list_ids(with_work=True), _STARTUP_DISCOVERY_S)
    except Exception as e:
        logger.error(
            "chat_sessions_discovery_failed",
            error=str(e) or type(e).__name__,
            detail="The server starts without its stored chat sessions; looking again in the background.",
        )
        app.state.chat_discovery = asyncio.get_running_loop().create_task(_discover_later(app, runnable, store))
        return []
    return await _restore_discovered(app, runnable, store, session_ids)


async def _discover_later(app: Any, agent: Any, store: ChatSessionStore) -> None:
    """Startup could not list the stored sessions: look again (1 s, 2 s, 5 s, 10 s, then every 30 s) until the
    store answers, then bring back those with work waiting, as startup would have."""
    delays = iter(_DISCOVERY_RETRY_S)
    delay = _DISCOVERY_RETRY_S[0]
    said: str | None = None
    while not getattr(app.state, "chat_closing", False):
        delay = next(delays, delay)
        await asyncio.sleep(delay)
        try:
            session_ids = await store.list_ids(with_work=True)
        except Exception as e:
            error = str(e) or type(e).__name__
            if error != said:  # once per cause, not once per attempt for as long as the store is away
                said = error
                logger.warning("chat_sessions_discovery_failed", error=error)
            continue
        restored = await _restore_discovered(app, agent, store, session_ids)
        logger.info("chat_sessions_discovered_late", count=len(restored))
        return


async def _restore_discovered(app: Any, runnable: Any, store: ChatSessionStore, session_ids: list[str]) -> list[str]:
    sessions = _app_sessions(app)
    restored, skipped = [], []
    for session_id in session_ids:
        if session_id in sessions:
            continue
        if getattr(app.state, "chat_closing", False):
            break
        try:
            await _restore_once(app, runnable, store, session_id)
            restored.append(session_id)
        except SessionLeased:
            skipped.append(session_id)  # another worker serves it, or a predecessor's lease has not run out yet
        except (LookupError, ValueError):
            continue
        except Exception as e:
            logger.error("chat_session_restore_failed", session_id=session_id, error=str(e))
            skipped.append(session_id)
    if restored:
        logger.info("chat_sessions_restored", count=len(restored))
    if skipped and not getattr(app.state, "chat_closing", False):
        # A replacement can start before the old container's lease runs out (and a new hostname, or a reused
        # pid, makes its holder look alive): try those again, or their queued work waits for a client.
        app.state.chat_restore_retry = asyncio.get_running_loop().create_task(
            _retry_restores(app, runnable, store, skipped)
        )
    return restored


async def _retry_restores(app: Any, agent: Any, store: ChatSessionStore, session_ids: list[str]) -> None:
    """Retry restoring sessions skipped at startup, every third of a lease, while they still have work waiting
    and nobody here holds them. A live worker that serves one keeps its lease, and clears the work."""
    waiting = set(session_ids)
    interval = max(store.lease_ttl_s / 3, 0.05)
    while waiting:
        await asyncio.sleep(interval)
        sessions = getattr(app.state, "chat_sessions", None) or {}
        for session_id in sorted(waiting):
            if session_id in sessions:
                waiting.discard(session_id)
                continue
            try:
                snapshot = await store.load(session_id)
            except Exception as e:
                logger.warning("chat_session_restore_retry_failed", session_id=session_id, error=str(e))
                continue
            if snapshot is None or not snapshot.has_work():
                waiting.discard(session_id)
                continue
            try:
                await _restore_once(app, agent, store, session_id)
            except SessionLeased:
                continue
            except (LookupError, ValueError):
                waiting.discard(session_id)
            except Exception as e:
                logger.warning("chat_session_restore_retry_failed", session_id=session_id, error=str(e))
            else:
                waiting.discard(session_id)
                logger.info("chat_session_restored_late", session_id=session_id)


def _agent(request: Request) -> Any:
    from ..core.agent import Agent

    runnable = request.app.state.runnable
    if not isinstance(runnable, Agent):
        raise HTTPException(status_code=400, detail="Chat sessions need the served runnable to be an Agent.")
    return runnable


async def _reap(sessions: dict[str, ChatSession]) -> None:
    """Let go of sessions idle for longer than the TTL (never one with a turn running or due to start one).
    A durable one is kept in its store: the next request for it restores it."""
    ttl = _session_ttl()
    if ttl <= 0:
        return
    now = time.monotonic()
    for session_id, session in list(sessions.items()):
        # `due`, not just `busy`: a session collecting a message (or waiting out a backoff) is neither busy
        # nor idle. Let go of it and nothing runs that message until a request asks for the session again,
        # which for an acknowledged message may be never.
        if session.closed or (not session.busy and not session.due and now - session.last_activity > ttl):
            # Closed before it leaves the map: a request meanwhile finds it handed off and waits, instead of
            # restoring a second copy while this one still holds the session's task store.
            await session.close(keep=True)
            await session.wait_closed()
            if sessions.get(session_id) is session:
                sessions.pop(session_id)


async def close_chat_sessions(app: Any) -> None:
    """Close every session (server shutdown). Durable ones are kept for the next process, a turn this cuts
    recovered there."""
    app.state.chat_closing = True  # no restore starts from here on
    for name in ("chat_discovery", "chat_restore_retry"):
        task = getattr(app.state, name, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
    # A restore already going is shielded from its callers: left alone, it would finish after this and its
    # session keep running, unclosed, through the shutdown. Wait for it (its session is then in the map,
    # closed below); one that takes too long is cancelled, which lets go of its lease and writes nothing.
    restoring = [task for task in (getattr(app.state, "chat_restoring", None) or {}).values() if not task.done()]
    if restoring:
        _, slow = await asyncio.wait(restoring, timeout=_SHUTDOWN_RESTORE_S)
        for task in slow:
            task.cancel()
        if slow:
            await asyncio.gather(*slow, return_exceptions=True)
    sessions = getattr(app.state, "chat_sessions", None) or {}
    for session in list(sessions.values()):
        with contextlib.suppress(Exception):
            await session.close(keep=True)
    sessions.clear()


_RETRY = {"Retry-After": "1"}


_TAKEN = "The request was taken (a turn has it), but it would not survive a restart: do not resend."


async def _stored(session: ChatSession, mark: int, undo: Any = None, *, taken: str = _TAKEN) -> None:
    """``session.flush()`` for a route about to acknowledge the change it made at ``mark`` (``session.mark()``,
    taken right after it). A session another process took is a ``409``. When the store fails:

    - the change itself is stored and a later one is what failed: returns, the acknowledgement holds;
    - the store holds nothing of it and ``undo`` takes it back here too (returns True: no turn has taken it):
      ``503`` with ``Retry-After``. Nothing was accepted, in memory or in the store, so a retry cannot run
      it twice, whether or not this process lives to store anything again;
    - a turn has it: ``503`` without ``Retry-After`` (``taken``)."""
    try:
        await session.flush()
    except SessionLeased as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except SessionNotStored as e:
        if session.stored_since(mark):
            return
        if undo is not None and undo():
            raise HTTPException(status_code=503, detail=f"{e}. Nothing was accepted: retry.", headers=_RETRY) from e
        raise HTTPException(status_code=503, detail=f"{e}. {taken}") from e


def _jsonable(item: Any) -> Any:
    dump = getattr(item, "model_dump", None)
    return dump(mode="json") if dump is not None else item


def _json(content: Any, status_code: int = 200) -> JSONResponse:
    # Approval inputs and tool payloads can hold values json cannot encode as is.
    return JSONResponse(status_code=status_code, content=json.loads(json.dumps(content, default=str)))


async def _body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="The body must be JSON.") from e
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="The body must be a JSON object.")
    return body


@router.get("/")
async def chat_page() -> HTMLResponse:
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/meta")
async def chat_meta(request: Request) -> JSONResponse:
    from ..core.agent import Agent

    runnable = request.app.state.runnable
    is_agent = isinstance(runnable, Agent)
    model = getattr(runnable, "model", None) if is_agent else None
    return _json(
        {
            "agent": getattr(runnable, "name", None),
            "is_agent": is_agent,
            "model": model if isinstance(model, str) else type(model).__name__ if model is not None else None,
            "steering": runnable.steering if is_agent else None,
            "modes": list(SEND_MODES),
            "busy_mode": "steer",
            "durable": is_agent and _store(request) is not None,
        }
    )


@router.post("/sessions")
async def create_session(request: Request) -> JSONResponse:
    """Open a session: ``{"parent_id"?, "busy_mode"?, "collect_window"?}``."""
    agent = _agent(request)
    body = await _body(request)
    sessions = _sessions(request)
    await _reap(sessions)
    try:
        session = ChatSession(
            agent,
            parent_id=body.get("parent_id"),
            busy_mode=body.get("busy_mode") or "steer",
            collect_window=float(body.get("collect_window", 1.0)),
            session_store=_store(request),
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    try:
        await session.restore()
    except Exception as e:
        await session.close()
        raise HTTPException(status_code=503, detail=f"Could not load the run the session continues: {e}") from e
    session.persist()
    mark = session.mark()
    try:
        await session.flush()
    except (SessionNotStored, SessionLeased) as e:
        if isinstance(e, SessionLeased) or not session.stored_since(mark):
            await session.close()  # never acknowledged: nothing of it stays
            raise HTTPException(status_code=503, detail=f"{e}. No session was opened: retry.", headers=_RETRY) from e
    sessions[session.id] = session
    return _json(session.state(), status_code=201)


@router.get("/sessions")
async def list_sessions(request: Request) -> JSONResponse:
    return _json([session.state() for session in _sessions(request).values() if not session.closed])


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> JSONResponse:
    return _json((await _session(request, session_id)).state())


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> JSONResponse:
    session = await _session(request, session_id)
    sessions = _sessions(request)
    try:
        # In the map until it is closed and gone from the store: a request meanwhile finds it closed (404),
        # not a gap through which to restore the row that is still there.
        await session.close()
        await session.wait_closed()  # a close already going (ownership lost) is this one: wait for its end
    finally:
        if sessions.get(session_id) is session:
            del sessions[session_id]
    if session.lost:
        # Another process took it before this copy noticed: nothing was deleted, it goes on there.
        raise HTTPException(status_code=409, detail=f"Chat session {session_id!r} is served by another process.")
    if session.close_error is not None:
        # Closed here, but the store still holds it as open: the next request would restore it. A retry
        # restores it and deletes again.
        raise HTTPException(
            status_code=503,
            detail=f"Chat session {session_id!r} was closed but not removed from the store: {session.close_error}",
            headers=_RETRY,
        )
    return _json({"closed": True})


@router.post("/sessions/{session_id}/messages")
async def send_message(request: Request, session_id: str) -> JSONResponse:
    """Send a user message: ``{"text", "mode"?}`` (``steer`` | ``followup`` | ``collect`` | ``interrupt``)."""
    session = await _session(request, session_id)
    body = await _body(request)
    text = body.get("text")
    mode = body.get("mode") or None
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'text' must be a string.")
    busy = session.busy
    try:
        # The message is only queued here: 'now' and 'interrupt' cut the running turn once it is stored, so a
        # 503 that says nothing was accepted has not cut anything either.
        event = session.send(text, mode=mode, defer_cut=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    mark = session.mark()
    await _stored(session, mark, lambda: session.withdraw(event.id))  # the 202 means it survives a restart
    session.cut_for(event, mode)
    return _json(
        {"message": {**event.receipt(), "text": event.text}, "mode": mode or session.busy_mode, "busy": busy},
        status_code=202,
    )


@router.post("/sessions/{session_id}/wake")
async def wake_session(request: Request, session_id: str) -> JSONResponse:
    """Start a turn for the application: ``{"text", "key"?, "tool_policy"?, "silent_ok"?}``.

    ``tool_policy`` defaults to ``"read_only"``; ``null`` gives the turn every
    tool, a list of names only those. The wake runs once the user has nothing
    pending, as a turn of its own, and may stay silent.
    """
    session = await _session(request, session_id)
    body = await _body(request)
    text = body.get("text")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'text' must be a string.")
    if body.get("key") is not None and not isinstance(body["key"], str):
        raise HTTPException(status_code=400, detail="'key' must be a string or null.")
    policy = body.get("tool_policy", "read_only")
    if policy is not None and not (policy == "read_only" or isinstance(policy, list)):
        raise HTTPException(status_code=400, detail="'tool_policy' must be 'read_only', a list of tool names, or null.")
    try:
        wake_id = session.wake(
            text, key=body.get("key"), tool_policy=policy, silent_ok=bool(body.get("silent_ok", True))
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    await _stored(session, session.mark(), lambda: session.withdraw(wake_id))
    return _json({"wake_id": wake_id, "busy": session.busy}, status_code=202)


@router.post("/sessions/{session_id}/interrupt")
async def interrupt_session(request: Request, session_id: str) -> JSONResponse:
    session = await _session(request, session_id)
    interrupted = session.interrupt()
    if interrupted:
        await _stored(
            session,
            session.mark(),
            taken="The session was interrupted, but its state could not be stored: it would not survive a restart.",
        )
    return _json({"interrupted": interrupted})


@router.post("/sessions/{session_id}/resume")
async def resume_session(request: Request, session_id: str) -> JSONResponse:
    """Answer a parked turn: ``{"resume": {approval_id: true | {...}, interaction_id: value}}``."""
    session = await _session(request, session_id)
    body = await _body(request)
    values = body.get("resume")
    if not isinstance(values, dict) or not values:
        raise HTTPException(status_code=400, detail="'resume' must be a non-empty object.")
    try:
        session.resume(values)
    except TypeError as e:
        # An answer the session store cannot hold (the body parser lets NaN and Infinity through).
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (RuntimeError, ValueError) as e:
        # Not waiting for an answer (yet), or an id already answered differently.
        raise HTTPException(status_code=409, detail=str(e)) from e
    await _stored(session, session.mark(), lambda: session.withdraw_answers(values))
    return _json({"resumed": True}, status_code=202)


@router.get("/sessions/{session_id}/log")
async def session_log(
    request: Request, session_id: str, after: int = 0, limit: int = 500, wait_ms: int = 0
) -> JSONResponse:
    """The session's log as JSON pages, for clients without SSE (mirrors ``/runs/{run_id}/events``).

    ``after`` is the last ``seq`` seen; ``wait_ms`` (up to 30000) long-polls
    while nothing is past it. ``gapped`` means entries past ``after`` were
    dropped: continue from ``next_cursor``. ``closed``: the session ended.
    """
    session = await _session(request, session_id, allow_closed=True)
    log = session.log
    limit = max(1, min(limit, 2000))
    if wait_ms > 0 and log.cursor_end <= after and not session.closed:
        await log.wait(after, min(wait_ms, 30_000) / 1000)
    events, next_cursor, gapped = log.read(after, limit)
    if gapped:
        events, next_cursor, _ = log.read(log.forgotten_through, limit)
    first = next_cursor - len(events) + 1
    return _json(
        {
            "session_id": session_id,
            "events": [{"seq": first + i, "data": _jsonable(item)} for i, item in enumerate(events)],
            "next_cursor": next_cursor,
            "gapped": gapped,
            "closed": session.closed,
        }
    )


@router.get("/sessions/{session_id}/events")
async def session_events(request: Request, session_id: str, after: int = 0) -> StreamingResponse:
    """The session's log as SSE, from cursor ``after`` (or ``Last-Event-ID``), live until it closes.

    Each ``data:`` is one event of a turn, or a ``{"type": "SESSION", ...}``
    record. ``id:`` is the cursor to reconnect with. ``event: gapped`` means
    the ring dropped entries past the cursor; the stream goes on from the
    oldest one it still holds.
    """
    session = await _session(request, session_id, allow_closed=True)
    last_event_id = request.headers.get("last-event-id")
    if last_event_id and last_event_id.isdigit():
        after = max(after, int(last_event_id))

    async def stream() -> AsyncGenerator[str, None]:
        items = session.events(after, numbered=True).__aiter__()
        pending: asyncio.Task | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(items.__anext__())
                done, _ = await asyncio.wait({pending}, timeout=KEEPALIVE_S)
                if not done:
                    yield ": keepalive\n\n"
                    continue
                task, pending = pending, None
                try:
                    seq, item = task.result()
                except StopAsyncIteration:
                    return
                if item is BACKGROUND_LOG_GAPPED:
                    yield f"event: gapped\ndata: {json.dumps({'next_cursor': seq})}\n\n"
                    continue
                yield f"id: {seq}\ndata: {json.dumps(_jsonable(item), default=str)}\n\n"
        finally:
            if pending is not None:
                pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending
            with contextlib.suppress(Exception):
                await items.aclose()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
