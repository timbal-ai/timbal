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
from ..state.chat_store import ChatSessionStore, SessionLeased, SqliteChatSessionStore

logger = structlog.get_logger("timbal.server.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

_HTML_PATH = Path(__file__).parent / "chat.html"

SESSION_TTL_ENV = "TIMBAL_CHAT_SESSION_TTL_S"
STORE_ENV = "TIMBAL_CHAT_STORE"
DEFAULT_SESSION_TTL_S = 3600.0
# An SSE comment this often keeps proxies from closing a quiet stream.
KEEPALIVE_S = 15.0


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
    if raw.lower() == "platform":
        from ..state.platform_chat_store import PlatformChatSessionStore

        return PlatformChatSessionStore()
    if raw:
        # SQLAlchemy's form: sqlite:///relative.db, sqlite:////absolute/path.db
        if not raw.startswith("sqlite:///") or len(raw) == len("sqlite:///"):
            raise RuntimeError(f"{STORE_ENV} must be 'memory', 'platform' or 'sqlite:///path/to.db', got {raw!r}")
        return SqliteChatSessionStore(raw[len("sqlite:///") :])
    provider = getattr(agent, "tracing_provider", None)
    if isinstance(provider, type) and issubclass(provider, SqliteTracingProvider) and provider._path is not None:
        return SqliteChatSessionStore(provider._path, synchronous=provider.synchronous)
    return None


def _store(request: Request) -> ChatSessionStore | None:
    state = request.app.state
    if not hasattr(state, "chat_store"):
        runnable = getattr(state, "runnable", None)
        state.chat_store = session_store_for(runnable) if runnable is not None else None
    return state.chat_store


async def _session(request: Request, session_id: str, *, allow_closed: bool = False) -> ChatSession:
    sessions = _sessions(request)
    session = sessions.get(session_id)
    if session is None:
        session = await _restore(request, session_id)
    if session is None or (session.closed and not allow_closed):
        raise HTTPException(status_code=404, detail=f"No chat session {session_id!r}.")
    return session


class _HeldElsewhere(HTTPException):
    """Another process holds the session's lease."""

    def __init__(self, session_id: str) -> None:
        super().__init__(status_code=409, detail=f"Chat session {session_id!r} is served by another process.")


async def _owned_or_queued(
    request: Request, session_id: str, kind: str, item: dict[str, Any]
) -> ChatSession | JSONResponse:
    """The session, or, when another process serves it and the store has an inbox, ``item`` queued for that
    process (``202 {"queued": "inbox", "inbox_id"}``: it applies it within moments, stored before the ack)."""
    try:
        return await _session(request, session_id)
    except _HeldElsewhere:
        store = _store(request)
        if store is None or not store.supports_inbox:
            raise
        try:
            inbox_id = await store.post_inbox(session_id, kind, item)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=f"No chat session {session_id!r}.") from e
        return _json({"queued": "inbox", "inbox_id": inbox_id, "session_id": session_id}, status_code=202)


async def _restore(request: Request, session_id: str) -> ChatSession | None:
    """A durable session this process does not hold, restored (once, however many requests ask at the same time)."""
    store = _store(request)
    if store is None:
        return None
    state = request.app.state
    restoring: dict[str, asyncio.Task] = getattr(state, "chat_restoring", None) or {}
    state.chat_restoring = restoring
    task = restoring.get(session_id)
    if task is None:
        task = restoring[session_id] = asyncio.ensure_future(ChatSession.from_store(_agent(request), store, session_id))
        task.add_done_callback(lambda _: restoring.pop(session_id, None))
    try:
        session = await asyncio.shield(task)
    except LookupError:
        return None
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except SessionLeased as e:
        raise _HeldElsewhere(session_id) from e
    _sessions(request).setdefault(session_id, session)
    return _sessions(request)[session_id]


async def restore_chat_sessions(app: Any) -> list[str]:
    """Server startup: bring back the durable sessions with work waiting (a queued message, a wake, a turn
    that died with the last process), so they go on without waiting for a client. Returns their ids."""
    from ..core.agent import Agent

    runnable = getattr(app.state, "runnable", None)
    if not isinstance(runnable, Agent):
        return []
    store = app.state.chat_store = session_store_for(runnable)
    if store is None:
        return []
    sessions = getattr(app.state, "chat_sessions", None)
    if sessions is None:
        sessions = app.state.chat_sessions = {}
    restored = []
    for session_id in await store.list_ids(with_work=True):
        if session_id in sessions:
            continue
        try:
            sessions[session_id] = await ChatSession.from_store(runnable, store, session_id)
            restored.append(session_id)
        except SessionLeased:
            continue  # another live worker serves it
        except Exception as e:
            logger.error("chat_session_restore_failed", session_id=session_id, error=str(e))
    if restored:
        logger.info("chat_sessions_restored", count=len(restored))
    return restored


def _agent(request: Request) -> Any:
    from ..core.agent import Agent

    runnable = request.app.state.runnable
    if not isinstance(runnable, Agent):
        raise HTTPException(status_code=400, detail="Chat sessions need the served runnable to be an Agent.")
    return runnable


async def _reap(sessions: dict[str, ChatSession]) -> None:
    """Let go of sessions idle for longer than the TTL (never one with a turn running). A durable one is kept
    in its store: the next request for it restores it."""
    ttl = _session_ttl()
    if ttl <= 0:
        return
    now = time.monotonic()
    for session_id, session in list(sessions.items()):
        if session.closed or (not session.busy and now - session.last_activity > ttl):
            sessions.pop(session_id, None)
            await session.close(keep=True)


async def close_chat_sessions(app: Any) -> None:
    """Close every session (server shutdown). Durable ones are kept for the next process, a turn this cuts
    recovered there."""
    sessions = getattr(app.state, "chat_sessions", None) or {}
    for session in list(sessions.values()):
        with contextlib.suppress(Exception):
            await session.close(keep=True)
    sessions.clear()


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
    sessions[session.id] = session
    session.persist()
    await session.flush()
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
    _sessions(request).pop(session_id, None)
    await session.close()
    return _json({"closed": True})


@router.post("/sessions/{session_id}/messages")
async def send_message(request: Request, session_id: str) -> JSONResponse:
    """Send a user message: ``{"text", "mode"?}`` (``steer`` | ``followup`` | ``collect`` | ``interrupt``)."""
    body = await _body(request)
    text = body.get("text")
    mode = body.get("mode") or None
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'text' must be a string.")
    if mode is not None and mode not in SEND_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {list(SEND_MODES)}, got {mode!r}")
    session = await _owned_or_queued(request, session_id, "message", {"text": text, "mode": mode})
    if isinstance(session, JSONResponse):
        return session
    busy = session.busy
    try:
        event = session.send(text, mode=mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    await session.flush()  # durable: the 202 means it survives a restart
    return _json(
        {"message": {**event.receipt(), "text": event.text}, "mode": mode or session.busy_mode, "busy": busy},
        status_code=202,
    )


_REJECTED_STATUS = {"not_accepted": 403, "blocked": 403, "closed": 409, "invalid": 422, "unknown": 404}


@router.post("/sessions/{session_id}/agent_messages")
async def agent_message(request: Request, session_id: str) -> JSONResponse:
    """A message from another agent (:class:`~timbal.core.peers.RemotePeer`):
    ``{"from", "text", "message_id"?, "reply_to"?, "priority"?, "hops"?}``.

    ``from`` is the sender's claim: the route trusts whoever may call this
    server's API (as for user messages). 403 when the session's ``accepts``
    refuses the sender or its input guardrails block the text.
    """
    from ..core.peers import PeerRejected

    body = await _body(request)
    sender, text, hops = body.get("from"), body.get("text"), body.get("hops", 0)
    if not isinstance(sender, str) or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'from' and 'text' must be strings.")
    if not isinstance(hops, int) or isinstance(hops, bool):
        raise HTTPException(status_code=400, detail="'hops' must be an integer.")
    for name in ("message_id", "reply_to"):
        if body.get(name) is not None and not isinstance(body[name], str):
            raise HTTPException(status_code=400, detail=f"{name!r} must be a string.")
    priority = body.get("priority") or "normal"
    if priority not in ("normal", "urgent"):
        raise HTTPException(status_code=422, detail=f"priority must be 'normal' or 'urgent', got {priority!r}")
    item = {key: body.get(key) for key in ("from", "text", "message_id", "reply_to")}
    session = await _owned_or_queued(request, session_id, "agent_message", {**item, "priority": priority, "hops": hops})
    if isinstance(session, JSONResponse):
        return session
    try:
        receipt = await session.receive_agent_message(
            text,
            sender=sender,
            message_id=body.get("message_id"),
            reply_to=body.get("reply_to"),
            priority=priority,
            hops=hops,
        )
    except PeerRejected as e:
        raise HTTPException(status_code=_REJECTED_STATUS.get(e.reason, 422), detail=str(e)) from e
    await session.flush()  # durable: the 202 means it survives a restart
    return _json(receipt, status_code=202)


@router.post("/sessions/{session_id}/wake")
async def wake_session(request: Request, session_id: str) -> JSONResponse:
    """Start a turn for the application: ``{"text", "key"?, "tool_policy"?, "silent_ok"?, "trigger"?}``.

    ``tool_policy`` defaults to ``"read_only"``; ``null`` gives the turn every
    tool, a list of names only those. The wake runs once the user has nothing
    pending, as a turn of its own, and may stay silent. ``trigger`` (an object)
    is merged into the turn's root ``metadata.trigger`` (a routine fire's ids).
    """
    body = await _body(request)
    text = body.get("text")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'text' must be a string.")
    policy = body.get("tool_policy", "read_only")
    if policy is not None and not (policy == "read_only" or isinstance(policy, list)):
        raise HTTPException(status_code=400, detail="'tool_policy' must be 'read_only', a list of tool names, or null.")
    trigger = body.get("trigger")
    if trigger is not None and not isinstance(trigger, dict):
        raise HTTPException(status_code=400, detail="'trigger' must be an object.")
    item = {"text": text, "key": body.get("key"), "tool_policy": policy, "silent_ok": bool(body.get("silent_ok", True))}
    session = await _owned_or_queued(request, session_id, "wake", {**item, "trigger": trigger})
    if isinstance(session, JSONResponse):
        return session
    try:
        wake_id = session.wake(
            text,
            key=body.get("key"),
            tool_policy=policy,
            silent_ok=bool(body.get("silent_ok", True)),
            trigger=trigger,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    await session.flush()
    return _json({"wake_id": wake_id, "busy": session.busy}, status_code=202)


@router.post("/sessions/{session_id}/interrupt")
async def interrupt_session(request: Request, session_id: str) -> JSONResponse:
    session = await _session(request, session_id)
    interrupted = session.interrupt()
    await session.flush()
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
    except (RuntimeError, ValueError) as e:
        # Not waiting for an answer (yet), or an id already answered differently.
        raise HTTPException(status_code=409, detail=str(e)) from e
    await session.flush()
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
