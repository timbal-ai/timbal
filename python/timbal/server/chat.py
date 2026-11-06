"""Chat sessions over HTTP (:class:`~timbal.core.chat_session.ChatSession`), and the chat playground.

``GET /chat/`` serves ``chat.html``. Sessions live in this process: a request
for a session another worker holds is a 404, like ``/runs/{run_id}/events``.
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

logger = structlog.get_logger("timbal.server.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

_HTML_PATH = Path(__file__).parent / "chat.html"

SESSION_TTL_ENV = "TIMBAL_CHAT_SESSION_TTL_S"
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


def _session(request: Request, session_id: str, *, allow_closed: bool = False) -> ChatSession:
    session = _sessions(request).get(session_id)
    if session is None or (session.closed and not allow_closed):
        raise HTTPException(status_code=404, detail=f"No chat session {session_id!r} in this process.")
    return session


def _agent(request: Request) -> Any:
    from ..core.agent import Agent

    runnable = request.app.state.runnable
    if not isinstance(runnable, Agent):
        raise HTTPException(status_code=400, detail="Chat sessions need the served runnable to be an Agent.")
    return runnable


async def _reap(sessions: dict[str, ChatSession]) -> None:
    """Close sessions idle for longer than the TTL (never one with a turn running)."""
    ttl = _session_ttl()
    if ttl <= 0:
        return
    now = time.monotonic()
    for session_id, session in list(sessions.items()):
        if session.closed or (not session.busy and now - session.last_activity > ttl):
            sessions.pop(session_id, None)
            await session.close()


async def close_chat_sessions(app: Any) -> None:
    """Close every session (server shutdown)."""
    sessions = getattr(app.state, "chat_sessions", None) or {}
    for session in list(sessions.values()):
        with contextlib.suppress(Exception):
            await session.close()
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
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    sessions[session.id] = session
    return _json(session.state(), status_code=201)


@router.get("/sessions")
async def list_sessions(request: Request) -> JSONResponse:
    return _json([session.state() for session in _sessions(request).values() if not session.closed])


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> JSONResponse:
    return _json(_session(request, session_id).state())


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> JSONResponse:
    session = _session(request, session_id)
    _sessions(request).pop(session_id, None)
    await session.close()
    return _json({"closed": True})


@router.post("/sessions/{session_id}/messages")
async def send_message(request: Request, session_id: str) -> JSONResponse:
    """Send a user message: ``{"text", "mode"?}`` (``steer`` | ``followup`` | ``collect`` | ``interrupt``)."""
    session = _session(request, session_id)
    body = await _body(request)
    text = body.get("text")
    mode = body.get("mode") or None
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'text' must be a string.")
    busy = session.busy
    try:
        event = session.send(text, mode=mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
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
    session = _session(request, session_id)
    body = await _body(request)
    text = body.get("text")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="'text' must be a string.")
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
    return _json({"wake_id": wake_id, "busy": session.busy}, status_code=202)


@router.post("/sessions/{session_id}/interrupt")
async def interrupt_session(request: Request, session_id: str) -> JSONResponse:
    return _json({"interrupted": _session(request, session_id).interrupt()})


@router.post("/sessions/{session_id}/resume")
async def resume_session(request: Request, session_id: str) -> JSONResponse:
    """Answer a parked turn: ``{"resume": {approval_id: true | {...}, interaction_id: value}}``."""
    session = _session(request, session_id)
    body = await _body(request)
    values = body.get("resume")
    if not isinstance(values, dict) or not values:
        raise HTTPException(status_code=400, detail="'resume' must be a non-empty object.")
    try:
        session.resume(values)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
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
    session = _session(request, session_id, allow_closed=True)
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
    session = _session(request, session_id, allow_closed=True)
    last_event_id = request.headers.get("last-event-id")
    if last_event_id and last_event_id.isdigit():
        after = max(after, int(last_event_id))

    async def stream() -> AsyncGenerator[str, None]:
        seq = after
        items = session.events(after).__aiter__()
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
                    item = task.result()
                except StopAsyncIteration:
                    return
                if item is BACKGROUND_LOG_GAPPED:
                    seq = session.log.forgotten_through
                    yield f"event: gapped\ndata: {json.dumps({'next_cursor': seq})}\n\n"
                    continue
                seq += 1
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
