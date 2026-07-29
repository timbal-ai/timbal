"""Single-session lifetime for serverless voice boxes (``TIMBAL_VOICE_SINGLE_SESSION=1``).

On serverless workforce deployments the platform spawns one server process
per voice call and reaps the box when the process exits — it cannot see the
call itself (WebRTC media flows browser ↔ TURN relay ↔ box, with no tunneled
socket), so the process must own its lifetime:

* Serve **exactly one** voice session (WebRTC or WebSocket), then exit 0.
  The exit waits for the recording upload (``PUT …/sessions/{session_id}``)
  to finish first — this process is the only thing holding that data.
* Exit 0 if no media connection is established within
  ``TIMBAL_VOICE_IDLE_EXIT_SECS`` (default 60) of server start. The window
  is boot → *media established*, so a browser that POSTs an offer and never
  completes ICE also exits after the window.
* Refuse a second session while one is live or after one has been served:
  409 on ``POST /voice/rtc``, close 1008 on ``/voice/ws``.

All exits here are code 0 — the platform logs non-zero exits as crashes.

Env is read at server start (the lifespan calls :func:`init_single_session_guard`),
never at import time: serverless boxes CRIU-restore from a framework-warm
snapshot with the real env arriving at restore time.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys

import structlog

logger = structlog.get_logger("timbal.server.single_session")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_DEFAULT_IDLE_EXIT_SECS = 60.0


def _exit_process(code: int) -> None:
    """Hard process exit.

    ``os._exit`` rather than ``sys.exit``: a ``SystemExit`` raised inside an
    asyncio task is swallowed by the event loop / uvicorn and the process
    would linger. Everything worth finalizing (recording upload, peer
    connection teardown) has already happened by the time this runs; flush
    stdio so the last log lines reach the platform's log tail.
    """
    with contextlib.suppress(Exception):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


class SingleSessionGuard:
    """One voice session per process: claim → connect → finish → exit.

    Shared by the WebSocket and WebRTC transports (whichever claims first
    owns the process). Not thread-safe — everything runs on the server's
    single event loop, where the plain-attribute state machine is atomic.
    """

    def __init__(self, *, idle_exit_secs: float) -> None:
        self.idle_exit_secs = idle_exit_secs
        self._claimed = False
        self._connected = False
        self._finished = False
        self._idle_task: asyncio.Task | None = None

    def start(self) -> None:
        """Arm the idle-exit timer. Requires a running event loop (lifespan)."""
        self._idle_task = asyncio.create_task(self._idle_exit(), name="voice-single-session-idle-exit")

    def shutdown(self) -> None:
        """Disarm the idle-exit timer (lifespan teardown / media connected)."""
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None

    async def _idle_exit(self) -> None:
        await asyncio.sleep(self.idle_exit_secs)
        # A claimed-but-never-connected session exits too: the window runs
        # boot → media established, so an offer whose ICE never completes
        # doesn't keep the box (and its concurrency slot) alive.
        logger.info(
            "voice_single_session_idle_exit",
            idle_exit_secs=self.idle_exit_secs,
            session_claimed=self._claimed,
        )
        # Detach before finish(): its shutdown() cancels self._idle_task,
        # which is this very task — cancelling ourselves would abort the
        # upload drain mid-flight.
        self._idle_task = None
        # Exit through finish(), not a bare _exit_process: the exit must
        # never race an in-flight recording upload, even in states that
        # shouldn't leave one behind while the timer is still armed.
        await self.finish()

    def claim(self) -> bool:
        """Reserve the one session slot; False while live or after it was served."""
        if self._claimed or self._finished:
            return False
        self._claimed = True
        return True

    def release(self) -> None:
        """Undo a claim whose session never started (e.g. a rejected offer)."""
        if not self._finished:
            self._claimed = False

    def mark_connected(self) -> None:
        """Media is established — disarm the idle-exit timer."""
        if not self._connected:
            self._connected = True
            self.shutdown()

    async def finish(self) -> None:
        """The one session ended: finalize and exit 0.

        Waits for in-flight recording uploads before exiting — bounded by
        the upload's own retry budget (~1h), under the platform's hard
        session cap. Idempotent.
        """
        if self._finished:
            return
        self._finished = True
        self._claimed = True
        self.shutdown()
        from .recording_upload import drain_upload_tasks

        try:
            await drain_upload_tasks()
        except Exception as e:
            logger.error("voice_single_session_drain_failed", error=str(e))
        logger.info("voice_single_session_exit")
        _exit_process(0)


def init_single_session_guard() -> SingleSessionGuard | None:
    """Build and arm the guard when ``TIMBAL_VOICE_SINGLE_SESSION`` is set, else None.

    Called from the server lifespan — i.e. at server start, after any CRIU
    restore has delivered the real env.
    """
    if os.environ.get("TIMBAL_VOICE_SINGLE_SESSION", "").strip().lower() not in _TRUTHY:
        return None
    raw = os.environ.get("TIMBAL_VOICE_IDLE_EXIT_SECS", "")
    try:
        idle_exit_secs = float(raw) if raw.strip() else _DEFAULT_IDLE_EXIT_SECS
    except ValueError:
        logger.warning("voice_single_session_bad_idle_exit_secs", value=raw)
        idle_exit_secs = _DEFAULT_IDLE_EXIT_SECS
    guard = SingleSessionGuard(idle_exit_secs=idle_exit_secs)
    guard.start()
    logger.info("voice_single_session_enabled", idle_exit_secs=idle_exit_secs)
    return guard
