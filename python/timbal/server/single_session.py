"""Single-session lifetime for serverless voice boxes (``TIMBAL_VOICE_SINGLE_SESSION=1``).

On serverless workforce deployments the platform spawns one server process
per voice call and reaps the box when the process exits — it cannot see the
call itself (WebRTC media flows browser ↔ TURN relay ↔ box, with no tunneled
socket), so the process must own its lifetime:

* Serve **exactly one** voice session (WebRTC, WebSocket, or LiveKit), then exit 0.
  The exit waits for the recording upload (``PUT …/sessions/{session_id}``)
  to finish first — this process is the only thing holding that data.
* Exit 0 if no media connection is established within
  ``TIMBAL_VOICE_IDLE_EXIT_SECS`` (default 60) of server start. The window
  is boot → *media established*, so a browser that POSTs an offer and never
  completes ICE also exits after the window.
* On the LiveKit path, a caller who drops is given
  ``TIMBAL_VOICE_ABANDON_SECS`` (default 45) to rejoin before the box exits.
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
from typing import Any

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

    def __init__(self, *, idle_exit_secs: float, abandon_exit_secs: float = 45.0) -> None:
        self.idle_exit_secs = idle_exit_secs
        self.abandon_exit_secs = abandon_exit_secs
        self._claimed = False
        self._connected = False
        self._finished = False
        self._idle_task: asyncio.Task | None = None
        self._abandon_task: asyncio.Task | None = None
        self._on_abandon: Any = None

    def start(self) -> None:
        """Arm the idle-exit timer. Requires a running event loop (lifespan)."""
        self._idle_task = asyncio.create_task(self._idle_exit(), name="voice-single-session-idle-exit")

    def shutdown(self) -> None:
        """Disarm idle and abandon timers (lifespan teardown / media connected)."""
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None
        self.mark_reconnected()

    async def _idle_exit(self) -> None:
        await asyncio.sleep(self.idle_exit_secs)
        # ICE can complete in the same event-loop turn as the timer firing.
        # Yield once so a queued connectionstatechange → mark_connected()
        # runs before we decide the box is unused; without this, clearing
        # _idle_task below would make a late mark_connected() a no-op and
        # finish() would kill a call that should count as media established.
        await asyncio.sleep(0)
        if self._connected or self._finished:
            return
        # A claimed-but-never-connected session exits too: the window runs
        # boot → media established, so an offer whose ICE never completes
        # doesn't keep the box (and its concurrency slot) alive.
        logger.info(
            "voice_single_session_idle_exit",
            idle_exit_secs=self.idle_exit_secs,
            session_claimed=self._claimed,
        )
        # Detach before awaiting so shutdown()/finish() cannot cancel this
        # very task mid-drain. Commit _finished only after the final
        # _connected check — media may still land while we drain uploads.
        self._idle_task = None
        if self._connected or self._finished:
            return
        from .recording_upload import drain_upload_tasks

        try:
            await drain_upload_tasks()
        except Exception as e:
            logger.error("voice_single_session_drain_failed", error=str(e))
        if self._connected or self._finished:
            # Connected (or the session already finished) while draining —
            # session teardown owns the process lifetime from here.
            return
        self._finished = True
        self._claimed = True
        logger.info("voice_single_session_exit")
        _exit_process(0)

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
            # Cancel idle only — do not touch an in-flight abandon timer
            # (mark_connected is once; abandon is armed later on a blip).
            if self._idle_task is not None:
                self._idle_task.cancel()
                self._idle_task = None

    def mark_disconnected(self, *, on_abandon: Any = None) -> None:
        """Media dropped but the call may resume — re-arm a bounded window.

        Distinct from the boot idle-exit, which is 'no one ever showed up'.
        Both end in ``_exit_process(0)``. ``on_abandon`` (typically
        ``session.close``) runs before the drain so the recording finalizes.
        """
        if self._finished or self._abandon_task is not None:
            return
        self._on_abandon = on_abandon
        self._abandon_task = asyncio.create_task(
            self._abandon_exit(), name="voice-single-session-abandon"
        )

    def mark_reconnected(self) -> None:
        """Caller rejoined — cancel the abandon window."""
        if self._abandon_task is not None:
            self._abandon_task.cancel()
            self._abandon_task = None
        self._on_abandon = None

    async def _abandon_exit(self) -> None:
        await asyncio.sleep(self.abandon_exit_secs)
        await asyncio.sleep(0)
        if self._finished:
            return
        logger.info(
            "voice_single_session_abandoned",
            abandon_exit_secs=self.abandon_exit_secs,
        )
        self._abandon_task = None
        on_abandon = self._on_abandon
        self._on_abandon = None
        if on_abandon is not None:
            try:
                result = on_abandon()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("voice_single_session_abandon_close_failed", error=str(e))
        if self._finished:
            return
        from .recording_upload import drain_upload_tasks

        try:
            await drain_upload_tasks()
        except Exception as e:
            logger.error("voice_single_session_drain_failed", error=str(e))
        if self._finished:
            return
        self._finished = True
        self._claimed = True
        logger.info("voice_single_session_exit")
        _exit_process(0)

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
    raw_abandon = os.environ.get("TIMBAL_VOICE_ABANDON_SECS", "")
    try:
        abandon_exit_secs = float(raw_abandon) if raw_abandon.strip() else 45.0
    except ValueError:
        logger.warning("voice_single_session_bad_abandon_secs", value=raw_abandon)
        abandon_exit_secs = 45.0
    guard = SingleSessionGuard(idle_exit_secs=idle_exit_secs, abandon_exit_secs=abandon_exit_secs)
    guard.start()
    logger.info(
        "voice_single_session_enabled",
        idle_exit_secs=idle_exit_secs,
        abandon_exit_secs=abandon_exit_secs,
    )
    return guard
