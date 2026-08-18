"""Per-process cap on concurrent voice sessions.

A voice session is expensive and mostly *not* free while it waits: an STT
socket, a TTS socket, resampling, turn detection (an ONNX model on some
configs), and — on LiveKit — a libwebrtc runtime with its own thread pool.
One process per call (serverless, ``TIMBAL_VOICE_SINGLE_SESSION``) never had
to think about this. A long-lived server does: without a ceiling, session
N+1 does not fail, it degrades the N calls already in progress, and choppy
audio is unrecoverable for everyone on the box. A rejection is the better
failure — the caller can retry or fall back, and the live calls are
untouched.

The ceiling is per **process**, deliberately: the process is the only party
that knows how many sessions it is actually running. A platform-side limit
sits in front of some deployments and not others, and can be several
schedulers wide.

**Off unless asked for.** ``TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS`` takes an
integer, or ``auto``; unset (and ``0``) means no cap. Opt-in because turning
a cap on is a behaviour change for a deployment that is already overcommitted
— rejecting its fifth call is the *right* answer, but it should be a decision
someone made, not one an upgrade made for them.

``auto`` sizes the cap from the CPU the process may actually use — the
**cgroup quota**, not ``os.cpu_count()``, which reports the host's cores and
would tell a 0.5-vCPU container it has 16.

``_PER_CPU`` is a starting point, not a measurement. The real number depends
on the turn detector and the transport; size it on one task under load
before leaning on it.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import structlog

logger = structlog.get_logger("timbal.server.capacity")

_ENV_VAR = "TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS"

_PER_CPU = 2.0

# Below this a cap does more harm than good: one session must always be
# admissible, and a second lets a caller reconnect before the first has
# finished tearing down.
_MIN_LIMIT = 2

_CGROUP_V2 = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _cgroup_cpus() -> float | None:
    """CPUs the cgroup is allowed, or None when unlimited/unreadable."""
    try:
        raw = _CGROUP_V2.read_text().split()
        if len(raw) == 2 and raw[0] != "max":
            quota, period = int(raw[0]), int(raw[1])
            if quota > 0 and period > 0:
                return quota / period
            return None
        if raw and raw[0] == "max":
            return None
    except (OSError, ValueError, IndexError):
        pass
    quota = _read_int(_CGROUP_V1_QUOTA)
    period = _read_int(_CGROUP_V1_PERIOD)
    # -1 is v1's "unlimited".
    if quota is not None and quota > 0 and period:
        return quota / period
    return None


def available_cpus() -> float:
    """The CPU this process may actually use.

    cgroup quota first (containers), then the affinity mask, then the host's
    core count. Never zero — a fractional quota still runs code.
    """
    quota = _cgroup_cpus()
    if quota is not None:
        return max(quota, 0.1)
    try:
        return float(len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        return float(os.cpu_count() or 1)


def _default_limit() -> int:
    return max(_MIN_LIMIT, math.floor(available_cpus() * _PER_CPU))


def _resolve_limit() -> int:
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        logger.info("voice_capacity_disabled", hint=f"set {_ENV_VAR} to a number or 'auto'")
        return 0
    if raw.lower() == "auto":
        limit = _default_limit()
        logger.info("voice_capacity_auto", limit=limit, cpus=round(available_cpus(), 2))
        return limit
    try:
        limit = int(raw)
    except ValueError:
        # Deriving a cap nobody asked for would start rejecting calls over a
        # typo; staying uncapped keeps the failure to a log line.
        logger.warning("voice_capacity_bad_limit", value=raw, using="no cap")
        return 0
    if limit <= 0:
        logger.info("voice_capacity_disabled")
        return 0
    logger.info("voice_capacity_configured", limit=limit)
    return limit


class _VoiceCapacity:
    """Counter, not a semaphore: a full box rejects rather than queues.

    Not locked — every caller runs on the server's event loop, and a waiter
    is exactly what this exists to avoid (a caller parked behind a 40-minute
    call is worse off than one told to try again).
    """

    def __init__(self) -> None:
        self._limit: int | None = None
        self._active = 0

    @property
    def limit(self) -> int:
        if self._limit is None:
            self._limit = _resolve_limit()
        return self._limit

    @property
    def active(self) -> int:
        return self._active

    def acquire(self) -> bool:
        limit = self.limit
        if limit and self._active >= limit:
            logger.warning("voice_capacity_rejected", active=self._active, limit=limit)
            return False
        self._active += 1
        return True

    def release(self) -> None:
        if self._active == 0:
            # A double release would make the box look emptier than it is,
            # which is the failure this cap exists to prevent.
            logger.error("voice_capacity_release_underflow")
            return
        self._active -= 1

    def reset(self) -> None:
        """Tests only: re-read the env and forget in-flight sessions."""
        self._limit = None
        self._active = 0


_capacity = _VoiceCapacity()


def acquire_session_slot() -> bool:
    """Claim a session slot. False means the process is at capacity."""
    return _capacity.acquire()


def release_session_slot() -> None:
    """Give a slot back. Must pair with a successful acquire."""
    _capacity.release()


def max_concurrent_sessions() -> int:
    """The resolved cap; 0 when disabled."""
    return _capacity.limit


def active_sessions() -> int:
    return _capacity.active


def reset_for_tests() -> None:
    _capacity.reset()
