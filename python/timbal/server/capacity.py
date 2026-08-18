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

**Off unless asked for, on the transports that predate it.**
``TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS`` takes an integer, or ``auto``; unset
means no cap on those. Opt-in because turning a cap on is a behaviour change
for a deployment that is already overcommitted — rejecting its fifth call is
the *right* answer, but it should be a decision someone made, not one an
upgrade made for them.

That argument does not cover the per-request LiveKit dial, which is new: no
deployment can regress to a ceiling on a path it never had, and it is the one
entry point that is unbounded by construction (a request can start session
N+1 forever). So ``acquire_session_slot(default_auto=True)`` — used only
there — falls back to the ``auto`` ceiling when nothing is configured. One
counter, two ceilings: an explicit env value still wins everywhere.

Which makes ``0`` distinct from unset: it is the way to say "I know, leave it
uncapped" and it turns the dial path's default off too. A typo reads as unset
(a log line, not a wave of rejections).

``auto`` sizes the cap from the CPU the process may actually use — the
**cgroup quota**, not ``os.cpu_count()``, which reports the host's cores and
would tell a 0.5-vCPU container it has 16 — divided by the worker count. The
quota covers the whole container while this counter is per process, so
without that divisor ``--workers 4`` on a 4-vCPU task would admit 8 per
worker: 32 sessions on 4 CPUs, the exact overcommit this module exists to
prevent.

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
# Exported by `run_server_cli` so a worker can size its share of the cgroup.
_ENV_WORKERS = "TIMBAL_SERVER_WORKERS"

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
        if raw and raw[0] == "max":
            return None
        if len(raw) == 2:
            quota, period = int(raw[0]), int(raw[1])
            if quota > 0 and period > 0:
                return quota / period
            # Nonsense values are not a quota — fall through to v1 the same way
            # an unreadable file does, rather than reporting "unlimited".
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


def _workers() -> int:
    try:
        return max(1, int(os.environ.get(_ENV_WORKERS, "") or 1))
    except ValueError:
        return 1


def _auto_limit() -> int:
    """What ``auto`` sizes to: this worker's share of the cgroup's CPU."""
    return max(_MIN_LIMIT, math.floor(available_cpus() * _PER_CPU / _workers()))


def _resolve_limit() -> int | None:
    """The configured ceiling, or None when nothing was configured.

    ``0`` and ``None`` are different answers: ``0`` is an operator saying "I
    know, leave it uncapped", which has to beat the ``auto`` default the
    per-request path would otherwise apply.
    """
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        return None
    if raw.lower() == "auto":
        return _auto_limit()
    try:
        limit = int(raw)
    except ValueError:
        # Deriving a cap nobody asked for would start rejecting calls over a
        # typo; treating it as unconfigured keeps the failure to a log line.
        logger.warning("voice_capacity_bad_limit", value=raw, using="unconfigured")
        return None
    return max(0, limit)


class _VoiceCapacity:
    """Counter, not a semaphore: a full box rejects rather than queues.

    Not locked — every caller runs on the server's event loop, and a waiter
    is exactly what this exists to avoid (a caller parked behind a 40-minute
    call is worse off than one told to try again).
    """

    def __init__(self) -> None:
        self._resolved = False
        self._limit: int | None = None
        self._auto: int | None = None
        self._active = 0

    @property
    def configured(self) -> int | None:
        """What the env asked for; None when it asked for nothing."""
        if not self._resolved:
            self._limit = _resolve_limit()
            self._resolved = True
        return self._limit

    @property
    def auto(self) -> int:
        """The ceiling ``auto`` would pick, whatever is configured."""
        if self._auto is None:
            self._auto = _auto_limit()
        return self._auto

    @property
    def active(self) -> int:
        return self._active

    def effective_limit(self, *, default_auto: bool = False) -> int:
        configured = self.configured
        if configured is not None:
            return configured
        return self.auto if default_auto else 0

    def acquire(self, *, default_auto: bool = False) -> bool:
        limit = self.effective_limit(default_auto=default_auto)
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
        self._resolved = False
        self._limit = None
        self._auto = None
        self._active = 0


_capacity = _VoiceCapacity()


def acquire_session_slot(*, default_auto: bool = False) -> bool:
    """Claim a session slot. False means the process is at capacity.

    ``default_auto`` applies the ``auto`` ceiling when nothing is configured —
    for entry points new enough that a default cap cannot regress anyone.
    """
    return _capacity.acquire(default_auto=default_auto)


def release_session_slot() -> None:
    """Give a slot back. Must pair with a successful acquire."""
    _capacity.release()


def max_concurrent_sessions(*, default_auto: bool = False) -> int:
    """The ceiling in force; 0 when uncapped."""
    return _capacity.effective_limit(default_auto=default_auto)


def active_sessions() -> int:
    return _capacity.active


def log_capacity() -> None:
    """State the ceilings at boot.

    Resolution is lazy, so without this an operator who set ``auto`` sees
    nothing until the first call lands — and a typo in the env var stays
    invisible until the moment it stops mattering.
    """
    configured = _capacity.configured
    logger.info(
        "voice_capacity",
        limit="unconfigured" if configured is None else (configured or "uncapped"),
        livekit_dial_limit=max_concurrent_sessions(default_auto=True) or "uncapped",
        cpus=round(available_cpus(), 2),
        workers=_workers(),
        hint=(
            f"set {_ENV_VAR} to a number or 'auto' to cap every transport"
            if configured is None
            else None
        ),
    )


def reset_for_tests() -> None:
    _capacity.reset()
