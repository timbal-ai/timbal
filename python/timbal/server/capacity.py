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
without that divisor ``--workers 4`` on a 4-vCPU task would admit four times
its share per worker: the exact overcommit this module exists to prevent.

What ``auto`` picks is measured rather than guessed — see
``benchmarks/voice/bench_cpu.py``, which ramps concurrent sessions against real
VAD and turn detection and fails a rung on event-loop lag and turn latency
rather than on CPU%, because punctuality is what a caller hears. Two findings
shape the sizing, and neither survives a single constant:

* Cost is dominated by the **turn detector**, not by "a session" — ~0.001 cores
  each with ``provider`` against ~0.056 with ``local``, a ~50x spread. So ``auto``
  resolves a per-deployment :class:`_Profile` (see :func:`configure_capacity`)
  instead of one number for every shape of voice app.
* The binding resource is usually the **event loop**, not the core count. The
  inline per-frame work (Silero every 32ms, the recorder's MP3 encode) is
  charged to one loop and a worker has exactly one, so scaling purely by cores
  overcommits a single-worker process on a big box. Hence the ``per_loop`` term,
  which is *not* divided by the worker count.

The old flat 2.0/cpu was wrong in both directions at once: it capped a
``provider`` deployment at ~3% of what it holds (2 sessions on a box that took
60), while allowing 20 audio-EOU sessions onto a loop that broke at 16.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger("timbal.server.capacity")

_ENV_VAR = "TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS"
# Exported by `run_server_cli` so a worker can size its share of the cgroup.
_ENV_WORKERS = "TIMBAL_SERVER_WORKERS"


@dataclass(frozen=True)
class _Profile:
    """What a session of this shape costs, and how many one loop will hold.

    ``per_cpu`` is divided by the worker count (the quota is shared);
    ``per_loop`` is not, because each worker is its own process with its own
    event loop and its own counter.
    """

    name: str
    per_cpu: float
    per_loop: int


# Both numbers come from `benchmarks/voice/bench_cpu.py`, discounted for what
# that harness cannot see: it mocks STT/TTS, so it omits a websocket per
# session with per-message JSON parsing and base64 audio decode, all on this
# same loop. Measured on 10 cores / 1 worker — cost per session, the last rung
# that held, and the first that failed three runs in a row:
#
#   provider  0.001 cores   60 sessions/loop, 64 broke   -> no_eou
#   lexical   0.007 cores   48 sessions/loop, 52 broke   -> text_eou
#   local     0.056 cores   12 sessions/loop, 16 broke   -> audio_eou
#
# The ~50x spread is why one constant could not work: the turn detector, not
# "a voice session", is what costs money. Recording costs 2.4x the CPU per
# session (0.017 vs 0.007 with `lexical`), an MP3 encode per mic chunk inline
# on the loop — see `_profile_for`.
#
# Requiring a failure to reproduce mattered more than any of the tuning here.
# Pooled turn-latency p95 trips on a stray GC pause, and an unlucky rung read
# as a ceiling: `provider` first measured 28 and is really 60. Anything below
# was anchored to background noise on a laptop.
_PROFILES = {
    # Smart Turn ONNX per utterance boundary. The only class where CPU, not
    # scheduling, binds: the 16 that broke drew 0.94 cores — one full core, on
    # one loop — and dropped a third of its expected turns.
    "audio_eou": _Profile("audio_eou", per_cpu=6.0, per_loop=8),
    # Silero VAD inline every 32ms, then punctuation scoring in ~0.1ms. At
    # ~0.25ms per frame of inference this would not saturate a loop until ~125
    # sessions, so like `no_eou` it breaks on scheduling, not on the VAD.
    "text_eou": _Profile("text_eou", per_cpu=20.0, per_loop=24),
    # No EOU model means no VAD endpointer either (the session logs
    # `vad_endpointing_unavailable`), so nothing runs per frame but bookkeeping.
    # 60 sessions drew 0.06 cores: the ceiling here is how many timers one loop
    # can service on time, and nowhere near a core's worth of work.
    "no_eou": _Profile("no_eou", per_cpu=32.0, per_loop=32),
}

# Deliberately the expensive one. `resolve_turn_detector(None)` builds Smart
# Turn + Namo whenever `timbal[voice]` is installed, so a deployment that never
# names a detector really is in the audio class — and one we cannot classify
# should not be assumed cheap.
_DEFAULT_PROFILE = _PROFILES["audio_eou"]

_DETECTOR_PROFILES = {
    "local": "audio_eou",
    "audio": "audio_eou",
    "smart_turn": "audio_eou",
    "lexical": "text_eou",
    "semantic": "text_eou",
    "punctuation": "text_eou",
    "provider": "no_eou",
    "stt": "no_eou",
    "heuristic": "no_eou",
    "raw": "no_eou",
    "none": "no_eou",
    "off": "no_eou",
}

# Provider ids `resolve_stt` sends to a native-EOU STT (Flux; Munsit), which
# does its own endpointing: `select_turn_detector_spec` then forces the provider
# detector and will not let a client escalate back to `local`. So such a
# deployment is in the cheap class no matter what its detector field says.
_NATIVE_EOU_STT_PROVIDERS = frozenset({"deepgram", "deepgram-flux", "munsit", "faseeh"})

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


def _detector_profile_name(spec: Any) -> str | None:
    """Cost class for a turn-detector spec, or None when it cannot be told.

    Only the documented mode names are classified. An instance or a factory
    could be anything — including a subclass that loads its own model — so it
    falls back to the conservative default rather than being guessed at.
    """
    if not isinstance(spec, str):
        return None
    key = spec.strip().lower()
    if key in ("", "default"):
        # What `resolve_turn_detector` builds for None: Smart Turn when the
        # voice extra is installed. Being wrong here in the cheap direction
        # would overcommit, so treat it as the audio class.
        return "audio_eou"
    return _DETECTOR_PROFILES.get(key)


def _profile_for(voice_config: Any) -> _Profile:
    if voice_config is None:
        return _DEFAULT_PROFILE
    stt = str(getattr(voice_config, "stt_provider", "") or "").strip().lower()
    name = (
        "no_eou"
        if stt in _NATIVE_EOU_STT_PROVIDERS
        else _detector_profile_name(getattr(voice_config, "turn_detector", None))
    )
    profile = _PROFILES.get(name or "", _DEFAULT_PROFILE)
    recording = getattr(voice_config, "recording", None)
    if recording is not None and getattr(recording, "dir", None):
        # An MP3 encode per mic chunk, synchronous on the session's own loop
        # (`voice/recording.py`: "call from the session's event loop only").
        return _Profile(
            f"{profile.name}+recording",
            per_cpu=profile.per_cpu / 2,
            per_loop=max(_MIN_LIMIT, profile.per_loop // 2),
        )
    return profile


_profile = _DEFAULT_PROFILE


def configure_capacity(voice_config: Any) -> None:
    """Size ``auto`` from what this deployment is actually configured to run.

    Called once at boot with the server's own ``VoiceConfig`` — never per
    request. A client can ask for a different turn detector in its hello, and
    honouring that here would let callers pick their own admission limit.

    The consequence is worth stating: on a deployment whose *default* detector
    is cheap, a client that asks for ``local`` gets a session costing ~30x what
    the cap was sized for. Native-EOU STT (Flux, Munsit) can't be escalated
    that way, but the others can — pin the detector server-side, or set the
    env var explicitly, if that matters to you.
    """
    global _profile
    _profile = _profile_for(voice_config)
    _capacity.resize()


def _auto_limit() -> int:
    """What ``auto`` sizes to.

    The smaller of this worker's share of the cgroup's CPU and what one event
    loop will hold. The second term is the one the benchmark says usually
    binds, and it is why raising the per-CPU figure alone would be wrong: a
    single worker on a big box would otherwise be handed a cap sized for cores
    it cannot reach from its one loop.
    """
    per_cpu = math.floor(available_cpus() * _profile.per_cpu / _workers())
    return max(_MIN_LIMIT, min(per_cpu, _profile.per_loop))


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

    def resize(self) -> None:
        """Re-resolve both ceilings after the profile changes.

        ``configured`` is memoized too, and when the env says ``auto`` that
        cached value came from the profile — so dropping only ``_auto`` would
        leave a stale ceiling behind. The in-flight count survives: this is a
        re-sizing, not a reset.
        """
        self._resolved = False
        self._limit = None
        self._auto = None

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
        auto=_auto_limit(),
        # Which detector the cap was sized for is the first thing to check when
        # `auto` looks surprising.
        profile=_profile.name,
        cpus=round(available_cpus(), 2),
        workers=_workers(),
        hint=(f"set {_ENV_VAR} to a number or 'auto' to cap every transport" if configured is None else None),
    )


def reset_for_tests() -> None:
    global _profile
    _profile = _DEFAULT_PROFILE
    _capacity.reset()
