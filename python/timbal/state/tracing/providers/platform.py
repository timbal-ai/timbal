from __future__ import annotations

import asyncio
import atexit
import contextlib
import contextvars
import functools
import gzip
import hashlib
import itertools
import json
import math
import os
import threading
import time
import weakref
from collections import OrderedDict, deque
from time import perf_counter_ns, time_ns
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

import structlog

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from .. import uploads
from ..trace import Trace
from . import base as _base
from .base import _COPIES, TracingProvider, _capture, _redacted_trace, _RedactedRunContextView

if TYPE_CHECKING:
    from ...config import PlatformConfig
    from ...context import RunContext

logger = structlog.get_logger(__name__)


@functools.lru_cache(maxsize=16)
def _parse_cache_limits(
    raw_capacity: str | None, raw_ttl: str | None, default_capacity: int, default_ttl: float
) -> tuple[int, float]:
    """Memoized per raw value, so an invalid setting warns once instead of on every store."""
    capacity, ttl = default_capacity, default_ttl
    if raw_capacity is not None:
        try:
            capacity = int(raw_capacity)
            if capacity < 0:
                raise ValueError
        except ValueError:
            logger.warning("invalid_trace_cache_capacity", value=raw_capacity, default=default_capacity)
            capacity = default_capacity
    if raw_ttl is not None:
        try:
            ttl = float(raw_ttl)
            if not math.isfinite(ttl) or ttl < 0:
                raise ValueError
        except ValueError:
            logger.warning("invalid_trace_cache_ttl", value=raw_ttl, default=default_ttl)
            ttl = default_ttl
    return capacity, ttl


class _Saves:
    """One run's saves in flight, and the newest snapshot among its saves that finished."""

    __slots__ = ("newest", "pending")

    def __init__(self) -> None:
        # snapshot -> resolved when that save finishes, successfully or not
        self.pending: dict[int, asyncio.Future[None]] = {}
        self.newest = -1


_MAX_ON_WIRE = 2
_UNSAVED_MAX = 256


class _Flight:
    """One destination's saves on one event loop, single-flight: the round to send next, the PATCHes on the wire
    and the task sending them. A round carries no state of its own: whatever sends it, it sends the save the
    destination is owed when it goes out (``_save``)."""

    __slots__ = ("arrival", "driver", "key", "loop", "next", "on_wire", "sent")

    def __init__(self, key: str, loop: asyncio.AbstractEventLoop) -> None:
        self.key = key
        self.loop = loop
        self.next: asyncio.Future[None] | None = None
        self.sent: list[asyncio.Future[None]] = []
        self.on_wire: dict[asyncio.Task, float] = {}  # save -> loop time it went out
        self.driver: asyncio.Task | None = None
        self.arrival: asyncio.Future[None] | None = None  # wakes the driver when a save joins

    def waits(self) -> list[asyncio.Future[None]]:
        """The rounds not settled yet: a read of this run must not overtake them."""
        rounds = [r for r in self.sent if not r.done()]
        if self.next is not None and not self.next.done():
            rounds.append(self.next)
        return rounds

    def terminal(self) -> bool:
        write = PlatformTracingProvider._unsaved.get(self.key)
        return write is not None and _ended(write.context)


def _ended(run_context: RunContext | None) -> bool:
    """The run's root finished: its next save is the end of the run."""
    root = run_context.root_span() if run_context is not None else None
    return root is not None and root.t1 is not None


def _retrieve(future: asyncio.Future) -> None:
    """Mark a round's error seen: every caller of it may have been cancelled."""
    if not future.cancelled():
        future.exception()


def _transient(exc: BaseException) -> bool:
    """A failure worth sending the same save again for: 429, 5xx, a timeout or a dropped connection
    (what ``_request`` already retried a few times), not a request the platform turned down."""
    import httpx

    from ....errors import PlatformError

    if isinstance(exc, PlatformError):
        status = exc.status_code
        return status is None or status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError | OSError | TimeoutError)


def _settle(flight: _Flight, joined: asyncio.Future[None], save: asyncio.Task) -> None:
    """A round's save finished. Success also settles earlier rounds still on the wire (a stalled
    PATCH): this one carried their state and newer."""
    flight.on_wire.pop(save, None)
    if joined not in flight.sent:
        # A newer round landed while this (stalled) one was still on the wire, and settled it: it carried this
        # state and newer, so how this one ended changes nothing.
        if not save.cancelled():
            save.exception()  # retrieved: nobody waits on it any more
        return
    if save.cancelled() or save.exception() is not None:
        flight.sent.remove(joined)
        if not joined.done():
            if save.cancelled():
                joined.cancel()
            else:
                joined.set_exception(save.exception())
        if not save.cancelled() and _transient(save.exception()) and flight.terminal():
            # No later save of an ended run is coming to carry this one.
            PlatformTracingProvider._schedule_retry(flight.key, flight.loop)
        return
    index = flight.sent.index(joined)
    settled, flight.sent[: index + 1] = flight.sent[: index + 1], []
    for future in settled:
        if not future.done():
            future.set_result(None)


def _flush_timeout() -> float:
    raw = os.environ.get("TIMBAL_TRACE_FLUSH_TIMEOUT_SECS")
    try:
        value = float(raw) if raw is not None else 5.0
        if not math.isfinite(value) or value < 0:
            raise ValueError
    except ValueError:
        logger.warning("invalid_trace_flush_timeout", value=raw, default=5.0)
        value = 5.0
    return value


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError
    except ValueError:
        logger.warning("invalid_trace_setting", name=name, value=raw, default=default)
        return default
    return value


# Set in the task a keep-alive save runs in: it is no progress of the run's own.
_KEEPALIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("timbal_trace_keepalive", default=False)
# Set by get_strict around its get(): that read asks for the whole trace, never the select=resume projection.
# A context variable, not an argument: a subclass that overrides get() is still the one get_strict calls.
_WHOLE_TRACE: contextvars.ContextVar[bool] = contextvars.ContextVar("timbal_trace_whole_read", default=False)
_KEEPALIVES_MAX = 2048


class _KeepAlive:
    """A run in flight: when it last saved for itself, and the timer for its next keep-alive save."""

    __slots__ = ("context", "handle", "loop", "progress", "provider")

    def __init__(self, context: RunContext, provider: type, loop: asyncio.AbstractEventLoop, progress: float):
        self.context = context
        self.provider = provider
        self.loop = loop
        self.progress = progress
        self.handle: asyncio.TimerHandle | None = None

    def cancel(self) -> None:
        if self.handle is not None:
            self.handle.cancel()
            self.handle = None


def _keepalive_done(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        # The next save of the run, or the next keep-alive (the timer was re-armed when it started), tries again.
        logger.warning("platform_trace_keepalive_failed", error=str(error))


def _flush_at_exit() -> None:
    """Scripts: ``asyncio.run`` cancels saves still pending when its coroutine returns. Resend those runs'
    latest state on a fresh loop. A server flushes in its lifespan shutdown, before this runs."""
    if not PlatformTracingProvider._unsaved:
        return
    try:
        asyncio.get_running_loop()
        return  # still inside a loop in this thread: nothing here can block on it
    except RuntimeError:
        pass
    with contextlib.suppress(Exception):
        asyncio.run(PlatformTracingProvider.flush())


DELTA_MEDIA_TYPE = "application/vnd.timbal.trace-delta+json"
_DELTA_RUNS_MAX = 2048
# Entries of each kind kept in the capability cache besides the hosts of app runs (``_host_caps``): declared
# families (one per org, say) and runs' own destinations (one per run), each bounded on its own.
_CAPS_FAMILIES_MAX = 1024
# Saves and reads run on event loops in more than one thread (a worker per thread, the flush at exit): the
# capability cache is read, filled, refreshed and evicted under this lock.
_CAPS_LOCK = threading.Lock()
_GZIP_INLINE_MAX = 256 * 1024


async def _gzip(body: bytes) -> bytes:
    """Large bodies compress off the event loop: a whole multi-MB trace would block it (a voice turn)
    for tens of ms. ``body`` is already encoded, so nothing changes under the thread."""
    if len(body) <= _GZIP_INLINE_MAX:
        return gzip.compress(body, compresslevel=5)
    return await asyncio.to_thread(gzip.compress, body, 5)


class _HostCaps:
    """What a platform host advertised it accepts (learned from its responses), and what it turned down."""

    __slots__ = ("delta", "delta_rejected", "gzip", "gzip_rejected", "select_resume")

    def __init__(self) -> None:
        self.gzip = False
        self.delta = False
        # A rejection outlives later advertisements (a proxy may pass the header but not the body) and
        # overrides a forced format, so one bad host never drops saves.
        self.gzip_rejected = False
        self.delta_rejected = False
        self.select_resume = False

    def learn(self, headers) -> None:
        found = _header_items(headers)
        if "gzip" in found.get("accept-encoding", "").lower():
            self.gzip = True
        if DELTA_MEDIA_TYPE in found.get("accept-patch", "").lower():
            self.delta = True
        if "resume" in found.get(SELECT_HEADER, "").lower():
            self.select_resume = True
        stored = found.get(SNAPSHOT_HEADER, "").strip()
        if stored.isdigit():
            _observe_snapshot(int(stored))


class RunPath(str):
    """A run path that says which runs share its endpoint: what ``_run_path`` returns when it knows.

    ``family`` names the endpoint and is a prefix of the path, so a declaration is tied to the route it
    declares, e.g. ``RunPath(f"orgs/{org}/chat/runs/{run_id}", family=f"orgs/{org}/chat/runs")``. What that
    endpoint advertised (gzip, span deltas, ``select=resume``) and refused is then learned once for all its
    runs. A plain string declares nothing: that run's destination keeps its own, which is always right (one
    destination is one endpoint) and costs relearning per run, including one refused save per run for a format
    the endpoint advertises (or is forced) and then refuses. Whoever computes the path declares it: a subclass
    or ``configured()`` copy that returns another path declares again or keeps the default. A family shared
    by routes that are not one endpoint (one too short a prefix) is the author's declaration, and trusted.
    """

    family: str

    def __new__(cls, path: str, family: str) -> RunPath:
        if not isinstance(path, str):
            raise TypeError("RunPath's path is a string")
        if not isinstance(family, str) or not family:
            raise TypeError("RunPath(family=...) is a non-empty string: the name every run of its endpoint shares")
        if not path.startswith(family):
            raise ValueError(f"RunPath family {family!r} is not a prefix of its path {path!r}")
        run_path = super().__new__(cls, path)
        run_path.family = family
        return run_path

    def __getnewargs__(self) -> tuple[str, str]:
        return str(self), self.family  # copy and pickle rebuild it with its family


class RunNotStored(LookupError):
    """Raised by a ``_run_path`` for a run this provider does not store (a registry that does not know it).

    Reads answer it as not stored (``None``), as they do a 404; saves fail on it. Only this exception: any other
    one ``_run_path`` raises, a ``KeyError`` included, is a bug or a misconfiguration and goes to the caller (read
    as "not stored", an HTTP resume would run without the tool policy of the run it resumes).
    """


def _header_items(headers) -> dict[str, str]:
    items = getattr(headers, "items", None)
    if items is None:
        return {}
    return {str(name).lower(): value for name, value in items() if isinstance(value, str)}


_TRACE_POOLS: dict[asyncio.AbstractEventLoop, object] = {}


@contextlib.contextmanager
def _as_run(run_context: RunContext):
    """Make ``run_context`` the current run for one request: ``_request`` takes host and credentials from the
    current run, and a save can be sent from outside its run (a shutdown flush, a resend at exit)."""
    from ... import get_run_context, set_run_context

    previous = get_run_context()
    if previous is run_context:
        yield
        return
    set_run_context(run_context)
    try:
        yield
    finally:
        set_run_context(previous)


@contextlib.contextmanager
def _pooled():
    """Trace requests on this loop share one connection pool, so a save does not pay a TLS handshake each time.

    Uses the platform HTTP session (no response cookies, per loop and process); a caller that already scoped
    one keeps it. Closed at :meth:`PlatformTracingProvider.flush`; a loop that ends without one only leaves
    idle keep-alive sockets for the GC.
    """
    from ....platform._http_session import _SESSION, _Session

    current = _SESSION.get()
    if current is not None and current.usable():
        yield
        return
    loop = asyncio.get_running_loop()
    for stale in [other for other in _TRACE_POOLS if other.is_closed()]:
        del _TRACE_POOLS[stale]
    session = _TRACE_POOLS.get(loop)
    if session is None or not session.usable():
        session = _TRACE_POOLS[loop] = _Session()
    token = _SESSION.set(session)
    try:
        yield
    finally:
        _SESSION.reset(token)


async def _close_pool() -> None:
    session = _TRACE_POOLS.pop(asyncio.get_running_loop(), None)
    if session is not None:
        session.retired = True
        await session.close_if_idle()


SNAPSHOT_HEADER = "x-timbal-snapshot"
APPLIED_HEADER = "x-timbal-snapshot-applied"
SELECT_HEADER = "x-timbal-select"
_last_snapshot = 0
# Event loops in other threads save too: two equal numbers would make the platform drop the second as stale.
# Also guards _unsaved and _LATEST, and a PATCH's number with the body it sends (see PlatformTracingProvider._save).
# Re-entrant: it is held while a trace is serialized and redacted (allocations, a redactor's own code), and a
# collection then may finish a save left suspended in a closed loop on this thread; nothing it runs may wait for it.
_snapshot_lock = threading.RLock()
# time_ns() - perf_counter_ns(), taken at an edge of the wall clock's tick (see _epoch_ns); None until first use.
_epoch_offset_ns: int | None = None


def _epoch_ns() -> int:
    """ns since the epoch with ``perf_counter``'s resolution.

    Before Python 3.13, Windows' ``time_ns()`` ticks every 15.6 ms: a process continuing a run in the same
    tick as the last save of another would send the same number, and the platform would drop it as stale.
    The offset to ``perf_counter_ns()`` (fine everywhere) is taken once, right as the wall clock ticks over,
    so it is exact; waiting for that edge costs at most one tick, once per process (nothing where the
    wall clock is fine).
    """
    global _epoch_offset_ns
    if _epoch_offset_ns is None:
        start = time_ns()
        while (now := time_ns()) == start:
            pass
        _epoch_offset_ns = now - perf_counter_ns()
    return _epoch_offset_ns + perf_counter_ns()


def _next_snapshot() -> int:
    """Strictly increasing within the process and, by the clock, across processes (ns since the epoch).

    The platform ignores a save whose snapshot is not newer than the last it applied to the run, and a
    run can be continued by another process (an approval resumed elsewhere): a per-process counter
    would restart below what the first process sent.
    """
    global _last_snapshot
    with _snapshot_lock:
        _last_snapshot = max(_last_snapshot + 1, _epoch_ns())
        return _last_snapshot


def _observe_snapshot(stored: int) -> None:
    """The platform said it holds ``stored``: number later saves above it, whatever this clock says."""
    global _last_snapshot
    with _snapshot_lock:
        _last_snapshot = max(_last_snapshot, stored)


class _DeltaRun:
    """Spans of one run the platform acknowledged, by fingerprint, and each span's start index (root = 0).

    ``root_stored``: the platform acknowledged a full root carrying ``memory`` (see ``_delta_body``). ``use``: the
    run context these were acknowledged for, and ``policy`` the redaction that made what was sent (``_policy``):
    another context of the run id, or a save redacted another way, starts over (a span acknowledged as one provider
    sent it is not what another provider's policy would store).
    """

    __slots__ = ("acked", "policy", "root_stored", "seq", "use")

    def __init__(self) -> None:
        self.acked: dict[str, str] = {}
        self.seq: dict[str, int] = {}
        self.root_stored = False
        self.use: int | None = None
        self.policy: tuple | None = None


class _Write:
    """A save asked of one destination (``put``): what is owed it until a PATCH carrying this request, or a newer
    one, lands there (``_unsaved``). Everything it is sent with is fixed when it is asked:

    ``context``: the run behind any redacted copies it was handed in (whose save it is); ``source``: what was handed
    in, sent as it is (the run itself, read when each PATCH goes out, or a copy, with whatever was done to it before
    it was handed on), through ``redactors``; ``provider``: the provider class that accepted it, whose configuration
    every PATCH of it uses; ``platform_config`` and ``run_path``: the destination it was accepted for, never worked
    out again (a ``_run_path`` that answers differently later routes later saves, not this one); ``seq``: its place
    among every save asked in this process; ``sent``: a PATCH of it has been serialized; ``out``: its PATCHes whose
    answer has not come back. Only a write that was sent and has none out may be pushed out of ``_unsaved``.
    """

    __slots__ = ("context", "out", "platform_config", "provider", "redactors", "run_path", "sent", "seq", "source")

    def __init__(
        self,
        context: RunContext,
        source: RunContext,
        redactors: tuple,
        provider: type,
        platform_config: PlatformConfig,
        run_path: str,
        seq: int,
    ) -> None:
        self.context = context
        self.source = source
        self.redactors = redactors
        self.provider = provider
        self.platform_config = platform_config
        self.run_path = run_path
        self.seq = seq
        self.sent = False
        self.out = 0

    @property
    def id(self) -> str:
        return self.context.id


# Every save asked in this process, in order (_Write.seq); and every run context saved (_DeltaRun.use).
_REQUESTS = itertools.count(1)
_USES = itertools.count(1)


def _unwrap(run_context: RunContext, own) -> tuple[RunContext, tuple]:
    """The run behind ``run_context``, and the redaction policies to apply to what was handed in: ``own``, unless a
    copy in the chain was already made with it (this provider's ``put`` made it for a subclass's own ``_store``).
    What was handed in is sent as it is: a field another provider or an exporter took out of its copy stays out."""
    context, applied = run_context, False
    while isinstance(context, _RedactedRunContextView):
        applied = applied or (own is not None and context.__dict__.get("_redactor") is own)
        context = context._original
    return context, (() if own is None or applied else (own,))


def _policy(write: _Write) -> tuple:
    """Which redaction made what ``write`` sends: the policies of the copies it came in (``None`` for a copy that
    does not say), then those applied on top."""
    chain = []
    view = write.source
    while isinstance(view, _RedactedRunContextView):
        redactor = view.__dict__.get("_redactor")
        chain.append(None if redactor is None else id(redactor))
        view = view._original
    return (*chain, "|", *(id(redactor) for redactor in write.redactors))


def _token(platform_config: PlatformConfig) -> str:
    """Which credentials: a run is read back from memory only by the credentials that stored it."""
    return hashlib.sha256(platform_config.auth.token.get_secret_value().encode()).hexdigest()


def _redacted(write: _Write) -> RunContext:
    """What ``write`` sends now: its source through its redaction policies."""
    sending = write.source
    for redactor in write.redactors:
        sending = _RedactedRunContextView(sending, _redacted_trace(sending._trace, redactor), redactor)
    return sending


class _Latest:
    """The newest run context saved to one destination (its ``_platform_use``). Held by every context saved there,
    and only weakly by ``_LATEST``: it lives exactly as long as a context that could still be saved there."""

    __slots__ = ("__weakref__", "use")

    def __init__(self, use: int) -> None:
        self.use = use


# destination key -> _Latest, under _snapshot_lock.
_LATEST: weakref.WeakValueDictionary[str, _Latest] = weakref.WeakValueDictionary()
# destination key -> the context (_platform_use) whose PATCH this process serialized last there, under _snapshot_lock:
# a context that finds another one there took the run id over (_owe). An int per destination, the most recent kept.
_ROW_OWNERS: OrderedDict[str, int] = OrderedDict()
_ROW_OWNERS_MAX = 16384


def _owe(target: tuple, run_context: RunContext, provider: type) -> _Write | None:
    """Note a save of ``run_context`` asked of the destination ``target`` (``_target``: config, run path, key) by
    ``provider``: it is what that destination is owed from now on (it covers every earlier request).

    ``None``, and nothing noted, for an older context of a run id used again (the HTTP job store allows a finished
    id to be used again): the id belongs to the newer run, and a late save of the old one (its keep-alive, work
    that outlived it, a buffered upload) must not take its place. ``None`` too for a copy captured before what was
    last noted or sent for the run there (``_platform_noted``: a copy's capture number, the number taken when a
    save of the run itself is noted and when a PATCH serializes it): a subclass's ``_store`` that awaited before
    passing its copy on, an exporter that forwarded a copy late, a keep-alive holding an older copy. It holds an
    older state than what the destination has or is owed.

    Never pushes out a request no PATCH has carried yet, or one a round is sending: ``_unsaved`` keeps at most
    ``_UNSAVED_MAX`` destinations that are waiting for a resend, and goes over it rather than drop a save nothing has
    sent yet."""
    platform_config, run_path, key = target
    context, redactors = _unwrap(run_context, provider._trace_redactor)
    made = run_context.__dict__.get("_made") if isinstance(run_context, _RedactedRunContextView) else None
    unsaved = PlatformTracingProvider._unsaved
    evicted = []
    with _snapshot_lock:
        own = context.__dict__
        use = own.get("_platform_use")
        if use is None:
            use = own["_platform_use"] = next(_USES)
        latest = _LATEST.get(key)
        if latest is None:
            latest = _LATEST[key] = _Latest(use)
        own.setdefault("_platform_latest", {})[key] = latest
        owner = _ROW_OWNERS.get(key)
        if use > latest.use or (owner is not None and owner != use and key not in own.get("_platform_owned", ())):
            # It takes over a run id another context saved here (one still alive, or one whose PATCH this process
            # sent): whole until a whole save of it lands (_save), so its first PATCH replaces the span rows the
            # other run left instead of being merged into them.
            own.setdefault("_platform_replacing", set()).add(key)
        noted = own.setdefault("_platform_noted", {})
        replaced = use < latest.use
        if made is None:
            made = next(_COPIES)  # the run itself, read when its PATCH goes out: no older than now
        if replaced or made < noted.get(key, 0):
            write = None
        else:
            noted[key] = made
            latest.use = use
            write = unsaved[key] = _Write(
                context, run_context, redactors, provider, platform_config, run_path, next(_REQUESTS)
            )
            unsaved.move_to_end(key)
            if len(unsaved) > _UNSAVED_MAX:
                # The oldest first, among those waiting for a resend: never one not sent yet (this one, or one another
                # thread noted and is about to send), nor one with a PATCH out (on any loop), nor one a round is
                # sending.
                flights = PlatformTracingProvider._flights
                waiting = [
                    other
                    for other in unsaved
                    if unsaved[other].sent and not unsaved[other].out and other not in flights
                ]
                for old in waiting[: len(unsaved) - _UNSAVED_MAX]:
                    evicted.append(unsaved.pop(old).id)
    if replaced:
        logger.warning("platform_trace_older_context_dropped", run_id=context.id)
    for run_id in evicted:
        # A shutdown flush can no longer send or report it.
        logger.error("platform_trace_unsaved_evicted", run_id=run_id, max_runs=_UNSAVED_MAX)
    return write


def _landed(key: str, seq: int) -> None:
    """A PATCH carrying the request numbered ``seq`` was stored at ``key``: what that destination is owed is
    settled if it is that request or an older one. A newer one (asked meanwhile, whose own save may have
    failed) stays owed."""
    unsaved = PlatformTracingProvider._unsaved
    with _snapshot_lock:
        write = unsaved.get(key)
        if write is not None and write.seq <= seq:
            del unsaved[key]


class PlatformTracingProvider(TracingProvider):
    """Platform tracing provider using platform-wide storage.

    Can be configured with an explicit ``PlatformConfig`` via ``configured()``::

        from timbal.state.config import PlatformConfig, PlatformAuth, PlatformAuthType, PlatformSubject

        provider = PlatformTracingProvider.configured(
            platform_config=PlatformConfig(
                host="api.timbal.ai",
                auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_..."),
                subject=PlatformSubject(org_id="1", app_id="400"),
            )
        )
        agent = Agent(model="...", tracing_provider=provider)

    When ``platform_config`` is not set, credentials are resolved from the
    run context and, as a last resort, from environment variables
    (``TIMBAL_API_KEY``, ``TIMBAL_ORG_ID``, ``TIMBAL_APP_ID``).

    The traces this process stored most recently are read back from memory:
    a voice call starts each turn from the one before it, in the same process,
    a moment after storing it, and would otherwise fetch it over the network
    twice (memory and session data) before the model is called.

    ``TIMBAL_TRACE_CACHE_MAX_ENTRIES`` (default 256) and
    ``TIMBAL_TRACE_CACHE_TTL_SECS`` (default 300) control this process-wide
    cache. Either set to zero disables caching. The cache also holds at most
    64 MiB of serialized traces. Expired entries are removed on cache reads
    and writes; there is no background cleanup task.

    Reading a run this process is still saving waits for those saves first,
    up to ``_save_wait_s`` (default 2 s), cache or no cache: until the PATCH
    lands, the platform may answer 404 or an older snapshot.
    """

    platform_config: PlatformConfig | None = None
    """The platform config of runs that use this provider, and of their traces: a run started without one
    takes it (``RunContext``), and its model router, tools and files use it too."""

    trace_platform_config: PlatformConfig | None = None
    """Credentials for the traces alone (``configured(trace_platform_config=...)``): every request of this
    provider uses it, and it never becomes a run's config, which stays what the run resolved for itself
    (passed, ``platform_config`` above, or the environment). For a provider that stores runs somewhere other
    than an app's runs (:meth:`_run_path`), or with a token of its own."""

    default_durability = "buffered"
    """A save is a PATCH round trip: awaiting one per finished span adds seconds to a long turn. Saves while
    the run goes are stored in the background instead; its end, and every other barrier, still waits.
    ``durability="sync"`` or ``TIMBAL_TRACE_DURABILITY=sync`` awaits every save again."""

    overlapping_saves = True
    """``_store`` keeps a run's saves in order itself, so ``uploads.write_through`` does not queue them: one PATCH on
    the wire per run and loop, every PATCH sending the newest save asked of the destination as it is at its
    number (``_save``), each numbered (``X-Timbal-Snapshot``) so the host drops one that is not newer than the
    last it applied."""

    compress: bool | None = None
    """Gzip each PATCH body (``Content-Encoding: gzip``). ``None``: when ``TIMBAL_TRACE_GZIP`` is set, or
    once the endpoint advertised it (``Accept-Encoding: gzip`` on a response, RFC 7694; for an app's runs, the
    host)."""

    deltas: bool | None = None
    """Send span deltas instead of whole traces (``Content-Type: application/vnd.timbal.trace-delta+json``):
    a save carries the finished spans the platform has not acknowledged and a root header; the run's end
    carries the full root. ``None``: once the endpoint advertised it (``Accept-Patch``, RFC 5789). An endpoint
    that rejects either format (400/415/422) while it takes the same save without it falls back to whole
    uncompressed traces for good, forced or not (per endpoint: the host, for an app's runs; see ``_family``)."""

    keepalive_s: float | None = None
    """While a run is in flight and has not saved for this long (a long tool call, a slow model), it saves
    again, so the platform never takes it for dead: past ``RUN_TIMEOUT_MINUTES`` (30 on the platform) without
    a save, a running run reads ``timeout``. With deltas the save is the root header alone. ``None``:
    ``TIMBAL_TRACE_KEEPALIVE_SECS``, default 300; 0 turns it off. Runs that save only at their end
    (``durability="end"``) never keep alive. Keep-alive saves are not the run's progress: one that has not
    saved for itself in ``TIMBAL_TRACE_KEEPALIVE_MAX_SECS`` (default 4 h; a hung tool) stops, and reads
    ``timeout`` again."""

    # host (app runs) or "host family" -> what it advertised and refused. Families are bounded (_CAPS_FAMILIES_MAX).
    _caps: ClassVar[OrderedDict[str, _HostCaps]] = OrderedDict()
    _delta_runs: ClassVar[OrderedDict[str, _DeltaRun]] = OrderedDict()
    # run key -> its keep-alive timer, while the run is in flight and has saved at least once.
    _keepalives: ClassVar[OrderedDict[str, _KeepAlive]] = OrderedDict()

    # run key -> (monotonic time of the last successful save, the exact PATCH body sent)
    _recent: ClassVar[OrderedDict[str, tuple[float, bytes, str | None]]] = OrderedDict()
    # Snapshots are numbered as they are taken (_next_snapshot): saves of one run overlap (parallel
    # tools, background children) and their PATCHes can finish in any order.
    # run key -> the run's saves in flight. Kept apart from _recent so eviction cannot lose it, and dropped
    # with the run's last save in flight: any later save takes a higher number.
    _saving: ClassVar[dict[str, _Saves]] = {}
    # run key -> its single-flight state, while it has a round queued or a driver sending.
    _flights: ClassVar[dict[str, _Flight]] = {}
    # How long a run's end waits behind a PATCH still on the wire before going anyway.
    _stall_s: ClassVar[float] = 10.0
    # run key -> its latest context, while a save asked for has not landed (failed, or still on its way).
    _unsaved: ClassVar[OrderedDict[str, _Write]] = OrderedDict()
    # run key -> the snapshots this process sent for it (0.1: a newer stored one that is not ours is a takeover).
    _own_snapshots: ClassVar[OrderedDict[str, deque[int]]] = OrderedDict()
    # Event loops a flush is running on: their runs' ends go now, and their retries and keep-alives stand down.
    _closing_loops: ClassVar[set] = set()
    _exit_hook: ClassVar[bool] = False
    # run key -> the background task resending an ended run whose last save failed on a transient error.
    _retries: ClassVar[dict[str, asyncio.Task]] = {}
    # run key -> runs the platform refused as too large (413): saved whole with long strings cut from then on.
    _oversized: ClassVar[OrderedDict[str, None]] = OrderedDict()
    # Waits between those resends (the last repeats), and how long to keep trying before giving up.
    _retry_delays_s: ClassVar[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
    _retry_window_s: ClassVar[float] = 600.0
    # The PATCH read timeout when the client has none (its default): the platform parses the trace first.
    _patch_read_timeout_s: ClassVar[float] = 120.0
    # How long a read waits for saves of that run already in flight before asking the platform.
    _save_wait_s: ClassVar[float] = 2.0
    # One entry per live conversation: concurrent calls sharing a process must not evict each other between turns.
    _recent_max: ClassVar[int] = 256
    _recent_max_bytes: ClassVar[int] = 64 * 1024 * 1024
    # Another process may write the same run later (an approval resumed
    # elsewhere); past this, the platform's copy is the one to trust.
    _recent_ttl_s: ClassVar[float] = 300.0

    @classmethod
    def _run_path(cls, platform_config: PlatformConfig, run_id: str) -> str:
        """Where the run ``run_id`` is stored and read: ``orgs/{org_id}/apps/{app_id}/runs/{run_id}``, an app's runs.

        A subclass that stores runs somewhere else on the platform (runs that belong to no app) overrides this,
        and ``_resolve_config`` for its credentials, and gets the whole save pipeline as it is: span deltas,
        gzip, snapshot ordering, keep-alive saves, single flight, retries, reads. Every request and every
        per-run key of this provider comes from it. What a host advertises is learned per host for an app's
        runs (as before), and for any other run per destination, unless the path it returns is a
        :class:`RunPath` that declares the family its runs share (:meth:`_family`). A run it cannot place (a registry that does
        not know it) raises :class:`RunNotStored`: reads answer it as not stored (``None``, as a 404), saves
        fail. Anything else it raises is a bug or a configuration error and goes to the caller.
        """
        subject = platform_config.subject
        if subject is None:
            raise RuntimeError("PlatformTracingProvider requires a subject (org_id + app_id) in platform_config.")
        if not subject.app_id:
            raise ValueError("Cannot use platform tracing provider without an app or project subject")
        return f"orgs/{subject.org_id}/apps/{subject.app_id}/runs/{run_id}"

    @staticmethod
    def _row_key(platform_config: PlatformConfig, run_path: str) -> str:
        """Where a run is stored: the destination its saves are owed and ordered by, whatever credentials send them
        (two providers saving one run there are one destination)."""
        return f"{platform_config.host}/{run_path}"

    @staticmethod
    def _recent_key(platform_config: PlatformConfig, run_path: str) -> str:
        """A run is read back only by the credentials that stored it, at the path it was stored at,
        so memory never answers a request the platform would have refused."""
        token = hashlib.sha256(platform_config.auth.token.get_secret_value().encode()).hexdigest()
        return f"{platform_config.host}/{run_path}#{token}"

    @classmethod
    def _recent_limits(cls) -> tuple[int, float]:
        """Read process settings; invalid values fall back to the defaults."""
        return _parse_cache_limits(
            os.environ.get("TIMBAL_TRACE_CACHE_MAX_ENTRIES"),
            os.environ.get("TIMBAL_TRACE_CACHE_TTL_SECS"),
            cls._recent_max,
            cls._recent_ttl_s,
        )

    @classmethod
    def _prune_recent(cls, now: float, capacity: int, ttl: float) -> None:
        recent = PlatformTracingProvider._recent
        size = sum(len(entry[1]) for entry in recent.values())
        # Entries are ordered by their last successful save, oldest first.
        # Purge expired entries even when the requested key is absent.
        while recent:
            saved_at, body, _ = next(iter(recent.values()))
            if len(recent) <= capacity and size <= cls._recent_max_bytes and now - saved_at < ttl:
                break
            recent.popitem(last=False)
            size -= len(body)

    @classmethod
    def _remember(cls, key: str, snapshot: int, body: bytes | None, token: str | None = None) -> None:
        """Settle a finished save of ``snapshot`` at the destination ``key``; ``body`` is None when the save failed,
        ``token`` the credentials that sent it (``_token``): the cached body is read back only by them. One entry
        per destination, whatever credentials sent it: the newest snapshot there replaces it."""
        capacity, ttl = cls._recent_limits()
        now = time.monotonic()
        newest = -1
        saves = PlatformTracingProvider._saving.get(key)
        if saves is not None:
            settled = saves.pending.pop(snapshot, None)
            if settled is not None and not settled.done():
                settled.set_result(None)
            if not saves.pending:
                del PlatformTracingProvider._saving[key]
            newest, saves.newest = saves.newest, max(saves.newest, snapshot)
        # An older snapshot finishing after a newer one leaves the cache as the newer one left it.
        if snapshot > newest:
            recent = PlatformTracingProvider._recent
            # Popped even when the body is not kept: the platform may hold this snapshot now.
            recent.pop(key, None)
            if body is not None and len(body) <= cls._recent_max_bytes:
                recent[key] = (now, body, token)
        cls._prune_recent(now, capacity, ttl)

    @classmethod
    def _recall(cls, key: str, token: str | None = None) -> Trace | None:
        capacity, ttl = cls._recent_limits()
        cls._prune_recent(time.monotonic(), capacity, ttl)
        entry = PlatformTracingProvider._recent.get(key)
        if entry is None or entry[2] != token:
            return None  # not stored here, or stored by other credentials: the platform decides
        # Parsed afresh each time, so a reader can never change what the next one sees.
        stored = json.loads(entry[1])
        return Trace(stored["trace"], parent_id=stored.get("parent_id"))

    @staticmethod
    def _host_caps(host: str, family: str = "", *, create: bool = True) -> _HostCaps:
        """What ``host`` advertised for the runs of ``family`` (:meth:`_family`; ``""``: an app's runs).

        ``create=False``: a family not seen yet is answered with nothing learned and not stored. App runs keep
        one entry per host. Declared families (``=``) and runs' own destinations (``@``) are each the least
        recently used ``_CAPS_FAMILIES_MAX``, bounded apart, so thousands of runs never push out the endpoint
        every one of them leans on, nor the reverse (one evicted learns again what its endpoint takes, and may
        have one save refused again).
        """
        caps_by_key = PlatformTracingProvider._caps
        key = f"{host} {family}" if family else host
        with _CAPS_LOCK:
            caps = caps_by_key.get(key)
            if caps is None:
                caps = _HostCaps()
                if not create:
                    return caps
                caps_by_key[key] = caps
                if family and len(caps_by_key) > _CAPS_FAMILIES_MAX:
                    # "=" declared, "@" a run's own destination: the character after the host (hosts have no
                    # space), not a substring of the name, which may contain anything.
                    same = [known for known in caps_by_key if known.partition(" ")[2][:1] == family[0]]
                    for oldest in same[: max(0, len(same) - _CAPS_FAMILIES_MAX)]:
                        del caps_by_key[oldest]
            elif family:
                caps_by_key.move_to_end(key)
        return caps

    @classmethod
    def _family(cls, run_path: str | None) -> str | None:
        """The key a run's capabilities are kept under besides its host.

        What a host advertised and refused is kept per endpoint (one may take deltas, gzip or the
        ``select=resume`` projection while another does not yet, and a refusal is for good, see ``_HostCaps``),
        and the provider does not guess which paths share one:

        - ``""`` for an app's runs (the default ``_run_path``): their host's entry, as before;
        - ``"=" + family`` for a :class:`RunPath` that declares it: every run of that endpoint shares it;
        - ``"@" + run_path`` otherwise: the run's own destination, one endpoint whatever routes there. Each run
          then learns on its own: its first save goes whole, a format the endpoint advertises (or is forced)
          and then refuses costs one refused save per run, a parent read in another process rarely gets the
          projection (bytes, never a wrong format; the cache's bound keeps the memory flat). So does a
          subclass whose ``_run_path`` only wraps ``super()._run_path``: it is not the default one;
        - ``None`` without a run (``takes_deltas``) for any run path but the default: no destination.
        """
        if getattr(cls._run_path, "__func__", None) is _DEFAULT_RUN_PATH:
            return ""
        if run_path is None:
            return None
        family = getattr(run_path, "family", None)
        return f"={family}" if family else f"@{run_path}"

    @classmethod
    def _family_log(cls, run_path: str) -> dict[str, str]:
        """The family a log line names: none for an app's runs, whose lines stay as they were."""
        family = cls._family(run_path)
        return {"family": family} if family else {}

    @classmethod
    def _family_caps(
        cls, platform_config: PlatformConfig, run_path: str | None = None, *, create: bool = True
    ) -> _HostCaps:
        """What the endpoint of ``run_path`` on ``platform_config``'s host advertised, and what it turned down
        (:meth:`_family`). Nothing learned, and nothing stored, without a run for a non-default ``_run_path``."""
        family = cls._family(run_path)
        if family is None:
            return _HostCaps()
        return cls._host_caps(platform_config.host, family, create=create)

    @classmethod
    def _compressing(cls, caps: _HostCaps) -> bool:
        if caps.gzip_rejected:
            return False
        if cls.compress is not None:
            return cls.compress
        if os.environ.get("TIMBAL_TRACE_GZIP", "").strip().lower() in ("1", "true", "yes"):
            return True
        return caps.gzip

    @classmethod
    def _sending_deltas(cls, caps: _HostCaps) -> bool:
        if caps.delta_rejected:
            return False
        return cls.deltas if cls.deltas is not None else caps.delta

    @classmethod
    def takes_deltas(cls, platform_config: PlatformConfig | None = None) -> bool:
        """Whether saves to this provider's endpoint go out as span deltas right now (each save is then small).

        For the config a save of a run whose own config is ``platform_config`` would use: ``_resolve_config``
        is asked with a stand-in run holding only that (so ``trace_platform_config`` comes first, then
        ``platform_config``, then the class's, then the environment's, and a subclass's own
        ``_resolve_config`` is followed; one that needs more of the run than its config answers False). False
        when it finds none. Asking stores nothing. For a ``_run_path`` other than the default there is no
        destination without a run: only a forced ``deltas=True`` answers True, whatever an endpoint refused.
        """
        try:
            cfg = cls._resolve_config(SimpleNamespace(platform_config=platform_config))
        except Exception:
            return False
        return cfg is not None and cls._sending_deltas(cls._family_caps(cfg, create=False))

    @staticmethod
    def _delta_run(key: str) -> _DeltaRun:
        runs = PlatformTracingProvider._delta_runs
        run = runs.get(key)
        if run is None:
            run = runs[key] = _DeltaRun()
            while len(runs) > _DELTA_RUNS_MAX:  # a run whose process died never ends
                runs.popitem(last=False)
        else:
            runs.move_to_end(key)
        return run

    @classmethod
    def _resolve_config(cls, run_context: RunContext):
        """Return platform_config, with the following priority:

        0. cls.trace_platform_config    — the traces' own credentials; returned as is, never written back
        1. run_context.platform_config  — set by auto-detection or a previous call
        2. cls.platform_config          — set explicitly via configured()
        3. env vars / config file       — resolved lazily, written back to run_context

        The resolved config is written back to run_context, so the run's other
        platform calls find it too. This provider's own requests carry the config
        this returns. Credentials for the traces alone go in
        ``trace_platform_config`` (or a subclass returns its own here, without
        writing them onto the run): the run keeps its config for its model
        router, tools and files.
        """
        if cls.trace_platform_config is not None:
            return cls.trace_platform_config  # for the traces alone: never written onto the run
        cfg = run_context.platform_config or cls.platform_config
        if cfg is None:
            from ....state.config_loader import resolve_platform_config

            cfg = resolve_platform_config(None)
            if cfg is None:
                raise RuntimeError(
                    "PlatformTracingProvider requires platform credentials. "
                    "Set TIMBAL_API_KEY, TIMBAL_ORG_ID, and TIMBAL_APP_ID environment variables, "
                    "or pass platform_config=... to configured()."
                )
        run_context.platform_config = cfg
        return cfg

    @classmethod
    @override
    async def get_strict(cls, run_context: RunContext) -> Trace | None:
        """See base class: a 404 is a run the platform does not have; other failures raise.

        Always the whole trace, never the ``select=resume`` projection ``get`` asks for (the root and the agent
        spans, what the next turn's memory needs). Strict reads are the ones that decide what a stored run
        left: restoring a session (``ChatSession``: the approvals and questions it waits for are on the tool
        spans), recovering a dead turn (which calls completed, by their tool spans and model calls), the tool
        policy a resume keeps (nested agents note theirs on the call's span), and the approval claims of a
        provider that makes them (this one does not, yet).
        """
        import httpx

        from ....errors import PlatformError

        whole = _WHOLE_TRACE.set(True)
        try:
            return await cls.get(run_context)
        except PlatformError as error:
            # _request turns a 4xx (but 429) into PlatformError before get() sees the response.
            if error.status_code == 404:
                return None
            raise
        except httpx.HTTPStatusError as error:
            if error.response is not None and error.response.status_code == 404:
                return None
            raise
        finally:
            _WHOLE_TRACE.reset(whole)

    @classmethod
    @override
    async def get(cls, run_context: RunContext) -> Trace | None:
        """See base class."""
        from ....platform.utils import _request

        platform_config = cls._resolve_config(run_context)
        try:
            run_path = cls._run_path(platform_config, run_context.parent_id)
        except RunNotStored:
            return None  # _run_path does not know the run (a registry miss): not stored here, as a 404
        key = cls._row_key(platform_config, run_path)
        loop = asyncio.get_running_loop()
        saves = PlatformTracingProvider._saving.get(key)
        flight = PlatformTracingProvider._flights.get(key)
        # A save on another event loop cannot be awaited here: its future belongs to that loop.
        waits = [save for save in saves.pending.values() if save.get_loop() is loop] if saves is not None else []
        if flight is not None and flight.loop is loop:
            waits += flight.waits()  # a round queued behind one on the wire is a save already asked for
        if waits:
            # The parent's save is still in flight (a voice turn's shielded save racing the next turn): the
            # platform could answer 404, or an older snapshot. Only saves already asked for are awaited, so a
            # parent that keeps saving (background children) does not hold every read for the full bound.
            await asyncio.wait(waits, timeout=cls._save_wait_s)
        recalled = cls._recall(key, _token(platform_config))
        if recalled is not None:
            return recalled
        caps = cls._family_caps(platform_config, run_path)
        with _pooled(), _as_run(run_context):
            res = await _request(
                method="GET",
                path=run_path,
                platform_config=platform_config,
                # The root and the agent spans: what resuming reads, without the LLM and tool spans' bytes. Not
                # for a strict read (get_strict), which needs the tool spans too.
                params={"select": "resume"} if caps.select_resume and not _WHOLE_TRACE.get() else None,
            )
        res.raise_for_status()
        caps.learn(getattr(res, "headers", None))
        res_json = res.json()
        trace = res_json.get("trace")
        if not trace:
            return None
        # `parent_id` is the parent's row id; `parent_idempotency_key` (newer hosts) is the run id this SDK
        # chained with, which its caches and in-flight saves are keyed by. A GET by either works.
        return Trace(trace, parent_id=res_json.get("parent_idempotency_key") or res_json.get("parent_id"))

    @classmethod
    @override
    async def put(cls, run_context: RunContext) -> None:
        """See base class. The built-in store keeps the run itself and redacts each PATCH when it is sent
        (``_save``): a copy made here could be sent after a newer save of the run, by a flush, a background resend
        or another event loop. A subclass that stores runs its own way (it overrides ``_store``) gets the redacted
        copy, as from the base class; if it hands it on to the built-in ``_store``, it hands on what it was given
        (edited or not) or a ``_RedactedRunContextView`` of it, never another ``RunContext``: a ``model_copy()`` is
        taken for the run itself, read when its PATCH goes out, and nothing can tell how old it is. Exporters get
        the redacted copy either way."""
        if getattr(cls._store, "__func__", None) is not _BUILTIN_STORE:
            await super().put(run_context)
            return
        exported = run_context
        if cls._trace_redactor is not None and cls._exporters:
            exported = _capture(run_context, cls._trace_redactor)
        await cls._store(run_context)
        for exporter in cls._exporters:
            try:
                await exporter.export(exported)
            except Exception:
                pass

    @classmethod
    @override
    async def _store(cls, run_context: RunContext) -> None:
        """See base class. Notes the save as what the run's destination is owed (``_owe``), with that destination
        resolved now and kept, then sends it (``_send``). A save of an older context of a run id used again, or of a
        copy older than what was noted, is dropped."""
        target = cls._target(run_context)
        if _owe(target, run_context, cls) is None:
            return  # an older context of a run id used again, or an older copy: the destination has newer
        key = target[2]
        cls._arm_keepalive(key, run_context, asyncio.get_running_loop())
        await cls._send(key)

    @classmethod
    async def _send(cls, key: str) -> None:
        """Send what the destination ``key`` is owed, on this loop. Single-flight per destination and loop: a send
        that arrives while a PATCH is on the wire joins the next round, which goes out once, with the newest save
        asked, when that one is back (or has stalled for ``_stall_s``). Waits for the round carrying it;
        cancelling the caller does not cancel the save."""
        loop = asyncio.get_running_loop()
        flights = PlatformTracingProvider._flights
        with _snapshot_lock:  # other loops' drivers replace and remove flights too
            flight = flights.get(key)
            if flight is None or flight.loop is not loop:
                flight = flights[key] = _Flight(key, loop)
        if not PlatformTracingProvider._exit_hook:
            PlatformTracingProvider._exit_hook = True
            atexit.register(_flush_at_exit)
        if flight.next is None:
            flight.next = loop.create_future()
            flight.next.add_done_callback(_retrieve)
        joined = flight.next
        if flight.arrival is not None and not flight.arrival.done():
            flight.arrival.set_result(None)
        if flight.driver is None:
            flight.driver = loop.create_task(cls._drive(flight))
        await asyncio.shield(joined)

    @classmethod
    def _keepalive_interval(cls) -> float:
        if cls.keepalive_s is not None:
            return cls.keepalive_s
        return _env_seconds("TIMBAL_TRACE_KEEPALIVE_SECS", 300.0)

    @classmethod
    def _arm_keepalive(cls, key: str, run_context: RunContext, loop: asyncio.AbstractEventLoop) -> None:
        """(Re)start the run's keep-alive timer from this save; an ended run, or one hung too long, drops it."""
        keepalives = PlatformTracingProvider._keepalives
        entry = keepalives.pop(key, None)
        if entry is not None:
            entry.cancel()
        interval = cls._keepalive_interval()
        if not interval or _ended(run_context):
            return
        now = loop.time()
        progress = now
        if _KEEPALIVE.get() and entry is not None:
            progress = entry.progress
            if now - progress >= _env_seconds("TIMBAL_TRACE_KEEPALIVE_MAX_SECS", 4 * 3600.0):
                logger.warning("platform_trace_keepalive_stopped", run_id=run_context.id, idle_s=round(now - progress))
                return
        entry = keepalives[key] = _KeepAlive(run_context, cls, loop, progress)
        entry.handle = loop.call_later(interval, PlatformTracingProvider._keep_alive, key)
        while len(keepalives) > _KEEPALIVES_MAX:
            _, dropped = keepalives.popitem(last=False)
            dropped.cancel()

    @staticmethod
    def _keep_alive(key: str) -> None:
        entry = PlatformTracingProvider._keepalives.get(key)
        if entry is None or entry.loop.is_closed() or entry.loop in PlatformTracingProvider._closing_loops:
            return
        entry.handle = None
        if _ended(entry.context):
            PlatformTracingProvider._keepalives.pop(key, None)  # its end is being saved, or was
            return
        idle = entry.loop.time() - entry.progress
        if idle >= _env_seconds("TIMBAL_TRACE_KEEPALIVE_MAX_SECS", 4 * 3600.0):
            logger.warning("platform_trace_keepalive_stopped", run_id=entry.context.id, idle_s=round(idle))
            PlatformTracingProvider._keepalives.pop(key, None)
            return
        # The next one now, whatever this save does: a save the store drops (an older copy) must not end the chain.
        # A save it takes arms it again from its own start (_arm_keepalive).
        interval = entry.provider._keepalive_interval()
        entry.handle = entry.loop.call_later(interval, PlatformTracingProvider._keep_alive, key)
        context = contextvars.copy_context()
        context.run(_KEEPALIVE.set, True)
        task = entry.loop.create_task(entry.provider._store(entry.context), context=context)
        task.add_done_callback(_keepalive_done)

    @classmethod
    def _send_in(cls, flight: _Flight, now: float) -> float:
        """Seconds until the next round may go out (0: now; inf: only once something changes)."""
        if flight.next is None:
            return math.inf
        if not flight.on_wire:
            return 0.0
        # A second PATCH only for the run's end, and only past a stalled one: mid-run rounds are superseded
        # anyway, and more overlap when the platform is slow would feed the slowness.
        if len(flight.on_wire) >= _MAX_ON_WIRE or not flight.terminal():
            return math.inf
        if flight.loop in PlatformTracingProvider._closing_loops:
            return 0.0
        return max(0.0, min(flight.on_wire.values()) + cls._stall_s - now)

    @classmethod
    async def _drive(cls, flight: _Flight) -> None:
        loop = asyncio.get_running_loop()
        try:
            while flight.next is not None or flight.on_wire:
                wait = cls._send_in(flight, loop.time())
                if wait == 0:
                    joined, flight.next = flight.next, None
                    flight.sent.append(joined)
                    save = loop.create_task(cls._save(flight.key))
                    flight.on_wire[save] = loop.time()
                    save.add_done_callback(functools.partial(_settle, flight, joined))
                    continue
                flight.arrival = loop.create_future()
                await asyncio.wait(
                    {*flight.on_wire, flight.arrival},
                    timeout=None if math.isinf(wait) else wait,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                flight.arrival = None
        finally:
            flight.driver = None
            if flight.next is not None and not flight.next.done():
                flight.next.cancel()  # the driver itself was cancelled (loop shutting down)
            with _snapshot_lock:  # another loop may be replacing it right now
                if PlatformTracingProvider._flights.get(flight.key) is flight:
                    del PlatformTracingProvider._flights[flight.key]

    @staticmethod
    def _schedule_retry(key: str, loop: asyncio.AbstractEventLoop) -> None:
        """Keep resending what an ended run's destination is owed after its save failed on a transient error, in
        the background, until one lands, a shutdown flush takes over, or ``_retry_window_s`` runs out.
        ``_request``'s own retries span under a second: a platform restart outlasts them, and the run would stay
        ``running`` until this process exits."""
        retries = PlatformTracingProvider._retries
        task = retries.get(key)
        if (task is not None and not task.done()) or loop in PlatformTracingProvider._closing_loops or loop.is_closed():
            return
        retries[key] = loop.create_task(PlatformTracingProvider._retry(key))

    @staticmethod
    async def _retry(key: str) -> None:
        loop = asyncio.get_running_loop()
        write = PlatformTracingProvider._unsaved.get(key)
        settings = write.provider if write is not None else PlatformTracingProvider
        deadline = loop.time() + settings._retry_window_s
        delays = settings._retry_delays_s
        try:
            for attempt in itertools.count():
                delay = delays[min(attempt, len(delays) - 1)]
                if loop.time() + delay > deadline:
                    break
                await asyncio.sleep(delay)
                write = PlatformTracingProvider._unsaved.get(key)
                if write is None or loop in PlatformTracingProvider._closing_loops:
                    return  # landed with another save, or a shutdown flush on this loop is sending it
                if uploads.tracked(write.context):
                    continue  # buffered uploads hold the run and retry it themselves; check again later
                try:
                    await write.provider._send(key)
                    logger.info("platform_trace_retry_landed", run_id=write.id, attempts=attempt + 1)
                    return
                except Exception as e:
                    if not _transient(e):
                        logger.error("platform_trace_retry_rejected", run_id=write.id, error=str(e))
                        return
                    logger.warning("platform_trace_retry_failed", run_id=write.id, attempt=attempt + 1, error=str(e))
            write = PlatformTracingProvider._unsaved.get(key)
            if write is not None and uploads.tracked(write.context):
                # Every turn was skipped for buffered uploads: they own the run, nothing was given up.
                logger.info("platform_trace_retry_handed_to_uploads", run_id=write.id)
            elif write is not None:
                logger.error("platform_trace_retry_exhausted", run_id=write.id, window_s=settings._retry_window_s)
        finally:
            if PlatformTracingProvider._retries.get(key) is asyncio.current_task():
                del PlatformTracingProvider._retries[key]

    @classmethod
    async def flush(cls, timeout: float | None = None) -> list[str]:
        """Send every save this process still owes the platform, and wait for them (process shutdown).

        Runs' ends go first and no longer wait out a stalled PATCH; runs whose last save failed, or died
        with an earlier event loop, are sent again from their latest state. Bounded by ``timeout``
        (default ``TIMBAL_TRACE_FLUSH_TIMEOUT_SECS``, 5 s: inside a pod's grace period). Returns, and
        logs, the run ids whose saves did not land.
        """
        if timeout is None:
            timeout = _flush_timeout()
        loop = asyncio.get_running_loop()
        PlatformTracingProvider._closing_loops.add(loop)
        keepalives = PlatformTracingProvider._keepalives
        for key in [key for key, entry in list(keepalives.items()) if entry.loop is loop or entry.loop.is_closed()]:
            keepalives.pop(key).cancel()
        try:
            for task in list(PlatformTracingProvider._retries.values()):
                if task.get_loop() is loop:
                    task.cancel()  # the flush sends their runs now, within its bound
            flights = PlatformTracingProvider._flights
            for flight in list(flights.values()):
                if flight.loop is loop and flight.arrival is not None and not flight.arrival.done():
                    flight.arrival.set_result(None)  # re-evaluate with _closing set
            with _snapshot_lock:  # other event loops ask and settle saves meanwhile
                owed = list(PlatformTracingProvider._unsaved.items())
            # Every destination owed, also one with a round of this loop still out: a run's end that failed past a
            # stalled PATCH has no round queued, and nothing else would send it. A send joins a round queued.
            owed.sort(key=lambda owed_write: not _ended(owed_write[1].context))
            # Each with the provider that accepted it (its redaction policy, its configuration).
            resends = [loop.create_task(write.provider._send(key)) for key, write in owed]
            for task in resends:
                task.add_done_callback(_retrieve)
            waits = [*resends, *(r for f in list(flights.values()) if f.loop is loop for r in f.waits())]
            if waits:
                await asyncio.wait(waits, timeout=timeout)
            with _snapshot_lock:
                left = [write.id for write in PlatformTracingProvider._unsaved.values()]
            if left:
                logger.error("platform_trace_flush_incomplete", run_ids=left, timeout_s=timeout)
            return left
        finally:
            PlatformTracingProvider._closing_loops.discard(loop)
            await _close_pool()

    @staticmethod
    def _own(key: str) -> deque[int]:
        owned = PlatformTracingProvider._own_snapshots
        snapshots = owned.get(key)
        if snapshots is None:
            snapshots = owned[key] = deque(maxlen=64)
            while len(owned) > _DELTA_RUNS_MAX:
                owned.popitem(last=False)
        else:
            owned.move_to_end(key)
        return snapshots

    @classmethod
    def _patch_timeout(cls):
        """The client's timeouts, with the read bounded when it is not: no save may stay on the wire
        forever now that callers do not cancel it. ``TIMBAL_HTTP_READ_TIMEOUT`` still wins."""
        import httpx

        from ....platform.utils import _default_timeout

        timeout = _default_timeout()
        if timeout.read is not None or os.environ.get("TIMBAL_HTTP_READ_TIMEOUT"):
            return timeout
        return httpx.Timeout(
            connect=timeout.connect, read=cls._patch_read_timeout_s, write=timeout.write, pool=timeout.pool
        )

    @classmethod
    def _target(cls, run_context: RunContext):
        """The config, the run path and the destination key (``_row_key``) of the run ``run_context`` saves."""
        platform_config = cls._resolve_config(run_context)
        run_path = cls._run_path(platform_config, run_context.id)
        return platform_config, run_path, cls._row_key(platform_config, run_path)

    @classmethod
    async def _save(cls, key: str, *, plain: bool = False, resend: bool = True) -> bool:
        """One PATCH of what the destination ``key`` is owed: the newest save asked of it (``_unsaved``), with the
        provider that accepted it, its run as it is now, through its redaction policies. The number and the body
        are taken in one step under ``_snapshot_lock``, so a PATCH numbered later never carries an older request
        or an older state, whichever path sends it (a round, a resend below, a flush on another loop, a retry).

        ``plain``: whole and uncompressed whatever the host advertised. ``resend``: send again, numbered above, if
        the platform kept a newer snapshot this process did not write. Returns True when a PATCH landed, False
        when nothing was owed (a save that landed meanwhile carried every request up to now)."""
        from ....errors import PlatformError
        from ....platform.utils import _request

        while True:
            with _snapshot_lock:
                write = PlatformTracingProvider._unsaved.get(key)
            if write is None:
                return False
            provider = write.provider
            platform_config, run_path = write.platform_config, write.run_path  # the destination it was accepted for
            # A run stored somewhere other than an app's runs (a subclass's _run_path) may have no subject.
            rev = platform_config.subject.rev if platform_config.subject is not None else None
            caps = provider._family_caps(platform_config, run_path)
            # A run the platform refused as too large goes whole with long strings cut, so its end still lands. Only
            # the context it was refused for: a fresh one of the run id starts over.
            shrink = PlatformTracingProvider._oversized.get(key) == write.context.__dict__.get("_platform_use")
            deltas = not plain and not shrink and provider._sending_deltas(caps)
            compressed = not plain and provider._compressing(caps)
            token = _token(platform_config)
            with _snapshot_lock:
                if PlatformTracingProvider._unsaved.get(key) is not write:
                    continue  # a newer save was asked meanwhile (another thread): that is the one to send
                snapshot = _next_snapshot()
                own = write.context.__dict__
                use = own["_platform_use"]  # set by _owe
                policy = _policy(write)
                known = PlatformTracingProvider._delta_runs.get(key)
                if known is not None and (known.use != use or known.policy != policy):
                    # Another context of the run id, or another redaction: what was acknowledged is not what this
                    # save would store. Start over.
                    del PlatformTracingProvider._delta_runs[key]
                # A context that took the run id over from another goes whole until a whole PATCH of it landed here:
                # only a whole trace replaces the span rows the other run left.
                delta = deltas and key not in own.get("_platform_replacing", ())
                # What this PATCH's acknowledgement updates: only the state it was built from.
                delta_run = None
                if deltas:
                    delta_run = provider._delta_run(key)
                    delta_run.use, delta_run.policy = use, policy
                root_stored = False
                # Serialized here, under the capture lock with its number taken first: model_dump() shares the
                # agent's live memory list, which keeps growing while the PATCH is in flight, a PATCH numbered after
                # this one must not carry an older state, and a copy captured before this is older than what it
                # sends (_owe).
                with _base._CAPTURE_LOCK:
                    if write.source is write.context:
                        own["_platform_noted"][key] = next(_COPIES)
                    sending = _redacted(write)
                    if delta:
                        body, cache_body, sent, root_stored = provider._delta_body(sending, rev, key, snapshot)
                    else:
                        trace = sending._trace.model_dump()
                        payload = {"parent_id": sending.parent_id, "trace": _shrink(trace) if shrink else trace}
                        if rev is not None:
                            payload["rev"] = rev
                        body = cache_body = _encode(payload)
                        sent = {}
                        root = next((span for span in trace if span.get("parent_call_id") is None), None)
                        # A whole running root with its memory is what a delta's first root copy is for (_delta_body).
                        root_stored = (
                            not shrink
                            and root is not None
                            and root.get("t1") is None
                            and root.get("memory") is not None
                        )
                # Only now, with a body to send: a serialization that raised (a redactor, a value JSON cannot carry)
                # leaves nothing recorded of a PATCH that never existed.
                write.sent = True
                write.out += 1
                own.setdefault("_platform_owned", set()).add(key)
                _ROW_OWNERS[key] = use
                _ROW_OWNERS.move_to_end(key)
                if len(_ROW_OWNERS) > _ROW_OWNERS_MAX:
                    _ROW_OWNERS.popitem(last=False)
                # Registered in this step, so an older PATCH's answer never finds a newer one of the run finished
                # and unregistered (_remember).
                PlatformTracingProvider._saving.setdefault(key, _Saves()).pending[snapshot] = (
                    asyncio.get_running_loop().create_future()
                )
            break
        try:
            provider._own(key).append(snapshot)
            headers = {
                "Prefer": "return=minimal",  # RFC 7240
                "Content-Type": DELTA_MEDIA_TYPE if delta else "application/json",
                # Lets the platform drop a save that lands after a newer one (it acks it): saves overlap
                # across processes, retries and stalls.
                "X-Timbal-Snapshot": str(snapshot),
            }
            if compressed:
                headers["Content-Encoding"] = "gzip"
            try:
                wire = await _gzip(body) if compressed else body
                with _pooled(), _as_run(write.context):
                    res = await _request(
                        method="PATCH",
                        path=run_path,
                        platform_config=platform_config,
                        headers=headers,
                        content=wire,
                        timeout=provider._patch_timeout(),
                    )
                res.raise_for_status()
            except PlatformError as e:
                provider._remember(key, snapshot, None, token)
                if getattr(e, "status_code", None) == 413 and not shrink:
                    # Over the platform's size limit: every later save of the run would carry the same span and be
                    # refused too, so the run would never store its end. From now on it goes whole and cut down.
                    oversized = PlatformTracingProvider._oversized
                    oversized[key] = write.context.__dict__["_platform_use"]
                    while len(oversized) > _DELTA_RUNS_MAX:
                        oversized.popitem(last=False)
                    logger.warning("platform_trace_too_large", run_id=write.id, bytes=len(body))
                    return await provider._save(key, plain=plain, resend=resend)
                if not (delta or compressed) or getattr(e, "status_code", None) not in (400, 415, 422):
                    raise
                # Today's server answers a gzip body with 400 (its JSON extractor cannot parse it), a server
                # with the layer with 415. The same save whole and uncompressed tells a rejected format from
                # a bad save: only when that one lands is the format off for this endpoint, starting now.
                if not await provider._save(key, plain=True, resend=resend):
                    return False  # nothing was sent whole (a newer save landed meanwhile): nothing learned
                caps.delta_rejected = caps.delta_rejected or delta
                caps.gzip_rejected = caps.gzip_rejected or compressed
                logger.warning(
                    "platform_trace_format_rejected",
                    host=platform_config.host,
                    status=e.status_code,
                    deltas=delta,
                    gzip=compressed,
                    **provider._family_log(run_path),
                )
                return True
            except BaseException:
                # The write may have landed anyway (timeout after commit): never keep serving the older snapshot.
                provider._remember(key, snapshot, None, token)
                raise
            caps.learn(getattr(res, "headers", None))  # also lifts the counter above the stored snapshot
            found = _header_items(getattr(res, "headers", None))
            stored = found.get(SNAPSHOT_HEADER, "").strip()
            if delta and not stored.isdigit():
                # A server that takes deltas orders saves by snapshot and echoes it. Without the echo the body was
                # not understood: today's API reads a delta as a PATCH with no `trace` and answers 204, so the
                # spans did not land. Same handling as a rejected format.
                provider._remember(key, snapshot, None, token)
                if not await provider._save(key, plain=True, resend=resend):
                    return False
                caps.delta_rejected = True
                logger.warning(
                    "platform_trace_delta_unconfirmed",
                    host=platform_config.host,
                    **provider._family_log(run_path),
                )
                return True
            if (
                found.get(APPLIED_HEADER, "").strip().lower() == "false"
                and stored.isdigit()
                and int(stored) > snapshot
                and int(stored) not in provider._own(key)
            ):
                # Another process wrote this run with a clock ahead of ours (a takeover that read nothing
                # first): this save was dropped as stale. Within this process the newer write was ours.
                provider._remember(key, snapshot, None, token)
                if not resend:
                    # Still behind after numbering above it: another process is writing this run right now. It holds the
                    # run; nothing this process was asked is left to send.
                    logger.warning("platform_trace_save_superseded", run_id=write.id, stored=int(stored))
                    _landed(key, write.seq)
                    return True
                logger.info("platform_trace_save_stale", run_id=write.id, sent=snapshot, stored=int(stored))
                return await provider._save(key, plain=plain, resend=False)
            if not delta:
                # A whole trace of this context replaced the platform's span rows: its saves may go as deltas from now on.
                write.context.__dict__.get("_platform_replacing", set()).discard(key)
            current = PlatformTracingProvider._delta_runs.get(key)
            if current is not None and current is delta_run:
                if delta:
                    current.acked.update(sent)
                else:
                    current.acked.clear()  # what the replaced rows acknowledged is gone
                if root_stored:
                    current.root_stored = True
            elif current is not None and not delta:
                current.acked.clear()
            provider._remember(key, snapshot, cache_body, token)
            _landed(key, write.seq)
            return True

        finally:
            with _snapshot_lock:
                write.out -= 1  # this PATCH's answer is in (or never will be)

    @classmethod
    def _delta_body(
        cls, run_context: RunContext, rev: str | None, key: str, snapshot: int
    ) -> tuple[bytes, bytes | None, dict, bool]:
        """A delta save: the spans the platform has not acknowledged, and the whole trace once the root ended.

        While the run goes, finished spans only (they no longer change) and a root header; the root's
        end sends the full root and every span whose content changed since it was acknowledged (a
        background child's late output). Until the platform has acknowledged a full root carrying
        ``memory``, the running root goes whole instead of as a header: a process that dies mid-turn
        then still leaves the turn's input, history and session data stored (one copy per run; a root
        with no memory, like a workflow's, stays small). Returns the body, the full-trace body to
        cache (only once the root ended: that is what the next turn reads back), the fingerprints
        sent, and whether this body stores the root whole with its memory.
        """
        run = cls._delta_run(key)
        spans = run_context._trace.as_records()
        root = next((span for span in spans if span.parent_call_id is None), None)
        ended = root is not None and root.t1 is not None
        out: list[dict] = []
        sent: dict[str, str] = {}
        full: list[dict] = []
        # The root header carries no seq: the platform puts the root at 0, so no other span may take 0, even
        # when a child precedes the root in the trace or the root is not there yet.
        if root is not None:
            run.seq.setdefault(root.call_id, 0)
        for span in spans:
            if span is root:
                continue
            run.seq.setdefault(span.call_id, len(run.seq) + (root is None or root.call_id not in run.seq))
            if not ended and (span.t1 is None or span.call_id in run.acked):
                continue
            dumped = span.model_dump()
            fingerprint = hashlib.blake2b(_encode(dumped), digest_size=16).hexdigest()
            if ended:
                full.append(dumped)
                if run.acked.get(span.call_id) == fingerprint:
                    continue
            out.append({"seq": run.seq[span.call_id], "span": dumped})
            if span.t1 is not None:
                sent[span.call_id] = fingerprint
        root_stored = False
        if root is None:
            root_part = None
        elif ended or not run.root_stored:
            root_part = root.model_dump()
            root_stored = not ended and root_part.get("memory") is not None
        else:
            root_part = {"call_id": root.call_id, "path": root.path, "parent_call_id": None, "t0": root.t0}
        payload = {
            "parent_id": run_context.parent_id,
            "snapshot": snapshot,
            "ended": ended,
            "root": root_part,
            "spans": out,
        }
        if rev is not None:
            payload["rev"] = rev
        cache_body = None
        if ended:
            cache_body = _encode({"parent_id": run_context.parent_id, "trace": [root_part, *full]})
        return _encode(payload), cache_body, sent, root_stored


# Longest strings kept in a trace the platform refused as too large, tried in order until it fits under
# the target (the platform's limit is 50 MB decompressed).
_SHRINK_CAPS = (64 * 1024, 4 * 1024, 512)
_SHRINK_TARGET_BYTES = 40 * 1024 * 1024


def _shrink(trace: list) -> list:
    """A copy of ``trace`` with long strings cut until it fits, the root marked ``metadata.trace_truncated``.

    Walks the whole structure, so it is also the copy taken before the PATCH awaits (``model_dump`` shares
    the agent's live memory). What the platform then stores is what a later read gets back.
    """
    for cap in _SHRINK_CAPS:
        cut = 0

        def walk(value, cap=cap):
            nonlocal cut
            if isinstance(value, str):
                if len(value) <= cap:
                    return value
                cut += 1
                return (
                    f"{value[:cap]}… [{len(value) - cap} characters cut: the trace was over the platform's size limit]"
                )
            if isinstance(value, dict):
                return {k: walk(v) for k, v in value.items()}
            if isinstance(value, list):
                return [walk(v) for v in value]
            return value

        shrunk = walk(trace)
        if len(_encode(shrunk)) <= _SHRINK_TARGET_BYTES:
            break
    root = next((s for s in shrunk if isinstance(s, dict) and s.get("parent_call_id") is None), None)
    if root is not None:
        root["metadata"] = {**(root.get("metadata") or {}), "trace_truncated": {"max_chars": cap, "strings_cut": cut}}
    return shrunk


# The default run path (an app's runs): a provider whose ``_run_path`` is this one shares what a host
# advertised by host alone (``_family``).
_DEFAULT_RUN_PATH = PlatformTracingProvider._run_path.__func__
# The built-in store: a provider that overrides it gets redacted copies (see PlatformTracingProvider.put).
_BUILTIN_STORE = PlatformTracingProvider._store.__func__


def _new_locks_after_fork() -> None:
    """In a forked child: a lock another thread of the parent held is copied held, with no thread left to let
    it go, and the child's first save or read would wait for it for good. And the parent's saves stay the
    parent's: a child that inherited them would resend, at its exit or its flush, the runs as they were when it was
    forked, numbered above what the parent sent since."""
    global _CAPS_LOCK, _snapshot_lock
    _CAPS_LOCK = threading.Lock()
    _snapshot_lock = threading.RLock()
    provider = PlatformTracingProvider
    for state in (
        provider._unsaved,
        provider._flights,
        provider._retries,
        provider._keepalives,
        provider._saving,
        provider._own_snapshots,
        provider._delta_runs,
    ):
        state.clear()
    provider._closing_loops.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_locks_after_fork)


def _encode(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
