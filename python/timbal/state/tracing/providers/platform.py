from __future__ import annotations

import asyncio
import functools
import gzip
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from time import time_ns
from typing import TYPE_CHECKING, ClassVar

import structlog

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..trace import Trace
from .base import TracingProvider

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


class _Flight:
    """One run's saves, single-flight: the round to send next, the rounds on the wire, and the task sending them."""

    __slots__ = ("context", "driver", "loop", "next", "sent")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.context: RunContext | None = None
        self.next: asyncio.Future[None] | None = None
        self.sent: list[asyncio.Future[None]] = []
        self.driver: asyncio.Task | None = None

    def waits(self) -> list[asyncio.Future[None]]:
        """The rounds not settled yet: a read of this run must not overtake them."""
        rounds = [r for r in self.sent if not r.done()]
        if self.next is not None and not self.next.done():
            rounds.append(self.next)
        return rounds


def _retrieve(future: asyncio.Future) -> None:
    """Mark a round's error seen: every caller of it may have been cancelled."""
    if not future.cancelled():
        future.exception()


def _settle(flight: _Flight, joined: asyncio.Future[None], save: asyncio.Task) -> None:
    """A round's save finished. Success also settles earlier rounds still on the wire (a stalled
    PATCH): this one carried their state and newer."""
    if save.cancelled() or save.exception() is not None:
        flight.sent.remove(joined)
        if not joined.done():
            if save.cancelled():
                joined.cancel()
            else:
                joined.set_exception(save.exception())
        return
    index = flight.sent.index(joined)
    settled, flight.sent[: index + 1] = flight.sent[: index + 1], []
    for future in settled:
        if not future.done():
            future.set_result(None)


DELTA_MEDIA_TYPE = "application/vnd.timbal.trace-delta+json"
_DELTA_RUNS_MAX = 2048


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


def _header_items(headers) -> dict[str, str]:
    items = getattr(headers, "items", None)
    if items is None:
        return {}
    return {str(name).lower(): value for name, value in items() if isinstance(value, str)}


SNAPSHOT_HEADER = "x-timbal-snapshot"
SELECT_HEADER = "x-timbal-select"
_last_snapshot = 0


def _next_snapshot() -> int:
    """Strictly increasing within the process and, by the clock, across processes (ns since the epoch).

    The platform ignores a save whose snapshot is not newer than the last it applied to the run, and a
    run can be continued by another process (an approval resumed elsewhere): a per-process counter
    would restart below what the first process sent.
    """
    global _last_snapshot
    _last_snapshot = max(_last_snapshot + 1, time_ns())
    return _last_snapshot


def _observe_snapshot(stored: int) -> None:
    """The platform said it holds ``stored``: number later saves above it, whatever this clock says."""
    global _last_snapshot
    _last_snapshot = max(_last_snapshot, stored)


class _DeltaRun:
    """Spans of one run the platform acknowledged, by fingerprint, and each span's start index (root = 0)."""

    __slots__ = ("acked", "seq")

    def __init__(self) -> None:
        self.acked: dict[str, str] = {}
        self.seq: dict[str, int] = {}


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

    compress: bool | None = None
    """Gzip each PATCH body (``Content-Encoding: gzip``). ``None``: when ``TIMBAL_TRACE_GZIP`` is set, or
    once the host advertised it (``Accept-Encoding: gzip`` on a response, RFC 7694)."""

    deltas: bool | None = None
    """Send span deltas instead of whole traces (``Content-Type: application/vnd.timbal.trace-delta+json``):
    a save carries the finished spans the platform has not acknowledged and a root header; the run's end
    carries the full root. ``None``: once the host advertised it (``Accept-Patch``, RFC 5789). A host that
    rejects either format (400/415/422) while it takes the same save without it falls back to whole
    uncompressed traces for good, forced or not."""

    _caps: ClassVar[dict[str, _HostCaps]] = {}
    _delta_runs: ClassVar[OrderedDict[str, _DeltaRun]] = OrderedDict()

    # run key -> (monotonic time of the last successful save, the exact PATCH body sent)
    _recent: ClassVar[OrderedDict[str, tuple[float, bytes]]] = OrderedDict()
    # Snapshots are numbered as they are taken (_next_snapshot): saves of one run overlap (parallel
    # tools, background children) and their PATCHes can finish in any order.
    # run key -> the run's saves in flight. Kept apart from _recent so eviction cannot lose it, and dropped
    # with the run's last save in flight: any later save takes a higher number.
    _saving: ClassVar[dict[str, _Saves]] = {}
    # run key -> its single-flight state, while it has a round queued or a driver sending.
    _flights: ClassVar[dict[str, _Flight]] = {}
    # How long the next round waits behind a PATCH still on the wire before going anyway.
    _stall_s: ClassVar[float] = 10.0
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

    @staticmethod
    def _recent_key(platform_config: PlatformConfig, subject_path: str, run_id: str) -> str:
        """A run is read back only by the credentials, org and app that stored it,
        so memory never answers a request the platform would have refused."""
        token = hashlib.sha256(platform_config.auth.token.get_secret_value().encode()).hexdigest()
        return f"{platform_config.host}/orgs/{platform_config.subject.org_id}/{subject_path}/runs/{run_id}#{token}"

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
        size = sum(len(body) for _, body in recent.values())
        # Entries are ordered by their last successful save, oldest first.
        # Purge expired entries even when the requested key is absent.
        while recent:
            saved_at, body = next(iter(recent.values()))
            if len(recent) <= capacity and size <= cls._recent_max_bytes and now - saved_at < ttl:
                break
            recent.popitem(last=False)
            size -= len(body)

    @classmethod
    def _remember(cls, key: str, snapshot: int, body: bytes | None) -> None:
        """Settle a finished save of ``snapshot``; ``body`` is None when the save failed."""
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
                recent[key] = (now, body)
        cls._prune_recent(now, capacity, ttl)

    @classmethod
    def _recall(cls, key: str) -> Trace | None:
        capacity, ttl = cls._recent_limits()
        cls._prune_recent(time.monotonic(), capacity, ttl)
        entry = PlatformTracingProvider._recent.get(key)
        if entry is None:
            return None
        # Parsed afresh each time, so a reader can never change what the next one sees.
        stored = json.loads(entry[1])
        return Trace(stored["trace"], parent_id=stored.get("parent_id"))

    @staticmethod
    def _host_caps(host: str) -> _HostCaps:
        caps = PlatformTracingProvider._caps.get(host)
        if caps is None:
            caps = PlatformTracingProvider._caps[host] = _HostCaps()
        return caps

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
        """Whether saves to this provider's host go out as span deltas right now (each save is then small)."""
        cfg = platform_config or cls.platform_config
        if cfg is None:
            from ....state.config_loader import resolve_platform_config

            cfg = resolve_platform_config(None)
        return cfg is not None and cls._sending_deltas(cls._host_caps(cfg.host))

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

        1. run_context.platform_config  — set by auto-detection or a previous call
        2. cls.platform_config          — set explicitly via configured()
        3. env vars / config file       — resolved lazily, written back to run_context

        The resolved config is written back to run_context so that _request(),
        which also reads run_context.platform_config directly, finds it.
        """
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
    async def get(cls, run_context: RunContext) -> Trace | None:
        """See base class."""
        from ....platform.utils import _request

        platform_config = cls._resolve_config(run_context)
        subject = platform_config.subject
        if subject is None:
            raise RuntimeError("PlatformTracingProvider.get requires a subject (org_id + app_id) in platform_config.")

        if subject.app_id:
            subject_path = f"apps/{subject.app_id}"
        else:
            raise ValueError("Cannot use platform tracing provider without an app or project subject")
        key = cls._recent_key(platform_config, subject_path, run_context.parent_id)
        saves = PlatformTracingProvider._saving.get(key)
        flight = PlatformTracingProvider._flights.get(key)
        waits = list(saves.pending.values()) if saves is not None else []
        if flight is not None and flight.loop is asyncio.get_running_loop():
            waits += flight.waits()  # a round queued behind one on the wire is a save already asked for
        if waits:
            # The parent's save is still in flight (a voice turn's shielded save racing the next turn): the
            # platform could answer 404, or an older snapshot. Only saves already asked for are awaited, so a
            # parent that keeps saving (background children) does not hold every read for the full bound.
            await asyncio.wait(waits, timeout=cls._save_wait_s)
        recalled = cls._recall(key)
        if recalled is not None:
            return recalled
        caps = cls._host_caps(platform_config.host)
        res = await _request(
            method="GET",
            path=f"orgs/{subject.org_id}/{subject_path}/runs/{run_context.parent_id}",
            # The root and the agent spans: what resuming reads, without the LLM and tool spans' bytes.
            params={"select": "resume"} if caps.select_resume else None,
        )
        res.raise_for_status()
        caps.learn(getattr(res, "headers", None))
        res_json = res.json()
        trace = res_json.get("trace")
        if not trace:
            return None
        return Trace(trace, parent_id=res_json.get("parent_id"))

    @classmethod
    @override
    async def _store(cls, run_context: RunContext) -> None:
        """See base class. Single-flight per run: a save that arrives while one is on the wire joins the
        next round, which goes out once, from the latest state, when that one is back (or has stalled
        for ``_stall_s``). The caller waits for the round carrying its state; cancelling it does not
        cancel the save."""
        _, _, key = cls._target(run_context)
        loop = asyncio.get_running_loop()
        flights = PlatformTracingProvider._flights
        flight = flights.get(key)
        if flight is None or flight.loop is not loop:
            flight = flights[key] = _Flight(loop)
        flight.context = run_context
        if flight.next is None:
            flight.next = loop.create_future()
            flight.next.add_done_callback(_retrieve)
        joined = flight.next
        if flight.driver is None:
            flight.driver = loop.create_task(cls._drive(key, flight))
        await asyncio.shield(joined)

    @classmethod
    async def _drive(cls, key: str, flight: _Flight) -> None:
        try:
            while flight.next is not None:
                joined, flight.next = flight.next, None
                flight.sent.append(joined)
                save = asyncio.get_running_loop().create_task(cls._save(flight.context))
                save.add_done_callback(functools.partial(_settle, flight, joined))
                # A PATCH can hang (read timeouts are unbounded by default): past this the next round
                # goes anyway, so a run's end is never held behind it. The platform keeps the newest
                # snapshot whichever lands last.
                await asyncio.wait({save}, timeout=cls._stall_s)
        finally:
            flight.driver = None
            if flight.next is not None and not flight.next.done():
                flight.next.cancel()  # the driver itself was cancelled (loop shutting down)
            if PlatformTracingProvider._flights.get(key) is flight:
                del PlatformTracingProvider._flights[key]

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
        platform_config = cls._resolve_config(run_context)
        subject = platform_config.subject
        if subject is None:
            raise RuntimeError(
                "PlatformTracingProvider._store requires a subject (org_id + app_id) in platform_config."
            )
        if subject.app_id:
            subject_path = f"apps/{subject.app_id}"
        else:
            raise ValueError("Cannot use platform tracing provider without an app or project subject")
        return platform_config, subject_path, cls._recent_key(platform_config, subject_path, run_context.id)

    @classmethod
    async def _save(cls, run_context: RunContext, *, plain: bool = False) -> None:
        """One PATCH. ``plain``: whole and uncompressed whatever the host advertised."""
        from ....errors import PlatformError
        from ....platform.utils import _request

        platform_config, subject_path, key = cls._target(run_context)
        subject = platform_config.subject
        caps = cls._host_caps(platform_config.host)
        delta = not plain and cls._sending_deltas(caps)
        # Taken with no await before model_dump(), so numbers follow snapshot order.
        snapshot = _next_snapshot()
        # Serialized before the await: model_dump() shares the agent's live memory list, which keeps growing
        # while the PATCH is in flight. Same encoding httpx applies to json=.
        if delta:
            body, cache_body, sent = cls._delta_body(run_context, subject.rev, key, snapshot)
        else:
            payload = {"parent_id": run_context.parent_id, "trace": run_context._trace.model_dump()}
            if subject.rev is not None:
                payload["rev"] = subject.rev
            body = cache_body = _encode(payload)
            sent = {}
        saves = PlatformTracingProvider._saving.setdefault(key, _Saves())
        saves.pending[snapshot] = asyncio.get_running_loop().create_future()
        headers = {
            "Prefer": "return=minimal",  # RFC 7240
            "Content-Type": DELTA_MEDIA_TYPE if delta else "application/json",
            # Lets the platform drop a save that lands after a newer one (it acks it): saves overlap
            # across processes, retries and stalls.
            "X-Timbal-Snapshot": str(snapshot),
        }
        compressed = not plain and cls._compressing(caps)
        wire = body
        if compressed:
            wire = gzip.compress(body, compresslevel=5)
            headers["Content-Encoding"] = "gzip"
        try:
            res = await _request(
                method="PATCH",
                path=f"orgs/{subject.org_id}/{subject_path}/runs/{run_context.id}",
                headers=headers,
                content=wire,
                timeout=cls._patch_timeout(),
            )
            res.raise_for_status()
        except PlatformError as e:
            cls._remember(key, snapshot, None)
            if not (delta or compressed) or getattr(e, "status_code", None) not in (400, 415, 422):
                raise
            # Today's server answers a gzip body with 400 (its JSON extractor cannot parse it), a server
            # with the layer with 415. The same save whole and uncompressed tells a rejected format from
            # a bad save: only when that one lands is the format off for this host, starting now.
            await cls._save(run_context, plain=True)
            caps.delta_rejected = caps.delta_rejected or delta
            caps.gzip_rejected = caps.gzip_rejected or compressed
            logger.warning(
                "platform_trace_format_rejected",
                host=platform_config.host,
                status=e.status_code,
                deltas=delta,
                gzip=compressed,
            )
            return
        except BaseException:
            # The write may have landed anyway (timeout after commit): never keep serving the older snapshot.
            cls._remember(key, snapshot, None)
            raise
        caps.learn(getattr(res, "headers", None))
        if sent:
            cls._delta_run(key).acked.update(sent)
        elif not delta and key in PlatformTracingProvider._delta_runs:
            # A whole trace replaces the platform's span rows: what they acknowledged is gone.
            PlatformTracingProvider._delta_runs[key].acked.clear()
        cls._remember(key, snapshot, cache_body)

    @classmethod
    def _delta_body(
        cls, run_context: RunContext, rev: str | None, key: str, snapshot: int
    ) -> tuple[bytes, bytes | None, dict]:
        """A delta save: the spans the platform has not acknowledged, and the whole trace once the root ended.

        While the run goes, finished spans only (they no longer change) and a root header; the root's
        end sends the full root and every span whose content changed since it was acknowledged (a
        background child's late output). Returns the body, the full-trace body to cache (only once
        the root ended: that is what the next turn reads back), and the fingerprints sent.
        """
        run = cls._delta_run(key)
        spans = run_context._trace.as_records()
        root = next((span for span in spans if span.parent_call_id is None), None)
        ended = root is not None and root.t1 is not None
        out: list[dict] = []
        sent: dict[str, str] = {}
        full: list[dict] = []
        for span in spans:
            run.seq.setdefault(span.call_id, len(run.seq))
            if span is root:
                continue
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
        if root is None:
            root_part = None
        elif ended:
            root_part = root.model_dump()
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
        return _encode(payload), cache_body, sent


def _encode(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
