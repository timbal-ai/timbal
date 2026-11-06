from __future__ import annotations

import asyncio
import functools
import hashlib
import itertools
import json
import math
import os
import time
from collections import OrderedDict
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

    # run key -> (monotonic time of the last successful save, the exact PATCH body sent)
    _recent: ClassVar[OrderedDict[str, tuple[float, bytes]]] = OrderedDict()
    # Numbers snapshots as they are taken: saves of one run overlap (parallel tools, background
    # children) and their PATCHes can finish in any order.
    _snapshots: ClassVar[itertools.count] = itertools.count()
    # run key -> the run's saves in flight. Kept apart from _recent so eviction cannot lose it, and dropped
    # with the run's last save in flight: any later save takes a higher number.
    _saving: ClassVar[dict[str, _Saves]] = {}
    # How long a read waits for saves of that run already in flight before asking the platform.
    _save_wait_s: ClassVar[float] = 2.0
    # One entry per live conversation: concurrent calls sharing a process must not evict each other between turns.
    _recent_max: ClassVar[int] = 256
    _recent_max_bytes: ClassVar[int] = 64 * 1024 * 1024
    # Another process may write the same run later (an approval resumed
    # elsewhere); past this, the platform's copy is the one to trust.
    _recent_ttl_s: ClassVar[float] = 300.0
    # (parent_id:approval_id) -> run_id that claimed it in this process.
    _claims: ClassVar[OrderedDict[str, str]] = OrderedDict()
    _claims_max: ClassVar[int] = 4096

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
        if saves is not None and saves.pending:
            # The parent's save is still in flight (a voice turn's shielded save racing the next turn): the
            # platform could answer 404, or an older snapshot. Only saves already started are awaited, so a
            # parent that keeps saving (background children) does not hold every read for the full bound.
            await asyncio.wait(list(saves.pending.values()), timeout=cls._save_wait_s)
        recalled = cls._recall(key)
        if recalled is not None:
            return recalled
        res = await _request(
            method="GET",
            path=f"orgs/{subject.org_id}/{subject_path}/runs/{run_context.parent_id}",
        )
        res.raise_for_status()
        res_json = res.json()
        trace = res_json.get("trace")
        if not trace:
            return None
        return Trace(trace, parent_id=res_json.get("parent_id"))

    @classmethod
    @override
    async def claim_approval(cls, parent_id: str | None, approval_id: str, run_id: str) -> bool:
        """Claim ``(parent_id, approval_id)`` within this worker process.

        Cross-host ``voice_call`` resumes are serialized on the platform by the
        ``SipCallWaits`` outbox (per-run Postgres advisory lock, pinned
        ``resume_run_id``, and a single-child check on the waiting run). This
        in-process map catches duplicate resumes that hit the same worker.
        """
        if parent_id is None:
            return True
        key = f"{parent_id}:{approval_id}"
        existing = cls._claims.get(key)
        if existing is not None:
            return existing == run_id
        cls._claims[key] = run_id
        while len(cls._claims) > cls._claims_max:
            cls._claims.popitem(last=False)
        return True

    @classmethod
    @override
    async def _store(cls, run_context: RunContext) -> None:
        """See base class."""
        from ....platform.utils import _request

        platform_config = cls._resolve_config(run_context)
        subject = platform_config.subject
        if subject is None:
            raise RuntimeError(
                "PlatformTracingProvider._store requires a subject (org_id + app_id) in platform_config."
            )

        payload = {"parent_id": run_context.parent_id, "trace": run_context._trace.model_dump()}
        if subject.app_id:
            subject_path = f"apps/{subject.app_id}"
        else:
            raise ValueError("Cannot use platform tracing provider without an app or project subject")
        if subject.rev is not None:
            payload["rev"] = subject.rev
        # Serialized before the await: model_dump() shares the agent's live memory list, which keeps growing
        # while the PATCH is in flight. Same encoding httpx applies to json=.
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        # Taken with no await since model_dump(), so numbers follow snapshot order.
        snapshot = next(PlatformTracingProvider._snapshots)
        key = cls._recent_key(platform_config, subject_path, run_context.id)
        saves = PlatformTracingProvider._saving.setdefault(key, _Saves())
        saves.pending[snapshot] = asyncio.get_running_loop().create_future()
        try:
            res = await _request(
                method="PATCH",
                path=f"orgs/{subject.org_id}/{subject_path}/runs/{run_context.id}",
                headers={"Prefer": "return=minimal", "Content-Type": "application/json"},  # RFC 7240
                content=body,
            )
            res.raise_for_status()
        except BaseException:
            # The write may have landed anyway (timeout after commit): never keep serving the older snapshot.
            cls._remember(key, snapshot, None)
            raise
        cls._remember(key, snapshot, body)
