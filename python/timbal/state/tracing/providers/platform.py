from __future__ import annotations

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
    """

    platform_config: PlatformConfig | None = None

    # run key -> (monotonic time the save finished, snapshot number, the exact PATCH body sent or None)
    _recent: ClassVar[OrderedDict[str, tuple[float, int, bytes | None]]] = OrderedDict()
    # Numbers snapshots as they are taken: saves of one run overlap (parallel tools, background
    # children) and their PATCHes can finish in any order.
    _snapshots: ClassVar[itertools.count] = itertools.count()
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
        size = sum(len(body) for _, _, body in recent.values() if body)
        # Entries are ordered by when their save finished, oldest first.
        # Purge expired entries even when the requested key is absent.
        while recent:
            saved_at, _, body = next(iter(recent.values()))
            if len(recent) <= capacity and size <= cls._recent_max_bytes and now - saved_at < ttl:
                break
            recent.popitem(last=False)
            size -= len(body) if body else 0

    @classmethod
    def _remember(cls, key: str, snapshot: int, body: bytes | None) -> None:
        """Settle a finished save of ``snapshot``; ``body`` is None when the save failed."""
        capacity, ttl = cls._recent_limits()
        now = time.monotonic()
        recent = PlatformTracingProvider._recent
        settled = recent.get(key)
        if settled is None or settled[1] < snapshot:
            recent.pop(key, None)
            # A failed or oversized save still leaves its number: the platform may hold this snapshot now,
            # so neither the older cached body nor an older save still in flight may answer reads.
            kept = body if body is not None and len(body) <= cls._recent_max_bytes else None
            recent[key] = (now, snapshot, kept)
        cls._prune_recent(now, capacity, ttl)

    @classmethod
    def _recall(cls, key: str) -> Trace | None:
        capacity, ttl = cls._recent_limits()
        cls._prune_recent(time.monotonic(), capacity, ttl)
        entry = PlatformTracingProvider._recent.get(key)
        if entry is None or entry[2] is None:
            return None
        # Parsed afresh each time, so a reader can never change what the next one sees.
        return Trace(json.loads(entry[2])["trace"])

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
        recalled = cls._recall(cls._recent_key(platform_config, subject_path, run_context.parent_id))
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
        return Trace(trace)

    @classmethod
    @override
    async def _store(cls, run_context: RunContext) -> None:
        """See base class."""
        from ....platform.utils import _request

        platform_config = cls._resolve_config(run_context)
        subject = platform_config.subject
        if subject is None:
            raise RuntimeError("PlatformTracingProvider._store requires a subject (org_id + app_id) in platform_config.")

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
