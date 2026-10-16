from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, ClassVar

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
    """

    platform_config: PlatformConfig | None = None

    _recent: ClassVar[OrderedDict[str, tuple[float, str]]] = OrderedDict()
    _recent_max: ClassVar[int] = 16
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
    def _remember(cls, key: str, trace: list) -> None:
        recent = PlatformTracingProvider._recent
        recent[key] = (time.monotonic(), json.dumps(trace))
        recent.move_to_end(key)
        while len(recent) > cls._recent_max:
            recent.popitem(last=False)

    @classmethod
    def _recall(cls, key: str) -> Trace | None:
        entry = PlatformTracingProvider._recent.get(key)
        if entry is None or time.monotonic() - entry[0] > cls._recent_ttl_s:
            return None
        # Parsed afresh each time, so a reader can never change what the next one sees.
        return Trace(json.loads(entry[1]))

    @classmethod
    def _resolve_config(cls, run_context: "RunContext"):
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
    async def get(cls, run_context: "RunContext") -> Trace | None:
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
    async def _store(cls, run_context: "RunContext") -> None:
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
        res = await _request(
            method="PATCH",
            path=f"orgs/{subject.org_id}/{subject_path}/runs/{run_context.id}",
            headers={"Prefer": "return=minimal"},  # RFC 7240
            json=payload,
        )
        res.raise_for_status()
        cls._remember(cls._recent_key(platform_config, subject_path, run_context.id), payload["trace"])
