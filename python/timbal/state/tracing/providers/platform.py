from typing import TYPE_CHECKING

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..trace import Trace
from .base import TracingProvider

if TYPE_CHECKING:
    from ...context import RunContext


class PlatformTracingProvider(TracingProvider):
    """Platform tracing provider using platform-wide storage."""

    @classmethod
    @override
    async def get(cls, run_context: "RunContext") -> Trace | None:
        """See base class."""
        from ....platform.utils import _request

        assert run_context.platform_config is not None, "PlatformTracingProvider.get called without platform config"
        subject = run_context.platform_config.subject
        assert subject is not None, "PlatformTracingProvider.get called without a subject"

        if subject.app_id:
            subject_path = f"apps/{subject.app_id}"
        elif subject.project_id:
            subject_path = f"projects/{subject.project_id}"
        else:
            raise ValueError("Cannot use platform tracing provider without an app or project subject")
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
    async def put(cls, run_context: "RunContext") -> None:
        """See base class."""
        from ....platform.utils import _request

        assert run_context.platform_config is not None, "PlatformTracingProvider.put called without platform config"
        subject = run_context.platform_config.subject
        assert subject is not None, "PlatformTracingProvider.put called without a subject"

        payload = {"parent_id": run_context.parent_id, "trace": run_context._trace.model_dump()}
        if subject.app_id:
            subject_path = f"apps/{subject.app_id}"
            payload["version_id"] = subject.version_id
        elif subject.project_id:
            subject_path = f"projects/{subject.project_id}"
        else:
            raise ValueError("Cannot use platform tracing provider without an app or project subject")
        res = await _request(
            method="PATCH",
            path=f"orgs/{subject.org_id}/{subject_path}/runs/{run_context.id}",
            headers={"Prefer": "return=minimal"},  # RFC 7240
            json=payload,
        )
        res.raise_for_status()
