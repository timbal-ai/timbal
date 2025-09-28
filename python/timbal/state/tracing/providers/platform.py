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
        subject = run_context.platform_config.subject
        res = await _request(
            method="GET",
            path=f"orgs/{subject.org_id}/apps/{subject.app_id}/runs/{run_context.parent_id}",
        )
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
        subject = run_context.platform_config.subject
        payload = {
            "version_id": subject.version_id,
            "parent_id": run_context.parent_id,
            "trace": run_context._trace.model_dump()
        }
        res = await _request(
            method="PATCH",
            path=f"orgs/{subject.org_id}/apps/{subject.app_id}/runs/{run_context.id}",
            json=payload,
        )
        res_json = res.json()
        # We'll need to update the run_context id with the one returned by the platform,
        # so we can link run ids with parent ids in subsequent runs
        # ? We'll have the logs with the auto-generated ids, but they will be persisted with different ones
        run_context.id = res_json.get("id", run_context.id)
