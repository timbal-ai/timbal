from typing import TYPE_CHECKING, override

from .. import Tracing
from .base import TracingProvider

if TYPE_CHECKING:
    from ...context import RunContext


class PlatformTracingProvider(TracingProvider):
    """Platform tracing provider using platform-wide storage."""
    
    @classmethod
    @override
    async def get(cls, run_context: "RunContext") -> Tracing | None:
        """See base class."""
        from ....utils import _platform_api_call
        subject = run_context.platform_config.subject
        res = await _platform_api_call(
            method="GET",
            path=f"orgs/{subject.org_id}/apps/{subject.app_id}/tracing",
            params={"run_id": run_context.parent_id},
        )
        res_json = res.json()
        return Tracing(res_json["tracing"])
    
    @classmethod
    @override
    async def put(cls, run_context: "RunContext") -> None:
        """See base class."""
        from ....utils import _platform_api_call
        subject = run_context.platform_config.subject
        payload = {
            "version_id": subject.version_id,
            "id": run_context.id,
            "parent_id": run_context.parent_id,
            "tracing": run_context.tracing.model_dump()
        }
        res = await _platform_api_call(
            method="POST",
            path=f"orgs/{subject.org_id}/apps/{subject.app_id}/tracing",
            json=payload,
        )
        res_json = res.json()
        # We'll need to update the run_context id with the one returned by the platform,
        # so we can link run ids with parent ids in subsequent runs
        if "run_id" in res_json:
            run_context.id = res_json["run_id"]
