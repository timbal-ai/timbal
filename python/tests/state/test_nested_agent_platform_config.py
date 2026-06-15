"""Regression guard for gotcha #1: a standalone Agent instantiated and called
inside a workflow step body forks a fresh RunContext (it was never ``nest()``ed,
so its ``_path`` has no "."). Previously that fork carried only ``parent_id`` +
``tracing_provider`` and re-resolved platform config from env/``~/.timbal``
only — dropping an explicitly-injected ``platform_config`` and breaking platform
API calls (trace save, integration creds) with "No platform config available".

The fix makes the forked child inherit the parent's ``platform_config``. These
tests assert that inheritance (no network needed). The autouse
``reset_platform_config_cache`` fixture forces ``resolve_platform_config()`` to
return None, so without inheritance the child would resolve to None.
"""

from timbal import Agent, Workflow
from timbal.core import Tool
from timbal.core.test_model import TestModel
from timbal.state import get_run_context, set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider


def _platform_config() -> PlatformConfig:
    return PlatformConfig(
        host="api.test.timbal.ai",
        auth=PlatformAuth(type=PlatformAuthType.BEARER, token="tok"),
        subject=PlatformSubject(org_id="1", app_id="42"),
    )


class TestNestedStandaloneAgentContext:
    async def test_nested_standalone_agent_inherits_platform_config(self):
        """A standalone Agent created inside a step body still forks a fresh
        context (different run_id), but now INHERITS the parent's platform_config."""
        observed: dict = {}

        def capture() -> None:
            ctx = get_run_context()
            observed["inner_platform_config"] = ctx.platform_config
            observed["inner_run_id"] = ctx.id

        async def step_handler() -> str:
            parent_ctx = get_run_context()
            observed["handler_platform_config"] = parent_ctx.platform_config
            observed["handler_run_id"] = parent_ctx.id

            # Pin the inner agent to in-memory tracing so the now-inherited
            # platform_config doesn't trigger a real platform trace write.
            inner = Agent(
                name="inner",
                model=TestModel(responses=["done"]),
                pre_hook=capture,
                tracing_provider=InMemoryTracingProvider,
            )
            result = await inner(prompt="hi").collect()
            return result.output.collect_text()

        workflow = Workflow(name="wf").step(Tool(name="step", handler=step_handler))

        set_run_context(RunContext(
            platform_config=_platform_config(),
            tracing_provider=InMemoryTracingProvider,
        ))
        result = await workflow().collect()
        assert result.status.code == "success", result.error

        # The step handler runs in the parent context → platform_config present.
        assert observed["handler_platform_config"] is not None
        assert observed["handler_platform_config"].subject.app_id == "42"

        # The nested standalone Agent still forks a distinct child run...
        assert observed["inner_run_id"] != observed["handler_run_id"]
        # ...but now inherits the parent's platform_config.
        assert observed["inner_platform_config"] is not None
        assert observed["inner_platform_config"].subject.app_id == "42"

    async def test_nested_agent_platform_call_succeeds(self):
        """With config inherited, the previously-failing platform API call inside
        the nested agent now resolves (no "No platform config available")."""
        from timbal.platform.utils import _resolve_url_and_headers

        path = "v1/orgs/1/apps/42/traces"
        observed: dict = {}

        def capture() -> None:
            try:
                url, _ = _resolve_url_and_headers(None, path, {})
                observed["inner_url"] = url
                observed["inner_error"] = None
            except ValueError as e:
                observed["inner_error"] = str(e)

        async def step_handler() -> str:
            # Same call resolves fine in the parent context.
            url, _ = _resolve_url_and_headers(None, path, {})
            observed["parent_url"] = url
            inner = Agent(
                name="inner",
                model=TestModel(responses=["done"]),
                pre_hook=capture,
                tracing_provider=InMemoryTracingProvider,
            )
            result = await inner(prompt="hi").collect()
            return result.output.collect_text()

        workflow = Workflow(name="wf").step(Tool(name="step", handler=step_handler))
        set_run_context(RunContext(
            platform_config=_platform_config(),
            tracing_provider=InMemoryTracingProvider,
        ))
        result = await workflow().collect()
        assert result.status.code == "success", result.error

        assert observed["parent_url"].startswith("https://api.test.timbal.ai/")
        assert observed["inner_error"] is None
        assert observed["inner_url"].startswith("https://api.test.timbal.ai/")

    async def test_agent_step_keeps_platform_config(self):
        """Contrast: a first-class Agent *step* is nest()ed (path has a "."),
        so it reuses the parent context and keeps platform_config."""
        observed: dict = {}

        def capture() -> None:
            ctx = get_run_context()
            observed["step_platform_config"] = ctx.platform_config
            observed["step_run_id"] = ctx.id

        inner = Agent(name="inner", model=TestModel(responses=["done"]), pre_hook=capture)
        workflow = Workflow(name="wf").step(inner)

        set_run_context(RunContext(
            platform_config=_platform_config(),
            tracing_provider=InMemoryTracingProvider,
        ))
        parent_run_id = get_run_context().id
        result = await workflow(prompt="hi").collect()
        assert result.status.code == "success", result.error

        # Same context, platform_config intact.
        assert observed["step_run_id"] == parent_run_id
        assert observed["step_platform_config"] is not None
        assert observed["step_platform_config"].subject.app_id == "42"
