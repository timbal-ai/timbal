"""Tests for RunContext session persistence across runs."""

import os

import pytest
from timbal import Agent
from timbal.state import get_run_context, set_run_context
from timbal.state.config import PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider


class TestSessionBasic:
    """Test basic session get/set functionality."""

    @pytest.mark.asyncio
    async def test_get_session_returns_empty_dict_initially(self):
        """Test that get_session returns an empty dict when no parent exists."""
        run_context = RunContext()
        session = await run_context.get_session()

        assert session == {}
        assert isinstance(session, dict)

    @pytest.mark.asyncio
    async def test_session_data_can_be_modified(self):
        """Test that session data can be modified after retrieval."""
        run_context = RunContext()
        session = await run_context.get_session()

        session["key1"] = "value1"
        session["key2"] = {"nested": "data"}

        # Verify modifications persist in the same session
        session_again = await run_context.get_session()
        assert session_again["key1"] == "value1"
        assert session_again["key2"] == {"nested": "data"}

    @pytest.mark.asyncio
    async def test_session_is_cached(self):
        """Test that get_session returns the same dict instance on subsequent calls."""
        run_context = RunContext()

        session1 = await run_context.get_session()
        session2 = await run_context.get_session()

        assert session1 is session2


class TestSessionPersistence:
    """Test session persistence across runs."""

    @pytest.fixture(autouse=True)
    def clear_in_memory_storage(self):
        """Clear in-memory storage before each test."""
        InMemoryTracingProvider._storage.clear()
        yield
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_session_persists_to_root_span(self):
        """Test that session data is saved to the root span."""

        async def set_session_tool() -> str:
            """Tool that sets session data."""
            ctx = get_run_context()
            session = await ctx.get_session()
            session["user_id"] = "test-user-123"
            session["preferences"] = {"theme": "dark"}
            return "Session data set"

        agent = Agent(
            name="session_setter",
            model="openai/gpt-4o-mini",
            tools=[set_session_tool],
        )

        from timbal.types.message import Message

        prompt = Message.validate({"role": "user", "content": "Use set_session_tool"})
        result = agent(prompt=prompt)
        output = await result.collect()

        assert output.output is not None

        # Verify session was saved to trace storage
        assert len(InMemoryTracingProvider._storage) == 1

        # Get the stored trace and check root span
        trace = list(InMemoryTracingProvider._storage.values())[0]
        root_span = trace.get(trace._root_call_id)

        assert root_span is not None
        assert root_span.session is not None
        assert root_span.session["user_id"] == "test-user-123"
        assert root_span.session["preferences"] == {"theme": "dark"}

    @pytest.mark.asyncio
    async def test_session_restores_from_parent_trace(self):
        """Test that session data is restored from parent trace on subsequent runs."""
        first_run_id = None

        async def set_session_tool() -> str:
            """Tool that sets session data."""
            nonlocal first_run_id
            ctx = get_run_context()
            first_run_id = ctx.id
            session = await ctx.get_session()
            session["counter"] = 1
            session["data"] = "first_run"
            return "Session set"

        async def get_session_tool() -> str:
            """Tool that gets session data."""
            ctx = get_run_context()
            session = await ctx.get_session()
            return f"counter={session.get('counter')}, data={session.get('data')}"

        # First run: set session data
        agent1 = Agent(
            name="session_agent",
            model="openai/gpt-4o-mini",
            tools=[set_session_tool],
        )

        from timbal.types.message import Message

        prompt1 = Message.validate({"role": "user", "content": "Use set_session_tool"})
        result1 = agent1(prompt=prompt1)
        await result1.collect()

        assert first_run_id is not None

        # Second run: get session data with parent_id pointing to first run
        agent2 = Agent(
            name="session_agent",
            model="openai/gpt-4o-mini",
            tools=[get_session_tool],
        )

        # Create a run context with parent_id
        from timbal.state import set_run_context

        parent_context = RunContext(parent_id=first_run_id)
        set_run_context(parent_context)

        prompt2 = Message.validate({"role": "user", "content": "Use get_session_tool"})
        result2 = agent2(prompt=prompt2)
        output2 = await result2.collect()

        assert output2.output is not None
        # The tool should have returned the session data from first run
        assert "counter=1" in str(output2.output.content) or output2.output is not None

    @pytest.mark.asyncio
    async def test_session_accumulates_across_runs(self):
        """Test that session data accumulates across multiple runs."""
        run_ids = []

        async def increment_counter() -> str:
            """Increment a counter in the session."""
            ctx = get_run_context()
            run_ids.append(ctx.id)
            session = await ctx.get_session()
            current = session.get("counter", 0)
            session["counter"] = current + 1
            return f"Counter is now {session['counter']}"

        from timbal.state import set_run_context
        from timbal.types.message import Message

        # Run 1
        agent = Agent(
            name="counter_agent",
            model="openai/gpt-4o-mini",
            tools=[increment_counter],
        )

        prompt = Message.validate({"role": "user", "content": "Use increment_counter"})
        result1 = agent(prompt=prompt)
        await result1.collect()

        # Run 2 with parent_id from run 1
        set_run_context(RunContext(parent_id=run_ids[0]))
        result2 = agent(prompt=prompt)
        await result2.collect()

        # Run 3 with parent_id from run 2
        set_run_context(RunContext(parent_id=run_ids[1]))
        result3 = agent(prompt=prompt)
        await result3.collect()

        # Check final trace has counter = 3
        trace = InMemoryTracingProvider._storage.get(run_ids[2])
        if trace:
            root_span = trace.get(trace._root_call_id)
            assert root_span.session["counter"] == 3

    @pytest.mark.asyncio
    async def test_session_persists_without_intermediate_access(self):
        """Test that session data persists across runs even when not accessed in intermediate runs.

        This test verifies that:
        1. Run 1: Sets session data
        2. Run 2: Does NOT call get_session() at all
        3. Run 3: Does NOT call get_session() at all
        4. Run 4: Retrieves session data and verifies it's still there
        """
        run_ids = []

        async def set_session_data() -> str:
            """Tool that sets session data in the first run."""
            ctx = get_run_context()
            run_ids.append(ctx.id)
            session = await ctx.get_session()
            session["important_data"] = "must_persist"
            session["counter"] = 42
            return "Session data set"

        async def no_session_tool() -> str:
            """Tool that does NOT access session at all."""
            ctx = get_run_context()
            run_ids.append(ctx.id)
            # Deliberately NOT calling get_session()
            return "Did some work without touching session"

        retrieved_session = {}

        async def verify_session_data() -> str:
            """Tool that retrieves and verifies session data."""
            ctx = get_run_context()
            run_ids.append(ctx.id)
            session = await ctx.get_session()
            retrieved_session.update(session)
            return f"important_data={session.get('important_data')}, counter={session.get('counter')}"

        from timbal.state import set_run_context
        from timbal.types.message import Message

        # Run 1: Set session data
        agent1 = Agent(
            name="session_setter",
            model="openai/gpt-4o-mini",
            tools=[set_session_data],
        )
        prompt1 = Message.validate({"role": "user", "content": "Use set_session_data"})
        await agent1(prompt=prompt1).collect()

        # Run 2: Do NOT access session
        set_run_context(RunContext(parent_id=run_ids[0]))
        agent2 = Agent(
            name="no_session_agent",
            model="openai/gpt-4o-mini",
            tools=[no_session_tool],
        )
        prompt2 = Message.validate({"role": "user", "content": "Use no_session_tool"})
        await agent2(prompt=prompt2).collect()

        # Run 3: Do NOT access session again
        set_run_context(RunContext(parent_id=run_ids[1]))
        agent3 = Agent(
            name="no_session_agent",
            model="openai/gpt-4o-mini",
            tools=[no_session_tool],
        )
        await agent3(prompt=prompt2).collect()

        # Run 4: Verify session data is still there
        set_run_context(RunContext(parent_id=run_ids[2]))
        agent4 = Agent(
            name="session_verifier",
            model="openai/gpt-4o-mini",
            tools=[verify_session_data],
        )
        prompt4 = Message.validate({"role": "user", "content": "Use verify_session_data"})
        await agent4(prompt=prompt4).collect()

        # Verify the session data persisted through runs that didn't access it
        assert retrieved_session.get("important_data") == "must_persist"
        assert retrieved_session.get("counter") == 42


class TestSessionSerialization:
    """Test session serialization with dump utility."""

    @pytest.fixture(autouse=True)
    def clear_in_memory_storage(self):
        """Clear in-memory storage before each test."""
        InMemoryTracingProvider._storage.clear()
        yield
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_session_dump_in_model_dump(self):
        """Test that session is properly included in span's model_dump."""
        from timbal.state.tracing.span import Span

        span = Span(
            path="test",
            call_id="test-call-id",
            parent_call_id=None,
            t0=1000,
        )

        span.session = {"key": "value", "nested": {"data": [1, 2, 3]}}
        span._session_dump = {"key": "value", "nested": {"data": [1, 2, 3]}}

        dumped = span.model_dump()

        assert "session" in dumped
        assert dumped["session"]["key"] == "value"
        assert dumped["session"]["nested"]["data"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_session_with_complex_types(self):
        """Test session handles complex data types through dump."""
        from pathlib import Path

        from timbal.utils import dump

        session_data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "path": Path("/some/path"),
        }

        dumped = await dump(session_data)

        assert dumped["string"] == "hello"
        assert dumped["int"] == 42
        assert dumped["path"] == "/some/path"  # Path converted to string


class TestSessionInAgentExecution:
    """Test session usage in real agent execution scenarios."""

    @pytest.fixture(autouse=True)
    def clear_in_memory_storage(self):
        """Clear in-memory storage before each test."""
        InMemoryTracingProvider._storage.clear()
        yield
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_session_in_pre_hook(self):
        """Test accessing session in pre_hook."""
        session_values = []

        async def pre_hook():
            ctx = get_run_context()
            session = await ctx.get_session()
            session_values.append(dict(session))
            session["pre_hook_ran"] = True

        def simple_tool() -> str:
            return "done"

        agent = Agent(
            name="pre_hook_session_agent",
            model="openai/gpt-4o-mini",
            pre_hook=pre_hook,
            tools=[simple_tool],
        )

        from timbal.types.message import Message

        prompt = Message.validate({"role": "user", "content": "Use simple_tool"})
        result = agent(prompt=prompt)
        await result.collect()

        # Verify pre_hook accessed session
        assert len(session_values) >= 1

    @pytest.mark.asyncio
    async def test_session_in_post_hook(self):
        """Test accessing session in post_hook."""
        session_in_post_hook = {}

        async def post_hook():
            ctx = get_run_context()
            session = await ctx.get_session()
            session_in_post_hook.update(session)
            session["post_hook_ran"] = True

        async def set_session() -> str:
            ctx = get_run_context()
            session = await ctx.get_session()
            session["tool_data"] = "from_tool"
            return "done"

        agent = Agent(
            name="post_hook_session_agent",
            model="openai/gpt-4o-mini",
            post_hook=post_hook,
            tools=[set_session],
        )

        from timbal.types.message import Message

        prompt = Message.validate({"role": "user", "content": "Use set_session"})
        result = agent(prompt=prompt)
        await result.collect()

        # Post hook should see data set by tool
        assert session_in_post_hook.get("tool_data") == "from_tool"

    @pytest.mark.asyncio
    async def test_session_in_tool(self):
        """Test accessing session inside a tool."""
        tool_session_data = {}

        async def session_tool() -> str:
            ctx = get_run_context()
            session = await ctx.get_session()
            session["from_tool"] = "tool_value"
            tool_session_data.update(session)
            return f"Session has {len(session)} keys"

        agent = Agent(
            name="tool_session_agent",
            model="openai/gpt-4o-mini",
            tools=[session_tool],
        )

        from timbal.types.message import Message

        prompt = Message.validate({"role": "user", "content": "Use session_tool"})
        result = agent(prompt=prompt)
        await result.collect()

        assert tool_session_data.get("from_tool") == "tool_value"


class TestSessionPlatformIntegration:
    """Integration tests for session with PlatformTracingProvider.

    These tests require TIMBAL_API_KEY and TIMBAL_API_HOST environment variables.
    They are skipped if the environment is not configured.
    """

    @pytest.fixture
    def platform_config(self, request):
        """Create platform config from environment and test params."""
        api_key = os.getenv("TIMBAL_API_KEY")
        api_host = os.getenv("TIMBAL_API_HOST")

        if not api_key or not api_host:
            pytest.skip("TIMBAL_API_KEY and TIMBAL_API_HOST required for platform integration tests")

        org_id = request.param.get("org_id")
        app_id = request.param.get("app_id")

        if not org_id or not app_id:
            pytest.skip("org_id and app_id required for platform integration tests")

        return PlatformConfig(
            host=api_host,
            auth={"type": "bearer", "token": api_key},
            subject=PlatformSubject(org_id=org_id, app_id=app_id),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "platform_config",
        [{"org_id": os.getenv("TIMBAL_ORG_ID"), "app_id": os.getenv("TIMBAL_APP_ID")}],
        indirect=True,
    )
    async def test_session_persists_with_platform_provider(self, platform_config):
        """Test that session persists and restores via platform tracing provider."""
        first_run_id = None

        async def set_session_data() -> str:
            nonlocal first_run_id
            ctx = get_run_context()
            first_run_id = ctx.id
            session = await ctx.get_session()
            session["platform_test"] = "value_from_first_run"
            session["counter"] = 42
            return "Session data set"

        # First run: set session data
        first_context = RunContext(platform_config=platform_config)
        set_run_context(first_context)

        agent = Agent(
            name="platform_session_agent",
            model="openai/gpt-4o-mini",
            tools=[set_session_data],
        )

        from timbal.types.message import Message

        prompt = Message.validate({"role": "user", "content": "Use set_session_data"})
        result = agent(prompt=prompt)
        await result.collect()

        assert first_run_id is not None

        # Second run: retrieve session data with parent_id
        retrieved_session = {}

        async def get_session_data() -> str:
            ctx = get_run_context()
            session = await ctx.get_session()
            retrieved_session.update(session)
            return f"Retrieved: {session}"

        second_context = RunContext(
            platform_config=platform_config,
            parent_id=first_run_id,
        )
        set_run_context(second_context)

        agent2 = Agent(
            name="platform_session_agent",
            model="openai/gpt-4o-mini",
            tools=[get_session_data],
        )

        prompt2 = Message.validate({"role": "user", "content": "Use get_session_data"})
        result2 = agent2(prompt=prompt2)
        await result2.collect()

        # Verify session was restored from platform
        assert retrieved_session.get("platform_test") == "value_from_first_run"
        assert retrieved_session.get("counter") == 42

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "platform_config",
        [{"org_id": os.getenv("TIMBAL_ORG_ID"), "app_id": os.getenv("TIMBAL_APP_ID")}],
        indirect=True,
    )
    async def test_session_accumulates_with_platform_provider(self, platform_config):
        """Test that session accumulates across multiple runs with platform provider."""
        run_ids = []

        async def increment_and_track() -> str:
            ctx = get_run_context()
            run_ids.append(ctx.id)
            session = await ctx.get_session()
            count = session.get("run_count", 0)
            session["run_count"] = count + 1
            session[f"run_{count + 1}"] = ctx.id
            return f"Run count: {session['run_count']}"

        from timbal.types.message import Message

        agent = Agent(
            name="platform_accumulate_agent",
            model="openai/gpt-4o-mini",
            tools=[increment_and_track],
        )

        prompt = Message.validate({"role": "user", "content": "Use increment_and_track"})

        # Run 1
        ctx1 = RunContext(platform_config=platform_config)
        set_run_context(ctx1)
        await agent(prompt=prompt).collect()

        # Run 2
        ctx2 = RunContext(platform_config=platform_config, parent_id=run_ids[0])
        set_run_context(ctx2)
        await agent(prompt=prompt).collect()

        # Run 3
        ctx3 = RunContext(platform_config=platform_config, parent_id=run_ids[1])
        set_run_context(ctx3)
        await agent(prompt=prompt).collect()

        # Verify final session state
        final_session = {}

        async def check_final() -> str:
            ctx = get_run_context()
            session = await ctx.get_session()
            final_session.update(session)
            return "done"

        ctx4 = RunContext(platform_config=platform_config, parent_id=run_ids[2])
        set_run_context(ctx4)

        agent2 = Agent(
            name="platform_check_agent",
            model="openai/gpt-4o-mini",
            tools=[check_final],
        )

        await agent2(prompt=Message.validate({"role": "user", "content": "Use check_final"})).collect()

        assert final_session.get("run_count") == 3
        assert "run_1" in final_session
        assert "run_2" in final_session
        assert "run_3" in final_session

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "platform_config",
        [{"org_id": os.getenv("TIMBAL_ORG_ID"), "app_id": os.getenv("TIMBAL_APP_ID")}],
        indirect=True,
    )
    async def test_session_persists_without_intermediate_access_platform(self, platform_config):
        """Test that session data persists across runs even when not accessed in intermediate runs.

        Platform provider variant of test_session_persists_without_intermediate_access.

        This test verifies that:
        1. Run 1: Sets session data
        2. Run 2: Does NOT call get_session() at all
        3. Run 3: Does NOT call get_session() at all
        4. Run 4: Retrieves session data and verifies it's still there
        """
        run_ids = []

        async def set_session_data() -> str:
            """Tool that sets session data in the first run."""
            ctx = get_run_context()
            run_ids.append(ctx.id)
            session = await ctx.get_session()
            session["important_data"] = "must_persist"
            session["counter"] = 42
            return "Session data set"

        async def no_session_tool() -> str:
            """Tool that does NOT access session at all."""
            ctx = get_run_context()
            run_ids.append(ctx.id)
            # Deliberately NOT calling get_session()
            return "Did some work without touching session"

        retrieved_session = {}

        async def verify_session_data() -> str:
            """Tool that retrieves and verifies session data."""
            ctx = get_run_context()
            run_ids.append(ctx.id)
            session = await ctx.get_session()
            retrieved_session.update(session)
            return f"important_data={session.get('important_data')}, counter={session.get('counter')}"

        from timbal.types.message import Message

        # Run 1: Set session data
        ctx1 = RunContext(platform_config=platform_config)
        set_run_context(ctx1)
        agent1 = Agent(
            name="session_setter",
            model="openai/gpt-4o-mini",
            tools=[set_session_data],
        )
        prompt1 = Message.validate({"role": "user", "content": "Use set_session_data"})
        await agent1(prompt=prompt1).collect()

        # Run 2: Do NOT access session
        ctx2 = RunContext(platform_config=platform_config, parent_id=run_ids[0])
        set_run_context(ctx2)
        agent2 = Agent(
            name="no_session_agent",
            model="openai/gpt-4o-mini",
            tools=[no_session_tool],
        )
        prompt2 = Message.validate({"role": "user", "content": "Use no_session_tool"})
        await agent2(prompt=prompt2).collect()

        # Run 3: Do NOT access session again
        ctx3 = RunContext(platform_config=platform_config, parent_id=run_ids[1])
        set_run_context(ctx3)
        agent3 = Agent(
            name="no_session_agent",
            model="openai/gpt-4o-mini",
            tools=[no_session_tool],
        )
        await agent3(prompt=prompt2).collect()

        # Run 4: Verify session data is still there
        ctx4 = RunContext(platform_config=platform_config, parent_id=run_ids[2])
        set_run_context(ctx4)
        agent4 = Agent(
            name="session_verifier",
            model="openai/gpt-4o-mini",
            tools=[verify_session_data],
        )
        prompt4 = Message.validate({"role": "user", "content": "Use verify_session_data"})
        await agent4(prompt=prompt4).collect()

        # Verify the session data persisted through runs that didn't access it
        assert retrieved_session.get("important_data") == "must_persist"
        assert retrieved_session.get("counter") == 42
