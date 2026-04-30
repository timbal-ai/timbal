import asyncio
from datetime import datetime

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.types.message import Message

from ..conftest import assert_has_output_event, assert_no_errors


class TestSystemPromptBasic:
    """Test basic system prompt functionality."""

    def test_agent_without_system_prompt(self):
        agent = Agent(name="no_prompt_agent", model=TestModel())
        assert agent.system_prompt is None
        assert agent._system_prompt_fn is None

    def test_agent_with_static_system_prompt(self):
        system_prompt = "You are a helpful assistant specialized in mathematics."
        agent = Agent(name="static_prompt_agent", model=TestModel(), system_prompt=system_prompt)
        assert agent.system_prompt == system_prompt
        assert agent._system_prompt_fn is None

    @pytest.mark.asyncio
    async def test_static_system_prompt_execution(self):
        system_prompt = "You are a math expert."
        agent = Agent(name="math_expert", model=TestModel(), system_prompt=system_prompt)

        prompt = Message.validate({"role": "user", "content": "What is 2+2?"})
        result = agent(prompt=prompt)
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        assert isinstance(output.output, Message)

    @pytest.mark.asyncio
    async def test_static_prompt_resolves_unchanged(self):
        system_prompt = "Static prompt"
        agent = Agent(name="static_agent", model=TestModel(), system_prompt=system_prompt)
        resolved = await agent._resolve_system_prompt()
        assert resolved == "Static prompt"

    @pytest.mark.asyncio
    async def test_none_system_prompt_resolves_to_none(self):
        agent = Agent(name="no_prompt_agent", model=TestModel())
        resolved = await agent._resolve_system_prompt()
        assert resolved is None


class TestCallableSystemPrompt:
    """Test system_prompt passed as a callable function."""

    def test_sync_callable_system_prompt(self):
        def get_prompt() -> str:
            return "You are a helpful assistant."

        agent = Agent(name="sync_callable_agent", model=TestModel(), system_prompt=get_prompt)

        assert agent.system_prompt is None
        assert agent._system_prompt_fn is get_prompt
        assert agent._system_prompt_fn_is_async is False

    def test_async_callable_system_prompt(self):
        async def get_prompt_async() -> str:
            await asyncio.sleep(0.01)
            return "You are an async assistant."

        agent = Agent(name="async_callable_agent", model=TestModel(), system_prompt=get_prompt_async)

        assert agent.system_prompt is None
        assert agent._system_prompt_fn is get_prompt_async
        assert agent._system_prompt_fn_is_async is True

    @pytest.mark.asyncio
    async def test_sync_callable_resolution(self):
        call_count = 0

        def get_prompt() -> str:
            nonlocal call_count
            call_count += 1
            return f"Call count: {call_count}"

        agent = Agent(name="sync_resolution_agent", model=TestModel(), system_prompt=get_prompt)

        resolved1 = await agent._resolve_system_prompt()
        assert resolved1 == "Call count: 1"

        resolved2 = await agent._resolve_system_prompt()
        assert resolved2 == "Call count: 2"

    @pytest.mark.asyncio
    async def test_async_callable_resolution(self):
        call_count = 0

        async def get_prompt_async() -> str:
            nonlocal call_count
            await asyncio.sleep(0.01)
            call_count += 1
            return f"Async call count: {call_count}"

        agent = Agent(name="async_resolution_agent", model=TestModel(), system_prompt=get_prompt_async)

        resolved1 = await agent._resolve_system_prompt()
        assert resolved1 == "Async call count: 1"

        resolved2 = await agent._resolve_system_prompt()
        assert resolved2 == "Async call count: 2"

    @pytest.mark.asyncio
    async def test_callable_with_dynamic_content(self):
        def get_dynamic_prompt() -> str:
            current_time = datetime.now().strftime("%H:%M:%S")
            return f"Current time is {current_time}. You are a time-aware assistant."

        agent = Agent(name="dynamic_callable_agent", model=TestModel(), system_prompt=get_dynamic_prompt)

        resolved = await agent._resolve_system_prompt()
        assert "Current time is" in resolved
        assert "You are a time-aware assistant" in resolved

    @pytest.mark.asyncio
    async def test_callable_returning_empty_string(self):
        def get_empty_prompt() -> str:
            return ""

        agent = Agent(name="empty_callable_agent", model=TestModel(), system_prompt=get_empty_prompt)

        resolved = await agent._resolve_system_prompt()
        assert resolved == ""

    @pytest.mark.asyncio
    async def test_lambda_as_system_prompt(self):
        agent = Agent(name="lambda_agent", model=TestModel(), system_prompt=lambda: "Lambda system prompt")

        assert agent._system_prompt_fn is not None
        assert agent._system_prompt_fn_is_async is False

        resolved = await agent._resolve_system_prompt()
        assert resolved == "Lambda system prompt"

    @pytest.mark.asyncio
    async def test_callable_vs_string_behavior(self):
        static_prompt = "Static prompt"
        agent_static = Agent(name="static_agent", model=TestModel(), system_prompt=static_prompt)

        def dynamic_prompt() -> str:
            return "Dynamic prompt"

        agent_dynamic = Agent(name="dynamic_agent", model=TestModel(), system_prompt=dynamic_prompt)

        assert agent_static.system_prompt == static_prompt
        assert agent_static._system_prompt_fn is None

        assert agent_dynamic.system_prompt is None
        assert agent_dynamic._system_prompt_fn is dynamic_prompt

        resolved_static = await agent_static._resolve_system_prompt()
        resolved_dynamic = await agent_dynamic._resolve_system_prompt()

        assert resolved_static == "Static prompt"
        assert resolved_dynamic == "Dynamic prompt"
