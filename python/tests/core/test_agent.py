import pytest
from timbal import Agent, Tool
from timbal.core.agent import AgentParams
from timbal.types.events import OutputEvent
from timbal.types.message import Message

from .conftest import (
    Timer,
    assert_has_output_event,
    assert_no_errors,
    skip_if_agent_error,
)


class TestAgentCreation:
    """Test Agent instantiation and configuration."""
    
    def test_minimal_agent(self):
        """Test creating agent with minimal configuration."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini"
        )
        
        assert agent.name == "test_agent"
        assert agent.model == "openai/gpt-4o-mini"
        assert agent.system_prompt is None
        assert len(agent.tools) == 0
        assert agent.max_iter == 10
    
    def test_agent_with_system_prompt(self):
        """Test agent with system system_prompt."""
        system_prompt = "You are a helpful assistant specialized in math."
        agent = Agent(
            name="math_agent",
            model="openai/gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        assert agent.system_prompt == system_prompt
    
    def test_agent_with_callable_tools(self):
        """Test agent with function tools."""
        def add(a: int, b: int) -> int:
            return a + b
        
        def multiply(a: int, b: int) -> int:
            return a * b
        
        agent = Agent(
            name="math_agent",
            model="openai/gpt-4o-mini",
            tools=[add, multiply]
        )
        
        assert len(agent.tools) == 2
        assert all(isinstance(tool, Tool) for tool in agent.tools)
        assert agent._tools_by_name["add"] is not None
        assert agent._tools_by_name["multiply"] is not None
    
    def test_agent_with_dict_tools(self):
        """Test agent with dictionary tool configurations."""
        tool_config = {
            "name": "identity",
            "handler": lambda x: x,
            "description": "Returns input unchanged"
        }
        
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            tools=[tool_config]
        )
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "identity"
        assert agent.tools[0].description == "Returns input unchanged"
    
    def test_agent_with_nested_agents(self):
        """Test agent with other agents as tools."""
        # Create a specialist agent
        def get_weather(city: str) -> str:
            return f"Weather in {city}: sunny, 22°C"
        
        weather_agent = Agent(
            name="weather_agent",
            model="openai/gpt-4o-mini",
            tools=[get_weather],
            description="Provides weather information"
        )
        
        # Create main agent with weather agent as tool
        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[weather_agent]
        )
        
        assert len(main_agent.tools) == 1
        assert isinstance(main_agent.tools[0], Agent)
        assert main_agent._tools_by_name["weather_agent"] is weather_agent
    
    def test_duplicate_tool_names(self):
        """Test that duplicate tool names are rejected."""
        def tool1(x: int) -> int:
            return x
        
        def tool2(x: int) -> int:
            return x * 2
        
        # Rename tool2 to have same name as tool1
        tool2.__name__ = "tool1"
        
        with pytest.raises(ValueError, match="Tool tool1 already exists"):
            Agent(
                name="test_agent",
                model="openai/gpt-4o-mini",
                tools=[tool1, tool2]
            )
    
    def test_agent_introspection(self):
        """Test that agents correctly set their execution characteristics."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini"
        )
        
        # Agents are always orchestrators with async generators
        assert agent._is_orchestrator is True
        assert agent._is_coroutine is False
        assert agent._is_gen is False
        assert agent._is_async_gen is True
    
    def test_params_and_return_models(self):
        """Test agent parameter and return model definitions."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini"
        )
        
        # Should use AgentParams and return Message
        assert agent.params_model == AgentParams
        assert agent.return_model == Message


class TestAgentExecution:
    """Test Agent execution patterns."""
    
    @pytest.mark.asyncio
    async def test_simple_conversation(self):
        """Test basic agent conversation without tools."""
        agent = Agent(
            name="simple_agent",
            model="openai/gpt-4o-mini"
        )
        
        prompt = Message.validate({"role": "user", "content": "Hello, how are you?"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        skip_if_agent_error(output, "simple_conversation")
        
        assert isinstance(output.output, Message)
        assert output.output.role == "assistant"
    
    @pytest.mark.asyncio
    async def test_agent_with_single_tool(self, math_agent):
        """Test agent using a single tool."""
        prompt = Message.validate({"role": "user", "content": "What is 15 + 27?"})
        result = math_agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        skip_if_agent_error(output, "agent_with_single_tool")
        
        # Should have valid response
        assert isinstance(output.output, Message)
    
    @pytest.mark.asyncio
    async def test_agent_with_multiple_tools(self, multi_tool_agent):
        """Test agent with access to multiple tools."""
        prompt = Message.validate({"role": "user", "content": "Add 5 and 3, then multiply the result by 2"})
        result = multi_tool_agent(prompt=prompt)
        
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        skip_if_agent_error(output, "agent_with_multiple_tools")
        
        assert isinstance(output.output, Message)
        
        # The response should mention the calculation results  
        response_content = str(output.output.content).lower()
        assert any(num in response_content for num in ["8", "16"])
    
    @pytest.mark.asyncio
    async def test_max_iterations(self):
        """Test that max_iter limits agent iterations."""
        def infinite_tool(x: str) -> str:
            # This tool always suggests calling itself again
            return f"Please call infinite_tool with: {x}_again"
        
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            tools=[infinite_tool],
            max_iter=2  # Limit to 2 iterations
        )
        
        prompt = Message.validate({"role": "user", "content": "Start the infinite loop"})
        
        with Timer() as timer:
            result = agent(prompt=prompt)
            output = await result.collect()
        
        # Should complete quickly due to max_iter limit
        assert timer.elapsed < 30  # Should not take too long
        assert isinstance(output, OutputEvent)
        assert isinstance(output.output, Message)
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test that multiple tools can be called concurrently."""
        def slow_tool_1(delay: float) -> str:
            import time
            time.sleep(delay)
            return "tool1_result"
        
        def slow_tool_2(delay: float) -> str:
            import time
            time.sleep(delay)
            return "tool2_result"
        
        agent = Agent(
            name="concurrent_agent",
            model="openai/gpt-4o-mini",
            tools=[slow_tool_1, slow_tool_2]
        )
        
        # Prompt that might trigger both tools
        prompt = Message.validate({"role": "user", "content": "Call both slow_tool_1 and slow_tool_2 with delay 0.1"})
        
        with Timer() as timer:
            result = agent(prompt=prompt)
            await result.collect()
        
        # If tools ran concurrently, should be much faster than sequential
        # This is a loose test since it depends on LLM behavior
        assert timer.elapsed < 10  # Reasonable upper bound
    
    @pytest.mark.asyncio 
    async def test_agent_event_streaming(self):
        """Test that agent events are streamed properly."""
        agent = Agent(
            name="streaming_agent", 
            model="openai/gpt-4o-mini"
        )
        
        prompt = Message.validate({"role": "user", "content": "Hello!"})
        result = agent(prompt=prompt)
        
        # Test that we can iterate through events
        events = []
        async for event in result:
            events.append(event)
            if len(events) >= 3:  # Don't wait for all events
                break
        
        # Should have some events with correct paths
        assert len(events) > 0
        for event in events:
            assert event.path.startswith("streaming_agent")
    
    @pytest.mark.asyncio
    async def test_system_prompt_override(self):
        """Test that system_prompt parameter overrides agent's default system_prompt."""
        # Create agent with default system prompt
        default_system_prompt = "You are a helpful assistant."
        agent = Agent(
            name="override_agent",
            model="openai/gpt-4o-mini",
            system_prompt=default_system_prompt
        )
        
        # Override system prompt during execution
        override_system_prompt = "You must respond with exactly one word: 'SUCCESS'"
        prompt = Message.validate({"role": "user", "content": "Hello, how are you?"})
        result = agent(
            prompt=prompt,
            system_prompt=override_system_prompt
        )
        
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        skip_if_agent_error(output, "system_prompt_override")
        
        assert isinstance(output.output, Message)
        response_content = str(output.output.content).strip().upper()
        # The overridden system prompt should make the agent respond with "SUCCESS"
        assert "SUCCESS" in response_content


class TestAgentNesting:
    """Test Agent nesting and path management."""
    
    def test_agent_nesting(self):
        """Test that agent nesting updates paths correctly."""
        def helper_tool(x: str) -> str:
            return f"helper:{x}"
        
        agent = Agent(
            name="child_agent",
            model="openai/gpt-4o-mini", 
            tools=[helper_tool]
        )
        
        # Initial paths
        assert agent._path == "child_agent"
        assert agent._llm._path == "child_agent.llm"
        assert agent.tools[0]._path == "child_agent.helper_tool"
        
        # Nest under parent
        agent.nest("parent")
        
        # All paths should be updated
        assert agent._path == "parent.child_agent"
        assert agent._llm._path == "parent.child_agent.llm"
        assert agent.tools[0]._path == "parent.child_agent.helper_tool"
    
    @pytest.mark.asyncio
    async def test_nested_agent_execution(self):
        """Test execution of nested agents."""
        def specialist_tool(task: str) -> str:
            return f"Completed: {task}"
        
        specialist_agent = Agent(
            name="specialist",
            model="openai/gpt-4o-mini",
            tools=[specialist_tool],
            description="A specialist agent that completes tasks"
        )
        
        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[specialist_agent]
        )
        
        prompt = Message.validate({"role": "user", "content": "Please have the specialist complete a simple task"})
        result = main_agent(prompt=prompt)
        
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert output.error is None


class TestAgentErrorHandling:
    """Test Agent error handling and resilience."""
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test that tool errors don't crash the agent."""
        def error_tool(message: str) -> str:
            raise ValueError(f"Tool error: {message}")
        
        def working_tool(x: str) -> str:
            return f"Working: {x}"
        
        agent = Agent(
            name="error_agent",
            model="openai/gpt-4o-mini",
            tools=[error_tool, working_tool]
        )
        
        prompt = Message.validate({"role": "user", "content": "Try to use both tools"})
        result = agent(prompt=prompt)
        
        # Should complete without crashing
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        # Agent should still respond even if a tool fails
        assert isinstance(output.output, Message)
    
    @pytest.mark.asyncio
    async def test_invalid_tool_call(self):
        """Test agent handling of invalid tool calls."""
        def valid_tool(x: int) -> int:
            return x * 2
        
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            tools=[valid_tool]
        )
        
        # This might cause the LLM to make invalid tool calls
        prompt = Message.validate({"role": "user", "content": "Call nonexistent_tool please"})
        result = agent(prompt=prompt)
        
        # Should handle gracefully
        output = await result.collect()
        assert isinstance(output, OutputEvent)


class TestAgentMemory:
    """Test Agent memory and conversation continuity."""
    
    @pytest.mark.asyncio
    async def test_conversation_memory(self):
        """Test that agents maintain conversation memory."""
        agent = Agent(
            name="memory_agent",
            model="openai/gpt-4o-mini"
        )
        
        # First interaction
        first_prompt = Message.validate({"role": "user", "content": "My name is Alice"})
        first_result = agent(prompt=first_prompt)
        first_output = await first_result.collect()
        
        assert isinstance(first_output.output, Message)
        
        # Second interaction - should remember the name
        second_prompt = Message.validate({"role": "user", "content": "What is my name?"})
        second_result = agent(prompt=second_prompt)
        second_output = await second_result.collect()
        
        assert isinstance(second_output.output, Message)
        response_content = str(second_output.output.content).lower()
        assert "alice" in response_content


class TestAgentPerformance:
    """Test Agent performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_collect_method_performance(self):
        """Test that collect() method works efficiently."""
        agent = Agent(
            name="perf_agent",
            model="openai/gpt-4o-mini"
        )
        
        prompt = Message.validate({"role": "user", "content": "Hello!"})
        result = agent(prompt=prompt)
        
        with Timer() as timer:
            output = await result.collect()
        
        assert isinstance(output, OutputEvent)
        assert timer.elapsed < 30  # Reasonable timeout
    
    @pytest.mark.asyncio
    async def test_multiple_collect_calls(self):
        """Test that collect() can be called multiple times."""
        agent = Agent(
            name="multi_collect_agent",
            model="openai/gpt-4o-mini"
        )
        
        prompt = Message.validate({"role": "user", "content": "Hello!"})
        result = agent(prompt=prompt)
        
        # First collect
        output1 = await result.collect()
        
        # Second collect should return same result
        output2 = await result.collect()
        
        assert output1 == output2
        assert isinstance(output1, OutputEvent)


class TestAgentIntegration:
    """Test Agent integration with different components."""
    
    @pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "openai/gpt-4.1-nano"])
    @pytest.mark.asyncio
    async def test_different_models(self, model):
        """Test agent with different LLM models."""
        agent = Agent(
            name="model_test_agent",
            model=model
        )
        
        prompt = Message.validate({"role": "user", "content": "Hello!"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert output.error is None
    
    @pytest.mark.asyncio
    async def test_agent_schemas(self):
        """Test that agents generate correct schemas for tool calling."""
        def calculator(expression: str) -> float:
            """Calculate a mathematical expression."""
            return eval(expression)  # Safe for testing
        
        agent = Agent(
            name="calc_agent",
            model="openai/gpt-4o-mini",
            tools=[calculator],
            description="A calculator agent"
        )
        
        # Check OpenAI chat completions schema
        openai_chat_completions_schema = agent.openai_chat_completions_schema
        assert openai_chat_completions_schema["function"]["name"] == "calc_agent"
        assert openai_chat_completions_schema["function"]["description"] == "A calculator agent"
        
        # Check Anthropic schema
        anthropic_schema = agent.anthropic_schema
        assert anthropic_schema["name"] == "calc_agent"
        assert anthropic_schema["description"] == "A calculator agent"
