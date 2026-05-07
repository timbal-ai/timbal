import asyncio

import pytest
from timbal import Agent, Tool
from timbal.core.agent import AgentParams
from timbal.core.test_model import TestModel
from timbal.types.content import CustomContent, TextContent, ToolResultContent, ToolUseContent
from timbal.types.events import OutputEvent
from timbal.types.message import Message

from ..conftest import Timer, assert_has_output_event, assert_no_errors


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
            model=TestModel(),
        )

        prompt = Message.validate({"role": "user", "content": "Hello, how are you?"})
        result = agent(prompt=prompt)

        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert_no_errors(output)
        assert isinstance(output.output, Message)
        assert output.output.role == "assistant"

    @pytest.mark.asyncio
    async def test_internal_llm_tool_does_not_emit_default_request_usage(self, math_agent):
        """Internal ``llm`` is a :class:`~timbal.core.tool.Tool` but must not bill ``llm:requests``."""
        prompt = Message.validate({"role": "user", "content": "What is 15 + 27?"})
        output = await math_agent(prompt=prompt).collect()
        assert_no_errors(output)
        assert output.usage.get("llm:requests", 0) == 0
        assert output.usage.get("calculate:requests") == 1
    
    @pytest.mark.asyncio
    async def test_agent_with_single_tool(self, math_agent):
        """Test agent using a single tool."""
        prompt = Message.validate({"role": "user", "content": "What is 15 + 27?"})
        result = math_agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        assert isinstance(output.output, Message)
    
    @pytest.mark.asyncio
    async def test_agent_with_multiple_tools(self, multi_tool_agent):
        """Test agent with access to multiple tools."""
        prompt = Message.validate({"role": "user", "content": "Add 5 and 3, then multiply the result by 2"})
        result = multi_tool_agent(prompt=prompt)
        
        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert_no_errors(output)
        assert isinstance(output.output, Message)
        # The fixture's TestModel computes 5+3=8 and 8*2=16 via actual tool calls
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
            model=TestModel(responses=[
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="c1", name="infinite_tool", input={"x": "start"})],
                    stop_reason="tool_use",
                ),
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="c2", name="infinite_tool", input={"x": "start_again"})],
                    stop_reason="tool_use",
                ),
                "Done.",
            ]),
            tools=[infinite_tool],
            max_iter=2,
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
        def slow_tool_1(_delay: float) -> str:
            import time
            time.sleep(0.2)
            return "tool1_result"

        def slow_tool_2() -> str:
            import time
            time.sleep(0.2)
            return "tool2_result"

        agent = Agent(
            name="concurrent_agent",
            model=TestModel(responses=[
                Message(
                    role="assistant",
                    content=[
                        ToolUseContent(id="c1", name="slow_tool_1", input={"delay": 0.2}),
                        ToolUseContent(id="c2", name="slow_tool_2", input={}),
                    ],
                    stop_reason="tool_use",
                ),
                "Both tools finished.",
            ]),
            tools=[slow_tool_1, slow_tool_2],
        )

        slow_tool_1_started = False
        slow_tool_2_started = False
        async for event in agent(prompt=Message.validate({"role": "user", "content": "go"})):
            if event.type == "START" and event.path == "concurrent_agent.slow_tool_1":
                slow_tool_1_started = True
            if event.type == "START" and event.path == "concurrent_agent.slow_tool_2":
                slow_tool_2_started = True
            if event.type == "OUTPUT" and event.path == "concurrent_agent.slow_tool_1":
                if not slow_tool_2_started:
                    pytest.fail("slow_tool_1 finished before slow_tool_2 started")
            if event.type == "OUTPUT" and event.path == "concurrent_agent.slow_tool_2":
                if not slow_tool_1_started:
                    pytest.fail("slow_tool_2 finished before slow_tool_1 started")

    
    @pytest.mark.asyncio
    async def test_agent_event_streaming(self):
        """Test that agent events are streamed properly."""
        agent = Agent(
            name="streaming_agent",
            model=TestModel(),
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
        agent = Agent(
            name="override_agent",
            model=TestModel(),
            system_prompt="You are a helpful assistant.",
        )

        prompt = Message.validate({"role": "user", "content": "Hello, how are you?"})
        result = agent(prompt=prompt, system_prompt="Overridden prompt.")

        output = await result.collect()
        assert isinstance(output, OutputEvent)
        assert_no_errors(output)
        assert isinstance(output.output, Message)


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
            model=TestModel(),
            tools=[specialist_tool],
            description="A specialist agent that completes tasks",
        )

        main_agent = Agent(
            name="main_agent",
            model=TestModel(responses=[
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="c1", name="specialist", input={"prompt": "do a task"})],
                    stop_reason="tool_use",
                ),
                "Done.",
            ]),
            tools=[specialist_agent],
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
            model=TestModel(responses=[
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="c1", name="error_tool", input={"message": "oops"})],
                    stop_reason="tool_use",
                ),
                "The tool failed but I recovered.",
            ]),
            tools=[error_tool, working_tool],
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
            model=TestModel(),
            tools=[valid_tool],
        )

        prompt = Message.validate({"role": "user", "content": "Call nonexistent_tool please"})
        result = agent(prompt=prompt)

        # Should handle gracefully — TestModel responds without calling any tool
        output = await result.collect()
        assert isinstance(output, OutputEvent)

    @pytest.mark.asyncio
    async def test_hallucinated_tool_does_not_hang(self):
        """LLM emits a tool_use for a tool that doesn't exist (e.g. a skill name
        called as if it were a tool). Agent must not hang on queue.get(); it must
        feed an error tool_result back to the LLM so it can self-correct."""
        def real_tool(x: int) -> int:
            return x * 2

        agent = Agent(
            name="hallucinate_agent",
            model=TestModel(responses=[
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="c1", name="not_a_tool", input={"x": 5})],
                    stop_reason="tool_use",
                ),
                "Sorry, I retried with the right tool.",
            ]),
            tools=[real_tool],
            max_iter=3,
        )

        with Timer() as timer:
            output = await asyncio.wait_for(
                agent(prompt="hi").collect(),
                timeout=10.0,
            )

        assert timer.elapsed < 5
        assert isinstance(output, OutputEvent)
        assert output.status.code == "success"
        assert output.error is None
        assert "retried" in output.output.collect_text().lower()

    @pytest.mark.asyncio
    async def test_hallucinated_tool_logs_warning_with_suggestion(self, caplog):
        """The error fed back to the LLM should include close matches as a hint."""
        import logging
        caplog.set_level(logging.WARNING)

        def make_pptx(slides: int) -> str:
            return f"made {slides} slides"

        # LLM hallucinates `make_ppt` (close to `make_pptx`)
        agent = Agent(
            name="suggest_agent",
            model=TestModel(responses=[
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="c1", name="make_ppt", input={"slides": 3})],
                    stop_reason="tool_use",
                ),
                "Got it, used the real tool.",
            ]),
            tools=[make_pptx],
            max_iter=3,
        )

        captured_results: list[str] = []
        async for event in agent(prompt="make a deck"):
            if isinstance(event, OutputEvent) and event.path.endswith(".make_ppt"):
                # synthetic event for the hallucinated tool
                captured_results.append(event.error)

        assert captured_results, "expected synthetic error event for hallucinated tool"
        msg = captured_results[0]
        assert "make_pptx" in msg
        assert "Did you mean" in msg


class TestAgentMemory:
    """Test Agent memory and conversation continuity."""

    @pytest.mark.asyncio
    async def test_memory_dump_matches_memory_across_turns(self):
        """_memory_dump must be the serialized form of memory at every turn.

        This exercises the _prev_memory_dump optimization: on turn 2+, we reuse
        the already-serialized previous span memory and only dump new messages.
        The resulting _memory_dump must be identical to what a full re-dump would
        produce — i.e. the optimization must be transparent to the trace.
        """
        from timbal.state.tracing.providers import InMemoryTracingProvider
        from timbal.utils import dump

        turn_count = 0

        def counting_handler(_messages):
            nonlocal turn_count
            turn_count += 1
            return f"response {turn_count}"

        provider = InMemoryTracingProvider.configured()
        agent = Agent(
            name="dump_agent",
            model=TestModel(handler=counting_handler),
            tracing_provider=provider,
        )

        # Run 4 turns — enough to exercise the incremental path multiple times
        run_ids = []
        for i in range(4):
            out = await agent(prompt=f"message {i}").collect()
            run_ids.append(out.run_id)

        # For each stored trace, verify _memory_dump == dump(memory) for agent spans
        for run_id in run_ids:
            trace = provider._storage.get(run_id)
            assert trace is not None, f"No trace stored for run {run_id}"
            for call_id, span in trace.items():
                if not hasattr(span, "_memory_dump") or span._memory_dump is None:
                    continue
                if not hasattr(span, "memory") or not isinstance(span.memory, list):
                    continue
                # Re-dump from scratch and compare — must equal the incremental result
                expected = await dump([Message.validate(m) for m in span.memory])
                assert span._memory_dump == expected, (
                    f"_memory_dump mismatch for span {call_id} in run {run_id}: "
                    f"got {len(span._memory_dump)} items, expected {len(expected)}"
                )

    @pytest.mark.asyncio
    async def test_conversation_memory(self):
        """Test that agents pass conversation history to the LLM on subsequent calls."""

        def memory_handler(messages):
            # Verify the framework passes prior turns in messages
            user_texts = [m.collect_text() for m in messages if m.role == "user"]
            if any("alice" in t.lower() for t in user_texts):
                return "Your name is Alice."
            return "Nice to meet you!"

        agent = Agent(name="memory_agent", model=TestModel(handler=memory_handler))

        first_output = await agent(prompt=Message.validate({"role": "user", "content": "My name is Alice"})).collect()
        assert isinstance(first_output.output, Message)

        second_output = await agent(prompt=Message.validate({"role": "user", "content": "What is my name?"})).collect()
        assert isinstance(second_output.output, Message)
        assert "alice" in str(second_output.output.content).lower()


    @pytest.mark.asyncio
    async def test_assistant_message_as_prompt_normalized_to_user(self):
        """When an assistant Message is passed as prompt (e.g. mapped from another
        agent's output in a workflow), it should be normalized to role='user' so
        the LLM doesn't receive an invalid conversation structure."""
        agent = Agent(
            name="receiving_agent",
            model=TestModel(),
        )

        # Simulate what happens when agent output is mapped as another agent's prompt.
        assistant_msg = Message.validate({"role": "assistant", "content": "The answer is 42."})
        result = agent(prompt=assistant_msg)
        output = await result.collect()

        assert_no_errors(output)
        assert isinstance(output.output, Message)


class TestAgentPerformance:
    """Test Agent performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_collect_method_performance(self):
        """Test that collect() method works efficiently."""
        agent = Agent(
            name="perf_agent",
            model=TestModel(),
        )

        prompt = Message.validate({"role": "user", "content": "Hello!"})
        result = agent(prompt=prompt)

        with Timer() as timer:
            output = await result.collect()

        assert isinstance(output, OutputEvent)
        assert timer.elapsed < 5  # Fast with TestModel
    
    @pytest.mark.asyncio
    async def test_multiple_collect_calls(self):
        """Test that collect() can be called multiple times."""
        agent = Agent(
            name="multi_collect_agent",
            model=TestModel(),
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
    
    @pytest.mark.asyncio
    async def test_different_models(self):
        """Test agent runs successfully regardless of model configuration."""
        for model in [TestModel(responses=["response A"]), TestModel(responses=["response B"])]:
            agent = Agent(name="model_test_agent", model=model)
            prompt = Message.validate({"role": "user", "content": "Hello!"})
            output = await agent(prompt=prompt).collect()
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
            model=TestModel(),
            tools=[calculator],
            description="A calculator agent",
        )
        
        # Check OpenAI chat completions schema
        openai_chat_completions_schema = agent.openai_chat_completions_schema
        assert openai_chat_completions_schema["function"]["name"] == "calc_agent"
        assert openai_chat_completions_schema["function"]["description"] == "A calculator agent"
        
        # Check Anthropic schema
        anthropic_schema = agent.anthropic_schema
        assert anthropic_schema["name"] == "calc_agent"
        assert anthropic_schema["description"] == "A calculator agent"


def _make_agent(model: str = "anthropic/claude-sonnet-4-6") -> Agent:
    kwargs = {"name": "test", "model": model, "tools": []}
    if model.startswith("anthropic/"):
        kwargs["max_tokens"] = 1024
    return Agent(**kwargs)


def _tool_use(id: str, *, server: bool = False) -> ToolUseContent:
    return ToolUseContent(id=id, name="fn", input={}, is_server_tool_use=server)


class TestSynthesizeMissingToolResults:
    """Unit tests for Agent._synthesize_missing_tool_results."""

    def test_no_tool_use_is_noop(self):
        """Last message has only text — nothing to synthesize."""
        agent = _make_agent()
        memory = [Message(role="assistant", content=[TextContent(text="hello")])]
        agent._synthesize_missing_tool_results(memory)
        assert len(memory) == 1
        assert len(memory[0].content) == 1

    def test_regular_tool_use_appends_error_result(self):
        """Interrupted regular tool_use gets a new tool-role message."""
        agent = _make_agent()
        memory = [Message(role="assistant", content=[_tool_use("tu_001")])]
        agent._synthesize_missing_tool_results(memory)
        assert len(memory) == 2
        result_msg = memory[1]
        assert result_msg.role == "tool"
        assert len(result_msg.content) == 1
        result = result_msg.content[0]
        assert isinstance(result, ToolResultContent)
        assert result.id == "tu_001"
        assert "ERROR" in result.content[0].text

    def test_multiple_regular_tool_uses_each_get_result(self):
        """Two interrupted tool_use blocks → two appended result messages."""
        agent = _make_agent()
        memory = [Message(role="assistant", content=[_tool_use("tu_001"), _tool_use("tu_002")])]
        agent._synthesize_missing_tool_results(memory)
        # One result message per tool_use
        assert len(memory) == 3
        result_ids = {memory[i].content[0].id for i in (1, 2)}
        assert result_ids == {"tu_001", "tu_002"}

    def test_server_tool_use_anthropic_appends_inline_error(self):
        """Interrupted server_tool_use on Anthropic → inline CustomContent added to same message."""
        agent = _make_agent("anthropic/claude-sonnet-4-6")
        memory = [Message(role="assistant", content=[_tool_use("svu_001", server=True)])]
        agent._synthesize_missing_tool_results(memory)
        # No new message — inline content appended to last message
        assert len(memory) == 1
        assert len(memory[0].content) == 2
        inline = memory[0].content[1]
        assert isinstance(inline, CustomContent)
        assert inline.value["tool_use_id"] == "svu_001"
        assert inline.value["type"] == "web_search_tool_result"

    def test_server_tool_use_non_anthropic_skipped(self):
        """Interrupted server_tool_use on a non-Anthropic provider is skipped (no mutation)."""
        agent = _make_agent("openai/gpt-4o")
        memory = [Message(role="assistant", content=[_tool_use("svu_001", server=True)])]
        agent._synthesize_missing_tool_results(memory)
        assert len(memory) == 1
        assert len(memory[0].content) == 1

    def test_server_tool_use_with_followup_is_skipped(self):
        """server_tool_use followed by text content means the LLM already continued — no synthesis."""
        agent = _make_agent("anthropic/claude-sonnet-4-6")
        memory = [
            Message(
                role="assistant",
                content=[_tool_use("svu_001", server=True), TextContent(text="Here is the answer")],
            )
        ]
        agent._synthesize_missing_tool_results(memory)
        assert len(memory) == 1
        assert len(memory[0].content) == 2  # unchanged
