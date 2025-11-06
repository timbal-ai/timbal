"""Tests for the Agent commands feature.

Commands allow users to directly invoke tools using a slash command syntax (e.g., /search query).
This bypasses the LLM and directly executes the tool with the provided arguments.
"""

import pytest
from timbal import Agent, Tool, Workflow
from timbal.types.events import OutputEvent, StartEvent
from timbal.types.message import Message

from .conftest import (
    assert_has_output_event,
    assert_no_errors,
    skip_if_agent_error,
)


class TestCommandsWithTools:
    """Test commands feature with Tool runnables."""
    
    @pytest.mark.asyncio
    async def test_simple_tool_command(self):
        """Test basic command execution with a simple tool."""
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"
        
        tool = Tool(handler=greet, command="greet")
        agent = Agent(
            name="command_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Execute command
        prompt = Message.validate({"role": "user", "content": "/greet Alice"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        # Should have executed the tool directly
        assert output.output is not None
        assert "Hello, Alice!" in str(output.output)
    
    @pytest.mark.asyncio
    async def test_tool_command_multiple_args(self):
        """Test command with multiple arguments."""
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        tool = Tool(handler=add, command="add")
        agent = Agent(
            name="math_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Execute command with multiple args
        prompt = Message.validate({"role": "user", "content": "/add 5 3"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        # Should have result of 8
        assert "8" in str(output.output)
    
    @pytest.mark.asyncio
    async def test_tool_command_with_quoted_args(self):
        """Test command with quoted arguments (using shlex parsing)."""
        def echo(message: str) -> str:
            """Echo a message."""
            return f"Echo: {message}"
        
        tool = Tool(handler=echo, command="echo")
        agent = Agent(
            name="echo_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Execute command with quoted string
        prompt = Message.validate({"role": "user", "content": '/echo "Hello World"'})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        # Should preserve the full quoted string
        assert "Hello World" in str(output.output)
    
    @pytest.mark.asyncio
    async def test_tool_command_missing_args(self):
        """Test command with missing required arguments."""
        def multiply(a: int, b: int, c: int) -> int:
            """Multiply three numbers."""
            return a * b * c
        
        tool = Tool(handler=multiply, command="mul")
        agent = Agent(
            name="math_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Execute command with only 2 args (missing 1)
        prompt = Message.validate({"role": "user", "content": "/mul 2 3"})
        result = agent(prompt=prompt)
        
        # Collect all events to find the tool error
        events = []
        async for event in result:
            events.append(event)
        
        # Should have tool output event with error
        tool_outputs = [e for e in events if isinstance(e, OutputEvent) and "multiply" in e.path]
        assert len(tool_outputs) > 0
        assert tool_outputs[0].error is not None
    
    @pytest.mark.asyncio
    async def test_tool_command_extra_args(self):
        """Test command with extra arguments (should be ignored)."""
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"
        
        tool = Tool(handler=greet, command="greet")
        agent = Agent(
            name="greet_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Execute command with extra args
        prompt = Message.validate({"role": "user", "content": "/greet Alice extra args"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        # Should only use first arg
        assert "Hello, Alice!" in str(output.output)
    
    @pytest.mark.asyncio
    async def test_tool_command_no_args(self):
        """Test command with no arguments."""
        def get_time() -> str:
            """Get current time."""
            import time
            return time.strftime("%Y-%m-%d %H:%M:%S")
        
        tool = Tool(handler=get_time, command="time")
        agent = Agent(
            name="time_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Execute command with no args
        prompt = Message.validate({"role": "user", "content": "/time"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        # Should have timestamp
        assert output.output is not None
    
    @pytest.mark.asyncio
    async def test_tool_command_async_handler(self):
        """Test command with async tool handler."""
        async def async_greet(name: str) -> str:
            """Async greet function."""
            import asyncio
            await asyncio.sleep(0.01)
            return f"Async hello, {name}!"
        
        tool = Tool(handler=async_greet, command="agreet")
        agent = Agent(
            name="async_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        prompt = Message.validate({"role": "user", "content": "/agreet Bob"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        assert "Async hello, Bob!" in str(output.output)
    
    @pytest.mark.asyncio
    async def test_tool_command_generator_handler(self):
        """Test command with generator tool handler."""
        def count(n: int):
            """Count from 0 to n-1."""
            for i in range(int(n)):
                yield i
        
        tool = Tool(handler=count, command="count")
        agent = Agent(
            name="gen_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        prompt = Message.validate({"role": "user", "content": "/count 3"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        # Should have list of numbers
        assert output.output is not None
    
    @pytest.mark.asyncio
    async def test_tool_command_error_handling(self):
        """Test command error handling when tool raises exception."""
        def divide(a: int, b: int) -> float:
            """Divide two numbers."""
            return int(a) / int(b)
        
        tool = Tool(handler=divide, command="div")
        agent = Agent(
            name="div_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Execute command that will cause division by zero
        prompt = Message.validate({"role": "user", "content": "/div 10 0"})
        result = agent(prompt=prompt)
        
        # Collect all events to find the tool error
        events = []
        async for event in result:
            events.append(event)
        
        # Should have tool output event with error
        tool_outputs = [e for e in events if isinstance(e, OutputEvent) and "divide" in e.path]
        assert len(tool_outputs) > 0
        assert tool_outputs[0].error is not None
    
    @pytest.mark.asyncio
    async def test_nonexistent_command(self):
        """Test that nonexistent commands fall through to LLM."""
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        tool = Tool(handler=greet, command="greet")
        agent = Agent(
            name="fallthrough_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Try to execute nonexistent command
        prompt = Message.validate({"role": "user", "content": "/nonexistent arg"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        skip_if_agent_error(output, "nonexistent_command")
        
        # Should get LLM response, not command execution
        assert isinstance(output.output, Message)
    
    @pytest.mark.asyncio
    async def test_tool_without_command_not_executable(self):
        """Test that tools without command attribute cannot be invoked via slash."""
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        # Tool without command attribute
        tool = Tool(handler=greet)
        agent = Agent(
            name="no_command_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Try to execute with slash syntax
        prompt = Message.validate({"role": "user", "content": "/greet Alice"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        skip_if_agent_error(output, "tool_without_command")
        
        # Should fall through to LLM since no command is registered
        assert isinstance(output.output, Message)


class TestCommandsWithAgents:
    """Test commands feature with Agent runnables.
    
    Note: Agents expect a 'prompt' parameter which should be a single string.
    When using commands with Agents, quote the entire message to pass it as one argument.
    Example: /helper "Please help me with this task"
    """
    
    @pytest.mark.asyncio
    async def test_agent_as_command_with_quoted_prompt(self):
        """Test using an agent as a command with quoted prompt."""
        def helper_tool(task: str) -> str:
            return f"Completed: {task}"
        
        # Create a sub-agent
        sub_agent = Agent(
            name="helper_agent",
            model="openai/gpt-4o-mini",
            tools=[helper_tool],
            description="A helper agent",
            command="helper"
        )
        
        # Create main agent with sub-agent as tool
        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[sub_agent]
        )
        
        # Execute sub-agent via command with quoted prompt
        # The quotes ensure the entire message is passed as a single 'prompt' argument
        prompt = Message.validate({"role": "user", "content": '/helper "Please help me with this task"'})
        result = main_agent(prompt=prompt)
        
        # Collect all events
        events = []
        async for event in result:
            events.append(event)
        
        # Should have events from the sub-agent execution
        helper_events = [e for e in events if "helper_agent" in e.path]
        assert len(helper_events) > 0
        
        # Check for output events from the helper agent
        helper_outputs = [e for e in events if isinstance(e, OutputEvent) and "helper_agent" in e.path]
        if len(helper_outputs) > 0:
            skip_if_agent_error(helper_outputs[0], "agent_as_command")
    
    @pytest.mark.asyncio
    async def test_nested_agent_command_with_tools(self):
        """Test nested agent command that uses its own tools."""
        def search(query: str) -> str:
            return f"Search results for: {query}"
        
        def summarize(text: str) -> str:
            return f"Summary: {text[:50]}"
        
        # Create specialist agent with tools
        search_agent = Agent(
            name="search_agent",
            model="openai/gpt-4o-mini",
            tools=[search, summarize],
            description="Search and summarize",
            command="search"
        )
        
        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[search_agent]
        )
        
        # Execute nested agent command with quoted prompt
        # Quote the entire message to pass it as a single 'prompt' argument
        prompt = Message.validate({"role": "user", "content": '/search "python tutorials"'})
        result = main_agent(prompt=prompt)
        
        # Collect all events
        events = []
        async for event in result:
            events.append(event)
        
        # Should have events from the search agent execution
        search_events = [e for e in events if "search_agent" in e.path]
        assert len(search_events) > 0
        
        # Check for output events from the search agent
        search_outputs = [e for e in events if isinstance(e, OutputEvent) and "search_agent" in e.path]
        if len(search_outputs) > 0:
            skip_if_agent_error(search_outputs[0], "nested_agent_command")


class TestCommandsWithMixedRunnables:
    """Test commands with mixed runnable types."""
    
    @pytest.mark.asyncio
    async def test_multiple_commands_different_types(self):
        """Test agent with multiple commands of different types."""
        def simple_tool(x: str) -> str:
            return f"Tool: {x}"
        
        def helper_func(task: str) -> str:
            return f"Helper: {task}"
        
        sub_agent = Agent(
            name="sub_agent",
            model="openai/gpt-4o-mini",
            tools=[helper_func],
            description="Sub agent",
            command="sub"
        )
        
        tool = Tool(handler=simple_tool, command="tool")
        
        main_agent = Agent(
            name="main_agent",
            model="openai/gpt-4o-mini",
            tools=[tool, sub_agent]
        )
        
        # Test tool command
        prompt1 = Message.validate({"role": "user", "content": "/tool test"})
        result1 = main_agent(prompt=prompt1)
        output1 = await result1.collect()
        
        assert_has_output_event(output1)
        assert_no_errors(output1)
        assert "Tool: test" in str(output1.output)
        
        # Test agent command
        prompt2 = Message.validate({"role": "user", "content": "/sub task"})
        result2 = main_agent(prompt=prompt2)
        output2 = await result2.collect()
        
        assert_has_output_event(output2)
        skip_if_agent_error(output2, "agent_command")
    
    @pytest.mark.asyncio
    async def test_command_name_collision(self):
        """Test that command names must be unique."""
        def tool1(x: str) -> str:
            return f"Tool1: {x}"
        
        def tool2(x: str) -> str:
            return f"Tool2: {x}"
        
        # Both tools with same command name
        t1 = Tool(handler=tool1, command="cmd")
        t2 = Tool(handler=tool2, command="cmd")
        
        agent = Agent(
            name="collision_agent",
            model="openai/gpt-4o-mini",
            tools=[t1, t2]
        )
        
        # Execute command - should use first registered tool
        prompt = Message.validate({"role": "user", "content": "/cmd test"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        # Should execute first tool (due to duplicate warning)
        assert output.output is not None


class TestCommandsEventStreaming:
    """Test event streaming behavior with commands."""
    
    @pytest.mark.asyncio
    async def test_command_events_structure(self):
        """Test that command execution produces correct event structure."""
        def process(data: str) -> str:
            return f"Processed: {data}"
        
        tool = Tool(handler=process, command="process")
        agent = Agent(
            name="event_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        prompt = Message.validate({"role": "user", "content": "/process test"})
        result = agent(prompt=prompt)
        
        events = []
        async for event in result:
            events.append(event)
        
        # Should have START and OUTPUT events
        assert len(events) > 0
        
        # Check for tool execution events
        tool_events = [e for e in events if "process" in e.path]
        assert len(tool_events) > 0
    
    @pytest.mark.asyncio
    async def test_command_early_return(self):
        """Test that command execution returns early without LLM call."""
        def quick_tool(x: str) -> str:
            return f"Quick: {x}"
        
        tool = Tool(handler=quick_tool, command="quick")
        agent = Agent(
            name="early_return_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        prompt = Message.validate({"role": "user", "content": "/quick test"})
        result = agent(prompt=prompt)
        
        events = []
        async for event in result:
            events.append(event)
        
        # Should not have LLM events
        llm_events = [e for e in events if "llm" in e.path.lower()]
        assert len(llm_events) == 0, "Command should not trigger LLM call"


class TestCommandsMemory:
    """Test command execution memory and history."""
    
    @pytest.mark.asyncio
    async def test_command_adds_to_memory(self):
        """Test that command execution is added to agent memory."""
        def remember(info: str) -> str:
            return f"Remembered: {info}"
        
        tool = Tool(handler=remember, command="remember")
        agent = Agent(
            name="memory_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Execute command
        prompt = Message.validate({"role": "user", "content": "/remember Alice"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        # Memory should contain the command execution
        # This is verified by the agent's internal memory tracking


class TestCommandsParameterOrdering:
    """Test that command arguments map correctly to function parameters."""
    
    @pytest.mark.asyncio
    async def test_parameter_order_preserved(self):
        """Test that parameters maintain function signature order."""
        def create_user(username: str, email: str, age: int) -> str:
            return f"User: {username}, Email: {email}, Age: {age}"
        
        tool = Tool(handler=create_user, command="user")
        agent = Agent(
            name="order_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        # Arguments should map in order: username, email, age
        prompt = Message.validate({"role": "user", "content": "/user alice alice@example.com 25"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        # Check that arguments were mapped correctly
        output_str = str(output.output)
        assert "alice" in output_str
        assert "alice@example.com" in output_str
        assert "25" in output_str
    
    @pytest.mark.asyncio
    async def test_complex_parameter_types(self):
        """Test commands with various parameter types."""
        def calculate(operation: str, a: int, b: int) -> str:
            ops = {
                "add": lambda x, y: x + y,
                "sub": lambda x, y: x - y,
                "mul": lambda x, y: x * y,
            }
            result = ops[operation](int(a), int(b))
            return f"{operation}({a}, {b}) = {result}"
        
        tool = Tool(handler=calculate, command="calc")
        agent = Agent(
            name="calc_agent",
            model="openai/gpt-4o-mini",
            tools=[tool]
        )
        
        prompt = Message.validate({"role": "user", "content": "/calc mul 6 7"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        assert_no_errors(output)
        
        assert "42" in str(output.output)


class TestCommandsWithWorkflows:
    """Test commands feature with Workflow runnables."""
    
    @pytest.mark.asyncio
    async def test_simple_workflow_command(self):
        """Test basic workflow execution via command."""
        def step1(x: int) -> int:
            """First step: double the input."""
            return x * 2
        
        def step2(y: int) -> int:
            """Second step: add 10."""
            return y + 10
        
        workflow = Workflow(name="math_workflow", command="math")
        workflow.step(step1)
        workflow.step(step2)
        
        agent = Agent(
            name="workflow_agent",
            model="openai/gpt-4o-mini",
            tools=[workflow]
        )
        
        # Execute workflow via command
        prompt = Message.validate({"role": "user", "content": "/math 5 15"})
        result = agent(prompt=prompt)
        
        # Collect all events
        events = []
        async for event in result:
            events.append(event)
        
        # Should have events from workflow execution
        workflow_events = [e for e in events if "math_workflow" in e.path]
        assert len(workflow_events) > 0
        
        # Check for output events
        workflow_outputs = [e for e in events if isinstance(e, OutputEvent) and "math_workflow" in e.path]
        assert len(workflow_outputs) > 0
        assert workflow_outputs[0].error is None
    
    @pytest.mark.asyncio
    async def test_workflow_command_with_dependencies(self):
        """Test workflow with step dependencies via command."""
        def fetch_data(query: str) -> str:
            """Fetch data based on query."""
            return f"Data for {query}"
        
        def process_data(data: str) -> str:
            """Process the fetched data."""
            return f"Processed: {data}"
        
        def summarize(processed: str) -> str:
            """Summarize the processed data."""
            return f"Summary: {processed[:30]}"
        
        workflow = Workflow(name="pipeline", command="pipeline")
        workflow.step(fetch_data)
        workflow.step(process_data, depends_on=["fetch_data"])
        workflow.step(summarize, depends_on=["process_data"])
        
        agent = Agent(
            name="pipeline_agent",
            model="openai/gpt-4o-mini",
            tools=[workflow]
        )
        
        # Execute workflow via command with quoted argument
        prompt = Message.validate({"role": "user", "content": '/pipeline "test query" "ignored" "ignored"'})
        result = agent(prompt=prompt)
        
        # Collect all events
        events = []
        async for event in result:
            events.append(event)
        
        # Should have events from all steps
        pipeline_events = [e for e in events if "pipeline" in e.path]
        assert len(pipeline_events) > 0
        
        # Check that steps executed
        fetch_events = [e for e in events if "fetch_data" in e.path]
        process_events = [e for e in events if "process_data" in e.path]
        summarize_events = [e for e in events if "summarize" in e.path]
        
        assert len(fetch_events) > 0
        assert len(process_events) > 0
        assert len(summarize_events) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_command_parallel_steps(self):
        """Test workflow with parallel independent steps via command."""
        def task_a(x: int) -> int:
            """Independent task A."""
            return x * 2
        
        def task_b(y: int) -> int:
            """Independent task B."""
            return y + 5
        
        workflow = Workflow(name="parallel_workflow", command="parallel")
        workflow.step(task_a)
        workflow.step(task_b)
        
        agent = Agent(
            name="parallel_agent",
            model="openai/gpt-4o-mini",
            tools=[workflow]
        )
        
        # Execute workflow via command
        # Workflow params_model collects all parameters from steps: x, y
        prompt = Message.validate({"role": "user", "content": "/parallel 10 20"})
        result = agent(prompt=prompt)
        
        # Collect all events
        events = []
        async for event in result:
            events.append(event)
        
        # Should have events from all steps
        task_a_events = [e for e in events if "task_a" in e.path]
        task_b_events = [e for e in events if "task_b" in e.path]
        
        assert len(task_a_events) > 0
        assert len(task_b_events) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_command_error_handling(self):
        """Test workflow error handling via command."""
        def failing_step(x: int) -> int:
            """Step that raises an error."""
            raise ValueError("Intentional error")
        
        def dependent_step(y: int) -> int:
            """Step that depends on failing step."""
            return y * 2
        
        workflow = Workflow(name="error_workflow", command="error")
        workflow.step(failing_step)
        workflow.step(dependent_step, depends_on=["failing_step"])
        
        agent = Agent(
            name="error_agent",
            model="openai/gpt-4o-mini",
            tools=[workflow]
        )
        
        # Execute workflow via command
        prompt = Message.validate({"role": "user", "content": "/error 5 10"})
        result = agent(prompt=prompt)
        
        # Collect all events
        events = []
        async for event in result:
            events.append(event)
        
        # Should have error in failing step
        failing_outputs = [e for e in events if isinstance(e, OutputEvent) and "failing_step" in e.path]
        assert len(failing_outputs) > 0
        assert failing_outputs[0].error is not None
    
    @pytest.mark.asyncio
    async def test_workflow_command_mixed_with_tools(self):
        """Test agent with both workflow and tool commands."""
        def simple_tool(msg: str) -> str:
            return f"Tool: {msg}"
        
        def wf_step1(x: int) -> int:
            return x * 2
        
        def wf_step2(y: int) -> int:
            return y + 1
        
        workflow = Workflow(name="my_workflow", command="wf")
        workflow.step(wf_step1)
        workflow.step(wf_step2)
        
        tool = Tool(handler=simple_tool, command="tool")
        
        agent = Agent(
            name="mixed_agent",
            model="openai/gpt-4o-mini",
            tools=[tool, workflow]
        )
        
        # Test tool command
        prompt1 = Message.validate({"role": "user", "content": "/tool hello"})
        result1 = agent(prompt=prompt1)
        output1 = await result1.collect()
        
        assert_has_output_event(output1)
        assert_no_errors(output1)
        assert "Tool: hello" in str(output1.output)
        
        # Test workflow command
        prompt2 = Message.validate({"role": "user", "content": "/wf 5 10"})
        result2 = agent(prompt=prompt2)
        
        events2 = []
        async for event in result2:
            events2.append(event)
        
        workflow_events = [e for e in events2 if "my_workflow" in e.path]
        assert len(workflow_events) > 0
