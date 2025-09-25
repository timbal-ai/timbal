import tempfile

import pytest
from timbal import Agent
from timbal.tools.bash import Bash


class TestBashToolInitialization:
    """Test Bash tool initialization and basic functionality."""
    
    def test_bash_tool_creation(self):
        """Test that Bash tool is created correctly."""
        cmd = "echo"
        bash_tool = Bash(cmd)
        
        assert bash_tool.name == "echo_tool"
        assert bash_tool.cmd == cmd
        assert bash_tool.handler is not None
    
    def test_bash_tool_handler_signature(self):
        """Test that the handler has the correct signature."""
        bash_tool = Bash("echo")
        
        # Check that handler accepts the expected parameters
        import inspect
        sig = inspect.signature(bash_tool.handler)
        params = list(sig.parameters.keys())
        
        expected_params = ['args', 'background', 'capture_output', 'cwd']
        for param in expected_params:
            assert param in params


class TestBashCommandExecution:
    """Test Bash command execution functionality."""
    
    @pytest.mark.asyncio
    async def test_simple_command_execution(self):
        """Test executing a simple command."""
        bash_tool = Bash("echo")
        result = bash_tool(args="hello world")
        
        output = await result.collect()
        assert output.error is None
        assert "hello world" in output.output['output']
    
    @pytest.mark.asyncio
    async def test_command_with_no_args(self):
        """Test executing command without additional arguments."""
        bash_tool = Bash("echo")
        result = bash_tool()
        
        output = await result.collect()
        assert output.error is None
        # Should execute just "echo" which typically outputs empty line
        assert isinstance(output.output['output'], str)
    
    @pytest.mark.asyncio
    async def test_command_with_cwd(self):
        """Test executing command with custom working directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bash_tool = Bash("pwd")
            result = bash_tool(cwd=temp_dir)
            
            output = await result.collect()
            assert output.error is None
            assert temp_dir in output.output['output']
    
    @pytest.mark.asyncio
    async def test_command_without_capture_output(self):
        """Test executing command without capturing output."""
        bash_tool = Bash("echo")
        result = bash_tool(args="test", capture_output=False)
        
        output = await result.collect()
        assert output.error is None
        # When not capturing output, stdout should be None => command line output 
        assert output.output['output'] is None
    
    # TODO: check behaviour
    @pytest.mark.asyncio
    async def test_command_that_fails(self):
        """Test executing a command that fails."""
        bash_tool = Bash("nonexistentcommand12345")
        result = bash_tool()
        
        output = await result.collect()
        # Empty output for nonexistent command
        assert output.output['output'] == ''

    @pytest.mark.asyncio
    async def test_command_with_shell_expansion(self):
        """Test executing command with shell expansion."""
        bash_tool = Bash("ls")
        result = bash_tool(args="*.py")
        
        output = await result.collect()
        assert output.error is None
        assert isinstance(output.output['output'], str)

    @pytest.mark.asyncio
    async def test_tool_with_agent_not_matching_cmd(self):
        """Test agent with Bash tool when the command doesn't match the request."""
        agent = Agent(
            name="code_analyzer",
            model="openai/gpt-4o-mini",
            system_prompt="""You are a helpful code analysis assistant.
            When writing code, make sure to test the code and check that it works as expected.
            Analyse whether the process executions are long and should be run in the background.
            """,
            tools=[Bash(cmd="mkdir")]
        )

        response = await agent(
            prompt="List all Python files in the current directory"
        ).collect()
        
        # The agent should recognize that mkdir is not appropriate for listing files
        # and either explain this or suggest an alternative
        assert response.error is None
        assert response.output is not None
        response_text = response.output.content[0].text.lower()
        
        # Should indicate that mkdir is not suitable for listing files
        assert any(keyword in response_text for keyword in [
            "mkdir", "directory", "create", "not appropriate", "cannot", "unable"
        ])

    @pytest.mark.asyncio
    async def test_tool_functionality(self):
        """Test agent with Bash tool when the command matches the request."""
        tool = Bash(cmd="ls")
        result = await tool(args="*.py").collect()
        
        assert "test_bash.py" in result.output['output']
    

class TestBashBackgroundExecution:
    """Test Bash background execution functionality."""
    
    @pytest.mark.asyncio
    async def test_background_execution(self):
        """Test executing command in background."""
        bash_tool = Bash("sleep")
        agent = Agent(
            name="bash_agent",
            model="openai/gpt-4o-mini",
            tools=[bash_tool]
        )
        
        # Agent starts a background task
        result = agent(
            prompt="Start a background sleep process for 4 seconds"
        )
        _ = await result.collect()
        assert bool(agent._bg_tasks)

        # Confirm agent has the required tools
        assert "sleep_tool" in agent._tools_by_name, "Agent should have sleep_tool"
        assert "get_process_output" in agent._tools_by_name, "Agent should have get_process_output tool"
        assert "terminate_process" in agent._tools_by_name, "Agent should have terminate_process tool"

    
    @pytest.mark.asyncio
    async def test_background_execution_with_agent(self):
        """Test termination of background process with agent."""
        # Create an agent with bash tool and termination tools
        bash_tool = Bash("ls")
        agent = Agent(
            name="bash_agent",
            model="openai/gpt-4o-mini",
            tools=[bash_tool]
        )
        
        # Agent starts a background task
        result = agent(
            prompt="List all files on a background process"
        )
        _ = await result.collect()
        
        assert bool(agent._bg_tasks)
        
        # Agent checks if process is done and terminates it
        result2 = agent(
            prompt=f"Get the output of the process."
        )
        _ = await result2.collect()

        print(agent._tools_by_name)

        # TODO: check
        assert "get_process_output" in agent._tools_by_name, "Agent should have get_process_output tool"
        assert "terminate_process" in agent._tools_by_name, "Agent should have terminate_process tool"

        # process must be removed from _bg_tasks
        assert not bool(agent._bg_tasks)

    @pytest.mark.asyncio
    async def test_output_process_not_finished(self):
        """Test termination of background process with agent and process output."""
        bash_tool = Bash("sleep")
        agent = Agent(
            name="bash_agent",
            model="openai/gpt-4o-mini",
            tools=[bash_tool]
        )
        
        # Agent starts a background task
        result = agent(
            prompt="Start a background sleep process for 10 seconds"
        )
        _ = await result.collect()

        result2 = agent(
            prompt=f"Get the output of the process."
        )
        output2 = await result2.collect()
        
        # Check that the process is still running (since it's a 4-second sleep)
        response_content = output2.output.content[0].text
        assert "running" in response_content.lower()
        
        

class TestBashToolIntegration:
    """Test Bash tool integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_multiple_bash_tools(self):
        """Test using multiple Bash tools."""
        echo_tool = Bash("echo")
        ls_tool = Bash("ls")
        
        result1 = echo_tool(args="hello")
        result2 = ls_tool(args="-la")
        
        output1 = await result1.collect()
        output2 = await result2.collect()
        
        assert output1.error is None
        assert output2.error is None
        assert "hello" in output1.output['output']
        assert isinstance(output2.output['output'], str)
    
    