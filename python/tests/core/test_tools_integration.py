import os

import pytest
from timbal import Agent
from timbal.tools import Bash  #, WebSearch


@pytest.fixture(params=[
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-mini-responses",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4-0",
    # ? Add more tests for other models.
])
def model(request):
    if request.param.startswith("openai"):
        if request.param.endswith("-responses"):
            # Responses API is now the default, so remove any disable flag
            os.environ.pop("TIMBAL_DISABLE_OPENAI_RESPONSES_API", None)
            return request.param.replace("-responses", "")
        else:
            # Disable responses API for non-responses tests
            os.environ["TIMBAL_DISABLE_OPENAI_RESPONSES_API"] = "true"
    return request.param


# class TestReadToolIntegration:
#     """Test Read tool integration with agents across different models."""

#     @pytest.mark.asyncio
#     async def test_read_file(self, model):
#         """Test agent using Read tool to read a file."""
#         model_params = {}
#         if model.startswith("anthropic"):
#             model_params["max_tokens"] = 2048
#         agent = Agent(
#             name="file_reader",
#             model=model,
#             model_params=model_params,
#             tools=[Read()]
#         )

#         response = await agent(
#             prompt="The image in ./python/tests/core/fixtures/test.png contains tabular data. You MUST use the read tool to open the file"
#         ).collect()



class TestBashToolIntegration:
    """Test Bash tool integration with agents across different models."""

    @pytest.mark.asyncio
    async def test_bash_simple_command_execution(self, model):
        """Test agent using Bash tool for simple command execution."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048

        bash_tool = Bash("echo *")
        agent = Agent(
            name="bash_tester",
            model=model,
            model_params=model_params,
            system_prompt="You are a helpful assistant that can execute bash commands. Use the bash tool to run commands as requested.",
            tools=[bash_tool]
        )

        # Collect all events to inspect tool calls
        events = []
        async for event in agent(prompt="Use the bash tool to echo 'Hello World'"):
            events.append(event)

        # Find bash tool execution events (OutputEvents with path ending in '.bash')
        bash_events = [e for e in events if hasattr(e, 'type') and e.type == 'OUTPUT' and e.path.endswith('.bash')]

        # Should have at least one bash tool call
        assert len(bash_events) > 0, "Agent should have called the bash tool"

        # Check that echo command was executed
        echo_calls = [e for e in bash_events if 'echo' in str(e.input).lower()]
        assert len(echo_calls) > 0, "Agent should have used echo command"

    @pytest.mark.asyncio
    async def test_bash_directory_operations(self, model):
        """Test agent using Bash tool for directory operations."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048
        agent = Agent(
            name="directory_agent",
            model=model,
            model_params=model_params,
            system_prompt="You can list directories and check current working directory using bash commands.",
            tools=[Bash(["ls *", "pwd"])]
        )

        response = await agent(
            prompt="What is the current directory and what files are in the python directory?"
        ).collect()

        assert response.error is None
        assert response.output is not None
        response_text = response.output.content[0].text.lower()

        # Should use the bash tools to get directory information
        assert any(keyword in response_text for keyword in [
            "directory", "pwd", "ls", "python", "current"
        ])

    @pytest.mark.asyncio
    async def test_bash_pattern_limitation_handling(self, model):
        """Test that agent handles bash tool pattern limitations gracefully."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048
        agent = Agent(
            name="limited_bash_agent",
            model=model,
            model_params=model_params,
            system_prompt="You can only run echo commands. You cannot run any other bash commands.",
            tools=[Bash("echo hello")]
        )

        response = await agent(
            prompt="List all files in the current directory"
        ).collect()

        assert response.error is None
        assert response.output is not None
        response_text = response.output.content[0].text.lower()

        # Agent should recognize limitation and explain it cannot list files
        assert any(keyword in response_text for keyword in [
            "cannot", "unable", "only", "echo", "limited", "restricted"
        ])

    @pytest.mark.asyncio
    async def test_bash_command_chaining(self, model):
        """Test agent using Bash tool with command chaining."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048
        agent = Agent(
            name="chaining_agent",
            model=model,
            model_params=model_params,
            system_prompt="You can run multiple bash commands using && to chain them together.",
            tools=[Bash(["echo *", "pwd"])]
        )

        response = await agent(
            prompt="Echo 'Starting' then show the current directory"
        ).collect()

        assert response.error is None
        assert response.output is not None
        response_text = response.output.content[0].text.lower()

        # Should attempt to use command chaining or explain the process
        assert any(keyword in response_text for keyword in [
            "starting", "directory", "echo", "pwd"
        ])
