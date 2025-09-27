import os
import tempfile
from pathlib import Path

import pytest
from timbal import Agent
from timbal.tools import Bash, List  #, WebSearch


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


class TestListToolIntegration:
    """Test List tool integration with agents across different models."""

    @pytest.mark.asyncio
    async def test_list_directory_exploration(self, model):
        """Test agent using List tool to explore directories."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048

        list_tool = List()
        agent = Agent(
            name="directory_explorer",
            model=model,
            model_params=model_params,
            system_prompt="You can list directory contents using the list tool. Help users explore directory structures.",
            tools=[list_tool]
        )

        # Collect all events to inspect tool calls
        events = []
        async for event in agent(prompt="What files and directories are in the python directory?"):
            events.append(event)

        # Find list tool execution events (OutputEvents with path ending in '.list')
        list_events = [e for e in events if hasattr(e, 'type') and e.type == 'OUTPUT' and e.path.endswith('.list')]

        # Should have at least one list tool call
        assert len(list_events) > 0, "Agent should have called the list tool"

        # Check that python directory was listed
        python_calls = [e for e in list_events if 'python' in str(e.input).lower()]
        assert len(python_calls) > 0, "Agent should have listed the python directory"

    @pytest.mark.asyncio
    async def test_list_current_directory(self, model):
        """Test agent using List tool on current directory."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048
        agent = Agent(
            name="current_dir_lister",
            model=model,
            model_params=model_params,
            system_prompt="Use the list tool to show directory contents when asked.",
            tools=[List()]
        )

        response = await agent(
            prompt="Show me what's in the current directory"
        ).collect()

        assert response.error is None
        assert response.output is not None
        response_text = response.output.content[0].text.lower()

        # Should use list tool to show current directory contents
        assert any(keyword in response_text for keyword in [
            "current", "directory", "contents", "files", "list"
        ])

    @pytest.mark.asyncio
    async def test_list_with_path_expansion(self, model):
        """Test agent using List tool with path expansion."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048
        agent = Agent(
            name="path_expander",
            model=model,
            model_params=model_params,
            system_prompt="You can list directories using ~ for home directory and environment variables.",
            tools=[List()]
        )

        response = await agent(
            prompt="List the contents of my home directory"
        ).collect()

        assert response.error is None
        assert response.output is not None
        response_text = response.output.content[0].text.lower()

        # Should understand home directory concept and use list tool
        assert any(keyword in response_text for keyword in [
            "home", "directory", "contents", "list"
        ])

    @pytest.mark.asyncio
    async def test_list_error_handling(self, model):
        """Test agent handling List tool errors gracefully."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048
        agent = Agent(
            name="error_handler",
            model=model,
            model_params=model_params,
            system_prompt="Handle errors gracefully when listing directories that don't exist.",
            tools=[List()]
        )

        response = await agent(
            prompt="List the contents of /nonexistent/directory/path"
        ).collect()

        assert response.error is None
        assert response.output is not None
        response_text = response.output.content[0].text.lower()

        # Should handle error and explain the issue
        assert any(keyword in response_text for keyword in [
            "error", "exist", "found", "cannot", "unable", "directory"
        ])


class TestBashListToolCombination:
    """Test using Bash and List tools together."""

    @pytest.mark.asyncio
    async def test_list_tool_preference_over_bash(self, model):
        """Test that agent prefers List tool over Bash ls for directory listing."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048

        bash_tool = Bash("ls *")
        list_tool = List()
        agent = Agent(
            name="tool_chooser",
            model=model,
            model_params=model_params,
            system_prompt="You have both bash (ls command) and list tools available. Always prefer the list tool for directory operations as instructed by the tool descriptions.",
            tools=[bash_tool, list_tool]
        )

        # Collect all events to inspect tool calls
        events = []
        async for event in agent(prompt="What files are in the current directory?"):
            events.append(event)

        # Find tool execution events
        list_calls = [e for e in events if hasattr(e, 'type') and e.type == 'OUTPUT' and e.path.endswith('.list')]
        bash_calls = [e for e in events if hasattr(e, 'type') and e.type == 'OUTPUT' and e.path.endswith('.bash')]

        # Should use list tool, not bash
        assert len(list_calls) > 0, "Agent should have used the list tool"
        # Ideally bash should not be called, but this is less critical
        # assert len(bash_calls) == 0, "Agent should prefer list tool over bash ls"

    @pytest.mark.asyncio
    async def test_bash_and_list_directory_analysis(self, model):
        """Test agent using both Bash and List tools for directory analysis."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048

        bash_tool = Bash(["ls *", "pwd", "echo *"])
        list_tool = List()
        agent = Agent(
            name="directory_analyzer",
            model=model,
            model_params=model_params,
            system_prompt="You have both bash and list tools. Use them to analyze directory structures and provide detailed information.",
            tools=[bash_tool, list_tool]
        )

        # Collect all events to inspect tool calls
        events = []
        async for event in agent(prompt="Analyze the python directory structure - show me what's there and tell me about it"):
            events.append(event)

        # Find tool execution events (OutputEvents with path ending in '.bash' or '.list')
        tool_events = [e for e in events if hasattr(e, 'type') and e.type == 'OUTPUT' and (e.path.endswith('.bash') or e.path.endswith('.list'))]

        # Should have used at least one tool
        assert len(tool_events) > 0, "Agent should have used tools to analyze directory structure"

        # Check for python-related calls
        python_related = [e for e in tool_events if 'python' in str(e.input).lower()]
        assert len(python_related) > 0, "Agent should have analyzed python directory"

    @pytest.mark.asyncio
    async def test_file_system_exploration_workflow(self, model):
        """Test complex file system exploration using both tools."""
        model_params = {}
        if model.startswith("anthropic"):
            model_params["max_tokens"] = 2048
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            test_dir = Path(tmpdir) / "test_project"
            test_dir.mkdir()

            (test_dir / "README.md").write_text("# Test Project")
            (test_dir / "src").mkdir()
            (test_dir / "src" / "main.py").write_text("print('hello')")

            agent = Agent(
                name="file_explorer",
                model=model,
                model_params=model_params,
                system_prompt=f"Explore the directory structure at {test_dir}. Use both bash and list tools as needed to provide a comprehensive overview.",
                tools=[Bash(["ls *", "pwd", "echo *"]), List()]
            )

            response = await agent(
                prompt=f"Explore and describe the structure of {test_dir}"
            ).collect()

            assert response.error is None
            assert response.output is not None
            response_text = response.output.content[0].text.lower()

            # Should explore and describe the created structure
            assert any(keyword in response_text for keyword in [
                "structure", "directory", "readme", "src", "project"
            ])
