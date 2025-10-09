import tempfile

import pytest
from timbal import Agent
from timbal.tools.bash import Bash


class TestBashToolInitialization:
    """Test Bash tool initialization and basic functionality."""

    def test_bash_tool_creation(self):
        """Test that Bash tool is created correctly."""
        pattern = "echo *"
        bash_tool = Bash(pattern)

        assert bash_tool.name == "bash"
        assert bash_tool.allowed_patterns == [pattern]
        assert bash_tool.handler is not None
    
    def test_bash_tool_handler_signature(self):
        """Test that the handler has the correct signature."""
        bash_tool = Bash("echo *")

        # Check that handler accepts the expected parameters
        import inspect
        sig = inspect.signature(bash_tool.handler)
        params = list(sig.parameters.keys())

        expected_params = ['command']
        for param in expected_params:
            assert param in params

    def test_bash_tool_multiple_patterns(self):
        """Test that Bash tool accepts multiple patterns."""
        patterns = ["echo *", "ls *", "pwd"]
        bash_tool = Bash(patterns)

        assert bash_tool.allowed_patterns == patterns

    def test_bash_tool_invalid_pattern(self):
        """Test that empty patterns raise errors."""
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            Bash("   ")


class TestBashCommandExecution:
    """Test Bash command execution functionality."""
    
    @pytest.mark.asyncio
    async def test_simple_command_execution(self):
        """Test executing a simple command."""
        bash_tool = Bash("echo *")
        result = bash_tool(command="echo hello world")

        output = await result.collect()
        assert output.error is None
        assert "hello world" in output.output['stdout']
    
    @pytest.mark.asyncio
    async def test_command_with_no_args(self):
        """Test executing basic echo command."""
        bash_tool = Bash("echo")
        result = bash_tool(command="echo")

        output = await result.collect()
        assert output.error is None
        # Should execute just "echo" which typically outputs empty line
        assert isinstance(output.output['stdout'], str)
    
    @pytest.mark.asyncio
    async def test_command_pwd(self):
        """Test executing pwd command."""
        bash_tool = Bash("pwd")
        result = bash_tool(command="pwd")

        output = await result.collect()
        assert output.error is None
        assert isinstance(output.output['stdout'], str)
        assert len(output.output['stdout'].strip()) > 0
    
    @pytest.mark.asyncio
    async def test_command_pattern_validation(self):
        """Test that commands are validated against patterns."""
        bash_tool = Bash("echo *")

        # This should work
        result = bash_tool(command="echo test")
        output = await result.collect()
        assert output.error is None

        # This should fail validation
        result_fail = bash_tool(command="ls -la")
        output_fail = await result_fail.collect()
        assert output_fail.error is not None
        assert output_fail.error["type"] == "ValueError"
    
    @pytest.mark.asyncio
    async def test_command_that_fails(self):
        """Test executing a command that fails."""
        bash_tool = Bash("nonexistentcommand*")
        result = bash_tool(command="nonexistentcommand12345")

        output = await result.collect()
        assert output.error is None
        # Should have returncode != 0 for nonexistent command
        assert output.output['returncode'] != 0
        assert output.output['stderr'] != ''

    @pytest.mark.asyncio
    async def test_command_with_shell_expansion(self):
        """Test executing command with shell expansion."""
        bash_tool = Bash("ls *")
        result = bash_tool(command="ls *.py")

        output = await result.collect()
        assert output.error is None
        assert isinstance(output.output['stdout'], str)

    @pytest.mark.asyncio
    async def test_tool_with_agent_not_matching_pattern(self):
        """Test agent with Bash tool when the command doesn't match allowed patterns."""
        agent = Agent(
            name="code_analyzer",
            model="openai/gpt-4o-mini",
            system_prompt="""You are a helpful assistant.
            You can only run mkdir commands using the bash tool.
            You cannot run any other commands.
            """,
            tools=[Bash("mkdir *")]
        )

        response = await agent(
            prompt="List all Python files in the current directory"
        ).collect()

        # The agent should recognize that it can only run mkdir commands
        assert response.error is None
        assert response.output is not None
        response_text = response.output.content[0].text.lower()

        # Should indicate that it can't list files or mention mkdir limitation
        assert any(keyword in response_text for keyword in [
            "mkdir", "cannot", "unable", "only", "directory", "create", "listing"
        ])

    @pytest.mark.asyncio
    async def test_tool_functionality_with_list(self):
        """Test Bash tool with multiple allowed patterns."""
        tool = Bash(["cd *", "ls *"])
        # First run cd command
        result1 = await tool(command="cd python/tests/tools").collect()
        assert result1.error is None

        # Then run ls command
        result2 = await tool(command="ls *.py").collect()
        assert result2.error is None
        assert isinstance(result2.output['stdout'], str)

    @pytest.mark.asyncio
    async def test_tool_functionality_with_chaining(self):
        """Test Bash tool with command chaining using &&."""
        tool = Bash(["cd *", "ls *"])  # Allow individual commands
        # Change to the test directory where test_bash.py exists and list files
        result = await tool(command="cd python/tests/tools && ls *.py").collect()

        assert result.error is None
        assert "test_bash.py" in result.output['stdout']

    @pytest.mark.asyncio
    async def test_command_piping_with_command_array(self):
        """Test piping with array of allowed commands."""
        tool = Bash(["echo *", "grep *", "wc *"])
        result = await tool(command="echo hello world | grep hello").collect()

        assert result.error is None
        assert "hello" in result.output['stdout']
        assert result.output['returncode'] == 0

    @pytest.mark.asyncio
    async def test_command_piping_multiple_with_array(self):
        """Test multiple pipes with array of allowed commands."""
        tool = Bash(["echo *", "grep *", "wc *"])
        result = await tool(command="echo hello world test | grep hello | wc -w").collect()

        assert result.error is None
        # grep hello will match the line "hello world test", wc -w counts words in that line (3)
        assert "3" in result.output['stdout'].strip()  # Should count 3 words
        assert result.output['returncode'] == 0

    @pytest.mark.asyncio
    async def test_command_chaining_with_array(self):
        """Test command chaining (&&) with array of allowed commands."""
        tool = Bash(["cd *", "ls *", "echo *"])
        result = await tool(command="cd python/tests/tools && ls *.py && echo done").collect()

        assert result.error is None
        assert "test_bash.py" in result.output['stdout']
        assert "done" in result.output['stdout']
        assert result.output['returncode'] == 0

    @pytest.mark.asyncio
    async def test_mixed_piping_and_chaining_with_array(self):
        """Test mixing pipes and chaining with array of commands."""
        tool = Bash(["echo *", "grep *", "wc *"])
        result = await tool(command="echo hello world | grep hello && echo success").collect()

        assert result.error is None
        assert "hello" in result.output['stdout']
        assert "success" in result.output['stdout']
        assert result.output['returncode'] == 0

    @pytest.mark.asyncio
    async def test_command_array_validation_failure(self):
        """Test that unlisted commands in array are rejected."""
        tool = Bash(["echo *", "grep *"])  # Only allow echo and grep
        result = await tool(command="echo hello | wc -w").collect()

        # Should fail because 'wc' is not in allowed commands
        assert result.error is not None
        assert result.error["type"] == "ValueError"


class TestBashAdvancedOperators:
    """Test advanced shell operators and edge cases."""

    @pytest.mark.asyncio
    async def test_or_operator(self):
        """Test OR operator (||)."""
        tool = Bash(["* || echo *"])  # Pattern that matches the full OR command
        result = await tool(command="nonexistent_command || echo fallback").collect()

        assert result.error is None
        assert "fallback" in result.output['stdout']
        assert result.output['returncode'] == 0

    @pytest.mark.asyncio
    async def test_sequential_operator(self):
        """Test sequential operator (;)."""
        tool = Bash(["echo *", "pwd"])
        result = await tool(command="echo first; pwd").collect()

        assert result.error is None
        assert "first" in result.output['stdout']
        assert result.output['returncode'] == 0

    @pytest.mark.asyncio
    async def test_mixed_operators(self):
        """Test mixing different operators."""
        tool = Bash(["echo *", "pwd", "ls *"])
        result = await tool(command="echo start && pwd || ls -la").collect()

        assert result.error is None
        assert result.output['returncode'] == 0

    @pytest.mark.asyncio
    async def test_operator_validation_failure(self):
        """Test that unauthorized commands in operator chains are rejected."""
        tool = Bash(["echo *"])  # Only allow echo
        result = await tool(command="echo hello || rm -rf /").collect()

        # Should fail because 'rm' is not allowed
        assert result.error is not None
        assert result.error["type"] == "ValueError"


class TestBashEdgeCases:
    """Test edge cases and whitespace handling."""

    @pytest.mark.asyncio
    async def test_extra_whitespace(self):
        """Test commands with extra whitespace."""
        tool = Bash(["echo *", "grep *"])
        result = await tool(command="echo hello world   |   grep hello").collect()

        assert result.error is None
        assert "hello" in result.output['stdout']

    @pytest.mark.asyncio
    async def test_multiple_sequential_operators(self):
        """Test multiple sequential operators."""
        tool = Bash(["echo *"])
        result = await tool(command="echo one; echo two; echo three").collect()

        assert result.error is None
        assert "one" in result.output['stdout']
        assert "two" in result.output['stdout']
        assert "three" in result.output['stdout']

    @pytest.mark.asyncio
    async def test_empty_command_parts(self):
        """Test handling of malformed commands with empty parts."""
        tool = Bash(["echo *", "ls *"])

        # This should be handled gracefully
        result = await tool(command="echo hello &&").collect()

        # The command should fail but not crash
        assert result.error is not None or result.output['returncode'] != 0

    @pytest.mark.asyncio
    async def test_long_command_chain(self):
        """Test very long command chains."""
        tool = Bash(["echo *"])
        long_chain = " && ".join([f"echo step{i}" for i in range(10)])
        result = await tool(command=long_chain).collect()

        assert result.error is None
        assert "step0" in result.output['stdout']
        assert "step9" in result.output['stdout']


class TestBashSecurity:
    """Test security-related scenarios."""

    @pytest.mark.asyncio
    async def test_command_injection_prevention(self):
        """Test that command injection attempts are blocked."""
        tool = Bash(["echo *"])  # Only allow echo

        # These should all fail
        injection_attempts = [
            "echo hello; rm -rf /",
            "echo hello && cat /etc/passwd",
            "echo $(whoami)",
            "echo hello | sh"
        ]

        for attempt in injection_attempts:
            result = await tool(command=attempt).collect()
            # Should fail because unauthorized commands are used
            if "rm" in attempt or "cat" in attempt or "sh" in attempt:
                assert result.error is not None
                assert result.error["type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_redirection_with_patterns(self):
        """Test commands with redirection operators."""
        tool = Bash(["echo * > *", "cat *"])

        # This should work if pattern allows redirection
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            result = await tool(command=f"echo test > {tmp.name}").collect()
            assert result.error is None

            # Clean up
            import os
            os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_pattern_with_special_chars(self):
        """Test patterns containing special characters."""
        # Test with literal patterns that should work
        tool = Bash(["ls *.py", "echo *"])

        result1 = await tool(command="ls test.py").collect()
        # This may fail if file doesn't exist, but should not have validation error
        assert result1.error is None or result1.output['returncode'] != 0

        result2 = await tool(command="echo hello-world").collect()
        assert result2.error is None

    @pytest.mark.asyncio
    async def test_wildcard_accepts_everything(self):
        """Test that '*' pattern accepts any command."""
        tool = Bash("*")  # Allow everything

        test_commands = [
            "echo hello",
            "ls -la /tmp",
            "pwd && echo done",
            "cat file.txt | grep pattern",
            "complex command with many args --flag=value",
            "rm -rf dangerous_command",
            "echo 'quoted string with | pipes'",
            "command; another; third"
        ]

        for cmd in test_commands:
            result = await tool(command=cmd).collect()
            # Should not have validation errors (though commands may fail to execute)
            assert result.error is None or result.output is not None
            # If there's an error, it should be execution error, not validation error
            if result.error:
                assert "does not match any allowed patterns" not in result.error.get("message", "")
