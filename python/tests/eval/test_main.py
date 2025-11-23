"""
Tests for the main eval_v2 functionality - the primary user interface.

These tests simulate how users actually interact with the eval system:
loading agents from Python files and running evaluations against them.
"""
import pytest
from timbal import Agent
from timbal.eval.__main__ import run_evals
from timbal.eval.engine import eval_file
from timbal.eval.types.result import EvalTestSuiteResult
from timbal.utils import ImportSpec


class TestMainWorkflow:
    """Test the main eval workflow that users actually use."""

    @pytest.mark.asyncio
    async def test_load_agent_from_file(self, sample_agent):
        """Test loading an agent from a Python file (--fqn functionality)."""
        assert isinstance(sample_agent, Agent)
        assert sample_agent.name == "sample_agent"
        assert len(sample_agent.tools) == 4  # get_current_time, add_numbers, greet_person, calculate_expression


    @pytest.mark.asyncio
    async def test_basic_command_line_usage(self, sample_agent, simple_test_file):
        """Simulate: python -m timbal.eval_v2 --fqn sample_agent.py::agent --tests eval_simple_test.yaml"""
        files = [simple_test_file]
        
        test_results = EvalTestSuiteResult()
        dumped_results = await run_evals(files, sample_agent, test_results)
        
        assert isinstance(dumped_results, dict)
        assert test_results.total_tests == 1



        
    @pytest.mark.asyncio
    async def test_agent_with_multiple_tools(self, sample_agent):
        """Test that our sample agent works with its various tools."""
        # Verify agent has expected tools
        tool_names = [tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in sample_agent.tools]
        expected_tools = ['get_current_time', 'add_numbers', 'greet_person', 'calculate_expression']
        
        for expected_tool in expected_tools:
            assert any(expected_tool in tool_name for tool_name in tool_names), f"Tool {expected_tool} not found"


    @pytest.mark.asyncio
    async def test_counting_files_and_tests(self, sample_agent, agent_test_file, simple_test_file):
        """Test that files and tests are counted correctly."""
        # Test single file with multiple tests
        files = [agent_test_file]
        
        test_results = EvalTestSuiteResult()
        await run_evals(files, sample_agent, test_results)
        
        assert test_results.total_files == 1  # 1 file processed
        assert test_results.total_tests == 5  # 5 tests in eval_agent.yaml
        
        # Test multiple files
        files = [agent_test_file, simple_test_file]
        
        test_results = EvalTestSuiteResult()
        await run_evals(files, sample_agent, test_results)
        
        assert test_results.total_files == 2  # 2 files processed
        assert test_results.total_tests == 6  # 5 from eval_agent.yaml + 1 from eval_simple_test.yaml


class TestMainErrorHandling:
    """Test error handling in main functionality."""

    @pytest.mark.asyncio
    async def test_nonexistent_agent_file(self, fixtures_dir):
        """Test error when agent file doesn't exist."""
        nonexistent_file = fixtures_dir / "nonexistent_agent.py"
        import_spec = ImportSpec(
            path=nonexistent_file,
            target="agent"
        )
        
        with pytest.raises(Exception):  # Should raise import/file error
            import_spec.load()

    @pytest.mark.asyncio
    async def test_invalid_agent_object(self, fixtures_dir):
        """Test error when specified agent object doesn't exist in file."""
        agent_file = fixtures_dir / "sample_agent.py"
        import_spec = ImportSpec(
            path=agent_file,
            target="nonexistent_agent"  # This object doesn't exist
        )
        
        with pytest.raises(Exception):  # Should raise attribute error
            import_spec.load()

    @pytest.mark.asyncio
    async def test_agent_without_object_name(self, fixtures_dir):
        """Test loading agent file without specifying object name."""
        # This simulates --fqn sample_agent.py (without ::agent)
        agent_file = fixtures_dir / "sample_agent.py"
        import_spec = ImportSpec(
            path=agent_file,
            target=None  # No specific object specified
        )
        
        # This should fail because load_module doesn't support loading entire modules
        with pytest.raises(NotImplementedError, match="Does not support loading entire module"):
            import_spec.load()

    @pytest.mark.asyncio
    async def test_nonexistent_test_file(self, sample_agent, fixtures_dir):
        """Test error when test file doesn't exist."""
        # Try to run against nonexistent test file
        nonexistent_test = fixtures_dir / "eval_nonexistent_test.yaml"
        test_results = EvalTestSuiteResult()
        
        with pytest.raises(FileNotFoundError):
            await eval_file(nonexistent_test, sample_agent, test_results)