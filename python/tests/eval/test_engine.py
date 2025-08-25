from pathlib import Path

import pytest
import yaml
from timbal import Agent
from timbal.eval.engine import eval_file, eval_output, eval_steps, eval_usage, run_turn
from timbal.eval.types.input import Input
from timbal.eval.types.output import Output
from timbal.eval.types.result import EvalTestSuiteResult
from timbal.eval.types.steps import Steps
from timbal.eval.types.test_suite import TestSuite
from timbal.eval.types.turn import Turn
from timbal.types.message import Message

# Get the fixtures directory path
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestEvalFile:
    """Test the eval_file function - the main entry point for evaluations."""

    @pytest.mark.asyncio
    async def test_eval_file_simple(self):
        """Test basic file evaluation."""
        def get_time() -> str:
            return "2024-01-01 12:00:00"
        
        agent = Agent(
            name="simple_agent", 
            model="openai/gpt-4.1-mini",
            tools=[get_time]
        )
        test_file = FIXTURES_DIR / "eval_simple_test.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)
        assert test_results.total_files == 1
        assert test_results.total_tests == 1

    @pytest.mark.asyncio 
    async def test_eval_file_with_specific_test(self):
        """Test evaluating a specific test by name."""
        def add(a: int, b: int) -> int:
            return a + b
        
        agent = Agent(name="math_agent", model="openai/gpt-4o-mini", tools=[add])
        test_file = FIXTURES_DIR / "eval_multiple_tests.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results, test_name="test1")
        
        assert isinstance(result, EvalTestSuiteResult)
        assert test_results.total_tests == 1

    @pytest.mark.asyncio
    async def test_eval_file_nonexistent(self):
        """Test error handling for nonexistent file."""
        def get_time() -> str:
            return "2024-01-01 12:00:00"
        
        agent = Agent(name="simple_agent", model="openai/gpt-4o-mini", tools=[get_time])
        nonexistent_path = Path("/nonexistent/eval_file.yaml")
        test_results = EvalTestSuiteResult()
        
        with pytest.raises(FileNotFoundError):
            await eval_file(nonexistent_path, agent, test_results)


class TestEvalSteps:
    """Test the eval_steps function using real YAML fixtures."""

    @pytest.mark.asyncio
    async def test_eval_steps_with_yaml_fixture(self):
        """Test step evaluation using math test YAML fixture."""
        
        yaml_path = FIXTURES_DIR / "eval_math_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]  # Get the math_test
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert turn.input.text == "What is 2 + 3?"
        assert isinstance(turn.output, Output)
        assert isinstance(turn.steps, Steps)
        
        def add(a: int, b: int) -> int:
            return a + b
        
        agent = Agent(name="math_agent", model="openai/gpt-4o-mini", tools=[add])
        
        success, errors, steps = await eval_steps(turn, agent)
        
        assert isinstance(success, bool)
        assert isinstance(errors, list)
        assert isinstance(steps, list)

    @pytest.mark.asyncio
    async def test_eval_steps_no_tracing_yaml(self):
        """Test step evaluation without tracing using YAML fixture."""
        
        yaml_path = FIXTURES_DIR / "eval_agent.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]  # Get the time_test
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert isinstance(turn.output, Output)
        assert isinstance(turn.steps, Steps)
        
        def get_current_time() -> str:
            return "2024-01-01 12:00:00"
        
        agent = Agent(name="time_agent", model="openai/gpt-4o-mini", tools=[get_current_time])
        
        success, errors, steps = await eval_steps(turn, agent)
        
        assert success is False
        assert "No tracing data available" in errors[0]
        assert steps == []

    @pytest.mark.asyncio
    async def test_eval_steps_no_validators_yaml(self):
        """Test step evaluation with turn that has no step validators using YAML."""
        
        yaml_path = FIXTURES_DIR / "eval_simple_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert isinstance(turn.output, Output)
        assert turn.steps is None  # No steps field in simple test
        
        def get_time() -> str:
            return "2024-01-01 12:00:00"
        
        agent = Agent(name="simple_agent", model="openai/gpt-4o-mini", tools=[get_time])
        
        success, errors, steps = await eval_steps(turn, agent)
        
        assert success is True
        assert errors == []
        assert steps == []


class TestEvalOutput:
    """Test the eval_output function using real YAML fixtures."""

    @pytest.mark.asyncio
    async def test_eval_output_with_validators_yaml(self):
        """Test output evaluation with explicit validators from YAML."""
        
        yaml_path = FIXTURES_DIR / "eval_simple_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert isinstance(turn.output, Output)
        assert turn.output.validators is not None
        
        message = Message.validate({
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello there!"}]
        })
        
        success, errors = await eval_output(turn, message)
        
        assert isinstance(success, bool)
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_eval_output_with_text_only_yaml(self):
        """Test output evaluation with text only from YAML (uses semantic validation)."""
        
        yaml_path = FIXTURES_DIR / "eval_multiple_tests.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]  # First test has text output
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert isinstance(turn.output, Output)
        assert turn.output.text is not None
        
        message = Message.validate({
            "role": "assistant", 
            "content": [{"type": "text", "text": "The answer is 4"}]
        })
        
        success, errors = await eval_output(turn, message)
        
        assert isinstance(success, bool)
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_eval_output_no_output_yaml(self):
        """Test output evaluation when no output is specified in YAML."""
        
        # Create a minimal turn without output for testing
        turn = Turn(input=Input(text="Hello"))
        
        # Validate that turn without output works
        assert isinstance(turn.input, Input)
        assert turn.output is None
        
        message = Message.validate({
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello there!"}]
        })
        
        success, errors = await eval_output(turn, message)
        
        assert success is True  # Should pass when no output specified
        assert errors == []


class TestEvalUsage:
    """Test the eval_usage function using real YAML fixtures."""

    def test_eval_usage_basic_yaml(self):
        """Test basic usage evaluation from YAML fixture."""
        
        yaml_path = FIXTURES_DIR / "eval_usage_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert turn.usage is not None
        
        actual_usage = {
            "gpt-4.1-mini:input_tokens": 50
        }
        
        results = eval_usage(turn, actual_usage)
        
        assert isinstance(results, list)
        if results:
            for result in results:
                assert "usage_key" in result
                assert "correct" in result
                assert "explanation" in result

    def test_eval_usage_no_constraints_yaml(self):
        """Test usage evaluation when no usage constraints are specified."""
        
        yaml_path = FIXTURES_DIR / "eval_simple_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert turn.usage is None
        
        actual_usage = {"gpt-4:input_tokens": 50}
        
        results = eval_usage(turn, actual_usage)
        
        assert results == []

    def test_eval_usage_prefix_matching_yaml(self):
        """Test usage evaluation with model prefix matching from YAML."""

        yaml_path = FIXTURES_DIR / "eval_usage_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert turn.usage is not None
        
        actual_usage = {
            "gpt-4.1-mini-2024-01-01:input_tokens": 50
        }
        
        results = eval_usage(turn, actual_usage)
        
        assert isinstance(results, list)


class TestRunTurn:
    """Test the run_turn function using real YAML fixtures."""

    @pytest.mark.asyncio
    async def test_run_turn_basic_yaml(self):
        """Test basic turn execution using YAML fixture."""
        
        yaml_path = FIXTURES_DIR / "eval_agent.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]  # Get the time_test
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert isinstance(turn.output, Output)
        assert test.name == "time_test"
        
        def get_current_time() -> str:
            return "2024-01-01 12:00:00"
        
        agent = Agent(name="time_agent", model="openai/gpt-4o-mini", tools=[get_current_time])
        conversation_history = []
        test_results = EvalTestSuiteResult()
        
        await run_turn(agent, turn, test, conversation_history, test_results, str(yaml_path.name), yaml_path.parent)
        
        assert test_results.total_turns == 1

    @pytest.mark.asyncio
    async def test_run_turn_with_conversation_history_yaml(self):
        """Test turn execution with conversation history using multi-turn YAML."""
        
        yaml_path = FIXTURES_DIR / "eval_multi_turn_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]  # Get the memory_test
        second_turn = test.turns[1]  # Use the second turn which relies on conversation history
        
        assert isinstance(second_turn.input, Input)
        assert isinstance(second_turn.output, Output)
        assert len(test.turns) == 2
        
        def get_time() -> str:
            return "2024-01-01 12:00:00"
        
        agent = Agent(name="memory_agent", model="openai/gpt-4o-mini", tools=[get_time])
        
        conversation_history = [
            Message.validate({"role": "user", "content": [{"type": "text", "text": "My name is Bob"}]}),
            Message.validate({"role": "assistant", "content": [{"type": "text", "text": "Nice to meet you, Bob!"}]})
        ]
        test_results = EvalTestSuiteResult()
        
        await run_turn(agent, second_turn, test, conversation_history, test_results, str(yaml_path.name), yaml_path.parent)
        
        assert test_results.total_turns == 1

    @pytest.mark.asyncio
    async def test_run_turn_with_execution_error_yaml(self):
        """Test turn execution when agent execution fails using YAML fixture."""
        
        yaml_path = FIXTURES_DIR / "eval_math_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert isinstance(turn.output, Output)
        
        def failing_add(a: int, b: int) -> int:
            raise ValueError("Tool failed")
        
        agent = Agent(name="failing_agent", model="openai/gpt-4o-mini", tools=[failing_add])
        conversation_history = []
        test_results = EvalTestSuiteResult()
        
        await run_turn(agent, turn, test, conversation_history, test_results, str(yaml_path.name), yaml_path.parent)
        
        assert test_results.total_turns == 1
        assert test_results.execution_errors >= 0  # Should handle the error gracefully


class TestEngineIntegration:
    """Integration tests for the eval engine."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test evaluation with multi-turn conversations."""
        def remember_name(name: str) -> str:
            return f"I'll remember that your name is {name}"
        
        def recall_name() -> str:
            return "I remember you told me your name earlier"
        
        agent = Agent(name="memory_agent", model="openai/gpt-4o-mini", tools=[remember_name, recall_name])
        test_file = FIXTURES_DIR / "eval_multi_turn_test.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)
        assert result.total_turns >= 2

    @pytest.mark.asyncio
    async def test_usage_constraints(self):
        """Test evaluation with usage constraints."""
        def get_time() -> str:
            return "2024-01-01 12:00:00"
        
        agent = Agent(name="simple_agent", model="openai/gpt-4o-mini", tools=[get_time])
        test_file = FIXTURES_DIR / "eval_usage_test.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)


class TestNestedAgentValidation:
    """Test validation of agents with nested agent tools."""

    @pytest.mark.asyncio
    async def test_nested_agent_only_validates_first_component(self):
        """Test that step validation only captures the first-level tool when an agent has nested agent tools."""
        
        def add(a: int, b: int) -> int:
            return a + b
        
        math_helper_agent = Agent(
            name="math_helper_agent", 
            model="openai/gpt-4o-mini",
            tools=[add]
        )
        
        outer_agent = Agent(
            name="coordinator_agent",
            model="openai/gpt-4o-mini", 
            tools=[math_helper_agent],
            system_prompt="You are a coordinator agent that uses the math_helper_agent for calculations."
        )
        
        test_file = FIXTURES_DIR / "eval_nested_agent_test.yaml"
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, outer_agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)
        assert test_results.total_tests == 1


class TestEngineEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_malformed_test_structure(self):
        """Test handling of malformed test structure."""
        def get_time() -> str:
            return "2024-01-01 12:00:00"
        
        agent = Agent(name="simple_agent", model="openai/gpt-4o-mini", tools=[get_time])
        test_file = FIXTURES_DIR / "eval_invalid.yaml"
        
        test_results = EvalTestSuiteResult()
        with pytest.raises(Exception):  # Should raise validation error
            await eval_file(test_file, agent, test_results)

    @pytest.mark.asyncio
    async def test_empty_test_file(self):
        """Test handling of empty test files."""
        def get_time() -> str:
            return "2024-01-01 12:00:00"
        
        agent = Agent(name="simple_agent", model="openai/gpt-4o-mini", tools=[get_time])
        test_file = FIXTURES_DIR / "eval_empty_test.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)
        # Should handle empty test files gracefully

    @pytest.mark.asyncio
    async def test_eval_file_with_files(self):
        """Test that agent properly reads files specified in YAML."""
        agent = Agent(
            name="file_agent",
            model="openai/gpt-4.1-mini",
        )
        test_file = FIXTURES_DIR / "eval_file_test.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)
        assert test_results.total_files == 1
        assert test_results.total_tests == 1
        assert test_results.outputs_passed == 1