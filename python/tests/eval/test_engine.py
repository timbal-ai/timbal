from pathlib import Path

import pytest
import yaml
from timbal import Agent
from timbal.eval.engine import eval_file, eval_output, eval_steps, run_turn
from timbal.eval.types.input import Input
from timbal.eval.types.output import Output
from timbal.eval.types.result import EvalTestSuiteResult
from timbal.eval.types.steps import Steps
from timbal.eval.types.test_suite import TestSuite
from timbal.eval.types.turn import Turn
from timbal.state import get_run_context
from timbal.tools import WebSearch
from timbal.types.message import Message

# Get the fixtures directory path
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def get_time() -> str:
    """Helper function for tests that need a time function."""
    return "2024-01-01 12:00:00"


def add(a: int, b: int) -> int:
    """Helper function for tests that need an add function."""
    return a + b


def get_current_time() -> str:
    """Helper function for tests that need a current time function."""
    return "2024-01-01 12:00:00"


def add_tool(a: int, b: int) -> int:
    """Helper function for tests that need an add tool function."""
    return a + b


class TestEvalFile:
    """Test the eval_file function - the main entry point for evaluations."""

    @pytest.mark.asyncio
    async def test_eval_file_simple(self):
        """Test basic file evaluation."""
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
        agent = Agent(name="time_agent", model="openai/gpt-4o-mini", tools=[get_time])
        test_file = FIXTURES_DIR / "eval_time_test.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results, test_name="time_validation_multiple")
        
        assert isinstance(result, EvalTestSuiteResult)
        assert test_results.total_tests == 1

    @pytest.mark.asyncio
    async def test_eval_file_nonexistent(self):
        """Test error handling for nonexistent file."""
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
        test = next(t for t in test_suite.tests if t.name == "math_test")
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert turn.input.prompt == ["What is 2 + 3?"]
        assert isinstance(turn.output, Output)
        assert isinstance(turn.steps, Steps)
        
        success, errors, steps = await eval_steps(turn)
        
        assert success is False
        assert len(errors) > 0
        assert isinstance(steps, list)

    @pytest.mark.asyncio
    async def test_eval_steps_no_tracing_yaml(self):
        """Test step evaluation without tracing using YAML fixture."""
        
        yaml_path = FIXTURES_DIR / "eval_agent.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert turn.output is None or isinstance(turn.output, Output)
        assert isinstance(turn.steps, Steps)
        
        success, errors, steps = await eval_steps(turn)
        
        assert success is False
        assert "No trace data available" in errors[0]
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
        assert turn.steps is None
        
        success, errors, steps = await eval_steps(turn)
        
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
        
        assert success is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_eval_output_with_text_only_yaml(self):
        """Test output evaluation with content only from YAML (no validation, just fixed record)."""
        
        yaml_path = FIXTURES_DIR / "eval_multi_turn_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert isinstance(turn.output, Output)
        assert turn.output.content is not None
        assert turn.output.validators is None
        
        message = Message.validate({
            "role": "assistant", 
            "content": [{"type": "text", "text": "The answer is 4"}]
        })
        
        success, errors = await eval_output(turn, message)
        
        assert success is None
        assert errors == []

    @pytest.mark.asyncio
    async def test_eval_output_no_output_yaml(self):
        """Test output evaluation when no output is specified in YAML."""
        
        turn = Turn(input=Input(prompt="Hello"))
        
        assert isinstance(turn.input, Input)
        assert turn.output is None
        
        message = Message.validate({
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello there!"}]
        })
        
        success, errors = await eval_output(turn, message)
        
        assert success is None
        assert errors == []


class TestUsageValidator:
    """Test the usage validator using real YAML fixtures."""

    def test_usage_validator_in_output_yaml(self):
        """Test usage validator in output from YAML fixture."""
        
        yaml_path = FIXTURES_DIR / "eval_usage_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert turn.output is not None
        assert turn.output.validators is not None
        usage_validators = turn.output.validators.get("usage", [])
        assert len(usage_validators) > 0
        assert any(hasattr(v, "name") and v.name == "usage" for v in usage_validators)

    def test_no_usage_validator_yaml(self):
        """Test when no usage validators are specified."""
        
        yaml_path = FIXTURES_DIR / "eval_simple_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        if turn.output and turn.output.validators:
            usage_validators = turn.output.validators.get("usage", [])
            assert len(usage_validators) == 0


class TestRunTurn:
    """Test the run_turn function using real YAML fixtures."""

    @pytest.mark.asyncio
    async def test_run_turn_basic_yaml(self):
        """Test basic turn execution using YAML fixture."""
        
        yaml_path = FIXTURES_DIR / "eval_agent.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = next(t for t in test_suite.tests if t.name == "time_test")
        turn = test.turns[0]
        
        assert isinstance(turn.input, Input)
        assert isinstance(turn.output, Output)
        assert test.name == "time_test"
        
        agent = Agent(name="time_agent", model="openai/gpt-4o-mini", tools=[get_current_time])
        conversation_history = []
        test_results = EvalTestSuiteResult()
        
        await run_turn(agent, turn, test, conversation_history, test_results, str(yaml_path.name), yaml_path.parent)

        assert test_results.total_validations == 2

    @pytest.mark.asyncio
    async def test_run_turn_with_conversation_history_yaml(self):
        """Test turn execution with conversation history using multi-turn YAML."""
        
        yaml_path = FIXTURES_DIR / "eval_multi_turn_test.yaml"
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        test_suite = TestSuite.model_validate(yaml_data)
        test = test_suite.tests[0]
        second_turn = test.turns[1]
        
        assert isinstance(second_turn.input, Input)
        assert isinstance(second_turn.output, Output)
        assert len(test.turns) == 2
        
        agent = Agent(name="memory_agent", model="openai/gpt-4o-mini", tools=[get_time])
        
        conversation_history = [
            Message.validate({"role": "user", "content": [{"type": "text", "text": "My name is Bob"}]}),
            Message.validate({"role": "assistant", "content": [{"type": "text", "text": "Nice to meet you, Bob!"}]})
        ]
        test_results = EvalTestSuiteResult()
        
        await run_turn(agent, second_turn, test, conversation_history, test_results, str(yaml_path.name), yaml_path.parent)
        
        assert test_results.total_validations == 1

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
        
        def add_tool(a: int, b: int) -> int:
            raise ValueError("Tool failed")
        
        agent = Agent(name="add_tool", model="openai/gpt-4o-mini", system_prompt="Use the add tool to add two numbers", tools=[add_tool])
        conversation_history = []
        test_results = EvalTestSuiteResult()
        
        await run_turn(agent, turn, test, conversation_history, test_results, str(yaml_path.name), yaml_path.parent)
        
        assert test_results.execution_errors == 1
        # When there's an execution error, validations are not executed
        assert test_results.total_validations == 0


    @pytest.mark.asyncio
    async def test_pre_hook_persistence_with_yaml_fixture(self):
        """Test pre_hook persistence using YAML fixture."""
        
        yaml_path = FIXTURES_DIR / "eval_pre_hook_persistence.yaml"
        
        async def pre_hook():
            span = get_run_context().current_span()
            prompt = span.input.get("prompt", "")
            if isinstance(prompt, list):
                prompt_text = " ".join(str(p) for p in prompt if not hasattr(p, 'path'))
            else:
                prompt_text = str(prompt)
            if "premium" in prompt_text:
                span.input["points"] = "10"
            return None

        agent = Agent(name="pre_hook_persistence_agent", model="openai/gpt-4.1-mini", pre_hook=pre_hook)
        test_results = EvalTestSuiteResult()
        
        result = await eval_file(yaml_path, agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)
        assert test_results.total_validations == 1

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
        assert result.total_validations == 1

    @pytest.mark.asyncio
    async def test_usage_constraints(self):
        """Test evaluation with usage constraints."""
        agent = Agent(name="simple_agent", model="openai/gpt-4o-mini", tools=[get_time, WebSearch()])
        test_file = FIXTURES_DIR / "eval_usage_test.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)
        assert test_results.total_validations == 10


class TestNestedAgentValidation:
    """Test validation of agents with nested agent tools."""

    @pytest.mark.asyncio
    async def test_nested_agent_only_validates_first_component(self):
        """Test that step validation only captures the first-level tool when an agent has nested agent tools."""
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
        assert test_results.steps_passed == 1 or test_results.steps_failed == 1


class TestEngineEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_malformed_test_structure(self):
        """Test handling of malformed test structure."""
        agent = Agent(name="simple_agent", model="openai/gpt-4o-mini", tools=[get_time])
        test_file = FIXTURES_DIR / "eval_invalid.yaml"
        
        test_results = EvalTestSuiteResult()
        with pytest.raises(Exception):
            await eval_file(test_file, agent, test_results)

    @pytest.mark.asyncio
    async def test_empty_test_file(self):
        """Test handling of empty test files."""
        agent = Agent(name="simple_agent", model="openai/gpt-4o-mini", tools=[get_time])
        test_file = FIXTURES_DIR / "eval_empty_test.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)
        assert test_results.total_tests == 0
        assert test_results.total_validations == 0

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

    @pytest.mark.asyncio
    async def test_eval_file_with_usage_validator(self):
        """Test eval_file with usage validator."""
        agent = Agent(name="math_agent", model="openai/gpt-4o-mini", tools=[add, WebSearch()])
        test_file = FIXTURES_DIR / "eval_usage_test.yaml"
        
        test_results = EvalTestSuiteResult()
        result = await eval_file(test_file, agent, test_results)
        
        assert isinstance(result, EvalTestSuiteResult)
        assert test_results.total_files == 1
        assert test_results.total_tests == 4
        assert test_results.total_validations == 10  # 3+2+2+3 validations across 4 tests


class TestOnlyLastTurnValidators:
    """Test that only the last turn can have validators."""

    def test_single_turn_with_validators(self):
        """Test that a single turn with validators is valid."""
        test_data = {
            "name": "single_turn_test",
            "turns": [
                {
                    "input": {"prompt": "Hello"},
                    "output": {
                        "content": {
                            "validators": {
                                "contains": ["hello"]
                            }
                        }
                    }
                }
            ]
        }
        from timbal.eval.types.test import Test
        test = Test.model_validate(test_data)
        assert test.name == "single_turn_test"
        assert len(test.turns) == 1

    def test_multiple_turns_only_last_has_validators(self):
        """Test that multiple turns with validators only in last turn is valid."""
        test_data = {
            "name": "multi_turn_test",
            "turns": [
                {
                    "input": {"prompt": "Hello"},
                    "output": "Hi!"
                },
                {
                    "input": {"prompt": "How are you?"},
                    "output": {
                        "content": {
                            "validators": {
                                "contains": ["good"]
                            }
                        }
                    }
                }
            ]
        }
        from timbal.eval.types.test import Test
        test = Test.model_validate(test_data)
        assert test.name == "multi_turn_test"
        assert len(test.turns) == 2

    def test_input_validators_in_non_last_turn_raises_error(self):
        """Test that input validators in non-last turn raises ValueError."""
        test_data = {
            "name": "invalid_test",
            "turns": [
                {
                    "input": {
                        "prompt": "Hello",
                        "validators": {
                            "usage": {
                                "gpt-4.1-mini:input_text_tokens": {"max": 100}
                            }
                        }
                    },
                    "output": "Hi!"
                },
                {
                    "input": {"prompt": "How are you?"},
                    "output": {
                        "content": {
                            "validators": {
                                "contains": ["good"]
                            }
                        }
                    }
                }
            ]
        }
        from timbal.eval.types.test import Test
        with pytest.raises(ValueError, match="Turn 1.*not the last turn.*input validators"):
            Test.model_validate(test_data)

    def test_output_validators_in_non_last_turn_raises_error(self):
        """Test that output validators in non-last turn raises ValueError."""
        test_data = {
            "name": "invalid_test",
            "turns": [
                {
                    "input": {"prompt": "Hello"},
                    "output": {
                        "content": {
                            "validators": {
                                "contains": ["hi"]
                            }
                        }
                    }
                },
                {
                    "input": {"prompt": "How are you?"},
                    "output": {
                        "content": {
                            "validators": {
                                "contains": ["good"]
                            }
                        }
                    }
                }
            ]
        }
        from timbal.eval.types.test import Test
        with pytest.raises(ValueError, match="Turn 1.*not the last turn.*output validators"):
            Test.model_validate(test_data)

    def test_steps_validators_in_non_last_turn_raises_error(self):
        """Test that steps validators in non-last turn raises ValueError."""
        test_data = {
            "name": "invalid_test",
            "turns": [
                {
                    "input": {"prompt": "Hello"},
                    "output": "Hi!",
                    "steps": {
                        "validators": {
                            "contains": [{"name": "some_tool"}]
                        }
                    }
                },
                {
                    "input": {"prompt": "How are you?"},
                    "output": {
                        "content": {
                            "validators": {
                                "contains": ["good"]
                            }
                        }
                    }
                }
            ]
        }
        from timbal.eval.types.test import Test
        with pytest.raises(ValueError, match="Turn 1.*not the last turn.*steps validators"):
            Test.model_validate(test_data)