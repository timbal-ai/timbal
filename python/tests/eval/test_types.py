from pathlib import Path

import pytest
from pydantic import ValidationError
from timbal.eval.types.input import Input
from timbal.eval.types.output import Output
from timbal.eval.types.result import EvalResult, EvalTestSuiteResult
from timbal.eval.types.steps import Steps
from timbal.eval.types.test import Test
from timbal.eval.types.test_suite import TestSuite
from timbal.eval.types.turn import Turn
from timbal.eval.validators import contains_output, contains_steps
from timbal.types.file import File

TEST_FILE = Path(__file__).parent / "fixtures" / "math_question.md"


class TestInput:
    """Test the Input model."""

    def test_input_string(self):
        """Test creating input with string (converts to prompt list)."""
        input_obj = Input(prompt="Hello, world!")
        
        assert input_obj.prompt == ["Hello, world!"]

    def test_input_prompt_list(self):
        """Test creating input with prompt as list."""
        input_obj = Input(prompt=["Hello", "world"])
        
        assert input_obj.prompt == ["Hello", "world"]

    def test_input_with_files_in_prompt(self):
        """Test creating input with files in prompt list."""
        input_obj = Input(prompt=["Process this file", str(TEST_FILE)])
        
        assert len(input_obj.prompt) == 2
        assert input_obj.prompt[0] == "Process this file"
        assert input_obj.prompt[1] == str(TEST_FILE)

    def test_input_files_only(self):
        """Test creating input with files only (no text)."""
        input_obj = Input(prompt=[str(TEST_FILE)])
        
        assert len(input_obj.prompt) == 1
        assert input_obj.prompt[0] == str(TEST_FILE)

    def test_input_empty(self):
        """Test creating empty input."""
        with pytest.raises(ValidationError):
            Input()  # Should require at least prompt

    def test_input_validation(self):
        """Test input validation."""
        # Valid inputs
        Input(prompt="Hello")
        Input(prompt=[str(TEST_FILE)])
        Input(prompt=["Hello", str(TEST_FILE)])
        
        # Invalid - prompt is None
        with pytest.raises(ValidationError):
            Input(prompt=None)

    def test_input_empty_prompt_list(self):
        """Test input with empty prompt list."""
        with pytest.raises(ValidationError):
            Input(prompt=[])  # Should require at least one item

    def test_input_multiple_files(self):
        """Test input with multiple files in prompt."""
        input_obj = Input(prompt=["Process these files", str(TEST_FILE), str(TEST_FILE)])
        
        assert len(input_obj.prompt) == 3
        assert input_obj.prompt[0] == "Process these files"

    def test_input_relative_path_resolution(self):
        """Test that relative paths are resolved correctly."""
        # Create an input with a relative path string in prompt
        input_obj = Input(prompt=["Process this file", "./math_question.md"])
        
        # Test with test_file_dir (should resolve)
        test_file_dir = Path(__file__).parent / "fixtures"
        message = input_obj.to_message(role="user", test_file_dir=test_file_dir)
        assert len(message.content) == 2  # text + file
        file_content = [c for c in message.content if hasattr(c, 'file')][0]
        expected_path = (test_file_dir / "math_question.md").resolve()
        assert str(file_content.file) == str(expected_path)
        
    def test_input_absolute_path_no_resolution(self):
        """Test that absolute paths are not modified."""
        # Create an input with an absolute path in prompt
        input_obj = Input(prompt=["Process this file", str(TEST_FILE)])
        
        # Test with test_file_dir (should not change absolute path)
        test_file_dir = Path(__file__).parent / "fixtures"
        message = input_obj.to_message(role="user", test_file_dir=test_file_dir)
        assert len(message.content) == 2  # text + file
        file_content = [c for c in message.content if hasattr(c, 'file')][0]
        assert str(file_content.file) == str(TEST_FILE)


class TestOutput:
    """Test the Output model."""

    def test_output_content_only(self):
        """Test creating output with content only."""
        output = Output(content=["Hello, world!"])
        
        assert output.content == ["Hello, world!"]
        assert output.validators is None

    def test_output_validators_only(self):
        """Test creating output with validators only."""
        output = Output(validators={"contains": ["hello"]})
        
        assert output.content is None
        assert output.validators is not None
        assert len(output.validators) == 1

    def test_output_content_and_validators(self):
        """Test creating output with both content and validators."""
        output = Output(content=["Hello"], validators={"contains": ["hello"]})
        
        assert output.content == ["Hello"]
        assert output.validators is not None

    def test_output_empty(self):
        """Test creating empty output."""
        output = Output()
        
        # Content should be None when not provided
        assert output.content is None
        # Validators should be None when not provided
        assert output.validators is None

    def test_output_multiple_validators(self):
        """Test output with multiple validators."""
        output = Output(validators={"contains": ["hello", "world"]})
        
        assert output.validators is not None
        assert len(output.validators) == 1

    def test_output_string_content(self):
        """Test output with string content (converts to list)."""
        output = Output(content="Hello")
        
        assert output.content == ["Hello"]

    def test_output_with_files_in_content(self):
        """Test output with files in content list."""
        output = Output(content=["Hello", str(TEST_FILE)])
        
        assert len(output.content) == 2
        assert output.content[0] == "Hello"
        assert output.content[1] == str(TEST_FILE)


class TestSteps:
    """Test the Steps model."""

    def test_steps_with_validators(self):
        """Test creating steps with validators."""
        validators = [contains_steps([{"name": "test_tool"}])]
        steps = Steps(validators=validators)
        
        assert steps.validators == validators

    def test_steps_empty_validators(self):
        """Test creating steps with empty validators."""
        steps = Steps(validators=[])
        assert steps.validators == []

    def test_steps_multiple_validators(self):
        """Test steps with multiple validators."""
        validators = [
            contains_steps([{"name": "tool1"}]),
            contains_steps([{"name": "tool2"}])
        ]
        steps = Steps(validators=validators)
        
        assert len(steps.validators) == 2

    def test_steps_none_validators(self):
        """Test creating steps with None validators."""
        steps = Steps()
        assert steps.validators is None


class TestTurn:
    """Test the Turn model."""

    def test_turn_basic(self):
        """Test creating a basic turn."""
        input_obj = Input(prompt="Hello")
        turn = Turn(input=input_obj)
        
        assert turn.input == input_obj
        assert turn.output is None
        assert turn.steps is None
        assert turn.usage is None

    def test_turn_full(self):
        """Test creating a turn with all fields."""
        input_obj = Input(prompt="Hello")
        output_obj = Output(content=["Hi there!"])
        steps_obj = Steps(validators=[contains_steps([{"name": "greet"}])])
        usage = [{"max": 1000, "type": "tokens"}]
        
        turn = Turn(
            input=input_obj,
            output=output_obj,
            steps=steps_obj,
            usage=usage
        )
        
        assert turn.input == input_obj
        assert turn.output == output_obj
        assert turn.steps == steps_obj
        assert turn.usage == usage

    def test_turn_missing_input(self):
        """Test turn validation requires input."""
        with pytest.raises(ValidationError):
            Turn()  # Missing required input field

    def test_turn_with_output_validators(self):
        """Test turn with output validators."""
        input_obj = Input(prompt="What is 2+2?")
        output_obj = Output(validators={"contains": ["4"]})
        
        turn = Turn(input=input_obj, output=output_obj)
        
        assert turn.output.validators is not None
        assert len(turn.output.validators) == 1

    def test_turn_usage_constraints(self):
        """Test turn with usage constraints."""
        input_obj = Input(prompt="Hello")
        usage = [
            {"max": 100, "type": "input_tokens"},
            {"min": 10, "type": "output_tokens"}
        ]
        
        turn = Turn(input=input_obj, usage=usage)
        
        assert turn.usage == usage
        assert len(turn.usage) == 2


class TestTest:
    """Test the Test model."""

    def test_test_basic(self):
        """Test creating a basic test."""
        turn = Turn(input=Input(prompt="Hello"))
        test = Test(name="test1", turns=[turn])
        
        assert test.name == "test1"
        assert test.description is None
        assert test.turns == [turn]

    def test_test_with_description(self):
        """Test creating test with description."""
        turn = Turn(input=Input(prompt="Hello"))
        test = Test(
            name="test1",
            description="A simple test",
            turns=[turn]
        )
        
        assert test.name == "test1"
        assert test.description == "A simple test"
        assert test.turns == [turn]

    def test_test_multiple_turns(self):
        """Test test with multiple turns."""
        turn1 = Turn(input=Input(prompt="Hello"))
        turn2 = Turn(input=Input(prompt="How are you?"))
        
        test = Test(name="multi_turn", turns=[turn1, turn2])
        
        assert len(test.turns) == 2
        assert test.turns[0] == turn1
        assert test.turns[1] == turn2

    def test_test_empty_turns(self):
        """Test test with empty turns list."""
        test = Test(name="empty_test", turns=[])
        assert test.turns == []

    def test_test_missing_name(self):
        """Test test validation requires name."""
        turn = Turn(input=Input(text="Hello"))
        with pytest.raises(ValidationError):
            Test(turns=[turn])  # Missing required name field

    def test_test_missing_turns(self):
        """Test test validation requires turns."""
        with pytest.raises(ValidationError):
            Test(name="test1")  # Missing required turns field

    def test_test_pytest_attribute(self):
        """Test that Test class has __test__ = False for pytest."""
        assert hasattr(Test, '__test__')
        assert Test.__test__ is False


class TestTestSuite:
    """Test the TestSuite model."""

    def test_test_suite_basic(self):
        """Test creating a basic test suite."""
        turn = Turn(input=Input(prompt="Hello"))
        test = Test(name="test1", turns=[turn])
        test_suite = TestSuite(tests=[test])
        
        assert test_suite.tests == [test]

    def test_test_suite_multiple_tests(self):
        """Test test suite with multiple tests."""
        turn1 = Turn(input=Input(prompt="Hello"))
        turn2 = Turn(input=Input(prompt="Goodbye"))
        test1 = Test(name="test1", turns=[turn1])
        test2 = Test(name="test2", turns=[turn2])
        
        test_suite = TestSuite(tests=[test1, test2])
        
        assert len(test_suite.tests) == 2
        assert test_suite.tests[0] == test1
        assert test_suite.tests[1] == test2

    def test_test_suite_empty(self):
        """Test empty test suite."""
        test_suite = TestSuite(tests=[])
        assert test_suite.tests == []

    def test_test_suite_from_list(self):
        """Test creating test suite from list (validator)."""
        turn = Turn(input=Input(prompt="Hello"))
        test = Test(name="test1", turns=[turn])
        
        # Should be able to create from list directly using model_validate
        test_suite = TestSuite.model_validate([test])
        assert test_suite.tests == [test]

    def test_test_suite_pytest_attribute(self):
        """Test that TestSuite class has __test__ = False for pytest."""
        assert hasattr(TestSuite, '__test__')
        assert TestSuite.__test__ is False


class TestEvalResult:
    """Test the EvalResult model."""

    def test_eval_result_basic(self):
        """Test creating a basic eval result."""
        result = EvalResult(
            test_name="test1",
            test_path="/path/to/test1.yaml",
            input="Hello",
            actual_output="Hi there"
        )
        
        assert result.test_name == "test1"
        assert result.test_path == "/path/to/test1.yaml"
        assert result.input == "Hello"
        assert result.actual_output == "Hi there"

    def test_eval_result_with_errors(self):
        """Test eval result with errors."""
        result = EvalResult(
            test_name="test1",
            test_path="/path/to/test1.yaml",
            input="Hello",
            actual_output="Error occurred",
            execution_error={"type": "RuntimeError", "message": "Test failed"}
        )
        
        assert result.test_name == "test1"
        assert result.execution_error is not None
        assert result.execution_error["type"] == "RuntimeError"

    def test_eval_result_with_turns(self):
        """Test eval result with turn data."""
        steps = [{"name": "greet", "input": {"message": "Hello"}, "output": "Hi"}]
        result = EvalResult(
            test_name="test1",
            test_path="/path/to/test1.yaml",
            input="Hello",
            actual_output="Hi",
            actual_steps=steps
        )
        
        assert result.actual_steps == steps
        assert len(result.actual_steps) == 1

    def test_eval_result_missing_fields(self):
        """Test eval result validation."""
        with pytest.raises(ValidationError):
            EvalResult()  # Missing required fields


class TestEvalTestSuiteResult:
    """Test the EvalTestSuiteResult model."""

    def test_eval_test_suite_result_basic(self):
        """Test creating a basic eval test suite result."""
        test_result = EvalResult(
            test_name="test1",
            test_path="/path/to/test1.yaml",
            input="Hello",
            actual_output="Hi"
        )
        
        result = EvalTestSuiteResult(
            total_tests=1,
            outputs_passed=1,
            tests_failed=[test_result]
        )
        
        assert result.total_tests == 1
        assert result.outputs_passed == 1
        assert result.tests_failed == [test_result]

    def test_eval_test_suite_result_with_errors(self):
        """Test eval test suite result with errors."""
        result = EvalTestSuiteResult(
            total_tests=2,
            outputs_failed=2,
            execution_errors=2
        )
        
        assert result.total_tests == 2
        assert result.outputs_failed == 2
        assert result.execution_errors == 2

    def test_eval_test_suite_result_multiple_tests(self):
        """Test eval test suite result with multiple test results."""
        test_result1 = EvalResult(
            test_name="test1",
            test_path="/path/to/test1.yaml",
            input="Hello",
            actual_output="Hi"
        )
        test_result2 = EvalResult(
            test_name="test2",
            test_path="/path/to/test2.yaml",
            input="Goodbye",
            actual_output="Bye"
        )
        
        result = EvalTestSuiteResult(
            total_tests=2,
            outputs_passed=1,
            outputs_failed=1,
            tests_failed=[test_result1, test_result2]
        )
        
        assert len(result.tests_failed) == 2
        assert result.tests_failed[0].test_name == "test1"
        assert result.tests_failed[1].test_name == "test2"

    def test_eval_test_suite_result_empty_tests(self):
        """Test eval test suite result with no tests."""
        result = EvalTestSuiteResult(
            total_tests=0,
            outputs_passed=0,
            tests_failed=[]
        )
        
        assert result.tests_failed == []
        assert result.total_tests == 0


class TestTypeIntegration:
    """Integration tests for the type system."""

    def test_complete_evaluation_structure(self):
        """Test creating a complete evaluation structure."""
        # Create a complete test with all components
        input_obj = Input(
            prompt=["Calculate 2 + 3", str(TEST_FILE)]
        )
        
        output_obj = Output(
            content=["The result is 5"],
            validators={"contains": ["5"]}
        )
        
        steps_obj = Steps(
            validators=[contains_steps([{"name": "add", "input": {"a": "2", "b": "3"}}])]
        )
        
        turn = Turn(
            input=input_obj,
            output=output_obj,
            steps=steps_obj,
            usage=[{"max": 1000, "type": "tokens"}]
        )
        
        test = Test(
            name="math_test",
            description="Test mathematical operations",
            turns=[turn]
        )
        
        test_suite = TestSuite(tests=[test])
        
        # Verify the complete structure
        assert test_suite.tests[0].name == "math_test"
        assert test_suite.tests[0].turns[0].input.prompt[0] == "Calculate 2 + 3"
        assert test_suite.tests[0].turns[0].output.content[0] == "The result is 5"
        assert len(test_suite.tests[0].turns[0].input.prompt) == 2  # text + file
        assert test_suite.tests[0].turns[0].output.validators is not None
        assert len(test_suite.tests[0].turns[0].steps.validators) == 1

    def test_serialization_roundtrip(self):
        """Test that types can be serialized and deserialized."""
        # Create a test object
        turn = Turn(
            input=Input(prompt="Hello"),
            output=Output(validators={"contains": ["hi"]})
        )
        test = Test(name="test1", turns=[turn])
        
        # Convert to dict and back
        test_dict = test.model_dump()
        # to_dict may have issues with validators, so use model_dump_json and parse
        reconstructed = Test.model_validate(test_dict)
        
        assert reconstructed.name == test.name
        assert len(reconstructed.turns) == len(test.turns)
        # Verify the structure is preserved
        assert reconstructed.turns[0].input.prompt == ["Hello"]
        assert reconstructed.turns[0].output.validators is not None

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored due to ConfigDict settings."""
        # This should not raise an error due to extra="ignore"
        test = Test.model_validate({
            "name": "test1",
            "turns": [],
            "extra_field": "should be ignored"
        })
        
        assert test.name == "test1"
        assert test.turns == []
        # extra_field should be ignored


class TestTypeValidation:
    """Test validation and error handling in types."""

    def test_input_validation_errors(self):
        """Test various input validation errors."""
        # Invalid - prompt is None
        with pytest.raises(ValidationError):
            Input(prompt=None)  # Should require prompt

    def test_turn_validation_errors(self):
        """Test turn validation errors."""
        # Invalid input type
        with pytest.raises(ValidationError):
            Turn(input=123)  # Should be Input object, not int

    def test_test_validation_errors(self):
        """Test test validation errors."""
        # Invalid turns type
        with pytest.raises(ValidationError):
            Test(name="test1", turns="not_a_list")
        
        # Invalid turn objects in list
        with pytest.raises(ValidationError):
            Test(name="test1", turns=["not_a_turn"])

    def test_test_suite_validation_errors(self):
        """Test test suite validation errors."""
        # Invalid tests type
        with pytest.raises(ValidationError):
            TestSuite(tests="not_a_list")
        
        # Invalid test objects in list
        with pytest.raises(ValidationError):
            TestSuite(tests=["not_a_test"])

    def test_result_validation_errors(self):
        """Test result validation errors."""
        # Missing required fields
        with pytest.raises(ValidationError):
            EvalResult(test_name="test1")  # Missing test_path, input, actual_output
        
        # EvalTestSuiteResult has no required fields, all have defaults
        result = EvalTestSuiteResult()  # This should work
        assert result.total_tests == 0
