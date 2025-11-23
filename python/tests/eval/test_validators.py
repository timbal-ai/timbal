
import pytest
from timbal.errors import EvalError
from timbal.eval.validators import (
    Validator,
    contains_any_output,
    contains_any_steps,
    contains_ordered_steps,
    contains_output,
    contains_steps,
    equals,
    equals_steps,
    regex,
    semantic_output,
    time,
    usage,
)
from timbal.state import RunContext, set_run_context
from timbal.types.message import Message


class TestValidator:
    """Test the Validator wrapper class."""

    def test_validator_creation(self):
        """Test creating a validator wrapper."""
        def dummy_func(x):
            return x
        
        validator = Validator(dummy_func, "test", "test_ref")
        
        assert validator.name == "test"
        assert validator.ref == "test_ref"
        assert validator.func == dummy_func

    def test_validator_call(self):
        """Test calling a validator."""
        def dummy_func(x):
            return f"processed_{x}"
        
        validator = Validator(dummy_func, "test", "test_ref")
        result = validator("input")
        
        assert result == "processed_input"

    def test_validator_repr(self):
        """Test validator string representation."""
        def dummy_func(x):
            return x
        
        validator = Validator(dummy_func, "test", "short_ref")
        repr_str = repr(validator)
        
        assert "Validator" in repr_str
        assert "test" in repr_str
        assert "short_ref" in repr_str

    def test_validator_repr_truncation(self):
        """Test validator repr with long reference."""
        def dummy_func(x):
            return x
        
        long_ref = "x" * 60  # Very long reference
        validator = Validator(dummy_func, "test", long_ref)
        repr_str = repr(validator)
        
        assert "..." in repr_str
        assert len(repr_str) < len(long_ref) + 50


class TestContainsOutput:
    """Test the contains_output validator."""

    def test_contains_output_basic(self):
        """Test basic contains_output functionality."""
        validator = contains_output("hello")
        
        assert isinstance(validator, Validator)
        assert validator.name == "contains"

    def test_contains_output_success(self):
        """Test contains_output with matching content."""
        validator = contains_output("world")  # matches case
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Should not raise an exception for matching content
        validator(message)  # No exception means success

    def test_contains_output_case_insensitive(self):
        """Test contains_output is case insensitive."""
        validator = contains_output("hello")  # lowercase
        message = Message.validate({
            "role": "assistant", 
            "content": "Hello, world!"  # uppercase
        })
        
        # Should not raise an exception for matching content (case insensitive)
        validator(message)

    def test_contains_output_failure(self):
        """Test contains_output with non-matching content."""
        validator = contains_output("goodbye")
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Should raise EvalError for non-matching content
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "goodbye" in str(exc_info.value)

    def test_contains_output_empty_content(self):
        """Test contains_output with empty content."""
        validator = contains_output("hello")
        message = Message.validate({
            "role": "assistant",
            "content": ""
        })
        
        # Should raise EvalError for empty content
        with pytest.raises(EvalError):
            validator(message)

    def test_contains_output_none_content(self):
        """Test contains_output with None content."""
        validator = contains_output("hello")
        message = Message.validate({
            "role": "assistant",
            "content": None
        })
        
        # Should raise EvalError for None content
        with pytest.raises(EvalError):
            validator(message)


class TestContainsSteps:
    """Test the contains_steps validator."""

    def test_contains_steps_basic(self):
        """Test basic contains_steps functionality."""
        validator = contains_steps([{"name": "test_tool"}])
        
        assert isinstance(validator, Validator)
        assert validator.name == "contains_steps"

    def test_contains_steps_success(self):
        """Test contains_steps with matching steps."""
        validator = contains_steps([{"name": "add"}])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}  # Use "tool" not "name"
        ]
        
        # Should not raise an exception for matching steps
        validator(steps)

    def test_contains_steps_with_input_validation(self):
        """Test contains_steps with input validation."""
        validator = contains_steps([{
            "name": "add",
            "input": {"a": "2", "b": "3"}
        }])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should not raise exception for matching steps with input validation
        validator(steps)

    def test_contains_steps_missing_tool(self):
        """Test contains_steps with missing tool."""
        validator = contains_steps([{"name": "missing_tool"}])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should raise EvalError for missing tool
        with pytest.raises(EvalError) as exc_info:
            validator(steps)
        
        assert "missing_tool" in str(exc_info.value)

    def test_contains_steps_wrong_input(self):
        """Test contains_steps with wrong input values."""
        validator = contains_steps([{
            "name": "add",
            "input": {"a": "5", "b": "6"}  # Different values
        }])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should raise EvalError for wrong input values
        with pytest.raises(EvalError):
            validator(steps)

    def test_contains_steps_multiple_tools(self):
        """Test contains_steps with multiple required tools."""
        validator = contains_steps([
            {"name": "add"},
            {"name": "multiply"}
        ])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5},
            {"tool": "multiply", "input": {"a": 2, "b": 4}, "output": 8}
        ]
        
        # Should not raise exception for matching multiple tools
        validator(steps)

    def test_contains_steps_empty_steps(self):
        """Test contains_steps with empty steps list."""
        validator = contains_steps([{"name": "add"}])
        steps = []
        
        # Should raise EvalError for empty steps when tool is required
        with pytest.raises(EvalError):
            validator(steps)

    def test_contains_steps_invalid_ref(self):
        """Test contains_steps with invalid reference format."""
        with pytest.raises(ValueError):
            contains_steps("invalid_format")

    def test_contains_steps_with_multiple_values(self):
        """Test contains_steps with list of possible values for input field (OR logic)."""
        validator = contains_steps([{
            "name": "greet_person",
            "input": {"name": ["emma", "Emma"]}
        }])
        steps = [
            {"tool": "greet_person", "input": {"name": "Emma"}, "output": "Hello, Emma!"}
        ]
        
        # Should not raise exception when any value in list matches
        validator(steps)

    def test_contains_steps_with_multiple_values_no_match(self):
        """Test contains_steps with list of values where none match."""
        validator = contains_steps([{
            "name": "greet_person",
            "input": {"name": ["alice", "bob"]}
        }])
        steps = [
            {"tool": "greet_person", "input": {"name": "Emma"}, "output": "Hello, Emma!"}
        ]
        
        # Should raise EvalError when none of the values match
        with pytest.raises(EvalError):
            validator(steps)


class TestEqualsSteps:
    """Test the equals_steps validator."""

    def test_equals_steps_basic(self):
        """Test basic equals_steps functionality."""
        validator = equals_steps([{"name": "test_tool"}])
        
        assert isinstance(validator, Validator)
        assert validator.name == "equals_steps"

    def test_equals_steps_success_exact_match(self):
        """Test equals_steps with exact match."""
        validator = equals_steps([{"name": "add"}])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should not raise an exception for exact match
        validator(steps)

    def test_equals_steps_success_multiple_steps(self):
        """Test equals_steps with multiple steps in exact order."""
        validator = equals_steps([
            {"name": "search"},
            {"name": "calculate"}
        ])
        steps = [
            {"tool": "search", "input": {"query": "test"}},
            {"tool": "calculate", "input": {"a": 2, "b": 3}}
        ]
        
        # Should not raise exception for exact match in order
        validator(steps)

    def test_equals_steps_wrong_order(self):
        """Test equals_steps with steps in wrong order."""
        validator = equals_steps([
            {"name": "search"},
            {"name": "calculate"}
        ])
        steps = [
            {"tool": "calculate", "input": {"a": 2, "b": 3}},
            {"tool": "search", "input": {"query": "test"}}
        ]
        
        # Should raise EvalError for wrong order
        with pytest.raises(EvalError) as exc_info:
            validator(steps)
        
        assert "expected tool 'search'" in str(exc_info.value)

    def test_equals_steps_additional_steps(self):
        """Test equals_steps with additional steps (should fail)."""
        validator = equals_steps([{"name": "add"}])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}},
            {"tool": "multiply", "input": {"a": 2, "b": 4}}
        ]
        
        # Should raise EvalError for additional steps
        with pytest.raises(EvalError) as exc_info:
            validator(steps)
        
        assert "Expected 1 step(s), but got 2 step(s)" in str(exc_info.value)

    def test_equals_steps_missing_steps(self):
        """Test equals_steps with missing steps."""
        validator = equals_steps([
            {"name": "search"},
            {"name": "calculate"}
        ])
        steps = [
            {"tool": "search", "input": {"query": "test"}}
        ]
        
        # Should raise EvalError for missing steps
        with pytest.raises(EvalError) as exc_info:
            validator(steps)
        
        assert "Expected 2 step(s), but got 1 step(s)" in str(exc_info.value)

    def test_equals_steps_with_input_validation(self):
        """Test equals_steps with input validation."""
        validator = equals_steps([{
            "name": "add",
            "input": {"a": "2", "b": "3"}
        }])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should not raise exception for matching steps with input validation
        validator(steps)

    def test_equals_steps_wrong_input(self):
        """Test equals_steps with wrong input values."""
        validator = equals_steps([{
            "name": "add",
            "input": {"a": "5", "b": "6"}
        }])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should raise EvalError for wrong input values
        with pytest.raises(EvalError) as exc_info:
            validator(steps)
        
        assert "input 'a' expected '5'" in str(exc_info.value)

    def test_equals_steps_invalid_ref(self):
        """Test equals_steps with invalid reference format."""
        with pytest.raises(ValueError):
            equals_steps("invalid_format")


class TestContainsOrderedSteps:
    """Test the contains_ordered_steps validator."""

    def test_contains_ordered_steps_basic(self):
        """Test basic contains_ordered_steps functionality."""
        validator = contains_ordered_steps([{"name": "test_tool"}])
        
        assert isinstance(validator, Validator)
        assert validator.name == "contains_ordered_steps"

    def test_contains_ordered_steps_success_exact_order(self):
        """Test contains_ordered_steps with steps in exact order."""
        validator = contains_ordered_steps([
            {"name": "search"},
            {"name": "calculate"}
        ])
        steps = [
            {"tool": "search", "input": {"query": "test"}},
            {"tool": "calculate", "input": {"a": 2, "b": 3}}
        ]
        
        # Should not raise exception for correct order
        validator(steps)

    def test_contains_ordered_steps_success_with_additional_steps(self):
        """Test contains_ordered_steps with additional steps between expected ones."""
        validator = contains_ordered_steps([
            {"name": "search"},
            {"name": "calculate"}
        ])
        steps = [
            {"tool": "search", "input": {"query": "test"}},
            {"tool": "other_tool", "input": {"x": 1}},  # Additional step
            {"tool": "calculate", "input": {"a": 2, "b": 3}}
        ]
        
        # Should not raise exception - additional steps are allowed
        validator(steps)

    def test_contains_ordered_steps_wrong_order(self):
        """Test contains_ordered_steps with steps in wrong order."""
        validator = contains_ordered_steps([
            {"name": "search"},
            {"name": "calculate"}
        ])
        steps = [
            {"tool": "calculate", "input": {"a": 2, "b": 3}},
            {"tool": "search", "input": {"query": "test"}}
        ]
        
        # Should raise EvalError for wrong order
        with pytest.raises(EvalError) as exc_info:
            validator(steps)
        
        assert "not found in the correct order" in str(exc_info.value)

    def test_contains_ordered_steps_missing_step(self):
        """Test contains_ordered_steps with missing step."""
        validator = contains_ordered_steps([
            {"name": "search"},
            {"name": "calculate"}
        ])
        steps = [
            {"tool": "search", "input": {"query": "test"}}
        ]
        
        # Should raise EvalError for missing step
        with pytest.raises(EvalError) as exc_info:
            validator(steps)
        
        assert "not found in the correct order" in str(exc_info.value)
        assert "calculate" in str(exc_info.value)

    def test_contains_ordered_steps_with_input_validation(self):
        """Test contains_ordered_steps with input validation."""
        validator = contains_ordered_steps([{
            "name": "add",
            "input": {"a": "2", "b": "3"}
        }])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should not raise exception for matching steps with input validation
        validator(steps)

    def test_contains_ordered_steps_wrong_input(self):
        """Test contains_ordered_steps with wrong input values."""
        validator = contains_ordered_steps([{
            "name": "add",
            "input": {"a": "5", "b": "6"}
        }])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should raise EvalError for wrong input values
        with pytest.raises(EvalError) as exc_info:
            validator(steps)
        
        assert "not found in the correct order" in str(exc_info.value)

    def test_contains_ordered_steps_multiple_steps_with_additional(self):
        """Test contains_ordered_steps with multiple expected steps and additional ones."""
        validator = contains_ordered_steps([
            {"name": "search"},
            {"name": "process"},
            {"name": "calculate"}
        ])
        steps = [
            {"tool": "search", "input": {"query": "test"}},
            {"tool": "other_tool1", "input": {}},  # Additional step
            {"tool": "process", "input": {"data": "x"}},
            {"tool": "other_tool2", "input": {}},  # Additional step
            {"tool": "calculate", "input": {"a": 2, "b": 3}}
        ]
        
        # Should not raise exception - order is correct, additional steps allowed
        validator(steps)

    def test_contains_ordered_steps_invalid_ref(self):
        """Test contains_ordered_steps with invalid reference format."""
        with pytest.raises(ValueError):
            contains_ordered_steps("invalid_format")


class TestSemanticOutput:
    """Test the semantic_output validator."""

    def test_semantic_output_basic(self):
        """Test basic semantic_output functionality."""
        validator = semantic_output("A friendly greeting")
        
        assert isinstance(validator, Validator)
        assert validator.name == "semantic"

    @pytest.mark.asyncio
    async def test_semantic_output_success(self):
        """Test semantic_output with matching semantic content."""
        validator = semantic_output("A mathematical calculation")
        message = Message.validate({
            "role": "assistant",
            "content": "The result of 2 + 2 is 4."
        })
        
        # Test the validator structure
        assert validator.name == "semantic"
        assert validator.ref == ["A mathematical calculation"]
        
        # Semantic validator doesn't return success/errors, it raises on failure
        # If this doesn't raise an exception, the validation passed
        await validator(message)

    def test_semantic_output_creation(self):
        """Test semantic_output validator creation."""
        ref = "A helpful response"
        validator = semantic_output(ref)
        
        assert validator.ref == [ref]
        assert validator.name == "semantic"


class TestEquals:
    """Test the equals validator."""

    def test_equals_basic(self):
        """Test basic equals functionality."""
        validator = equals("Hello, world!")
        
        assert isinstance(validator, Validator)
        assert validator.name == "equals"

    def test_equals_success(self):
        """Test equals with exact match."""
        validator = equals("Hello, world!")
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Should not raise an exception for exact match
        validator(message)

    def test_equals_failure(self):
        """Test equals with non-exact match."""
        validator = equals("Hello, world!")
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, World!"  # Different case
        })
        
        # Should raise EvalError for non-exact match
        with pytest.raises(EvalError):
            validator(message)

    def test_equals_whitespace_handling(self):
        """Test equals handles whitespace by stripping."""
        validator = equals("Hello")
        message = Message.validate({
            "role": "assistant",
            "content": " Hello "  # Extra whitespace should be stripped
        })
        
        # Should not raise exception because equals strips whitespace
        validator(message)


class TestRegexOutput:
    """Test the regex validator."""

    def test_regex_basic(self):
        """Test basic regex functionality."""
        validator = regex(r"\d+")
        
        assert isinstance(validator, Validator)
        assert validator.name == "regex"

    def test_regex_success(self):
        """Test regex with matching pattern."""
        validator = regex(r"\d{4}")  # 4 digits
        message = Message.validate({
            "role": "assistant",
            "content": "The year is 2024."
        })
        
        # Should not raise an exception for matching pattern
        validator(message)

    def test_regex_failure(self):
        """Test regex with non-matching pattern."""
        validator = regex(r"\d{4}")  # 4 digits
        message = Message.validate({
            "role": "assistant",
            "content": "No numbers here!"
        })
        
        # Should raise EvalError for non-matching pattern
        with pytest.raises(EvalError):
            validator(message)

    def test_regex_complex_pattern(self):
        """Test regex with complex pattern."""
        validator = regex(r"[A-Z][a-z]+, [A-Z][a-z]+!")  # "Hello, World!" pattern
        message = Message.validate({
            "role": "assistant", 
            "content": "Hello, World!"
        })
        
        # Should not raise an exception for matching complex pattern
        validator(message)

    def test_regex_invalid_pattern(self):
        """Test regex with invalid regex pattern."""
        with pytest.raises(Exception):  # Should raise a regex compilation error
            regex("[invalid")



class TestValidatorIntegration:
    """Integration tests for validators."""

    def test_multiple_validators_success(self):
        """Test using multiple validators that all pass."""
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, the answer is 42."
        })
        
        validators = [
            contains_output("Hello"),  # Fix case sensitivity
            contains_output("42"),
            regex(r"\d+")
        ]
        
        # All validators should pass without raising exceptions
        for validator in validators:
            validator(message)

    def test_multiple_validators_mixed(self):
        """Test using multiple validators with mixed results."""
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        passing_validator = contains_output("Hello")  # Fix case
        failing_validator = contains_output("goodbye")
        
        # Passing validator should not raise exception
        passing_validator(message)
        
        # Failing validator should raise EvalError
        with pytest.raises(EvalError):
            failing_validator(message)

    def test_validator_error_handling(self):
        """Test validator error handling with invalid inputs."""
        validator = contains_output("test")
        
        # Test with invalid message
        with pytest.raises((TypeError, AttributeError, EvalError)):
            validator("not_a_message")


class TestValidatorEdgeCases:
    """Test edge cases and error conditions for validators."""

    def test_empty_reference_strings(self):
        """Test validators with empty reference strings."""
        validator = contains_output("")
        message = Message.validate({
            "role": "assistant",
            "content": "Any content"
        })
        
        # Empty string should match any content (no exception raised)
        validator(message)

    def test_unicode_content(self):
        """Test validators with Unicode content."""
        validator = contains_output("café")
        message = Message.validate({
            "role": "assistant",
            "content": "I love café ☕"
        })
        
        # Should not raise exception for Unicode content match
        validator(message)

    def test_very_long_content(self):
        """Test validators with very long content."""
        long_content = "x" * 10000
        validator = contains_output("x")
        message = Message.validate({
            "role": "assistant",
            "content": long_content
        })
        
        # Should not raise exception for long content match
        validator(message)

class TestContainsAnyOutput:
    """Test the contains_any_output validator."""

    def test_contains_any_output_basic(self):
        """Test basic contains_any_output functionality."""
        validator = contains_any_output(["hello", "hi"])
        
        assert isinstance(validator, Validator)
        assert validator.name == "contains_any"

    def test_contains_any_output_single_string(self):
        """Test contains_any_output with single string (converted to list)."""
        validator = contains_any_output("hello")
        
        assert validator.ref == ["hello"]

    def test_contains_any_output_success_first_match(self):
        """Test contains_any_output with first item matching."""
        validator = contains_any_output(["hello", "goodbye"])
        message = Message.validate({
            "role": "assistant",
            "content": "hello world"
        })
        
        # Should not raise an exception for matching content
        validator(message)

    def test_contains_any_output_success_second_match(self):
        """Test contains_any_output with second item matching."""
        validator = contains_any_output(["hello", "world"])
        message = Message.validate({
            "role": "assistant",
            "content": "greetings world"
        })
        
        # Should not raise an exception for matching content
        validator(message)

    def test_contains_any_output_success_multiple_match(self):
        """Test contains_any_output with multiple items matching."""
        validator = contains_any_output(["hello", "world"])
        message = Message.validate({
            "role": "assistant",
            "content": "hello world"
        })
        
        # Should not raise an exception for matching content
        validator(message)

    def test_contains_any_output_failure(self):
        """Test contains_any_output with no matches."""
        validator = contains_any_output(["goodbye", "farewell"])
        message = Message.validate({
            "role": "assistant",
            "content": "hello world"
        })
        
        # Should raise EvalError for non-matching content
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "goodbye" in str(exc_info.value)
        assert "farewell" in str(exc_info.value)
        assert "does not contain any of" in str(exc_info.value)

    def test_contains_any_output_empty_content(self):
        """Test contains_any_output with empty content."""
        validator = contains_any_output(["hello", "world"])
        message = Message.validate({
            "role": "assistant",
            "content": ""
        })
        
        # Should raise EvalError for empty content
        with pytest.raises(EvalError):
            validator(message)

    def test_contains_any_output_case_sensitive(self):
        """Test contains_any_output is case sensitive."""
        validator = contains_any_output(["Hello", "World"])
        message = Message.validate({
            "role": "assistant",
            "content": "hello world"
        })
        
        # Should raise EvalError for case mismatch
        with pytest.raises(EvalError):
            validator(message)


class TestContainsAnySteps:
    """Test the contains_any_steps validator."""

    def test_contains_any_steps_basic(self):
        """Test basic contains_any_steps functionality."""
        validator = contains_any_steps([{"name": "tool1"}, {"name": "tool2"}])
        
        assert isinstance(validator, Validator)
        assert validator.name == "contains_any_steps"

    def test_contains_any_steps_success_first_tool(self):
        """Test contains_any_steps with first tool matching."""
        validator = contains_any_steps([{"name": "add"}, {"name": "multiply"}])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should not raise an exception for matching steps
        validator(steps)

    def test_contains_any_steps_success_second_tool(self):
        """Test contains_any_steps with second tool matching."""
        validator = contains_any_steps([{"name": "add"}, {"name": "multiply"}])
        steps = [
            {"tool": "multiply", "input": {"a": 2, "b": 3}, "output": 6}
        ]
        
        # Should not raise an exception for matching steps
        validator(steps)

    def test_contains_any_steps_success_with_input_validation(self):
        """Test contains_any_steps with input validation."""
        validator = contains_any_steps([
            {"name": "add", "input": {"a": "2"}},
            {"name": "multiply", "input": {"x": "5"}}
        ])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should not raise exception for matching tool with input validation
        validator(steps)

    def test_contains_any_steps_no_match(self):
        """Test contains_any_steps with no matching tools."""
        validator = contains_any_steps([{"name": "divide"}, {"name": "subtract"}])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should raise EvalError for no matching tools
        with pytest.raises(EvalError) as exc_info:
            validator(steps)
        
        assert "divide" in str(exc_info.value)
        assert "subtract" in str(exc_info.value)
        assert "No step found with any of the tools" in str(exc_info.value)

    def test_contains_any_steps_empty_steps(self):
        """Test contains_any_steps with empty steps list."""
        validator = contains_any_steps([{"name": "add"}, {"name": "multiply"}])
        steps = []
        
        # Should raise EvalError for empty steps when tools are required
        with pytest.raises(EvalError):
            validator(steps)

    def test_contains_any_steps_invalid_ref(self):
        """Test contains_any_steps with invalid reference format."""
        with pytest.raises(ValueError):
            contains_any_steps("invalid_format")

    def test_contains_any_steps_input_mismatch(self):
        """Test contains_any_steps where tool matches but input doesn't."""
        validator = contains_any_steps([
            {"name": "add", "input": {"a": "5"}},  # Wrong value
            {"name": "multiply", "input": {"x": "10"}}  # Tool not present
        ])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5}
        ]
        
        # Should raise EvalError since tool matches but input doesn't match criteria
        with pytest.raises(EvalError):
            validator(steps)

    def test_contains_any_steps_multiple_steps_one_match(self):
        """Test contains_any_steps with multiple steps, only one matching."""
        validator = contains_any_steps([{"name": "multiply"}, {"name": "divide"}])
        steps = [
            {"tool": "add", "input": {"a": 2, "b": 3}, "output": 5},
            {"tool": "multiply", "input": {"a": 2, "b": 4}, "output": 8},
            {"tool": "subtract", "input": {"a": 5, "b": 2}, "output": 3}
        ]
        
        # Should not raise exception since multiply is found
        validator(steps)


class TestTime:
    """Test the time validator."""

    def test_time_basic(self):
        """Test basic time functionality."""
        validator = time({"max": 5.0})
        
        assert isinstance(validator, Validator)
        assert validator.name == "time"

    def test_time_success_max(self):
        """Test time validator with execution time below max."""
        validator = time({"max": 5.0})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with execution time
        run_context = RunContext()
        run_context._last_execution_time = 3.0
        set_run_context(run_context)
        
        # Should not raise an exception for time within limit
        validator(message)

    def test_time_success_min(self):
        """Test time validator with execution time above min."""
        validator = time({"min": 1.0})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with execution time
        run_context = RunContext()
        run_context._last_execution_time = 2.5
        set_run_context(run_context)
        
        # Should not raise an exception for time above min
        validator(message)

    def test_time_success_between(self):
        """Test time validator with execution time between min and max."""
        validator = time({"min": 1.0, "max": 5.0})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with execution time
        run_context = RunContext()
        run_context._last_execution_time = 3.0
        set_run_context(run_context)
        
        # Should not raise an exception for time within range
        validator(message)

    def test_time_failure_exceeds_max(self):
        """Test time validator with execution time exceeding max."""
        validator = time({"max": 5.0})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with execution time exceeding max
        run_context = RunContext()
        run_context._last_execution_time = 6.0
        set_run_context(run_context)
        
        # Should raise EvalError for time exceeding max
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "greater than or equal to max value" in str(exc_info.value)
        assert "6.000" in str(exc_info.value)
        assert "5.0" in str(exc_info.value)

    def test_time_failure_below_min(self):
        """Test time validator with execution time below min."""
        validator = time({"min": 2.0})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with execution time below min
        run_context = RunContext()
        run_context._last_execution_time = 1.0
        set_run_context(run_context)
        
        # Should raise EvalError for time below min
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "less than or equal to min value" in str(exc_info.value)
        assert "1.000" in str(exc_info.value)
        assert "2.0" in str(exc_info.value)

    def test_time_failure_no_run_context(self):
        """Test time validator with no run context."""
        validator = time({"max": 5.0})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Clear run context
        set_run_context(None)
        
        # Should raise EvalError for missing run context
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "no run context available" in str(exc_info.value)

    def test_time_failure_no_execution_time(self):
        """Test time validator with no execution time in run context."""
        validator = time({"max": 5.0})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context without execution time
        run_context = RunContext()
        set_run_context(run_context)
        
        # Should raise EvalError for missing execution time
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "execution time not available" in str(exc_info.value)

    def test_time_invalid_ref_not_dict(self):
        """Test time validator with invalid reference (not a dict)."""
        with pytest.raises(ValueError) as exc_info:
            time("invalid")
        
        assert "expected dict" in str(exc_info.value)

    def test_time_invalid_ref_no_max_or_min(self):
        """Test time validator with invalid reference (no max or min)."""
        with pytest.raises(ValueError) as exc_info:
            time({})
        
        assert "at least one of 'max' or 'min' keys" in str(exc_info.value)

    def test_time_invalid_max_not_number(self):
        """Test time validator with invalid max value (not a number)."""
        with pytest.raises(ValueError) as exc_info:
            time({"max": "not_a_number"})
        
        assert "time.max must be a number" in str(exc_info.value)

    def test_time_invalid_min_not_number(self):
        """Test time validator with invalid min value (not a number)."""
        with pytest.raises(ValueError) as exc_info:
            time({"min": "not_a_number"})
        
        assert "time.min must be a number" in str(exc_info.value)

    def test_time_boundary_max_exact(self):
        """Test time validator with execution time exactly at max boundary."""
        validator = time({"max": 5.0})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with execution time exactly at max
        run_context = RunContext()
        run_context._last_execution_time = 5.0
        set_run_context(run_context)
        
        # Should raise EvalError (max is exclusive: > not >=)
        with pytest.raises(EvalError):
            validator(message)

    def test_time_boundary_min_exact(self):
        """Test time validator with execution time exactly at min boundary."""
        validator = time({"min": 2.0})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with execution time exactly at min
        run_context = RunContext()
        run_context._last_execution_time = 2.0
        set_run_context(run_context)
        
        # Should raise EvalError (min is exclusive: < not <=)
        with pytest.raises(EvalError):
            validator(message)

    def test_time_not_supported_with_steps(self):
        """Test time validator does not work with steps (time validator only works for input/output)."""
        validator = time({"max": 5.0})
        steps = [{"tool": "add", "input": {"a": 2, "b": 3}}]
        
        # Set up run_context with steps execution time
        run_context = RunContext()
        run_context._last_steps_execution_time = 3.0
        set_run_context(run_context)
        
        # Should raise an exception (time validator is not supported at step level)
        with pytest.raises(EvalError, match="Time validator is not supported at step"):
            validator(steps)


class TestUsage:
    """Test the usage validator."""

    def test_usage_basic(self):
        """Test basic usage functionality."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"max": 1000}})
        
        assert isinstance(validator, Validator)
        assert validator.name == "usage"

    def test_usage_success_max(self):
        """Test usage validator with usage below max."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"max": 1000}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with usage
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini:input_text_tokens": 500
        }
        set_run_context(run_context)
        
        # Should not raise an exception for usage within limit
        validator(message)

    def test_usage_success_min(self):
        """Test usage validator with usage above min."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"min": 10}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with usage
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini:input_text_tokens": 50
        }
        set_run_context(run_context)
        
        # Should not raise an exception for usage above min
        validator(message)

    def test_usage_success_between(self):
        """Test usage validator with usage between min and max."""
        validator = usage({
            "gpt-4.1-mini:input_text_tokens": {
                "min": 10,
                "max": 1000
            }
        })
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with usage
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini:input_text_tokens": 500
        }
        set_run_context(run_context)
        
        # Should not raise an exception for usage within range
        validator(message)

    def test_usage_failure_exceeds_max(self):
        """Test usage validator with usage exceeding max."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"max": 1000}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with usage exceeding max
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini:input_text_tokens": 1500
        }
        set_run_context(run_context)
        
        # Should raise EvalError for usage exceeding max
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "greater than max value" in str(exc_info.value)
        assert "1500" in str(exc_info.value)
        assert "1000" in str(exc_info.value)

    def test_usage_failure_below_min(self):
        """Test usage validator with usage below min."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"min": 100}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with usage below min
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini:input_text_tokens": 50
        }
        set_run_context(run_context)
        
        # Should raise EvalError for usage below min
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "less than min value" in str(exc_info.value)
        assert "50" in str(exc_info.value)
        assert "100" in str(exc_info.value)

    def test_usage_failure_no_run_context(self):
        """Test usage validator with no run context."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"max": 1000}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Clear run context
        set_run_context(None)
        
        # Should raise EvalError for missing run context
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "no run context available" in str(exc_info.value)

    def test_usage_failure_no_usage_data(self):
        """Test usage validator with no usage data in run context."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"max": 1000}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context without usage
        run_context = RunContext()
        set_run_context(run_context)
        
        # Should raise EvalError for missing usage data
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "usage data not available" in str(exc_info.value)

    def test_usage_invalid_ref_not_dict(self):
        """Test usage validator with invalid reference (not a dict)."""
        with pytest.raises(ValueError) as exc_info:
            usage("invalid")
        
        assert "expected dict" in str(exc_info.value)

    def test_usage_invalid_limits_not_dict(self):
        """Test usage validator with invalid limits (not a dict)."""
        with pytest.raises(ValueError) as exc_info:
            usage({"gpt-4.1-mini:input_text_tokens": "invalid"})
        
        assert "expected dict" in str(exc_info.value)

    def test_usage_invalid_no_max_or_min(self):
        """Test usage validator with invalid limits (no max or min)."""
        with pytest.raises(ValueError) as exc_info:
            usage({"gpt-4.1-mini:input_text_tokens": {}})
        
        assert "at least one of 'max', 'min', or 'equals' keys" in str(exc_info.value)

    def test_usage_multiple_constraints(self):
        """Test usage validator with multiple usage constraints."""
        validator = usage({
            "gpt-4.1-mini:input_text_tokens": {"max": 1000},
            "gpt-4.1-mini:output_text_tokens": {"max": 500}
        })
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with usage
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini:input_text_tokens": 500,
            "gpt-4.1-mini:output_text_tokens": 200
        }
        set_run_context(run_context)
        
        # Should not raise an exception for usage within limits
        validator(message)

    def test_usage_prefix_matching(self):
        """Test usage validator with model prefix matching."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"max": 1000}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with usage (model name with date suffix)
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini-2024-01-01:input_text_tokens": 500
        }
        set_run_context(run_context)
        
        # Should not raise an exception (prefix matching should work)
        validator(message)

    def test_usage_prefix_matching_multiple_matches(self):
        """Test usage validator with multiple model matches (should fail)."""
        validator = usage({"gpt-4.1:input_text_tokens": {"max": 1000}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with multiple matching models
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini:input_text_tokens": 500,
            "gpt-4.1-turbo:input_text_tokens": 300
        }
        set_run_context(run_context)
        
        # Should raise EvalError for multiple matches
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "Multiple models match prefix" in str(exc_info.value)

    def test_usage_no_matching_model(self):
        """Test usage validator with no matching model."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"max": 1000}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with different model
        run_context = RunContext()
        run_context._last_usage = {
            "claude-3:input_text_tokens": 500
        }
        set_run_context(run_context)
        
        # Should raise EvalError for no matching model
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "no matching model found" in str(exc_info.value)

    def test_usage_works_with_steps(self):
        """Test usage validator works with steps (not just messages)."""
        validator = usage({"gpt-4.1-mini:input_text_tokens": {"max": 1000}})
        steps = [{"tool": "add", "input": {"a": 2, "b": 3}}]
        
        # Set up run_context with steps usage
        run_context = RunContext()
        run_context._last_steps_usage = {
            "gpt-4.1-mini:input_text_tokens": 500
        }
        set_run_context(run_context)
        
        # Should not raise an exception (usage validator works with steps)
        validator(steps)

    def test_usage_sum_operator(self):
        """Test usage validator with sum operator (+)."""
        validator = usage({"gpt-4.1-mini:input_text_tokens+output_text_tokens": {"max": 2000}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with usage
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini:input_text_tokens": 800,
            "gpt-4.1-mini:output_text_tokens": 600
        }
        set_run_context(run_context)
        
        # Should not raise an exception (sum = 1400 < 2000)
        validator(message)

    def test_usage_sum_operator_exceeds(self):
        """Test usage validator with sum operator exceeding max."""
        validator = usage({"gpt-4.1-mini:input_text_tokens+output_text_tokens": {"max": 1000}})
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Set up run_context with usage
        run_context = RunContext()
        run_context._last_usage = {
            "gpt-4.1-mini:input_text_tokens": 600,
            "gpt-4.1-mini:output_text_tokens": 500
        }
        set_run_context(run_context)
        
        # Should raise EvalError (sum = 1100 > 1000)
        with pytest.raises(EvalError) as exc_info:
            validator(message)
        
        assert "greater than max value" in str(exc_info.value)

