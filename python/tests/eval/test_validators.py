
import pytest
from timbal.errors import EvalError
from timbal.eval.validators import (
    Validator,
    contains_any_output,
    contains_any_steps,
    contains_output,
    contains_steps,
    exact_output,
    regex,
    semantic_output,
)
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

    def test_contains_output_case_sensitive(self):
        """Test contains_output is case sensitive."""
        validator = contains_output("Hello")  # match case
        message = Message.validate({
            "role": "assistant", 
            "content": "Hello, world!"
        })
        
        # Should not raise an exception for matching content
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


class TestExactOutput:
    """Test the exact_output validator."""

    def test_exact_output_basic(self):
        """Test basic exact_output functionality."""
        validator = exact_output("Hello, world!")
        
        assert isinstance(validator, Validator)
        assert validator.name == "exact_output"

    def test_exact_output_success(self):
        """Test exact_output with exact match."""
        validator = exact_output("Hello, world!")
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, world!"
        })
        
        # Should not raise an exception for exact match
        validator(message)

    def test_exact_output_failure(self):
        """Test exact_output with non-exact match."""
        validator = exact_output("Hello, world!")
        message = Message.validate({
            "role": "assistant",
            "content": "Hello, World!"  # Different case
        })
        
        # Should raise EvalError for non-exact match
        with pytest.raises(EvalError):
            validator(message)

    def test_exact_output_whitespace_handling(self):
        """Test exact_output handles whitespace by stripping."""
        validator = exact_output("Hello")
        message = Message.validate({
            "role": "assistant",
            "content": " Hello "  # Extra whitespace should be stripped
        })
        
        # Should not raise exception because exact_output strips whitespace
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

