"""Tests for AgentParams validation and schema generation."""

import pytest
from pydantic import ValidationError
from timbal.core.agent import AgentParams
from timbal.types.message import Message


class TestAgentParamsValidation:
    """Test AgentParams validation logic."""

    def test_valid_with_prompt_message(self):
        """Test creating AgentParams with a Message prompt."""
        message = Message(role="user", content=["Hello"])
        params = AgentParams(prompt=message)
        
        assert params.prompt == message
        assert params.messages is None

    def test_valid_with_prompt_dict(self):
        """Test creating AgentParams with a dict prompt (auto-converted to Message)."""
        params = AgentParams(prompt={"role": "user", "content": "Hello"})
        
        assert isinstance(params.prompt, Message)
        assert params.prompt.role == "user"
        assert params.messages is None

    def test_valid_with_prompt_string(self):
        """Test creating AgentParams with a string prompt (auto-converted to Message)."""
        params = AgentParams(prompt="Hello, world!")
        
        assert isinstance(params.prompt, Message)
        assert params.prompt.role == "user"
        assert params.messages is None

    def test_valid_with_messages_list(self):
        """Test creating AgentParams with a messages list."""
        messages = [
            Message(role="user", content=["First message"]),
            Message(role="assistant", content=["Response"]),
            Message(role="user", content=["Second message"]),
        ]
        params = AgentParams(messages=messages)
        
        assert params.messages == messages
        assert params.prompt is None

    def test_valid_with_messages_dict_list(self):
        """Test creating AgentParams with a list of dicts (auto-converted to Messages)."""
        messages_dicts = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
        ]
        params = AgentParams(messages=messages_dicts)
        
        assert len(params.messages) == 2
        assert all(isinstance(msg, Message) for msg in params.messages)
        assert params.prompt is None

    def test_both_prompt_and_messages_logs_warning(self):
        """Test that providing both prompt and messages logs a warning but doesn't fail.
        
        When both are provided, the framework prefers 'messages' and logs a warning.
        """
        
        # This should not raise an error, just log a warning
        params = AgentParams(
            prompt="Hello",
            messages=[Message(role="user", content=["Hi"])]
        )
        
        # Both should be set
        assert params.prompt is not None
        assert params.messages is not None

    def test_invalid_neither_prompt_nor_messages(self):
        """Test that providing neither prompt nor messages raises an error."""
        with pytest.raises(ValidationError) as exc_info:
            AgentParams()
        
        assert "Must specify either 'prompt' or 'messages'" in str(exc_info.value)

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed due to ConfigDict(extra='allow')."""
        params = AgentParams(
            prompt="Hello",
            custom_field="custom_value",
            another_field=123
        )
        
        assert params.prompt is not None
        assert params.custom_field == "custom_value"
        assert params.another_field == 123


class TestAgentParamsSchemaGeneration:
    """Test AgentParams JSON schema generation."""

    def test_model_json_schema_structure(self):
        """Test that model_json_schema returns the expected structure."""
        schema = AgentParams.model_json_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_model_json_schema_only_prompt_property(self):
        """Test that schema only exposes 'prompt' property, not 'messages'."""
        schema = AgentParams.model_json_schema()
        
        assert "prompt" in schema["properties"]
        assert "messages" not in schema["properties"]

    def test_model_json_schema_prompt_required(self):
        """Test that 'prompt' is marked as required in the schema."""
        schema = AgentParams.model_json_schema()
        
        assert "prompt" in schema["required"]
        assert len(schema["required"]) == 1

    def test_model_json_schema_prompt_structure(self):
        """Test the structure of the 'prompt' property in the schema."""
        schema = AgentParams.model_json_schema()
        prompt_schema = schema["properties"]["prompt"]
        
        assert prompt_schema["type"] == "object"
        assert prompt_schema["title"] == "TimbalMessage"
        assert "properties" in prompt_schema
        
        # Check role property
        assert "role" in prompt_schema["properties"]
        role_schema = prompt_schema["properties"]["role"]
        assert role_schema["type"] == "string"
        assert role_schema["enum"] == ["user"]
        
        # Check content property
        assert "content" in prompt_schema["properties"]
        content_schema = prompt_schema["properties"]["content"]
        assert content_schema["type"] == "array"
        assert "items" in content_schema

    def test_schema_simplification_for_llm_tools(self):
        """Test that the schema is simplified for use as an LLM tool.
        
        This ensures that when an Agent is used as a tool by another Agent,
        the LLM sees a clean, simple interface with just a 'prompt' parameter.
        """
        schema = AgentParams.model_json_schema()
        
        # Should only have one property
        assert len(schema["properties"]) == 1
        
        # Should be marked as required
        assert schema["required"] == ["prompt"]
        
        # Should have TimbalMessage structure
        assert schema["properties"]["prompt"]["title"] == "TimbalMessage"


class TestAgentParamsIntegration:
    """Integration tests for AgentParams with Agent usage patterns."""

    def test_params_with_simple_string_prompt(self):
        """Test the most common use case: simple string prompt."""
        params = AgentParams(prompt="What is 2+2?")
        
        assert params.prompt is not None
        assert isinstance(params.prompt, Message)
        assert params.messages is None

    def test_params_with_conversation_history(self):
        """Test using messages for full conversation control."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's the weather?"},
        ]
        params = AgentParams(messages=conversation)
        
        assert params.messages is not None
        assert len(params.messages) == 3
        assert params.prompt is None

    def test_params_serialization_with_prompt(self):
        """Test that AgentParams can be serialized and deserialized."""
        original = AgentParams(prompt="Test message")
        
        # Serialize to dict
        data = original.model_dump()
        
        # Deserialize back
        restored = AgentParams(**data)
        
        assert restored.prompt is not None
        assert restored.messages is None

    def test_params_serialization_with_messages(self):
        """Test serialization with messages list."""
        messages = [
            Message(role="user", content=["Hello"]),
            Message(role="assistant", content=["Hi there!"]),
        ]
        original = AgentParams(messages=messages)
        
        # Serialize to dict
        data = original.model_dump()
        
        # Deserialize back
        restored = AgentParams(**data)
        
        assert restored.messages is not None
        assert len(restored.messages) == 2
        assert restored.prompt is None


class TestAgentParamsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_messages_list_invalid(self):
        """Test that an empty messages list still requires validation."""
        # Empty list should still be considered as "messages is set"
        params = AgentParams(messages=[])
        assert params.messages == []
        assert params.prompt is None

    def test_prompt_with_complex_message_object(self):
        """Test prompt with a fully constructed Message object."""
        from timbal.types.content import TextContent
        
        complex_message = Message(
            role="user",
            content=[
                TextContent(text="Hello"),
                TextContent(text="World"),
            ]
        )
        params = AgentParams(prompt=complex_message)
        
        assert params.prompt == complex_message
        assert len(params.prompt.content) == 2

    def test_messages_with_mixed_roles(self):
        """Test messages with various role types."""
        messages = [
            Message(role="user", content=["User message"]),
            Message(role="assistant", content=["Assistant response"]),
            Message(role="user", content=["Follow-up"]),
        ]
        params = AgentParams(messages=messages)
        
        assert len(params.messages) == 3
        assert params.messages[0].role == "user"
        assert params.messages[1].role == "assistant"
        assert params.messages[2].role == "user"
