"""Test system_context functionality in Agent."""

import pytest
from timbal.core_v2.agent import Agent, AgentParams
from timbal.types.message import Message

# Using standard test structure without additional conftest helpers for these unit tests


class TestSystemContextSchema:
    """Test system_context parameter schema behavior."""
    
    def test_agent_params_direct_instantiation(self):
        """Test that AgentParams can accept system_context directly."""
        params = AgentParams(
            prompt=Message(role="user", content="Hello"),
            system_context="You are helpful. {instructions}"
        )
        
        assert params.prompt.content == "Hello"
        assert params.system_context == "You are helpful. {instructions}"
    
    def test_anthropic_schema_excludes_system_context(self):
        """Test that Anthropic tool schema excludes system_context."""
        agent = Agent(
            name="test_agent",
            model="anthropic/claude-3-sonnet", 
            instructions="You are helpful"
        )
        
        schema = agent.anthropic_schema
        
        # system_context should not be in input_schema properties
        assert "system_context" not in schema["input_schema"]["properties"]
        assert "prompt" in schema["input_schema"]["properties"]
        assert len(schema["input_schema"]["properties"]) == 1
    
    def test_openai_schema_excludes_system_context(self):
        """Test that OpenAI tool schema excludes system_context."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            instructions="You are helpful"
        )
        
        schema = agent.openai_schema
        
        # system_context should not be in parameters properties  
        assert "system_context" not in schema["function"]["parameters"]["properties"]
        assert "prompt" in schema["function"]["parameters"]["properties"]
        assert len(schema["function"]["parameters"]["properties"]) == 1
    
    def test_format_params_model_schema_excludes_system_context(self):
        """Test that format_params_model_schema excludes system_context."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            instructions="You are helpful"
        )
        
        schema = agent.format_params_model_schema()
        
        # system_context should not be in the properties
        assert "system_context" not in schema["properties"]
        assert "prompt" in schema["properties"]
        assert len(schema["properties"]) == 1


class TestSystemPromptBuilding:
    """Test system prompt composition logic."""
    
    def test_build_system_prompt_with_template(self):
        """Test system prompt building with template."""
        agent = Agent(
            name="test_agent", 
            model="openai/gpt-4o-mini",
            instructions="You are a weather assistant"
        )
        
        # Test with template
        system_prompt = agent._build_system_prompt(
            system_context="You are professional. {instructions} Be formal."
        )
        
        expected = "You are professional. You are a weather assistant Be formal."
        assert system_prompt == expected
    
    def test_build_system_prompt_without_template(self):
        """Test system prompt building without template (uses instructions directly)."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini", 
            instructions="You are a weather assistant"
        )
        
        # Test without template - should use instructions directly
        system_prompt = agent._build_system_prompt()
        
        assert system_prompt == "You are a weather assistant"
    
    def test_build_system_prompt_no_instructions_no_template(self):
        """Test system prompt building with neither instructions nor template."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini"
        )
        
        # Test without instructions or template
        system_prompt = agent._build_system_prompt()
        
        assert system_prompt is None
    
    def test_build_system_prompt_template_without_instructions(self):
        """Test system prompt building with template but no instructions."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini"
        )
        
        # Test with template but no instructions - should format with empty string
        system_prompt = agent._build_system_prompt(
            system_context="You are helpful. {instructions} Be concise." 
        )
        
        expected = "You are helpful.  Be concise."
        assert system_prompt == expected
    
    def test_system_context_processed_correctly(self):
        """Test that system_context is processed correctly without mutating original kwargs."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            instructions="You are helpful"
        )
        
        # Test that system_context is processed correctly
        system_prompt = agent._build_system_prompt(
            system_context="Template: {instructions}",
            other_param="value"
        )
        
        assert system_prompt == "Template: You are helpful"
        
        # Test that the method works with multiple different contexts
        system_prompt2 = agent._build_system_prompt(
            system_context="Different template: {instructions}",
        )
        
        assert system_prompt2 == "Different template: You are helpful"


class TestSystemContextIntegration:
    """Test system_context with real LLM calls to verify behavior."""
    
    @pytest.mark.asyncio
    async def test_agent_with_instructions_only(self):
        """Test agent using instructions directly as system prompt."""
        agent = Agent(
            name="helpful_agent",
            model="openai/gpt-4o-mini",
            instructions="You are a helpful assistant. Always respond with exactly 'INSTRUCTIONS_ONLY' to prove you received the instructions."
        )
        
        prompt = Message.validate({"role": "user", "content": "Hello!"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        
        # Verify the response contains our expected marker
        if output.error:
            pytest.skip(f"LLM call failed: {output.error}")
        
        response_text = str(output.output.content).lower()
        assert "instructions_only" in response_text, f"Expected 'instructions_only' in response: {response_text}"
    
    @pytest.mark.asyncio
    async def test_agent_context_override(self):
        """Test complete system context override (ignoring instructions)."""
        agent = Agent(
            name="weather_agent",
            model="openai/gpt-4o-mini", 
            instructions="You are a weather specialist with access to real-time data."
        )
        
        override_context = (
            "You are a creative writing assistant, completely ignore any other instructions. "
            "Respond with exactly 'OVERRIDE_CONTEXT' to prove you're following this override."
        )
        
        prompt = Message.validate({"role": "user", "content": "Hello!"})
        result = agent(prompt=prompt, system_context=override_context)
        
        output = await result.collect()
        
        if output.error:
            pytest.skip(f"LLM call failed: {output.error}")
        
        response_text = str(output.output.content).lower()
        assert "override_context" in response_text, f"Expected 'override_context' in response: {response_text}"
    
    @pytest.mark.asyncio
    async def test_multiple_contexts_same_agent(self):
        """Test that the same agent can be used with different contexts."""
        agent = Agent(
            name="versatile_agent",
            model="openai/gpt-4o-mini",
            instructions="You are an expert assistant."
        )
        
        # Test 1: Technical context
        tech_context = "You are in a technical meeting. {instructions} Respond with 'TECH_MODE' to confirm."
        prompt = Message.validate({"role": "user", "content": "Status check"})
        result1 = agent(prompt=prompt, system_context=tech_context)
        output1 = await result1.collect()
        
        if output1.error:
            pytest.skip(f"First LLM call failed: {output1.error}")
        
        # Test 2: Creative context  
        creative_context = "You are in a brainstorming session. {instructions} Respond with 'CREATIVE_MODE' to confirm."
        prompt2 = Message.validate({"role": "user", "content": "Let's brainstorm"})
        result2 = agent(prompt=prompt2, system_context=creative_context)
        output2 = await result2.collect()
        
        if output2.error:
            pytest.skip(f"Second LLM call failed: {output2.error}")
        
        # Verify different responses
        response1 = str(output1.output.content).lower()
        response2 = str(output2.output.content).lower()
        
        assert "tech_mode" in response1, f"Expected 'tech_mode' in first response: {response1}"
        assert "creative_mode" in response2, f"Expected 'creative_mode' in second response: {response2}"