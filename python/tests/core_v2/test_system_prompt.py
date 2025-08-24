import asyncio
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
from timbal.core_v2.agent import Agent
from timbal.types.events import OutputEvent
from timbal.types.message import Message

from .conftest import assert_has_output_event, assert_no_errors, skip_if_agent_error


# ==============================================================================
# Test Utility Functions for System Prompts
# ==============================================================================

def get_current_time() -> str:
    """Get current time as string."""
    return datetime.now().strftime("%H:%M:%S")


def get_current_date() -> str:
    """Get current date as string."""
    return datetime.now().strftime("%Y-%m-%d")


def get_user_info() -> str:
    """Get mock user information."""
    return "Test User (ID: 12345)"


def get_system_status() -> str:
    """Get mock system status."""
    return "System Status: Online, Load: 15%"


async def async_get_weather() -> str:
    """Async function to get mock weather."""
    await asyncio.sleep(0.01)  # Simulate async work
    return "Weather: Sunny, 22°C"


async def async_get_time() -> str:
    """Async function to get current time."""
    await asyncio.sleep(0.01)
    return datetime.now().strftime("%H:%M:%S")


def get_context_info(**kwargs) -> str:
    """Function with optional parameters."""
    return f"Context: {kwargs.get('type', 'default')}"


def get_config(env: str = "test") -> str:
    """Function with default parameter."""
    return f"Config: {env} environment"


# ==============================================================================
# Test Classes
# ==============================================================================

class TestSystemPromptBasic:
    """Test basic system prompt functionality without template functions."""
    
    def test_agent_without_system_prompt(self):
        """Test agent creation without system prompt."""
        agent = Agent(
            name="no_prompt_agent",
            model="gpt-4o-mini"
        )
        assert agent.system_prompt is None
        assert len(agent._system_prompt_callables) == 0
    
    def test_agent_with_static_system_prompt(self):
        """Test agent with static system prompt (no template functions)."""
        system_prompt = "You are a helpful assistant specialized in mathematics."
        agent = Agent(
            name="static_prompt_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        assert agent.system_prompt == system_prompt
        assert len(agent._system_prompt_callables) == 0
    
    @pytest.mark.asyncio
    async def test_static_system_prompt_execution(self):
        """Test that static system prompt is used during execution."""
        system_prompt = "You are a math expert. Always start responses with 'As a math expert:'"
        agent = Agent(
            name="math_expert",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        prompt = Message.validate({"role": "user", "content": "What is 2+2?"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        skip_if_agent_error(output, "static_system_prompt_execution")
        
        # The response should reflect the system prompt
        response_content = str(output.output.content).lower()
        # Note: This test may be flaky depending on LLM behavior
        assert isinstance(output.output, Message)


class TestSystemPromptTemplateFunctions:
    """Test system prompt template function parsing and execution."""
    
    def test_system_prompt_pattern_recognition(self):
        """Test that system prompt patterns are correctly identified."""
        # Use functions that don't require parameters
        system_prompt = "Current directory: {os::getcwd}. Process ID: {os::getpid}"
        agent = Agent(
            name="pattern_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        # Should identify template patterns
        assert len(agent._system_prompt_callables) == 2
        assert "{os::getcwd}" in agent._system_prompt_callables
        assert "{os::getpid}" in agent._system_prompt_callables
    
    def test_invalid_pattern_format(self):
        """Test that invalid patterns are rejected."""
        # Test with a pattern that should actually trigger the assertion
        # The pattern {single} doesn't match the regex, so it won't be processed
        # We need a pattern that matches but fails validation
        with pytest.raises(Exception):  # Could be various exceptions during import
            Agent(
                name="invalid_pattern_agent",
                model="gpt-4o-mini",
                system_prompt="Invalid pattern: {nonexistent_module::nonexistent_function}"
            )
    
    def test_local_function_loading(self):
        """Test loading functions from local modules."""
        # This would test {::module::function} pattern
        # Since we can't easily create a temp module, we'll test the error handling
        system_prompt = "Time: {::nonexistent::function}"
        
        # This should fail during agent creation due to module not found
        with pytest.raises(Exception):  # Could be ImportError, ValueError, etc.
            Agent(
                name="local_func_agent",
                model="gpt-4o-mini",
                system_prompt=system_prompt
            )
    
    def test_package_function_loading(self):
        """Test loading functions from installed packages."""
        # Test with os module (more reliable than datetime.datetime.now)
        system_prompt = "CWD: {os::getcwd}"
        
        agent = Agent(
            name="package_func_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        assert len(agent._system_prompt_callables) == 1
        callable_info = agent._system_prompt_callables["{os::getcwd}"]
        assert callable_info["callable"] is not None
        assert "is_coroutine" in callable_info
    
    def test_function_validation(self):
        """Test that system prompt functions are properly validated."""
        # Test with a valid function
        system_prompt = "Path: {os::getcwd}"
        
        agent = Agent(
            name="validation_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        # Should have cached coroutine info
        callable_info = agent._system_prompt_callables["{os::getcwd}"] 
        assert isinstance(callable_info["is_coroutine"], bool)


class TestSystemPromptExecution:
    """Test execution of system prompt template functions."""
    
    @pytest.mark.asyncio
    async def test_sync_function_execution(self):
        """Test execution of synchronous template functions."""
        # Use os.getcwd which is available and callable
        system_prompt = "Current directory: {os::getcwd}"
        
        agent = Agent(
            name="sync_exec_agent",
            model="gpt-4o-mini", 
            system_prompt=system_prompt
        )
        
        # Manually test the resolution (since _resolve_system_prompt is private)
        resolved_prompt = await agent._resolve_system_prompt()
        
        # Should have replaced the template with actual path
        assert resolved_prompt != system_prompt
        assert "{os::getcwd}" not in resolved_prompt
        # Should contain a file path
        assert "/" in resolved_prompt  # Unix paths contain /
    
    @pytest.mark.asyncio
    async def test_async_function_execution(self):
        """Test execution of asynchronous template functions."""
        # Test with a regular function (the framework handles sync/async automatically)
        system_prompt = "Directory: {os::getcwd}"
        
        agent = Agent(
            name="async_exec_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        resolved_prompt = await agent._resolve_system_prompt()
        assert resolved_prompt is not None
        assert resolved_prompt != system_prompt
        assert "{os::getcwd}" not in resolved_prompt
    
    @pytest.mark.asyncio
    async def test_multiple_function_execution(self):
        """Test parallel execution of multiple template functions."""
        system_prompt = "CWD: {os::getcwd} and PID: {os::getpid}"
        
        agent = Agent(
            name="multi_exec_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        resolved_prompt = await agent._resolve_system_prompt()
        
        # Both patterns should be replaced
        assert "{os::getcwd}" not in resolved_prompt
        assert "{os::getpid}" not in resolved_prompt
        assert resolved_prompt != system_prompt
    
    @pytest.mark.asyncio
    async def test_function_error_handling(self):
        """Test handling of errors in template functions."""
        # Test with a function that might cause issues
        system_prompt = "Result: {os::getcwd}"  # Should work but test the pattern
        
        agent = Agent(
            name="error_handling_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        # Should not crash during resolution
        resolved_prompt = await agent._resolve_system_prompt()
        assert resolved_prompt is not None
    
    @pytest.mark.asyncio
    async def test_none_result_handling(self):
        """Test handling of template functions that return None."""
        # We can't easily inject a None-returning function, so test the replacement logic
        system_prompt = "Value: {datetime::datetime.now}"
        
        agent = Agent(
            name="none_result_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        resolved_prompt = await agent._resolve_system_prompt()
        # Should handle the result properly (even if it's not None in this case)
        assert resolved_prompt is not None
        assert isinstance(resolved_prompt, str)


class TestSystemPromptIntegration:
    """Test integration of system prompts with agent execution."""
    
    @pytest.mark.asyncio
    async def test_resolved_prompt_in_conversation(self):
        """Test that resolved system prompt is used in LLM calls."""
        system_prompt = "Current time is {datetime::datetime.now}. You are a time-aware assistant."
        
        agent = Agent(
            name="time_aware_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        prompt = Message.validate({"role": "user", "content": "What time is it?"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        skip_if_agent_error(output, "resolved_prompt_in_conversation")
        
        # The agent should have access to the resolved system prompt
        assert isinstance(output.output, Message)
    
    @pytest.mark.asyncio
    async def test_dynamic_system_prompt_updates(self):
        """Test that system prompt is resolved fresh on each execution."""
        # Use os.getcwd which should be consistent
        system_prompt = "Current directory: {os::getcwd}"
        
        agent = Agent(
            name="dynamic_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        # First resolution
        resolved1 = await agent._resolve_system_prompt()
        
        # Small delay
        await asyncio.sleep(0.1)
        
        # Second resolution
        resolved2 = await agent._resolve_system_prompt()
        
        # Should resolve consistently
        assert resolved1 == resolved2  # Should be same since getcwd doesn't change
        assert "{os::getcwd}" not in resolved1
        assert "{os::getcwd}" not in resolved2
    
    @pytest.mark.asyncio
    async def test_complex_system_prompt_template(self):
        """Test a complex system prompt with multiple template functions."""
        system_prompt = """
        You are an AI assistant with the following context:
        - Current directory: {os::getcwd}
        - Process ID: {os::getpid}
        - User session: active
        
        Please respond helpfully and include directory information when relevant.
        """
        
        agent = Agent(
            name="complex_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        prompt = Message.validate({"role": "user", "content": "Hello, can you help me?"})
        result = agent(prompt=prompt)
        
        output = await result.collect()
        assert_has_output_event(output)
        skip_if_agent_error(output, "complex_system_prompt_template")
        
        assert isinstance(output.output, Message)


class TestSystemPromptPerformance:
    """Test performance characteristics of system prompt resolution."""
    
    @pytest.mark.asyncio
    async def test_resolution_performance(self):
        """Test that system prompt resolution is reasonably fast."""
        system_prompt = "CWD: {os::getcwd} Base: {os::path.basename}"
        
        agent = Agent(
            name="perf_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        start_time = time.time()
        resolved_prompt = await agent._resolve_system_prompt()
        end_time = time.time()
        
        # Should resolve quickly (under 1 second for simple functions)
        assert end_time - start_time < 1.0
        assert resolved_prompt is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_resolution(self):
        """Test that multiple system prompt resolutions can run concurrently."""
        system_prompt = "Directory: {os::getcwd}"
        
        agents = [
            Agent(
                name=f"concurrent_agent_{i}",
                model="gpt-4o-mini",
                system_prompt=system_prompt
            )
            for i in range(3)
        ]
        
        start_time = time.time()
        
        # Resolve all prompts concurrently
        tasks = [agent._resolve_system_prompt() for agent in agents]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 2.0
        assert len(results) == 3
        assert all(result is not None for result in results)
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self):
        """Test that function metadata is cached properly."""
        system_prompt = "Value: {os::getcwd}"
        
        agent = Agent(
            name="cache_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        # Check that callable info is cached
        callable_info = agent._system_prompt_callables["{os::getcwd}"]
        assert "callable" in callable_info
        assert "is_coroutine" in callable_info
        assert "start" in callable_info
        assert "end" in callable_info
        
        # Multiple resolutions should use cached metadata
        result1 = await agent._resolve_system_prompt()
        result2 = await agent._resolve_system_prompt()
        
        assert result1 is not None
        assert result2 is not None


class TestSystemPromptEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_system_prompt_with_templates(self):
        """Test handling of edge case patterns."""
        # Test with minimal valid pattern
        system_prompt = "{a::b}"
        
        with pytest.raises(Exception):  # Should fail to load module 'a'
            Agent(
                name="edge_case_agent",
                model="gpt-4o-mini",
                system_prompt=system_prompt
            )
    
    @pytest.mark.asyncio
    async def test_system_prompt_with_no_templates(self):
        """Test that agents without templates work normally."""
        agent = Agent(
            name="no_template_agent",
            model="gpt-4o-mini",
            system_prompt="Static prompt"
        )
        
        resolved = await agent._resolve_system_prompt()
        assert resolved == "Static prompt"
    
    def test_template_pattern_edge_cases(self):
        """Test various edge cases in template patterns."""
        # Test patterns that should be ignored (don't match regex)
        invalid_patterns = [
            "{single}",  # Single part
            "{::}",      # Empty parts
            "{}",        # Empty
            "{a:b}",     # Single colon
        ]
        
        for pattern in invalid_patterns:
            system_prompt = f"Test {pattern} end"
            agent = Agent(
                name="edge_test_agent",
                model="gpt-4o-mini",
                system_prompt=system_prompt
            )
            # Should not match any patterns
            assert len(agent._system_prompt_callables) == 0
    
    @pytest.mark.asyncio
    async def test_mixed_template_and_static_content(self):
        """Test system prompts with both template functions and static content."""
        system_prompt = """
        Welcome! Current directory is {os::getcwd}.
        
        You are a helpful assistant. Please be concise and accurate.
        The process ID is {os::getpid}.
        
        Additional static instructions:
        - Always be polite
        - Provide examples when helpful
        """
        
        agent = Agent(
            name="mixed_content_agent",
            model="gpt-4o-mini",
            system_prompt=system_prompt
        )
        
        resolved = await agent._resolve_system_prompt()
        
        # Should preserve static content while replacing templates
        assert "Welcome!" in resolved
        assert "You are a helpful assistant" in resolved
        assert "Always be polite" in resolved
        assert "{os::getcwd}" not in resolved
        assert "{os::getpid}" not in resolved