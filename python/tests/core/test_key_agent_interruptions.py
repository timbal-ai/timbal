import asyncio
import os

# Configure logging to include DELTA events for this test file
os.environ["TIMBAL_LOG_EVENTS"] = "START,OUTPUT,DELTA"
os.environ["TIMBAL_DELTA_EVENTS"] = "true"

import pytest
from timbal import Agent, Tool
from timbal.tools import WebSearch
from timbal.types.content.tool_use import ToolUseContent
from timbal.types.message import Message

from .conftest import assert_has_output_event

MODELS_TO_TEST = [
    pytest.param("anthropic/claude-haiku-4-5", {"max_tokens": 8092}, id="anthropic-claude-haiku-4-5"),  # messages
    pytest.param("openai/gpt-5.2", {}, id="openai-gpt-5.2"),  # responses
    pytest.param("google/gemini-2.5-flash-lite", {}, id="google-gemini-2.5-flash-lite"),  # chat completions
]


class TestKeyAgentInterruptions:
    """Test interrupting agents at key points during LLM generation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,model_params", MODELS_TO_TEST)
    async def test_llm_interrupt_and_continue(self, model: str, model_params: dict):
        """
        Test interrupting an agent mid-LLM step and then continuing the conversation.

        This test:
        1. Creates an agent with the specified model
        2. Asks it to generate a long story
        3. Interrupts after 2 seconds
        4. Sends a follow-up message asking to continue
        5. Verifies the agent can respond to the follow-up
        """
        agent_kwargs = {
            "name": f"continuable_agent_{model.replace('/', '_')}",
            "model": model,
        }
        if model_params:
            agent_kwargs["model_params"] = model_params

        agent = Agent(**agent_kwargs)

        # First prompt - ask for a long story
        prompt1 = (
            "Write a very long and detailed epic tale about a dragon who befriends a knight. "
            "Include extensive world-building, character backstories, multiple plot twists."
        )

        # Start agent execution and interrupt it
        task1 = asyncio.create_task(agent(prompt=prompt1).collect())
        await asyncio.sleep(3)
        task1.cancel()
        result1 = await task1

        # Verify first request was interrupted
        assert result1.status.code == "cancelled", f"Expected 'cancelled' status, got '{result1.status.code}'"
        assert result1.output is not None, "Expected partial output from interrupted request"
        assert isinstance(result1.output, Message), f"Expected Message output, got {type(result1.output)}"
        assert result1.output.content, "Expected non-empty content from interrupted request"

        # Now send a follow-up message - the agent should be able to respond
        prompt2 = "Actually, just tell me: what is 2 + 2?"

        # This should complete successfully
        result2 = await agent(prompt=prompt2).collect()

        # Verify the agent recovered and responded
        assert_has_output_event(result2)
        assert result2.status.code == "success", f"Expected 'success' status, got '{result2.status.code}'"
        assert result2.output is not None, "Expected output from the agent"
        assert isinstance(result2.output, Message), f"Expected Message output, got {type(result2.output)}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,model_params", MODELS_TO_TEST)
    async def test_server_tool_interrupt_and_continue(self, model: str, model_params: dict):
        """
        Test interrupting an agent while a server-side tool (WebSearch) is running.

        WebSearch is only compatible with OpenAI Responses API and Anthropic.
        For Google (Chat Completions), we expect a ValueError.

        This test:
        1. Creates an agent with WebSearch tool
        2. Asks it to search for something that requires time
        3. Interrupts while the search is running
        4. Verifies the interruption
        5. Sends a follow-up message
        6. Verifies the agent can respond to the follow-up
        """
        agent_kwargs = {
            "name": f"websearch_agent_{model.replace('/', '_')}",
            "model": model,
            "tools": [WebSearch()],
        }
        if model_params:
            agent_kwargs["model_params"] = model_params

        agent = Agent(**agent_kwargs)

        # For Google (Chat Completions), WebSearch is not compatible - expect error
        if "google" in model:
            prompt1 = "Search the web for the latest news about artificial intelligence breakthroughs in 2026."
            result = await agent(prompt=prompt1).collect()
            assert result.status.code == "error", f"Expected 'error' status for Google, got '{result.status.code}'"
            assert result.error is not None, "Expected error for Google with WebSearch"
            assert "WebSearch is not compatible" in str(result.error), f"Expected WebSearch error, got {result.error}"
            return

        # First prompt - ask to search for something that takes time
        prompt1 = "When does Real Madrid play next?"

        # Start agent execution and interrupt it while search is running
        task1 = asyncio.create_task(agent(prompt=prompt1).collect())
        # Wait long enough for the LLM to decide to use web search and for the search to start
        await asyncio.sleep(2)
        task1.cancel()
        result1 = await task1

        # Verify first request was interrupted
        assert result1.status.code == "cancelled", f"Expected 'cancelled' status, got '{result1.status.code}'"
        assert result1.status.reason == "interrupted", f"Expected 'interrupted' reason, got '{result1.status.reason}'"
        assert result1.error is None, f"Expected no error, got {result1.error}"

        # Now send a follow-up message - the agent should be able to respond
        prompt2 = "Never mind the search. What is 3 + 3?"

        # This should complete successfully
        result2 = await agent(prompt=prompt2).collect()

        # Verify the agent recovered and responded
        assert_has_output_event(result2)
        assert result2.status.code == "success", f"Expected 'success' status, got '{result2.status.code}'"
        assert result2.output is not None, "Expected output from the agent"
        assert isinstance(result2.output, Message), f"Expected Message output, got {type(result2.output)}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,model_params", MODELS_TO_TEST)
    async def test_immediate_interrupt_and_continue(self, model: str, model_params: dict):
        """
        Test interrupting an agent almost immediately and then continuing the conversation.

        This test:
        1. Creates an agent with the specified model
        2. Starts a request
        3. Interrupts almost immediately (0.1 seconds)
        4. Verifies the interruption
        5. Sends a follow-up message
        6. Verifies the agent can respond to the follow-up
        """
        agent_kwargs = {
            "name": f"immediate_agent_{model.replace('/', '_')}",
            "model": model,
        }
        if model_params:
            agent_kwargs["model_params"] = model_params

        agent = Agent(**agent_kwargs)

        # First prompt
        prompt1 = "Tell me a story about a brave adventurer."

        # Start agent execution and interrupt almost immediately
        task1 = asyncio.create_task(agent(prompt=prompt1).collect())
        await asyncio.sleep(0.1)  # Very fast interruption
        task1.cancel()
        result1 = await task1

        # Verify first request was interrupted
        assert result1.status.code == "cancelled", f"Expected 'cancelled' status, got '{result1.status.code}'"
        assert result1.status.reason == "interrupted", f"Expected 'interrupted' reason, got '{result1.status.reason}'"
        assert result1.error is None, f"Expected no error, got {result1.error}"
        # Output may or may not be present depending on how fast the LLM responded

        # Now send a follow-up message - the agent should be able to respond
        prompt2 = "What is 5 + 5?"

        # This should complete successfully
        result2 = await agent(prompt=prompt2).collect()

        # Verify the agent recovered and responded
        assert_has_output_event(result2)
        assert result2.status.code == "success", f"Expected 'success' status, got '{result2.status.code}'"
        assert result2.output is not None, "Expected output from the agent"
        assert isinstance(result2.output, Message), f"Expected Message output, got {type(result2.output)}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,model_params", MODELS_TO_TEST)
    async def test_tool_interrupt_and_continue(self, model: str, model_params: dict):
        """
        Test interrupting an agent mid-tool execution and then continuing the conversation.

        This test:
        1. Creates an agent with a slow tool that simulates internal search
        2. Asks it to use the tool
        3. Interrupts while the tool is running
        4. Verifies the interruption output
        5. Sends a follow-up message
        6. Verifies the agent can respond to the follow-up
        """

        async def internal_search(query: str) -> str:
            """Searches the internal database for information."""
            await asyncio.sleep(10)
            return f"Found results for: {query}"

        internal_search_tool = Tool(
            name="internal_search",
            description="Searches the internal database for information. Use this tool when the user asks about internal data.",
            handler=internal_search,
        )

        agent_kwargs = {
            "name": f"tool_agent_{model.replace('/', '_')}",
            "model": model,
            "tools": [internal_search_tool],
        }
        if model_params:
            agent_kwargs["model_params"] = model_params

        agent = Agent(**agent_kwargs)

        # First prompt - ask to use the internal search tool
        prompt1 = "Please search the internal database for information about project alpha."

        # Start agent execution and interrupt it while tool is running
        task1 = asyncio.create_task(agent(prompt=prompt1).collect())
        # Wait long enough for the LLM to decide to use the tool and for the tool to start
        await asyncio.sleep(5)
        task1.cancel()
        result1 = await task1

        # Verify first request was interrupted
        assert result1.status.code == "cancelled", f"Expected 'cancelled' status, got '{result1.status.code}'"
        assert result1.status.reason == "interrupted", f"Expected 'interrupted' reason, got '{result1.status.reason}'"
        assert result1.error is None, f"Expected no error, got {result1.error}"

        # Verify the output contains the tool use that was in progress
        assert result1.output is not None, "Expected output from interrupted request"
        assert isinstance(result1.output, Message), f"Expected Message output, got {type(result1.output)}"
        assert result1.output.content, "Expected non-empty content from interrupted request"
        tool_use = None
        for content in result1.output.content:
            if isinstance(content, ToolUseContent):
                tool_use = content
        assert tool_use is not None, f"Expected ToolUseContent, got {type(tool_use)}"
        assert tool_use.name == "internal_search", f"Expected tool name 'internal_search', got '{tool_use.name}'"

        # Now send a follow-up message - the agent should be able to respond
        prompt2 = "Never mind the search. What is 2 + 2?"

        # This should complete successfully
        result2 = await agent(prompt=prompt2).collect()

        # Verify the agent recovered and responded
        assert_has_output_event(result2)
        assert result2.status.code == "success", f"Expected 'success' status, got '{result2.status.code}'"
        assert result2.output is not None, "Expected output from the agent"
        assert isinstance(result2.output, Message), f"Expected Message output, got {type(result2.output)}"
