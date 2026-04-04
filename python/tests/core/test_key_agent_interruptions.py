import asyncio

import pytest
from timbal import Agent, Tool
from timbal.tools import WebSearch
from timbal.types.content.tool_use import ToolUseContent
from timbal.types.events import OutputEvent
from timbal.types.events.delta import DeltaEvent
from timbal.types.message import Message

from ..conftest import assert_has_output_event

MODELS_TO_TEST = [
    pytest.param("anthropic/claude-haiku-4-5", 8092, id="anthropic-claude-haiku-4-5"),  # messages
    pytest.param("openai/gpt-5.2", None, id="openai-gpt-5.2"),  # responses
    pytest.param("google/gemini-2.5-flash-lite", None, id="google-gemini-2.5-flash-lite"),  # chat completions
]


async def _collect_and_cancel_on(agent, event_filter, timeout=30, **kwargs):
    """Run an agent via .collect() in a task and cancel when a specific event is observed.

    Uses a separate monitoring task that iterates events via the raw generator
    to watch for the trigger, while the main collection happens via .collect().

    Args:
        agent: The agent to run.
        event_filter: A callable(event) -> bool. Cancel after it returns True.
        timeout: Max seconds to wait for the event before failing the test.
        **kwargs: Passed to agent().

    Returns:
        The OutputEvent from the cancelled run.
    """
    triggered = asyncio.Event()
    event_queue = asyncio.Queue()

    gen = agent(**kwargs)

    async def _iterate():
        """Iterate the generator, forwarding events to the queue and watching for trigger."""
        async for event in gen:
            if not triggered.is_set() and event_filter(event):
                triggered.set()
            await event_queue.put(event)
        await event_queue.put(None)  # sentinel

    task = asyncio.create_task(_iterate())

    # Wait for the trigger or timeout. Also handle the case where the task
    # completes before we can cancel (fast models).
    try:
        await asyncio.wait_for(triggered.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        if not triggered.is_set() and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            pytest.fail(
                f"Event filter was not triggered within {timeout}s — "
                "agent may not have reached expected state"
            )

    # Cancel the iteration task if still running
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Drain the queue to find the OutputEvent (produced by the generator's finally block)
    result = None
    while not event_queue.empty():
        event = event_queue.get_nowait()
        if isinstance(event, OutputEvent):
            result = event

    # If no OutputEvent in queue yet, the finally block may still be running.
    # Give it a moment by closing the generator explicitly and checking again.
    if result is None:
        try:
            await gen.aclose()
        except Exception:
            pass

    assert result is not None, "Agent did not produce an OutputEvent after cancellation"
    return result


class TestKeyAgentInterruptions:
    """Test interrupting agents at key points during LLM generation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,max_tokens", MODELS_TO_TEST)
    async def test_llm_interrupt_and_continue(self, model: str, max_tokens: int | None):
        """
        Test interrupting an agent mid-LLM step and then continuing the conversation.

        Cancels after the first DeltaEvent (proves the LLM is streaming),
        then sends a follow-up and verifies the agent recovers.
        """
        agent_kwargs = {
            "name": f"continuable_agent_{model.replace('/', '_')}",
            "model": model,
        }
        if max_tokens:
            agent_kwargs["max_tokens"] = max_tokens

        agent = Agent(**agent_kwargs)

        prompt1 = (
            "Write a very long and detailed epic tale about a dragon who befriends a knight. "
            "Include extensive world-building, character backstories, multiple plot twists."
        )

        # Cancel after the first streaming chunk — proves LLM is mid-generation
        result1 = await _collect_and_cancel_on(
            agent,
            event_filter=lambda e: isinstance(e, DeltaEvent),
            prompt=prompt1,
        )

        assert result1.status.code == "cancelled", f"Expected 'cancelled' status, got '{result1.status.code}'"
        assert result1.output is not None, "Expected partial output from interrupted request"
        assert isinstance(result1.output, Message), f"Expected Message output, got {type(result1.output)}"
        assert result1.output.content, "Expected non-empty content from interrupted request"

        # Follow-up should complete successfully
        prompt2 = "Actually, just tell me: what is 2 + 2?"
        result2 = await agent(prompt=prompt2).collect()

        assert_has_output_event(result2)
        assert result2.status.code == "success", f"Expected 'success' status, got '{result2.status.code}'"
        assert result2.output is not None, "Expected output from the agent"
        assert isinstance(result2.output, Message), f"Expected Message output, got {type(result2.output)}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,max_tokens", MODELS_TO_TEST)
    async def test_server_tool_interrupt_and_continue(self, model: str, max_tokens: int | None):
        """
        Test interrupting an agent while a server-side tool (WebSearch) is running.

        WebSearch is only compatible with OpenAI Responses API and Anthropic.
        For Google (Chat Completions), we expect a ValueError.
        """
        agent_kwargs = {
            "name": f"websearch_agent_{model.replace('/', '_')}",
            "model": model,
            "tools": [WebSearch()],
        }
        if max_tokens:
            agent_kwargs["max_tokens"] = max_tokens

        agent = Agent(**agent_kwargs)

        # For Google (Chat Completions), WebSearch is not compatible - expect error
        if "google" in model:
            prompt1 = "Search the web for the latest news about artificial intelligence breakthroughs in 2026."
            result = await agent(prompt=prompt1).collect()
            assert result.status.code == "error", f"Expected 'error' status for Google, got '{result.status.code}'"
            assert result.error is not None, "Expected error for Google with WebSearch"
            assert "WebSearch is not compatible" in str(result.error), f"Expected WebSearch error, got {result.error}"
            return

        prompt1 = "When does Real Madrid play next?"

        # Cancel after the first DeltaEvent — the LLM is streaming
        result1 = await _collect_and_cancel_on(
            agent,
            event_filter=lambda e: isinstance(e, DeltaEvent),
            prompt=prompt1,
        )

        assert result1.status.code == "cancelled", f"Expected 'cancelled' status, got '{result1.status.code}'"
        assert result1.status.reason == "interrupted", f"Expected 'interrupted' reason, got '{result1.status.reason}'"
        assert result1.error is None, f"Expected no error, got {result1.error}"

        # Follow-up should complete successfully
        prompt2 = "Never mind the search. What is 3 + 3?"
        result2 = await agent(prompt=prompt2).collect()

        assert_has_output_event(result2)
        assert result2.status.code == "success", f"Expected 'success' status, got '{result2.status.code}'"
        assert result2.output is not None, "Expected output from the agent"
        assert isinstance(result2.output, Message), f"Expected Message output, got {type(result2.output)}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,max_tokens", MODELS_TO_TEST)
    async def test_immediate_interrupt_and_continue(self, model: str, max_tokens: int | None):
        """
        Test interrupting an agent almost immediately and then continuing.

        Cancels after the agent's own START event (before LLM has responded),
        then sends a follow-up.
        """
        agent_kwargs = {
            "name": f"immediate_agent_{model.replace('/', '_')}",
            "model": model,
        }
        if max_tokens:
            agent_kwargs["max_tokens"] = max_tokens

        agent = Agent(**agent_kwargs)

        prompt1 = "Tell me a story about a brave adventurer."

        # Cancel as quickly as possible. Unlike other tests, we don't wait
        # for a specific event — the point is to cancel early and verify
        # that the agent can still serve a follow-up regardless.
        task1 = asyncio.create_task(agent(prompt=prompt1).collect())
        await asyncio.sleep(0.1)
        task1.cancel()
        result1 = await task1

        # Fast models may complete before we cancel — that's fine.
        assert result1.status.code in ("cancelled", "success"), (
            f"Expected 'cancelled' or 'success' status, got '{result1.status.code}'"
        )
        if result1.status.code == "cancelled":
            assert result1.status.reason == "interrupted"
        assert result1.error is None, f"Expected no error, got {result1.error}"

        # Follow-up should complete successfully
        prompt2 = "What is 5 + 5?"
        result2 = await agent(prompt=prompt2).collect()

        assert_has_output_event(result2)
        assert result2.status.code == "success", f"Expected 'success' status, got '{result2.status.code}'"
        assert result2.output is not None, "Expected output from the agent"
        assert isinstance(result2.output, Message), f"Expected Message output, got {type(result2.output)}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model,max_tokens", MODELS_TO_TEST)
    async def test_tool_interrupt_and_continue(self, model: str, max_tokens: int | None):
        """
        Test interrupting an agent mid-tool execution and then continuing.

        Uses a slow tool and cancels after the tool's START event
        (proves the tool is running), then sends a follow-up.
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
        if max_tokens:
            agent_kwargs["max_tokens"] = max_tokens

        agent = Agent(**agent_kwargs)

        prompt1 = "Please search the internal database for information about project alpha."

        # Cancel after the tool's START event — proves the tool is running
        result1 = await _collect_and_cancel_on(
            agent,
            event_filter=lambda e: e.type == "START" and "internal_search" in e.path,
            prompt=prompt1,
        )

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

        # Follow-up should complete successfully
        prompt2 = "Never mind the search. What is 2 + 2?"
        result2 = await agent(prompt=prompt2).collect()

        assert_has_output_event(result2)
        assert result2.status.code == "success", f"Expected 'success' status, got '{result2.status.code}'"
        assert result2.output is not None, "Expected output from the agent"
        assert isinstance(result2.output, Message), f"Expected Message output, got {type(result2.output)}"
