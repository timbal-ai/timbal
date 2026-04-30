"""Tests for preserving partial output and memory across interrupted agent runs.

The voice session scenario: an Agent streams a response, the user barges in
(interrupt), and the next turn should see the partial assistant reply in its
conversation history.

Two interruption vectors:
1. ``asyncio.CancelledError`` — task.cancel() while the agent is streaming.
2. ``GeneratorExit`` — consumer breaks out of ``async for`` (closing the generator).

Both must snapshot the collector's partial result into span.output so that
``resolve_memory`` on the next chained run includes the assistant message.
"""

import asyncio
from collections.abc import AsyncGenerator

from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state import set_run_context
from timbal.state.context import RunContext
from timbal.types.events import OutputEvent
from timbal.types.message import Message


class _SlowPostYieldModel(TestModel):
    """TestModel that yields the Message immediately but then sleeps before returning.

    This simulates a streaming model that has finished producing content but the
    generator hasn't closed yet.  The MessageCollector captures the Message on
    the first yield, so collector.result() returns a valid Message even if
    CancelledError fires during the post-yield sleep.
    """

    __test__ = False

    def __init__(self, responses=None, handler=None, post_yield_delay: float = 1.0):
        super().__init__(responses=responses, handler=handler)
        self._post_yield_delay = post_yield_delay

    async def stream(self, messages: list, **kwargs) -> AsyncGenerator:
        async for item in super().stream(messages, **kwargs):
            yield item
        # Simulate streaming tail — CancelledError can fire here
        await asyncio.sleep(self._post_yield_delay)


class TestInterruptedAgentPreservesOutput:
    """When an Agent run is interrupted mid-stream, span.output must contain
    the partial assistant Message (not None)."""

    async def test_cancelled_agent_has_partial_output(self):
        """CancelledError during streaming → Agent OutputEvent.output is the partial Message."""
        model = _SlowPostYieldModel(responses=["Hello! How can I help you today?"], post_yield_delay=2.0)
        agent = Agent(name="test_agent", model=model)

        prompt = Message.validate({"role": "user", "content": "Hi"})
        task = asyncio.create_task(agent(prompt=prompt).collect())

        # Wait long enough for the Message to be yielded but cancel before post-yield sleep finishes
        await asyncio.sleep(0.3)
        task.cancel()
        result = await task

        assert isinstance(result, OutputEvent)
        assert result.status.code == "cancelled"
        assert result.output is not None, (
            "Interrupted agent must preserve partial output in span. "
            "Got output=None — collector.result() was not snapshotted."
        )
        assert isinstance(result.output, Message)

    async def test_generator_exit_agent_has_partial_output(self):
        """Breaking out of async for after LLM OutputEvent → Agent span has output + memory."""
        model = TestModel(responses=["Hello! How can I help you today?"])
        agent = Agent(name="test_agent", model=model)

        run_context = RunContext()
        set_run_context(run_context)

        prompt = Message.validate({"role": "user", "content": "Hi"})

        # Break right after the LLM OutputEvent (event 3), before Agent OutputEvent
        collected = []
        async for event in agent(prompt=prompt):
            collected.append(event)
            if isinstance(event, OutputEvent) and event.path.endswith(".llm"):
                break

        # Allow the async generator finalizer to run
        await asyncio.sleep(0.1)

        agent_spans = [s for s in run_context._trace.values() if s.path == "test_agent"]
        assert len(agent_spans) == 1
        span = agent_spans[0]
        assert span.status is not None, "span.status must not be None after GeneratorExit"
        assert span.status.code == "cancelled"
        # Memory should include the assistant Message (appended by handler before yield,
        # or salvaged by _salvage_interrupted_llm_output)
        assert span.memory is not None
        roles = [m.role for m in span.memory]
        assert "assistant" in roles, (
            f"Agent memory must include assistant reply after GeneratorExit. "
            f"Got roles: {roles}."
        )


class TestInterruptedMemoryChaining:
    """Multi-turn: after an interrupted first turn, the second turn's memory
    must include the partial assistant reply from turn 1."""

    async def test_memory_includes_assistant_reply_after_normal_turn(self):
        """Turn 1 completes → Turn 2 messages include the assistant reply from turn 1."""
        call_count = [0]
        captured_messages = [None]

        def spy_handler(messages):
            call_count[0] += 1
            captured_messages[0] = list(messages)
            if call_count[0] == 1:
                return "Hello! I'm doing great, thank you for asking."
            return "Sure, 2 + 2 = 4."

        model = TestModel(handler=spy_handler)
        agent = Agent(name="voice_agent", model=model)

        # --- Turn 1: complete normally ---
        run_context = RunContext()
        set_run_context(run_context)

        prompt1 = Message.validate({"role": "user", "content": "Hello, how are you?"})
        result1 = await agent(prompt=prompt1).collect()
        assert result1.status.code == "success"
        assert result1.output is not None

        # --- Turn 2: chained via parent_id from turn 1's run ---
        turn2_ctx = RunContext(parent_id=result1.run_id)
        set_run_context(turn2_ctx)
        prompt2 = Message.validate({"role": "user", "content": "What is 2 + 2?"})
        result2 = await agent(prompt=prompt2).collect()

        assert result2.status.code == "success"
        assert captured_messages[0] is not None
        turn2_messages = captured_messages[0]
        roles = [m.role for m in turn2_messages]
        assert "assistant" in roles, (
            f"Turn 2 memory must include assistant reply from turn 1. "
            f"Got roles: {roles}"
        )

    async def test_cancelled_turn_memory_still_chains(self):
        """Turn 1 cancelled mid-stream → Turn 2 memory includes partial assistant reply."""

        call_count = [0]
        captured_messages = [None]

        def spy_handler(messages):
            call_count[0] += 1
            if call_count[0] == 2:
                captured_messages[0] = list(messages)
            return "Some response text here."

        model = _SlowPostYieldModel(handler=spy_handler, post_yield_delay=2.0)
        agent = Agent(name="voice_agent", model=model)

        # --- Turn 1: cancel after Message yielded but before generator closes ---
        prompt1 = Message.validate({"role": "user", "content": "Hello"})
        task = asyncio.create_task(agent(prompt=prompt1).collect())
        await asyncio.sleep(0.3)
        task.cancel()
        result1 = await task

        assert result1.status.code == "cancelled"
        turn1_run_id = result1.run_id

        # --- Turn 2: chained via parent_id from turn 1 ---
        turn2_ctx = RunContext(parent_id=turn1_run_id)
        set_run_context(turn2_ctx)

        fast_model = TestModel(handler=spy_handler)
        agent2 = Agent(name="voice_agent", model=fast_model)
        prompt2 = Message.validate({"role": "user", "content": "What is 2+2?"})
        await agent2(prompt=prompt2).collect()

        assert captured_messages[0] is not None, "Turn 2 handler should have been called"
        roles = [m.role for m in captured_messages[0]]
        assert "assistant" in roles, (
            f"Turn 2 memory must include (partial) assistant reply from cancelled turn 1. "
            f"Got roles: {roles}. This means resolve_memory lost the interrupted turn."
        )

    async def test_generator_exit_turn_memory_chains(self):
        """Turn 1 interrupted via break after LLM OutputEvent → Turn 2 has assistant reply."""
        call_count = [0]
        captured_messages = [None]

        def spy_handler(messages):
            call_count[0] += 1
            if call_count[0] == 2:
                captured_messages[0] = list(messages)
            if call_count[0] == 1:
                return "Hello! I'm doing great."
            return "2 + 2 = 4."

        model = TestModel(handler=spy_handler)
        agent = Agent(name="voice_agent", model=model)

        run_context = RunContext()
        set_run_context(run_context)

        # --- Turn 1: break after LLM OutputEvent ---
        prompt1 = Message.validate({"role": "user", "content": "Hello"})
        turn1_run_id = None
        async for event in agent(prompt=prompt1):
            if isinstance(event, OutputEvent) and event.path.endswith(".llm"):
                turn1_run_id = event.run_id
                break
        await asyncio.sleep(0.1)

        assert turn1_run_id is not None

        # --- Turn 2: chained via parent_id ---
        # Clear stale call_id/parent_call_id from turn 1's interrupted generator
        from timbal.state import set_call_id, set_parent_call_id
        set_call_id(None)
        set_parent_call_id(None)

        turn2_ctx = RunContext(parent_id=turn1_run_id)
        set_run_context(turn2_ctx)
        prompt2 = Message.validate({"role": "user", "content": "What is 2+2?"})
        await agent(prompt=prompt2).collect()

        assert captured_messages[0] is not None, "Turn 2 handler should have been called"
        roles = [m.role for m in captured_messages[0]]
        assert "assistant" in roles, (
            f"Turn 2 memory must include assistant reply from GeneratorExit-interrupted turn 1. "
            f"Got roles: {roles}."
        )
