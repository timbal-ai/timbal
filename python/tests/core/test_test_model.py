"""Tests for TestModel — offline agent testing without LLM API calls."""

import asyncio

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


class TestTestModelBasics:
    def test_default_response(self):
        model = TestModel()
        assert model.responses == ["Test response."]
        assert model.call_count == 0

    def test_custom_responses(self):
        model = TestModel(responses=["hello", "world"])
        assert model.responses == ["hello", "world"]

    def test_handler(self):
        model = TestModel(handler=lambda msgs: "dynamic")
        assert model.handler is not None

    def test_cannot_provide_both(self):
        with pytest.raises(ValueError, match="not both"):
            TestModel(responses=["x"], handler=lambda msgs: "y")

    def test_str(self):
        assert str(TestModel()) == "test/model"


class TestTestModelWithAgent:
    @pytest.mark.asyncio
    async def test_fixed_response(self):
        model = TestModel(responses=["Hello from test!"])
        agent = Agent(name="a", model=model)
        result = await agent(prompt="hi").collect()
        assert result.output.collect_text() == "Hello from test!"
        assert model.call_count == 1

    @pytest.mark.asyncio
    async def test_step_detection_by_message_history(self):
        """Responses index by conversation step (assistant message count), not global counter.
        Each independent run starts at step 0 — same response every time."""
        model = TestModel(responses=["always this"])
        agent = Agent(name="a", model=model)

        r1 = await agent(prompt="first").collect()
        r2 = await agent(prompt="second").collect()
        r3 = await agent(prompt="third").collect()

        # Every independent run sees 0 assistant messages → step 0 → responses[0]
        assert r1.output.collect_text() == "always this"
        assert r2.output.collect_text() == "always this"
        assert r3.output.collect_text() == "always this"
        assert model.call_count == 3

    @pytest.mark.asyncio
    async def test_multi_step_responses_with_tools(self):
        """Responses list indexes by step within a single tool-calling run."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        model = TestModel(responses=[
            # Step 0: no assistant messages yet → call the tool
            Message(
                role="assistant",
                content=[ToolUseContent(id="c1", name="add", input={"a": 3, "b": 4})],
                stop_reason="tool_use",
            ),
            # Step 1: 1 assistant message in history → return final answer
            "3 + 4 = 7.",
        ])
        agent = Agent(name="a", model=model, tools=[add])
        result = await agent(prompt="what is 3 + 4?").collect()
        assert result.output.collect_text() == "3 + 4 = 7."
        assert model.call_count == 2

    @pytest.mark.asyncio
    async def test_handler_receives_messages(self):
        captured = []

        def my_handler(messages):
            captured.append(messages)
            return "captured!"

        agent = Agent(name="a", model=TestModel(handler=my_handler))
        result = await agent(prompt="hello").collect()
        assert result.output.collect_text() == "captured!"
        assert len(captured) == 1
        assert captured[0][-1].collect_text() == "hello"

    @pytest.mark.asyncio
    async def test_tool_call_sequence_with_handler(self):
        """Full agent loop via handler: model emits tool call, agent runs tool, model answers."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def tool_call_then_answer(messages):
            # Stateless step detection — count assistant messages
            step = sum(1 for m in messages if m.role == "assistant")
            if step == 0:
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="call_1", name="add", input={"a": 3, "b": 4})],
                    stop_reason="tool_use",
                )
            return "3 + 4 = 7."

        agent = Agent(name="a", model=TestModel(handler=tool_call_then_answer), tools=[add])
        result = await agent(prompt="what is 3 + 4?").collect()
        assert result.output.collect_text() == "3 + 4 = 7."
        assert agent.model.call_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_runs_on_shared_model(self):
        """Multiple concurrent runs on one shared TestModel instance — each gets correct responses."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        model = TestModel(responses=[
            Message(
                role="assistant",
                content=[ToolUseContent(id="c1", name="add", input={"a": 1, "b": 2})],
                stop_reason="tool_use",
            ),
            "done",
        ])
        agent = Agent(name="a", model=model, tools=[add])

        results = await asyncio.gather(
            agent(prompt="run 1").collect(),
            agent(prompt="run 2").collect(),
            agent(prompt="run 3").collect(),
        )

        # All three should complete successfully with the same final answer
        for r in results:
            assert r.output.collect_text() == "done"
            assert r.error is None

    @pytest.mark.asyncio
    async def test_no_api_keys_needed(self):
        """TestModel must work even when no API keys are present in the environment."""
        import os

        env_backup = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        try:
            agent = Agent(name="a", model=TestModel(responses=["works offline"]))
            result = await agent(prompt="hello").collect()
            assert result.output.collect_text() == "works offline"
        finally:
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v
