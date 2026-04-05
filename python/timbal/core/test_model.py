"""Drop-in model replacement for testing agents without hitting a real LLM API."""

from collections.abc import Callable
from typing import Any


class TestModel:
    __test__ = False  # prevent pytest from treating this as a test class
    _is_test_model = True  # marker for duck-type checks — avoids importing this module at all
    """A drop-in model replacement for testing. Makes no network calls.

    Usage::

        agent = Agent(model=TestModel(responses=["Hello!", "The answer is 42."]))
        result = await agent.collect(prompt="hi")
        assert result.output.collect_text() == "Hello!"

        result = await agent.collect(prompt="what is 6x7")
        assert result.output.collect_text() == "The answer is 42."

    Responses cycle from the last entry once exhausted::

        model = TestModel(responses=["ok"])
        # Every call returns "ok"

    For tool-calling agents, pass a Message with ToolUseContent::

        from timbal.types.message import Message
        from timbal.types.content import ToolUseContent
        model = TestModel(responses=[
            Message(role="assistant", content=[
                ToolUseContent(id="call_1", name="add", input={"a": 1, "b": 2})
            ], stop_reason="tool_use"),
            "The result is 3.",
        ])

    Use a handler for dynamic, context-aware responses::

        def my_handler(messages):
            if "weather" in messages[-1].collect_text().lower():
                return "It's sunny."
            return "I don't know."

        agent = Agent(model=TestModel(handler=my_handler))

    Inspect call count after a run::

        model = TestModel(responses=["ok"])
        agent = Agent(model=model)
        await agent.collect(prompt="go")
        assert model.call_count == 1
    """

    def __init__(
        self,
        responses: list[Any] | None = None,
        handler: Callable[..., Any] | None = None,
    ) -> None:
        if responses is None and handler is None:
            responses = ["Test response."]
        if responses is not None and handler is not None:
            raise ValueError("Provide either 'responses' or 'handler', not both.")
        self.responses = responses
        self.handler = handler
        self.call_count: int = 0

    def __str__(self) -> str:
        return "test/model"

    def __repr__(self) -> str:
        return f"TestModel(call_count={self.call_count})"
