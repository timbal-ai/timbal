"""Drop-in model replacement for testing agents without hitting a real LLM API."""

from collections.abc import AsyncGenerator, Callable
from typing import Any


class TestModel:
    __test__ = False  # prevent pytest from treating this as a test class
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

    provider: str = "test"
    model_name: str = "model"

    _collector_registered: bool = False

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

    async def stream(self, messages: list, **_kwargs: Any) -> AsyncGenerator:
        """Yield a single Message — no network call, no streaming.

        Implements the same async-generator interface as ``_llm_router`` so
        the agent loop can treat it identically to a real provider.
        """
        # Lazy imports keep SDK dependencies out of production import paths.
        from ..collectors import _collector_registry
        from ..collectors.impl.message import MessageCollector
        from ..state import get_or_create_run_context
        from ..types.content import TextContent
        from ..types.message import Message

        # Register once across all TestModel instances (class-level flag).
        if not TestModel._collector_registered:
            _collector_registry.register(MessageCollector)
            TestModel._collector_registered = True

        self.call_count += 1

        if self.handler is not None:
            raw = self.handler(messages)
        else:
            # Stateless step detection: count assistant messages already in the
            # conversation.  Each prior assistant reply = one prior LLM call
            # within this run.  Safe for concurrent asyncio.gather() on a
            # shared instance because each run has its own message history.
            step = sum(1 for m in messages if m.role == "assistant")
            idx = min(step, len(self.responses) - 1)
            raw = self.responses[idx]

        if isinstance(raw, str):
            response = Message(
                role="assistant",
                content=[TextContent(text=raw)],
                stop_reason="end_turn",
            )
        else:
            # Assume it's already a Message; default stop_reason so the agent loop terminates.
            if raw.stop_reason is None:
                raw = Message(role=raw.role, content=raw.content, stop_reason="end_turn")
            response = raw

        # Approximate token counts (1 token ≈ 4 chars) for UsageLimits compatibility.
        run_context = get_or_create_run_context()
        input_tokens = max(1, sum(len(str(c)) for m in messages for c in m.content) // 4)
        output_tokens = max(1, sum(len(str(c)) for c in response.content) // 4)
        run_context.update_usage("test/model:input_text_tokens", input_tokens)
        run_context.update_usage("test/model:output_text_tokens", output_tokens)

        yield response

    def __str__(self) -> str:
        return "test/model"

    def __repr__(self) -> str:
        return f"TestModel(call_count={self.call_count})"
