#!/usr/bin/env python3
# ruff: noqa: T201
"""
Timbal hooks vs Google ADK callback benchmark.

Both sides run a single-tool agent loop with no-op lifecycle callbacks enabled:
  prompt -> LLM -> add tool -> LLM -> answer

Timbal uses Runnable pre/post hooks on the agent and tool. Google ADK uses
before/after model callbacks and before/after tool callbacks.

Run:
    uv run --with google-adk python benchmarks/google_adk/bench_callbacks.py
    uv run --with google-adk python benchmarks/google_adk/bench_callbacks.py --quick
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path

logging.disable(logging.WARNING)
os.environ.setdefault("TIMBAL_LOG_LEVEL", "CRITICAL")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from timbal import Agent as TimbalAgent  # noqa: E402
from timbal.core.test_model import TestModel  # noqa: E402
from timbal.core.tool import Tool  # noqa: E402
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider  # noqa: E402
from timbal.types.content import TextContent, ToolResultContent, ToolUseContent  # noqa: E402
from timbal.types.message import Message  # noqa: E402

from benchmarks.google_adk.bench_transfer import (  # noqa: E402
    BOLD,
    DIM,
    HAS_ADK,
    RESET,
    WIDTH,
    collect,
    needed_runs,
    print_results,
)


def _noop() -> None:
    return None


def _clear_timbal_traces() -> None:
    InMemoryTracingProvider._storage.clear()


def _count_timbal_tool_results(messages) -> int:
    return sum(
        1
        for msg in messages
        for content in (msg.content if hasattr(msg, "content") and msg.content else [])
        if isinstance(content, ToolResultContent)
    )


def _make_timbal_hooked_agent() -> TimbalAgent:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def handler(messages):
        if _count_timbal_tool_results(messages) == 0:
            return Message(
                role="assistant",
                content=[ToolUseContent(type="tool_use", id="c1", name="add", input={"a": 1, "b": 2})],
            )
        return Message(role="assistant", content=[TextContent(type="text", text="3")], stop_reason="end_turn")

    tool = Tool(name="add", handler=add, pre_hook=_noop, post_hook=_noop)
    return TimbalAgent(
        name="hooked_agent",
        model=TestModel(handler=handler),
        tools=[tool],
        pre_hook=_noop,
        post_hook=_noop,
    )


async def _run_timbal(agent: TimbalAgent) -> str:
    _clear_timbal_traces()
    result = await agent(prompt="calculate").collect()
    return result.output.collect_text()


if HAS_ADK:
    from google.adk.agents import Agent as AdkAgent  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.models.base_llm import BaseLlm  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.models.llm_response import LlmResponse  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.runners import Runner  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.adk.sessions import InMemorySessionService  # pyright: ignore[reportMissingImports]  # noqa: E402
    from google.genai import types as genai_types  # noqa: E402

    def _count_adk_function_responses(contents) -> int:
        return sum(
            1
            for content in (contents or [])
            for part in (content.parts or [])
            if getattr(part, "function_response", None)
        )

    def _before_model_callback(*args, **kwargs):
        del args, kwargs
        return None

    def _after_model_callback(*args, **kwargs):
        del args, kwargs
        return None

    def _before_tool_callback(*args, **kwargs):
        del args, kwargs
        return None

    def _after_tool_callback(*args, **kwargs):
        del args, kwargs
        return None

    class _FakeCallbackLlm(BaseLlm):
        def supported_models(self) -> list[str]:
            return ["fake-adk"]

        async def generate_content_async(self, llm_request, stream: bool = False):
            del stream
            if _count_adk_function_responses(llm_request.contents) == 0:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part.from_function_call(name="add", args={"a": 1, "b": 2})],
                    )
                )
            else:
                yield LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part.from_text(text="3")],
                    )
                )

    class AdkCallbackBench:
        def __init__(self) -> None:
            self.app_name = "google_adk_callback_bench"
            self.user_id = "bench_user"
            self.session_service = InMemorySessionService()

            def add(a: int, b: int) -> int:
                """Add two numbers."""
                return a + b

            agent = AdkAgent(
                name="callback_agent",
                model=_FakeCallbackLlm(model="fake-adk"),
                instruction="Use the available tool to calculate the answer.",
                tools=[add],
                before_model_callback=_before_model_callback,
                after_model_callback=_after_model_callback,
                before_tool_callback=_before_tool_callback,
                after_tool_callback=_after_tool_callback,
            )
            self.runner = Runner(app_name=self.app_name, agent=agent, session_service=self.session_service)
            self._sessions: list[str] = []

        async def prepare(self, n: int) -> None:
            for _ in range(n):
                session = await self.session_service.create_session(app_name=self.app_name, user_id=self.user_id)
                self._sessions.append(session.id)

        async def run(self) -> str:
            session_id = self._sessions.pop()
            final_text = ""
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=genai_types.Content(role="user", parts=[genai_types.Part.from_text(text="calculate")]),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = event.content.parts[0].text or ""
            return final_text


async def main() -> None:
    print()
    print(f"{BOLD}{'=' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal hooks vs Google ADK callbacks benchmark{RESET}")
    print(f"{BOLD}{'=' * WIDTH}{RESET}")

    t_agent = _make_timbal_hooked_agent()
    msg = f"  Timbal: {'OK' if await _run_timbal(t_agent) == '3' else 'FAIL'}"

    adk_bench = None
    if HAS_ADK:
        adk_bench = AdkCallbackBench()
        await adk_bench.prepare(needed_runs() + 1)
        msg += f"  |  ADK: {'OK' if await adk_bench.run() == '3' else 'FAIL'}"
    print(msg, flush=True)

    rows = [("Timbal hooks", await collect("Timbal hooks", lambda: _run_timbal(t_agent)))]
    if adk_bench:
        rows.append(("ADK callbacks", await collect("ADK callbacks", adk_bench.run)))

    print_results("Callback-enabled single-tool loop", rows)
    print()
    print(f"{DIM}{'-' * WIDTH}")
    print("  Both columns run no-op lifecycle callbacks around model/tool execution.")
    print("  This measures callback-enabled agent-loop overhead, not workflow scheduling.")
    print(f"{'-' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
