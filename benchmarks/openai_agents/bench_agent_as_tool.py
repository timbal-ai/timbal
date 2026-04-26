#!/usr/bin/env python3
"""
Timbal delegation vs OpenAI Agents SDK agent-as-tool benchmark.

This is the most direct OpenAI Agents orchestration comparison:
  supervisor -> worker-as-tool -> supervisor final answer

Run:
    uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_agent_as_tool.py
    uv run --with openai-agents --with 'openai>=2.26,<3' python benchmarks/openai_agents/bench_agent_as_tool.py --quick
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.openai_agents.bench_handoff import (
    BOLD,
    DIM,
    RESET,
    WIDTH,
    _collect,
    _make_timbal_delegation,
    _print_results,
)

HAS_OPENAI_AGENTS = False

try:
    from agents import Agent as OAIAgent
    from agents import RunConfig, Runner
    from agents.items import ModelResponse
    from agents.models.interface import Model as OAIModel
    from agents.usage import Usage
    from openai.types.responses import ResponseFunctionToolCall

    from benchmarks.openai_agents.bench_handoff import _text

    def _count_tool_outputs(input_items) -> int:
        if not isinstance(input_items, list):
            return 0
        return sum(1 for item in input_items if item.get("type") == "function_call_output")

    class _FakeAgentToolModel(OAIModel):
        def __init__(self, role: str) -> None:
            self.role = role

        async def get_response(
            self,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            *,
            previous_response_id,
            conversation_id,
            prompt,
        ) -> ModelResponse:
            _ = (
                system_instructions,
                model_settings,
                output_schema,
                handoffs,
                tracing,
                previous_response_id,
                conversation_id,
                prompt,
            )
            if self.role == "supervisor" and _count_tool_outputs(input) == 0:
                output = [
                    ResponseFunctionToolCall(
                        type="function_call",
                        id="fc_worker",
                        call_id="c1",
                        name=tools[0].name,
                        arguments=json.dumps({"input": "go"}),
                        status="completed",
                    )
                ]
            elif self.role == "supervisor":
                output = [_text("worker-done")]
            else:
                output = [_text("worker-done")]
            return ModelResponse(output=output, usage=Usage(), response_id=None)

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    def _make_oai_agent_as_tool() -> OAIAgent:
        run_config = RunConfig(tracing_disabled=False)
        worker = OAIAgent(name="worker", model=_FakeAgentToolModel("worker"))
        worker_tool = worker.as_tool(
            tool_name="worker_tool",
            tool_description="Delegate work to the worker agent.",
            run_config=run_config,
        )
        return OAIAgent(name="supervisor", model=_FakeAgentToolModel("supervisor"), tools=[worker_tool])

    HAS_OPENAI_AGENTS = True
except ImportError as e:
    print(f"OpenAI Agents SDK not found or incompatible: {e}")


async def main() -> None:
    print()
    print(f"{BOLD}{'=' * WIDTH}{RESET}")
    print(f"{BOLD}  Timbal delegation vs OpenAI Agents SDK agent-as-tool benchmark{RESET}")
    print(f"{BOLD}{'=' * WIDTH}{RESET}")

    t_agent = _make_timbal_delegation()
    t_res = await t_agent(prompt="go").collect()
    msg = f"  Timbal: {'OK' if t_res.output.collect_text() == 'worker-done' else 'FAIL'}"
    if HAS_OPENAI_AGENTS:
        oai_res = await Runner.run(_make_oai_agent_as_tool(), "go", run_config=RunConfig(tracing_disabled=False))
        msg += f"  |  OAI: {'OK' if str(oai_res.final_output) == 'worker-done' else 'FAIL'}"
    print(msg, flush=True)

    rows = [("Timbal delegation", await _collect("Timbal delegation", lambda: t_agent(prompt="go").collect()))]
    if HAS_OPENAI_AGENTS:
        oai_agent = _make_oai_agent_as_tool()
        rows.append(
            (
                "OAI agent tool",
                await _collect(
                    "OAI agent tool",
                    lambda: Runner.run(oai_agent, "go", run_config=RunConfig(tracing_disabled=False), max_turns=5),
                ),
            )
        )

    _print_results(rows)
    print()
    print(f"{DIM}{'-' * WIDTH}")
    print("  Both columns use supervisor -> worker-as-tool -> supervisor final answer.")
    print("  OpenAI Agents SDK uses Agent.as_tool(); Timbal uses an async worker-agent tool.")
    print(f"{'-' * WIDTH}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
