"""Repeatable regression and agent comparison probe using the comparison's existing factories.

Run in fresh processes with each revision's python/ directory on PYTHONPATH.
The same interpreter, dependencies and this harness must be used for both.
Prints JSON; no LLM calls. Times execution only, with in-memory tracing enabled.
"""

# ruff: noqa: T201 - benchmark output

import argparse
import asyncio
import gc
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Run only a named case; repeat to select several")
    parser.add_argument("--skip-competitors", action="store_true")
    args = parser.parse_args()
    import timbal
    from timbal.state.tracing.providers import InMemoryTracingProvider

    agent = load("bench_agent")
    workflow = load("bench_workflow")
    parallel = load("bench_parallel")
    double = load("bench_double_fanout")
    await agent._verify_agent_loops()
    cases = [(f"agent_{n}", agent._make_timbal_agent(n), {"prompt": "go"}) for n in (1, 2, 3)]
    cases += [
        (f"dag_{name}", getattr(workflow, "_timbal_" + name)(), {"x": 5})
        for name in ("sequential", "fanout", "diamond")
    ]
    cases += [(f"parallel_{n}", parallel._timbal_wide(n), {"x": 5}) for n in (4, 8, 16, 32, 64)]
    cases += [(f"double_{n}", double._timbal_double(n), {"x": 5}) for n in (16, 32, 64, 128)]
    if args.case:
        cases = [case for case in cases if case[0] in args.case]
        if len(cases) != len(set(args.case)):
            parser.error("Unknown case name")
    results = {}
    for name, runnable, kwargs in cases:
        for _ in range(10):
            out = await agent._fresh_collect(runnable, **kwargs)
            assert out.status.code == "success", out.status
        InMemoryTracingProvider._storage.clear()
        gc.collect()
        samples = []
        for _ in range(200):
            start = time.perf_counter_ns()
            out = await agent._fresh_collect(runnable, **kwargs)
            samples.append((time.perf_counter_ns() - start) / 1000)
            assert out.status.code == "success", out.status
            assert out.run_id in InMemoryTracingProvider._storage, "Tracing was not enabled"
        InMemoryTracingProvider._storage.clear()
        results[name] = {
            "p50_us": statistics.median(samples),
            "p95_us": sorted(samples)[190],
            "mean_us": statistics.mean(samples),
            "samples_us": samples,
        }
    # Repeated comparison against both APIs, using the same model and tools.
    competitors = {}
    for api in () if args.skip_competitors else ("legacy", "current"):
        agent._args.agent_api = api
        await agent._verify_agent_loops()
        for scenario in (1, 2, 3):
            graph = agent._make_lc_graph(scenario)
            for _ in range(10):
                await graph.ainvoke(agent._LG_INPUT)
            gc.collect()
            samples = []
            for _ in range(100):
                start = time.perf_counter_ns()
                await graph.ainvoke(agent._LG_INPUT)
                samples.append((time.perf_counter_ns() - start) / 1000)
            competitors[f"{api}_agent_{scenario}"] = {
                "p50_us": statistics.median(samples),
                "p95_us": sorted(samples)[95],
                "samples_us": samples,
            }
    print(
        json.dumps(
            {"timbal_source": timbal.__file__, "python": sys.version, "results": results, "langgraph": competitors},
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
