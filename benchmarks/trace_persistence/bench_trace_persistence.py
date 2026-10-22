"""Turn latency and upload volume of synchronous vs buffered trace persistence.

    uv run python benchmarks/trace_persistence/bench_trace_persistence.py
    uv run python benchmarks/trace_persistence/bench_trace_persistence.py --rounds 25 --payload-kb 60 --runs 5
    uv run python benchmarks/trace_persistence/bench_trace_persistence.py --fail-for-s 3   # a platform outage

A scripted agent (no network) makes ``--rounds`` model calls of ``--model-ms`` each; every call asks for
``--tools`` parallel tool calls whose results are ``--payload-kb`` each, so the trace grows like a long
coding turn. The provider stores a real JSON snapshot of the trace (serialization is measured), then
waits like the platform's trace PATCH: ``--base-ms`` plus ``--ms-per-mb`` per MB sent. The defaults come
from the monolith's request log for composer trace PATCHes (about 50 ms under 0.2 MB, about 1 s at 5 MB).
With ``--fail-for-s``, every store fails for that long after the run starts.

For each mode it prints p50/p95 of the run's seconds, the seconds it waited on saves, saves, MB sent and
serialization seconds.
"""

# ruff: noqa: T201 - a benchmark reports on stdout
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time

from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state import RunContext, get_run_context, set_run_context
from timbal.state.tracing.providers.base import TracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


class TimedModel(TestModel):
    def __init__(self, *, model_s: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_s = model_s

    async def stream(self, messages, **kwargs):
        await asyncio.sleep(self.model_s)
        async for chunk in super().stream(messages, **kwargs):
            yield chunk


def make_platform(args: argparse.Namespace, started: dict[str, float]):
    class Platform(TracingProvider):
        saves = 0
        failures = 0
        bytes_sent = 0
        serialize_s = 0.0

        @classmethod
        async def get(cls, _run_context):
            return None

        @classmethod
        async def _store(cls, run_context):
            t0 = time.perf_counter()
            body = json.dumps(run_context._trace.model_dump()).encode()
            cls.serialize_s += time.perf_counter() - t0
            await asyncio.sleep((args.base_ms + args.ms_per_mb * len(body) / 1e6) / 1000)
            if time.monotonic() - started["at"] < args.fail_for_s:
                cls.failures += 1
                raise RuntimeError("platform unavailable")
            cls.saves += 1
            cls.bytes_sent += len(body)

    return Platform


async def one_run(args: argparse.Namespace, mode: str) -> dict[str, float]:
    started = {"at": time.monotonic()}
    platform = make_platform(args, started).configured(
        durability=mode, flush_interval_s=args.flush_ms / 1000, max_lag_s=args.max_lag_s
    )
    payload = "x" * (args.payload_kb * 1024)

    async def lookup(query: str) -> str:
        return f"{query}: {payload}"

    responses: list = [
        Message(
            role="assistant",
            content=[
                ToolUseContent(id=f"c{r}_{i}", name="lookup", input={"query": f"{r}-{i}"}) for i in range(args.tools)
            ],
        )
        for r in range(args.rounds)
    ]
    original = get_run_context()
    try:
        set_run_context(RunContext(tracing_provider=platform))
        agent = Agent(
            name="bench",
            model=TimedModel(model_s=args.model_ms / 1000, responses=[*responses, "done"]),
            tools=[lookup],
            tracing_provider=platform,
        )
        started["at"] = time.monotonic()
        output = await agent(prompt="go").collect()
        seconds = time.monotonic() - started["at"]
    finally:
        set_run_context(original)
    assert output.status.code == "success", output.error
    return {
        "seconds": seconds,
        "waited": output.usage.get("tracing:wait_ms", 0) / 1000,
        "saves": platform.saves,
        "failures": platform.failures,
        "mb": platform.bytes_sent / 1e6,
        "serialize": platform.serialize_s,
    }


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--tools", type=int, default=2)
    parser.add_argument("--payload-kb", type=int, default=40)
    parser.add_argument("--model-ms", type=float, default=300.0)
    parser.add_argument("--base-ms", type=float, default=50.0)
    parser.add_argument("--ms-per-mb", type=float, default=200.0)
    parser.add_argument("--flush-ms", type=float, default=500.0)
    parser.add_argument("--max-lag-s", type=float, default=30.0)
    parser.add_argument("--fail-for-s", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    # Sync mode writes no tracing counters unless asked; the comparison needs its wait too.
    os.environ["TIMBAL_TRACE_METRICS"] = "1"

    print(
        f"{args.rounds} model calls x {args.tools} tools, {args.payload_kb} KB per result, {args.model_ms:.0f} ms "
        f"per model call; store {args.base_ms:.0f} ms + {args.ms_per_mb:.0f} ms/MB; flush {args.flush_ms:.0f} ms"
        + (f"; stores fail for the first {args.fail_for_s:.1f} s" if args.fail_for_s else "")
    )
    print(
        f"{'mode':9} {'run p50':>8} {'run p95':>8} {'waited':>7} {'saves':>6} {'failed':>6} {'MB sent':>8} {'serialize':>9}"
    )
    for mode in ("sync", "buffered"):
        runs = [await one_run(args, mode) for _ in range(args.runs)]
        seconds = [r["seconds"] for r in runs]
        mean = {k: statistics.mean(r[k] for r in runs) for k in ("waited", "saves", "failures", "mb", "serialize")}
        print(
            f"{mode:9} {pct(seconds, 0.5):7.2f}s {pct(seconds, 0.95):7.2f}s {mean['waited']:6.2f}s "
            f"{mean['saves']:6.0f} {mean['failures']:6.0f} {mean['mb']:8.1f} {mean['serialize']:8.2f}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
