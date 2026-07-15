import asyncio
import json
import subprocess
import sys
from pathlib import Path

from timbal.core.tool import Tool
from timbal.evals.models import Eval, EvalResult
from timbal.evals.reporters import JsonReporter, Reporter
from timbal.evals.runner import run_evals

AGENT_MODULE = """\
from timbal import Agent
from timbal.core.test_model import TestModel

agent = Agent(name="agent", model=TestModel(responses=["ok"]), tools=[])
"""

EVAL_FILE = """\
- name: greeting_test
  runnable: agent.py::agent
  params:
    prompt: "Hi there!"
  output:
    not_null!: true
"""


def make_eval(name: str, delay: float, text: str) -> Eval:
    async def handler() -> str:
        await asyncio.sleep(delay)
        print(text)  # noqa: T201 — this output is what the capture tests assert on
        return text

    tool = Tool(name=name, handler=handler)
    return Eval.model_validate({"path": Path(f"{name}.yaml"), "name": name, "runnable": tool})


class RecordingReporter(Reporter):
    def __init__(self) -> None:
        self.order: list[str] = []

    async def result(self, result: EvalResult) -> None:
        self.order.append(result.eval.name)


class TestSequentialRun:
    async def test_results_in_file_order(self):
        evals = [make_eval("slow", 0.05, "slow out"), make_eval("fast", 0.0, "fast out")]
        reporter = RecordingReporter()

        summary = await run_evals(evals, reporter=reporter)

        assert reporter.order == ["slow", "fast"]
        assert summary.passed == 2

    async def test_capture_attributes_output_per_eval(self):
        evals = [make_eval("a", 0.0, "output from a"), make_eval("b", 0.0, "output from b")]

        summary = await run_evals(evals, reporter=Reporter())

        by_name = {r.eval.name: r for r in summary.results}
        assert "output from a" in by_name["a"].captured_stdout
        assert "output from b" not in by_name["a"].captured_stdout
        assert "output from b" in by_name["b"].captured_stdout


class TestParallelRun:
    async def test_results_stream_in_completion_order(self):
        evals = [make_eval("slow", 0.3, "slow out"), make_eval("fast", 0.01, "fast out")]
        reporter = RecordingReporter()

        summary = await run_evals(evals, reporter=reporter, max_concurrency=2)

        assert reporter.order == ["fast", "slow"]
        assert summary.passed == 2

    async def test_parallel_capture_does_not_interleave(self):
        evals = [make_eval(f"eval_{i}", 0.01 * (i % 3), f"output from eval_{i}") for i in range(6)]

        summary = await run_evals(evals, reporter=Reporter(), max_concurrency=4)

        assert summary.total == 6
        for result in summary.results:
            name = result.eval.name
            assert f"output from {name}\n" in result.captured_stdout
            other_names = {e.eval.name for e in summary.results} - {name}
            for other in other_names:
                assert f"output from {other}\n" not in result.captured_stdout

    async def test_parallel_restores_std_streams(self):
        stdout_before, stderr_before = sys.stdout, sys.stderr
        evals = [make_eval("a", 0.0, "a out"), make_eval("b", 0.0, "b out")]

        await run_evals(evals, reporter=Reporter(), max_concurrency=2)

        assert sys.stdout is stdout_before
        assert sys.stderr is stderr_before

    async def test_parallel_is_actually_concurrent(self):
        import time

        evals = [make_eval(f"e{i}", 0.2, "out") for i in range(4)]
        t0 = time.perf_counter()

        await run_evals(evals, reporter=Reporter(), max_concurrency=4)

        elapsed = time.perf_counter() - t0
        # Sequential would take >= 0.8s; parallel should be well under that.
        assert elapsed < 0.6


class TestJsonReporterStream:
    async def test_emits_start_result_summary(self):
        import io

        stream = io.StringIO()
        reporter = JsonReporter(stream=stream)
        evals = [make_eval("a", 0.0, "a out"), make_eval("b", 0.0, "b out")]

        await run_evals(evals, reporter=reporter, max_concurrency=2)

        lines = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert lines[0]["event"] == "start"
        assert lines[0]["total"] == 2
        result_events = [line for line in lines if line["event"] == "result"]
        assert {e["name"] for e in result_events} == {"a", "b"}
        assert all(e["passed"] for e in result_events)
        assert lines[-1]["event"] == "summary"
        assert lines[-1]["passed"] == 2


class TestCliJsonFormat:
    def test_format_json_streams_jsonl_to_stdout(self, tmp_path):
        (tmp_path / "agent.py").write_text(AGENT_MODULE)
        (tmp_path / "evals_simple.yaml").write_text(EVAL_FILE)

        proc = subprocess.run(
            [sys.executable, "-m", "timbal.evals.cli", str(tmp_path), "--format", "json", "-j", "2"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=60,
        )

        assert proc.returncode == 0, proc.stderr
        lines = [json.loads(line) for line in proc.stdout.splitlines()]
        assert [line["event"] for line in lines] == ["start", "result", "summary"]
        assert lines[1]["name"] == "greeting_test"
        assert lines[1]["passed"] is True

    def test_format_json_emits_events_when_no_evals_found(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "timbal.evals.cli", str(tmp_path), "--format", "json"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=60,
        )

        assert proc.returncode == 0, proc.stderr
        lines = [json.loads(line) for line in proc.stdout.splitlines()]
        assert [line["event"] for line in lines] == ["start", "summary"]
        assert lines[0]["total"] == 0
        assert lines[1] == {"event": "summary", "total": 0, "passed": 0, "failed": 0, "total_duration": 0.0}

    def test_dash_output_writes_document_when_no_evals_found(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "timbal.evals.cli", str(tmp_path), "-o", "-"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=60,
        )

        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data["total"] == 0
        assert data["results"] == []
        assert "No evals found" in proc.stderr

    def test_format_json_with_dash_output_is_rejected(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "timbal.evals.cli", str(tmp_path), "--format", "json", "-o", "-"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=60,
        )

        assert proc.returncode == 1


class TestCodegenEvalsOperation:
    def test_runs_evals_against_workspace_entry_point(self, tmp_path):
        (tmp_path / "agent.py").write_text(AGENT_MODULE)
        (tmp_path / "timbal.yaml").write_text('fqn: "agent.py::agent"\n')
        # No 'runnable' key: the workspace entry point must be used as default.
        (tmp_path / "evals_simple.yaml").write_text(
            "- name: greeting_test\n"
            "  params:\n"
            '    prompt: "Hi there!"\n'
            "  output:\n"
            "    not_null!: true\n"
        )

        proc = subprocess.run(
            [sys.executable, "-m", "timbal.codegen", "--path", str(tmp_path), "evals"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=60,
        )

        assert proc.returncode == 0, proc.stderr
        lines = [json.loads(line) for line in proc.stdout.splitlines()]
        assert [line["event"] for line in lines] == ["start", "result", "summary"]
        assert lines[1]["name"] == "greeting_test"
        assert lines[1]["passed"] is True

    def test_missing_timbal_yaml_errors(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "timbal.codegen", "--path", str(tmp_path), "evals"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=60,
        )

        assert proc.returncode == 1
        assert "timbal.yaml not found" in proc.stderr
