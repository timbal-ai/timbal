import json
import subprocess
import sys
from pathlib import Path

import pytest
from timbal.evals.models import EvalSummary
from timbal.evals.runner import run_eval
from timbal.evals.utils import discover_config, discover_eval_files, dump_summary, parse_eval_file

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


def touch(path: Path, contents: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


class TestDiscoverEvalFiles:
    def test_missing_path_returns_empty(self, tmp_path):
        assert discover_eval_files(tmp_path / "nope") == []

    def test_file_path_returned_as_is(self, tmp_path):
        f = touch(tmp_path / "anything.yaml")
        assert discover_eval_files(f) == [f]

    def test_non_yaml_file_raises(self, tmp_path):
        f = touch(tmp_path / "evals.json")
        with pytest.raises(ValueError, match="Invalid evals path"):
            discover_eval_files(f)

    def test_matches_eval_prefix_and_suffix(self, tmp_path):
        prefix = touch(tmp_path / "evals_simple.yaml")
        suffix = touch(tmp_path / "smoke_eval.yaml")
        touch(tmp_path / "config.yaml")
        touch(tmp_path / "evalconf.txt")
        assert discover_eval_files(tmp_path) == sorted([prefix, suffix])

    def test_recurses_into_subdirectories(self, tmp_path):
        nested = touch(tmp_path / "evals" / "nested" / "eval_agent.yaml")
        assert discover_eval_files(tmp_path) == [nested]

    def test_skips_hidden_directories(self, tmp_path):
        touch(tmp_path / ".venv" / "lib" / "site-packages" / "timbal" / "evals" / "fixtures" / "eval_examples.yaml")
        touch(tmp_path / ".git" / "eval_leftover.yaml")
        visible = touch(tmp_path / "eval_mine.yaml")
        assert discover_eval_files(tmp_path) == [visible]

    def test_skips_non_hidden_virtualenvs(self, tmp_path):
        touch(tmp_path / "venv" / "pyvenv.cfg")
        touch(tmp_path / "venv" / "lib" / "eval_examples.yaml")
        visible = touch(tmp_path / "eval_mine.yaml")
        assert discover_eval_files(tmp_path) == [visible]

    def test_skips_node_modules_pycache_and_site_packages(self, tmp_path):
        touch(tmp_path / "node_modules" / "some-pkg" / "eval_examples.yaml")
        touch(tmp_path / "__pycache__" / "eval_cached.yaml")
        touch(tmp_path / "site-packages" / "timbal" / "eval_fixtures.yaml")
        visible = touch(tmp_path / "eval_mine.yaml")
        assert discover_eval_files(tmp_path) == [visible]

    def test_regular_subdirectories_are_still_scanned(self, tmp_path):
        a = touch(tmp_path / "evals" / "eval_a.yaml")
        b = touch(tmp_path / "more_evals" / "deep" / "b_eval.yaml")
        assert discover_eval_files(tmp_path) == sorted([a, b])


class TestParseEvalFileRunnableResolution:
    def test_runnable_relative_to_eval_file_dir(self, tmp_path):
        touch(tmp_path / "agent.py", AGENT_MODULE)
        eval_file = touch(tmp_path / "evals_simple.yaml", EVAL_FILE)

        evals = parse_eval_file(eval_file)

        assert len(evals) == 1
        assert evals[0].name == "greeting_test"

    def test_runnable_falls_back_to_cwd(self, tmp_path, monkeypatch):
        """Eval files in an evals/ subdirectory can reference project-root modules."""
        touch(tmp_path / "agent.py", AGENT_MODULE)
        eval_file = touch(tmp_path / "evals" / "evals_simple.yaml", EVAL_FILE)
        monkeypatch.chdir(tmp_path)

        evals = parse_eval_file(eval_file)

        assert len(evals) == 1
        assert evals[0].name == "greeting_test"

    def test_missing_runnable_raises(self, tmp_path):
        eval_file = touch(tmp_path / "evals" / "evals_simple.yaml", EVAL_FILE)

        with pytest.raises(ValueError, match="Failed to load runnable"):
            parse_eval_file(eval_file)

    def test_no_runnable_anywhere_raises(self, tmp_path):
        eval_file = touch(
            tmp_path / "evals_simple.yaml",
            "- name: greeting_test\n  params:\n    prompt: hi\n",
        )

        with pytest.raises(ValueError, match="No runnable specified"):
            parse_eval_file(eval_file)


class TestDumpSummary:
    async def test_dump_summary_is_json_serializable(self, tmp_path):
        touch(tmp_path / "agent.py", AGENT_MODULE)
        eval_file = touch(tmp_path / "evals_simple.yaml", EVAL_FILE)
        evals = parse_eval_file(eval_file)

        result = await run_eval(evals[0], capture=True)
        summary = EvalSummary(results=[result])

        data = await dump_summary(summary)
        json.dumps(data)  # must not raise

        assert data["total"] == 1
        assert data["passed"] == 1
        assert data["failed"] == 0
        r = data["results"][0]
        assert r["name"] == "greeting_test"
        assert r["passed"] is True
        assert r["error"] is None
        assert r["output"] is not None
        assert len(r["validators"]) == 1
        assert r["validators"][0]["name"] == "not_null!"
        assert r["validators"][0]["passed"] is True


class TestCliJsonOutput:
    def test_stdout_is_pure_json_with_dash_output(self, tmp_path):
        """With `-o -`, stdout must contain only the JSON document; human output goes to stderr."""
        touch(tmp_path / "agent.py", AGENT_MODULE)
        touch(tmp_path / "evals_simple.yaml", EVAL_FILE)

        proc = subprocess.run(
            [sys.executable, "-m", "timbal.evals.cli", str(tmp_path), "-o", "-"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=60,
        )

        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)  # must parse without any surrounding noise
        assert data["total"] == 1
        assert data["passed"] == 1
        assert "Timbal Evals" in proc.stderr


class TestDiscoverConfig:
    def test_walks_up_to_find_config(self, tmp_path):
        touch(tmp_path / "evalconf.yaml", "runnable: agent.py::agent\n")
        nested = tmp_path / "evals" / "nested"
        nested.mkdir(parents=True)

        assert discover_config(nested) == {"runnable": "agent.py::agent"}

    def test_returns_empty_when_missing(self, tmp_path):
        assert discover_config(tmp_path) == {}
