"""Regression coverage for home paths and literal tilde-prefixed workspace names."""

import sys
from pathlib import Path

import pytest
from timbal.evals.utils import _load_runnable, parse_eval_file
from timbal.utils.import_spec import ImportSpec


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    return home


def test_tilde_workspace_base(isolated_home):
    spec = ImportSpec.from_fqn("entry.py::handler", base_path=Path("~/project"))
    assert spec.path == isolated_home / "project" / "entry.py"


@pytest.mark.parametrize("name", ["~definitely_not_a_user_734.py", "~definitely_not_a_user_734/entry.py"])
def test_literal_tilde_name_in_workspace(tmp_path, name):
    spec = ImportSpec.from_fqn(name + "::handler", base_path=tmp_path)
    assert spec.path == tmp_path / name


def test_eval_home_runnable_loads_from_home(tmp_path, isolated_home):
    (isolated_home / "review_home_entry.py").write_text("handler = 42\n")
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    assert _load_runnable("~/review_home_entry.py::handler", eval_dir / "eval.yaml") == 42


@pytest.mark.parametrize("mode", ["relative", "absolute", "cwd", "literal_tilde"])
@pytest.mark.usefixtures("isolated_home")
def test_eval_existing_resolution_modes(tmp_path, monkeypatch, mode):
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    name = "~review_literal.py" if mode == "literal_tilde" else "review_control.py"
    target = (cwd if mode == "cwd" else eval_dir) / name
    target.write_text("handler = 77\n")
    supplied = str(target) if mode == "absolute" else name
    assert _load_runnable(supplied + "::handler", eval_dir / "eval.yaml") == 77


def test_parse_eval_home_runnable(tmp_path, isolated_home):
    module = isolated_home / "review_eval_agent.py"
    module.write_text(
        "from timbal import Agent\n"
        "from timbal.core.test_model import TestModel\n"
        'agent = Agent(name="review-agent", model=TestModel(responses=["ok"]), tools=[])\n'
    )
    path = tmp_path / "evals.yaml"
    path.write_text(
        "- name: home-runnable\n"
        "  runnable: ~/review_eval_agent.py::agent\n"
        "  params:\n"
        "    prompt: hello\n"
        "  output:\n"
        "    not_null!: true\n"
    )
    result = parse_eval_file(path)
    assert len(result) == 1
    assert result[0].runnable.name == "review-agent"
