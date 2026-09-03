"""Tests for utils/import_spec.py — ImportSpec.from_fqn() and load()."""

import sys
import textwrap
import types
from pathlib import Path

import pytest
from timbal.utils.import_spec import ImportSpec


@pytest.fixture
def workforce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A flat workforce dir with an entry module and a sibling helper.

    The helper is only imported *inside* the handler, so the import runs
    after ``load()`` has returned — the shape that broke in production.
    cwd is moved away so the tmp dir can't leak onto ``sys.path`` via ``''``.
    """
    (tmp_path / "helper.py").write_text('VALUE = "from-helper"\n')
    (tmp_path / "main.py").write_text(
        textwrap.dedent(
            """
            import top_level_helper


            def handler() -> str:
                from helper import VALUE  # lazy sibling import

                return f"{top_level_helper.TAG}:{VALUE}"
            """
        )
    )
    (tmp_path / "top_level_helper.py").write_text('TAG = "top"\n')
    monkeypatch.chdir(tmp_path.parent)
    yield tmp_path
    for name in ("main", "helper", "top_level_helper"):
        sys.modules.pop(name, None)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


class TestFromFqn:
    def test_parses_path_and_target(self, tmp_path: Path):
        spec = ImportSpec.from_fqn("main.py::handler", base_path=tmp_path)
        assert spec.path == (tmp_path / "main.py").resolve()
        assert spec.target == "handler"

    def test_rejects_missing_separator(self):
        with pytest.raises(ValueError, match="Expected format"):
            ImportSpec.from_fqn("main.py")


class TestLoad:
    def test_returns_target(self, workforce: Path):
        handler = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        assert callable(handler)

    def test_module_dir_stays_on_sys_path(self, workforce: Path):
        assert str(workforce) not in sys.path
        ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        assert sys.path[0] == str(workforce)

    def test_lazy_sibling_import_resolves_after_load(self, workforce: Path):
        handler = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        # `helper` was never imported at module top-level; this is its first import.
        assert "helper" not in sys.modules
        assert handler() == "top:from-helper"

    def test_lazy_sibling_import_fails_once_module_dir_leaves_sys_path(self, workforce: Path):
        """Negative control reproducing the pre-fix behaviour, where load()
        removed the module dir in a `finally`. Proves the fixture is not
        rescued by cwd / '' on sys.path — the lookup really rides on the
        entry it leaves behind."""
        handler = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        if str(workforce) in sys.path:
            sys.path.remove(str(workforce))
        with pytest.raises(ModuleNotFoundError, match="No module named 'helper'"):
            handler()

    def test_does_not_duplicate_existing_sys_path_entry(self, workforce: Path):
        sys.path.insert(0, str(workforce))
        ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        assert sys.path.count(str(workforce)) == 1

    def test_missing_target_raises(self, workforce: Path):
        with pytest.raises(ValueError, match="has no target"):
            ImportSpec.from_fqn("main.py::nope", base_path=workforce).load()


class TestSysModulesRegistration:
    def test_entry_module_is_registered_under_its_stem(self, workforce: Path):
        handler = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        assert sys.modules["main"].handler is handler

    def test_sibling_import_of_entry_does_not_reexecute_it(self, workforce: Path):
        """``from main import X`` inside a sibling must hit the cached module.

        Without registration it re-runs the entry file: a second Agent built,
        tools/tracing set up twice, isinstance failing across the two copies.
        """
        (workforce / "exec_log.py").write_text("RUNS: list[str] = []\n")
        (workforce / "main.py").write_text(
            textwrap.dedent(
                """
                import exec_log

                exec_log.RUNS.append("main")
                CONST = object()


                def handler():
                    from helper import get_const

                    return get_const()
                """
            )
        )
        (workforce / "helper.py").write_text(
            textwrap.dedent(
                """
                def get_const():
                    from main import CONST

                    return CONST
                """
            )
        )
        try:
            handler = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
            assert handler() is sys.modules["main"].CONST
            assert sys.modules["exec_log"].RUNS == ["main"]
        finally:
            sys.modules.pop("exec_log", None)

    def test_dataclass_with_future_annotations_in_entry(self, workforce: Path):
        """Registration must happen *before* exec, not after.

        ``dataclasses`` resolves string annotations through
        ``sys.modules[cls.__module__].__dict__`` at class-creation time; with
        the entry module unregistered that is ``None.__dict__`` — an
        AttributeError on every supported Python for any ``@dataclass`` in an
        entry file using ``from __future__ import annotations``.
        """
        (workforce / "main.py").write_text(
            textwrap.dedent(
                """
                from __future__ import annotations

                from dataclasses import dataclass


                @dataclass
                class Point:
                    x: int


                def handler():
                    return Point(1)
                """
            )
        )
        handler = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        assert handler().x == 1

    def test_reload_replaces_own_registration(self, workforce: Path):
        first = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        second = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        assert first is not second
        assert sys.modules["main"].handler is second

    def test_never_clobbers_a_foreign_module(self, workforce: Path, monkeypatch: pytest.MonkeyPatch):
        foreign = types.ModuleType("main")
        monkeypatch.setitem(sys.modules, "main", foreign)
        handler = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        assert callable(handler)
        assert sys.modules["main"] is foreign

    def test_failed_exec_leaves_no_registration(self, workforce: Path):
        (workforce / "broken.py").write_text("raise RuntimeError('boom')\n")
        with pytest.raises(RuntimeError, match="boom"):
            ImportSpec.from_fqn("broken.py::x", base_path=workforce).load()
        assert "broken" not in sys.modules

    def test_failed_reload_restores_previous_registration(self, workforce: Path):
        good = ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        (workforce / "main.py").write_text("raise RuntimeError('boom')\n")
        with pytest.raises(RuntimeError, match="boom"):
            ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        assert sys.modules["main"].handler is good
