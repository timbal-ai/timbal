"""Tests for utils/import_spec.py — ImportSpec.from_fqn() and load()."""

import sys
import textwrap
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

    def test_does_not_duplicate_existing_sys_path_entry(self, workforce: Path):
        sys.path.insert(0, str(workforce))
        ImportSpec.from_fqn("main.py::handler", base_path=workforce).load()
        assert sys.path.count(str(workforce)) == 1

    def test_missing_target_raises(self, workforce: Path):
        with pytest.raises(ValueError, match="has no target"):
            ImportSpec.from_fqn("main.py::nope", base_path=workforce).load()
