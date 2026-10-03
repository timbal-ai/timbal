"""Catalog tooling must work with Windows' legacy default text encoding."""

import builtins
import importlib.util
import io
import sys
from pathlib import Path

import pytest
import yaml
from timbal.codegen import model_discovery
from timbal.core import models

ROOT = Path(__file__).resolve().parents[3]


def _script(name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def windows_encoding(monkeypatch):
    original_open = io.open

    def cp1252_open(file, mode="r", buffering=-1, encoding=None, errors=None,
                    newline=None, closefd=True, opener=None):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(io, "open", cp1252_open)
    monkeypatch.setattr(builtins, "open", cp1252_open)


def test_offline_audit_reads_utf8_docs(monkeypatch):
    audit = _script("audit_models", monkeypatch)
    catalog = audit._load_models()
    assert audit._check_models_py_sync(catalog) == []
    assert audit._check_docs_sync(catalog) == []


def test_runtime_and_discovery_preserve_unicode():
    expected = yaml.safe_load((ROOT / "python/timbal/models.yaml").read_text(encoding="utf-8"))["models"]
    models._load_models.cache_clear()
    try:
        assert models._load_models() == {model["id"]: model for model in expected}
        assert model_discovery.get_models() == expected
    finally:
        models._load_models.cache_clear()


def test_model_generation_preserves_utf8_source(monkeypatch, tmp_path):
    generate = _script("generate_models", monkeypatch)
    source = '# Unicode comment: “模型”\n# Model type with provider prefixes\nModel = Literal["old"]\n'
    dest = tmp_path / "models.py"
    dest.write_text(source, encoding="utf-8")
    monkeypatch.setattr(generate, "MODELS_PY", dest)
    generate.main()
    result = dest.read_text(encoding="utf-8")
    assert result.startswith('# Unicode comment: “模型”\n')
    assert '"openai/gpt-6.1-sol"' in result


def test_cost_generation_preserves_utf8(monkeypatch, tmp_path):
    generate = _script("generate_costs_sql", monkeypatch)
    catalog = tmp_path / "models.yaml"
    catalog.write_text(yaml.safe_dump({"models": [{
        "id": "test/model", "provider": "test", "display_name": "“模型”",
        "input_price": 1, "output_price": 2,
    }]}, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(generate, "MODELS_YAML", catalog)
    dest = tmp_path / "costs.sql"
    generate.main(output_path=dest)
    assert "“模型”" in dest.read_text(encoding="utf-8")
