"""Offline checks for recurring documentation failures; never execute snippets."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit_docs", ROOT / "scripts/audit_docs.py")
assert spec and spec.loader
audit_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_docs)


@pytest.fixture
def docs_fixture(tmp_path):
    docs = tmp_path / "docs"
    (docs / "models").mkdir(parents=True)
    (docs / "api-reference").mkdir()
    (docs / "docs.json").write_text(
        json.dumps({"navigation": {"pages": ["index", "models/example"]}}),
        encoding="utf-8",
    )
    (docs / "api-reference/openapi.json").write_text('{"paths": {}}', encoding="utf-8")
    (docs / "index.mdx").write_text(
        '---\ntitle: "Café 🎙️"\n---\n[Models](/models/example)\n```python\nfrom timbal import Agent\n```\n',
        encoding="utf-8",
    )
    (docs / "models/example.mdx").write_text(
        '---\ntitle: "Models"\n---\n<Card>\n`openai/example`\n'
        "&#36;2 input / &#36;10 output; cached input &#36;0.1\n"
        "Above 272K: &#36;4 / &#36;15\n</Card>\n",
        encoding="utf-8",
    )
    catalog = tmp_path / "models.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "openai/example",
                        "input_price": 2,
                        "output_price": 10,
                        "cached_input_price": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return docs, catalog


def test_audit_accepts_unicode_and_base_price_before_long_context(docs_fixture):
    errors, counts = audit_docs.audit(*docs_fixture, runtime=True)
    assert errors == []
    assert counts["price_pairs"] == 1
    assert counts["cache_prices"] == 1


def test_audit_reads_third_price_in_input_output_cache_triple(docs_fixture):
    docs, catalog = docs_fixture
    (docs / "models/example.mdx").write_text(
        '---\ntitle: "Models"\n---\n<Card>\n`openai/example`\n'
        "Standard input / output / cached input: &#36;2 / &#36;10 / &#36;0.1\n</Card>\n",
        encoding="utf-8",
    )
    errors, counts = audit_docs.audit(docs, catalog)
    assert errors == []
    assert counts["cache_prices"] == 1


@pytest.mark.parametrize(
    ("bad_content", "expected"),
    [
        ("```python\nagent = Agent(\n```", "Python syntax"),
        ("```python\nfrom timbal.platform.kbs.tables import query\n```", "cannot import"),
        ("```python\nfrom timbal import NotAnExport\n```", "missing import"),
        ('```python\nprint("unterminated fence")', "unclosed code fence"),
        ("[Gone](/missing-page)", "broken local link"),
        ('```python\nmodel="openai/retired-model"\n```', "model not in catalog"),
    ],
)
def test_audit_rejects_broken_examples(docs_fixture, bad_content, expected):
    docs, catalog = docs_fixture
    (docs / "index.mdx").write_text('---\ntitle: "Example"\n---\n' + bad_content, encoding="utf-8")
    errors, _ = audit_docs.audit(docs, catalog, runtime=True)
    assert any(expected in error for error in errors)


def test_audit_rejects_navigation_api_and_price_drift(docs_fixture):
    docs, catalog = docs_fixture
    (docs / "index.mdx").write_text(
        '---\ntitle: "API"\nopenapi: "POST /removed"\n---\n',
        encoding="utf-8",
    )
    (docs / "orphan.mdx").write_text('---\ntitle: "Orphan"\n---\n', encoding="utf-8")
    card = docs / "models/example.mdx"
    card.write_text(
        card.read_text(encoding="utf-8").replace("&#36;10", "&#36;11").replace("&#36;0.1", "&#36;0.2"), encoding="utf-8"
    )
    errors, _ = audit_docs.audit(docs, catalog)
    for expected in ("unlisted page", "unknown OpenAPI operation", "card prices", "cached input"):
        assert any(expected in error for error in errors)


def test_repository_docs_pass_offline_audit():
    errors, _ = audit_docs.audit(ROOT / "docs", ROOT / "python/timbal/models.yaml")
    assert errors == []
