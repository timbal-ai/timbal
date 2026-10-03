"""Billing regressions: emitted collector units must have the published SQL rates."""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml
from timbal.core.models import base_usage_metric, service_tier_usage_suffix

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("generate_costs_sql", ROOT / "scripts/generate_costs_sql.py")
costs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(costs)


def _rates(sql, model):
    rows = {}
    for line in sql.splitlines():
        if not line.startswith('INSERT INTO') or f"VALUES ('{model}'," not in line:
            continue
        match = re.search(r", '([^']+)', 'USD', ([0-9.]+)\) ON CONFLICT", line)
        assert match, line
        rows[match[1]] = float(match[2]) * 1_000_000
    return rows


def test_sol_61_costs_include_cache_and_combined_tiers(tmp_path):
    dest = tmp_path / "costs.sql"
    costs.main(output_path=dest, provider_filter="openai")
    rates = _rates(dest.read_text(encoding="utf-8"), "openai/gpt-6.1-sol")
    for context, expected in [("", [2, 10, 0.1, 2.5]), ("_long_context", [4, 15, 0.2, 5])]:
        for tier, multiplier in [("", 1), ("_flex", 0.5), ("_fast", 2)]:
            for unit, price in zip(
                ["input_text_tokens", "output_text_tokens", "input_cached_tokens", "input_cache_write_tokens"],
                expected,
                strict=True,
            ):
                assert rates[unit + context + tier] == pytest.approx(price * multiplier)
    assert "input_audio_tokens" not in rates
    assert "web_search_requests" not in rates


def test_ultrafast_usage_units_and_prices(tmp_path):
    suffix = service_tier_usage_suffix("openai/gpt-6-astra", "ultrafast")
    assert suffix == "_ultrafast"
    assert service_tier_usage_suffix("openai/gpt-6.1-sol", "ultrafast") == ""
    assert base_usage_metric("output_text_tokens_long_context_ultrafast") == "output_text_tokens"
    dest = tmp_path / "costs.sql"
    costs.main(output_path=dest, provider_filter="openai")
    rates = _rates(dest.read_text(encoding="utf-8"), "openai/gpt-6-astra")
    assert rates["input_text_tokens_ultrafast"] == pytest.approx(60)
    assert rates["output_text_tokens_long_context_ultrafast"] == pytest.approx(450)
    assert rates["input_cache_write_tokens_long_context_ultrafast"] == pytest.approx(150)


def test_unknown_and_dedicated_prices_are_never_invented(tmp_path, monkeypatch):
    catalog = tmp_path / "models.yaml"
    catalog.write_text(yaml.safe_dump({"models": [
        {"id": "test/unknown", "provider": "test", "input_price": 1, "output_price": None,
         "capabilities": ["audio"]},
        {"id": "test/dedicated", "provider": "test", "dedicated_only": True,
         "input_price": None, "output_price": None},
        {"id": "anthropic/unknown-cache", "provider": "anthropic", "input_price": 1, "output_price": 5,
         "cached_input_price": None, "cache_write_price": None},
    ]}), encoding="utf-8")
    monkeypatch.setattr(costs, "MODELS_YAML", catalog)
    dest = tmp_path / "costs.sql"
    costs.main(output_path=dest)
    assert _rates(dest.read_text(encoding="utf-8"), "test/unknown") == {"input_text_tokens": 1}
    assert _rates(dest.read_text(encoding="utf-8"), "test/dedicated") == {}
    anthropic = _rates(dest.read_text(encoding="utf-8"), "anthropic/unknown-cache")
    assert "cache_read_input_tokens" not in anthropic
    assert "cache_creation_input_tokens" not in anthropic
