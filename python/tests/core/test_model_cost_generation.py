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
    assert "ephemeral_5m_input_tokens" not in anthropic
    assert "ephemeral_1h_input_tokens" not in anthropic


@pytest.mark.parametrize("missing", [True, False], ids=["omitted", "null"])
@pytest.mark.parametrize("published", [None, "cached_input_price", "cache_write_price", "cache_write_1h_price"])
def test_anthropic_cache_prices_are_independently_optional(tmp_path, monkeypatch, missing, published):
    fields = {
        "cached_input_price": ["cache_read_input_tokens"],
        "cache_write_price": ["cache_creation_input_tokens", "ephemeral_5m_input_tokens"],
        "cache_write_1h_price": ["ephemeral_1h_input_tokens"],
    }
    model = {"id": "anthropic/test", "provider": "anthropic", "input_price": 1, "output_price": 5,
             "service_tiers": {"fast": 2}}
    if not missing:
        model.update(dict.fromkeys(fields))
    if published is not None:
        # Deliberately differs from the traditional multipliers: use the catalog verbatim.
        model[published] = 0.7
    catalog = tmp_path / "models.yaml"
    catalog.write_text(yaml.safe_dump({"models": [model]}), encoding="utf-8")
    monkeypatch.setattr(costs, "MODELS_YAML", catalog)
    dest = tmp_path / "costs.sql"
    costs.main(output_path=dest)
    rates = _rates(dest.read_text(encoding="utf-8"), model["id"])
    for suffix, multiplier in [("", 1), ("_fast", 2)]:
        assert rates["input_tokens" + suffix] == pytest.approx(multiplier)
        assert rates["output_tokens" + suffix] == pytest.approx(5 * multiplier)
        for field, units in fields.items():
            for unit in units:
                if field == published:
                    assert rates[unit + suffix] == pytest.approx(0.7 * multiplier)
                else:
                    assert unit + suffix not in rates


def test_anthropic_published_cache_rates_and_fast_mode(tmp_path):
    dest = tmp_path / "costs.sql"
    costs.main(output_path=dest, provider_filter="anthropic")
    rates = _rates(dest.read_text(encoding="utf-8"), "anthropic/claude-opus-5-5")
    for suffix, multiplier in [("", 1), ("_fast", 2)]:
        for unit, price in {
            "cache_read_input_tokens": 0.2,
            "cache_creation_input_tokens": 5,
            "ephemeral_5m_input_tokens": 5,
            "ephemeral_1h_input_tokens": 8,
        }.items():
            assert rates[unit + suffix] == pytest.approx(price * multiplier)


@pytest.mark.parametrize("model,cache_read", [
    ("anthropic/claude-sonnet-5-5", 0.1),
    ("anthropic/claude-sonnet-5", 0.2),
])
def test_sonnet_published_cache_read_prices(tmp_path, model, cache_read):
    dest = tmp_path / "costs.sql"
    costs.main(output_path=dest, provider_filter="anthropic")
    assert _rates(dest.read_text(encoding="utf-8"), model) == pytest.approx({
        "input_tokens": 2,
        "output_tokens": 10,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": 2.5,
        "ephemeral_5m_input_tokens": 2.5,
        "ephemeral_1h_input_tokens": 4,
        "web_fetch_requests": 0,
        "web_search_requests": 10_000,
    })


def test_haiku_55_costs_include_all_long_context_token_units(tmp_path):
    dest = tmp_path / "costs.sql"
    costs.main(output_path=dest, provider_filter="anthropic")
    rates = _rates(dest.read_text(encoding="utf-8"), "anthropic/claude-haiku-5-5")
    expected = {"web_fetch_requests": 0, "web_search_requests": 10_000}
    for suffix, prices in [("", [0.1, 0.5, 0.01, 0.125, 0.2]),
                           ("_long_context", [0.5, 2.5, 0.05, 0.625, 1])]:
        inp, out, read, write_5m, write_1h = prices
        expected.update({
            "input_tokens" + suffix: inp,
            "output_tokens" + suffix: out,
            "cache_read_input_tokens" + suffix: read,
            "cache_creation_input_tokens" + suffix: write_5m,
            "ephemeral_5m_input_tokens" + suffix: write_5m,
            "ephemeral_1h_input_tokens" + suffix: write_1h,
        })
    assert rates == pytest.approx(expected)


@pytest.mark.parametrize("field,unit", [
    ("cached_input_price", "cache_read_input_tokens"),
    ("cache_write_price", "ephemeral_5m_input_tokens"),
    ("cache_write_1h_price", "ephemeral_1h_input_tokens"),
])
@pytest.mark.parametrize("missing", [True, False], ids=["omitted", "null"])
def test_anthropic_long_context_cache_prices_are_not_inferred(tmp_path, monkeypatch, field, unit, missing):
    long_context = {"threshold": 100_000, "input_price": 2, "output_price": 10}
    if not missing:
        long_context[field] = None
    model = {"id": "anthropic/test", "provider": "anthropic", "input_price": 1, "output_price": 5,
             field: 0.7, "long_context": long_context, "service_tiers": {"fast": 2}}
    catalog = tmp_path / "models.yaml"
    catalog.write_text(yaml.safe_dump({"models": [model]}), encoding="utf-8")
    monkeypatch.setattr(costs, "MODELS_YAML", catalog)
    dest = tmp_path / "costs.sql"
    costs.main(output_path=dest)
    rates = _rates(dest.read_text(encoding="utf-8"), model["id"])
    for suffix, multiplier in [("", 1), ("_fast", 2)]:
        assert rates[unit + suffix] == pytest.approx(0.7 * multiplier)
        assert rates["input_tokens_long_context" + suffix] == pytest.approx(2 * multiplier)
        assert rates["output_tokens_long_context" + suffix] == pytest.approx(10 * multiplier)
        assert unit + "_long_context" + suffix not in rates
