"""Tests for core/models.py — get_context_window and model metadata."""

import pytest
from timbal.core.models import (
    FAST_USAGE_SUFFIX,
    FLEX_USAGE_SUFFIX,
    LONG_CONTEXT_USAGE_SUFFIX,
    base_usage_metric,
    get_context_window,
    get_long_context_threshold,
    has_cache_write_pricing,
    service_tier_usage_suffix,
    uses_long_context_pricing,
)


class TestGetContextWindow:
    def test_known_anthropic_model(self):
        result = get_context_window("anthropic/claude-sonnet-4-6")
        assert isinstance(result, int)
        assert result > 0

    def test_known_openai_model(self):
        result = get_context_window("openai/gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_unknown_model_returns_none(self):
        result = get_context_window("fake/nonexistent-model-xyz")
        assert result is None

    def test_empty_string_returns_none(self):
        result = get_context_window("")
        assert result is None

    def test_result_is_cached(self):
        """Calling twice should return the same value (lru_cache active)."""
        r1 = get_context_window("anthropic/claude-sonnet-4-6")
        r2 = get_context_window("anthropic/claude-sonnet-4-6")
        assert r1 == r2

    def test_multiple_providers_have_context_windows(self):
        """Smoke test that several providers have entries in models.yaml."""
        providers = [
            "anthropic/claude-haiku-4-5",
            "openai/gpt-4o-mini",
        ]
        for model_id in providers:
            result = get_context_window(model_id)
            assert result is not None, f"{model_id} should have a context window"
            assert result > 0


class TestGetLongContextThreshold:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("openai/gpt-6-astra", 272_000),
            ("openai/gpt-5.6-sol", 272_000),
            ("openai/gpt-5.5", 272_000),
            ("openai/gpt-5.4", 272_000),
            ("xai/grok-4.6", 200_000),
            ("byteplus/seed-2-0-pro-260328", 128_000),
        ],
    )
    def test_models_with_long_context_tier(self, model_id: str, expected: int):
        assert get_long_context_threshold(model_id) == expected

    @pytest.mark.parametrize("model_id", ["openai/gpt-4o", "anthropic/claude-sonnet-4-6", "openai/gpt-5.4-nano"])
    def test_models_without_long_context_tier(self, model_id: str):
        assert get_long_context_threshold(model_id) is None

    def test_unknown_model_returns_none(self):
        assert get_long_context_threshold("fake/nonexistent-model-xyz") is None

    def test_provider_specific_boundary_semantics(self):
        assert uses_long_context_pricing("openai/gpt-6-astra", 272_000) is False
        assert uses_long_context_pricing("openai/gpt-6-astra", 272_001) is True
        assert uses_long_context_pricing("xai/grok-4.6", 199_999) is False
        assert uses_long_context_pricing("xai/grok-4.6", 200_000) is True

    def test_threshold_is_below_context_window(self):
        """The tier must be reachable: threshold strictly inside the advertised window."""
        for model_id in ("openai/gpt-6-astra", "xai/grok-4.6", "byteplus/seed-2-0-lite-260228"):
            threshold = get_long_context_threshold(model_id)
            window = get_context_window(model_id)
            assert threshold is not None and window is not None
            assert threshold < window


class TestCacheWritePricing:
    @pytest.mark.parametrize("model_id", ["openai/gpt-6-astra", "openai/gpt-5.6-sol", "openai/gpt-5.6-luna"])
    def test_models_with_published_cache_write_rate(self, model_id: str):
        assert has_cache_write_pricing(model_id) is True

    @pytest.mark.parametrize("model_id", ["openai/gpt-5.5", "openai/gpt-4o", "xai/grok-4.6", "fake/nope"])
    def test_models_without_cache_write_rate(self, model_id: str):
        assert has_cache_write_pricing(model_id) is False


class TestBaseUsageMetric:
    def test_strips_long_context_suffix(self):
        assert base_usage_metric(f"output_text_tokens{LONG_CONTEXT_USAGE_SUFFIX}") == "output_text_tokens"

    def test_strips_combined_pricing_suffixes(self):
        metric = f"output_text_tokens{LONG_CONTEXT_USAGE_SUFFIX}{FAST_USAGE_SUFFIX}"
        assert base_usage_metric(metric) == "output_text_tokens"

    def test_leaves_other_metrics_alone(self):
        assert base_usage_metric("output_text_tokens") == "output_text_tokens"
        assert base_usage_metric("web_search_requests") == "web_search_requests"


class TestServiceTierUsageSuffix:
    def test_astra_tiers(self):
        assert service_tier_usage_suffix("openai/gpt-6-astra", "fast") == FAST_USAGE_SUFFIX
        assert service_tier_usage_suffix("openai/gpt-6-astra", "priority") == FAST_USAGE_SUFFIX
        assert service_tier_usage_suffix("openai/gpt-6-astra", "flex") == FLEX_USAGE_SUFFIX

    def test_unpriced_tier_is_ignored(self):
        assert service_tier_usage_suffix("openai/gpt-4o", "fast") == ""
        assert service_tier_usage_suffix("openai/gpt-6-astra", "default") == ""
