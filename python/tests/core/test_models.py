"""Tests for core/models.py — get_context_window and model metadata."""

import pytest
from timbal.core.models import get_context_window, get_long_context_threshold


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

    def test_threshold_is_below_context_window(self):
        """The tier must be reachable: threshold strictly inside the advertised window."""
        for model_id in ("openai/gpt-6-astra", "xai/grok-4.6", "byteplus/seed-2-0-lite-260228"):
            threshold = get_long_context_threshold(model_id)
            window = get_context_window(model_id)
            assert threshold is not None and window is not None
            assert threshold < window
