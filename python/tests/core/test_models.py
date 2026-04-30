"""Tests for core/models.py — get_context_window and model metadata."""

from timbal.core.models import get_context_window


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
