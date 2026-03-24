"""Tests for WebSearch and XSearch specification-only tools."""

import pytest
from timbal.tools import WebSearch, XSearch


class TestWebSearchSchema:
    """Test WebSearch schema generation across providers."""

    def test_basic_responses_schema(self):
        ws = WebSearch()
        schema = ws.openai_responses_schema
        assert schema == {"type": "web_search"}

    def test_allowed_domains(self):
        ws = WebSearch(allowed_domains=["example.com", "docs.python.org"])
        schema = ws.openai_responses_schema
        assert schema["filters"]["allowed_domains"] == ["example.com", "docs.python.org"]

    def test_blocked_domains_mapped_to_excluded(self):
        """blocked_domains should map to excluded_domains in the Responses API filters."""
        ws = WebSearch(blocked_domains=["spam.com", "ads.com"])
        schema = ws.openai_responses_schema
        assert schema["filters"]["excluded_domains"] == ["spam.com", "ads.com"]
        assert "blocked_domains" not in schema.get("filters", {})

    def test_both_allowed_and_blocked_domains(self):
        ws = WebSearch(allowed_domains=["good.com"], blocked_domains=["bad.com"])
        schema = ws.openai_responses_schema
        assert schema["filters"]["allowed_domains"] == ["good.com"]
        assert schema["filters"]["excluded_domains"] == ["bad.com"]

    def test_user_location(self):
        loc = {"type": "approximate", "country": "GB", "city": "London"}
        ws = WebSearch(user_location=loc)
        schema = ws.openai_responses_schema
        assert schema["user_location"] == loc

    def test_no_filters_when_empty(self):
        ws = WebSearch()
        schema = ws.openai_responses_schema
        assert "filters" not in schema

    def test_anthropic_schema(self):
        ws = WebSearch(allowed_domains=["example.com"], blocked_domains=["spam.com"])
        schema = ws.anthropic_schema
        assert schema["type"] == "web_search_20250305"
        assert schema["allowed_domains"] == ["example.com"]
        assert schema["blocked_domains"] == ["spam.com"]

    def test_chat_completions_raises(self):
        ws = WebSearch()
        with pytest.raises(ValueError, match="not compatible"):
            _ = ws.openai_chat_completions_schema


class TestXSearchSchema:
    """Test XSearch schema generation and provider restrictions."""

    def test_responses_schema(self):
        xs = XSearch()
        assert xs.openai_responses_schema == {"type": "x_search"}

    def test_default_name(self):
        xs = XSearch()
        assert xs.name == "x_search"

    def test_custom_name(self):
        xs = XSearch(name="my_x_search")
        assert xs.name == "my_x_search"

    def test_anthropic_raises(self):
        xs = XSearch()
        with pytest.raises(ValueError, match="only supported by xAI"):
            _ = xs.anthropic_schema

    def test_chat_completions_raises(self):
        xs = XSearch()
        with pytest.raises(ValueError, match="only supported by xAI"):
            _ = xs.openai_chat_completions_schema

    def test_serialization_uses_responses_schema(self):
        """Serialization should not crash (uses responses schema, not anthropic)."""
        xs = XSearch()
        serialized = xs.serialize()
        assert serialized == {"type": "x_search"}
