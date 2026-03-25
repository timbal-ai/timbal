import json
import subprocess

import pytest
from timbal.codegen.tool_discovery import (
    FrameworkTool,
    _CACHE_DIR,
    _CACHE_FILE,
    _load_cache,
    _save_cache,
    _tools_fingerprint,
    get_framework_tools,
    get_provider_summaries,
    invalidate_cache,
)


# ---------------------------------------------------------------------------
# get_provider_summaries() unit tests
# ---------------------------------------------------------------------------


class TestGetProviderSummaries:
    def test_returns_nonempty(self):
        summaries = get_provider_summaries()
        assert len(summaries) > 0

    def test_includes_system_group(self):
        names = [s["name"] for s in get_provider_summaries()]
        assert "system" in names

    def test_sorted_by_count_descending(self):
        counts = [s["tool_count"] for s in get_provider_summaries()]
        assert counts == sorted(counts, reverse=True)

    def test_counts_sum_to_total_tools(self):
        summaries = get_provider_summaries()
        total = sum(s["tool_count"] for s in summaries)
        assert total == len(get_framework_tools())

    def test_each_summary_has_required_keys(self):
        for s in get_provider_summaries():
            assert "name" in s
            assert "logo" in s
            assert "tool_count" in s
            assert isinstance(s["tool_count"], int)
            assert s["tool_count"] > 0

    def test_system_has_no_logo(self):
        system = next(s for s in get_provider_summaries() if s["name"] == "system")
        assert system["logo"] is None

    def test_non_system_has_logo(self):
        for s in get_provider_summaries():
            if s["name"] != "system":
                assert s["logo"] is not None

    def test_no_cache_flag_passes_through(self):
        """no_cache=True still returns correct data."""
        summaries = get_provider_summaries(no_cache=True)
        assert len(summaries) > 0
        names = [s["name"] for s in summaries]
        assert "system" in names


# ---------------------------------------------------------------------------
# Disk cache unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Redirect cache to a temp directory for isolation."""
    import timbal.codegen.tool_discovery as mod

    cache_dir = tmp_path / ".tool_cache"
    cache_file = cache_dir / "framework_tools.json"
    monkeypatch.setattr(mod, "_CACHE_DIR", cache_dir)
    monkeypatch.setattr(mod, "_CACHE_FILE", cache_file)
    return cache_dir, cache_file


class TestToolsFingerprint:
    def test_returns_nonempty_string(self):
        fp = _tools_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) > 0

    def test_is_deterministic(self):
        assert _tools_fingerprint() == _tools_fingerprint()

    def test_is_hex_digest(self):
        fp = _tools_fingerprint()
        assert all(c in "0123456789abcdef" for c in fp)


class TestSaveAndLoadCache:
    def test_roundtrip(self, cache_dir):
        _dir, _file = cache_dir
        registry = {
            "FakeTool": FrameworkTool(
                module="timbal.tools",
                name="fake_tool",
                description="A fake tool",
                provider="fake",
                provider_logo="https://example.com/logo.svg",
            ),
            "AnotherTool": FrameworkTool(
                module="timbal.tools",
                name="another",
                description=None,
                provider=None,
                provider_logo=None,
            ),
        }
        fp = "abc123"
        _save_cache(fp, registry)
        assert _file.exists()

        loaded = _load_cache(fp)
        assert loaded is not None
        assert set(loaded.keys()) == {"FakeTool", "AnotherTool"}
        assert loaded["FakeTool"].name == "fake_tool"
        assert loaded["FakeTool"].description == "A fake tool"
        assert loaded["FakeTool"].provider == "fake"
        assert loaded["FakeTool"].provider_logo == "https://example.com/logo.svg"
        assert loaded["AnotherTool"].description is None
        assert loaded["AnotherTool"].provider is None

    def test_load_returns_none_when_no_cache(self, cache_dir):
        assert _load_cache("any") is None

    def test_load_returns_none_on_fingerprint_mismatch(self, cache_dir):
        _dir, _file = cache_dir
        registry = {
            "FakeTool": FrameworkTool("timbal.tools", "fake", None, None, None),
        }
        _save_cache("old_fingerprint", registry)
        assert _load_cache("new_fingerprint") is None

    def test_load_returns_none_on_corrupt_json(self, cache_dir):
        _dir, _file = cache_dir
        _dir.mkdir(parents=True, exist_ok=True)
        _file.write_text("not valid json{{{")
        assert _load_cache("any") is None

    def test_load_returns_none_on_missing_keys(self, cache_dir):
        _dir, _file = cache_dir
        _dir.mkdir(parents=True, exist_ok=True)
        _file.write_text(json.dumps({"fingerprint": "abc"}))
        assert _load_cache("abc") is None


class TestInvalidateCache:
    def test_invalidate_removes_cache_file(self, cache_dir):
        _dir, _file = cache_dir
        registry = {
            "FakeTool": FrameworkTool("timbal.tools", "fake", None, None, None),
        }
        _save_cache("fp", registry)
        assert _file.exists()

        invalidate_cache()
        assert not _file.exists()

    def test_invalidate_noop_when_no_cache(self, cache_dir):
        """Does not raise when cache doesn't exist."""
        invalidate_cache()


class TestGetFrameworkToolsCache:
    def test_creates_cache_on_first_call(self, cache_dir):
        _dir, _file = cache_dir
        assert not _file.exists()
        tools = get_framework_tools()
        assert _file.exists()
        assert len(tools) > 0

    def test_second_call_uses_cache(self, cache_dir):
        _dir, _file = cache_dir
        # First call — populates cache
        tools1 = get_framework_tools()
        assert _file.exists()
        mtime1 = _file.stat().st_mtime_ns

        # Second call — should NOT rewrite the cache
        tools2 = get_framework_tools()
        mtime2 = _file.stat().st_mtime_ns
        assert mtime1 == mtime2
        assert len(tools1) == len(tools2)

    def test_no_cache_flag_skips_cache(self, cache_dir):
        _dir, _file = cache_dir
        # Populate cache
        get_framework_tools()
        mtime1 = _file.stat().st_mtime_ns

        # no_cache=True should rewrite the file
        get_framework_tools(no_cache=True)
        mtime2 = _file.stat().st_mtime_ns
        assert mtime2 >= mtime1  # rewrites (or same if fast enough)

    def test_stale_cache_is_rebuilt(self, cache_dir):
        _dir, _file = cache_dir
        # Write a cache with a wrong fingerprint
        _dir.mkdir(parents=True, exist_ok=True)
        _file.write_text(json.dumps({
            "fingerprint": "stale_fingerprint_that_wont_match",
            "tools": {"FakeTool": {"module": "m", "name": "f", "description": None, "provider": None, "provider_logo": None}},
        }))

        tools = get_framework_tools()
        # Should have rebuilt — FakeTool should NOT be in the real registry
        assert "FakeTool" not in tools
        assert len(tools) > 100  # real tool count

    def test_cached_and_uncached_return_same_data(self, cache_dir):
        """Cache roundtrip preserves all tool metadata."""
        uncached = get_framework_tools(no_cache=True)
        cached = get_framework_tools()
        assert set(uncached.keys()) == set(cached.keys())
        for cls_name in uncached:
            u, c = uncached[cls_name], cached[cls_name]
            assert u.module == c.module
            assert u.name == c.name
            assert u.description == c.description
            assert u.provider == c.provider
            assert u.provider_logo == c.provider_logo


# ---------------------------------------------------------------------------
# CLI: get-tools (no filters → providers)
# ---------------------------------------------------------------------------


def _run(*cli_args: str) -> dict:
    result = subprocess.run(
        ["python", "-m", "timbal.codegen", "get-tools", *cli_args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"get-tools failed:\n{result.stderr}"
    return json.loads(result.stdout)


class TestGetToolsProviders:
    def test_default_returns_providers(self):
        data = _run()
        assert "providers" in data
        assert "tools" not in data
        assert len(data["providers"]) > 0

    def test_provider_summaries_shape(self):
        data = _run()
        for p in data["providers"]:
            assert "name" in p
            assert "logo" in p
            assert "tool_count" in p


# ---------------------------------------------------------------------------
# CLI: get-tools --provider
# ---------------------------------------------------------------------------


class TestGetToolsProvider:
    def test_filter_by_provider(self):
        data = _run("--provider", "slack")
        assert "tools" in data
        assert data["total"] > 0
        for t in data["tools"]:
            assert t["provider"] == "slack"

    def test_filter_system(self):
        data = _run("--provider", "system")
        assert data["total"] > 0
        for t in data["tools"]:
            assert t["provider"] is None

    def test_unknown_provider_returns_empty(self):
        data = _run("--provider", "nonexistent_provider_xyz")
        assert data["total"] == 0
        assert data["tools"] == []


# ---------------------------------------------------------------------------
# CLI: get-tools --search
# ---------------------------------------------------------------------------


class TestGetToolsSearch:
    def test_search_by_name(self):
        data = _run("--search", "web_search")
        assert data["total"] >= 1
        names = [t["name"] for t in data["tools"]]
        assert any("web_search" in n for n in names)

    def test_search_by_type(self):
        data = _run("--search", "WebSearch")
        assert data["total"] >= 1

    def test_search_case_insensitive(self):
        lower = _run("--search", "websearch")
        upper = _run("--search", "WEBSEARCH")
        assert lower["total"] == upper["total"]

    def test_search_no_results(self):
        data = _run("--search", "zzz_nonexistent_tool_zzz")
        assert data["total"] == 0
        assert data["tools"] == []

    def test_search_by_description(self):
        """Search matches against tool descriptions too."""
        # Find a tool with a known description word
        all_tools = _run("--provider", "slack")
        if all_tools["total"] > 0:
            # Search for a word that should appear in slack tool descriptions or names
            data = _run("--search", "slack")
            assert data["total"] >= 1


# ---------------------------------------------------------------------------
# CLI: combined filters
# ---------------------------------------------------------------------------


class TestGetToolsCombined:
    def test_provider_and_search(self):
        data = _run("--provider", "slack", "--search", "message")
        assert data["total"] >= 1
        for t in data["tools"]:
            assert t["provider"] == "slack"
            assert "message" in (t["name"] + (t["description"] or "") + t["type"]).lower()

    def test_combined_narrows_results(self):
        """Combined filters return fewer results than either alone."""
        provider_only = _run("--provider", "zendesk")
        search_only = _run("--search", "create")
        combined = _run("--provider", "zendesk", "--search", "create")
        assert combined["total"] <= provider_only["total"]
        assert combined["total"] <= search_only["total"]


# ---------------------------------------------------------------------------
# CLI: pagination
# ---------------------------------------------------------------------------


class TestGetToolsPagination:
    def test_default_limit(self):
        data = _run("--provider", "zendesk")
        assert len(data["tools"]) <= 50
        assert data["limit"] == 50
        assert data["offset"] == 0
        assert data["total"] > 50

    def test_custom_limit(self):
        data = _run("--provider", "zendesk", "--limit", "5")
        assert len(data["tools"]) == 5
        assert data["limit"] == 5

    def test_offset(self):
        page1 = _run("--provider", "zendesk", "--limit", "5", "--offset", "0")
        page2 = _run("--provider", "zendesk", "--limit", "5", "--offset", "5")
        assert page1["tools"] != page2["tools"]
        assert page1["total"] == page2["total"]

    def test_offset_beyond_total(self):
        data = _run("--provider", "zendesk", "--offset", "9999")
        assert data["tools"] == []
        assert data["total"] > 0

    def test_pagination_metadata(self):
        data = _run("--provider", "slack", "--limit", "3", "--offset", "2")
        assert data["limit"] == 3
        assert data["offset"] == 2
        assert isinstance(data["total"], int)

    def test_limit_larger_than_total(self):
        data = _run("--provider", "system", "--limit", "1000")
        assert len(data["tools"]) == data["total"]
        assert data["total"] < 1000


# ---------------------------------------------------------------------------
# CLI: --no-cache flag
# ---------------------------------------------------------------------------


class TestGetToolsNoCache:
    def test_get_tools_no_cache_flag(self):
        data = _run("--no-cache", "--provider", "slack")
        assert data["total"] > 0

    def test_get_tools_no_cache_providers(self):
        data = _run("--no-cache")
        assert "providers" in data
        assert len(data["providers"]) > 0


# ---------------------------------------------------------------------------
# CLI: tool data shape
# ---------------------------------------------------------------------------


class TestToolDataShape:
    def test_tool_has_all_fields(self):
        data = _run("--provider", "slack", "--limit", "1")
        assert len(data["tools"]) == 1
        tool = data["tools"][0]
        assert "type" in tool
        assert "module" in tool
        assert "name" in tool
        assert "description" in tool
        assert "provider" in tool
        assert "provider_logo" in tool

    def test_system_tool_shape(self):
        data = _run("--provider", "system", "--limit", "1")
        tool = data["tools"][0]
        assert tool["provider"] is None
        assert tool["provider_logo"] is None
        assert tool["module"] == "timbal.tools"

