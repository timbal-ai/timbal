"""Tests for ~/.timbal config loading and platform config resolution."""

from pathlib import Path

import pytest
import timbal.state.config_loader as config_loader
from timbal.state.config_loader import FileConfig, load_file_config, resolve_platform_config


def _write_minimal_config(tmp_path: Path, sync_traces_line: str) -> None:
    (tmp_path / "config").write_text(
        "[default]\n"
        "base_url = https://api.example.com\n"
        "org = org-from-file\n"
        f"{sync_traces_line}",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("sync_traces_line", "expected"),
    [
        ("", None),
        ("sync_traces = 1\n", True),
        ("sync_traces = true\n", True),
        ("sync_traces = TRUE\n", True),
        ("sync_traces = t\n", True),
        ("sync_traces = y\n", True),
        ("sync_traces = yes\n", True),
        ("sync_traces = enabled\n", True),
        ("sync_traces = on\n", True),
        ("sync_traces = 0\n", False),
        ("sync_traces = false\n", False),
        ("sync_traces = bogus\n", False),
    ],
)
def test_load_file_config_sync_traces_truthy_list(
    tmp_path: Path,
    sync_traces_line: str,
    expected: bool | None,
) -> None:
    _write_minimal_config(tmp_path, sync_traces_line)
    fc = load_file_config(profile="default", config_dir=tmp_path)
    assert fc.sync_traces_enabled == expected


def test_resolve_sync_traces_defaults_to_true(
    tmp_path: Path,
) -> None:
    creds = tmp_path / "credentials"
    creds.write_text(
        "[default]\napi_key = secret-key\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "config"
    cfg.write_text(
        "[default]\nbase_url = https://api.example.com\norg = o1\n",
        encoding="utf-8",
    )
    pc = resolve_platform_config(profile="default", config_dir=tmp_path)
    assert pc is not None
    assert pc.sync_traces_enabled is True


def test_force_refresh_bypasses_stale_cached_none(monkeypatch) -> None:
    """A default call that cached None must not short-circuit env credentials set
    later in the process when force_refresh=True."""
    monkeypatch.setattr(config_loader, "load_file_config", lambda *_a, **_k: FileConfig(None, None, None, None))
    monkeypatch.setattr(config_loader, "_cached_default_config", None)
    monkeypatch.setattr(config_loader, "_default_config_resolved", False)
    for var in ("TIMBAL_API_KEY", "TIMBAL_API_HOST", "TIMBAL_API_TOKEN", "TIMBAL_ORG_ID"):
        monkeypatch.delenv(var, raising=False)

    # First default call with no creds caches None.
    assert resolve_platform_config() is None
    assert config_loader._default_config_resolved is True

    # Credentials appear after the cache was poisoned.
    monkeypatch.setenv("TIMBAL_API_KEY", "sk-platform")
    monkeypatch.setenv("TIMBAL_API_HOST", "api.example.com")
    monkeypatch.setenv("TIMBAL_ORG_ID", "org-123")

    # Without force_refresh, the stale cached None is returned.
    assert resolve_platform_config() is None

    # force_refresh re-reads env and refreshes the cache.
    pc = resolve_platform_config(force_refresh=True)
    assert pc is not None
    assert pc.subject is not None
    assert pc.subject.org_id == "org-123"
    assert resolve_platform_config() is not None
