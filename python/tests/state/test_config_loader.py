"""Tests for ~/.timbal config loading and platform config resolution."""

from pathlib import Path

import pytest
from pydantic import SecretStr
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.config_loader import load_file_config, resolve_platform_config


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


def test_resolve_sync_traces_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TIMBAL_SYNC_TRACES", raising=False)
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
    monkeypatch.setenv("TIMBAL_APP_ID", "app1")
    monkeypatch.setenv("TIMBAL_SYNC_TRACES", "0")

    pc = resolve_platform_config(profile="default", config_dir=tmp_path)
    assert pc is not None
    assert pc.sync_traces_enabled is False
    assert pc.subject is not None
    assert pc.subject.app_id == "app1"


def test_resolve_sync_traces_explicit_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TIMBAL_SYNC_TRACES", "0")
    existing = PlatformConfig(
        host="h",
        auth=PlatformAuth(type=PlatformAuthType.BEARER, token=SecretStr("t")),
        subject=PlatformSubject(org_id="o", app_id="a"),
        sync_traces_enabled=True,
    )
    pc = resolve_platform_config(platform_config=existing)
    assert pc is not None
    assert pc.sync_traces_enabled is True


def test_resolve_sync_traces_defaults_to_true(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TIMBAL_SYNC_TRACES", raising=False)
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
