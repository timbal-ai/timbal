"""Platform credentials remain available when tracing is local or disabled (#230)."""

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state import config_loader, get_run_context, set_run_context
from timbal.state.config import PlatformConfig
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider, JsonlTracingProvider, PlatformTracingProvider


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    for name in (
        "TIMBAL_API_HOST",
        "TIMBAL_API_KEY",
        "TIMBAL_API_TOKEN",
        "TIMBAL_ORG_ID",
        "TIMBAL_APP_ID",
        "TIMBAL_PROJECT_ID",
        "TIMBAL_REV",
        "TIMBAL_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_loader, "TIMBAL_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_loader, "_default_config_resolved", False)
    monkeypatch.setattr(config_loader, "_cached_default_config", None)
    monkeypatch.setattr(config_loader, "_file_config_cache", {})
    return tmp_path


@pytest.fixture(params=["environment", "files"])
def credentials(request, isolated_config, monkeypatch):
    if request.param == "environment":
        monkeypatch.setenv("TIMBAL_API_HOST", "api.example.test")
        monkeypatch.setenv("TIMBAL_API_KEY", "test-platform-key")
        monkeypatch.setenv("TIMBAL_ORG_ID", "test-org")
    else:
        (isolated_config / "config").write_text(
            "[default]\nbase_url = https://api.example.test\norg = test-org\n", encoding="utf-8"
        )
        (isolated_config / "credentials").write_text("[default]\napi_key = test-platform-key\n", encoding="utf-8")
    # A valid app would enable platform tracing if explicit selection were ignored.
    monkeypatch.setenv("TIMBAL_APP_ID", "test-app")


@pytest.mark.parametrize("provider", [None, InMemoryTracingProvider])
@pytest.mark.usefixtures("credentials")
def test_explicit_tracing_keeps_platform_proxy_credentials(provider, monkeypatch):
    from unittest.mock import Mock

    from timbal.core.llm import _PROVIDERS, clients

    ctx = RunContext(tracing_provider=provider)
    assert ctx.platform_config is not None
    assert ctx.platform_config.host == "api.example.test"
    assert ctx.platform_config.subject.org_id == "test-org"
    assert ctx.platform_config.subject.app_id == "test-app"
    assert ctx._tracing_provider is provider

    # Exercise real proxy selection, replacing only SDK construction (no network).
    create_client = Mock()
    monkeypatch.setattr(clients, "_get_client", create_client)
    _, base_url = clients._resolve_client("timbal", _PROVIDERS["timbal"], None, None, ctx)
    assert base_url.startswith("https://api.example.test/orgs/test-org/proxies/")
    assert create_client.call_args.args[1] == "Bearer test-platform-key"


@pytest.mark.usefixtures("credentials")
async def test_agent_with_jsonl_tracing_resolves_platform_config(tmp_path):
    provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
    observed = []

    def capture():
        observed.append(get_run_context())

    previous = get_run_context()
    set_run_context(None)
    try:
        agent = Agent(
            name="local_trace", model=TestModel(responses=["done"]), tracing_provider=provider, pre_hook=capture
        )
        result = await agent(prompt="hello").collect()
        assert result.status.code == "success", result.error
        assert observed[0].platform_config is not None
        assert observed[0].platform_config.auth.token.get_secret_value() == "test-platform-key"
        assert observed[0]._tracing_provider is provider
        assert (tmp_path / "traces.jsonl").is_file()
    finally:
        set_run_context(previous)


@pytest.mark.parametrize("provider", [None, InMemoryTracingProvider])
@pytest.mark.usefixtures("isolated_config")
def test_explicit_tracing_still_works_without_credentials(provider):
    ctx = RunContext(tracing_provider=provider)
    assert ctx.platform_config is None
    assert ctx._tracing_provider is provider


@pytest.mark.usefixtures("credentials")
def test_default_tracing_still_selects_platform():
    assert RunContext()._tracing_provider is PlatformTracingProvider


@pytest.mark.usefixtures("credentials")
def test_explicit_platform_config_keeps_precedence():
    config = PlatformConfig(host="explicit.example.test", auth={"type": "bearer", "token": "explicit-key"})
    ctx = RunContext(platform_config=config, tracing_provider=None)
    assert ctx.platform_config.host == "explicit.example.test"
    assert ctx.platform_config.auth.token.get_secret_value() == "explicit-key"
    assert ctx.platform_config.subject.org_id == "test-org"
    assert ctx._tracing_provider is None
