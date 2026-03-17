import pytest

from timbal.platform.utils import _resolve_url_and_headers
from timbal.state import set_run_context
from timbal.state.context import RunContext


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Isolate from real ~/.timbal config so tests are deterministic."""
    # Point config loader at an empty dir so it doesn't pick up real credentials.
    monkeypatch.setattr("timbal.state.config_loader.TIMBAL_CONFIG_DIR", tmp_path)
    monkeypatch.delenv("TIMBAL_API_KEY", raising=False)
    monkeypatch.delenv("TIMBAL_API_TOKEN", raising=False)
    monkeypatch.delenv("TIMBAL_API_HOST", raising=False)
    monkeypatch.delenv("TIMBAL_ORG_ID", raising=False)
    monkeypatch.delenv("TIMBAL_PROJECT_ENV_ID", raising=False)
    monkeypatch.delenv("TIMBAL_START_API_PORT", raising=False)
    monkeypatch.delenv("TIMBAL_START_UI_PORT", raising=False)
    monkeypatch.delenv("TIMBAL_START_WORKFORCE", raising=False)
    monkeypatch.delenv("TIMBAL_WORKFORCE", raising=False)
    set_run_context(RunContext())


class TestLocalDev:
    """Resolution via TIMBAL_START_* env vars (local development)."""

    def test_api(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_API_PORT", "3001")
        url, headers = _resolve_url_and_headers("api", "users", {})
        assert url == "http://localhost:3001/users"
        assert "Authorization" not in headers

    def test_ui(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_UI_PORT", "3737")
        url, headers = _resolve_url_and_headers("ui", "index", {})
        assert url == "http://localhost:3737/index"
        assert "Authorization" not in headers

    def test_workforce_member(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_WORKFORCE", "abc123:4455,def456:4456")
        url, _ = _resolve_url_and_headers("abc123", "run", {})
        assert url == "http://localhost:4455/run"
        url, _ = _resolve_url_and_headers("def456", "stream", {})
        assert url == "http://localhost:4456/stream"

    def test_workforce_legacy_fallback(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_WORKFORCE", "abc123:4455")
        url, _ = _resolve_url_and_headers("abc123", "run", {})
        assert url == "http://localhost:4455/run"

    def test_workforce_prefers_start_over_legacy(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_WORKFORCE", "abc123:5000")
        monkeypatch.setenv("TIMBAL_WORKFORCE", "abc123:4455")
        url, _ = _resolve_url_and_headers("abc123", "run", {})
        assert url == "http://localhost:5000/run"

    def test_preserves_extra_headers(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_API_PORT", "3001")
        _, headers = _resolve_url_and_headers("api", "users", {"X-Custom": "val"})
        assert headers["X-Custom"] == "val"

    def test_attaches_auth_when_platform_config_available(self, monkeypatch):
        """If TIMBAL_API_KEY is set, platform_config gets auto-resolved and auth is attached."""
        monkeypatch.setenv("TIMBAL_API_KEY", "sk-test")
        monkeypatch.setenv("TIMBAL_API_HOST", "api.timbal.ai")
        monkeypatch.setenv("TIMBAL_START_API_PORT", "3001")
        set_run_context(RunContext())
        url, headers = _resolve_url_and_headers("api", "users", {})
        assert url == "http://localhost:3001/users"
        assert headers["Authorization"] == "Bearer sk-test"


class TestRemote:
    """Resolution via TIMBAL_PROJECT_ENV_ID (deployed environments)."""

    @pytest.fixture(autouse=True)
    def _setup_remote(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_PROJECT_ENV_ID", "454")
        monkeypatch.setenv("TIMBAL_API_KEY", "sk-test")
        monkeypatch.setenv("TIMBAL_API_HOST", "api.timbal.ai")
        set_run_context(RunContext())

    def test_api(self):
        url, headers = _resolve_url_and_headers("api", "users", {})
        assert url == "https://proj-env-454.deployments.timbal.ai/api/users"
        assert headers["Authorization"] == "Bearer sk-test"

    def test_ui(self):
        url, headers = _resolve_url_and_headers("ui", "index", {})
        assert url == "https://proj-env-454.deployments.timbal.ai/index"
        assert headers["Authorization"] == "Bearer sk-test"

    def test_workforce_member(self):
        url, headers = _resolve_url_and_headers("agent-abc", "run", {})
        assert url == "https://proj-env-454.deployments.timbal.ai/api/workforce/agent-abc/run"
        assert headers["Authorization"] == "Bearer sk-test"

    def test_raises_without_platform_config(self, monkeypatch):
        monkeypatch.delenv("TIMBAL_API_KEY")
        set_run_context(RunContext())
        with pytest.raises(ValueError, match="No platform config"):
            _resolve_url_and_headers("api", "users", {})


class TestRemoteTakesPrecedence:
    """TIMBAL_PROJECT_ENV_ID takes priority — if set, it's remote."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_PROJECT_ENV_ID", "454")
        monkeypatch.setenv("TIMBAL_API_KEY", "sk-test")
        monkeypatch.setenv("TIMBAL_API_HOST", "api.timbal.ai")
        set_run_context(RunContext())

    def test_api_remote_over_local(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_API_PORT", "3001")
        url, _ = _resolve_url_and_headers("api", "users", {})
        assert url == "https://proj-env-454.deployments.timbal.ai/api/users"

    def test_workforce_remote_over_local(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_WORKFORCE", "abc123:4455")
        url, _ = _resolve_url_and_headers("abc123", "run", {})
        assert url == "https://proj-env-454.deployments.timbal.ai/api/workforce/abc123/run"


class TestNoConfig:
    """Raises when no env vars are set."""

    def test_raises_for_api(self):
        with pytest.raises(ValueError, match="TIMBAL_START_API_PORT is not set"):
            _resolve_url_and_headers("api", "users", {})

    def test_raises_for_ui(self):
        with pytest.raises(ValueError, match="TIMBAL_START_UI_PORT is not set"):
            _resolve_url_and_headers("ui", "index", {})

    def test_raises_for_unknown_workforce_member(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_WORKFORCE", "abc123:4455")
        with pytest.raises(ValueError, match="not found in TIMBAL_START_WORKFORCE"):
            _resolve_url_and_headers("unknown", "run", {})

    def test_raises_for_no_workforce_env(self):
        with pytest.raises(ValueError, match="TIMBAL_START_WORKFORCE is not set"):
            _resolve_url_and_headers("some-agent", "run", {})
