"""Tests for timbal.platform.utils — _resolve_url_and_headers, _request, _stream."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from timbal.errors import PlatformError
from timbal.platform.utils import _request, _resolve_url_and_headers, _stream
from timbal.state import set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_platform_config(
    host: str = "api.timbal.ai",
    org_id: str = "org_123",
    app_id: str = "app_456",
    version_id: str = "v1",
) -> PlatformConfig:
    return PlatformConfig(
        host=host,
        auth=PlatformAuth(type=PlatformAuthType.BEARER, token="token"),
        subject=PlatformSubject(org_id=org_id, app_id=app_id, version_id=version_id),
    )


def _run_context_with_config(cfg: PlatformConfig | None = None) -> RunContext:
    """Create a RunContext with an explicit tracing_provider=None to prevent network calls."""
    rc = RunContext(platform_config=cfg, tracing_provider=None)
    set_run_context(rc)
    return rc


# ---------------------------------------------------------------------------
# Fixture: isolate env and context var between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Strip all relevant env vars and reset the run-context before each test."""
    for var in (
        "TIMBAL_PROJECT_ENV_ID",
        "TIMBAL_START_API_PORT",
        "TIMBAL_START_UI_PORT",
        "TIMBAL_START_WORKFORCE",
        "TIMBAL_WORKFORCE",
        "TIMBAL_API_KEY",
        "TIMBAL_API_HOST",
        "TIMBAL_ORG_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    # Reset config-loader cache so RunContext() doesn't pick up a stale hit
    monkeypatch.setattr("timbal.state.config_loader._cached_default_config", None)
    monkeypatch.setattr("timbal.state.config_loader._default_config_resolved", False)
    set_run_context(None)  # type: ignore[arg-type]


# ===========================================================================
# TestResolveUrlAndHeaders
# ===========================================================================


class TestResolveUrlAndHeaders:
    """Unit tests for _resolve_url_and_headers."""

    # --- service=None (platform API calls) ----------------------------------

    def test_service_none_uses_platform_config(self):
        cfg = _make_platform_config()
        _run_context_with_config(cfg)
        url, headers = _resolve_url_and_headers(None, "v1/runs", {})
        assert url == "https://api.timbal.ai/v1/runs"
        assert headers["Authorization"] == f"Bearer {cfg.auth.token.get_secret_value()}"

    def test_service_none_no_platform_config_raises(self):
        _run_context_with_config(None)
        with pytest.raises(ValueError, match="No platform config"):
            _resolve_url_and_headers(None, "v1/runs", {})

    # --- service set + TIMBAL_PROJECT_ENV_ID (remote/gateway) ---------------

    def test_service_api_with_env_id(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_PROJECT_ENV_ID", "env42")
        cfg = _make_platform_config()
        _run_context_with_config(cfg)
        url, headers = _resolve_url_and_headers("api", "health", {})
        assert url == "https://proj-env-env42.deployments.timbal.ai/api/health"
        assert "Authorization" in headers

    def test_service_ui_with_env_id(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_PROJECT_ENV_ID", "env42")
        cfg = _make_platform_config()
        _run_context_with_config(cfg)
        url, headers = _resolve_url_and_headers("ui", "dashboard", {})
        assert url == "https://proj-env-env42.deployments.timbal.ai/dashboard"
        assert "Authorization" in headers

    def test_service_worker_with_env_id_uses_workforce_gateway(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_PROJECT_ENV_ID", "env42")
        cfg = _make_platform_config()
        _run_context_with_config(cfg)
        url, _ = _resolve_url_and_headers("myworker", "run", {})
        assert url == "https://proj-env-env42.deployments.timbal.ai/api/workforce/myworker/run"

    # --- service set, no env_id (local dev) ---------------------------------

    def test_service_api_local_dev(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_API_PORT", "3001")
        _run_context_with_config(None)
        url, headers = _resolve_url_and_headers("api", "health", {})
        assert url == "http://localhost:3001/health"
        assert "Authorization" not in headers

    def test_service_api_no_port_raises(self):
        _run_context_with_config(None)
        with pytest.raises(ValueError, match="TIMBAL_START_API_PORT is not set"):
            _resolve_url_and_headers("api", "health", {})

    def test_service_ui_local_dev(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_UI_PORT", "4200")
        _run_context_with_config(None)
        url, _ = _resolve_url_and_headers("ui", "app", {})
        assert url == "http://localhost:4200/app"

    def test_service_ui_no_port_raises(self):
        _run_context_with_config(None)
        with pytest.raises(ValueError, match="TIMBAL_START_UI_PORT is not set"):
            _resolve_url_and_headers("ui", "app", {})

    def test_service_worker_timbal_start_workforce(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_WORKFORCE", "myworker:8080")
        _run_context_with_config(None)
        url, _ = _resolve_url_and_headers("myworker", "run", {})
        assert url == "http://localhost:8080/run"

    def test_service_worker_timbal_workforce_fallback(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_WORKFORCE", "myworker:8080")
        _run_context_with_config(None)
        url, _ = _resolve_url_and_headers("myworker", "run", {})
        assert url == "http://localhost:8080/run"

    def test_service_worker_timbal_start_workforce_takes_priority(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_WORKFORCE", "myworker:9999")
        monkeypatch.setenv("TIMBAL_WORKFORCE", "myworker:8080")
        _run_context_with_config(None)
        url, _ = _resolve_url_and_headers("myworker", "run", {})
        assert url == "http://localhost:9999/run"

    def test_service_missing_from_workforce_raises(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_WORKFORCE", "myworker:8080")
        _run_context_with_config(None)
        with pytest.raises(ValueError, match="not found in TIMBAL_START_WORKFORCE"):
            _resolve_url_and_headers("missing", "run", {})

    def test_no_workforce_env_set_raises(self):
        _run_context_with_config(None)
        with pytest.raises(ValueError, match="TIMBAL_START_WORKFORCE is not set"):
            _resolve_url_and_headers("some-agent", "run", {})

    def test_extra_headers_preserved(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_API_PORT", "3001")
        _run_context_with_config(None)
        _, headers = _resolve_url_and_headers("api", "health", {"X-Custom": "yes"})
        assert headers["X-Custom"] == "yes"


# ===========================================================================
# Helpers shared by _request and _stream tests
# ===========================================================================


def _mock_httpx_client(mock_http):
    """Return a context-manager mock that yields mock_http from __aenter__."""
    mock_client_cm = MagicMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_http)
    mock_client_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_client_cm


def _make_http_status_error(status_code: int, reason: str = "Error") -> "httpx.HTTPStatusError":
    import httpx

    request = httpx.Request("GET", "https://api.timbal.ai/test")
    response = httpx.Response(status_code, request=request)
    # Provide a text attribute so error_body extraction doesn't blow up
    return httpx.HTTPStatusError(reason, request=request, response=response)


# ===========================================================================
# TestRequest
# ===========================================================================


class TestRequest:
    """Tests for the _request async helper."""

    @pytest.fixture(autouse=True)
    def _setup_context(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_API_PORT", "3001")
        _run_context_with_config(None)

    async def _call(self, patch_target="timbal.platform.utils.httpx.AsyncClient", **kwargs):
        """Helper that patches httpx.AsyncClient and drives _request."""
        mock_http = AsyncMock()
        mock_client_cm = _mock_httpx_client(mock_http)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_http.request = AsyncMock(return_value=mock_response)
        with patch(patch_target, return_value=mock_client_cm):
            result = await _request("GET", "health", service="api", max_retries=1, **kwargs)
        return result, mock_http

    @pytest.mark.asyncio
    async def test_success_returns_response(self):
        result, _ = await self._call()
        assert result is not None

    @pytest.mark.asyncio
    async def test_4xx_non_429_raises_immediately_without_retry(self):
        import httpx

        mock_http = AsyncMock()
        mock_client_cm = _mock_httpx_client(mock_http)
        exc = _make_http_status_error(404)
        mock_http.request = AsyncMock(side_effect=exc)

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("timbal.platform.utils.asyncio.sleep", new=AsyncMock()) as mock_sleep:
                with pytest.raises(PlatformError):
                    await _request("GET", "health", service="api", max_retries=3)
        # No sleep means no retry was attempted
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_429_retries_then_succeeds(self):
        import httpx

        mock_http = AsyncMock()
        mock_client_cm = _mock_httpx_client(mock_http)
        exc_429 = _make_http_status_error(429, "Too Many Requests")
        ok_response = MagicMock()
        ok_response.raise_for_status = MagicMock()
        mock_http.request = AsyncMock(side_effect=[exc_429, ok_response])

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("timbal.platform.utils.asyncio.sleep", new=AsyncMock()):
                result = await _request("GET", "health", service="api", max_retries=3)
        assert result is ok_response

    @pytest.mark.asyncio
    async def test_5xx_retries_then_raises_platform_error(self):
        mock_http = AsyncMock()
        mock_client_cm = _mock_httpx_client(mock_http)
        exc_500 = _make_http_status_error(500, "Internal Server Error")
        mock_http.request = AsyncMock(side_effect=exc_500)

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("timbal.platform.utils.asyncio.sleep", new=AsyncMock()):
                with pytest.raises(PlatformError):
                    await _request("GET", "health", service="api", max_retries=2)
        assert mock_http.request.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_network_exception_retries_then_reraises(self):
        mock_http = AsyncMock()
        mock_client_cm = _mock_httpx_client(mock_http)
        mock_http.request = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("timbal.platform.utils.asyncio.sleep", new=AsyncMock()):
                with pytest.raises(ConnectionError):
                    await _request("GET", "health", service="api", max_retries=2)
        assert mock_http.request.call_count == 3

    @pytest.mark.asyncio
    async def test_json_payload_forwarded(self):
        mock_http = AsyncMock()
        mock_client_cm = _mock_httpx_client(mock_http)
        ok_response = MagicMock()
        ok_response.raise_for_status = MagicMock()
        mock_http.request = AsyncMock(return_value=ok_response)

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            await _request("POST", "data", service="api", json={"key": "value"}, max_retries=0)

        _, kwargs = mock_http.request.call_args
        assert kwargs.get("json") == {"key": "value"}

    @pytest.mark.asyncio
    async def test_content_payload_forwarded(self):
        mock_http = AsyncMock()
        mock_client_cm = _mock_httpx_client(mock_http)
        ok_response = MagicMock()
        ok_response.raise_for_status = MagicMock()
        mock_http.request = AsyncMock(return_value=ok_response)

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            await _request("POST", "data", service="api", content=b"raw bytes", max_retries=0)

        _, kwargs = mock_http.request.call_args
        assert kwargs.get("content") == b"raw bytes"

    @pytest.mark.asyncio
    async def test_files_payload_forwarded(self):
        mock_http = AsyncMock()
        mock_client_cm = _mock_httpx_client(mock_http)
        ok_response = MagicMock()
        ok_response.raise_for_status = MagicMock()
        mock_http.request = AsyncMock(return_value=ok_response)
        files = {"file": ("report.pdf", b"%PDF-1.4", "application/pdf")}

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            await _request("POST", "upload", service="api", files=files, max_retries=0)

        _, kwargs = mock_http.request.call_args
        assert kwargs.get("files") == files


# ===========================================================================
# TestStream
# ===========================================================================


class TestStream:
    """Tests for the _stream async-generator helper."""

    @pytest.fixture(autouse=True)
    def _setup_context(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_START_API_PORT", "3001")
        _run_context_with_config(None)

    def _build_stream_mock(self, lines: list[str]):
        """Return (mock_http, mock_stream_cm) configured to yield *lines*."""
        mock_http = AsyncMock()

        mock_stream_cm = MagicMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_response.aiter_lines = _aiter_lines
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
        mock_http.stream = MagicMock(return_value=mock_stream_cm)

        mock_client_cm = _mock_httpx_client(mock_http)
        return mock_client_cm, mock_http, mock_stream_cm, mock_response

    @pytest.mark.asyncio
    async def test_yields_parsed_json_from_data_lines(self):
        import json

        payload = {"event": "delta", "text": "hello"}
        lines = [f"data: {json.dumps(payload)}"]
        mock_client_cm, mock_http, _, _ = self._build_stream_mock(lines)

        results = []
        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            async for item in _stream("POST", "stream", service="api"):
                results.append(item)

        assert results == [payload]

    @pytest.mark.asyncio
    async def test_skips_non_data_lines(self):
        import json

        payload = {"text": "world"}
        lines = [
            "event: message",
            "",
            f"data: {json.dumps(payload)}",
        ]
        mock_client_cm, mock_http, _, _ = self._build_stream_mock(lines)

        results = []
        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            async for item in _stream("POST", "stream", service="api"):
                results.append(item)

        assert results == [payload]

    @pytest.mark.asyncio
    async def test_skips_done_sentinel(self):
        import json

        payload = {"text": "done check"}
        lines = [f"data: {json.dumps(payload)}", "data: [DONE]"]
        mock_client_cm, _, _, _ = self._build_stream_mock(lines)

        results = []
        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            async for item in _stream("POST", "stream", service="api"):
                results.append(item)

        # [DONE] must not appear as a result
        assert results == [payload]

    @pytest.mark.asyncio
    async def test_skips_invalid_json_with_warning(self):
        lines = ["data: not-valid-json"]
        mock_client_cm, _, _, _ = self._build_stream_mock(lines)

        results = []
        mock_logger = MagicMock()
        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("timbal.platform.utils.logger", mock_logger):
                async for item in _stream("POST", "stream", service="api"):
                    results.append(item)

        assert results == []
        mock_logger.warning.assert_called_once()
        warning_args = mock_logger.warning.call_args[0]
        assert any("non-JSON" in str(a) for a in warning_args)

    @pytest.mark.asyncio
    async def test_4xx_raises_platform_error(self):
        mock_http = AsyncMock()
        mock_stream_cm = MagicMock()
        mock_response = AsyncMock()

        exc = _make_http_status_error(403, "Forbidden")
        mock_response.raise_for_status = MagicMock(side_effect=exc)
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
        mock_http.stream = MagicMock(return_value=mock_stream_cm)
        mock_client_cm = _mock_httpx_client(mock_http)

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("timbal.platform.utils.asyncio.sleep", new=AsyncMock()) as mock_sleep:
                with pytest.raises(PlatformError):
                    async for _ in _stream("POST", "stream", service="api", max_retries=3):
                        pass
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_5xx_retries_then_raises_platform_error(self):
        mock_http = AsyncMock()

        # Build a re-usable raise_for_status that always raises 500
        exc = _make_http_status_error(500, "Internal Server Error")

        calls = []

        def _make_stream_cm():
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock(side_effect=exc)
            mock_stream_cm = MagicMock()
            mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
            calls.append(mock_stream_cm)
            return mock_stream_cm

        mock_http.stream = MagicMock(side_effect=lambda *a, **kw: _make_stream_cm())
        mock_client_cm = _mock_httpx_client(mock_http)

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("timbal.platform.utils.asyncio.sleep", new=AsyncMock()):
                with pytest.raises(PlatformError):
                    async for _ in _stream("POST", "stream", service="api", max_retries=2):
                        pass

        assert mock_http.stream.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_stream_json_payload_forwarded(self):
        """Cover line 189: payload_kwargs['json'] = json inside _stream."""
        mock_client_cm, mock_http, _, _ = self._build_stream_mock([])
        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            async for _ in _stream("POST", "stream", service="api", json={"k": "v"}):
                pass
        _, kwargs = mock_http.stream.call_args
        assert kwargs.get("json") == {"k": "v"}

    @pytest.mark.asyncio
    async def test_stream_content_payload_forwarded(self):
        """Cover line 191: payload_kwargs['content'] = content inside _stream."""
        mock_client_cm, mock_http, _, _ = self._build_stream_mock([])
        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            async for _ in _stream("POST", "stream", service="api", content=b"raw"):
                pass
        _, kwargs = mock_http.stream.call_args
        assert kwargs.get("content") == b"raw"

    @pytest.mark.asyncio
    async def test_stream_files_payload_forwarded(self):
        """Cover line 193: payload_kwargs['files'] = files inside _stream."""
        files = {"file": ("report.pdf", b"%PDF", "application/pdf")}
        mock_client_cm, mock_http, _, _ = self._build_stream_mock([])
        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            async for _ in _stream("POST", "upload", service="api", files=files):
                pass
        _, kwargs = mock_http.stream.call_args
        assert kwargs.get("files") == files

    @pytest.mark.asyncio
    async def test_stream_error_body_read_failure_sets_none(self):
        """Cover lines 224-225: error_body=None when aread() itself raises."""
        import httpx

        request = httpx.Request("GET", "https://api.timbal.ai/test")
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.reason_phrase = "Internal Server Error"
        mock_response.aread = AsyncMock(side_effect=RuntimeError("cannot read"))
        exc = httpx.HTTPStatusError("Error", request=request, response=mock_response)

        mock_http = AsyncMock()
        mock_stream_cm = MagicMock()
        bad_response = AsyncMock()
        bad_response.raise_for_status = MagicMock(side_effect=exc)
        mock_stream_cm.__aenter__ = AsyncMock(return_value=bad_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
        mock_http.stream = MagicMock(return_value=mock_stream_cm)
        mock_client_cm = _mock_httpx_client(mock_http)

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("timbal.platform.utils.asyncio.sleep", new=AsyncMock()):
                with pytest.raises(PlatformError):
                    async for _ in _stream("POST", "stream", service="api", max_retries=0):
                        pass

    @pytest.mark.asyncio
    async def test_stream_network_error_retries_then_reraises(self):
        """Cover lines 250-261: except Exception block in _stream for network errors."""
        mock_http = AsyncMock()

        def _make_failing_stream_cm():
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(side_effect=ConnectionError("network down"))
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        mock_http.stream = MagicMock(side_effect=lambda *a, **kw: _make_failing_stream_cm())
        mock_client_cm = _mock_httpx_client(mock_http)

        with patch("timbal.platform.utils.httpx.AsyncClient", return_value=mock_client_cm):
            with patch("timbal.platform.utils.asyncio.sleep", new=AsyncMock()):
                with pytest.raises(ConnectionError):
                    async for _ in _stream("POST", "stream", service="api", max_retries=2):
                        pass

        assert mock_http.stream.call_count == 3  # initial + 2 retries
