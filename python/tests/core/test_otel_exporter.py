import json
import os
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from timbal.state.context import RunContext
from timbal.state.tracing.exporters.otel import OTelExporter
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.state.tracing.span import Span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_span(
    call_id: str = "c1",
    path: str = "agent.step",
    parent_call_id: str | None = None,
    t0: int = 1_000,
    t1: int = 2_000,
    error: str | None = None,
    usage: dict | None = None,
    input_dump=None,
    output_dump=None,
) -> Span:
    span = Span(
        path=path,
        call_id=call_id,
        parent_call_id=parent_call_id,
        t0=t0,
        t1=t1,
        error=error,
        usage=usage or {},
    )
    span._input_dump = input_dump
    span._output_dump = output_dump
    span._memory_dump = None
    span._session_dump = None
    return span


def _make_run_context(run_id: str = "run-1", spans: list[Span] | None = None) -> RunContext:
    provider = JsonlTracingProvider.configured()
    ctx = RunContext(tracing_provider=provider)
    object.__setattr__(ctx, "id", run_id)
    object.__setattr__(ctx, "parent_id", None)
    for span in spans or [_make_span()]:
        ctx._trace[span.call_id] = span
    return ctx


def _mock_client(raise_exc: Exception | None = None):
    """Return a mock httpx.AsyncClient whose post() succeeds or raises."""
    mock_response = MagicMock()
    if raise_exc:
        mock_response.raise_for_status.side_effect = raise_exc
    else:
        mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


async def _drain(exporter: OTelExporter) -> None:
    """Wait for all background tasks to complete."""
    await exporter.close()


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

class TestIds:
    def test_trace_id_is_32_hex_chars(self):
        tid = OTelExporter._trace_id("run-abc")
        assert len(tid) == 32
        assert all(c in "0123456789abcdef" for c in tid)

    def test_span_id_is_16_hex_chars(self):
        sid = OTelExporter._span_id("call-abc")
        assert len(sid) == 16
        assert all(c in "0123456789abcdef" for c in sid)

    def test_trace_id_is_deterministic(self):
        assert OTelExporter._trace_id("run-x") == OTelExporter._trace_id("run-x")

    def test_span_id_is_deterministic(self):
        assert OTelExporter._span_id("c1") == OTelExporter._span_id("c1")

    def test_different_run_ids_give_different_trace_ids(self):
        assert OTelExporter._trace_id("run-a") != OTelExporter._trace_id("run-b")

    def test_different_call_ids_give_different_span_ids(self):
        assert OTelExporter._span_id("c1") != OTelExporter._span_id("c2")


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------

class TestTimeConversion:
    def test_ms_to_ns(self):
        assert OTelExporter._ms_to_ns(1_000) == "1000000000"

    def test_ms_to_ns_returns_string(self):
        assert isinstance(OTelExporter._ms_to_ns(42), str)

    def test_zero(self):
        assert OTelExporter._ms_to_ns(0) == "0"


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

class TestBuildPayload:
    def _exporter(self, service_name="my-svc") -> OTelExporter:
        return OTelExporter(service_name=service_name)

    def test_top_level_structure(self):
        ctx = _make_run_context()
        payload = self._exporter()._build_payload(ctx)
        assert "resourceSpans" in payload
        assert len(payload["resourceSpans"]) == 1

    def test_service_name_in_resource(self):
        ctx = _make_run_context()
        payload = self._exporter("svc-x")._build_payload(ctx)
        attrs = payload["resourceSpans"][0]["resource"]["attributes"]
        names = {a["key"]: a["value"].get("stringValue") for a in attrs}
        assert names["service.name"] == "svc-x"

    def test_scope_name_is_timbal(self):
        ctx = _make_run_context()
        payload = self._exporter()._build_payload(ctx)
        scope = payload["resourceSpans"][0]["scopeSpans"][0]["scope"]
        assert scope["name"] == "timbal"

    def test_span_count_matches_trace(self):
        spans = [_make_span("c1"), _make_span("c2", path="step2", parent_call_id="c1")]
        ctx = _make_run_context(spans=spans)
        payload = self._exporter()._build_payload(ctx)
        otel_spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(otel_spans) == 2

    def test_trace_id_matches_run_id(self):
        ctx = _make_run_context("run-42")
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otel_span["traceId"] == OTelExporter._trace_id("run-42")

    def test_span_id_matches_call_id(self):
        ctx = _make_run_context(spans=[_make_span("my-call")])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otel_span["spanId"] == OTelExporter._span_id("my-call")

    def test_span_name_is_path(self):
        ctx = _make_run_context(spans=[_make_span(path="agent.llm")])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otel_span["name"] == "agent.llm"

    def test_start_end_times_in_nanoseconds(self):
        ctx = _make_run_context(spans=[_make_span(t0=1_000, t1=2_000)])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otel_span["startTimeUnixNano"] == "1000000000"
        assert otel_span["endTimeUnixNano"] == "2000000000"

    def test_parent_span_id_present_when_parent_call_id_set(self):
        span = _make_span(call_id="child", parent_call_id="parent-call")
        ctx = _make_run_context(spans=[span])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otel_span["parentSpanId"] == OTelExporter._span_id("parent-call")

    def test_parent_span_id_absent_when_no_parent(self):
        ctx = _make_run_context(spans=[_make_span(parent_call_id=None)])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert "parentSpanId" not in otel_span

    def test_input_attribute_present(self):
        ctx = _make_run_context(spans=[_make_span(input_dump={"x": 1})])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attr_keys = {a["key"] for a in otel_span["attributes"]}
        assert "timbal.input" in attr_keys

    def test_input_attribute_absent_when_none(self):
        ctx = _make_run_context(spans=[_make_span(input_dump=None)])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attr_keys = {a["key"] for a in otel_span["attributes"]}
        assert "timbal.input" not in attr_keys

    def test_output_attribute_present(self):
        ctx = _make_run_context(spans=[_make_span(output_dump={"result": 42})])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attr_keys = {a["key"] for a in otel_span["attributes"]}
        assert "timbal.output" in attr_keys

    def test_error_attribute_and_status_when_error_set(self):
        ctx = _make_run_context(spans=[_make_span(error="boom")])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attr_keys = {a["key"] for a in otel_span["attributes"]}
        assert "timbal.error" in attr_keys
        assert otel_span["status"]["code"] == 2  # ERROR

    def test_ok_status_when_no_error(self):
        ctx = _make_run_context(spans=[_make_span()])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otel_span["status"]["code"] == 1  # OK

    def test_usage_attribute_present_when_non_empty(self):
        ctx = _make_run_context(spans=[_make_span(usage={"tokens": 42})])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attr_keys = {a["key"] for a in otel_span["attributes"]}
        assert "timbal.usage" in attr_keys

    def test_usage_attribute_absent_when_empty(self):
        ctx = _make_run_context(spans=[_make_span(usage={})])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attr_keys = {a["key"] for a in otel_span["attributes"]}
        assert "timbal.usage" not in attr_keys

    def test_input_value_is_valid_json_string(self):
        ctx = _make_run_context(spans=[_make_span(input_dump={"nested": [1, 2]})])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attr = next(a for a in otel_span["attributes"] if a["key"] == "timbal.input")
        parsed = json.loads(attr["value"]["stringValue"])
        assert parsed == {"nested": [1, 2]}

    def test_t1_none_falls_back_to_t0(self):
        span = Span(path="p", call_id="c", t0=5_000, t1=None)
        span._input_dump = None
        span._output_dump = None
        span._memory_dump = None
        span._session_dump = None
        ctx = _make_run_context(spans=[span])
        payload = self._exporter()._build_payload(ctx)
        otel_span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otel_span["startTimeUnixNano"] == otel_span["endTimeUnixNano"]


# ---------------------------------------------------------------------------
# Resource attributes
# ---------------------------------------------------------------------------

class TestResourceAttributes:
    def test_includes_service_name(self):
        exporter = OTelExporter(service_name="my-svc")
        keys = {a["key"] for a in exporter._resource_attributes()}
        assert "service.name" in keys

    def test_includes_telemetry_sdk_name(self):
        exporter = OTelExporter()
        keys = {a["key"] for a in exporter._resource_attributes()}
        assert "telemetry.sdk.name" in keys

    def test_includes_host_name(self):
        exporter = OTelExporter()
        attrs = {a["key"]: a for a in exporter._resource_attributes()}
        assert "host.name" in attrs
        assert attrs["host.name"]["value"]["stringValue"] == socket.gethostname()

    def test_includes_process_pid(self):
        exporter = OTelExporter()
        attrs = {a["key"]: a for a in exporter._resource_attributes()}
        assert "process.pid" in attrs
        assert attrs["process.pid"]["value"]["intValue"] == str(os.getpid())


# ---------------------------------------------------------------------------
# Fire-and-forget + client lifecycle
# ---------------------------------------------------------------------------

class TestFireAndForget:
    @pytest.mark.asyncio
    async def test_export_returns_before_post_completes(self):
        posted = []

        async def slow_post(*args, **kwargs):
            import asyncio as _asyncio
            await _asyncio.sleep(0.05)
            posted.append(True)
            r = MagicMock()
            r.raise_for_status = MagicMock()
            return r

        exporter = OTelExporter()
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = slow_post
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        # export() returned — post hasn't finished yet
        assert posted == []

        await _drain(exporter)
        assert posted == [True]

    @pytest.mark.asyncio
    async def test_close_drains_all_pending_tasks(self):
        posted = []

        async def record_post(*args, **kwargs):
            posted.append(True)
            r = MagicMock()
            r.raise_for_status = MagicMock()
            return r

        exporter = OTelExporter()
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = record_post
        exporter._client = mock_client

        for i in range(5):
            await exporter.export(_make_run_context(run_id=f"run-{i}"))

        assert len(posted) < 5  # not all done yet (may vary by scheduler)
        await exporter.close()
        assert len(posted) == 5

    @pytest.mark.asyncio
    async def test_pending_tasks_cleared_after_close(self):
        exporter = OTelExporter()
        mock_client = _mock_client()
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        await exporter.close()

        assert len(exporter._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_async_context_manager_drains_on_exit(self):
        posted = []

        async def record_post(*args, **kwargs):
            posted.append(True)
            r = MagicMock()
            r.raise_for_status = MagicMock()
            return r

        async with OTelExporter() as exporter:
            mock_client = AsyncMock()
            mock_client.is_closed = False
            mock_client.post = record_post
            exporter._client = mock_client
            await exporter.export(_make_run_context())

        assert posted == [True]

    @pytest.mark.asyncio
    async def test_get_client_returns_same_instance(self):
        exporter = OTelExporter()
        c1 = exporter._get_client()
        c2 = exporter._get_client()
        assert c1 is c2
        await exporter.close()

    @pytest.mark.asyncio
    async def test_export_reuses_client_across_calls(self):
        exporter = OTelExporter()
        mock_client = _mock_client()
        exporter._client = mock_client

        await exporter.export(_make_run_context(run_id="run-1"))
        await exporter.export(_make_run_context(run_id="run-2"))
        await _drain(exporter)

        assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        exporter = OTelExporter(retry_delays=(1.0,))
        mock_client = _mock_client()
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        await _drain(exporter)

        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        exporter = OTelExporter(retry_delays=(0.0, 0.0))
        fail_response = MagicMock()
        fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=MagicMock()
        )
        ok_response = MagicMock()
        ok_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=[fail_response, ok_response])
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        await _drain(exporter)

        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_silently_discards_after_all_attempts_exhausted(self):
        exporter = OTelExporter(retry_delays=(0.0,))
        exc = httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
        mock_client = _mock_client(raise_exc=exc)
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        await _drain(exporter)  # should not raise

        assert mock_client.post.call_count == 2  # 1 attempt + 1 retry

    @pytest.mark.asyncio
    async def test_no_retries_when_empty_retry_delays(self):
        exporter = OTelExporter(retry_delays=())
        exc = httpx.ConnectError("refused")
        mock_client = _mock_client(raise_exc=exc)
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        await _drain(exporter)  # should not raise

        assert mock_client.post.call_count == 1


# ---------------------------------------------------------------------------
# HTTP export
# ---------------------------------------------------------------------------

class TestExport:
    @pytest.mark.asyncio
    async def test_posts_to_correct_url(self):
        exporter = OTelExporter(endpoint="http://collector:4318")
        mock_client = _mock_client()
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        await _drain(exporter)

        call_url = mock_client.post.call_args[0][0]
        assert call_url == "http://collector:4318/v1/traces"

    @pytest.mark.asyncio
    async def test_content_type_header_set(self):
        exporter = OTelExporter()
        mock_client = _mock_client()
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        await _drain(exporter)

        headers = mock_client.post.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_custom_headers_forwarded(self):
        exporter = OTelExporter(headers={"x-honeycomb-team": "secret"})
        mock_client = _mock_client()
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        await _drain(exporter)

        headers = mock_client.post.call_args.kwargs["headers"]
        assert headers["x-honeycomb-team"] == "secret"

    @pytest.mark.asyncio
    async def test_payload_sent_as_json(self):
        exporter = OTelExporter()
        mock_client = _mock_client()
        exporter._client = mock_client

        await exporter.export(_make_run_context())
        await _drain(exporter)

        payload = mock_client.post.call_args.kwargs["json"]
        assert "resourceSpans" in payload

    @pytest.mark.asyncio
    async def test_trailing_slash_stripped_from_endpoint(self):
        exporter = OTelExporter(endpoint="http://collector:4318/")
        assert exporter.endpoint == "http://collector:4318"

    @pytest.mark.asyncio
    async def test_integrated_with_provider(self, tmp_path):
        """OTelExporter fires when attached to a JsonlTracingProvider."""
        exported = []

        class FakeOTelExporter(OTelExporter):
            async def export(self, run_context):
                exported.append(self._build_payload(run_context))

        provider = JsonlTracingProvider.configured(
            _path=tmp_path / "traces.jsonl",
            _exporters=[FakeOTelExporter()],
        )
        ctx = _make_run_context(spans=[_make_span()])
        await provider.put(ctx)

        assert len(exported) == 1
        assert "resourceSpans" in exported[0]
        assert (tmp_path / "traces.jsonl").exists()
