import asyncio
import hashlib
import json
import os
import socket
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from ..providers.base import Exporter
from ..span import Span

if TYPE_CHECKING:
    from ...context import RunContext

logger = structlog.get_logger(__name__)

_DEFAULT_RETRY_DELAYS = (1.0, 2.0, 4.0)  # seconds between attempts


class OTelExporter(Exporter):
    """Forward traces to an OpenTelemetry-compatible backend via OTLP HTTP/JSON.

    Sends one POST to ``{endpoint}/v1/traces`` per run using the OTLP JSON
    encoding. Compatible with any OTLP-capable backend: Jaeger, Honeycomb,
    Datadog, Grafana Tempo, the OpenTelemetry Collector, etc.

    Usage::

        from timbal.state.tracing.exporters import OTelExporter
        from timbal.state.tracing.providers import JsonlTracingProvider

        provider = JsonlTracingProvider.configured(
            _path=Path("traces.jsonl"),
            _exporters=[
                OTelExporter(
                    endpoint="http://localhost:4318",
                    service_name="my-agent",
                    headers={"x-honeycomb-team": "YOUR_KEY"},
                ),
            ],
        )
        agent = Agent(..., tracing_provider=provider)

    **Fire-and-forget**

    ``export()`` returns immediately after scheduling the HTTP call as a
    background task. This means tracing never adds latency to the critical
    path — the run completes as soon as ``_store()`` finishes (a fast local
    write). Retries happen entirely in the background.

    To ensure all in-flight exports complete before process exit, call
    ``close()`` (or use the async context manager)::

        async with OTelExporter(...) as exporter:
            ...  # all exports are awaited before __aexit__ returns

    **Retries**

    On HTTP errors or network failures, the background task retries up to
    ``len(retry_delays)`` times with the specified delays between attempts.
    If all attempts fail, a warning is logged and the exception is discarded.

    **Span and trace ID generation**

    OTel requires stable 64-bit span IDs and 128-bit trace IDs expressed as
    lowercase hex strings. These are derived by SHA-256 hashing the Timbal
    ``call_id`` and ``run_id`` respectively, then taking the first N bytes.
    Collisions are astronomically unlikely for typical workloads.

    **Attribute mapping**

    Each Timbal span is mapped to one OTel span with the following attributes:

    - ``timbal.input``  — JSON-serialised input (if present)
    - ``timbal.output`` — JSON-serialised output (if present)
    - ``timbal.error``  — stringified error (if present, also sets ERROR status)
    - ``timbal.usage``  — JSON-serialised usage counters (if non-empty)

    **Time format**

    Timbal ``t0``/``t1`` are Unix epoch milliseconds (``int(time.time() * 1000)``).
    OTel requires Unix nanoseconds as string-encoded uint64. Conversion: ``ms * 1_000_000``.

    Args:
        endpoint: Base URL of the OTLP HTTP receiver, without a trailing slash.
                  Default: ``http://localhost:4318`` (standard OTLP port).
        service_name: Value of the ``service.name`` resource attribute.
        headers: Extra HTTP headers, e.g. authentication tokens.
        timeout: HTTP request timeout in seconds. Default: 5.0.
        retry_delays: Sequence of delays (seconds) between retry attempts.
                      Default: ``(1.0, 2.0, 4.0)`` — up to 3 retries.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        service_name: str = "timbal",
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
        retry_delays: tuple[float, ...] = _DEFAULT_RETRY_DELAYS,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.service_name = service_name
        self.headers = headers or {}
        self.timeout = timeout
        self.retry_delays = retry_delays
        self._client: httpx.AsyncClient | None = None
        self._pending_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Drain all in-flight exports, then close the HTTP client.

        Call this before process exit to ensure no traces are dropped.
        """
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OTelExporter":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trace_id(run_id: str) -> str:
        """Derive a 128-bit OTel trace ID (32 hex chars) from a run ID."""
        return hashlib.sha256(run_id.encode()).hexdigest()[:32]

    @staticmethod
    def _span_id(call_id: str) -> str:
        """Derive a 64-bit OTel span ID (16 hex chars) from a call ID."""
        return hashlib.sha256(call_id.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    @staticmethod
    def _ms_to_ns(ms: int) -> str:
        """Convert Unix milliseconds to nanoseconds string (OTel uint64 wire format)."""
        return str(ms * 1_000_000)

    @staticmethod
    def _build_attributes(span: Span) -> list[dict[str, Any]]:
        attrs = []
        input_dump = getattr(span, "_input_dump", None)
        if input_dump is not None:
            attrs.append({
                "key": "timbal.input",
                "value": {"stringValue": json.dumps(input_dump, default=str)},
            })
        output_dump = getattr(span, "_output_dump", None)
        if output_dump is not None:
            attrs.append({
                "key": "timbal.output",
                "value": {"stringValue": json.dumps(output_dump, default=str)},
            })
        if span.error is not None:
            attrs.append({
                "key": "timbal.error",
                "value": {"stringValue": str(span.error)},
            })
        if span.usage:
            attrs.append({
                "key": "timbal.usage",
                "value": {"stringValue": json.dumps(span.usage)},
            })
        return attrs

    @staticmethod
    def _span_status(span: Span) -> dict[str, Any]:
        # OTel StatusCode: 0=UNSET, 1=OK, 2=ERROR
        if span.error is not None:
            return {"code": 2, "message": str(span.error)}
        return {"code": 1}

    def _resource_attributes(self) -> list[dict[str, Any]]:
        attrs = [
            {"key": "service.name", "value": {"stringValue": self.service_name}},
            {"key": "telemetry.sdk.name", "value": {"stringValue": "timbal"}},
            {"key": "telemetry.sdk.language", "value": {"stringValue": "python"}},
        ]
        try:
            attrs.append({"key": "host.name", "value": {"stringValue": socket.gethostname()}})
        except Exception:
            pass
        try:
            attrs.append({"key": "process.pid", "value": {"intValue": str(os.getpid())}})
        except Exception:
            pass
        return attrs

    def _build_payload(self, run_context: "RunContext") -> dict[str, Any]:
        """Build the OTLP JSON payload for a completed run.

        This is a pure method (no I/O) — easy to unit-test independently.
        """
        trace_id = self._trace_id(str(run_context.id))
        otel_spans = []
        for span in run_context._trace.values():
            t0_ns = self._ms_to_ns(span.t0)
            t1_ns = self._ms_to_ns(span.t1 if span.t1 is not None else span.t0)
            otel_span: dict[str, Any] = {
                "traceId": trace_id,
                "spanId": self._span_id(span.call_id),
                "name": span.path,
                "kind": 1,  # SPAN_KIND_INTERNAL
                "startTimeUnixNano": t0_ns,
                "endTimeUnixNano": t1_ns,
                "attributes": self._build_attributes(span),
                "status": self._span_status(span),
            }
            if span.parent_call_id:
                otel_span["parentSpanId"] = self._span_id(span.parent_call_id)
            otel_spans.append(otel_span)

        return {
            "resourceSpans": [{
                "resource": {"attributes": self._resource_attributes()},
                "scopeSpans": [{
                    "scope": {"name": "timbal"},
                    "spans": otel_spans,
                }],
            }]
        }

    # ------------------------------------------------------------------
    # Background posting
    # ------------------------------------------------------------------

    async def _post_with_retry(self, payload: dict[str, Any], run_id: str) -> None:
        """POST the payload with retries. Runs as a background task."""
        url = f"{self.endpoint}/v1/traces"
        request_headers = {"Content-Type": "application/json", **self.headers}
        last_exc: Exception | None = None

        for attempt, delay in enumerate(
            [None, *self.retry_delays],  # first attempt has no preceding delay
            start=1,
        ):
            if delay is not None:
                await asyncio.sleep(delay)
            try:
                response = await self._get_client().post(url, json=payload, headers=request_headers)
                response.raise_for_status()
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "otel_export_attempt_failed",
                    run_id=run_id,
                    attempt=attempt,
                    max_attempts=1 + len(self.retry_delays),
                    error=str(exc),
                )

        logger.error(
            "otel_export_failed",
            run_id=run_id,
            max_attempts=1 + len(self.retry_delays),
            error=str(last_exc),
        )

    # ------------------------------------------------------------------
    # Exporter interface
    # ------------------------------------------------------------------

    async def export(self, run_context: "RunContext") -> None:
        """Schedule the export as a background task and return immediately.

        The payload is built synchronously before scheduling so the
        run_context can be safely garbage-collected after this call returns.
        ``close()`` awaits all pending tasks before shutting down.
        """
        payload = self._build_payload(run_context)
        run_id = str(run_context.id)
        task = asyncio.create_task(self._post_with_retry(payload, run_id))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
