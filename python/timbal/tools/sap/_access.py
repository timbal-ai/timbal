"""Execution limits and call records for SAP framework tools.

Which SAP operations an agent may use is decided with the SDK's own gates: the run's tool policy
(``read_only``) and approvals (``requires_approval``). Every call records on its span who selected
it (an agent or a deterministic workflow), its route, the SAP object and, for writes, the document
types it can create.
"""

import asyncio
import threading
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal

import httpx
import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    PrivateAttr,
    StringConstraints,
    model_validator,
)
from pydantic_core import to_json

from ...core.tool import Tool
from ...state import get_call_id, get_run_context

Ref = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# What kind of SAP operation a call is, recorded on its span with who selected it.
Route = Literal[
    "hana_view",
    "hana_sql",
    "hana_write",
    "published_api",
    "cds_view",
    "sap_agent",
    "gui_script",
    "hana_toolkit",
    "sap_mcp_gateway",
    "third_party_hana",
    "third_party_api",
    "third_party_cds",
]
Scalar = str | int | float | bool | None
logger = structlog.get_logger("timbal.sap")


def json_size(value: Any) -> int:
    return len(to_json(value, bytes_mode="base64"))


def _selecting_agent() -> str | None:
    """Path of the nearest Timbal agent that invoked this call, directly or through workflows."""
    from ...core.agent import Agent

    ctx = get_run_context()
    if ctx and get_call_id():
        for _, span in ctx.call_stack():
            if isinstance(span.runnable, Agent):
                return span.path
    return None


def _record(details: dict[str, Any]) -> None:
    """Record the call on its span. Direct handler calls outside a run have none."""
    ctx = get_run_context()
    if not (ctx and get_call_id()):
        return
    try:
        span = ctx.current_span()
    except RuntimeError:
        return
    span.metadata["sap"] = details


def parse_host(raw: Any) -> tuple[str, int | None]:
    """A connection's host field: a bare host name, optionally with a port or an ``https://`` prefix."""
    value = str(raw or "").strip()
    try:
        url = httpx.URL(value if "://" in value else f"https://{value}") if value else None
    except httpx.InvalidURL:
        url = None
    if url is None or url.scheme != "https" or not url.host or url.userinfo or url.query or url.fragment:
        raise ValueError("The SAP connection needs a host name, optionally with a port.")
    if url.path not in ("", "/"):
        raise ValueError("The SAP connection's host cannot include a path.")
    return url.host.lower(), url.port


class SAPCustomizing(BaseModel):
    """The designated non-production system and client of a customizing change, and its transport request."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    nonproduction_system: Ref
    client: Annotated[str, StringConstraints(pattern=r"^\d{3}$")]
    transport_request: Ref


class SAPCall:
    """State of a single admitted operation, including any outstanding native driver call."""

    def __init__(self, access: "SAPAccess", consumer: str, application: str) -> None:
        self.access = access
        self.consumer = consumer
        # Reported to SAP sessions that carry one: the selecting agent's path or the tool's name.
        self.application = application
        # host, port and SAP client of the bound connection, once an HTTP tool resolved it.
        self.connection: dict[str, Any] | None = None
        # Cache key of the token this call uses, so a token SAP rejects is not reused.
        self.token_key: str | None = None
        self.deadline = time.monotonic() + access.timeout_seconds
        self.cancelled = threading.Event()
        self.worker: asyncio.Task | None = None
        self.input_bytes = 0

    def account_input(self, value: Any) -> None:
        self.input_bytes += json_size(value)
        if self.input_bytes > self.access.max_input_bytes:
            raise ValueError("SAP request exceeds the configured input byte limit.")

    def check_active(self) -> None:
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise TimeoutError("SAP operation expired.")

    async def blocking(self, fn: Callable) -> Any:
        # A cancelled asyncio waiter cannot kill a native driver thread. Keep its admission
        # slot until its own timeout/cleanup finishes; never replay an ambiguous write.
        self.worker = asyncio.create_task(asyncio.to_thread(fn))
        return await asyncio.shield(self.worker)


class SAPHTTPError(RuntimeError):
    def __init__(self, service: str, status_code: int):
        self.status_code = status_code
        super().__init__(f"SAP {service} returned HTTP {status_code}.")


class SAPAccess(BaseModel):
    """Limits shared by the SAP tools that use the same system and quota in this worker."""

    model_config = ConfigDict(extra="forbid")
    consumer: Ref = "workflow"
    # Set when a model outside Timbal selects the operations before they reach Timbal.
    agent_selected: bool = False
    max_rows: PositiveInt = 1_000
    max_bytes: PositiveInt = 5_000_000
    max_input_bytes: PositiveInt = 5_000_000
    calls_per_minute: PositiveInt = 60
    consumer_calls_per_minute: PositiveInt = 60
    concurrency: PositiveInt = 4
    consumer_concurrency: PositiveInt = 4
    timeout_seconds: PositiveInt = 60
    failure_threshold: PositiveInt = 5
    cooldown_seconds: PositiveInt = 30

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _recent: deque = PrivateAttr(default_factory=deque)
    _active: Counter = PrivateAttr(default_factory=Counter)
    _failures: int = PrivateAttr(default=0)
    _blocked_until: float = PrivateAttr(default=0)

    def _charge(self, consumer: str, *, start: bool) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._blocked_until:
                raise RuntimeError("SAP integration is cooling down after upstream failures.")
            if self._blocked_until:
                self._blocked_until = self._failures = 0
            while self._recent and self._recent[0][0] <= now - 60:
                self._recent.popleft()
            if (
                len(self._recent) >= self.calls_per_minute
                or sum(key == consumer for _, key in self._recent) >= self.consumer_calls_per_minute
            ):
                raise RuntimeError("SAP request rate limit reached.")
            if start:
                if self._active.total() >= self.concurrency or self._active[consumer] >= self.consumer_concurrency:
                    raise RuntimeError("SAP concurrency limit reached.")
                self._active[consumer] += 1
            self._recent.append((now, consumer))

    def _release(self, consumer: str) -> None:
        with self._lock:
            self._active[consumer] -= 1
            if not self._active[consumer]:
                del self._active[consumer]

    async def invoke(
        self,
        tool_name: str,
        target: str,
        operation: Callable[[SAPCall], Awaitable[Any]],
        *,
        route: Route,
        document_types: tuple[str, ...] | None = None,
        customizing: SAPCustomizing | None = None,
    ) -> Any:
        """Run one SAP operation. ``document_types`` is passed by operations that can write."""
        agent_path = _selecting_agent()
        agent = agent_path is not None or self.agent_selected
        actor = agent_path or self.consumer
        details: dict[str, Any] = {
            "selected_by": "agent" if agent else "workflow",
            "route": route,
            "object": target,
        }
        if agent_path:
            details["agent"] = agent_path
        if document_types is not None:
            details["document_types"] = list(document_types)
        if customizing:
            details["customizing"] = customizing.model_dump()
        _record(details)
        call = SAPCall(self, actor, agent_path or tool_name)
        start = time.monotonic()
        admitted, volume, status = False, 0, "denied"
        http_status = None
        try:
            if agent and route == "gui_script":
                raise PermissionError(
                    "SAP GUI scripts run only in deterministic workflows, not when an agent selects them."
                )
            self._charge(actor, start=True)
            admitted, status = True, "error"
            async with asyncio.timeout(self.timeout_seconds):
                result = await operation(call)
                volume = json_size(result)
                if volume > self.max_bytes:
                    raise ValueError("SAP result exceeds the configured byte limit.")
            with self._lock:
                self._failures = 0
            status = "ok"
            return result
        except (ValueError, PermissionError):
            raise
        except Exception as exc:
            if isinstance(exc, SAPHTTPError):
                http_status = exc.status_code
            if admitted:
                with self._lock:
                    self._failures += 1
                    if self._failures >= self.failure_threshold:
                        self._blocked_until = time.monotonic() + self.cooldown_seconds
            raise
        finally:
            call.cancelled.set()
            if admitted:
                if call.worker and not call.worker.done():

                    def finished(task: asyncio.Task) -> None:
                        if not task.cancelled():
                            task.exception()
                        self._release(actor)

                    call.worker.add_done_callback(finished)
                else:
                    self._release(actor)
            log = logger.info if status == "ok" else logger.warning
            log(
                "sap_invocation",
                route=route,
                selected_by=details["selected_by"],
                tool=tool_name,
                object=target,
                consumer=actor,
                status=status,
                http_status=http_status,
                bytes=volume,
                input_bytes=call.input_bytes,
                latency_ms=round((time.monotonic() - start) * 1000),
                call_id=get_call_id(),
                transport_request=customizing.transport_request if customizing else None,
            )


class SAPTool(Tool):
    access: SAPAccess = Field(exclude=True)

    @model_validator(mode="after")
    def _check_document_access(self):
        if self.read_only and (getattr(self, "document_types", ()) or getattr(self, "customizing", None)):
            raise ValueError("A read-only SAP operation cannot create documents or execute customizing.")
        return self

    def __init__(self, **kwargs: Any) -> None:
        handler = self._invoke
        # Bind a deployment's business schema to the public tool signature.
        if input_model := kwargs.get("input_model"):

            async def invoke(payload: input_model):
                return await self._invoke(input_model.model_validate(payload))

            handler = invoke
        super().__init__(handler=handler, **kwargs)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        # SAP inputs and results (rows, SQL, documents) stay out of the host's event logs.
        self._log_events = set()
