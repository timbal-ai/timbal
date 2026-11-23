"""Execution limits and deployment-owned authorization for SAP framework tools."""

import asyncio
import hmac
import secrets
import threading
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    PrivateAttr,
    SecretStr,
    StringConstraints,
    model_validator,
)
from pydantic_core import to_json

from ...core.tool import Tool
from ...state import get_call_id, get_run_context

Ref = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Grant = Literal["B(i)(a)", "B(i)(b)", "B(i)(c)", "B(ii)", "B(iii)", "B(v)"]
Scalar = str | int | float | bool | None
logger = structlog.get_logger("timbal.sap")


def json_size(value: Any) -> int:
    return len(to_json(value, bytes_mode="base64"))


def _identity(consumer: str) -> tuple[str, str | None, bool]:
    from ...core.agent import Agent

    ctx = get_run_context()
    user = getattr(ctx, "sap_user_id", None)
    agent = None
    if ctx and get_call_id():
        for context, span in ctx.call_stack():
            if not user:
                user = getattr(context, "sap_user_id", None)
            if agent is None and isinstance(span.runnable, Agent):
                agent = span.path
    return agent or consumer, user, agent is not None


class SAPCustomizing(BaseModel):
    """Written designation of the target non-production client and its change transport."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    nonproduction_system: Ref
    client: Annotated[str, StringConstraints(pattern=r"^\d{3}$")]
    transport_request: Ref
    authorization: Ref


class SAPCall:
    """State of a single admitted operation, including any outstanding native driver call."""

    def __init__(self, access: "SAPAccess", consumer: str, user: str | None) -> None:
        self.access = access
        self.consumer = consumer
        self.user = user
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
    """Share one instance among tools using the same integration and quota in this worker."""

    model_config = ConfigDict(extra="forbid")
    catalog_id: Ref
    consumer: Ref = "workflow"
    agent_selected: bool = False
    scope_reference: Ref | None = None
    audit_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_hex(32)), exclude=True, repr=False)
    authorizations: dict[Grant, Ref] = Field(default_factory=dict)
    document_authorizations: dict[Ref, Ref] = Field(default_factory=dict)
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
        route: str,
        agent_route: str | None = None,
        hana: bool = False,
        document_types: tuple[str, ...] = (),
        customizing: SAPCustomizing | None = None,
    ) -> Any:
        consumer, user, agent = _identity(self.consumer)
        agent = agent or self.agent_selected
        route = agent_route if agent and agent_route else route
        key = self.audit_key.get_secret_value().encode()
        actor = hmac.digest(key, consumer.encode(), "sha256").hex()[:24]
        user_id = hmac.digest(key, str(user).encode(), "sha256").hex()[:24] if user else None
        call = SAPCall(self, actor, user_id)
        start = time.monotonic()
        admitted, volume, status = False, 0, "denied"
        http_status = None
        try:
            if hana and not self.scope_reference:
                raise PermissionError("The HANA integration needs a written scope reference.")
            if agent and route == "B(v)":
                raise PermissionError("SAP GUI scripts require a deterministic workflow.")
            if route.startswith("B") and not self.authorizations.get(route, "").strip():
                raise PermissionError(f"SAP route {route} requires a configured written authorization.")
            if any(not self.document_authorizations.get(doc, "").strip() for doc in document_types):
                raise PermissionError("SAP route B(iv) requires authorization for each declared document type.")
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
                catalog=self.catalog_id,
                route=route,
                tool=tool_name,
                object=target,
                consumer=actor,
                end_user=user_id,
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
