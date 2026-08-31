import asyncio
import os
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from uuid_extensions import uuid7

from ..errors import SpanNotFound
from .config import PlatformConfig
from .tracing.providers import (
    TRACING_UNSET,
    InMemoryTracingProvider,
    PlatformTracingProvider,
    TracingProvider,
    _TracingProviderUnset,
)
from .tracing.span import Span
from .tracing.trace import Trace


def _get_logger():
    import structlog

    return structlog.get_logger("timbal.state.context")


class _NoDefault:
    """INTERNAL: Sentinel to distinguish 'no default provided' from 'default is None'."""

    def __repr__(self) -> str:
        return "<no default>"


_NO_DEFAULT = _NoDefault()


def _emit_sink_unpickle() -> None:
    """Unpickle target for :class:`_EmitSink` — a sink never survives serialization."""
    return None


class _EmitSink:
    """INTERNAL: per-call delivery channel for :meth:`RunContext.emit`.

    Attached to a span (``span._emit_sink``) for the duration of one runnable
    invocation, pointing wherever that call's processed handler events go:

    - Buffered (foreground): emitted events accumulate here and the owning
      ``Runnable._stream`` drains them at yield boundaries, interleaving them
      with the handler's own events. They never pass through the collector, so
      they cannot alter the call's output.
    - Forwarding (detached background child): events go straight to the
      child's background record log, which after detach is the only copy of
      its stream.

    Thread-safe: handlers may run off-loop (``offload_blocking`` /
    ``sync_to_async_gen`` executor threads). The owning loop is captured at
    creation; off-loop puts are marshalled with ``call_soon_threadsafe``.
    Fire-and-forget: no backpressure, no await, never raises — the same
    durability contract as the background ``put_nowait`` path.
    """

    __slots__ = ("_loop", "_forward", "_buffer", "closed")

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        forward: Callable[[Any], None] | None = None,
    ) -> None:
        self._loop = loop
        self._forward = forward
        # Lazily allocated on first emit — most calls never emit.
        self._buffer: deque[Any] | None = None
        self.closed = False

    def put(self, event: Any) -> None:
        """Deliver one event. Safe to call from any thread; never raises."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._put_on_loop(event)
            return
        try:
            self._loop.call_soon_threadsafe(self._put_on_loop, event)
        except RuntimeError:
            # Owning loop already closed — accepted fire-and-forget loss.
            _get_logger().debug("emit() after the owning event loop closed; event dropped.")

    def _put_on_loop(self, event: Any) -> None:
        if self.closed:
            _get_logger().debug("emit() after the call's stream closed; event dropped.", path=event.path)
            return
        if self._forward is not None:
            self._forward(event)
            return
        if self._buffer is None:
            self._buffer = deque()
        self._buffer.append(event)

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def drain(self) -> list[Any]:
        """Take all buffered events. Loop-thread only; atomic (no await points)."""
        buffer = self._buffer
        if not buffer:
            return []
        drained = list(buffer)
        buffer.clear()
        return drained

    def close(self) -> None:
        """Stop accepting events. Already-buffered events remain drainable."""
        self.closed = True

    # A sink lives on its span (``span._emit_sink``) and holds the event loop
    # and possibly futures. Serializing providers snapshot traces via
    # deepcopy/pickle; the copy has no live stream to deliver to, so the sink
    # collapses to None instead of dragging asyncio internals along.
    def __copy__(self) -> None:
        return None

    def __deepcopy__(self, memo: dict) -> None:
        return None

    def __reduce__(self) -> tuple:
        return (_emit_sink_unpickle, ())


class RunContext(BaseModel):
    """Runtime execution context shared across all components in a run.

    The RunContext provides a centralized location for:
    - Execution tracing and monitoring
    - Data sharing between steps and components
    - Usage tracking and statistics
    - Parent-child run relationships

    This context is automatically created and managed by the framework and is accessible
    through get_run_context() in runtime callables like default param callables and hooks.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    id: str = Field(
        default_factory=lambda: uuid7(as_type="hex"),  # type: ignore
        description="Unique identifier for the run.",
    )
    parent_id: str | None = Field(
        default=None,
        description="Whether this run is a direct child of another run.",
    )

    platform_config: PlatformConfig | None = Field(
        default=None,
        description="Platform configuration for the run.",
    )

    tracing_provider: Any = Field(
        default=TRACING_UNSET,
        description=(
            "Tracing provider to use for this run. "
            "TRACING_UNSET (default) → auto-detect from env/config. "
            "None → disable tracing entirely. "
            "A TracingProvider subclass → use that provider."
        ),
        exclude=True,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_platform_config(cls, data: Any) -> Any:
        """Normalize platform_config from legacy 'timbal_platform_config' key."""
        if isinstance(data, dict):
            # Only use legacy key if platform_config is not present
            if "timbal_platform_config" in data and "platform_config" not in data:
                data["platform_config"] = data.pop("timbal_platform_config")
            elif "timbal_platform_config" in data:
                del data["timbal_platform_config"]
        return data

    # NOTE — runtime state is stored as plain instance attributes assigned in
    # model_post_init, NOT pydantic PrivateAttr declarations. PrivateAttr reads
    # route through BaseModel.__getattr__ (~20x slower than __dict__ lookup) and
    # _trace/_resume_values/_tracing_provider are read on every runnable call:
    #   _base_path: Path | None
    #   _trace: Trace
    #   _tracing_provider: type[TracingProvider] | None
    #   _session_data: dict | None
    #   _bg_store: BackgroundTaskStore — session bag of detached children;
    #       inherited across sequential turns via parent_id (see background.py)
    #   _resume_values: dict — active resume values keyed by id, supplied via
    #       ``resume=`` to fulfill a paused run. The id is an approval_id for an
    #       approval gate (value normalized to ``ApprovalResolution``) or a
    #       suspension_id for a ``suspend()`` call (value is arbitrary).
    #   _used_resume_ids: set — resume ids that matched a gate or ``suspend()``
    #       during this run; used to warn about unrecognized resume values.
    # Do not add class-level annotations for them — pydantic would turn any
    # annotated underscore name back into a (slow) private attribute.

    def pending_approvals(self) -> list[dict[str, Any]]:
        """Return metadata for every span currently waiting on approval.

        Useful when an OutputEvent's status is cancelled/approval_required and
        the caller wants to enumerate every approval that needs a decision
        before retrying the run with ``resume={...}``. Includes
        ``expired``/``expired_at`` when the previous decision was rejected for
        TTL reasons so a UI can flag stale approvals to the operator.

        Tolerates both ``RunStatus`` instances and dicts, since traces loaded
        from JSONL/SQLite providers carry status as a dict.
        """
        pending: list[dict[str, Any]] = []
        for span in self._trace.values():
            status = span.status
            code = status.get("code") if isinstance(status, dict) else getattr(status, "code", None)
            reason = status.get("reason") if isinstance(status, dict) else getattr(status, "reason", None)
            if code == "cancelled" and reason == "approval_required":
                approval = (span.metadata or {}).get("approval")
                if approval and approval.get("id"):
                    entry = {
                        "approval_id": approval["id"],
                        "path": span.path,
                        "call_id": span.call_id,
                        "tool_call_id": approval.get("tool_call_id"),
                        "prompt": approval.get("prompt"),
                        "description": approval.get("description"),
                        "kind": approval.get("kind"),
                        "ui": approval.get("ui"),
                        "input_schema": approval.get("input_schema"),
                        "metadata": approval.get("metadata", {}),
                        "input": approval.get("input"),
                    }
                    if approval.get("expired"):
                        entry["expired"] = True
                        entry["expired_at"] = approval.get("expired_at")
                    pending.append(entry)
        return pending

    def pending_interactions(self) -> list[dict[str, Any]]:
        """Return metadata for every span currently suspended on input.

        Mirror of :meth:`pending_approvals` for ``suspend()``-based interactions
        (e.g. ``ask_user``, ``confirm``, or any custom interaction tool). When an
        OutputEvent's status is cancelled/input_required, enumerate every
        suspension so a UI can render each prompt and the caller can retry the
        run with ``resume={interaction_id: value, ...}``.

        Tolerates both ``RunStatus`` instances and dicts, since traces loaded
        from JSONL/SQLite providers carry status as a dict.
        """
        pending: list[dict[str, Any]] = []
        for span in self._trace.values():
            status = span.status
            code = status.get("code") if isinstance(status, dict) else getattr(status, "code", None)
            reason = status.get("reason") if isinstance(status, dict) else getattr(status, "reason", None)
            if code == "cancelled" and reason == "input_required":
                suspension = (span.metadata or {}).get("suspension")
                if suspension and suspension.get("id"):
                    pending.append(
                        {
                            "interaction_id": suspension["id"],
                            "kind": suspension.get("kind", "suspend"),
                            "path": span.path,
                            "call_id": span.call_id,
                            "tool_call_id": suspension.get("tool_call_id"),
                            "payload": suspension.get("payload", {}),
                            "response_schema": suspension.get("response_schema"),
                        }
                    )
        return pending

    def model_post_init(self, __context: Any) -> None:
        """Initialize the RunContext after Pydantic model creation.

        Sets up the tracing provider based on available configuration.
        Defaults to in-memory tracing if no custom provider is configured.

        If no platform_config is provided, attempts to resolve it from
        environment variables and ~/.timbal/ config files.
        """
        from .background import bind_background_store
        from .config_loader import resolve_platform_config

        # Plain instance attributes (see NOTE above).
        self._base_path: Path | None = None
        self._session_data: dict[str, Any] | None = None
        # Captured once per run so off-loop emit() can marshal onto it.
        # None until the first Runnable._stream on this context.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._resume_values: dict[str, Any] = {}
        self._used_resume_ids: set[str] = set()
        self._trace = Trace()
        # Inherit the parent session's background-task bag (usually None) so a
        # finished parent turn can still list/peek/cancel running builders.
        # Allocates nothing unless this session actually has children.
        bind_background_store(self)

        # Explicit provider set on the runnable — skip auto-detection entirely.
        # None means tracing is disabled; a class means use that provider.
        if not isinstance(self.tracing_provider, _TracingProviderUnset):
            if self.tracing_provider is not None and (
                not isinstance(self.tracing_provider, type) or not issubclass(self.tracing_provider, TracingProvider)
            ):
                raise TypeError(
                    f"tracing_provider must be a TracingProvider subclass, None, or TRACING_UNSET — "
                    f"got {self.tracing_provider!r}. "
                    f"Pass the class itself (e.g. MyProvider), not an instance. "
                    f"Use MyProvider.configured(...) to set provider-specific options."
                )
            self._tracing_provider = self.tracing_provider
            return

        self.platform_config = resolve_platform_config(self.platform_config)

        if self.platform_config:
            use_platform_traces = self.platform_config.sync_traces_enabled is not False
            if use_platform_traces and self.platform_config.subject and self.platform_config.subject.app_id:
                _get_logger().info(
                    f"Platform configuration found (subject: {self.platform_config.subject}). "
                    "Using platform tracing provider.",
                    event_name="tracing_setup",
                    run_id=self.id,
                )
                self._tracing_provider = PlatformTracingProvider
                return
            if self.platform_config.sync_traces_enabled is False:
                _get_logger().info(
                    "Sync traces disabled (sync_traces_enabled=False). Using in-memory tracing provider.",
                    event_name="tracing_setup",
                    run_id=self.id,
                )
            else:
                _get_logger().warning(
                    "Platform configuration found but no valid subject. "
                    "Please set TIMBAL_ORG_ID and TIMBAL_APP_ID environment variables to enable platform tracing.",
                    event_name="tracing_setup",
                    run_id=self.id,
                )
        _get_logger().info(
            "Using in-memory tracing provider.",
            event_name="tracing_setup",
            run_id=self.id,
        )
        self._tracing_provider = InMemoryTracingProvider

    async def _get_parent_trace(self) -> Trace | None:
        """Load the trace data for the parent run.

        INTERNAL METHOD: This method is intended for internal framework use.
        Use with caution as it involves async I/O operations and direct
        interaction with the tracing provider.

        Returns:
            The parent run's tracing data, or None if this is a root run or the
            provider could not produce it. Provider failures are deliberately
            not fatal: unlike the in-memory provider, the platform provider
            *raises* on a missing or unreachable run, and a dangling or flaky
            ``parent_id`` must degrade to "continue without memory" — the same
            path callers already take for ``None`` — not abort a run (or end a
            voice call) that can otherwise proceed.
        """
        if self.parent_id and self._tracing_provider is not None:
            try:
                return await self._tracing_provider.get(self)
            except Exception as e:
                _get_logger().error(
                    "Parent trace fetch failed. Continuing without it...",
                    parent_id=self.parent_id,
                    run_id=self.id,
                    error=str(e),
                )
                return None
        return None

    async def _save_trace(self) -> None:
        """Save the trace data for the run.

        INTERNAL METHOD: This method is intended for internal framework use.
        It persists the current run's trace data using the configured
        tracing provider. Manual calls to this method may interfere with
        the framework's automatic tracing lifecycle.
        """
        # Sync session data to root span before saving. Skip when the session
        # was never populated ({} from get_session()) and never synced before —
        # avoids a dump() per span completion for the common no-session case.
        root = self.root_span()
        if root is not None and self._session_data is not None and (self._session_data or root.session is not None):
            from ..utils import dump

            root.session = self._session_data
            root._session_dump = await dump(self._session_data)
        if self._tracing_provider is not None:
            await self._tracing_provider.put(self)

    async def get_session(self) -> dict[str, Any]:
        """Get session data that persists across runs."""
        if self._session_data is None:
            self._session_data = {}
            if self.parent_id and self._tracing_provider is not None:
                try:
                    trace = await self._tracing_provider.get(self)
                except Exception as e:
                    # Same degrade-to-empty as a missing trace below: the
                    # platform provider raises on a dangling/unreachable run
                    # where in-memory returns None, and session data is
                    # auxiliary — a run that can proceed without it should.
                    _get_logger().error(
                        "Parent trace fetch failed. Continuing without session data...",
                        parent_id=self.parent_id,
                        run_id=self.id,
                        error=str(e),
                    )
                    return self._session_data
                if trace is None or trace._root_call_id is None:
                    _get_logger().error(
                        "Parent trace not found. Continuing without session data...",
                        parent_id=self.parent_id,
                        run_id=self.id,
                    )
                    return self._session_data
                root_span = trace.get(trace._root_call_id)
                assert root_span is not None, "Root span not found"
                if root_span.session is not None:
                    self._session_data.update(root_span.session)
        return self._session_data

    def root_span(self) -> Span | None:
        """Get the root span of the trace (the first span with no parent)."""
        if self._trace._root_call_id is None:
            return None
        return self._trace.get(self._trace._root_call_id)

    def parent_of(self, span: Span) -> Span | None:
        """Get the parent of a span. If the span has no parent, it's the root."""
        if span.parent_call_id is None:
            return None
        return self._trace.get(span.parent_call_id)

    def current_span(self) -> Span:
        """Get the span for the current call."""
        from . import get_call_id

        call_id = get_call_id()
        span = self._trace.get(call_id)
        if not span:
            raise RuntimeError(f"Could not resolve current span for call ID {call_id}")
        return span

    def parent_span(self) -> Span:
        """Get the span for the parent call."""
        from . import get_parent_call_id

        parent_call_id = get_parent_call_id()
        parent_span = self._trace.get(parent_call_id)
        if not parent_span:
            raise RuntimeError(f"Could not resolve parent span for call ID {parent_call_id}")
        return parent_span

    def step_span(self, name: str, default: Any = _NO_DEFAULT) -> Span | Any:
        """Get the span for a neighbor step by name.

        Uses get_parent_call_id() to find sibling spans. This allows workflows to
        temporarily set parent_call_id before evaluating lambdas, enabling step_span
        to find the correct siblings without requiring a span for the current call.

        Args:
            name: The name of the step to find.
            default: Value to return if span not found. If not provided, raises SpanNotFound.

        Returns:
            The span for the requested step, or the default value if provided and not found.

        Raises:
            SpanNotFound: If the step's span doesn't exist and no default is provided.
        """
        from . import get_parent_call_id

        parent_call_id = get_parent_call_id()
        # Last-wins: looping steps (while_) produce multiple spans with the same
        # path; callers (while_ conditions, downstream lambdas) want the latest.
        # Non-looping steps still produce exactly one span, so behavior is
        # unchanged. Trace insertion order is chronological, so scanning the
        # underlying dict in reverse finds the latest match first and keeps the
        # early exit (this is a hot path for param lambdas / conditions).
        for span in reversed(self._trace.data.values()):
            if span.parent_call_id == parent_call_id and span.path.endswith("." + name):
                return span

        if isinstance(default, _NoDefault):
            raise SpanNotFound(name)
        return default

    def emit(self, data: Any) -> None:
        """Broadcast a custom DELTA event on the current call's event stream.

        Fire-and-forget, ambient, out-of-band: the event is interleaved with
        the current call's processed handler events — on the parent event
        stream for a foreground call, or in the background log/transcript for
        a detached background child — always ordered before that call's
        OUTPUT. It never passes through the handler's collector, so unlike a
        generator ``yield`` it cannot alter the call's output or the persisted
        span. The per-call sink is created on first emit, not on every
        invocation (the no-emit path is a slot check). Plain handlers flush
        at completion; generator handlers drain at chunk boundaries.

        Works from any handler shape (plain sync/coroutine, sync/async
        generator) and from any thread the framework runs handlers on
        (``offload_blocking``, ``sync_to_async_gen``). Never raises and never
        blocks: when the current call cannot be resolved or no live stream
        exists, the event is logged and dropped.

        Args:
            data: JSON-serializable payload. Wrapped in a
                :class:`~timbal.types.events.delta.Custom` item
                (``item.type == "custom"``) whose ``id`` is the current call id.
        """
        from ..types.events.delta import Custom, DeltaEvent
        from . import get_call_id

        call_id = get_call_id()
        span = self._trace.get(call_id) if call_id else None
        if span is None:
            _get_logger().debug("emit() outside an active call; event dropped.", run_id=self.id)
            return
        event = DeltaEvent(
            run_id=self.id,
            parent_run_id=self.parent_id,
            path=span.path,
            call_id=span.call_id,
            parent_call_id=span.parent_call_id,
            item=Custom(id=span.call_id, data=data),
        )
        # 1. This span already has a live sink — use it even if t1 is set.
        #    Background spawn finalizes the launching span, then rebinds
        #    ``_emit_sink`` to a forwarding sink; those emits must not drop.
        # 2. Still in-flight, no sink yet — create on THIS span. Do not walk
        #    parents here: a parent that already emitted would steal the
        #    child's first emit (ids would be the child's, drain would not).
        # 3. Finished, no live sink — walk parents (stale call id / hooks).
        sink = span._emit_sink
        if sink is not None and not sink.closed:
            sink.put(event)
            return
        if span.t1 is None:
            sink = self._ensure_emit_sink(span)
            if sink is not None:
                sink.put(event)
                return
        parent_id = span.parent_call_id
        while parent_id:
            parent = self._trace.get(parent_id)
            if parent is None:
                break
            sink = parent._emit_sink
            if sink is not None and not sink.closed:
                sink.put(event)
                return
            parent_id = parent.parent_call_id
        _get_logger().debug("emit() found no live event stream; event dropped.", run_id=self.id, path=event.path)

    def _ensure_emit_sink(self, span: Any) -> Any:
        """INTERNAL: attach a buffered sink to ``span`` if it does not have a live one."""
        sink = span._emit_sink
        if sink is not None and not sink.closed:
            return sink
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return None
            self._loop = loop
        sink = _EmitSink(loop)
        span._emit_sink = sink
        return sink

    def update_usage(self, key: str, value: int) -> None:
        """Update usage statistics for the current call and all parent calls.

        This method propagates usage statistics up the call stack, ensuring
        that parent components can track cumulative usage from their children.
        Commonly used for tracking token usage, API calls, or other metrics.

        Note: This method is safe under asyncio concurrency because it contains
        no await points — the entire read-modify-write is atomic with respect
        to the event loop. Do not add await points inside this method.

        Args:
            key: The usage metric key (e.g., 'tokens', 'api_calls')
            value: The value to add to the current usage for this key
        """
        from . import get_call_id

        call_id = get_call_id() or self._trace._root_call_id
        # Update usage for all parents in the call stack
        while call_id:
            assert call_id in self._trace, f"RunContext.update_usage: Call ID {call_id} not found in trace."
            span = self._trace[call_id]
            span.usage[key] = span.usage.get(key, 0) + value
            call_id = span.parent_call_id

    def resolve_cwd(self, path: str | None = None) -> Path:
        """Get the current working directory or resolve a path relative to it.

        This method handles:
        - Returning the base_path (CWD) when no path is provided
        - Environment variable expansion (e.g., $HOME)
        - User home directory expansion (e.g., ~)
        - Relative path resolution (relative to base_path/CWD if set)
        - Security validation (ensures path is within base_path/CWD if set)

        Args:
            path: Optional file path to resolve (can be relative or absolute).
                  If None, returns the current working directory (base_path).

        Returns:
            Resolved absolute Path object

        Raises:
            ValueError: If base_path is set and the resolved path is outside it
        """
        # If no path provided, return the CWD (base_path or current directory)
        if path is None:
            if self._base_path is not None:
                return self._base_path.resolve()
            else:
                return Path.cwd()

        # Expand environment variables and user home directory
        expanded_path = Path(os.path.expandvars(os.path.expanduser(path)))

        # Only enforce base_path restrictions if explicitly set
        if self._base_path is not None:
            base_path = self._base_path.resolve()

            # If path is relative, resolve it relative to base_path
            if not expanded_path.is_absolute():
                resolved_path = (base_path / expanded_path).resolve()
            else:
                resolved_path = expanded_path.resolve()

            # Security check: ensure resolved path is within base_path
            try:
                resolved_path.relative_to(base_path)
            except ValueError:
                raise ValueError(
                    f"Access denied: path '{path}' resolves to '{resolved_path}' "
                    f"which is outside the allowed base path '{base_path}'"
                ) from None

            return resolved_path
        else:
            # No base_path restriction - just resolve the path normally
            return expanded_path.resolve()
