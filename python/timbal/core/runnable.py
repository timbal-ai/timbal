import ast
import asyncio
import contextvars
import hashlib
import inspect
import json
import logging
import os
import secrets
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from functools import cached_property
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
    model_serializer,
)
from uuid_extensions import uuid7

from ..collectors import get_collector_registry
from ..errors import (
    ApprovalPolicyError,
    EarlyExit,
    InterruptError,
    PauseRequired,
    RunCancelled,
    Suspend,
    WorkflowStepError,
)
from ..state import (
    get_call_id,
    get_parent_call_id,
    get_run_context,
    set_call_id,
    set_parent_call_id,
    set_run_context,
)
from ..state.context import RunContext
from ..state.dependency_analyzer import RunContextDependencyAnalyzer
from ..state.tracing.providers import TRACING_UNSET
from ..state.tracing.span import Span
from ..types.approval import ApprovalPolicyDecision, ApprovalResolution, Cancel
from ..types.events import (
    ApprovalEvent,
    BaseEvent,
    Event,
    InteractionEvent,
    OutputEvent,
    StartEvent,
)
from ..types.events.delta import Custom, DeltaEvent, DeltaItem
from ..types.message import Message
from ..types.run_status import RunStatus
from ..utils import dump, invalidate_message_dump_caches, sync_to_async_gen


def _get_logger():
    import structlog

    return structlog.get_logger("timbal.core.runnable")


_stdlib_events_logger = logging.getLogger("timbal.core.runnable")


def _events_logging_enabled() -> bool:
    """Cheap gate for per-event INFO logging.

    ``event.model_dump()`` is expensive and is evaluated eagerly as a call
    argument, so callers must check this BEFORE building the log kwargs.
    Timbal's structlog setup is stdlib-backed (see logs.setup_logging), so the
    stdlib effective level (which also respects ``logging.disable``) is
    authoritative.
    """
    return _stdlib_events_logger.isEnabledFor(logging.INFO)


def _collector_output_on_interrupt(collector: Any) -> Any:
    """Best-effort partial output when the handler async generator stops abruptly.

    Core collectors' ``result()`` implementations are trivial and do not raise, but
    user-defined collectors may validate in ``result()`` and fail on incomplete
    streams.  We log failures and fall back to known private fields on bundled
    collectors (duck-typed to avoid import cycles).
    """
    raw: Any
    try:
        raw = collector.result()
    except Exception as e:
        _get_logger().warning(
            "collector_result_failed_on_interrupt",
            collector_type=type(collector).__name__,
            error=str(e),
            exc_info=True,
        )
        raw = None
        ev = getattr(collector, "_output_event", None)
        if ev is not None:
            raw = ev
        else:
            msg = getattr(collector, "_message", None)
            if msg is not None:
                raw = msg
    if isinstance(raw, OutputEvent):
        return raw.output
    return raw


_Tool = None
_record_tool_requests = None


def _emit_default_tool_usage(runnable: Any) -> None:
    """On successful Tool completion, record ``{tool.name}:requests`` for billing defaults.

    Runs on every successful call — the imports are cached in module globals
    after the first invocation.
    """
    global _Tool, _record_tool_requests
    if _Tool is None:
        from ..state import _record_tool_requests as _record_fn
        from .tool import Tool

        _Tool = Tool
        _record_tool_requests = _record_fn

    if not isinstance(runnable, _Tool):
        return
    if not runnable.record_default_request_usage:
        return
    _record_tool_requests(runnable.name)


_TimbalCollector = None
"""Lazily imported TimbalCollector class — avoids importing collectors at module load."""


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

_BLOCKING_HANDLER_WARN_MS = float(os.getenv("TIMBAL_BLOCKING_WARN_MS", "100"))
"""Sync handlers running inline for longer than this (ms) get a one-time
warning suggesting offload_blocking=True / an async handler. Set to 0 to warn
on any sync handler; raise it (or set very high) to silence."""


ApprovalPolicy = bool | Callable[..., bool | ApprovalPolicyDecision | dict[str, Any]]
ApprovalPrompt = str | Callable[..., str | None] | None
ApprovalUi = dict[str, Any] | BaseModel | Callable[..., dict[str, Any] | BaseModel | None] | None


def _normalize_resume_values(raw: Any) -> dict[str, Any]:
    """Normalize caller-provided resume values keyed by id.

    The id is an approval_id (for an approval gate) or a suspension_id (for a
    ``suspend()`` call). Values are arbitrary: the approval gate normalizes its
    value to an ``ApprovalResolution`` lazily (see ``_coerce_approval_resolution``),
    while ``suspend()`` returns the value verbatim.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("resume must be a mapping of id to a resume value.")
    # A cancel may arrive as a Cancel instance or, over HTTP/JSON, as the tagged
    # dict {"type": "timbal.cancel", ...}. Normalize the wire form to a Cancel so
    # both the approval gate and suspend() see a real instance.
    normalized: dict[str, Any] = {}
    for k, v in raw.items():
        if not isinstance(v, Cancel) and Cancel.matches(v):
            v = Cancel.model_validate(v)
        normalized[str(k)] = v
    return normalized


def _coerce_approval_resolution(value: Any) -> ApprovalResolution:
    """Coerce a resume value landing on an approval gate into an ``ApprovalResolution``.

    Accepts ``bool``, ``dict``, or ``ApprovalResolution``. Raises ``ValueError``
    for anything else so a misrouted resume value fails fast at the gate rather
    than silently approving/denying.
    """
    if isinstance(value, ApprovalResolution):
        return value
    if isinstance(value, bool):
        return ApprovalResolution(approved=value)
    if isinstance(value, dict):
        return ApprovalResolution.model_validate(value)
    raise ValueError(
        "resume value for an approval gate must be a bool, dict, or ApprovalResolution; "
        f"got {type(value).__name__}."
    )


def _approval_id_for(path: str, input_dump: Any) -> str:
    """Compute a stable approval_id for an invocation.

    Hashes ``(path, validated_input)`` so the same decision resumes any retry
    of the same call. Treat the returned id as opaque: the derivation is an
    internal contract and may change across SDK versions, so do not persist
    ids beyond the lifetime of a single pending run.
    """
    payload = {"path": path, "input": input_dump}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


class Runnable(ABC, BaseModel):
    """Abstract base class for all runnable components in the Timbal framework.

    A Runnable represents an executable unit that can process inputs and produce outputs
    through an async generator interface. Runnables can be nested to form complex
    execution graphs and support various execution patterns (sync, async, generators).

    Key features:
    - Parameter validation using Pydantic models
    - Schema generation for LLM tool calling (OpenAI/Anthropic formats)
    - Event streaming with collection support for real-time processing
    - Comprehensive execution tracing and monitoring
    - Flexible parameter filtering and transformation
    - Runtime hooks (pre_hook/post_hook) for cross-cutting concerns
    - Context-aware execution with state management
    - Support for sync/async handlers with automatic context propagation
    - Nested execution patterns for complex workflows
    - Automatic error handling and recovery
    - Integration with collectors system for output processing
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str
    """The unique identifier for this runnable component."""
    description: str | None = None
    """Optional description of what this runnable does, used in LLM tool schemas."""
    metadata: dict[str, Any] = {}
    """Optional metadata for this runnable."""
    requires_approval: ApprovalPolicy = False
    """Whether this runnable invocation requires approval before handler execution.

    A callable receives the validated runnable input and may return a bool, dict,
    or ApprovalPolicyDecision.
    """
    approval_prompt: ApprovalPrompt = None
    """Optional approval prompt, or a callable that receives the validated runnable input."""
    approval_description: str | None = None
    """Optional approval description shown in ApprovalEvent."""
    approval_kind: str | None = None
    """Optional renderer discriminator for a rich approval card. Surfaced on
    ``ApprovalEvent.kind`` so the frontend can dispatch ``(kind, ui)`` to a
    component, exactly like it does ``(kind, payload)`` for interactions."""
    approval_ui: ApprovalUi = None
    """Optional structured approval card data. A dict or pydantic ``BaseModel``,
    or a callable receiving the (redacted) validated input by name and returning
    one. Pydantic models are dumped to JSON. The result is surfaced on
    ``ApprovalEvent.ui`` for the frontend to render. Presentation only — the
    handler still receives the unredacted input on resume."""
    approval_redactor: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    """Optional callable to redact sensitive fields before they reach any
    public approval surface (``ApprovalEvent.input``, persisted ``span.input``,
    ``span.metadata['approval']['input']``, exporters).

    Receives a copy of the validated input dict and must return a dict.
    The handler still runs with the unredacted validated input on resume.
    Takes precedence over ``approval_redact_keys`` when both are set.
    A redactor that raises or returns a non-dict falls back to a placeholder
    so the secret never leaks — see :meth:`_redact_validated_input`.
    """
    approval_redact_keys: list[str] | None = None
    """Ergonomic shortcut for ``approval_redactor``: each listed key in the
    validated input is replaced with ``"***"`` on the public approval
    surfaces. The handler still receives the unredacted input."""

    schema_params_mode: Literal["all", "required"] = "all"
    """Parameter inclusion mode: 'all' includes all params, 'required' only required ones."""
    schema_include_params: list[str] | None = None
    """Specific parameter names to include in the schema (additive to schema_params_mode)."""
    schema_exclude_params: list[str] | None = None
    """Specific parameter names to exclude from the schema."""

    default_params: dict[str, Any] = {}
    """Runtime default parameter injection.
    These parameters are added to the handler's parameters when the handler is called."""

    pre_hook: Callable[[], Any] | None = None
    """Pre-execution hook: parameterless callable; use get_run_context() for state.

    Runs after input resolution and before Pydantic validation, so the params
    model can see in-place changes to ``span.input`` (e.g. STT, middleware). A
    callable ``requires_approval`` policy is evaluated *after* validation, but
    this hook still runs on the first attempt, including when the run later cancels
    for ``approval_required``. Defer expensive work until after approval by using
    the handler or a nested Runnable.
    """
    post_hook: Callable[[], Any] | None = None
    """Post-execution hook for runtime processing. Must be a parameterless callable.
    Use get_run_context() to access execution state and output data.
    """

    background_mode: Literal["auto", "always", "never"] = "never"
    """Background execution mode"""

    offload_blocking: bool = False
    """If True, plain sync handlers run in the default thread pool so blocking
    code doesn't stall the event loop. Default False: sync handlers run inline
    on the event loop, which avoids a threadpool round-trip per call. Set this
    on tools whose sync handler blocks (network/disk I/O, heavy CPU) and that
    must not serialize concurrent runs — or better, make the handler async.
    Sync generator handlers always stream via the thread pool regardless of
    this flag."""

    tracing_provider: Any = Field(
        default=TRACING_UNSET,
        description=(
            "Tracing provider for runs started by this runnable. "
            "Unset (default) → auto-detect from env/config. "
            "None → disable tracing. "
            "A TracingProvider subclass → use that provider."
        ),
        exclude=True,
    )
    """Controls which tracing provider is used when this runnable creates a new RunContext.
    Only applies to the outermost runnable in a call chain — nested runnables inherit
    the RunContext (and provider) created by the outermost caller.
    """

    command: str | None = None
    """Optional command string that triggers automatic invocation of this runnable.

    When specified, this runnable will be automatically invoked when the command is received,
    bypassing LLM decision-making. This is particularly useful for Agents where you want
    direct command-based routing (e.g., '/help', '/search') without requiring the LLM to
    decide which tool to call.

    Note: This feature is primarily designed for Agent orchestrators and may not be
    applicable to all Runnable types.
    """

    # Workflow wiring. Declared as real (excluded-from-schema) fields rather than
    # relying on extra="allow" attributes: declared fields live in the instance
    # __dict__ and read ~8x faster than __pydantic_extra__ lookups, and these are
    # read on every step execution. Set by Workflow.step(); None outside workflows.
    previous_steps: Any = Field(default=None, exclude=True)
    """Names of steps this step waits for (set by Workflow.step)."""
    next_steps: Any = Field(default=None, exclude=True)
    """Names of steps that depend on this step (set by Workflow.step)."""
    previous_steps_kinds: Any = Field(default=None, exclude=True)
    """Per-source edge kinds ('ordering' | 'when' | 'param') for introspection."""
    when: Any = Field(default=None, exclude=True)
    """Optional {'callable', 'is_coroutine', ...} guard evaluated before the step runs."""

    # NOTE — hot runtime attributes are plain instance attributes assigned in
    # model_post_init, NOT pydantic PrivateAttr declarations. PrivateAttr reads
    # route through BaseModel.__getattr__ (~20x slower than __dict__ lookup) and
    # these are read multiple times on every call:
    #   _path, _is_orchestrator, _is_coroutine, _is_gen, _is_async_gen,
    #   _dependencies, _default_fixed_params, _default_runtime_params,
    #   _pre_hook_is_coroutine, _pre_hook_dependencies,
    #   _post_hook_is_coroutine, _post_hook_dependencies,
    #   _log_events, _bg_tasks
    # Do not add class-level annotations for them — pydantic would turn any
    # annotated underscore name back into a (slow) private attribute.

    @classmethod
    def _inspect_callable(
        cls,
        fn: Any,
        allow_required_params: bool = False,
        allow_coroutine: bool = True,
        allow_gen: bool = False,
        allow_async_gen: bool = False,
    ) -> dict[str, Any]:
        """Inspect a runtime callable, return if the callable is a coroutine and its step dependencies."""
        if not callable(fn):
            raise ValueError(f"fn must be a callable, got {type(fn)}")

        is_coroutine = inspect.iscoroutinefunction(fn)
        if not allow_coroutine and is_coroutine:
            raise NotImplementedError("Coroutine functions are not supported for runtime callables yet.")
        is_gen = inspect.isgeneratorfunction(fn)
        if not allow_gen and is_gen:
            raise NotImplementedError("Generator functions are not supported for runtime callables yet.")
        is_async_gen = inspect.isasyncgenfunction(fn)
        if not allow_async_gen and is_async_gen:
            raise NotImplementedError("Async generator functions are not supported for runtime callables yet.")

        if not allow_required_params:
            sig = inspect.signature(fn)
            required_params = [
                name
                for name, param in sig.parameters.items()
                if param.default is inspect.Parameter.empty
                and param.kind != inspect.Parameter.VAR_POSITIONAL
                and param.kind != inspect.Parameter.VAR_KEYWORD
            ]
            if required_params:
                raise ValueError(f"Callable must not have any required parameters, got required: {required_params}")

        dependencies = []
        try:
            source_file = inspect.getsourcefile(fn)
        except TypeError:
            source_file = None
        if source_file and source_file != "<string>":
            try:
                with open(source_file, encoding="utf-8") as f:
                    full_file_source = f.read()

                tree = ast.parse(full_file_source)
                func_name = fn.__name__

                # Collect all named function nodes matching func_name in one pass.
                candidates = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == func_name
                ]

                if len(candidates) == 1:
                    target_node = candidates[0]
                elif len(candidates) > 1:
                    # Narrow by source line range when multiple definitions share the same name.
                    source_lines, start_line = inspect.getsourcelines(fn)
                    target_node = next(
                        (n for n in candidates if start_line <= n.lineno <= start_line + len(source_lines)),
                        candidates[0],
                    )
                else:
                    # Fallback for lambdas: match by first line number.
                    first_line = fn.__code__.co_firstlineno
                    target_node = next(
                        (n for n in ast.walk(tree) if isinstance(n, ast.Lambda) and n.lineno == first_line),
                        None,
                    )

                if target_node:
                    analyzer = RunContextDependencyAnalyzer()
                    analyzer.visit(target_node)
                    dependencies = analyzer.dependencies
            except Exception:
                _get_logger().debug("Could not determine step dependencies for runtime callable.")

        return {
            "is_coroutine": is_coroutine,
            "is_gen": is_gen,
            "is_async_gen": is_async_gen,
            "dependencies": dependencies,
        }

    @field_validator("pre_hook", "post_hook")
    @classmethod
    def _validate_hooks(cls, v: Any) -> Callable[[], Any] | None:
        """Validate a hook, raise ValueError if invalid.

        Inspection *results* (is_coroutine, dependencies) are stored per instance
        in model_post_init. They used to be written to ``cls`` here, which let two
        instances of the same class with different sync/async hooks clobber each
        other's flags.
        """
        if v is None:
            return v
        cls._inspect_callable(v)
        return v

    def _prepare_default_params(self, default_params: dict[str, Any]) -> None:
        """Separates default_params into fixed (static) and runtime (callable) parameters."""
        if not isinstance(default_params, dict):
            raise ValueError("default_params must be a dictionary")
        for param_name, param_value in default_params.items():
            self.default_params[param_name] = param_value
            if callable(param_value):
                # Validate and store callable parameter
                inspect_result = self._inspect_callable(param_value)
                self._default_runtime_params[param_name] = {"callable": param_value, **inspect_result}
            else:
                # Store static parameter
                self._default_fixed_params[param_name] = param_value

    def model_post_init(self, __context: Any) -> None:
        """Initialize the Runnable after Pydantic model creation."""
        # Plain instance attributes (see NOTE above the field declarations).
        self._dependencies: list[str] = []
        self._default_fixed_params: dict[str, Any] = {}
        self._default_runtime_params: dict[str, dict[str, Any]] = {}
        self._bg_tasks: dict[str, Any] = {}
        self._blocking_warned: bool = False
        if self.pre_hook is not None:
            pre_hook_inspect = self._inspect_callable(self.pre_hook)
            self._pre_hook_is_coroutine: bool | None = pre_hook_inspect["is_coroutine"]
            self._pre_hook_dependencies: list[str] = pre_hook_inspect["dependencies"]
        else:
            self._pre_hook_is_coroutine = None
            self._pre_hook_dependencies = []
        if self.post_hook is not None:
            post_hook_inspect = self._inspect_callable(self.post_hook)
            self._post_hook_is_coroutine: bool | None = post_hook_inspect["is_coroutine"]
            self._post_hook_dependencies: list[str] = post_hook_inspect["dependencies"]
        else:
            self._post_hook_is_coroutine = None
            self._post_hook_dependencies = []

        log_events = os.getenv("TIMBAL_LOG_EVENTS", "START,OUTPUT").split(",")
        self._log_events = set(event.strip() for event in log_events)
        self._prepare_default_params(self.default_params)
        # Allow users to override the type in metadata if desired. Else, use the class name.
        if "type" not in self.metadata:
            self.metadata["type"] = self.__class__.__name__

    @staticmethod
    def _partial_schema(annotation: Any) -> dict[str, Any]:
        """Build a JSON schema for a type annotation, marking non-serialisable
        variants (e.g. Callable) instead of failing entirely.
        """
        import typing

        args = typing.get_args(annotation)
        if not args:
            return {}

        seen: set[str] = set()
        variants: list[dict[str, Any]] = []
        for arg in args:
            if arg is type(None):
                if "null" not in seen:
                    seen.add("null")
                    variants.append({"type": "null"})
                continue
            try:
                schema = TypeAdapter(arg).json_schema()
                schema.pop("title", None)
                variants.append(schema)
            except Exception:
                if "callable" not in seen:
                    seen.add("callable")
                    variants.append({"type": "callable"})

        if len(variants) == 1:
            return variants[0]
        return {"anyOf": variants}

    @staticmethod
    def _coerce_to_json_safe(value: Any) -> Any:
        """Coerce a config value into something json.dumps can serialise.

        Pydantic models are dumped, lists/tuples/sets/dicts are recursed,
        primitives pass through, everything else falls back to ``str(value)``
        (or a ``"<ClassName>"`` placeholder if even that fails).
        """
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(k): Runnable._coerce_to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [Runnable._coerce_to_json_safe(v) for v in value]
        if hasattr(value, "model_name"):
            chain = getattr(value, "model_name")
            if isinstance(chain, str):
                return chain
        try:
            return str(value)
        except Exception:
            return f"<{type(value).__name__}>"

    def _annotate_config(
        self,
        values: dict[str, Any],
        required: set[str] | None = None,
    ) -> dict[str, Any]:
        """Annotate config values with their JSON schema from Pydantic model fields.

        For each key in *values*, generates the full JSON schema from the
        field's ``FieldInfo`` (type + default + validators + description)
        using ``TypeAdapter`` and merges it with the current value. For union
        types with non-serialisable variants (e.g. ``str | Callable | None``),
        serialisable variants get their JSON schema and non-serialisable ones
        are marked with ``{"_type": "callable"}``.

        Fields listed in *required* are marked ``"required": True`` and their
        ``None`` variant is stripped from ``anyOf`` unions so the schema
        advertises only the concrete type.
        """
        import typing
        from typing import Annotated

        from ..platform.integrations import Integration

        required = required or set()
        model_fields = self.__class__.model_fields

        result: dict[str, Any] = {}
        for key, value in values.items():
            field_info = model_fields.get(key)
            field_schema: dict[str, Any] = {}
            if field_info is not None and field_info.annotation is not None:
                try:
                    field_schema = TypeAdapter(Annotated[field_info.annotation, field_info]).json_schema()
                    field_schema.pop("title", None)
                except Exception:
                    pass

                # TypeAdapter silently drops non-serialisable union variants
                # (e.g. Callable). Use _partial_schema to get the full picture.
                union_args = typing.get_args(field_info.annotation)
                if union_args:
                    full = self._partial_schema(field_info.annotation)
                    full_variants = full.get("anyOf", [full] if full else [])
                    schema_variants = field_schema.get("anyOf", [field_schema] if field_schema else [])
                    if len(full_variants) > len(schema_variants):
                        # Preserve FieldInfo metadata (default, description, etc.)
                        # from the TypeAdapter result, but use the full anyOf.
                        field_schema["anyOf"] = full_variants

            # For required fields, unwrap the anyOf to just the concrete type.
            if key in required and "anyOf" in field_schema:
                non_null = [v for v in field_schema["anyOf"] if v != {"type": "null"}]
                if len(non_null) == 1:
                    field_schema.pop("anyOf")
                    field_schema.pop("default", None)
                    field_schema.update(non_null[0])

            if isinstance(value, Integration):
                field_schema["value"] = str(value)
            else:
                field_schema["value"] = self._coerce_to_json_safe(value)
            result[key] = field_schema
        return result

    def get_config(self) -> dict[str, Any]:
        """Return the configurable parameters for this runnable.

        Each field is a dict with JSON schema properties (type, anyOf, etc.)
        plus a ``value`` key holding the current value. Override in subclasses
        to expose additional construction-time settings.
        """
        return self._annotate_config({"name": self.name, "description": self.description})

    @abstractmethod
    def nest(self, parent_path: str) -> None:
        """Set the nested path for this runnable within a parent context.

        This method is called when a runnable is nested inside another runnable
        (e.g., tools within an agent) to establish the hierarchical path.

        Args:
            parent_path: The path of the parent runnable
        """
        pass

    # NOTE: Pydantic's `@computed_field` and functool's `@cached_property` interfere
    # with the abstract method's ability to force an implementation at instantiation time
    # @computed_field
    # @cached_property
    @abstractmethod
    def params_model(self) -> BaseModel:
        """Return the Pydantic model defining the input parameters for this runnable.

        This model is used for:
        - Input validation when the runnable is called
        - Schema generation for LLM tool calling
        - Parameter filtering based on schema_params_mode, schema_include_params, schema_exclude_params

        Returns:
            A Pydantic BaseModel class defining the expected input parameters
        """
        pass

    @computed_field
    @cached_property
    def params_model_schema(self) -> dict[str, Any]:
        """Get the JSON schema for the params model.

        Returns:
            The JSON schema representation of the params_model
        """
        params_model_schema = self.params_model.model_json_schema()
        return params_model_schema

    # NOTE: Pydantic's `@computed_field` and functool's `@cached_property` interfere
    # with the abstract method's ability to force an implementation at instantiation time
    # @computed_field
    # @cached_property
    @abstractmethod
    def return_model(self) -> Any:
        """Return the type/model defining the expected output of this runnable.

        This is used for:
        - Type checking and validation
        - Schema generation for documentation
        - LLM integration where output types matter

        Returns:
            A type, Pydantic model, or other type annotation representing the output
        """
        pass

    @computed_field
    @cached_property
    def return_model_schema(self) -> dict[str, Any]:
        """Get the JSON schema for the return model.

        Returns:
            The JSON schema representation of the return_model
        """
        return_model_schema = TypeAdapter(self.return_model).json_schema()
        return return_model_schema

    @cached_property
    def _formatted_params_schema(self) -> dict[str, Any]:
        """Cached implementation of format_params_model_schema().

        Computed once per instance; depends on schema_params_mode,
        schema_include_params, schema_exclude_params, and background_mode —
        all of which are fixed at construction time.
        """
        selected_params = set()
        # Start with either all params or just required ones
        if self.schema_params_mode == "required":
            selected_params = set(self.params_model_schema.get("required", []))
        else:
            selected_params = set(self.params_model_schema["properties"].keys())

        # Add any explicitly included params
        if self.schema_include_params is not None:
            selected_params.update(self.schema_include_params)

        # Remove any explicitly excluded params
        if self.schema_exclude_params is not None:
            selected_params.difference_update(self.schema_exclude_params)

        # Filter properties to only include selected params
        properties = {}
        for k, v in self.params_model_schema["properties"].items():
            if k not in selected_params:
                continue
            # Simplify the structure of messages for LLM tool calling
            if v.get("title") == "TimbalMessage":
                properties[k] = {
                    "type": "string",
                    "description": "The input message to the agent.",
                }
            else:
                properties[k] = v

        # When background mode is auto, we'll expose this parameter to the LLM to let it decide
        if self.background_mode != "never":
            properties["run_in_background"] = {
                "type": "boolean",
                "default": True if self.background_mode == "always" else False,
                "description": "Run in the background",
            }

        # Strip excluded params from required
        required = [r for r in self.params_model_schema.get("required", []) if r in selected_params]

        return {
            **self.params_model_schema,
            "properties": properties,
            "required": required,
        }

    def format_params_model_schema(self) -> dict[str, Any]:
        """Format the parameter schema based on filtering rules.

        Applies the schema_params_mode, schema_include_params, and schema_exclude_params settings
        to filter which parameters are included in the final schema.

        Returns:
            A filtered JSON schema containing only the selected parameters
        """
        return self._formatted_params_schema

    @computed_field
    @cached_property
    def openai_chat_completions_schema(self) -> dict[str, Any]:
        """Tool schema in OpenAI Chat Completions format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": self._formatted_params_schema,
            },
        }

    @computed_field
    @cached_property
    def openai_responses_schema(self) -> dict[str, Any]:
        """Tool schema in OpenAI Responses API format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description or "",
            "parameters": self._formatted_params_schema,
        }

    @computed_field
    @cached_property
    def anthropic_schema(self) -> dict[str, Any]:
        """Tool schema in Anthropic format."""
        return {
            "name": self.name,
            "description": self.description or "",
            "input_schema": self._formatted_params_schema,
        }

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        """We use the simpler anthropic schema for serialization."""
        return self.anthropic_schema

    def get_background_task(self, task_id: str) -> dict[str, Any]:
        """Get the status and events of a background task."""
        if task_id not in self._bg_tasks:
            return {"status": "not_found", "events": []}

        task_info = self._bg_tasks[task_id]
        task = task_info["task"]

        # Get all available events
        events = []
        queue = task_info["event_queue"]
        while not queue.empty():
            try:
                events.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Determine status
        if task.done():
            # del self._bg_tasks[task_id] # Do not remove, to keep track of all background tasks
            if task.cancelled():
                return {"status": "cancelled", "events": events, "name": task_info["name"], "input": task_info["input"]}
            elif task.exception():
                return {
                    "status": "error",
                    "error": str(task.exception()),
                    "events": events,
                    "name": task_info["name"],
                    "input": task_info["input"],
                }
            else:
                return {
                    "status": "completed",
                    "result": task.result(),
                    "events": events,
                    "name": task_info["name"],
                    "input": task_info["input"],
                }
        else:
            return {"status": "running", "events": events, "name": task_info["name"], "input": task_info["input"]}

    async def _execute_runtime_callable(self, fn: Callable[..., Any], is_coroutine: bool) -> Any:
        """Execute a runtime callable handling async context automatically.

        Sync callables run inline on the event loop. These are param/`when`
        lambdas and hooks — documented as cheap accessors over the run context
        (e.g. ``step_span()``) — so a threadpool round-trip per evaluation is
        pure overhead. Running inline also lets them see the live context vars
        (the workflow sets ``parent_call_id`` right before evaluating them)
        without a context copy.
        """
        if is_coroutine:
            return await fn()
        return fn()

    async def _execute_approval_callable(self, fn: Callable[..., Any], validated_input: dict[str, Any]) -> Any:
        """Execute an approval policy callable with matching validated input parameters."""
        sig = inspect.signature(fn)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
            kwargs = validated_input
        else:
            kwargs = {name: validated_input[name] for name in sig.parameters if name in validated_input}

        if inspect.iscoroutinefunction(fn):
            return await fn(**kwargs)

        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()

        def fn_with_ctx():
            return ctx.run(fn, **kwargs)

        return await loop.run_in_executor(None, fn_with_ctx)

    async def _resolve_approval_decision(self, validated_input: dict[str, Any]) -> ApprovalPolicyDecision:
        """Normalize approval configuration for this invocation.

        Wraps **all** policy resolution errors — callable exceptions, invalid
        return types, malformed dicts that fail pydantic validation, prompt
        callable exceptions — in :class:`ApprovalPolicyError` so the gate
        surfaces a dedicated ``approval_policy_error`` reason rather than a
        generic handler error. The wrapping spans the whole function so any
        future code added here inherits the same contract.
        """
        try:
            raw_policy = self.requires_approval
            if callable(raw_policy):
                raw_decision = await self._execute_approval_callable(raw_policy, validated_input)
            else:
                raw_decision = raw_policy

            if isinstance(raw_decision, ApprovalPolicyDecision):
                decision = raw_decision
            elif isinstance(raw_decision, bool):
                decision = ApprovalPolicyDecision(required=raw_decision)
            elif isinstance(raw_decision, dict):
                decision = ApprovalPolicyDecision.model_validate(raw_decision)
            else:
                raise TypeError(
                    "requires_approval must be a bool or callable returning bool, dict, or ApprovalPolicyDecision; "
                    f"got {type(raw_decision).__name__}."
                )

            prompt = decision.prompt
            if decision.required and prompt is None and self.approval_prompt is not None:
                if callable(self.approval_prompt):
                    prompt = await self._execute_approval_callable(self.approval_prompt, validated_input)
                else:
                    prompt = self.approval_prompt

            # Resolve the structured card. Callables receive the *redacted* input
            # so a redactor/redact_keys config keeps secrets out of the card for
            # free; static dict/BaseModel values are used as-authored.
            ui = decision.ui
            if decision.required and ui is None and self.approval_ui is not None:
                if callable(self.approval_ui) and not isinstance(self.approval_ui, BaseModel):
                    redacted = self._redact_validated_input(validated_input)
                    ui = await self._execute_approval_callable(self.approval_ui, redacted)
                else:
                    ui = self.approval_ui
            if isinstance(ui, BaseModel):
                ui = ui.model_dump(mode="json")

            return ApprovalPolicyDecision(
                required=decision.required,
                prompt=prompt,
                description=decision.description or self.approval_description,
                kind=decision.kind or self.approval_kind,
                ui=ui,
                metadata=decision.metadata,
            )
        except ApprovalPolicyError:
            raise
        except Exception as exc:
            raise ApprovalPolicyError(self._path, exc) from exc

    def _redact_validated_input(self, validated_input: dict[str, Any]) -> dict[str, Any]:
        """Apply ``approval_redactor`` / ``approval_redact_keys`` to produce
        the public-facing input snapshot.

        The original ``validated_input`` is never mutated. The returned dict
        is what flows into :class:`ApprovalEvent`, ``span.input`` (when the
        gate fires), and ``span.metadata['approval']['input']``. The handler
        on resume keeps receiving the unredacted validated input.

        Defensive: a redactor that raises or returns a non-dict is treated
        as a config bug — we log and fall back to a placeholder so the
        secret never reaches a public surface.
        """
        if self.approval_redactor is None and not self.approval_redact_keys:
            return dict(validated_input)

        if self.approval_redactor is not None:
            try:
                redacted = self.approval_redactor(dict(validated_input))
            except Exception as exc:
                _get_logger().warning(
                    "approval_redactor raised; falling back to placeholder so the secret does not leak.",
                    runnable_path=self._path,
                    error=repr(exc),
                )
                return {"_approval_redaction_error": True}
            if not isinstance(redacted, dict):
                _get_logger().warning(
                    "approval_redactor must return a dict; falling back to placeholder.",
                    runnable_path=self._path,
                    returned_type=type(redacted).__name__,
                )
                return {"_approval_redaction_error": True}
            return redacted

        redacted = dict(validated_input)
        for key in self.approval_redact_keys or ():
            if key in redacted:
                redacted[key] = "***"
        return redacted

    async def _resolve_input_params(self, input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge fixed defaults, runtime defaults (lambdas), and input. Input takes priority."""
        input = input or {}
        resolved = dict(self._default_fixed_params)

        # Resolve runtime params (lambdas), skipping any already in input
        if self._default_runtime_params:
            pending = {
                name: self._execute_runtime_callable(info["callable"], info["is_coroutine"])
                for name, info in self._default_runtime_params.items()
                if name not in input
            }
            if pending:
                results = await asyncio.gather(*pending.values())
                resolved.update(zip(pending.keys(), results, strict=True))

        # Input takes priority over defaults
        resolved.update(input)
        return resolved

    async def _execute_simple(self, validated_input: dict[str, Any]) -> Any:
        """Execute a non-streaming (plain sync or coroutine) handler and return its output.

        Fast path used by __call__ for handlers that cannot yield events — skips
        the async-generator/tuple protocol of :meth:`_execute_handler` entirely.
        Subclasses may override to add fallback behavior (see Tool's proxy path).
        """
        if self._is_coroutine:
            return await self.handler(**validated_input)
        if self.offload_blocking:
            loop = asyncio.get_running_loop()
            ctx = contextvars.copy_context()

            def handler_func():
                return ctx.run(self.handler, **validated_input)

            return await loop.run_in_executor(None, handler_func)
        # Sync handlers run inline on the event loop: while one runs, every
        # other coroutine on this worker waits. Cheap handlers benefit (no
        # threadpool hop); blocking ones degrade concurrent request latency,
        # so flag them once with an actionable warning.
        t0 = time.perf_counter()
        try:
            return self.handler(**validated_input)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            if elapsed_ms >= _BLOCKING_HANDLER_WARN_MS and not self._blocking_warned:
                self._blocking_warned = True
                _get_logger().warning(
                    "Sync handler blocked the event loop; concurrent runs stall while it executes. "
                    "Make the handler async, or set offload_blocking=True to run it in a thread.",
                    runnable_path=self._path,
                    handler_elapsed_ms=round(elapsed_ms, 1),
                    threshold_ms=_BLOCKING_HANDLER_WARN_MS,
                )

    async def _execute_handler(
        self, validated_input: dict[str, Any], run_context: Any, span: Any, event_queue: asyncio.Queue | None = None
    ) -> AsyncGenerator[tuple[Event | None, Any, Any], None]:
        """Execute the handler with optional event streaming.

        Yields tuples of (event, output, collector) where output is None until the final iteration.
        Collector is yielded so it can be accessed for partial results on interruption.
        """
        handler_start = time.perf_counter()
        async_gen = None
        output = None
        collector = None

        if self._is_gen:
            loop = asyncio.get_running_loop()
            ctx = contextvars.copy_context()
            gen = self.handler(**validated_input)
            async_gen = sync_to_async_gen(gen, loop, ctx)
        elif self._is_async_gen:
            async_gen = self.handler(**validated_input)
        else:
            output = await self._execute_simple(validated_input)

        if async_gen:
            output = None
            # Peek at first element to determine collector type.
            # An empty generator (e.g. a Workflow with zero steps) is valid: leave
            # output as None and skip collector setup. We must catch
            # StopAsyncIteration here because letting it escape the body of
            # this async generator becomes a RuntimeError per PEP 479/525.
            try:
                first_chunk = await async_gen.__anext__()
            except StopAsyncIteration:
                first_chunk = None
                async_gen = None
        if async_gen:
            collector_type = get_collector_registry().get_collector_type(first_chunk)
            if collector_type:
                collector = collector_type(async_gen=async_gen, start=handler_start)

                # Yield collector immediately so it's available for interruption handling
                yield (None, None, collector)

                def process_event(event):
                    # If it's already a BaseEvent, it means we have already processed and logged it
                    if isinstance(event, BaseEvent):
                        return event
                    # Wrap non-delta events in a CustomItem
                    if not isinstance(event, DeltaItem):
                        # We use the runnable call id to aggregate events from the same call
                        event = Custom(id=span.call_id, data=event)
                    event = DeltaEvent(
                        run_id=run_context.id,
                        parent_run_id=run_context.parent_id,
                        path=span.path,
                        call_id=span.call_id,
                        parent_call_id=span.parent_call_id,
                        item=event,
                    )
                    if event.type in self._log_events and _events_logging_enabled():
                        _get_logger().info(event.type, **event.model_dump())
                    if event_queue:
                        event_queue.put_nowait(event)
                    return event

                # We need to manually process the first chunk, since we removed it from the generator.
                # Some collectors (e.g. ChatCompletionCollector) can emit multiple stream items
                # per source chunk — process() returns the first and queues the rest.
                first_event = collector.process(first_chunk)
                pending_events = [first_event] if first_event is not None else []
                pop_pending = getattr(collector, "pop_pending_stream_item", None)
                if callable(pop_pending):
                    while True:
                        pending_item = pop_pending()
                        if pending_item is None:
                            break
                        pending_events.append(pending_item)
                for raw_event in pending_events:
                    event = process_event(raw_event)
                    if event is not None:
                        yield (event, None, collector)
                # Process remaining events
                async for event in collector:
                    event = process_event(event)
                    if event is not None:
                        yield (event, None, collector)
                # Keep the final result
                output = collector.result()

        # Yield a final marker with the output and collector
        yield (None, output, collector)

    async def _apply_approval_gate(
        self,
        approval_decision: ApprovalPolicyDecision,
        validated_input: dict[str, Any],
        span: Any,
        run_context: Any,
    ) -> tuple[bool, "ApprovalEvent | None", dict[str, Any]]:
        """Run the human-approval gate for one invocation.

        All span status/output/metadata mutations for the gate happen here.

        Returns ``(proceed, approval_event, validated_input)``:

        - ``proceed=False, event=None`` — the gate ended the run (claimed by
          another worker, cancelled, or denied); the caller just returns.
        - ``proceed=False, event=ApprovalEvent`` — the gate is pending; the
          caller yields the event and returns.
        - ``proceed=True`` — approved; ``validated_input`` may carry the
          human's edit-on-approve overrides.
        """
        # ``approval_id`` MUST be derived from the unredacted input
        # so the resume call (which carries the full input) lands
        # on the same id as the original gate.
        input_dump = await dump(validated_input)
        approval_id = _approval_id_for(span.path, input_dump)

        # Compute the redacted view once and use it for every
        # public surface. The unredacted ``validated_input`` is
        # still what the handler sees on resume.
        redacted_input = self._redact_validated_input(validated_input)
        redaction_active = self.approval_redactor is not None or bool(self.approval_redact_keys)
        if redaction_active:
            span.input = redacted_input
            span._input_dump = await dump(redacted_input)

        try:
            input_schema = self.format_params_model_schema()
        except Exception:
            input_schema = None
        # Set by the agent on the START event when this gate fires inside a
        # tool call (see Agent._safe_tool_dispatch). None for direct calls.
        tool_call_id = span.metadata.get("tool_call_id")
        span.metadata["approval"] = {
            "id": approval_id,
            "required": True,
            "prompt": approval_decision.prompt,
            "description": approval_decision.description,
            "kind": approval_decision.kind,
            "ui": approval_decision.ui,
            "input_schema": input_schema,
            "tool_call_id": tool_call_id,
            "metadata": approval_decision.metadata,
            "input": redacted_input,
        }
        approval_resolution = None
        approval_cancel: Cancel | None = None
        if approval_id in run_context._resume_values:
            run_context._used_resume_ids.add(approval_id)
            raw_resume = run_context._resume_values[approval_id]
            # A Cancel value aborts the whole run rather than approving or
            # denying. It is not a valid ApprovalResolution input, so it
            # routes around coercion — but still flows through the durable
            # claim below so a cancel and an approve racing on two workers
            # can't both win.
            if isinstance(raw_resume, Cancel):
                approval_cancel = raw_resume
            else:
                approval_resolution = _coerce_approval_resolution(raw_resume)
        if approval_resolution is not None and approval_resolution.is_expired():
            span.metadata["approval"]["expired"] = True
            span.metadata["approval"]["expired_at"] = approval_resolution.expires_at
            # Counter fires for the *expired* resolution. The gate
            # then re-emits below, which adds a fresh :required tick.
            run_context.update_usage("approvals:expired", 1)
            approval_resolution = None

        # A decision OR a cancel claims the gate; the first claimer wins and
        # any later duplicate (resolve or cancel) stops here.
        if (approval_resolution is not None or approval_cancel is not None) and (
            run_context._tracing_provider is not None
        ):
            claimed = await run_context._tracing_provider.claim_approval(
                str(run_context.parent_id) if run_context.parent_id else None,
                approval_id,
                str(run_context.id),
            )
            if not claimed:
                span.metadata["approval"]["claim"] = {
                    "claimed": False,
                    "parent_id": str(run_context.parent_id) if run_context.parent_id else None,
                }
                span.status = RunStatus(
                    code="cancelled",
                    reason="approval_already_claimed",
                    message="Approval was already claimed by another resume run.",
                )
                span.output = {
                    "approval_id": approval_id,
                    "status": "approval_already_claimed",
                }
                span._output_dump = await dump(span.output)
                return False, None, validated_input

        if approval_cancel is not None:
            run_context.update_usage("approvals:cancelled", 1)
            message = approval_cancel.reason or "Run cancelled by user."
            span.metadata["approval"]["cancelled"] = True
            span.status = RunStatus(code="cancelled", reason="cancelled", message=message)
            span.output = {"approval_id": approval_id, "status": "cancelled", "reason": message}
            span._output_dump = await dump(span.output)
            return False, None, validated_input

        if approval_resolution is None:
            # Status/output MUST be set BEFORE the caller yields the event.
            # If the consumer breaks the stream right after seeing the
            # ApprovalEvent, GeneratorExit fires at the yield and we'd
            # otherwise persist this span as 'interrupted'.
            span.status = RunStatus(
                code="cancelled",
                reason="approval_required",
                message="Approval required before runnable execution.",
            )
            span.output = {
                "approval_id": approval_id,
                "status": "approval_required",
                "prompt": approval_decision.prompt,
            }
            span._output_dump = await dump(span.output)

            approval_event = ApprovalEvent(
                run_id=run_context.id,
                parent_run_id=run_context.parent_id,
                path=span.path,
                call_id=span.call_id,
                parent_call_id=span.parent_call_id,
                t0=int(time.time() * 1000),
                approval_id=approval_id,
                runnable_path=span.path,
                runnable_name=self.name,
                runnable_type=self.metadata.get("type", self.__class__.__name__),
                tool_call_id=tool_call_id,
                input=redacted_input,
                input_schema=input_schema,
                prompt=approval_decision.prompt,
                description=approval_decision.description,
                kind=approval_decision.kind,
                ui=approval_decision.ui,
                metadata=approval_decision.metadata,
            )
            run_context.update_usage("approvals:required", 1)
            return False, approval_event, validated_input

        # Resolution found and not expired — capture the audit
        # snapshot before deciding the gate's outcome. Typed fields
        # are surfaced under ``resolution`` so trace consumers can
        # query e.g. ``approval.resolution.approver_id`` directly.
        span.metadata["approval"]["resolution"] = {
            "approved": approval_resolution.approved,
            "reason": approval_resolution.reason,
            "approver_id": approval_resolution.approver_id,
            "comment": approval_resolution.comment,
            "decided_at": approval_resolution.decided_at,
            "expires_at": approval_resolution.expires_at,
            "override_input": approval_resolution.override_input,
            "metadata": approval_resolution.metadata,
        }
        if not approval_resolution.approved:
            run_context.update_usage("approvals:denied", 1)
            span.status = RunStatus(
                code="cancelled",
                reason="approval_denied",
                message=approval_resolution.reason or "Approval denied.",
            )
            span.output = {
                "approval_id": approval_id,
                "status": "approval_denied",
                "reason": approval_resolution.reason,
            }
            span._output_dump = await dump(span.output)
            return False, None, validated_input

        run_context.update_usage("approvals:approved", 1)

        # Edit-on-approve: merge the human's overrides over the proposed
        # input (override wins) and re-validate so the handler runs with
        # the corrected, type-checked values. The audit snapshot above
        # already recorded what was overridden.
        if approval_resolution.override_input:
            merged_input = {**validated_input, **approval_resolution.override_input}
            validated_input = dict(self.params_model.model_validate(merged_input))
            # Reflect what actually runs on the public input surface (redacted).
            span.input = self._redact_validated_input(validated_input)
            span._input_dump = await dump(span.input)
            span.metadata["approval"]["effective_input"] = span.input

        return True, None, validated_input

    async def _finalize_suspend(self, suspend_signal: Suspend, span: Any, run_context: Any) -> InteractionEvent:
        """Record a ``suspend()`` pause on the span and build its InteractionEvent.

        The same durable resume rails as approvals (parent_id + tracing
        provider) re-fire the handler with the supplied value on resume.
        """
        span.status = RunStatus(
            code="cancelled",
            reason="input_required",
            message="Input required to resume.",
        )
        span.output = {
            "suspension_id": suspend_signal.suspension_id,
            "status": "input_required",
            "kind": suspend_signal.kind,
            "payload": suspend_signal.payload,
        }
        span._output_dump = await dump(span.output)
        # Set by the agent on the START event when this suspension fires
        # inside a tool call. None for direct (non-agent) calls.
        tool_call_id = span.metadata.get("tool_call_id")
        span.metadata["suspension"] = {
            "id": suspend_signal.suspension_id,
            "kind": suspend_signal.kind,
            "payload": suspend_signal.payload,
            "response_schema": suspend_signal.response_schema,
            "tool_call_id": tool_call_id,
        }
        run_context.update_usage("suspends:required", 1)
        return InteractionEvent(
            run_id=run_context.id,
            parent_run_id=run_context.parent_id,
            path=span.path,
            call_id=span.call_id,
            parent_call_id=span.parent_call_id,
            t0=int(time.time() * 1000),
            interaction_id=suspend_signal.suspension_id,
            kind=suspend_signal.kind,
            runnable_path=span.path,
            runnable_name=self.name,
            runnable_type=self.metadata.get("type", self.__class__.__name__),
            tool_call_id=tool_call_id,
            payload=suspend_signal.payload,
            response_schema=suspend_signal.response_schema,
        )

    def _spawn_background_task(
        self, validated_input: dict[str, Any], input: dict[str, Any], run_context: Any, span: Any
    ) -> dict[str, Any]:
        """Spawn the handler as a background task registered on the parent runnable.

        Returns the ``{"task_id", "status"}`` placeholder output for the
        launching span. The task streams its events into a queue polled via
        :meth:`get_background_task` and records its real output on the span
        when it completes.
        """
        parent_span = run_context.parent_span()
        if not parent_span:
            raise ValueError("Parent span not found. Cannot run in background.")
        task_id = "".join(secrets.choice(ALPHABET) for _ in range(6))
        event_queue = asyncio.Queue()

        async def _bg_handler_execution():
            output = None
            try:
                async for _, final_output, _handler_collector in self._execute_handler(
                    validated_input, run_context, span, event_queue
                ):
                    if final_output is not None:
                        output = final_output

                # Post hook might modify the output, so we dump afterwards
                span._output_dump = await dump(output)
                span.output = output
                _emit_default_tool_usage(self)

                set_parent_call_id(span.parent_call_id)
                set_call_id(span.call_id)
                if self.post_hook is not None:
                    await self._execute_runtime_callable(self.post_hook, self._post_hook_is_coroutine)

            except asyncio.CancelledError:
                # Re-raise so asyncio marks the task as cancelled
                raise

        task = asyncio.create_task(_bg_handler_execution(), context=contextvars.copy_context())

        # Store task with event queue in parent runnable if available
        parent_span.runnable._bg_tasks[task_id] = {
            "task": task,
            "event_queue": event_queue,
            "name": self.name,
            "input": input,
        }
        return {"task_id": task_id, "status": "running"}

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the runnable, returning a TimbalCollector over its event stream.

        This is the public entry point. The collector is the API boundary: it
        supports ``async for`` iteration and ``.collect()``, and enriches the
        final OutputEvent with pending approvals/interactions. Framework
        internals (workflow steps, agent tools) iterate :meth:`_stream`
        directly, skipping the per-event collector layer.

        Args:
            **kwargs: Runtime parameters for the runnable execution.

        Returns:
            A TimbalCollector that yields Events and provides collect().

        Raises:
            ValidationError: If input parameters don't match the params_model
            Exception: Any exception raised during handler execution (captured in OutputEvent)
        """
        global _TimbalCollector
        if _TimbalCollector is None:
            from ..collectors.impl.timbal import TimbalCollector

            _TimbalCollector = TimbalCollector
        return _TimbalCollector(async_gen=self._stream(**kwargs))

    async def _stream(self, **kwargs: Any) -> AsyncGenerator[Event, None]:
        """Raw event stream for one runnable execution (internal entry point).

        Handles:
        - Parameter validation and merging with default_params
        - Run context management and tracing setup
        - Event streaming (StartEvent, DeltaEvents, OutputEvent)
        - Error handling and cleanup
        """
        t0 = int(time.time() * 1000)

        if self.background_mode == "auto":
            run_in_background = kwargs.pop("run_in_background", False)
        elif self.background_mode == "always":
            run_in_background = True
        else:
            run_in_background = False

        resume_values: dict[str, Any] = {}
        if "resume" in kwargs:
            resume_values.update(_normalize_resume_values(kwargs.pop("resume")))
        explicit_parent_id = kwargs.pop("parent_id", None)

        # Generate new context or reset it if appropriate
        _parent_call_id = get_parent_call_id()
        _call_id = get_call_id()
        run_context = get_run_context()
        # Entry snapshot for the finally block. A non-None entry call id means
        # we're executing inside another runnable's call on this task; if we
        # swap in a fresh RunContext below (top-level runnable invoked from a
        # handler, e.g. an agent called inside a tool), the caller's context
        # must be restored on exit or the swap leaks into the caller's run.
        _entry_run_context = run_context
        _entry_call_id = _call_id
        if run_context is None:
            run_context = RunContext(parent_id=explicit_parent_id, tracing_provider=self.tracing_provider)
            _parent_call_id = None
            _call_id = None
        elif "." not in self._path and run_context._trace:
            # Top-level runnable sees an existing context with traces.
            # If the root span has completed (t1 is set), this is a finished
            # previous run — chain session data via parent_id.
            # If the root span is still running (t1 is None), this context
            # belongs to a concurrent sibling — create a fresh context.
            # Inherit the parent's platform_config: a forked child run (linked
            # by parent_id, same process/deployment) shares it. Without this, the
            # fresh context re-resolves from env only and an explicitly-injected
            # platform_config is lost — breaking platform API calls (e.g. a
            # standalone Agent instantiated inside a step body).
            _inherited_platform_config = run_context.platform_config
            root = run_context.root_span()
            if root is not None and root.t1 is not None:
                run_context = RunContext(
                    parent_id=explicit_parent_id or run_context.id,
                    tracing_provider=self.tracing_provider,
                    platform_config=_inherited_platform_config,
                )
            else:
                run_context = RunContext(
                    parent_id=explicit_parent_id,
                    tracing_provider=self.tracing_provider,
                    platform_config=_inherited_platform_config,
                )
            _parent_call_id = None
            _call_id = None
        # Session data is loaded once per run; nested calls see it already set.
        if run_context._session_data is None:
            await run_context.get_session()
        previous_resume_values = dict(run_context._resume_values)
        run_context._resume_values.update(resume_values)
        set_run_context(run_context)

        _new_parent_call_id = _call_id
        _new_call_id: str = uuid7(as_type="hex")  # type: ignore
        set_parent_call_id(_new_parent_call_id)
        set_call_id(_new_call_id)

        assert _new_call_id not in run_context._trace, f"Call ID {_new_call_id} already exists in trace."
        span = Span(
            path=self._path,
            call_id=_new_call_id,
            parent_call_id=_new_parent_call_id,
            t0=t0,
            metadata={**self.metadata},  # Shallow copy
            runnable=self,
        )
        run_context._trace[_new_call_id] = span

        def _restore_context():
            """Restore this invocation's context vars.

            Between yields, another coroutine sharing the same asyncio Task
            may overwrite the context vars. Call this after every yield to
            reclaim ownership. Skips the writes if context is already correct
            (the common single-consumer case). The run-context check matters
            independently of the call-id one: a nested top-level runnable may
            have swapped in (and restored the ids around) a fresh RunContext.
            """
            if get_call_id() != _new_call_id or get_run_context() is not run_context:
                set_run_context(run_context)
                set_parent_call_id(_new_parent_call_id)
                set_call_id(_new_call_id)

        # We store a preliminary version of the input and output in the span, in case resolution fails
        input, output, error = kwargs, None, None
        span.input = input
        span._input_dump = None  # ? await dump(input)
        span._output_dump = None
        collector = None
        suspend_signal: Suspend | None = None
        _generator_closed = False
        try:
            start_event = StartEvent(
                run_id=run_context.id,
                parent_run_id=run_context.parent_id,
                path=span.path,
                call_id=span.call_id,
                parent_call_id=span.parent_call_id,
            )
            if start_event.type in self._log_events and _events_logging_enabled():
                _get_logger().info(start_event.type, **start_event.model_dump())
            yield start_event
            _restore_context()

            # Resolve input params (merging fixed defaults, runtime defaults, and provided input)
            # We then store the unvalidated input, as sent by the user to ensure full replayability of the run.
            input = await self._resolve_input_params(input)
            span.input = input
            span._input_dump = await dump(input)

            # Pydantic model_validate() does not mutate the input dict
            validated_input = dict(self.params_model.model_validate(input))

            # Fast path: requires_approval=False (the default) skips the whole
            # gate — no policy resolution, no ApprovalPolicyDecision construction.
            if self.requires_approval is not False:
                approval_decision = await self._resolve_approval_decision(validated_input)
                if approval_decision.required:
                    proceed, approval_event, validated_input = await self._apply_approval_gate(
                        approval_decision, validated_input, span, run_context
                    )
                    if approval_event is not None:
                        if approval_event.type in self._log_events and _events_logging_enabled():
                            _get_logger().info(approval_event.type, **approval_event.model_dump())
                        yield approval_event
                        _restore_context()
                    if not proceed:
                        return

            # pre_hook runs only when we're actually going to execute the
            # handler. We deliberately defer it past the approval gate so
            # external side-effects don't fire on gated/denied attempts.
            if self.pre_hook is not None:
                await self._execute_runtime_callable(self.pre_hook, self._pre_hook_is_coroutine)
                set_parent_call_id(_new_parent_call_id)
                set_call_id(_new_call_id)

            # Background task
            if run_in_background:
                output = self._spawn_background_task(validated_input, input, run_context, span)
            elif not self._is_async_gen and not self._is_gen:
                # Fast path: plain sync/coroutine handlers cannot yield events,
                # so skip the async-generator/tuple protocol entirely.
                try:
                    output = await self._execute_simple(validated_input)
                except Suspend as susp:
                    suspend_signal = susp
            else:
                # Iterate over events from handler and yield them
                try:
                    async for event, final_output, handler_collector in self._execute_handler(
                        validated_input, run_context, span
                    ):
                        # Update collector immediately so it's available for interruption handling
                        if handler_collector is not None:
                            collector = handler_collector
                        if event is not None:
                            # If a child gates/suspends, set our own status BEFORE the
                            # yield so that a consumer breaking the stream right after
                            # the ApprovalEvent/InteractionEvent doesn't end up with our
                            # span recorded as 'interrupted' (GeneratorExit clobbers
                            # late-set status). See test_agent_break_on_approval_event.
                            if span.status is None and isinstance(event, ApprovalEvent | InteractionEvent):
                                reason, message = (
                                    ("approval_required", "Approval required before runnable execution.")
                                    if isinstance(event, ApprovalEvent)
                                    else ("input_required", "Input required to resume.")
                                )
                                span.status = RunStatus(code="cancelled", reason=reason, message=message)
                            yield event
                            _restore_context()
                        if final_output is not None:
                            output = final_output
                except Suspend as susp:
                    # This runnable's own handler called suspend(). Pause: emit an
                    # InteractionEvent and end with status input_required. The same
                    # durable resume rails as approvals (parent_id + tracing provider)
                    # re-fire the handler with the supplied value on resume.
                    suspend_signal = susp

            if suspend_signal is not None:
                interaction_event = await self._finalize_suspend(suspend_signal, span, run_context)
                if interaction_event.type in self._log_events and _events_logging_enabled():
                    _get_logger().info(interaction_event.type, **interaction_event.model_dump())
                yield interaction_event
                _restore_context()
            else:
                # If the output is an OutputEvent, we extract the output
                # to avoid nesting an output event inside another output event
                status_already_set = False
                if isinstance(output, OutputEvent):
                    if output.status.code in {"cancelled", "error"}:
                        span.status = output.status
                        if output.error is not None:
                            span.error = output.error
                        status_already_set = True
                    output = output.output

                if not status_already_set:
                    # Determine stop_reason from Message output (LLM responses)
                    stop_reason = output.stop_reason if isinstance(output, Message) else None
                    span.status = RunStatus(code="success", reason=stop_reason, message=None)

                span.output = output

                if not run_in_background and span.status.code == "success":
                    _emit_default_tool_usage(self)

                set_parent_call_id(_new_parent_call_id)
                set_call_id(_new_call_id)
                if self.post_hook is not None and not run_in_background:
                    await self._execute_runtime_callable(self.post_hook, self._post_hook_is_coroutine)
                    # Hooks may mutate message content in place; drop any cached
                    # dumps on the output so the re-dump below sees the changes.
                    invalidate_message_dump_caches(span.output)

                # Post hook might modify the output, so we dump afterwards
                span._output_dump = await dump(span.output)

        except GeneratorExit:
            _generator_closed = True
            # Only overwrite status if no earlier branch (e.g. approval gate)
            # already set it. This keeps approval_required, early_exit etc.
            # intact when a consumer breaks the stream right after their event.
            if span.status is None:
                span.status = RunStatus(code="cancelled", reason="interrupted", message="")
                if collector is not None:
                    span.output = _collector_output_on_interrupt(collector)
            raise

        except RunCancelled as cancelled:
            # A human supplied a Cancel on the resume channel (approval or
            # suspend). Terminal: the whole run ends cancelled and nothing is
            # fed back to the model. Distinct from a denial (approval_denied).
            span.status = RunStatus(code="cancelled", reason="cancelled", message=cancelled.message)
            span.output = {"status": "cancelled", "reason": cancelled.message}
            span._output_dump = await dump(span.output)

        except EarlyExit as early_exit:
            reason = "early_exit" if early_exit.propagate else "early_exit_local"
            span.status = RunStatus(code="cancelled", reason=reason, message=early_exit.message)
            span.output = None
            span._output_dump = None

        except PauseRequired as pause_required:
            # A child runnable paused — either on an approval gate
            # (approval_required) or a suspend() call (input_required). It already
            # emitted its Approval/Interaction event; we just propagate the paused
            # status so this run also ends paused and persists for resume.
            output_event = pause_required.output_event
            span.status = output_event.status
            span.output = output_event.output
            span._output_dump = (
                output_event._output_dump if hasattr(output_event, "_output_dump") else await dump(span.output)
            )

        except ApprovalPolicyError as policy_err:
            original = policy_err.original
            span.status = RunStatus(
                code="error",
                reason="approval_policy_error",
                message=str(original),
            )
            span.error = {
                "type": type(original).__name__,
                "message": str(original),
                "traceback": "".join(traceback.format_exception(type(original), original, original.__traceback__)),
                "runnable_path": policy_err.runnable_path,
            }

        except (asyncio.CancelledError, InterruptError) as e:
            # Set status FIRST before any awaits. A second CancelledError can arrive
            # at any subsequent await (e.g. dump() below) and exit this handler before
            # the original assignment, leaving span.status=None and causing a Pydantic
            # ValidationError in the finally block (OutputEvent.status is required).
            span.status = RunStatus(code="cancelled", reason="interrupted", message=str(e))
            if isinstance(e, InterruptError):
                _get_logger().warning(
                    "Interrupted",
                    run_id=run_context.id,
                    call_id=span.call_id,
                    type="timbal.InterruptError",
                    from_call_id=e.call_id,
                )
                span.output = e.output
                span._output_dump = await dump(e.output)
            else:
                _get_logger().warning(
                    "Interrupted",
                    run_id=run_context.id,
                    call_id=span.call_id,
                    type="asyncio.CancelledError",
                )
                if collector is not None:
                    output = _collector_output_on_interrupt(collector)
                    span.output = output
                    span._output_dump = await dump(output)

        except WorkflowStepError as workflow_step_err:
            step_error = workflow_step_err.step_error
            span.status = RunStatus(
                code="error",
                reason="step_failed",
                message=step_error.get("message") if step_error else str(workflow_step_err),
            )
            if step_error is not None:
                span.error = step_error
            else:
                span.error = {
                    "type": type(workflow_step_err).__name__,
                    "message": str(workflow_step_err),
                    "traceback": traceback.format_exc(),
                    "step_name": workflow_step_err.step_name,
                }

        except Exception as err:
            # Set status FIRST before any operations that could raise (str(err),
            # traceback.format_exc()).  If those fail, span.status is already valid.
            span.status = RunStatus(
                code="error",
                reason=None,  # TODO
                message=None,  # TODO
            )
            error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }
            span.error = error  # No need to model dump the error. It's already a json compatible dict

        except (KeyboardInterrupt, SystemExit):
            span.status = RunStatus(code="cancelled", reason="interrupted", message="")
            raise

        except BaseException as err:
            # Any remaining BaseException subclass that is not an Exception —
            # e.g. custom BaseException subclasses from user code.
            span.status = RunStatus(code="error", reason=None, message=None)
            span.error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }
            raise

        finally:
            t1 = int(time.time() * 1000)
            span.t1 = t1
            output_event = OutputEvent(
                run_id=run_context.id,
                parent_run_id=run_context.parent_id,
                path=span.path,
                call_id=span.call_id,
                parent_call_id=span.parent_call_id,
                input=span.input,
                status=span.status,
                output=span.output,
                error=span.error,
                t0=span.t0,
                t1=span.t1,
                usage=span.usage,
                metadata=span.metadata,
            )
            output_event._input_dump = span._input_dump
            output_event._output_dump = span._output_dump
            await run_context._save_trace()
            # Warn about resume values that didn't match any gate or suspend()
            # so callers find typos / stale IDs instead of silently dropping
            # them. Only check the IDs introduced by THIS call; nested children
            # don't re-introduce them.
            if resume_values:
                unused = [rid for rid in resume_values if rid not in run_context._used_resume_ids]
                if unused:
                    _get_logger().warning(
                        "Unrecognized resume values ignored — these IDs did not "
                        "match any approval gate or suspend() during this run.",
                        unused_resume_ids=unused,
                        runnable_path=self._path,
                    )
            run_context._resume_values = previous_resume_values
            set_parent_call_id(_parent_call_id)
            set_call_id(_call_id)
            if _entry_call_id is not None:
                # Nested invocation: restore the caller's run context. A
                # top-level runnable invoked from a handler may have swapped in
                # a fresh RunContext above; with direct in-task iteration
                # (agent tool fast path, linear workflow steps) that swap would
                # otherwise leak into the caller's run. Top-level invocations
                # (entry call id None) deliberately leave their context set so
                # sequential same-task runs chain sessions implicitly.
                set_run_context(_entry_run_context)
            if output_event.type in self._log_events and _events_logging_enabled():
                _get_logger().info(output_event.type, **output_event.model_dump())
            if not _generator_closed:
                yield output_event


RunnableLike = Runnable | dict[str, Any] | Callable[..., Any]
"""Type alias for objects that can be used as tools for an agent or steps in a workflow."""
