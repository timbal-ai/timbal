import ast
import asyncio
import contextvars
import inspect
import os
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from functools import cached_property
from typing import Any, Literal

from nanoid import generate
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationInfo,
    computed_field,
    field_validator,
    model_serializer,
)
from uuid_extensions import uuid7

from ..collectors import get_collector_registry
from ..errors import EarlyExit, InterruptError
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
from ..types.events import (
    BaseEvent,
    Event,
    OutputEvent,
    StartEvent,
)
from ..types.events.delta import Custom, DeltaEvent, DeltaItem, TextDelta
from ..types.message import Message
from ..types.run_status import RunStatus
from ..utils import dump, sync_to_async_gen


def _get_logger():
    import structlog

    return structlog.get_logger("timbal.core.runnable")


def _timbal_collector_wrap(fn):
    """Lazy wrapper for @TimbalCollector.wrap — avoids importing the collector at module load."""
    from functools import wraps

    @wraps(fn)
    def wrapper(self, **kwargs):
        from ..collectors.impl.timbal import TimbalCollector

        return TimbalCollector(async_gen=fn(self, **kwargs))

    return wrapper


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


# TODO Add timeout
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

    _is_timbal_runnable: bool = True  # Marker for fast isinstance check in dump() without circular imports

    name: str
    """The unique identifier for this runnable component."""
    description: str | None = None
    """Optional description of what this runnable does, used in LLM tool schemas."""
    metadata: dict[str, Any] = {}
    """Optional metadata for this runnable."""

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
    """Pre-execution hook for runtime processing. Must be a parameterless callable.
    Use get_run_context() to access execution state and data.
    """
    post_hook: Callable[[], Any] | None = None
    """Post-execution hook for runtime processing. Must be a parameterless callable.
    Use get_run_context() to access execution state and output data.
    """

    background_mode: Literal["auto", "always", "never"] = "never"
    """Background execution mode"""

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

    _path: str = PrivateAttr()
    _is_orchestrator: bool = PrivateAttr()
    _is_coroutine: bool = PrivateAttr()
    _is_gen: bool = PrivateAttr()
    _is_async_gen: bool = PrivateAttr()
    _dependencies: list[str] = PrivateAttr(default_factory=list)
    _default_fixed_params: dict[str, Any] = PrivateAttr(default_factory=dict)
    _default_runtime_params: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _pre_hook_is_coroutine: bool | None = PrivateAttr()
    _pre_hook_dependencies: list[str] = PrivateAttr(default_factory=list)
    _post_hook_is_coroutine: bool | None = PrivateAttr()
    _post_hook_dependencies: list[str] = PrivateAttr(default_factory=list)
    _log_events: set[str] = PrivateAttr()
    _bg_tasks: dict[str, Any] = PrivateAttr(default_factory=dict)

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
            if not source_file:
                raise ValueError("Could not determine source file for runtime callable.")
            with open(source_file, encoding="utf-8") as f:
                full_file_source = f.read()

            tree = ast.parse(full_file_source)
            target_node = None
            # Strategy 1: Find by function name (most reliable for decorated functions)
            func_name = fn.__name__
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    and hasattr(node, "name")
                    and node.name == func_name
                ):
                    target_node = node
                    break
            # Strategy 2: If we have multiple functions with the same name, use source lines to narrow it down
            if not target_node:
                try:
                    source_lines, start_line = inspect.getsourcelines(fn)
                    # Look for FunctionDef nodes within a few lines of the start_line
                    for node in ast.walk(tree):
                        if (
                            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                            and hasattr(node, "name")
                            and node.name == func_name
                            and hasattr(node, "lineno")
                            and start_line <= node.lineno <= start_line + len(source_lines)
                        ):
                            target_node = node
                            break
                except:  # noqa: E722
                    pass
            # Strategy 3: Fallback to line number matching for lambdas
            if not target_node:
                first_line = fn.__code__.co_firstlineno
                for node in ast.walk(tree):
                    if hasattr(node, "lineno") and node.lineno == first_line and isinstance(node, ast.Lambda):
                        target_node = node
                        break
            if target_node:
                target_node_analyzer = RunContextDependencyAnalyzer()
                target_node_analyzer.visit(target_node)
                dependencies = target_node_analyzer.dependencies
        except Exception as e:
            _get_logger().error("Could not determine step dependencies for runtime callable.", exc_info=e)

        return {
            "is_coroutine": is_coroutine,
            "is_gen": is_gen,
            "is_async_gen": is_async_gen,
            "dependencies": dependencies,
        }

    @field_validator("pre_hook", "post_hook")
    @classmethod
    def _validate_hooks(cls, v: Any, info: ValidationInfo) -> Callable[[], Any] | None:
        """Validate a hook, raise ValueError if invalid."""
        if v is None:
            return v
        inspect_result = cls._inspect_callable(v)
        if info.field_name == "pre_hook":
            cls._pre_hook_is_coroutine = inspect_result["is_coroutine"]
            cls._pre_hook_dependencies = inspect_result["dependencies"]
        elif info.field_name == "post_hook":
            cls._post_hook_is_coroutine = inspect_result["is_coroutine"]
            cls._post_hook_dependencies = inspect_result["dependencies"]
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
                field_schema["value"] = value
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

    def format_params_model_schema(self) -> dict[str, Any]:
        """Format the parameter schema based on filtering rules.

        Applies the schema_params_mode, schema_include_params, and schema_exclude_params settings
        to filter which parameters are included in the final schema.

        Returns:
            A filtered JSON schema containing only the selected parameters
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

    @computed_field
    @cached_property
    def openai_chat_completions_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible tool schema for this runnable.

        Returns:
            A dictionary conforming to OpenAI's function calling schema format
        """
        formatted_params_model_schema = self.format_params_model_schema()
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": formatted_params_model_schema,
            },
        }
        return schema

    @computed_field
    @cached_property
    def openai_responses_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible tool schema for this runnable.

        Returns:
            A dictionary conforming to OpenAI's function calling schema format
        """
        formatted_params_model_schema = self.format_params_model_schema()
        schema = {
            "type": "function",
            "name": self.name,
            "description": self.description or "",
            "parameters": formatted_params_model_schema,
        }
        return schema

    @computed_field
    @cached_property
    def anthropic_schema(self) -> dict[str, Any]:
        """Generate Anthropic-compatible tool schema for this runnable.

        Returns:
            A dictionary conforming to Anthropic's tool calling schema format
        """
        formatted_params_model_schema = self.format_params_model_schema()
        anthropic_schema = {
            "name": self.name,
            "description": self.description or "",
            "input_schema": formatted_params_model_schema,
        }
        return anthropic_schema

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
        """Execute a runtime callable handling async context automatically."""
        if is_coroutine:
            return await fn()
        else:
            loop = asyncio.get_running_loop()
            ctx = contextvars.copy_context()

            def fn_with_ctx():
                return ctx.run(fn)

            return await loop.run_in_executor(None, fn_with_ctx)

    async def _resolve_input_params(self, input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge fixed defaults, runtime defaults (lambdas), and input. Input takes priority."""
        input = input or {}
        resolved = dict(self._default_fixed_params)

        # Resolve runtime params (lambdas), skipping any already in input
        if self._default_runtime_params:
            tasks = []
            callable_param_names = []
            for param_name, callable_info in self._default_runtime_params.items():
                if param_name in input:
                    continue  # Already provided, skip resolution
                tasks.append(self._execute_runtime_callable(callable_info["callable"], callable_info["is_coroutine"]))
                callable_param_names.append(param_name)

            if tasks:
                results = await asyncio.gather(*tasks)
                for param_name, result in zip(callable_param_names, results, strict=False):
                    resolved[param_name] = result

        # Input takes priority over defaults
        resolved.update(input)
        return resolved

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

        if not self._is_async_gen and not self._is_coroutine:
            loop = asyncio.get_running_loop()
            ctx = contextvars.copy_context()
            if self._is_gen:
                gen = self.handler(**validated_input)
                async_gen = sync_to_async_gen(gen, loop, ctx)
            else:

                def handler_func():
                    return ctx.run(self.handler, **validated_input)

                output = await loop.run_in_executor(None, handler_func)
        elif self._is_coroutine:
            output = await self.handler(**validated_input)
        else:
            async_gen = self.handler(**validated_input)

        if async_gen:
            output = None
            # Peek at first element to determine collector type
            first_chunk = await async_gen.__anext__()
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
                    if event.type in self._log_events:
                        _get_logger().info(event.type, **event.model_dump())
                    if event_queue:
                        event_queue.put_nowait(event)
                    return event

                # We need to manually process the first chunk, since we removed it from the generator
                first_event = collector.process(first_chunk)
                if first_event is not None:
                    first_event = process_event(first_event)
                    if first_event is not None:
                        yield (first_event, None, collector)
                # Process remaining events
                async for event in collector:
                    event = process_event(event)
                    if event is not None:
                        yield (event, None, collector)
                # Keep the final result
                output = collector.result()

        # Yield a final marker with the output and collector
        yield (None, output, collector)

    @_timbal_collector_wrap
    async def __call__(self, **kwargs: Any) -> AsyncGenerator[Event, None]:
        """Execute the runnable with the given parameters.

        This is the main entry point for executing a runnable. It handles:
        - Parameter validation and merging with default_params
        - Run context management and tracing setup
        - Event streaming (StartEvent, DeltaEvents, OutputEvent)
        - Error handling and cleanup
        - Integration with the collectors system

        The @collectable decorator wraps the returned async generator to add
        a .collect() method for easy result collection.

        Args:
            **kwargs: Runtime parameters for the runnable execution.

        Returns:
            A BaseCollector that yields Events and provides collect() method

        Raises:
            ValidationError: If input parameters don't match the params_model
            Exception: Any exception raised during handler execution (captured in OutputEvent)
        """
        t0 = int(time.time() * 1000)

        if self.background_mode == "auto":
            run_in_background = kwargs.pop("run_in_background", False)
        elif self.background_mode == "always":
            run_in_background = True
        else:
            run_in_background = False

        # Generate new context or reset it if appropriate
        _parent_call_id = get_parent_call_id()
        _call_id = get_call_id()
        run_context = get_run_context()
        if run_context is None:
            run_context = RunContext(tracing_provider=self.tracing_provider)
            _parent_call_id = None
            _call_id = None
        elif "." not in self._path and run_context._trace:
            # Top-level runnable sees an existing context with traces.
            # If the root span has completed (t1 is set), this is a finished
            # previous run — chain session data via parent_id.
            # If the root span is still running (t1 is None), this context
            # belongs to a concurrent sibling — create a fresh context.
            root = run_context.root_span()
            if root is not None and root.t1 is not None:
                run_context = RunContext(parent_id=run_context.id, tracing_provider=self.tracing_provider)
            else:
                run_context = RunContext(tracing_provider=self.tracing_provider)
            _parent_call_id = None
            _call_id = None
        await run_context.get_session()
        set_run_context(run_context)

        _new_parent_call_id = _call_id
        _new_call_id: str = uuid7(as_type="str").replace("-", "")  # type: ignore
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
            (the common single-consumer case).
            """
            if get_call_id() != _new_call_id:
                set_run_context(run_context)
                set_parent_call_id(_new_parent_call_id)
                set_call_id(_new_call_id)

        # We store a preliminary version of the input and output in the span, in case resolution fails
        input, output, error = kwargs, None, None
        span.input = input
        span._input_dump = None  # ? await dump(input)
        span._output_dump = None
        collector = None
        _generator_closed = False
        try:
            start_event = StartEvent(
                run_id=run_context.id,
                parent_run_id=run_context.parent_id,
                path=span.path,
                call_id=span.call_id,
                parent_call_id=span.parent_call_id,
            )
            if start_event.type in self._log_events:
                _get_logger().info(start_event.type, **start_event.model_dump())
            yield start_event
            _restore_context()

            # Resolve input params (merging fixed defaults, runtime defaults, and provided input)
            # We then store the unvalidated input, as sent by the user to ensure full replayability of the run.
            input = await self._resolve_input_params(input)
            span.input = input
            span._input_dump = await dump(input)

            if self.pre_hook is not None:
                await self._execute_runtime_callable(self.pre_hook, self._pre_hook_is_coroutine)
                set_parent_call_id(_new_parent_call_id)
                set_call_id(_new_call_id)

            # Pydantic model_validate() does not mutate the input dict
            validated_input = dict(self.params_model.model_validate(input))

            # Background task
            if run_in_background:
                parent_span = run_context.parent_span()
                if not parent_span:
                    raise ValueError("Parent span not found. Cannot run in background.")
                # task_id = uuid7(as_type="str").replace("-", "")
                task_id = generate(alphabet=ALPHABET, size=6)
                event_queue = asyncio.Queue()

                async def _bg_handler_execution():
                    nonlocal output, collector
                    try:
                        async for _, final_output, handler_collector in self._execute_handler(
                            validated_input, run_context, span, event_queue
                        ):
                            if handler_collector is not None:
                                collector = handler_collector
                            if final_output is not None:
                                output = final_output

                        # Post hook might modify the output, so we dump afterwards
                        span._output_dump = await dump(output)
                        span.output = output

                        set_parent_call_id(_new_parent_call_id)
                        set_call_id(_new_call_id)
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
                output = {"task_id": task_id, "status": "running"}
            else:
                # Iterate over events from handler and yield them
                async for event, final_output, handler_collector in self._execute_handler(
                    validated_input, run_context, span
                ):
                    # Update collector immediately so it's available for interruption handling
                    if handler_collector is not None:
                        collector = handler_collector
                    if event is not None:
                        yield event
                        _restore_context()
                    if final_output is not None:
                        output = final_output

            # If the output is an OutputEvent, we extract the output
            # to avoid nesting an output event inside another output event
            status_already_set = False
            if isinstance(output, OutputEvent):
                if output.status.code == "cancelled":
                    span.status = output.status
                    status_already_set = True
                output = output.output

            if not status_already_set:
                # Determine stop_reason from Message output (LLM responses)
                stop_reason = output.stop_reason if isinstance(output, Message) else None
                span.status = RunStatus(code="success", reason=stop_reason, message=None)

            span.output = output

            set_parent_call_id(_new_parent_call_id)
            set_call_id(_new_call_id)
            if self.post_hook is not None and not run_in_background:
                await self._execute_runtime_callable(self.post_hook, self._post_hook_is_coroutine)

            # Post hook might modify the output, so we dump afterwards
            span._output_dump = await dump(span.output)

        except GeneratorExit:
            _generator_closed = True
            span.status = RunStatus(code="cancelled", reason="interrupted", message="")
            raise

        except EarlyExit as early_exit:
            reason = "early_exit" if early_exit.propagate else "early_exit_local"
            span.status = RunStatus(code="cancelled", reason=reason, message=early_exit.message)
            span.output = None
            span._output_dump = None

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
                    output = collector.result()
                    # When a tool is cancelled, the CancelledError is raised by the Agent._multiplex_tools
                    if isinstance(output, OutputEvent):
                        output = output.output
                    span.output = output
                    span._output_dump = await dump(output)

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
            set_parent_call_id(_parent_call_id)
            set_call_id(_call_id)
            if output_event.type in self._log_events:
                _get_logger().info(output_event.type, **output_event.model_dump())
            if not _generator_closed:
                yield output_event


RunnableLike = Runnable | dict[str, Any] | Callable[..., Any]
"""Type alias for objects that can be used as tools for an agent or steps in a workflow."""
