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

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    TypeAdapter,
    ValidationInfo,
    computed_field,
    field_validator,
    model_serializer,
)
from uuid_extensions import uuid7

from ..collectors import get_collector_registry
from ..collectors.impl.timbal import TimbalCollector
from ..errors import EarlyExit, InterruptError
from ..state import (
    get_or_create_run_context,
    get_parent_call_id,
    set_call_id,
    set_parent_call_id,
    set_run_context,
)
from ..state.context import RunContext
from ..state.dependency_analyzer import RunContextDependencyAnalyzer
from ..state.tracing.span import Span
from ..types.events import (
    BaseEvent,
    ChunkEvent,
    Event,
    OutputEvent,
    StartEvent,
)
from ..types.events.delta import Custom, DeltaEvent, DeltaItem, TextDelta
from ..types.run_status import RunStatus
from ..utils import dump, sync_to_async_gen

logger = structlog.get_logger("timbal.core.runnable")

TIMBAL_DELTA_EVENTS = os.getenv("TIMBAL_DELTA_EVENTS", "false").lower() in [
    "true",
    "1",
    "t",
    "yes",
    "y",
    "enabled",
    "on",
]
if not TIMBAL_DELTA_EVENTS:
    logger.warning(
        "ChunkEvents will be deprecated in a future release. Enable TIMBAL_DELTA_EVENTS=true to use the new structured DeltaEvents system."
    )


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
    """The full path of the Runnable in the run context."""
    _is_orchestrator: bool = PrivateAttr()
    """Whether the Runnable is an orchestrator, i.e. it calls other Runnables."""
    _is_coroutine: bool = PrivateAttr()
    """Whether the Runnable handler is a coroutine."""
    _is_gen: bool = PrivateAttr()
    """Whether the Runnable handler is a generator."""
    _is_async_gen: bool = PrivateAttr()
    """Whether the Runnable handler is an async generator."""
    _dependencies: list[str] = PrivateAttr(default_factory=list)
    """List of sibling node names that the handler depends on via step_span() calls."""
    _default_fixed_params: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Dictionary of static default parameters."""
    _default_runtime_params: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    """Dictionary mapping parameter names to their callable functions and metadata."""
    _pre_hook_is_coroutine: bool | None = PrivateAttr()
    """Whether the pre_hook is a coroutine."""
    _pre_hook_dependencies: list[str] = PrivateAttr(default_factory=list)
    # ? Can we store all pre_hook related stuff together?
    _post_hook_is_coroutine: bool | None = PrivateAttr()
    """Whether the post_hook is a coroutine."""
    _post_hook_dependencies: list[str] = PrivateAttr(default_factory=list)
    # ? Can we store all post_hook related stuff together?
    _log_events: set[str] = PrivateAttr()
    """Which timbal events to log."""
    _bg_tasks: dict[str, asyncio.Task] = PrivateAttr(default_factory=dict)
    """Background tasks running for this runnable."""

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
            logger.error("Could not determine step dependencies for runtime callable.", exc_info=e)

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

        return {
            **self.params_model_schema,
            "properties": properties,
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
            del self._bg_tasks[task_id]
            if task.exception():
                return {"status": "error", "error": str(task.exception()), "events": events}
            else:
                return {"status": "completed", "result": task.result(), "events": events}
        else:
            return {"status": "running", "events": events}

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

    async def _resolve_default_params(self) -> dict[str, Any]:
        """Resolve default parameters by executing any callable values.

        Merges static default parameters with the results of executing
        runtime callable parameters in parallel.

        Returns:
            Dictionary containing resolved default parameters
        """
        resolved_params = dict(self._default_fixed_params)
        if not self._default_runtime_params:
            return resolved_params

        tasks = []
        callable_param_names = []
        for param_name, callable_info in self._default_runtime_params.items():
            tasks.append(self._execute_runtime_callable(callable_info["callable"], callable_info["is_coroutine"]))
            callable_param_names.append(param_name)

        results = await asyncio.gather(*tasks)
        for param_name, result in zip(callable_param_names, results, strict=False):
            resolved_params[param_name] = result
        return resolved_params

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
                    if TIMBAL_DELTA_EVENTS:
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
                    else:
                        if isinstance(event, TextDelta):
                            event = event.text_delta
                        elif isinstance(event, DeltaItem):
                            # Filter out all LLM emitted delta events that are not text
                            return None
                        event = ChunkEvent(
                            run_id=run_context.id,
                            parent_run_id=run_context.parent_id,
                            path=span.path,
                            call_id=span.call_id,
                            parent_call_id=span.parent_call_id,
                            chunk=event,
                        )
                    if event.type in self._log_events:
                        logger.info(event.type, **event.model_dump())
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

    @TimbalCollector.wrap
    async def __call__(self, **kwargs: Any) -> AsyncGenerator[Event, None]:
        """Execute the runnable with the given parameters.

        This is the main entry point for executing a runnable. It handles:
        - Parameter validation and merging with default_params
        - Run context management and tracing setup
        - Event streaming (StartEvent, ChunkEvents, OutputEvent)
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

        _parent_call_id = get_parent_call_id()
        _call_id = uuid7(as_type="str").replace("-", "")
        set_call_id(_call_id)
        if self._is_orchestrator:
            set_parent_call_id(_call_id)

        # Generate new context or reset it if appropriate
        run_context = get_or_create_run_context()
        if not _parent_call_id and run_context._trace:
            run_context = RunContext(parent_id=run_context.id)
            set_run_context(run_context)

        assert _call_id not in run_context._trace, f"Call ID {_call_id} already exists in trace."
        span = Span(
            path=self._path,
            call_id=_call_id,
            parent_call_id=_parent_call_id,
            t0=t0,
            metadata={**self.metadata},  # Shallow copy
            runnable=self,
        )
        run_context._trace[_call_id] = span

        # Execute the 'when' condition inside the appropriate runnable context
        if hasattr(self, "when") and self.when:
            should_run = await self._execute_runtime_callable(self.when["callable"], self.when["is_coroutine"])
            if not should_run:
                # Remove the span entry. As if this was never run
                run_context._trace.pop(_call_id)
                # Clean exit. The async for loop will complete normally but won't iterate over anything
                return

        # We store a preliminary version of the input and output in the span, in case resolution fails
        input, output, error = kwargs, None, None
        span.input = input
        span._input_dump = None  # ? await dump(input)
        span._output_dump = None
        collector = None
        try:
            start_event = StartEvent(
                run_id=run_context.id,
                parent_run_id=run_context.parent_id,
                path=span.path,
                call_id=span.call_id,
                parent_call_id=span.parent_call_id,
            )
            if start_event.type in self._log_events:
                logger.info(start_event.type, **start_event.model_dump())
            yield start_event

            # We store the unvalidated input, as sent by the user.
            # This will ensure full replayability of the run.
            # Resolve default params (executing any callable values)
            resolved_default_params = await self._resolve_default_params()
            input = {**resolved_default_params, **input}
            span.input = input
            span._input_dump = await dump(input)

            if self.pre_hook is not None:
                await self._execute_runtime_callable(self.pre_hook, self._pre_hook_is_coroutine)

            # Pydantic model_validate() does not mutate the input dict
            validated_input = dict(self.params_model.model_validate(input))

            # Background task
            if run_in_background:
                parent_span = run_context.parent_span()
                if not parent_span:
                    raise ValueError("Parent span not found. Cannot run in background.")
                task_id = uuid7(as_type="str").replace("-", "")
                event_queue = asyncio.Queue()

                async def _bg_handler_execution():
                    nonlocal output, collector
                    async for _, final_output, handler_collector in self._execute_handler(
                        validated_input, run_context, span, event_queue
                    ):
                        if handler_collector is not None:
                            collector = handler_collector
                        if final_output is not None:
                            output = final_output

                task = asyncio.create_task(_bg_handler_execution())

                # Store task with event queue in parent runnable if available
                parent_span.runnable._bg_tasks[task_id] = {"task": task, "event_queue": event_queue}
                output = {"task_id": task_id, "status": "running"}
            else:
                # Iterate over events from handler and yield them
                async for event, final_output, handler_collector in self._execute_handler(
                    validated_input, run_context, span
                ):
                    # Update collector immediately so it's available for interruption handling
                    if handler_collector is not None:
                        collector = handler_collector
                        span._collector = collector  # Store in span for interruption access
                    if event is not None:
                        yield event
                    if final_output is not None:
                        output = final_output

            span.status = RunStatus(
                code="success",
                reason=None,  # TODO
                message=None,  # TODO
            )
            # If the output is an OutputEvent, we extract the output
            # to avoid nesting an output event inside another output event
            if isinstance(output, OutputEvent):
                if output.status.code == "cancelled":
                    span.status = output.status
                output = output.output
            span.output = output

            # Restore the call context to this runnable before executing post_hook
            # This ensures post_hook modifies the correct span, not any nested ones
            set_call_id(_call_id)
            if self._is_orchestrator:
                set_parent_call_id(_call_id)
            if self.post_hook is not None:
                await self._execute_runtime_callable(self.post_hook, self._post_hook_is_coroutine)

            # Post hook might modify the output, so we dump afterwards
            span._output_dump = await dump(span.output)

        except EarlyExit as early_exit:
            span.status = RunStatus(code="cancelled", reason="early_exit", message=str(early_exit))
            span.output = None
            span._output_dump = None

        except (asyncio.CancelledError, InterruptError) as e:
            logger.warning("Interrupted", run_id=run_context.id, call_id=span.call_id)
            # Create a message with collected chunks or cancellation info
            if collector is not None:
                span.output = collector.result()
                span._output_dump = await dump(span.output)
            span.status = RunStatus(code="cancelled", reason="interrupted", message=str(e))
            yield InterruptError(_call_id)

        except Exception as err:
            error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }
            span.error = error  # No need to model dump the error. It's already a json compatible dict
            span.status = RunStatus(
                code="error",
                reason=None,  # TODO
                message=None,  # TODO
            )

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
            if _parent_call_id is None:
                # We don't want to propagate this between runs. We use this variable to check if we're at an entry point
                set_parent_call_id(None)
            if output_event.type in self._log_events:
                logger.info(output_event.type, **output_event.model_dump())
            yield output_event


RunnableLike = Runnable | dict[str, Any] | Callable[..., Any]
"""Type alias for objects that can be used as tools for an agent or steps in a workflow."""
