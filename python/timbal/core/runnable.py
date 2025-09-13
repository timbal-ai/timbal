import ast
import asyncio
import contextvars
import inspect
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from functools import cached_property, wraps
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
from ..state import (
    get_or_create_run_context,
    get_parent_call_id,
    set_call_id,
    set_parent_call_id,
    set_run_context,
)
from ..state.context import RunContext
from ..state.data import RunContextDataAccessAnalyzer
from ..state.tracing.trace import Trace
from ..types.events import (
    BaseEvent,
    ChunkEvent,
    Event,
    OutputEvent,
    StartEvent,
)
from ..utils import dump, sync_to_async_gen

logger = structlog.get_logger("timbal.core.runnable")


class CollectableAsyncGenerator:
    """
    Wrapper that adds collect() method to async generators.
    
    This class implements the async iterator protocol by wrapping an existing
    async generator and adding the ability to collect all results.
    """
    
    def __init__(self, async_gen: AsyncGenerator[Event, None], runnable: 'Runnable', kwargs: dict[str, Any]):
        """Initialize the CollectableAsyncGenerator wrapper.
        
        Args:
            async_gen: The original async generator from the Runnable's __call__ method
            runnable: Reference to the Runnable instance that created the generator
            kwargs: Original keyword arguments passed to the __call__ method
        """
        self._async_gen = async_gen  # The original async generator from __call__
        self._runnable = runnable    # Reference to the Runnable instance
        self._kwargs = kwargs        # Original kwargs passed to __call__
        self._collected = False      # Track if the generator has been fully consumed
        self._output_event = None    # Cache the OutputEvent when we encounter it
        self._events = []            # Cache all yielded events for later access
    
    def __aiter__(self):
        """Return the async iterator object (self).
        
        In Python's async iterator protocol:
        - __aiter__() is called when you do `async for item in obj:`
        - It should return an object that implements __anext__()
        - We return `self` because this class implements __anext__()
        
        This is equivalent to how regular iterators work:
        - __iter__() returns an iterator object with __next__()
        
        Returns:
            Self as the async iterator
        """
        return self
    
    async def __anext__(self):
        """Get the next item from the async iterator.
        
        This is called by `async for` loops and is the core of the async iterator protocol.
        When the generator is exhausted, it raises StopAsyncIteration to signal completion.
        
        We intercept each event and cache it in self._events for later use by collect().
        
        Returns:
            The next Event from the underlying async generator
            
        Raises:
            StopAsyncIteration: When the generator is exhausted
        """
        try:
            # Get the next event from the wrapped generator
            event = await self._async_gen.__anext__()
            # Cache this event so collect() can access it later
            self._events.append(event)
            # Cache OutputEvent directly when we encounter it
            if isinstance(event, OutputEvent) and event.path == self._runnable._path:
                self._output_event = event
            return event
        except StopAsyncIteration:
            # The generator is exhausted - mark as collected and re-raise
            self._collected = True
            raise  # This stops the `async for` loop
    
    async def aclose(self):
        """Close the generator gracefully.
        
        This is called when the generator needs to be cleaned up,
        either explicitly or when the generator is garbage collected.
        """
        await self._async_gen.aclose()
        self._collected = True
    
    async def collect(self) -> Any:
        """Collect the final output by consuming the entire stream.
        
        This method consumes all remaining events from the async generator
        and returns the final OutputEvent. It can be called multiple times
        safely - subsequent calls return the cached result.
        
        How this works:
        1. If we already have the OutputEvent cached, return it immediately
        2. Otherwise, consume remaining events using `async for event in self:`
           - This calls our __anext__() method which caches the OutputEvent when found
        3. Return the cached OutputEvent
        
        Returns:
            The final OutputEvent from the stream, or None if no OutputEvent was yielded
        """
        # If we already found and cached the OutputEvent, return it
        if self._output_event is not None:
            return self._output_event
        
        # Generator not fully consumed yet - consume remaining events
        try:
            # This calls our __aiter__() and __anext__() methods
            # __anext__() will cache the OutputEvent when it encounters it
            async for _ in self:
                # We could break early if we found the OutputEvent, but typically
                # the OutputEvent is the last event, so we consume everything
                pass
        except StopAsyncIteration:
            pass  # Expected when generator is exhausted
        
        # Return the cached OutputEvent (will be None if no OutputEvent was yielded)
        return self._output_event


def collectable(func):
    """Decorator that wraps async generator return with CollectableAsyncGenerator."""
    @wraps(func)
    def wrapper(self, **kwargs) -> CollectableAsyncGenerator:
        async_gen = func(self, **kwargs)
        return CollectableAsyncGenerator(async_gen, self, kwargs)
    return wrapper


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
    """List of sibling node names that the handler depends on via step_trace() calls."""
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
                name for name, param in sig.parameters.items()
                if param.default is inspect.Parameter.empty
                and param.kind != inspect.Parameter.VAR_POSITIONAL
                and param.kind != inspect.Parameter.VAR_KEYWORD
            ]
            if required_params:
                raise ValueError(f"Callable must not have any required parameters, got required: {required_params}")

        dependencies = []
        # TODO Fix errors with system prompt callables
        try:
            source_file = inspect.getsourcefile(fn)
            if not source_file:
                raise ValueError("Could not determine source file for runtime callable.")
            with open(source_file) as f:
                full_file_source = f.read()

            tree = ast.parse(full_file_source)
            target_node = None
            # Strategy 1: Find by function name (most reliable for decorated functions)
            func_name = fn.__name__
            for node in ast.walk(tree):
                if (isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and
                    hasattr(node, 'name') and node.name == func_name):
                    target_node = node
                    break
            # Strategy 2: If we have multiple functions with the same name, use source lines to narrow it down
            if not target_node:
                try:
                    source_lines, start_line = inspect.getsourcelines(fn)
                    # Look for FunctionDef nodes within a few lines of the start_line
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and
                            hasattr(node, 'name') and node.name == func_name and
                            hasattr(node, 'lineno') and
                            start_line <= node.lineno <= start_line + len(source_lines)):
                            target_node = node
                            break
                except: # noqa: E722
                    pass
            # Strategy 3: Fallback to line number matching for lambdas
            if not target_node:
                first_line = fn.__code__.co_firstlineno
                for node in ast.walk(tree):
                    if (hasattr(node, "lineno") and node.lineno == first_line and
                        isinstance(node, ast.Lambda)):
                        target_node = node
                        break
            if target_node:
                target_node_analyzer = RunContextDataAccessAnalyzer()
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
        properties = {
            k: v 
            for k, v in self.params_model_schema["properties"].items()
            if k in selected_params
        }
        return {
            **self.params_model_schema,
            "properties": properties,
        }

    
    @computed_field
    @cached_property
    def openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible tool schema for this runnable.
        
        Returns:
            A dictionary conforming to OpenAI's function calling schema format
        """
        formatted_params_model_schema = self.format_params_model_schema()
        openai_schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": formatted_params_model_schema,
            }
        }
        return openai_schema

    
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


    @collectable
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
            A CollectableAsyncGenerator that yields Events and provides collect() method
            
        Raises:
            ValidationError: If input parameters don't match the params_model
            Exception: Any exception raised during handler execution (captured in OutputEvent)
        """
        t0 = int(time.time() * 1000)

        _parent_call_id = get_parent_call_id()
        _call_id = uuid7(as_type="str").replace("-", "")
        set_call_id(_call_id)
        if self._is_orchestrator:
            set_parent_call_id(_call_id)

        # Generate new context or reset it if appropriate
        run_context = get_or_create_run_context()
        if not _parent_call_id and run_context._tracing:
            run_context = RunContext(parent_id=run_context.id)
            set_run_context(run_context)

        assert _call_id not in run_context._tracing, f"Call ID {_call_id} already exists in tracing."
        trace = Trace(
            path=self._path,
            call_id=_call_id,
            parent_call_id=_parent_call_id,
            t0=t0,
            metadata={**self.metadata}, # Shallow copy
        )
        run_context._tracing[_call_id] = trace

        # Execute the 'when' condition inside the appropriate runnable context
        if hasattr(self, "when") and self.when:
            should_run = await self._execute_runtime_callable(self.when["callable"], self.when["is_coroutine"])
            if not should_run:
                # Remove the trace entry. As if this was never run
                run_context._tracing.pop(_call_id)
                # Clean exit. The async for loop will complete normally but won't iterate over anything
                return

        # We store a preliminary version of the input and output in the trace, in case resolution fails
        input, output, error = kwargs, None, None
        trace.input = input
        trace._input_dump = None # ? await dump(input)
        trace._output_dump = None
        try:
            start_event = StartEvent(
                run_id=run_context.id,
                parent_run_id=run_context.parent_id,
                path=trace.path,
                call_id=trace.call_id,
                parent_call_id=trace.parent_call_id,
            )
            logger.info("start_event", **start_event.model_dump())
            yield start_event

            # We store the unvalidated input, as sent by the user. 
            # This will ensure full replayability of the run.
            # Resolve default params (executing any callable values)
            resolved_default_params = await self._resolve_default_params()
            input = {**resolved_default_params, **input}
            trace.input = input
            trace._input_dump = await dump(input)

            if self.pre_hook is not None:
                await self._execute_runtime_callable(self.pre_hook, self._pre_hook_is_coroutine)
            
            # Pydantic model_validate() does not mutate the input dict
            validated_input = dict(self.params_model.model_validate(input))

            handler_start = time.perf_counter()
            async_gen = None
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
                if self._is_orchestrator:
                    async_gen = self.handler(**validated_input)
                else:
                    async_gen = self.handler(**validated_input)
            
            if async_gen:
                collector = None
                async for chunk in async_gen:
                    # Get or create collector for this event type
                    if collector is None:
                        collector_type = get_collector_registry().get_collector_type(chunk)
                        if collector_type:
                            collector = collector_type(run_context, handler_start)
                    
                    if collector:
                        processed_chunk = collector.process(chunk)
                        # If the processed chunk is not None, it means we have streaming content
                        if processed_chunk is not None:
                            # If it's already a BaseEvent, it means we have already emitted it.
                            if not isinstance(processed_chunk, BaseEvent):
                                chunk_event = ChunkEvent(
                                    run_id=run_context.id,
                                    parent_run_id=run_context.parent_id,
                                    path=trace.path,
                                    call_id=trace.call_id,
                                    parent_call_id=trace.parent_call_id,
                                    chunk=processed_chunk,
                                )
                                logger.info("chunk_event", **chunk_event.model_dump())
                                yield chunk_event
                            else:
                                yield processed_chunk
                output = collector.collect() if collector else None
            trace.output = output
            trace._output_dump = await dump(output)
            
            if self.post_hook is not None:
                await self._execute_runtime_callable(self.post_hook, self._post_hook_is_coroutine)
            
        except Exception as err:
            error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }
            trace.error = error # No need to model dump the error. It's already a json compatible dict
        
        finally:
            t1 = int(time.time() * 1000)
            trace.t1 = t1
            output_event = OutputEvent(
                run_id=run_context.id,
                parent_run_id=run_context.parent_id,
                path=trace.path,
                call_id=trace.call_id,
                parent_call_id=trace.parent_call_id,
                input=trace.input,
                output=trace.output,
                error=trace.error,
                t0=trace.t0,
                t1=trace.t1,
                usage=trace.usage,
                metadata=trace.metadata,
            )
            output_event._input_dump = trace._input_dump
            output_event._output_dump = trace._output_dump
            if _parent_call_id is None:
                await run_context._save_tracing()
                # We don't want to propagate this between runs. We use this variable to check if we're at an entry point
                set_parent_call_id(None)
            logger.info("output_event", **output_event.model_dump())
            yield output_event


RunnableLike = Runnable | dict[str, Any] | Callable[..., Any]
"""Type alias for objects that can be used as tools for an agent or steps in a workflow."""
