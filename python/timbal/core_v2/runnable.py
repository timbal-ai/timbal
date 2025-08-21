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
    computed_field,
    field_validator,
    model_serializer,
)
from uuid_extensions import uuid7

from ..collectors import get_collector_registry
from ..collectors.utils import sync_to_async_gen
from ..state import get_call_id, get_run_context, set_call_id, set_run_context
from ..state.context import RunContext
from ..types.events import (
    BaseEvent,
    ChunkEvent,
    Event,
    OutputEvent,
    StartEvent,
)

logger = structlog.get_logger("timbal.core_v2.runnable")


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
    - Event streaming with collection support
    - Execution tracing and monitoring
    - Flexible parameter filtering and transformation
    
    Hook Patterns:
        Both pre_hook and post_hook follow a middleware-style pattern with in-place modification.
        
        IMPORTANT: Hooks receive mutable references and should modify them in-place.
        No return value is expected - all changes happen by mutating the passed objects.
        
        Basic patterns:
        • Reading data: value = data['key']
        • Modifying data: data['key'] = new_value
        • Adding data: data['new_key'] = computed_value
        • Removing data: del data['unwanted_key']
        • Logging/monitoring: logger.info(f"Processing {data}")
        
        Advanced patterns:
        • Access execution context: run_context = get_run_context()
        • Store data for other hooks: run_context.data['timestamp'] = time.time()
        • Cross-hook communication: user_id = run_context.data.get('user_id')
        • Conditional processing: if run_context.data.get('debug_mode'): add_debug_info()
        • Performance tracking: run_context.data['start_time'] = time.time()
        
        Functional-style transformations while maintaining in-place semantics:
            # Replace entire dict contents
            transformed = transform_data(dict(data))
            data.clear()
            data.update(transformed)
            
            # For lists/other mutables
            if isinstance(data, list):
                data[:] = transform_list(data)
        
        Common use cases:
        • Input validation/normalization before handler execution
        • Adding computed parameters (timestamps, user context, etc.)
        • Output transformation and metadata addition
        • Logging, monitoring, and debugging
        • Conditional parameter injection based on context
        • Data filtering and sanitization
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str
    """The unique identifier for this runnable component."""
    description: str | None = None
    """Optional description of what this runnable does, used in LLM tool schemas."""
    params_mode: Literal["all", "required"] = "all"
    """Parameter inclusion mode: 'all' includes all params, 'required' only required ones."""
    include_params: list[str] | None = None
    """Specific parameter names to include in the schema (additive to params_mode)."""
    exclude_params: list[str] | None = None
    """Specific parameter names to exclude from the schema."""
    fixed_params: dict[str, Any] = {}
    """Parameters that are fixed at initialization and merged with runtime kwargs."""
    pre_hook: Callable[..., Any] | None = None
    """Pre-execution hook for input processing. See 'Hook Patterns' in class docstring.
    Signature: async def pre_hook(input: dict[str, Any]) -> None
    """
    post_hook: Callable[..., Any] | None = None
    """Post-execution hook for output processing. See 'Hook Patterns' in class docstring.
    Signature: async def post_hook(output: Any) -> None
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


    @field_validator("pre_hook", "post_hook")
    @classmethod
    def _validate_hooks(cls, v: Callable[..., Any] | None) -> Callable[..., Any] | None:
        """Validate that hooks are async callables."""
        if v is None:
            return v
        if not callable(v):
            raise ValueError(f"Hook must be callable, got {type(v)}")
        # Check for generators first since they affect the iscoroutinefunction check
        if inspect.isgeneratorfunction(v) or inspect.isasyncgenfunction(v):
            raise ValueError("Hook must not be a generator or async generator")
        if not inspect.iscoroutinefunction(v):
            raise ValueError(f"Hook must be an async function, got {type(v)}")
        return v


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
        - Parameter filtering based on params_mode, include_params, exclude_params
        
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
        
        Applies the params_mode, include_params, and exclude_params settings
        to filter which parameters are included in the final schema.
        
        Returns:
            A filtered JSON schema containing only the selected parameters
        """
        selected_params = set()
        # Start with either all params or just required ones
        if self.params_mode == "required":
            selected_params = set(self.params_model_schema.get("required", []))
        else:
            selected_params = set(self.params_model_schema["properties"].keys())
        
        # Add any explicitly included params
        if self.include_params is not None:
            selected_params.update(self.include_params)

        # Remove any explicitly excluded params
        if self.exclude_params is not None:
            selected_params.difference_update(self.exclude_params)

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

    
    @collectable
    async def __call__(self, **kwargs: Any) -> AsyncGenerator[Event, None]:
        """Execute the runnable with the given parameters.
        
        This is the main entry point for executing a runnable. It handles:
        - Parameter validation and merging with fixed_params
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

        _parent_call_id = get_call_id()
        _call_id = uuid7(as_type="str").replace("-", "")
        if self._is_orchestrator:
            set_call_id(_call_id)

        # Generate new context or reset it if appropriate
        run_context = get_run_context()
        if not run_context:
            run_context = RunContext()
            set_run_context(run_context)
        if not _parent_call_id and run_context.tracing:
            run_context = RunContext(parent_id=run_context.id)
            set_run_context(run_context)

        assert _call_id not in run_context.tracing, f"Call ID {_call_id} already exists in tracing."
        trace = {
            "path": self._path,
            "parent_call_id": _parent_call_id,
            "usage": {},
            "t0": t0,
        }
        run_context.tracing[_call_id] = trace

        start_event = await StartEvent.build(
            run_id=run_context.id,
            path=self._path,
        )
        logger.info("start_event", **start_event.dump)
        yield start_event

        # We store the unvalidated input, as sent by the user. 
        # This will ensure full replayability of the run.
        # TODO Evaluate runtime mappings
        input = {**self.fixed_params, **kwargs}
        trace["input"] = input

        output, error = None, None
        try:
            # TODO We should not mutate the input dict. If we want to modify or add new variables we should store them someplace else
            if self.pre_hook is not None:
                await self.pre_hook(input)
            
            # Pydantic model_validate() does not mutate the input dict
            validated_input = dict(self.params_model.model_validate(input))

            async_gen = None
            if not self._is_async_gen and not self._is_coroutine:
                loop = asyncio.get_running_loop()
                ctx = contextvars.copy_context()
                if self._is_gen:
                    gen = self.handler(**validated_input)
                    async_gen = sync_to_async_gen(gen, loop, ctx, self._path, _call_id)
                else:
                    def handler_func(_call_id=_call_id):
                        return ctx.run(self.handler, **validated_input)
                    output = await loop.run_in_executor(None, handler_func)
            
            elif self._is_coroutine:
                output = await self.handler(**validated_input)
            
            else:
                if self._is_orchestrator:
                    async_gen = self.handler(**validated_input, _parent_call_id=_call_id)
                else:
                    async_gen = self.handler(**validated_input)
            
            if async_gen:
                collector = None
                async for chunk in async_gen:
                    # Get or create collector for this event type
                    if collector is None:
                        collector_type = get_collector_registry().get_collector_type(chunk)
                        if collector_type:
                            collector = collector_type(run_context)
                    
                    if collector:
                        processed_chunk = collector.process(chunk)
                        # If the processed chunk is not None, it means we have streaming content
                        if processed_chunk is not None:
                            # If it's already a BaseEvent, it means we have already emitted it.
                            if not isinstance(chunk, BaseEvent):
                                chunk_event = await ChunkEvent.build(
                                    run_id=run_context.id,
                                    path=self._path,
                                    chunk=processed_chunk,
                                )
                                logger.info("chunk_event", **chunk_event.dump)
                                yield chunk_event
                            else:
                                yield chunk

                output = collector.collect() if collector else None
            
            if self.post_hook is not None:
                await self.post_hook(output)
            
        except Exception as err:
            error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }
            trace["error"] = error
        
        finally:
            t1 = int(time.time() * 1000)
            trace["t1"] = t1
            output_event = await OutputEvent.build(
                run_id=run_context.id,
                path=self._path,
                t0=t0,
                t1=t1,
                input=input,
                output=output,
                error=error,
                usage=trace["usage"],
            )
            # TODO Not a fan of storing the dumps...
            trace["output"] = output_event.dump["output"]
            if _parent_call_id is None:
                await run_context.save_tracing()
                # We don't want to propagate this between runs. We use this variable to check if we're at an entry point
                set_call_id(None)
            logger.info("output_event", **output_event.dump)
            yield output_event


RunnableLike = Runnable | dict[str, Any] | Callable[..., Any]
"""Type alias for objects that can be used as tools for an agent or steps in a workflow."""
