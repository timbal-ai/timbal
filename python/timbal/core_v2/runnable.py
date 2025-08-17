import asyncio
import contextvars
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from functools import cached_property, wraps
from typing import Any, Literal

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    TypeAdapter,
    computed_field,
    model_serializer,
)

from ..collectors import get_collector_registry
from ..collectors.utils import sync_to_async_gen
from ..state import get_run_context, set_run_context
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
        self._async_gen = async_gen  # The original async generator from __call__
        self._runnable = runnable    # Reference to the Runnable instance (not used currently)
        self._kwargs = kwargs        # Original kwargs passed to __call__ (not used currently)
        self._collected = False      # Track if the generator has been fully consumed
        self._output_event = None    # Cache the OutputEvent when we encounter it
        self._events = []            # Cache all yielded events for later access
    
    def __aiter__(self):
        """
        Return the async iterator object (self).
        
        In Python's async iterator protocol:
        - __aiter__() is called when you do `async for item in obj:`
        - It should return an object that implements __anext__()
        - We return `self` because this class implements __anext__()
        
        This is equivalent to how regular iterators work:
        - __iter__() returns an iterator object with __next__()
        """
        return self
    
    async def __anext__(self):
        """
        Get the next item from the async iterator.
        
        This is called by `async for` loops and is the core of the async iterator protocol.
        When the generator is exhausted, it raises StopAsyncIteration to signal completion.
        
        We intercept each event and cache it in self._events for later use by collect().
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
        """
        Close the generator gracefully.
        
        This is called when the generator needs to be cleaned up.
        """
        await self._async_gen.aclose()
        self._collected = True
    
    async def collect(self) -> Any:
        """
        Collect the final output by consuming the entire stream.
        
        How this works:
        1. If we already have the OutputEvent cached, return it immediately
        2. Otherwise, consume remaining events using `async for event in self:`
           - This calls our __anext__() method which caches the OutputEvent when found
        3. Return the cached OutputEvent
        
        This method can be called multiple times safely.
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
    """"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str
    """"""
    description: str | None = None
    """"""
    params_mode: Literal["all", "required"] = "all"
    """"""
    include_params: list[str] | None = None
    """"""
    exclude_params: list[str] | None = None
    """"""
    fixed_params: dict[str, Any] = {}
    """"""

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


    @abstractmethod
    def nest(self, parent_path: str) -> None:
        """"""
        pass


    # NOTE: Pydantic's `@computed_field` and functool's `@cached_property` interfere
    # with the abstract method's ability to force an implementation at instantiation time
    # @computed_field 
    # @cached_property 
    @abstractmethod
    def params_model(self) -> BaseModel:
        """"""
        pass

    
    @computed_field 
    @cached_property 
    def params_model_schema(self) -> dict[str, Any]:
        """"""
        params_model_schema = self.params_model.model_json_schema()
        return params_model_schema


    # NOTE: Pydantic's `@computed_field` and functool's `@cached_property` interfere
    # with the abstract method's ability to force an implementation at instantiation time
    # @computed_field 
    # @cached_property 
    @abstractmethod
    def return_model(self) -> Any:
        """"""
        pass

    
    @computed_field 
    @cached_property 
    def return_model_schema(self) -> dict[str, Any]:
        """"""
        return_model_schema = TypeAdapter(self.return_model).json_schema()
        return return_model_schema

    
    def format_params_model_schema(self) -> dict[str, Any]:
        """"""
        selected_params = set()
        if self.params_mode == "required":
            selected_params = set(self.params_model_schema.get("required", []))
        else:
            selected_params = set(self.params_model_schema["properties"].keys())
        
        if self.include_params is not None:
            selected_params.update(self.include_params)

        if self.exclude_params is not None:
            selected_params.difference_update(self.exclude_params)

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
        """"""
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
        """"""
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
        """"""
        t0 = int(time.time() * 1000)

        _call_id = kwargs.pop("_call_id", None)
        _parent_call_id = kwargs.pop("_parent_call_id", None)

        # Generate new context or reset it if appropriate
        run_context = get_run_context()
        if not run_context:
            run_context = RunContext()
            set_run_context(run_context)
        if not _call_id and run_context.tracing:
            run_context = RunContext(parent_id=run_context.id)
            set_run_context(run_context)

        assert _call_id not in run_context.tracing, f"Call ID {_call_id} already exists in tracing."
        run_context.tracing[_call_id] = {
            "path": self._path,
            "parent_call_id": _parent_call_id,
            "usage": {},
        }

        start_event = await StartEvent.build(
            run_id=run_context.id,
            path=self._path,
        )
        logger.info("start_event", **start_event.dump)
        yield start_event

        # At initialization, we might want to fix some parameters for the handler.
        # We'll use these fixed parameters as default values.
        input = {**self.fixed_params, **kwargs}
        output, error = None, None

        try:
            input = dict(self.params_model.model_validate(input))

            async_gen = None
            if not self._is_async_gen and not self._is_coroutine:
                loop = asyncio.get_running_loop()
                ctx = contextvars.copy_context()
                if self._is_gen:
                    gen = self.handler(**input)
                    async_gen = sync_to_async_gen(gen, loop, ctx, self._path, _call_id)
                else:
                    def handler_func(_call_id=_call_id):
                        return ctx.run(self.handler, **input)
                    output = await loop.run_in_executor(None, handler_func)
            
            elif self._is_coroutine:
                output = await self.handler(**input)
            
            else:
                if self._is_orchestrator:
                    async_gen = self.handler(**input, _parent_call_id=_call_id)
                else:
                    async_gen = self.handler(**input)
            
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
            
        except Exception as err:
            error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }
        
        finally:
            t1 = int(time.time() * 1000)
            trace = run_context.tracing[_call_id]
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
            # TODO Think where to put this
            trace.update({
                "t0": t0,
                "t1": t1,
                "input": output_event.dump["input"],
                "output": output_event.dump["output"],
                "error": output_event.dump["error"],
            })
            if _call_id is None: # root
                await run_context.save_tracing()
            logger.info("output_event", **output_event.dump)
            yield output_event
