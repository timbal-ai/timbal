import asyncio
import contextvars
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from functools import cached_property
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

    _is_coroutine: bool = PrivateAttr()
    """"""
    _is_gen: bool = PrivateAttr()
    """"""
    _is_async_gen: bool = PrivateAttr()
    """"""
    _path: str = PrivateAttr()
    """"""


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

    
    async def __call__(self, **kwargs: Any) -> AsyncGenerator[Event, None]:
        """"""
        t0 = int(time.time() * 1000)

        run_context = get_run_context()
        if not run_context:
            run_context = RunContext()
            set_run_context(run_context)

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

        # Extract and store internal call ID for usage tracking
        _call_id = input.pop("_call_id", None)
        # Make sure the path and call_id exist in the tracing dictionary
        if self._path not in run_context.tracing:
            run_context.tracing[self._path] = {}
        if _call_id not in run_context.tracing[self._path]:
            run_context.tracing[self._path][_call_id] = {"usage": {}}

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
            trace = run_context.tracing.get(self._path, {}).get(_call_id, {})
            output_event = await OutputEvent.build(
                run_id=run_context.id,
                path=self._path,
                t0=t0,
                t1=t1,
                input=input,
                output=output,
                error=error,
                usage=trace.get("usage", {}),
            )
            # TODO Think where to put this
            trace.update({
                "t0": t0,
                "t1": t1,
                "input": output_event.dump["input"],
                "output": output_event.dump["output"],
                "error": output_event.dump["error"],
            })
            logger.info("output_event", **output_event.dump)
            yield output_event
