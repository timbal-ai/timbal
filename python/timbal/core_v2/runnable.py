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
from uuid_extensions import uuid7

from .collectors.base import BaseCollector
from .collectors.default import DefaultCollector
from .context import RunContext, get_run_context, set_run_context
from .events.base import Event
from .events.chunk import ChunkEvent, ChunkEventData
from .events.output import OutputEvent, OutputEventData
from .events.start import StartEvent, StartEventData
from .utils import dump, sync_to_async_gen

logger = structlog.get_logger("timbal.core_v2.runnable")


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
    force_exit: bool = False
    """"""
    params_mode: Literal["all", "required"] = "all"
    """"""
    include_params: list[str] | None = None
    """"""
    exclude_params: list[str] | None = None
    """"""
    fixed_params: dict[str, Any] = {}
    """"""
    collector_cls: type[BaseCollector] = DefaultCollector
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

        formatted_params_model_schema = {
            "type": "object",
            "properties": {
                k: v 
                for k, v in self.params_model_schema["properties"].items()
                if k in selected_params
            }
        }
        return formatted_params_model_schema

    
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
        """"""
        return self.openai_schema

    
    async def __call__(self, **kwargs: Any) -> AsyncGenerator[Event, None]:
        """"""
        t0 = int(time.time() * 1000)

        run_context = get_run_context()
        if not run_context:
            run_context = RunContext(id=uuid7(as_type="str"))
            set_run_context(run_context)

        start_event = StartEvent(
            run_id=run_context.id,
            path=self._path,
            data=StartEventData(),
        )
        start_event_dump = dump(start_event)

        logger.info("start_event", **start_event_dump)
        yield start_event

        # At initialization, we might want to fix some parameters for the handler.
        # We'll use these fixed parameters as default values.
        input = {**self.fixed_params, **kwargs}
        output, error = None, None

        try:
            # ? Should we capture and raise a special timbal error?
            input = dict(self.params_model.model_validate(input))

            async_gen = None
            if not self._is_async_gen and not self._is_coroutine:
                loop = asyncio.get_running_loop()
                ctx = contextvars.copy_context()

                if self._is_gen:
                    gen = self.handler(**input)
                    async_gen = sync_to_async_gen(gen, loop, ctx)
                else:
                    output = await loop.run_in_executor(None, lambda: ctx.run(self.handler, **input))
            
            elif self._is_coroutine:
                output = await self.handler(**input)
            
            else:
                async_gen = self.handler(**input)
            
            if async_gen:
                collector = self.collector_cls()
                async for chunk in async_gen:
                    chunk = collector.handle_chunk(chunk)
                    # If the handled chunk is None, it means we don't want to yield anything.
                    if chunk is not None:
                        # If it's already a base event, it means we have already emitted it.
                        if not isinstance(chunk, Event):
                            chunk_event = ChunkEvent(
                                run_id=run_context.id,
                                path=self._path,
                                data=ChunkEventData(chunk=chunk),
                            )
                            chunk_event_dump = dump(chunk_event)

                            logger.info("chunk_event", **chunk_event_dump)
                            yield chunk_event

                        else:
                            yield chunk

                output = collector.collect()
            
        except Exception as err:
            error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }
        
        finally:
            t1 = int(time.time() * 1000)

            output_event = OutputEvent(
                run_id=run_context.id,
                path=self._path,
                data=OutputEventData(
                    t0=t0,
                    t1=t1,
                    input=input,
                    output=output,
                    error=error,
                    # TODO This grabs all the usage, not just the one by this runnable component
                    usage=run_context.usage,
                ),
            )
            output_event_dump = dump(output_event)

            logger.info("output_event", **output_event_dump)
            yield output_event
