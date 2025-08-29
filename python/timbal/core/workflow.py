import asyncio
from collections.abc import AsyncGenerator
from functools import cached_property
from typing import Any, override

import structlog
from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field, create_model

from ..types.events.output import OutputEvent
from .runnable import Runnable, RunnableLike
from .tool import Tool

logger = structlog.get_logger("timbal.core.workflow")


class Workflow(Runnable):
    """"""

    _steps: dict[str, Runnable] = PrivateAttr(default_factory=dict)
    """List of steps to execute in the workflow."""


    def model_post_init(self, __context: Any) -> None:
        """"""
        super().model_post_init(__context)
        self._path = self.name

        # Workflows are always orchestrators with async generator handlers
        self._is_orchestrator = True
        self._is_coroutine = False
        self._is_gen = False
        self._is_async_gen = True


    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"
        # Update paths for internal LLM and all tools
        for step in self._steps:
            step.nest(self._path)
    

    @override 
    @computed_field
    @cached_property
    def params_model(self) -> BaseModel:
        """See base class."""
        fields = {}
        for step in self._steps.values():
            for param, field_info in step.params_model.__pydantic_fields__.items():
                # If a default is set for the param, we remove this from the model, but allow
                # extra properties to enable overriding these values from kwargs
                if param not in step.default_params:
                    fields[param] = (field_info.annotation, field_info)
        params_model_name = self.name.title().replace("_", "") + "Params"
        return create_model(params_model_name, __config__=ConfigDict(extra="allow"), **fields)


    @override 
    @computed_field
    @cached_property
    def return_model(self) -> Any:
        """See base class."""
        # TODO Implement
        return Any


    def _is_dag(self) -> bool:
        """Checks if the workflow is a directed acyclic graph (DAG)."""
        # States: 0 = unvisited, 1 = visiting, 2 = visited
        state = {step_name: 0 for step_name in self._steps.keys()}
        def dfs(step_name):
            if state[step_name] == 1:
                return False
            if state[step_name] == 2:
                return True
            state[step_name] = 1
            for next_step_name in self._steps[step_name].next_steps:
                if not dfs(next_step_name):
                    return False
            state[step_name] = 2
            return True
        for step_name in self._steps.keys():
            if state[step_name] == 0:
                if not dfs(step_name):
                    return False
        return True
    

    def link(self, source: str, target: str) -> "Workflow":
        """"""
        self._steps[source].next_steps.add(target)
        self._steps[target].previous_steps.add(source)
        if not self._is_dag():
            raise ValueError(f"Linking {source} -> {target} would create a cycle in the workflow.")
        return self


    # TODO Think how we handle agent model_params vs default_params
    def step(self, runnable: RunnableLike, **kwargs: Any) -> "Workflow":
        """"""
        if not isinstance(runnable, Runnable):
            if isinstance(runnable, dict):
                runnable = Tool(**runnable)
            else:
                runnable = Tool(handler=runnable)
            
        if runnable.name in self._steps:
            raise ValueError(f"Step {runnable.name} already exists in the workflow.")
        
        runnable.previous_steps = set()
        runnable.next_steps = set()

        # TODO Set execution order if we find refs
        for k, v in kwargs.items():
            runnable.default_params[k] = v
            runnable._validate_default_param(k, v)

        runnable.nest(self._path)
        self._steps[runnable.name] = runnable
        return self

    
    def _skip_next_steps(self, step_name: str, completions: dict[str, asyncio.Event]) -> None:
        """"""
        completions[step_name].set()
        for next_name in self._steps[step_name].next_steps:
            self._skip_next_steps(next_name, completions)


    async def _enqueue_step_events(self, step: Runnable, queue: asyncio.Queue, completions: dict[str, asyncio.Event], **kwargs: Any) -> None:
        """"""
        # Await for the completion of all ancestors
        await asyncio.gather(*[completions[step_name].wait() for step_name in step.previous_steps])
        # This serves multiple purposes. 
        # - It ensures that the step is not executed multiple times.
        # - It allows the step to be skipped from other steps, e.g. if a previous step failed.
        if completions[step.name].is_set():
            logger.info(f"Skipping {step.name} as it's already marked as completed.")
            await queue.put(None)
            return

        # TODO Evaluate conditions

        async for event in step(**kwargs):
            await queue.put(event)
            if isinstance(event, OutputEvent) and event.error is not None:
                self._skip_next_steps(step.name, completions)

        completions[step.name].set()
        await queue.put(None)


    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """"""
        queue = asyncio.Queue()
        completions = {step_name: asyncio.Event() for step_name in self._steps.keys()}
        tasks = [
            asyncio.create_task(self._enqueue_step_events(step, queue, completions, **kwargs)) 
            for step in self._steps.values()
        ]
        remaining = len(tasks)
        while remaining > 0:
            event = await queue.get()
            if event is None:
                remaining -= 1
            yield event
        # TODO How to collect the final output?
