import asyncio
from collections.abc import AsyncGenerator, Callable
from functools import cached_property
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field, create_model

from timbal.types.events.start import StartEvent

from ..errors import InterruptError
from ..types.events.output import OutputEvent
from .runnable import Runnable, RunnableLike
from .tool import Tool

logger = structlog.get_logger("timbal.core.workflow")


class Workflow(Runnable):
    """A Workflow is a Runnable that orchestrates execution of multiple steps in a directed acyclic graph (DAG).
    
    Workflows implement a step-based execution pattern where:
    1. Steps are added as Runnable components with explicit dependencies
    2. Steps can be linked to form execution chains based on data dependencies
    3. All steps execute concurrently while respecting dependency constraints
    4. Failed steps automatically skip their dependent steps to prevent cascading failures
    5. The workflow completes when all executable steps finish
    
    Workflows support:
    - Automatic step linking based on data key dependencies (e.g., step1.output -> step2.input)
    - Concurrent execution of independent steps for optimal performance
    - DAG validation to prevent circular dependencies
    - Graceful error handling with dependent step skipping
    - Dynamic parameter collection from all constituent steps
    """

    _steps: dict[str, Runnable] = PrivateAttr(default_factory=dict)
    """List of steps to execute in the workflow."""


    def model_post_init(self, __context: Any) -> None:
        """Initialize workflow as an orchestrator with async generator handler."""
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
        """Check if the workflow forms a valid DAG using depth-first search cycle detection."""
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
    

    def _link(self, source: str, target: str) -> "Workflow":
        """Internal method to link workflow steps."""
        if source not in self._steps:
            raise ValueError(f"Source step {source} not found in workflow.")
        if target not in self._steps:
            raise ValueError(f"Target step {target} not found in workflow.")
        self._steps[source].next_steps.add(target)
        self._steps[target].previous_steps.add(source)
        if not self._is_dag():
            raise ValueError(f"Linking {source} -> {target} would create a cycle in the workflow.")
        return self


    # TODO Think how we handle agent model_params vs default_params
    def step(
        self, 
        runnable: RunnableLike, 
        depends_on: list[str] | None = None, 
        when: Callable[[], bool] | None = None, 
        **kwargs: Any,
    ) -> "Workflow":
        """Add a step to the workflow with automatic dependency linking.
        
        Adds a runnable component as a workflow step and automatically creates
        dependency links based on data key analysis. If step parameters reference
        other steps' outputs (e.g., step1.result), those dependencies are
        automatically linked.
        
        The runnable can be:
        - A Runnable instance
        - A dictionary that will be converted to a Tool
        - A callable that will be wrapped in a Tool
        
        Args:
            runnable: The runnable component to add as a step
            depends_on: Optional list of steps that must complete before this step
            when: Optional callable that returns a boolean to conditionally execute the step
            **kwargs: Default parameters for the step, also used for dependency analysis
            
        Returns:
            Self for method chaining
        """
        if not isinstance(runnable, Runnable):
            if isinstance(runnable, dict):
                runnable = Tool(**runnable)
            else:
                runnable = Tool(handler=runnable)

        if runnable.name in self._steps:
            raise ValueError(f"Step {runnable.name} already exists in the workflow.")

        runnable.nest(self._path)
        self._steps[runnable.name] = runnable
        runnable.previous_steps = set()
        runnable.next_steps = set()
        runnable.when = None

        # Explicit dependencies
        if depends_on and not isinstance(depends_on, list):
            raise ValueError("depends_on must be a list of step names")
        depends_on = set(depends_on or []) # Deduplicate here to avoid duplicate _is_dag calls

        depends_on.update(runnable._dependencies)
        depends_on.update(runnable._pre_hook_dependencies)
        depends_on.update(runnable._post_hook_dependencies)

        # Optional handler to determine whether to execute the step, and inspect it to automatically link steps
        if when:
            inspect_result = runnable._inspect_callable(when)
            runnable.when = {"callable": when, **inspect_result}
            depends_on.update(inspect_result["dependencies"])

        # Use kwargs as default params for the runnable, and inspect callables to automatically link steps
        runnable._prepare_default_params(kwargs)
        for v in runnable._default_runtime_params.values():
            depends_on.update(v["dependencies"])

        for dep in depends_on:
            logger.info("Linking steps", previous_step=dep, next_step=runnable.name)
            self._link(dep, runnable.name)

        return self

    
    def _skip_next_steps(self, step_name: str, completions: dict[str, asyncio.Event]) -> None:
        """Recursively mark a step and all its dependents as completed (skipped)."""
        completions[step_name].set()
        for next_name in self._steps[step_name].next_steps:
            self._skip_next_steps(next_name, completions)


    async def _enqueue_step_events(self, step: Runnable, queue: asyncio.Queue, completions: dict[str, asyncio.Event], **kwargs: Any) -> None:
        """Execute a single workflow step and enqueue its events to the shared queue."""
        # Await for the completion of all ancestors
        await asyncio.gather(*[completions[step_name].wait() for step_name in step.previous_steps])
        # This serves multiple purposes. 
        # - It ensures that the step is not executed multiple times.
        # - It allows the step to be skipped from other steps, e.g. if a previous step failed.
        if completions[step.name].is_set():
            logger.info(f"Skipping {step.name} as it's already marked as completed.")
            await queue.put(None)
            return

        step_started = False
        try:
            async for event in step(**kwargs):
                await queue.put(event)
                if isinstance(event, StartEvent):
                    step_started = True
                if isinstance(event, OutputEvent) and event.error is not None:
                    logger.info(f"Skipping step {step.name} successors...")
                    self._skip_next_steps(step.name, completions)
        except Exception as e:
            await queue.put(e)
            return
        
        if not step_started:
            logger.info(f"Skipping step {step.name} and all successors...")
            self._skip_next_steps(step.name, completions)

        completions[step.name].set()
        await queue.put(None)


    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Main workflow execution handler implementing concurrent step orchestration.
        
        This is the core workflow logic that implements concurrent step execution:
        1. Creates completion events for all steps to coordinate dependencies
        2. Launches all steps concurrently as async tasks
        3. Each step waits for its prerequisites before executing
        4. Multiplexes events from all steps as they become available
        5. Continues until all steps complete or are skipped
        
        The workflow provides optimal performance by executing independent steps
        in parallel while maintaining dependency order through completion events.
        
        Args:
            **kwargs: Execution parameters distributed to appropriate steps
                
        Yields:
            Events from step executions as they become available
        """
        queue = asyncio.Queue()
        completions = {step_name: asyncio.Event() for step_name in self._steps.keys()}
        tasks = [
            asyncio.create_task(self._enqueue_step_events(step, queue, completions, **kwargs)) 
            for step in self._steps.values()
        ]
        
        try:
            remaining = len(tasks)
            while remaining > 0:
                event = await queue.get()
                if isinstance(event, InterruptError):
                    # Propagate interrupt error - will be handled by finally block
                    raise event
                if isinstance(event, Exception):
                    raise event
                elif event is None:
                    remaining -= 1
                else:
                    yield event
        except (asyncio.CancelledError, InterruptError):
            # Cancellation or interrupt - clean up gracefully
            raise
        finally:
            # Cancel all pending step tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all cancellations to complete, suppressing errors
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
