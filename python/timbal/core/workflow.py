import asyncio
from collections.abc import AsyncGenerator, Callable
from enum import Enum
from functools import cached_property
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field, create_model

from ..errors import InterruptError, SpanNotFound
from ..state import get_call_id, get_parent_call_id, set_parent_call_id
from ..types.events.output import OutputEvent
from .runnable import Runnable, RunnableLike
from .tool import Tool


class StepState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class StepStatus:
    __slots__ = ("state", "done")

    def __init__(self) -> None:
        self.state: StepState = StepState.PENDING
        self.done: asyncio.Event = asyncio.Event()


logger = structlog.get_logger("timbal.core.workflow")


class Workflow(Runnable):
    """Orchestrates execution of multiple steps in a DAG with automatic dependency linking."""

    _steps: dict[str, Runnable] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._path = self.name
        self._is_orchestrator = True
        self._is_coroutine = False
        self._is_gen = False
        self._is_async_gen = True

    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"
        # Update paths for internal LLM and all tools
        for step in self._steps.values():
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
        """Add a step to the workflow with automatic dependency linking."""
        if not isinstance(runnable, Runnable):
            if isinstance(runnable, dict):
                runnable = Tool(**runnable)
            else:
                runnable = Tool(handler=runnable)  # type: ignore[call-arg]

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
        depends_on = set(depends_on or [])  # Deduplicate here to avoid duplicate _is_dag calls

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

    async def _enqueue_step_events(
        self,
        step: Runnable,
        queue: asyncio.Queue,
        statuses: dict[str, StepStatus],
        **kwargs: Any,
    ) -> None:
        """Execute a single workflow step and enqueue its events to the shared queue."""
        status = statuses[step.name]

        # Await for the completion of all ancestors
        await asyncio.gather(*[statuses[step_name].done.wait() for step_name in step.previous_steps])
        # This serves multiple purposes.
        # - It ensures that the step is not executed multiple times.
        # - It allows the step to be skipped from other steps, e.g. if a previous step failed.
        if status.done.is_set():
            logger.info(f"Skipping {step.name} as it's already marked as done.")
            await queue.put(None)
            return

        # To evaluate `when` conditions and resolve parameters, lambdas call step_span()
        # which looks for sibling spans by parent_call_id. We temporarily set parent_call_id
        # to the workflow's call_id so step_span() finds the correct sibling steps.
        workflow_call_id = get_call_id()
        original_parent_call_id = get_parent_call_id()
        set_parent_call_id(workflow_call_id)

        try:
            if step.when:
                should_run = await step._execute_runtime_callable(step.when["callable"], step.when["is_coroutine"])
                if not should_run:
                    logger.info(f"Skipping {step.name} because `when` condition returned False.")
                    status.state = StepState.SKIPPED
                    status.done.set()
                    await queue.put(None)
                    return

            resolved_input = await step._resolve_input_params(kwargs)

        except SpanNotFound as e:
            logger.info(f"Skipping {step.name} because it needs span from skipped step {e.step_name}.")
            status.state = StepState.SKIPPED
            status.done.set()
            await queue.put(None)
            return

        except Exception as e:
            logger.info(f"Failing {step.name} due to error during evaluation: {e}")
            status.state = StepState.FAILED
            status.done.set()
            await queue.put(None)
            return

        finally:
            set_parent_call_id(original_parent_call_id)

        status.state = StepState.RUNNING
        try:
            async for event in step(**resolved_input):
                await queue.put(event)
                if isinstance(event, OutputEvent) and event.error is not None:
                    logger.info(f"Step {step.name} completed with error.")
                    status.state = StepState.FAILED
                    status.done.set()
                    await queue.put(None)
                    return
            status.state = StepState.COMPLETED
        except Exception as e:
            status.state = StepState.FAILED
            await queue.put(e)
            return
        finally:
            status.done.set()

        await queue.put(None)

    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Execute all steps concurrently, respecting dependencies."""
        queue = asyncio.Queue()
        statuses = {step_name: StepStatus() for step_name in self._steps.keys()}
        tasks = [
            asyncio.create_task(self._enqueue_step_events(step, queue, statuses, **kwargs))
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
