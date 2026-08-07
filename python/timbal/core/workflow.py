import asyncio
import traceback
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
from pydantic import BaseModel, ConfigDict, computed_field, create_model

from ..errors import InterruptError, PauseRequired, RunCancelled, SpanNotFound, WorkflowStepError
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
    __slots__ = ("state", "done", "error", "signal")

    def __init__(self) -> None:
        self.state: StepState = StepState.PENDING
        self.done: asyncio.Event = asyncio.Event()
        self.error: dict[str, Any] | None = None
        self.signal: BaseException | None = None
        """Deferred outcome signal recorded by _run_step: PauseRequired,
        RunCancelled, or the exception raised while streaming the step.
        Aggregated by handler() after the step finishes."""


logger = structlog.get_logger("timbal.core.workflow")


class Workflow(Runnable):
    """Orchestrates execution of multiple steps in a DAG with automatic dependency linking."""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        # Plain instance attributes (not PrivateAttr) — hot reads on every run.
        self._steps: dict[str, Runnable] = {}
        self._is_linear: bool = True
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
        while_: Callable[[], bool] | int | None = None,
        **kwargs: Any,
    ) -> "Workflow":
        """Add a step to the workflow with automatic dependency linking.

        ``while_`` (optional) repeats the step after each successful iteration
        until the condition is false or the count is reached. Semantics are
        do-while: the step always runs at least once when ``when`` allows it.
        Each iteration produces its own span; ``step_span(name)`` returns the
        latest. Self-references in a ``while_`` callable are not treated as
        graph edges (the step's own span does not exist before iteration 1).

        Loop semantics to be aware of:

        - An int count must be >= 1 (bools are rejected). ``while_=1`` is
          equivalent to no loop.
        - Step params are resolved ONCE, before the first iteration — lambdas
          are not re-evaluated per iteration. A step that needs a cursor or
          accumulator owns that state itself (e.g. reads its own previous
          output via ``step_span``).
        - A callable condition has no built-in iteration cap; a condition that
          never returns falsy loops forever. Prefer an int count or make the
          condition provably terminating.
        - Pausing mid-loop (an approval gate or ``suspend()`` inside the step)
          restarts the loop from iteration 1 on resume: loop progress is not
          persisted, so side effects of already-completed iterations run
          again. Additionally, approval ids are derived from ``(path, input)``
          and the input is fixed across iterations, so a single approval
          resolution covers EVERY iteration of the loop in the same resume
          run. Avoid combining ``while_`` with approval-gated or suspending
          steps unless that is acceptable.
        """
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
        # Per-source edge kinds: maps a previous step name -> set of kinds
        # ("ordering" | "when" | "while" | "param"). Used by introspection
        # (get_flow) to distinguish explicit sequencing from param/when/while-
        # induced dependencies. Runtime sequencing only ever looks at
        # ``previous_steps``.
        runnable.previous_steps_kinds = {}
        runnable.when = None
        runnable.while_ = None

        # Explicit dependencies
        if depends_on and not isinstance(depends_on, list):
            raise ValueError("depends_on must be a list of step names")

        # Track dependency origins so introspection can tell explicit ordering
        # (depends_on / hooks) apart from param- and when-induced wiring. These
        # all merge into ``previous_steps`` for execution, but the distinction is
        # lossy once merged (a step can be both an explicit dep and a param dep),
        # so we record it here while the sources are still separate.
        ordering_deps = set(depends_on or [])
        # Hooks read sibling outputs via step_span(); treat as ordering so they
        # are never silently collapsed in compact views.
        ordering_deps.update(runnable._pre_hook_dependencies)
        ordering_deps.update(runnable._post_hook_dependencies)
        # Handler-body step_span() references are data wiring, not explicit ordering.
        param_deps = set(runnable._dependencies)
        when_deps: set[str] = set()
        while_deps: set[str] = set()

        # Optional handler to determine whether to execute the step, and inspect it to automatically link steps
        if when:
            inspect_result = runnable._inspect_callable(when)
            runnable.when = {"callable": when, **inspect_result}
            when_deps.update(inspect_result["dependencies"])

        if isinstance(while_, bool):
            # bool is an int subclass — while_=True would silently mean "run
            # once" rather than "loop forever". Reject it outright.
            raise ValueError("while_ must be a callable, an int >= 1, or None (got a bool)")
        if isinstance(while_, int):
            if while_ < 1:
                raise ValueError(f"while_ count must be >= 1, got {while_}")
            runnable.while_ = {"count": while_}
        elif callable(while_):
            inspect_result = runnable._inspect_callable(while_)
            # Self-references are expected (condition reads this step's latest
            # output) and must not become a DAG self-edge.
            deps = set(inspect_result["dependencies"])
            deps.discard(runnable.name)
            inspect_result["dependencies"] = list(deps)
            runnable.while_ = {"callable": while_, **inspect_result}
            while_deps.update(deps)
        elif while_ is not None:
            raise ValueError("while_ must be a callable, an int, or None")

        # Use kwargs as default params for the runnable, and inspect callables to automatically link steps
        runnable._prepare_default_params(kwargs)
        for v in runnable._default_runtime_params.values():
            param_deps.update(v["dependencies"])

        edge_kinds: dict[str, set[str]] = {}
        for dep in ordering_deps:
            edge_kinds.setdefault(dep, set()).add("ordering")
        for dep in when_deps:
            edge_kinds.setdefault(dep, set()).add("when")
        for dep in while_deps:
            edge_kinds.setdefault(dep, set()).add("while")
        for dep in param_deps:
            edge_kinds.setdefault(dep, set()).add("param")
        runnable.previous_steps_kinds = edge_kinds

        # Deduplicate (set union) to avoid duplicate _is_dag calls per shared dep.
        for dep in edge_kinds:
            logger.info("Linking steps", previous_step=dep, next_step=runnable.name)
            self._link(dep, runnable.name)

        # A workflow is a linear chain when each step depends exactly on its
        # predecessor (and the first on nothing). handler() then skips the
        # task/queue fan-in machinery entirely.
        steps = list(self._steps.values())
        self._is_linear = all(
            (not s.previous_steps) if i == 0 else s.previous_steps == {steps[i - 1].name}
            for i, s in enumerate(steps)
        )

        return self

    async def _run_step(
        self,
        step: Runnable,
        statuses: dict[str, StepStatus],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Execute a single workflow step and yield its events.

        The step's outcome is recorded on ``statuses[step.name]``:
        ``state``/``error`` for regular completion/failure/skip, and ``signal``
        for deferred control flow (PauseRequired, RunCancelled, or the raised
        exception). ``done`` is ALWAYS set on exit so dependents never hang.

        Shared by both execution modes in :meth:`handler` — the sequential
        fast path iterates it directly; the concurrent path drives it from a
        task via :meth:`_enqueue_step_events`.
        """
        status = statuses[step.name]

        try:
            # Await for the completion of all ancestors. Sequential awaits are
            # equivalent to gather() for waiting on ALL events, without a task
            # per dependency; the is_set() guard skips already-completed ones.
            for step_name in step.previous_steps:
                dep_done = statuses[step_name].done
                if not dep_done.is_set():
                    await dep_done.wait()
            # This serves multiple purposes.
            # - It ensures that the step is not executed multiple times.
            # - It allows the step to be skipped from other steps, e.g. if a previous step failed.
            if status.done.is_set():
                logger.info(f"Skipping {step.name} as it's already marked as done.")
                return

            # To evaluate `when` conditions and resolve parameters, lambdas call step_span()
            # which looks for sibling spans by parent_call_id. We temporarily set parent_call_id
            # to the workflow's call_id so step_span() finds the correct sibling steps.
            workflow_call_id = get_call_id()
            original_parent_call_id = get_parent_call_id()
            set_parent_call_id(workflow_call_id)

            try:
                if step.when:
                    should_run = await step._execute_runtime_callable(
                        step.when["callable"], step.when["is_coroutine"]
                    )
                    if not should_run:
                        logger.info(f"Skipping {step.name} because `when` condition returned False.")
                        status.state = StepState.SKIPPED
                        status.done.set()
                        return

                step_kwargs = {k: v for k, v in kwargs.items() if k not in step._default_runtime_params}
                resolved_input = await step._resolve_input_params(step_kwargs)

            except SpanNotFound as e:
                logger.info(f"Skipping {step.name} because it needs span from skipped step {e.step_name}.")
                status.state = StepState.SKIPPED
                status.done.set()
                return

            except Exception as e:
                logger.info(f"Failing {step.name} due to error during evaluation: {e}")
                status.state = StepState.FAILED
                status.error = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                status.done.set()
                return

            finally:
                set_parent_call_id(original_parent_call_id)

            status.state = StepState.RUNNING
            iteration = 0
            try:
                # Do-while: always run once, then decide whether to continue.
                # Each iteration is a full _stream → its own span. Downstream
                # waits on status.done, which fires after all iterations.
                # resolved_input is fixed across iterations; the step owns any
                # cursor/accumulator state itself.
                while True:
                    # Iterate the raw stream: the TimbalCollector wrapper is only
                    # needed at the public API boundary (.collect(), pending-gate
                    # enrichment); a per-event collector layer here is pure overhead.
                    async for event in step._stream(**resolved_input):
                        yield event
                        if (
                            isinstance(event, OutputEvent)
                            and event.status.code == "cancelled"
                            and event.status.reason
                            in {"approval_required", "approval_denied", "input_required"}
                        ):
                            logger.info(f"Step {step.name} paused ({event.status.reason}).")
                            status.state = StepState.FAILED
                            status.signal = PauseRequired(event)
                            return
                        if (
                            isinstance(event, OutputEvent)
                            and event.status.code == "cancelled"
                            and event.status.reason == "cancelled"
                        ):
                            # A human cancelled this step via Cancel(); terminate the
                            # whole workflow run rather than continuing other steps.
                            logger.info(f"Step {step.name} cancelled by user.")
                            status.state = StepState.FAILED
                            status.signal = RunCancelled(
                                event.status.message or "Run cancelled by user."
                            )
                            return
                        if isinstance(event, OutputEvent) and event.error is not None:
                            logger.info(f"Step {step.name} completed with error.")
                            status.state = StepState.FAILED
                            status.error = event.error
                            status.done.set()
                            return

                    iteration += 1

                    if not step.while_:
                        break
                    if "count" in step.while_:
                        if iteration >= step.while_["count"]:
                            break
                    else:
                        set_parent_call_id(workflow_call_id)
                        try:
                            should_continue = await step._execute_runtime_callable(
                                step.while_["callable"], step.while_["is_coroutine"]
                            )
                        finally:
                            set_parent_call_id(original_parent_call_id)
                        if not should_continue:
                            break

                status.state = StepState.COMPLETED
            except Exception as e:
                status.state = StepState.FAILED
                status.error = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                status.signal = e
                return
            finally:
                status.done.set()

        except BaseException as e:
            # Catch BaseException subclasses that bypass the inner `except Exception`
            # (custom BaseException from user `when`/resolver callables, consumer
            # GeneratorExit, task-level CancelledError, etc.). Flagging the step
            # as FAILED and setting `done` keeps downstream-step state consistent.
            # Re-raise so the consumer (or task cleanup) proceeds.
            logger.warning(
                "Step %s exited via BaseException %s.",
                step.name,
                type(e).__name__,
            )
            status.state = StepState.FAILED
            if not status.done.is_set():
                status.done.set()
            raise

        finally:
            if not status.done.is_set():
                status.done.set()

    async def _enqueue_step_events(
        self,
        step: Runnable,
        queue: asyncio.Queue,
        statuses: dict[str, StepStatus],
        **kwargs: Any,
    ) -> None:
        """Drive one step for concurrent execution, forwarding its events to the queue.

        Sentinel contract: exactly one sentinel — the step's own StepStatus —
        is pushed on every exit path. The consumer in :meth:`handler`
        decrements ``remaining`` per StepStatus and reads its ``signal``; a
        missed sentinel would hang the consumer on ``queue.get()`` forever,
        so the finally guarantees it even for BaseException exits.
        """
        status = statuses[step.name]
        try:
            async for event in self._run_step(step, statuses, **kwargs):
                # put_nowait: the queue is unbounded, so put() never suspends —
                # awaiting it is pure coroutine overhead per event.
                queue.put_nowait(event)
        finally:
            try:
                queue.put_nowait(status)
            except Exception:
                logger.exception("Failed to enqueue sentinel for step %s", step.name)

    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Execute all steps, respecting dependencies.

        Linear chains (each step depends exactly on its predecessor — the
        common sequential pipeline) run in a direct loop with no tasks or
        queue: concurrency is impossible, so the fan-in machinery is pure
        overhead. Everything else runs one task per step multiplexed
        through a queue.

        When multiple parallel steps pause (approval gate or suspend()), every
        pause is drained (so each step emits its OutputEvent + Approval/
        Interaction event) before the workflow itself raises ``PauseRequired``.
        This lets a caller collect every pending id from a single run and resume
        with all of them at once via ``resume={...}``. Mirrors the agent's
        tool-multiplexing behaviour and prevents the first pause from cancelling
        later ones.
        """
        statuses = {step_name: StepStatus() for step_name in self._steps.keys()}
        first_pending_pause: PauseRequired | None = None
        first_pending_exception: Exception | None = None

        def _record_signal(signal: BaseException | None) -> None:
            nonlocal first_pending_pause, first_pending_exception
            if isinstance(signal, InterruptError):
                raise signal
            if isinstance(signal, PauseRequired):
                if first_pending_pause is None:
                    first_pending_pause = signal
            elif isinstance(signal, Exception):
                if first_pending_exception is None:
                    first_pending_exception = signal

        if self._is_linear:
            # Sequential fast path — insertion order is topological (links can
            # only point at already-registered steps).
            current_task = asyncio.current_task()

            def _cancel_pending() -> bool:
                # A cancelled step records 'interrupted' on its own span and
                # swallows the CancelledError. In concurrent mode the workflow
                # still sees the cancellation at queue.get() (and never yields
                # the interrupted step's final event); here we share the task
                # with the step, so detect the still-pending cancel request and
                # re-raise before forwarding post-cancel events.
                return current_task is not None and current_task.cancelling()

            for step in self._steps.values():
                try:
                    async for event in self._run_step(step, statuses, **kwargs):
                        if _cancel_pending():
                            raise asyncio.CancelledError
                        yield event
                except (asyncio.CancelledError, GeneratorExit, InterruptError):
                    raise
                except BaseException:  # noqa: BLE001
                    # Parity with concurrent mode, where a BaseException from a
                    # user `when`/resolver callable dies inside the step task
                    # (gather(..., return_exceptions=True)) and the workflow
                    # surfaces the failure via failed_steps below. _run_step
                    # already marked the step FAILED and logged.
                    pass
                if _cancel_pending():
                    raise asyncio.CancelledError
                _record_signal(statuses[step.name].signal)
        else:
            queue = asyncio.Queue()
            tasks = [
                asyncio.create_task(self._enqueue_step_events(step, queue, statuses, **kwargs))
                for step in self._steps.values()
            ]
            try:
                remaining = len(tasks)
                while remaining > 0:
                    item = await queue.get()
                    if isinstance(item, StepStatus):
                        remaining -= 1
                        _record_signal(item.signal)
                    else:
                        yield item
            except (asyncio.CancelledError, InterruptError):
                raise
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        # A pause (approval or suspend) takes precedence over a step error in
        # the surfaced status; both ride the same durable resume rails.
        if first_pending_pause is not None:
            raise first_pending_pause
        if first_pending_exception is not None:
            raise first_pending_exception
        failed_steps = sorted(
            (name, step_status)
            for name, step_status in statuses.items()
            if step_status.state == StepState.FAILED
        )
        if failed_steps:
            step_name, step_status = failed_steps[0]
            raise WorkflowStepError(step_name, step_status.error)
