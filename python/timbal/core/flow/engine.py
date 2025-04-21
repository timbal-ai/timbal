import asyncio
import copy
import inspect
import re
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import structlog
from pydantic import BaseModel, Field, TypeAdapter
from uuid_extensions import uuid7

from ...errors import (
    DataKeyError,
    FlowExecutionError,
    InvalidLinkError,
    StepExecutionError,
    StepKeyError,
)
from ...state import RunContext, Snapshot
from ...state.data import (
    BaseData,
    DataError,
    DataMap,
    DataValue,
    get_data_key,
)
from ...state.savers.base import BaseSaver
from ...steps.llms.gateway import handler as llm
from ...types import (
    Message,
    TextContent,
    Tool,
    ToolResultContent,
    ToolUseContent,
)
from ...types.events import (
    ChunkEvent,
    OutputEvent,
    StartEvent,
)
from ...types.models import create_model_from_fields, dump, merge_model_fields
from ..base import BaseStep
from ..shared import RunnableLike
from ..step import Step
from ..stream import AsyncGenState, handle_event, sync_to_async_gen
from .link import Link
from .utils import Dag, get_ancestors, get_sources, get_successors, is_dag

logger = structlog.get_logger("timbal.core.flow.engine")


class Flow(BaseStep):
    """A class representing a flow of connected steps for executing complex workflows.

    A Flow is a directed acyclic graph (DAG) where nodes are steps and edges define execution order
    and data dependencies. Steps can be functions, LLMs, or other flows. The Flow class handles:

    - Step management and execution ordering
    - Data passing between steps
    - Parameter validation
    - LLM memory management
    - Tool integration for LLM agents
    - State persistence
    - Streaming results

    Attributes:
        steps: Dictionary mapping step IDs to step instances
        links: Dictionary mapping link IDs to link instances
        data: Dictionary storing flow data and mappings
        outputs: Dictionary mapping output names to data keys
        state_saver: Optional saver for persisting flow state
        is_llm: Whether this flow acts as an LLM (always False)
        is_coroutine: Whether this flow returns a coroutine
        is_async_gen: Whether this flow returns an async generator
        _is_compiled: Whether the flow has been compiled
    """
    
    def __init__(
        self, 
        id: str = "flow", 
        path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Flow instance.

        Args:
            id: Optional identifier for the flow. Defaults to "flow".
            path: Optional path for the flow. Use when directly adding this to a flow or subflow.
            **kwargs: Keyword arguments will be passed to the BaseStep's __init__ method.
        """
        if path is None:
            path = id
        super().__init__(id=id, path=path, **kwargs)

        self.steps: dict[str, BaseStep] = {}
        self.links: dict[str, Link] = {}
        self.data: dict[str, BaseData] = {}
        self.outputs: dict[str, str] = {}

        self.state_saver: BaseSaver | None = None

        # ? Think if a subflow could ever act as an LLM (for the moment no, because it does not return anthropic or openai events).
        self.is_llm = False
        # These are used to determine if we need to run the step in an executor.
        self.is_coroutine = False
        self.is_async_gen = True

        self._is_compiled = False

    
    def prefix_path(self, prefix: str) -> None:
        """Prefix the flow's path with a given path."""
        self.path = f"{prefix}.{self.id}"
        for step in self.steps.values():
            step.prefix_path(self.path)

    
    def get_dags(self) -> tuple[Dag, Dag]:
        """Returns the DAG representing the flow's execution order and the reverse DAG for the flow."""
        if hasattr(self, '_dag') and hasattr(self, '_rev_dag'):
            return self._dag, self._rev_dag

        dag = {step_id: set() for step_id in self.steps}
        rev_dag = {step_id: set() for step_id in self.steps}
        for link in self.links.values():
            dag[link.step_id].add(link.next_step_id)
            rev_dag[link.next_step_id].add(link.step_id)
        return dag, rev_dag


    def get_sources(self) -> list[str]:
        """Returns the sources of the flow (i.e. steps with no incoming edges)."""
        if hasattr(self, '_sources'): 
            return self._sources

        dag, _ = self.get_dags()
        sources = get_sources(dag)
        return list(sources)

    
    def params_model(self) -> BaseModel:
        """Returns the Pydantic model defining the expected parameters for this step."""
        if hasattr(self, '_params_model'):
            return self._params_model

        params_model_fields = defaultdict(list)
        for data_key, data_value in self.data.items():
            if not isinstance(data_value, DataMap):
                continue
            map_key = data_value.key
            if map_key in self.data or "." in map_key:
                continue
            step_id, step_param_name = data_key.split(".")
            step = self.steps[step_id]
            step_params = step.params_model()
            step_field_info = step_params.model_fields[step_param_name]
            params_model_fields[map_key].append(step_field_info)

        for param_name, fields_infos in params_model_fields.items():
            if len(fields_infos) == 1:
                params_model_fields[param_name] = fields_infos[0]
            else: # len(steps_fields_infos) > 1
                params_model_fields[param_name] = merge_model_fields(*fields_infos)

        params_model_name = f"Flow_{self.id}_params"
        params_model = create_model_from_fields(
            name=params_model_name,
            model_fields=params_model_fields
        )
        return params_model
        

    def params_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the step's parameter model."""
        if hasattr(self, '_params_model_schema'):
            return self._params_model_schema

        return self.params_model().model_json_schema()


    def return_model(self) -> Any:
        """Returns the expected return type for this step."""
        if hasattr(self, '_return_model'):
            return self._return_model

        fields = {}
        # auto_loops = self.get_auto_loops()
        for return_param_name, data_key in self.outputs.items():

            # If there's a match in the data dictionary, we create a model with the type of the data value.
            try: 
                data_value = get_data_key(self.data, data_key)
                fields[return_param_name] = Field(...)
                fields[return_param_name].annotation = type(data_value)
                continue
            # KeyError is raised when we found a match in the data dictionary, but the specifier 
            # is wrong (e.g. attempting to access a property of a dict that does not exist).
            except KeyError:
                fields[return_param_name] = Field(...)
                fields[return_param_name].annotation = Any
                continue
            except DataKeyError:
                pass

            data_key_parts = data_key.split(".")
            if len(data_key_parts) == 2 and data_key_parts[-1] == "return":
                step_id = data_key_parts[0]
                step = self.steps[step_id]
                step_return_model = step.return_model()
                # Generate a FieldInfo with the model.
                fields[return_param_name] = Field(...)
                fields[return_param_name].annotation = step_return_model
            # TODO If step_id.return.0 -> fetch the type of the list
            # TODO If step_id.return.abc -> fetch the type of the dict value / base model field
            # ? Perhaps we could create an analogous function to get_data_key for navigating the fields definitions
            else:
                fields[return_param_name] = Field(...)
                fields[return_param_name].annotation = Any
        
        if not len(fields):
            return None

        return_model_name = f"Flow_{self.id}_return"
        return_model = create_model_from_fields(
            name=return_model_name, 
            model_fields=fields
        )
        return return_model


    def return_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the flow return model."""
        if hasattr(self, '_return_model_schema'):
            return self._return_model_schema

        return TypeAdapter(self.return_model()).json_schema()


    def _collect_outputs(self, data: dict[str, BaseData]) -> Any:
        """Aux method to collect the outputs from the data dictionary."""

        errors = set()
        outputs = {}
        for output_name, data_key in self.outputs.items():
            try:
                output_value = get_data_key(data, data_key)
                if isinstance(output_value, DataError): 
                    flow_execution_error_key = f"{self.path}.{data_key.split('.')[0]}"
                    errors.add(flow_execution_error_key)
                else:
                    outputs[output_name] = output_value
            except DataKeyError:
                flow_execution_error_key = f"{self.path}.{data_key.split('.')[0]}"
                errors.add(flow_execution_error_key)
        
        if len(errors):
            raise FlowExecutionError(f"Error collecting outputs {errors}.")
        return outputs
    

    @staticmethod
    def _cut_memory(memory: list[Message], max_window_size: int) -> list[Message]:
        """Cuts the memory list to maintain the specified window size while preserving tool context.

        This method ensures that when truncating the conversation history:
        1. The memory stays within the specified max window size
        2. Tool use/result pairs remain together - if a tool result is in the window, 
           its corresponding tool use call is also preserved
        
        Args:
            memory: List of Message objects representing the conversation history
            max_window_size: Maximum number of messages to keep in the window

        Returns:
            list[Message]: Truncated memory list that preserves tool context
            
        Example:
            If max_window_size=3 but message 4 contains a tool result referencing 
            a tool use in message 1, the returned window will include message 1 
            to maintain the tool context:
            [msg1(tool_use), msg2, msg3, msg4(tool_result)] -> [msg1, msg3, msg4]
        """
        if len(memory) <= max_window_size:
            return memory
        
        final_window_size = max_window_size
        window_tool_use_ids = set()
        window_tool_result_ids = set()
        for message in memory[-max_window_size:]:
            for content in message.content:
                if isinstance(content, ToolResultContent):
                    window_tool_result_ids.add(content.id)
                if isinstance(content, ToolUseContent):
                    window_tool_use_ids.add(content.id)
        window_tool_result_ids_missing = window_tool_result_ids - window_tool_use_ids

        i = len(memory) - max_window_size - 1
        while i >= 0:
            if not len(window_tool_result_ids_missing):
                break
            message = memory[i]
            for content in message.content:
                if isinstance(content, ToolUseContent):
                    if content.id in window_tool_result_ids_missing:
                        window_tool_result_ids_missing.remove(content.id)
                        # ? We could study grabbing the message that triggered the tool call as well for context. 

            final_window_size += 1
            i -= 1

        return memory[-final_window_size:]


    def _resolve_step_args(
        self, 
        step_id: str, 
        data: dict[str, BaseData],
    ) -> dict[str, Any]:
        """Resolves the arguments for a step given a data state.

        Args:
            step_id: Identifier of the step to resolve arguments for
            data: Dictionary containing all flow data, including inputs and previous step results

        Returns:
            Dictionary containing the resolved arguments for the step

        Note:
            These arguments won't be validated against the step's parameter model. 
            Validation must be performed by the caller.
        """
        step = self.steps[step_id]
        step_params_model = step.params_model()
        step_args = {}

        for step_param_name in step_params_model.model_fields.keys():
            step_param_key = f"{step_id}.{step_param_name}"
            step_param_data = data.get(step_param_key, None)

            if step_param_data is None:
                continue

            step_param = step_param_data.resolve(context_data=data)

            # This param will go directly to the step params model validation.
            # We don't want to pass an explicit None. 
            # The pydantic model validation will handle this if the field is optional.
            if step_param is None:
                continue

            step_args[step_param_name] = step_param
        
        _, rev_dag = self.get_dags()

        for ancestor_id in rev_dag[step_id]:
            link = self.links[f"{ancestor_id}-{step_id}"]

            if link.is_tool:
                ancestor_output = data.get(f"{link.step_id}.return", None)
                if not ancestor_output:
                    continue
                ancestor_output = ancestor_output.resolve(context_data=data)
                if not isinstance(ancestor_output, Message):
                    continue
                for content in ancestor_output.content:
                    if isinstance(content, ToolUseContent):
                        if content.name == step_id:
                            step_args = {**content.input, **step_args}

            if link.is_tool_result:
                ancestor_output = data.get(f"{link.step_id}.return", None)
                if ancestor_output is None:
                    continue 
                ancestor_output = ancestor_output.resolve(context_data=data)
                # Check memory to fetch the tool use ID. SDKs will need this to be able to match results to tool calls.
                step_memory = data.get(f"{step_id}.memory", None)
                if step_memory is None:
                    continue
                step_memory = step_memory.resolve(context_data=data)
                step_memory_tool_result_inserted = False
                for message in step_memory[::-1]: # Check in reverse. Tool calls will likely be the last messages.
                    for content in message.content:
                        if isinstance(content, ToolUseContent):
                            if content.name == link.step_id and not step_memory_tool_result_inserted:
                                tool_result_message = Message(
                                    role="user",
                                    content=[ToolResultContent(
                                        id=content.id,
                                        content=[TextContent(text=str(ancestor_output))] # TODO Rethink str() cast.
                                    )]
                                )
                                step_memory.append(tool_result_message)
                                step_memory_tool_result_inserted = True

        # Hack for injecting a prompt into the LLM's memory as last message.
        if step.is_llm:
            step_prompt = step_args.pop("prompt", None)
            if step_prompt is not None:
                # Ensure the prompt is a user message.
                if isinstance(step_prompt, Message) and step_prompt.role != "user":
                    step_prompt = Message(
                        role="user",
                        content=step_prompt.content
                    )
                # If the prompt comes from an unvalidated pydantic model field, we need to validate it.
                else:
                    step_prompt = Message.validate(step_prompt)
                
                if not isinstance(step_prompt, Message):
                    raise ValueError(f"Prompt must be an instance of Message. Got {type(step_prompt)} for step {f'{self.path}.{step_id}'}.")
                
                if "memory" in step_args:
                    step_args["memory"].append(step_prompt)
                    # Limit the memory size. This is individual per llm step. Hence, we can have multiple LLMs pointing
                    # to the same memory key but have different memory window sizes.
                    if step._memory_window_size is not None:
                        step_args["memory"] = self._cut_memory(step_args["memory"], step._memory_window_size)
                else:
                    step_args["memory"] = [step_prompt]

        return step_args


    async def _run_step(
        self, 
        step_id: str, 
        data: dict[str, Any],
        context: RunContext,
    ) -> tuple[Any, Any]:
        """Executes a single step in the flow and returns its result.

        Resolves step arguments from the data dictionary, validates them against the step's parameter model,
        and executes the step. Handles both synchronous and asynchronous step execution.

        Args:
            step_id: Identifier of the step to execute
            data: Dictionary containing all flow data, including inputs and previous step results
            context: Run context


        Returns:
            The step's input arguments
            The step's output (direct value, generator, or awaited coroutine)

        Note:
            While this function is async, it doesn't await generator results - those are handled by the caller.
            Only direct coroutines from step.run() are awaited here.
        """
        step = self.steps[step_id]

        step_input = self._resolve_step_args(step_id, data)
        # We need to copy to avoid the LLM memories being modified by reference.
        step_input_dump = dump(step_input, context)

        try:
            # Flow input is validated in the .run() method. That is because we want to validate
            # the first initial call to the parent flow as well.
            if not isinstance(step, Flow):
                step_input = dict(step.params_model().model_validate(step_input))
            # If we're dealing with a regular sync function, we need to run it in an executor to 
            # avoid blocking the event loop.
            if not step.is_coroutine and not step.is_async_gen:
                loop = asyncio.get_running_loop()
                step_result = await loop.run_in_executor(None, lambda: step.run(
                    context=context,
                    **step_input
                ))
                if inspect.isgenerator(step_result):
                    return step_input, sync_to_async_gen(step_result, loop)
                return step_input, step_result
            
            step_result = step.run(
                context=context,
                **step_input
            )

            if step.is_coroutine:
                step_result = await step_result
            
            return step_input_dump, step_result

        except Exception as e:
            raise StepExecutionError(step_input_dump, e) from e


    async def run(
        self, 
        context: RunContext | None = None,
        **kwargs: Any
    ) -> Any:
        """Executes the step's processing logic.
        
        Args:
            context: RunContext
            **kwargs: Additional keyword arguments required for step execution.
        
        Returns:
            Any: The step's processing result. Can be any object, a coroutine or an async generator.
        """
        if context is None:
            context = RunContext(id=uuid7(as_type="str"))
        elif context.id is None:
            context.id = uuid7(as_type="str")

        t0 = int(time.time() * 1000)

        flow_start_event = StartEvent(
            run_id=context.id,
            path=self.path,
        )

        logger.info("start_event", start_event=flow_start_event)
        yield flow_start_event

        # Copy the data to avoid potential bugs with data being modified by reference.
        data = copy.deepcopy(self.data)
        data.update({k: DataValue(value=v) for k, v in kwargs.items()})

        # Here we'll store all the steps outputs and run data.
        steps = {}
        flow_usage = {}

        # Load LLM memories.
        # Hence if this is a root run (no parent_id), we don't need to load any previous snapshot.
        # We do this now so if the validation fails or anything happens we store the last snapshot
        # with the last available information always.
        # TODO Optimize this. There's no need to load previous snapshot if there are no memories to load.
        if self.state_saver is not None and context.parent_id is not None:
            try:
                if self._is_state_saver_get_async:
                    last_snapshot = await self.state_saver.get_last(path=self.path, context=context)
                else:
                    last_snapshot = self.state_saver.get_last(path=self.path, context=context)
            except Exception as err:
                logger.error("get_memory_error", err=err)
                last_snapshot = None

            if last_snapshot is not None:
                last_snapshot_data = last_snapshot.data
                window_sizes = {}

                for _, step in self.steps.items():
                    if step.is_llm and hasattr(step, "_memory_key"):
                        memory_key = step._memory_key
                        if memory_key not in window_sizes:
                            window_sizes[memory_key] = step._memory_window_size
                        elif window_sizes[memory_key] is not None and step._memory_window_size is not None:
                            window_sizes[memory_key] = max(window_sizes[memory_key], step._memory_window_size)
                        else:
                            window_sizes[memory_key] = None
                        data[memory_key] = copy.deepcopy(last_snapshot_data[memory_key])
                        # We could defer this to pydantic inbuilt validation. We do it here to avoid issues when
                        # processing tools results, where code expects memory to contain a list of proper messages.
                        if isinstance(data[memory_key], DataValue):
                            data[memory_key] = DataValue(value=[Message.validate(message) for message in data[memory_key].resolve()])
                
                # Limit the memory sizes. Take into account the maximum amount of memory required by any of the steps.
                # If we can cut it, we cut it to avoid storing always all the data (potentially growing indefinitely).
                for memory_key, max_window_size in window_sizes.items():
                    if max_window_size is not None:
                        if max_window_size == 0:
                            data[memory_key] = DataValue(value=[])
                        else:
                            memory = data[memory_key].resolve(context_data=last_snapshot_data)
                            if len(memory) > max_window_size:
                                memory = self._cut_memory(memory, max_window_size)
                                data[memory_key] = DataValue(value=memory)
        
        # Copy the input as is, so we have the traces without validated data and defaults.
        flow_input = dump(kwargs, context=context)

        # Validate kwargs to store the validated inputs in the snapshot.
        try:
            kwargs = dict(self.params_model().model_validate(kwargs))
        except Exception as e:
            error = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

            if self.state_saver is not None:
                t1 = int(time.time() * 1000)
                snapshot = Snapshot(
                    v="0.2.0",
                    id=context.id,
                    parent_id=context.parent_id,
                    path=self.path,
                    input=flow_input,
                    output=None,
                    error=error,
                    t0=t0,
                    t1=t1,
                    data=data,
                )

                # We don't want to cancel the execution if this errors. 
                try:
                    if self._is_state_saver_put_async:
                        await self.state_saver.put(snapshot=snapshot, context=context)
                    else:
                        self.state_saver.put(snapshot=snapshot, context=context)
                except Exception as err:
                    logger.error("put_memory_error", err=err)

            # TODO Change this error.
            raise FlowExecutionError(f"Error validating kwargs for step {self.path}.") from e
        
        # Compute (or retrieve the pre-computed) DAGs that determine the flow's execution order.
        dag, rev_dag = self.get_dags()

        # Create events for each step to track their completion.
        tasks_completions = {step_id: asyncio.Event() for step_id in self.steps}

        # When working with agents, we'll need to keep track of the steps that don't need to be executed.
        steps_skipped = set()
        # When an error occurs, we'll need to keep track of the steps that need to be skipped.
        steps_to_skip = set()

        # Create tasks for the flow's sources.
        sources_ids = self.get_sources()
        tasks = []
        start_times = {}
        for source_id in sources_ids:
            step_start_event = StartEvent(
                run_id=context.id,
                path=self.steps[source_id].path,
            )

            logger.info("start_event", start_event=step_start_event)
            yield step_start_event
            
            task = asyncio.create_task(self._run_step(
                step_id=source_id, 
                data=data,
                context=context,
            ))
            task.step_id = source_id
            tasks.append(task)
            start_times[source_id] = int(time.time() * 1000)

        async_gens: dict[str, AsyncGenState] = {}

        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks[:] = pending
            
            for completed_task in done:
                step_id = completed_task.step_id
                step_path = self.steps[step_id].path
                step_input = None
                step_result = None
                step_error = None
                step_usage = {}

                if step_id in async_gens:
                    # When dealing with an async generator, we need to handle any possible
                    # exceptions that might occur while consuming the generator.
                    async_gen_state = async_gens[step_id]
                    step_input = async_gen_state.input
                    step_usage = async_gen_state.usage
                    try:
                        step_chunk = await completed_task
                        task = asyncio.create_task(async_gen_state.gen.__anext__())
                        task.step_id = step_id
                        tasks.append(task)
                        step_chunk = handle_event(event=step_chunk, async_gen_state=async_gen_state)

                        if step_chunk is not None:
                            step_chunk_event = ChunkEvent(
                                run_id=context.id,
                                path=step_path,
                                chunk=step_chunk,
                            )

                            logger.info("chunk_event", chunk_event=step_chunk_event)
                            yield step_chunk_event

                        continue

                    except StopAsyncIteration:
                        step_result = async_gen_state.collect()
                        # Tool results are not marked as llm steps. These can include citations (e.g. perplexity).
                        # We add these citations directly formatting the markdown of the text result.
                        if hasattr(async_gen_state, "citations"):
                            step_citations = async_gen_state.citations
                            if isinstance(step_citations, list) and len(step_citations):
                                for i, citation in enumerate(step_citations, start=1):
                                    step_result[-1]["text"] = step_result[-1]["text"].replace(f"[{i}]", f"[{i}]({citation})")
                        # If the step is an LLM, we add the result message to the corresponding memory.
                        if self.steps[step_id].is_llm:
                            step_result = Message.validate({
                                "role": "assistant",
                                "content": step_result,
                            })
                            # Add the message to the appropriate memory.
                            memory_key = f"{step_id}.memory"
                            if memory_key in data:
                                memory = data[memory_key]
                                # TODO We could handle an arbitrary number of nested memories here
                                if isinstance(memory, DataMap): 
                                    assert memory.key.endswith(".memory"), \
                                        f"Memory data map should point to a memory key. Found '{memory.key}'."
                                    memory_key = memory.key
                                memory = memory.resolve(context_data=data)
                                memory.append(step_result)
                                data[memory_key] = DataValue(value=memory)

                    except Exception as e:
                        step_error = {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        }
                        # Stop all the successors to this step, without interrupting the execution
                        # of other possible branches in the flow.
                        step_successors = get_successors(step_id, dag)
                        steps_to_skip.update(step_successors)
                        logger.error(
                            f"Step '{step_path}' failed. Skipping successors {step_successors}.",
                            error=step_error,
                        )
                
                else:
                    # Else the try except is handled in the _run_step method.
                    try:
                        step_input, step_result = await completed_task
                        if inspect.isasyncgen(step_result):
                            async_gens[step_id] = AsyncGenState(
                                gen=step_result,
                                input=step_input,
                            )
                            task = asyncio.create_task(step_result.__anext__())
                            task.step_id = step_id
                            tasks.append(task)
                            continue
                
                    except StepExecutionError as e:
                        step_input = e.input
                        step_error = {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        }
                        # Stop all the successors to this step, without interrupting the execution
                        # of other possible branches in the flow.
                        step_successors = get_successors(step_id, dag)
                        steps_to_skip.update(step_successors)
                        logger.error(
                            f"Step '{step_path}' failed. Skipping successors {step_successors}.",
                            error=step_error,
                        )

                # Store the result in the data dictionary. These will be used for data maps and 
                # collecting outputs from the overall flow execution.
                if step_error is None:
                    data[f"{step_id}.return"] = DataValue(value=step_result)
                else:
                    data[f"{step_id}.return"] = DataError(error=step_error)
                
                # Mark this node as completed. This will unblock the tasks waiting for this node.
                tasks_completions[step_id].set()
                
                # Create tasks for the step successors.
                for successor_id in dag[step_id]:
                    if successor_id in steps_to_skip:
                        continue

                    ancestors = [
                        (ancestor_id, self.links[f"{ancestor_id}-{successor_id}"])
                        for ancestor_id in rev_dag[successor_id]
                    ]

                    if all(
                        tasks_completions[ancestor_id].is_set() 
                        for ancestor_id, _ in ancestors
                        if ancestor_id not in steps_skipped
                    ):
                        # Link conditions are only evaluated once all the ancestors are completed.
                        # If any of the conditions is true, we start the next step.
                        if any(ancestor_link.evaluate_condition(data) for _, ancestor_link in ancestors):
                            step_start_event = StartEvent(
                                run_id=context.id,
                                path=self.steps[successor_id].path,
                            )

                            logger.info("start_event", start_event=step_start_event)
                            yield step_start_event

                            task = asyncio.create_task(self._run_step(
                                step_id=successor_id, 
                                data=data,
                                context=context,
                            ))
                            task.step_id = successor_id
                            tasks.append(task)
                            start_times[successor_id] = int(time.time() * 1000)
                        else:
                            steps_skipped.add(successor_id)
                
                start_time = start_times[step_id]
                end_time = int(time.time() * 1000)

                step_output_event = OutputEvent(
                    run_id=context.id,
                    path=step_path,
                    input=step_input,
                    output=step_result,
                    error=step_error,
                    t0=start_time,
                    t1=end_time,
                    usage=step_usage,
                )

                logger.info("output_event", output_event=step_output_event)
                yield step_output_event

                steps[step_path] = dump(step_output_event, context=context)

                for k, v in step_usage.items():
                    current_key_value = flow_usage.get(k, 0)
                    flow_usage[k] = current_key_value + v

        # Grab the properties as defined in the return_model.
        output = None
        error = None
        exception = None
        try: 
            output = self._collect_outputs(data)
        except FlowExecutionError as e:
            error = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            exception = e

        t1 = int(time.time() * 1000)

        if self.state_saver is not None:
            snapshot = Snapshot(
                v="0.2.0",
                id=context.id,
                parent_id=context.parent_id,
                path=self.path,
                input=kwargs,
                output=output,
                error=error,
                t0=t0,
                t1=t1,
                data=data,
                steps=steps,
                usage=flow_usage,
            )

            # We don't want to cancel the execution if this errors. 
            try:
                if self._is_state_saver_put_async:
                    await self.state_saver.put(snapshot=snapshot, context=context)
                else:
                    self.state_saver.put(snapshot=snapshot, context=context)
            except Exception as err:
                logger.error("put_memory_error", err=err)

        if exception is not None:
            raise exception
        
        flow_output_event = OutputEvent(
            run_id=context.id,
            path=self.path,
            input=flow_input,
            output=output,
            error=None, # If it reaches this point, the flow has completed successfully.
            t0=t0,
            t1=t1,
            usage=flow_usage,
        )

        logger.info("output_event", output_event=flow_output_event)
        yield flow_output_event
    

    async def complete(
        self,
        context: RunContext | None = None,
        **kwargs: Any
    ) -> OutputEvent:
        """Flow.run() wrapper method that completes the flow execution.
        
        Args: 
            context: RunContext
            **kwargs: Additional keyword arguments required for step execution.
        
        Returns:
            OutputEvent: The flow's selected outputs.
        """
        async for event in self.run(context=context, **kwargs):
            if isinstance(event, OutputEvent) and event.path == self.path:
                return event


    def add_step(
        self,
        step: str | RunnableLike,
        action: RunnableLike | None = None,
        # TODO timeout: int | None = None,
        # TODO retry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "Flow":
        """Add a step to the flow.

        Args:
            step: Either:
                - A string ID to identify the step (requires action parameter)
                - The step implementation (BaseStep or callable)
            action: The step implementation when using string ID
            **kwargs: Additional arguments to set as step parameters. Must match the step's param model fields

        Returns:
            The flow instance for chaining

        Raises:
            ValueError: If a step with the same id already exists in the flow
            NotImplementedError: If step type is invalid
            StepKeyError: If any kwarg doesn't match a valid step parameter
        """
        if isinstance(step, str) and action is None:
            raise ValueError("Action is required when adding a step by id.")
        
        if (isinstance(step, BaseStep | Callable) and action is not None and isinstance(action, BaseStep | Callable)):
            raise ValueError("Cannot add a step with two actions.")
        
        if isinstance(step, str):
            id = step
            step = action
        elif callable(step):
            id = step.__name__
        elif isinstance(step, BaseStep):
            # Unfortunatelly we cannot grab the object name since python does not have access 
            # to the variable name the flow is assigned to.
            id = step.id
        
        if id in self.steps:
            raise ValueError(f"Step {id} already exists in the flow.")

        if isinstance(step, BaseStep):
            step = copy.deepcopy(step)
            step.id = id
            # This recursively updates recursively the steps paths so that we can uniquely 
            # identify each step, and subflow within the parent flow.
            step.prefix_path(self.path) # TODO Add tests of how this works with nested subflows and nested agents.
            # When adding a subflow, ensure it is compiled with all the possible optimizations.
            if isinstance(step, Flow) and not step._is_compiled:
                step.compile()
            self.steps[id] = step
        elif callable(step):
            step = Step(id=id, path=f"{self.path}.{id}", handler_fn=step)
            self.steps[id] = step
        else:
            raise NotImplementedError(f"Invalid step type {step}.")

        if not kwargs:
            return self

        # We use the additional kwargs to set the step's params.
        step_params_model_fields = step.params_model().model_fields
        
        for k, v in kwargs.items():
            if k not in step_params_model_fields:
                raise StepKeyError(f"Parameter {k} does not exist in step {id}.")
            data_key = f"{id}.{k}"

            if isinstance(v, DataMap):
                self.set_data_map(data_key, v.key)
            else:
                # Otherwise create a DataValue
                self.set_data_value(data_key, v)

        return self
        
    
    def add_llm(
        self, 
        id: str | None = None, 
        model: str = "gpt-4o-mini", 
        memory_id: str | None = None,
        memory_window_size: int | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        frequency_penalty: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        parallel_tool_calls: bool | None = None,
        json_schema: dict | None = None,
    ) -> "Flow":
        """Adds an LLM step to the flow.

        This method is just a wrapper around add_step that sets the LLM step parameters,
        and exposes the parameters to the user for linters and IDEs.
        Additionally, it marks the step as an LLM step to implement agentic behavior.

        Args:
            id: Unique identifier for this LLM step
            model: Name of the LLM model to use (e.g. "gpt-4-mini")
            memory_id: Optional memory id to use for this LLM. If None, no memory will be used. 
                       If same as step id, creates new memory. If different, reuses existing memory.
            memory_window_size: Maximum number of messages to keep in the memory.
            max_tokens: Maximum number of tokens to generate.
            system_prompt: System prompt to guide the LLM's behavior and role.
            tool_choice: How the model should use the provided tools.
            frequency_penalty: Positive values penalize token frequency to reduce repetition. 
                               Ranges from -2.0 to 2.0.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of top log probabilities to return.
            presence_penalty: Positive values penalize tokens based on presence to encourage new topics. 
                              Ranges from -2.0 to 2.0.
            seed: Seed for deterministic sampling.
            stop: Where the model will stop generating.
            temperature: Amount of randomness injected into the response.
            top_p: Nucleus sampling parameter.
            top_k: Only sample from the top K options for each subsequent token.
            parallel_tool_calls: Whether to call tools in parallel.
            json_schema: JSON schema to use for this LLM.

        Returns:
            Flow: The flow instance for method chaining

        These are some of the models that could be used:
        - OpenAI: gpt-4o, gpt-4o-mini, o1, o3-mini, o1-mini
        - Anthropic: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, 
        claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
        - TogetherAI: deepseek-ai/DeepSeek-R1, deepseek-ai/DeepSeek-V3, 
        meta-llama/Llama-3.3-70B-Instruct-Turbo, meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo, 
        meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo, meta-llama/Llama-3.2-3B-Instruct-Turbo,
        Qwen/Qwen2.5-Coder-32B-Instruct, Qwen/Qwen2-VL-72B-Instruct, 
        mistralai/Mistral-Small-24B-Instruct-2501, mistralai/Mistral-7B-Instruct-v0.3,
        mistralai/Mixtral-8x22B-Instruct-v0.1, meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo
        - Gemini: gemini-2.0-flash-lite-preview-02-05, gemini-2.0-flash, gemini-1.5-flash, 
        gemini-1.5-flash-8b, text-embedding-004
        """
        add_step_kwargs = {"model": model}

        if system_prompt is not None:
            add_step_kwargs["system_prompt"] = system_prompt
        if max_tokens is not None:
            add_step_kwargs["max_tokens"] = max_tokens
        if tool_choice is not None:
            add_step_kwargs["tool_choice"] = tool_choice
        if frequency_penalty is not None:
            add_step_kwargs["frequency_penalty"] = frequency_penalty
        if logprobs is not None:
            add_step_kwargs["logprobs"] = logprobs
        if top_logprobs is not None:
            add_step_kwargs["top_logprobs"] = top_logprobs
        if presence_penalty is not None:
            add_step_kwargs["presence_penalty"] = presence_penalty
        if seed is not None:
            add_step_kwargs["seed"] = seed
        if stop is not None:
            add_step_kwargs["stop"] = stop
        if temperature is not None:
            add_step_kwargs["temperature"] = temperature
        if top_p is not None:
            add_step_kwargs["top_p"] = top_p
        if top_k is not None:
            add_step_kwargs["top_k"] = top_k
        if parallel_tool_calls is not None:
            add_step_kwargs["parallel_tool_calls"] = parallel_tool_calls
        if json_schema is not None:
            add_step_kwargs["json_schema"] = json_schema
        
        if id is None:
            id = "llm"

        self.add_step(id, llm, **add_step_kwargs)
    
        # Mark the step as an LLM step. This will be used for easier checks in other places to implement agentic behavior.
        self.steps[id].is_llm = True

        if memory_id is not None:
            memory_key = f"{memory_id}.memory"
            # We're reusing the memory of another step.
            if memory_key in self.data:
                data_key = f"{id}.memory"
                self.set_data_map(data_key, memory_key, autolink=False)
            # We're creating a new memory with the specified key. 
            else:
                self.set_data_value(memory_key, [])
                if memory_id != id:
                    self.set_data_map(f"{id}.memory", memory_key, autolink=False)
            
            # Store memory config in the llm step, for resolving the memory at runtime.
            self.steps[id]._memory_key = memory_key
            self.steps[id]._memory_window_size = memory_window_size

        return self


    def remove_step(
        self, 
        id: str,
    ) -> "Flow":
        """Removes a step from the flow.

        This will remove the step from the flow, and all its associated data, outputs and links.
        The expected behavior is that if we were to create a new step with the same id, it would be
        a new step, and not inherit all the old data, outputs and links.

        Args:
            id: The id of the step to remove.

        Returns:
            The flow instance for method chaining.
        """
        links_to_remove = [
            link.id 
            for link in self.links.values()
            if link.step_id == id or link.next_step_id == id
        ]
        for link_id in links_to_remove:
            self.remove_link(link_id)

        data_key_prefix_to_remove = f"{id}."

        outputs_to_remove = [
            output_name 
            for output_name, output_data_key in self.outputs.items() 
            if output_data_key.startswith(data_key_prefix_to_remove)
        ]
        for output_name in outputs_to_remove:
            self.remove_output(output_name)
        
        data_key_prefix_to_remove = f"{id}."
        data_to_remove = [
            data_key 
            for data_key, data in self.data.items() 
            if data_key.startswith(data_key_prefix_to_remove) or
                (isinstance(data, DataMap) and data.key.startswith(data_key_prefix_to_remove))
        ]
        for data_key in data_to_remove:
            self.remove_data(data_key)

        self.steps.pop(id, None)
        return self

    
    # TODO Consider the idea of disabling tool params from here.
    def add_link(
        self, 
        step_id: str, 
        next_step_id: str, 
        condition: str | None = None, 
        is_tool: bool = False,
        description: str | None = None,
        is_tool_result: bool = False,
    ) -> "Flow":
        """Adds a link between two steps.

        Links are used to define the flow execution order. 
        A link from step A to step B means that step B will only start after step A has completed.
        Links can be used so that next steps or previous steps are handled as tools for LLMs.

        Args:
            step_id: The id of the step to link from.
            next_step_id: The id of the step to link to.
            condition: The condition to evaluate to determine if the link should be followed.
                       If the condition is true, the link is followed.
            is_tool: Whether the link is a tool call.
            description: The description of the tool.
            is_tool_result: Whether the link is a tool result.
        
        Returns:
            The flow instance for method chaining.
        """
        if step_id not in self.steps:
            raise InvalidLinkError(f"Step {step_id} not found.")

        if next_step_id not in self.steps:
            raise InvalidLinkError(f"Step {next_step_id} not found.")
        
        link_id = f"{step_id}-{next_step_id}"
        if link_id in self.links:
            raise ValueError(f"Link from {step_id} to {next_step_id} already exists.")

        dag, rev_dag = self.get_dags()
        next_step_ancestors = get_ancestors(node_id=next_step_id, rev_dag=rev_dag)
        if step_id in next_step_ancestors:
            raise InvalidLinkError(f"Step {step_id} is already an ancestor of step {next_step_id}.")

        dag[step_id].add(next_step_id)
        if not is_dag(graph=dag):
            raise InvalidLinkError(f"Link from {step_id} to {next_step_id} creates a cycle.")

        if is_tool:
            if not self.steps[step_id].is_llm:
                raise InvalidLinkError(f"Cannot add link '{step_id}-{next_step_id}' as a tool, " \
                                       f"because step '{step_id}' is not an LLM.")

            tool_input_schema = self.steps[next_step_id].params_model_schema()

            tool = Tool(
                name=next_step_id,
                description=description if description else "",
                input_schema=tool_input_schema,
            )

            existing_tools = self.data.get(f"{step_id}.tools", None)
            if existing_tools:
                assert isinstance(existing_tools, DataValue), "Tools should always be instances of DataValue."
                existing_tools = existing_tools.resolve(context_data=None) 
                existing_tools_names = [existing_tool.name for existing_tool in existing_tools]
            else:
                existing_tools = []
                existing_tools_names = []

            if next_step_id not in existing_tools_names:
                existing_tools.append(tool)
            self.set_data_value(f"{step_id}.tools", existing_tools)
        
        if is_tool_result:
            if not self.steps[next_step_id].is_llm:
                raise InvalidLinkError(f"Cannot add link '{step_id}-{next_step_id}' as a tool result, " \
                                       f"because step '{next_step_id}' is not an LLM.")
            # TODO We could further validate that the next step shares de memory with the previous tool step.
            # ! I don't love this. 'memory' is not an argument of the tool per se. We're abusing the data maps.
            self.set_data_map(f"{step_id}.memory", f"{next_step_id}.memory", autolink=False)
        
        link = Link(
            step_id=step_id, 
            next_step_id=next_step_id, 
            condition=condition, 
            is_tool=is_tool,
            is_tool_result=is_tool_result,
        )
        self.links[link.id] = link

        return self
    

    def remove_link(self, id: str) -> "Flow":
        """Removes a link from the flow.

        Args:
            id: The id of the link to remove.

        Returns:
            The flow instance for method chaining.
        """
        self.links.pop(id, None)
        return self


    def set_input(
        self,
        data_key: str, 
        input_key: str | None = None,
    ) -> "Flow":
        """Sets an input mapping to the flow.

        Args:
            data_key: The key in the data dict to retrieve as an input.
            input_key: Whether to map this input value to a specific key.
        
        Returns:
            The flow instance for method chaining.
        """
        self.set_data_map(data_key, input_key)
        return self
    

    def set_output(
        self, 
        data_key: str,
        output_key: str | None = None, # TODO
    ) -> "Flow":
        """Adds an output mapping to the flow.

        Careful this does not validate the data key or anything.
        If there's a mapping error or missing key it will return an error at runtime.

        This overrides the previous output mapping for the same name.

        Args:
            data_key: The key in the data dict to retrieve as an output.
            output_key: Whether to map this output value to a specific key.
        
        Returns:
            The flow instance for method chaining.
        """
        self.outputs[output_key] = data_key
        return self


    def remove_output(
        self, 
        output_key: str,
    ) -> "Flow":
        """Removes an output mapping from the flow.

        Args:
            output_key: The key of the output to remove.

        Returns:
            The flow instance for method chaining.
        """
        self.outputs.pop(output_key, None)
        return self


    def set_data_value(
        self,
        data_key: str,
        data_value: Any,
        autolink: bool = True,
    ) -> "Flow":
        """Sets a data value for a data key.

        This method does not check if the data key exists in the flow data.
        It also does not check if the data value is valid (e.g. if the data key is a set param).
        Careful, if the data value provided to a step param is not valid, it will raise an error at runtime.

        Args:
            data_key: The data key to set the value for.
            data_value: The value to set.
            autolink: If string interpolation is used to map to other steps values, 
                      whether to automatically add a link between the source and target steps.
        Returns:
            The flow instance for method chaining.
        """
        self.data[data_key] = DataValue(value=data_value)

        if not autolink:
            return self

        if isinstance(data_value, str) and "{{" in data_value and "}}" in data_value:
            target_step_id = data_key.split(".")[0] if "." in data_key else None
            template_refs = re.findall(r"\{\{([\w\.]+)\}\}", data_value)
            
            for ref in template_refs:
                if "." not in ref:
                    continue
                source_step_id = ref.split(".")[0]
                if f"{source_step_id}-{target_step_id}" not in self.links:
                    self.add_link(source_step_id, target_step_id)

        return self

                
    def set_data_map(
        self,
        data_key: str,
        map_key: str,
        default: Any | None = "__NO_DEFAULT__",
        autolink: bool = True,
    ) -> "Flow":
        """Sets a map from a step param to a data key.

        This method does not check if the data key or the map key exist
        in the flow data. This allows for dynamic mapping during runtime (and other cool stuff).
        Careful, if the mapping is not valid at runtime, it will raise an error.

        Args:
            data_key: The data key of the step param to map.
            map_key: The key to map the data key to.
            default: The default value to return if the data key is not found.
            autolink: Whether to automatically add a link between the source and target steps.
        
        Returns:
            The flow instance for method chaining.
        """
        # TODO We could validate that the data key is of an appropriate type.

        if data_key in self.data:
            existing_data = self.data[data_key]
            if isinstance(existing_data, DataMap):
                # ? We could study overriding the existing map.
                raise ValueError(f"Cannot set multiple data maps for the same data key: {data_key}.")
            elif isinstance(existing_data, DataValue):
                # We use the previous data value as the default for the new map.
                if default is None:
                    default = existing_data.value
            else:
                raise NotImplementedError(
                    f"Cannot set data map for data key: {data_key} with value of type {type(existing_data)}.")

        self.data[data_key] = DataMap(key=map_key, default=default)

        if autolink and "." in data_key and "." in map_key:
            target_step_id = data_key.split(".")[0]
            if target_step_id not in self.steps:
                logger.warning(f"Cannot auto-add link for {data_key} -> {map_key}")
                return self

            source_step_id = map_key.split(".")[0]
            if source_step_id not in self.steps:
                logger.warning(f"Cannot auto-add link for {data_key} -> {map_key}")
                return self

            if f"{source_step_id}-{target_step_id}" not in self.links:
                self.add_link(source_step_id, target_step_id)

        return self

    
    def remove_data(
        self,
        key: str,
    ) -> "Flow":
        """Removes a data key from the flow.

        Args:
            key: The data key to remove.

        Returns:
            The flow instance for method chaining.
        """
        self.data.pop(key, None)
        return self


    def compile(
        self,
        state_saver: BaseSaver | None = None,
    ) -> "Flow":
        """Compiles the flow by pre-computing commonly used properties and 'caching' them.

        This method:
        - Pre-computes and caches the DAG and reverse DAG for execution order.
        - Pre-computes and caches the parameter and return models and schemas.
        - Sets the state saver for persisting flow state.

        Args:
            state_saver: Optional BaseSaver instance to persist flow state during execution.
        
        Returns:
            The flow instance for method chaining.
        """
        dag, rev_dag = self.get_dags()
        self._dag = dag
        self._rev_dag = rev_dag

        sources = self.get_sources()
        self._sources = sources

        self._params_model = self.params_model()
        self._params_model_schema = self.params_model_schema()
        self._return_model = self.return_model()
        self._return_model_schema = self.return_model_schema()

        self.state_saver = state_saver
        if self.state_saver is not None:
            self._is_state_saver_get_async = inspect.iscoroutinefunction(self.state_saver.get_last)
            self._is_state_saver_put_async = inspect.iscoroutinefunction(self.state_saver.put)

        # Mark the flow as compiled to prevent re-compiling when importing as a subflow.
        self._is_compiled = True

        return self
    