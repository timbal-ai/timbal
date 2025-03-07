import asyncio
import copy
import inspect
import re
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import structlog
from pydantic import BaseModel, Field

from ..errors import DataKeyError, InvalidLinkError, StepKeyError
from ..state import Snapshot
from ..state.data import BaseData, DataMap, DataValue, get_data_key
from ..state.savers.base import BaseSaver
from ..steps.llms.gateway import handler as llm
from ..types import (
    Message,
    TextContent,
    Tool,
    ToolResultContent,
    ToolUseContent,
)
from ..types.events import (
    FlowOutputEvent,
    StepChunkEvent,
    StepOutputEvent,
    StepStartEvent,
)
from ..types.models import create_model_from_fields, issubclass_safe, merge_model_fields
from .base import BaseStep
from .link import Link
from .step import Step
from .stream import AsyncGenState, handle_event, sync_to_async_gen
from .utils import Dag, get_ancestors, get_sources, is_dag

logger = structlog.get_logger("timbal.graph.flow")

RunnableLike = BaseStep | Callable[..., Any]

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
        self.is_agent = False
        # These are used to determine if we need to run the step in an executor.
        self.is_coroutine = False
        self.is_async_gen = True

        self._is_compiled = False

    
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

    
    def params_model(self) -> BaseModel:
        """Returns the Pydantic model defining the expected parameters for this step."""
        if hasattr(self, '_params_model'):
            return self._params_model

        params_model_fields = defaultdict(list)
        for data_key, data_value in self.data.items():
            if not isinstance(data_value, DataMap):
                continue
            map_key = data_value.key
            if map_key in self.data:
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


    def return_model(self) -> BaseModel:
        """Returns the Pydantic model defining the expected return outputs for this flow."""
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
                assert issubclass_safe(step_return_model, BaseModel), "Step return model must be a Pydantic model."
                # if step_id in auto_loops:
                #     step_return = List[step_return]
                if "return" in step_return_model.model_fields:
                    fields[return_param_name] = step_return_model.model_fields["return"]
                else:
                    # Generate a FieldInfo with the model.
                    fields[return_param_name] = Field(...)
                    fields[return_param_name].annotation = step_return_model
            # TODO If step_id.return.0 -> fetch the type of the list
            # TODO If step_id.return.abc -> fetch the type of the dict value / base model field
            # ? Perhaps we could create an analogous function to get_data_key for navigating the fields definitions
            else:
                fields[return_param_name] = Field(...)
                fields[return_param_name].annotation = Any
        
        return_model_name = f"Flow_{self.id}_return"
        return_model = create_model_from_fields(name=return_model_name, model_fields=fields)
        return return_model


    def return_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the flow return model."""
        if hasattr(self, '_return_model_schema'):
            return self._return_model_schema

        return self.return_model().model_json_schema()


    def _collect_outputs(self, data: dict[str, BaseData]) -> dict[str, Any]:
        """Aux method to collect the outputs from the data dictionary."""
        outputs = {}

        if self.is_agent:
            _, rev_dag = self.get_dags()
            rev_sources = list(get_sources(rev_dag))

            assert len(rev_sources) == 1, "Tool to LLM agent mode should have a single LLM as last step"
            last_llm_output = None
            last_llm_id = rev_sources[0]
            while last_llm_output is None:
                try: 
                    last_llm_output = get_data_key(data, f"{last_llm_id}.return")
                except DataKeyError:
                    last_llm_output = None
                    last_tools = list(rev_dag[last_llm_id])
                    assert len(last_tools) >= 1, "The last LLM of a tool to LLM agent must have at least one tool result."
                    last_llms = list(rev_dag[last_tools[0]])
                    assert len(last_llms) == 1, "Tool to LLM agent mode should have a single LLM as last step"
                    last_llm_id = last_llms[0]
            outputs["response"] = last_llm_output

        else:
            for output_name, data_key in self.outputs.items():
                # If we don't find the key, we set the output to None.
                # During the flow execution, a step could not be executed, thus its output will not be defined.
                try:
                    output_value = get_data_key(data, data_key)
                except DataKeyError:
                    output_value = None
                except Exception as e:
                    raise e
                outputs[output_name] = output_value

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

        # Hack for injecting a prompt into the LLM's memory as last message
        if step.is_llm:
            step_prompt = step_args.pop("prompt", None)
            if step_prompt is not None:
                # If the prompt comes from a previous LLM step, we need to transform it from assistant to user message
                if isinstance(step_prompt, Message):
                    message = Message(
                        role="user",
                        content=step_prompt.content
                    )
                # Message in dict format
                elif isinstance(step_prompt, dict) and "role" in step_prompt and "content" in step_prompt:
                    message = Message.validate(step_prompt)
                # For any other format, attempt to convert it to a message as content
                else:
                    message = Message.validate({
                        "role": "user",
                        "content": step_prompt
                    })

                if "memory" in step_args:
                    step_args["memory"].append(message)
                    # Limit the memory size. This is individual per llm step. Hence, we can have multiple LLMs pointing
                    # to the same memory key but have different memory window sizes.
                    if step._memory_window_size is not None:
                        step_args["memory"] = self._cut_memory(step_args["memory"], step._memory_window_size)
                else:
                    step_args["memory"] = [message]

        return step_args


    async def _run_step(
        self, 
        step_id: str, 
        data: dict[str, Any],
        run_id: str | None = None, # noqa: ARG002
        run_parent_id: str | None = None, # noqa: ARG002
        run_group_id: str | None = None, # noqa: ARG002
        dump_context: dict[str, Any] | None = None, # noqa: ARG002
    ) -> Any:
        """Executes a single step in the flow and returns its result.

        Resolves step arguments from the data dictionary, validates them against the step's parameter model,
        and executes the step. Handles both synchronous and asynchronous step execution.

        Args:
            step_id: Identifier of the step to execute
            data: Dictionary containing all flow data, including inputs and previous step results
            ...


        Returns:
            The step's result (direct value, generator, or awaited coroutine)

        Note:
            While this function is async, it doesn't await generator results - those are handled by the caller.
            Only direct coroutines from step.run() are awaited here.
        """
        step = self.steps[step_id]

        step_args = self._resolve_step_args(step_id, data)
        # TODO Think if we should return the actual inputs or all the validated params.
        step_args = step.params_model().model_validate(step_args)

        # If we're dealing with a regular sync function, we need to run it in an executor to 
        # avoid blocking the event loop.
        if not step.is_coroutine and not step.is_async_gen:
            loop = asyncio.get_running_loop()
            step_result = await loop.run_in_executor(None, lambda: step.run(
                run_id=run_id,
                run_parent_id=run_parent_id,
                run_group_id=run_group_id,
                dump_context=dump_context,
                **dict(step_args)
            ))
            if inspect.isgenerator(step_result):
                return step_args, sync_to_async_gen(step_result, loop)
            return step_args, step_result
        
        step_result = step.run(
            run_id=run_id,
            run_parent_id=run_parent_id,
            run_group_id=run_group_id,
            dump_context=dump_context,
            **dict(step_args)
        )

        if step.is_coroutine:
            step_result = await step_result
        
        return step_args, step_result


    async def run(
        self, 
        run_id: str | None = None,
        run_parent_id: str | None = None,
        run_group_id: str | None = None,
        dump_context: dict[str, Any] | None = None, # noqa: ARG002
        **kwargs: Any
    ) -> Any:
        """Executes the step's processing logic.
        
        Args:
            run_id: Identifier for the single run. 
                Handled separately from kwargs to avoid passing it downstream.
            run_parent_id: Identifier for the parent run.
                Handled separately from kwargs to avoid passing it downstream.
            run_group_id: Identifier for the group of runs.
                Handled separately from kwargs to avoid passing it downstream.
            dump_context: Context for dumping intermediate results. 
                Handled separately from kwargs to avoid passing it downstream.
            **kwargs: Additional keyword arguments required for step execution.
        
        Returns:
            Any: The step's processing result. Can be any object, a coroutine or an async generator.
        """
        t0 = int(time.time() * 1000)

        # Copy the data to avoid potential bugs with data being modified by reference.
        data = copy.deepcopy(self.data)

        data.update({k: DataValue(value=v) for k, v in kwargs.items()})

        # Load LLM memories.
        if self.state_saver is not None:
            last_snapshots = self.state_saver.get_last(
                n=1, 
                parent_id=run_parent_id,
                group_id=run_group_id,
                flow_path=self.path,
            )
            if len(last_snapshots) > 0:
                last_snapshot = last_snapshots[0]
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

        # Compute (or retrieve the pre-computed) DAGs that determine the flow's execution order.
        dag, rev_dag = self.get_dags()

        # Create events for each step to track their completion.
        tasks_completions = {step_id: asyncio.Event() for step_id in self.steps}

        # When working with agents, we'll need to keep track of the steps that don't need to be executed.
        skipped_steps = set()

        # Create tasks for the flow's sources.
        sources_ids = get_sources(dag)
        tasks = []
        start_times = {}
        for source_id in sources_ids:
            yield StepStartEvent(
                run_id=run_id,
                parent_step_id=self.id,
                step_id=source_id,
            )
            task = asyncio.create_task(self._run_step(
                step_id=source_id, 
                data=data,
                run_id=run_id,
                run_parent_id=run_parent_id,
                run_group_id=run_group_id,
                dump_context=dump_context,
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
                step_args = None
                step_result = None
                step_usage = {}
                try:
                    step_result = await completed_task

                    if step_id in async_gens:
                        async_gen_state = async_gens[step_id]
                        task = asyncio.create_task(async_gen_state.gen.__anext__())
                        task.step_id = step_id
                        tasks.append(task)

                        step_chunk = handle_event(event=step_result, async_gen_state=async_gen_state)
                        if step_chunk is not None:
                            yield StepChunkEvent(
                                run_id=run_id,
                                parent_step_id=self.id,
                                step_id=step_id,
                                step_chunk=step_chunk,
                            )
                        continue

                    step_args, step_result = step_result

                    if inspect.isasyncgen(step_result):
                        async_gens[step_id] = AsyncGenState(
                            gen=step_result,
                            inputs=step_args,
                        )
                        task = asyncio.create_task(step_result.__anext__())
                        task.step_id = step_id
                        tasks.append(task)
                        continue
                
                except StopAsyncIteration:
                    async_gen_state = async_gens[step_id]
                    step_args = async_gen_state.inputs
                    step_result = async_gen_state.results
                    step_usage = async_gen_state.usage

                    # Tool results are not marked as llm steps. These can include citations (e.g. perplexity).
                    # We add these citations directly formatting the markdown of the text result.
                    if hasattr(async_gen_state, "citations"):
                        step_citations = async_gen_state.citations
                        if isinstance(step_citations, list) and len(step_citations):
                            for i, citation in enumerate(step_citations, start=1):
                                step_result[-1]["text"] = step_result[-1]["text"].replace(f"[{i}]", f"[{i}]({citation})")

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

                # Store the result in the data dictionary. These will be used for data maps and 
                # collecting outputs from the overall flow execution.
                data[f"{step_id}.return"] = DataValue(value=step_result)
                
                # Mark this node as completed. This will unblock the tasks waiting for this node.
                tasks_completions[step_id].set()
                
                # Create tasks for the step successors.
                for successor_id in dag[step_id]:

                    ancestors = [
                        (ancestor_id, self.links[f"{ancestor_id}-{successor_id}"])
                        for ancestor_id in rev_dag[successor_id]
                    ]

                    if all(
                        tasks_completions[ancestor_id].is_set() 
                        for ancestor_id, _ in ancestors
                        if ancestor_id not in skipped_steps
                    ):
                        # Link conditions are only evaluated once all the ancestors are completed.
                        # If any of the conditions is true, we start the next step.
                        if any(ancestor_link.evaluate_condition(data) for _, ancestor_link in ancestors):
                            yield StepStartEvent(
                                run_id=run_id,
                                parent_step_id=self.id,
                                step_id=successor_id,
                            )
                            task = asyncio.create_task(self._run_step(
                                step_id=successor_id, 
                                data=data,
                                run_id=run_id,
                                run_parent_id=run_parent_id,
                                run_group_id=run_group_id,
                                dump_context=dump_context,
                            ))
                            task.step_id = successor_id
                            tasks.append(task)
                            start_times[successor_id] = int(time.time() * 1000)
                        else:
                            skipped_steps.add(successor_id)
                
                start_time = start_times[step_id]
                end_time = int(time.time() * 1000)
                elapsed_time = end_time - start_time

                yield StepOutputEvent(
                    run_id=run_id,
                    parent_step_id=self.id,
                    step_id=step_id,
                    step_args=step_args,
                    step_result=step_result,
                    step_usage=step_usage,
                    elapsed_time=elapsed_time,
                )

        # Grab the properties as defined in the return_model.
        output = self._collect_outputs(data)
        t1 = int(time.time() * 1000)
        elapsed_time = t1 - t0

        if self.state_saver is not None:
            snapshot = Snapshot(
                v="0.2.0",
                id=run_id,
                parent_id=run_parent_id,
                group_id=run_group_id,
                flow_path=self.path,
                input=kwargs,
                output=output,
                t0=t0,
                t1=t1,
                status="success", # TODO
                steps=[], # TODO
                data=copy.deepcopy(data),
            )
            self.state_saver.put(snapshot)
        
        yield FlowOutputEvent(
            run_id=run_id,
            step_id=self.id,
            outputs=output,
            elapsed_time=elapsed_time,
        )

    
    async def complete(
        self,
        run_id: str | None = None,
        run_parent_id: str | None = None,
        run_group_id: str | None = None,
        dump_context: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Flow.run() wrapper method that completes the flow execution.
        
        Args: 
            run_id: Identifier for the single run. 
                Handled separately from kwargs to avoid passing it downstream.
            run_parent_id: Identifier for the parent run.
                Handled separately from kwargs to avoid passing it downstream.
            run_group_id: Identifier for the group of runs.
                Handled separately from kwargs to avoid passing it downstream.
            dump_context: Context for dumping intermediate results. 
                Handled separately from kwargs to avoid passing it downstream.
            **kwargs: Additional keyword arguments required for step execution.
        
        Returns:
            dict[str, Any]: The flow's selected outputs.
        """
        async for event in self.run(
            run_id=run_id,
            run_parent_id=run_parent_id,
            run_group_id=run_group_id,
            dump_context=dump_context,
            **kwargs,
        ):
            if isinstance(event, FlowOutputEvent):
                return event.outputs


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

        def _update_path(step: BaseStep, path: str) -> None:
            if isinstance(step, Step):
                step.path = f"{path}.{step.id}"
                return
            elif isinstance(step, Flow):
                step.path = f"{path}.{step.id}"
                for sub_step in step.steps.values():
                    _update_path(sub_step, step.path)
            else:
                raise NotImplementedError(f"Invalid step type {type(step)}.")

        if isinstance(step, BaseStep):
            step = copy.deepcopy(step)
            step.id = id
            # This recursively updates recursively the steps paths so that we can uniquely 
            # identify each step, and subflow within the parent flow.
            _update_path(step, self.path)
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


    def set_output(
        self, 
        name: str, 
        data_key: str,
    ) -> "Flow":
        """Adds an output mapping to the flow.

        Careful this does not validate the data key or anything.
        If there's a mapping error or missing key it will return an error at runtime.

        This overrides the previous output mapping for the same name.

        Args:
            name: The name of the output to add.
            data_key: The data key to map the output to.
        
        Returns:
            The flow instance for method chaining.
        """
        self.outputs[name] = data_key
        return self


    def remove_output(
        self, 
        name: str,
    ) -> "Flow":
        """Removes an output mapping from the flow.

        Args:
            name: The name of the output to remove.

        Returns:
            The flow instance for method chaining.
        """
        self.outputs.pop(name, None)
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

        self._params_model = self.params_model()
        self._params_model_schema = self.params_model_schema()
        self._return_model = self.return_model()
        self._return_model_schema = self.return_model_schema()

        self.state_saver = state_saver

        # Mark the flow as compiled to prevent re-compiling when importing as a subflow.
        self._is_compiled = True

        return self


    def add_agent(
        self,
        id: str | None = None,
        tools: list[Callable | BaseStep | dict[str, Any]] = [],
        max_iter: int = 1,
        state_saver: BaseSaver | None = None,
        model: str | None = None,
        # We don't allow to pass a memory_id here. Agents will always use the same memory.
        # And they must have one, so they can use tool results with the corresponding tool calls.
        memory_window_size: int | None = None,
        **kwargs,
        # TODO Remove kwargs. Specify all the params.
    ) -> "Flow":
        """Adds an agent to the flow.

        An agent is an LLM-powered component that can reason about and interact with provided tools
        to accomplish tasks. The agent will:
        1. Receive an input prompt
        2. Decide whether to use available tools or respond directly
        3. If using tools, call them and incorporate their results
        4. Continue this process until reaching a final response

        Args:
            TODO        

        Returns:
            The flow instance for method chaining.

        Note:
            The agent will maintain conversation history between iterations, allowing it to
            build upon previous interactions with tools to reach its final response.
        """
        if id is None:
            id = "agent"

        # Create initial LLM step that receives the prompt
        entrypoint_llm_id = f"{id}_llm_0"
        memory_id = entrypoint_llm_id
        agent = (
            Flow()
            .add_llm(
                id=entrypoint_llm_id,
                model=model,
                memory_id=memory_id,
                memory_window_size=memory_window_size,
            )
            .set_data_map(f"{entrypoint_llm_id}.prompt", "prompt")
            .set_data_map(f"{entrypoint_llm_id}.system_prompt", "system_prompt")
        )

        if len(tools):
            # Default system prompt when processing tool results. Give the ability to the user to override this from the outside.
            tool_result_llm_system_prompt = """Please process the tool result given the call. 
            Do not remove references or citations from text. Leave them in markdown format."""
            kwargs_system_prompt = kwargs.pop("system_prompt", None)
            if kwargs_system_prompt is not None:
                tool_result_llm_system_prompt = kwargs_system_prompt

            prev_llm_id = entrypoint_llm_id
            # Create additional LLM steps for each iteration, allowing the agent to use tools multiple times
            for i in range(max_iter):
                iter_llm_id = f"{id}_llm_{i + 1}"
                agent.add_llm(
                    id=iter_llm_id,
                    memory_id=memory_id,
                    memory_window_size=memory_window_size,
                    system_prompt=tool_result_llm_system_prompt,
                )
                agent.set_data_map(f"{iter_llm_id}.model", f"{entrypoint_llm_id}.model", autolink=False)

                # For each tool, create a step and add links for tool use and tool result
                for tool_item in tools:
                    if isinstance(tool_item, dict):
                        tool = tool_item["tool"]
                        description = tool_item.get("description")
                    else:
                        tool = tool_item
                        description = None

                    if callable(tool):
                        tool_id = tool.__name__
                    elif isinstance(tool, BaseStep):
                        tool_id = tool.id

                    iter_tool_id = f"{id}_{tool_id}_{i}"

                    if iter_tool_id not in self.steps:
                        agent.add_step(iter_tool_id, tool)

                    agent.add_link(
                        step_id=prev_llm_id,
                        description=description,
                        next_step_id=iter_tool_id,
                        is_tool=True,
                    )

                    agent.add_link(
                        step_id=iter_tool_id,
                        next_step_id=iter_llm_id,
                        is_tool_result=True,
                    )
        
                prev_llm_id = iter_llm_id

        # Agents collect outputs differently. See _collect_outputs.
        agent.is_agent = True
        agent.compile(state_saver=state_saver)

        # Add the subflow to the parent flow
        self.add_step(id, agent)

        return self
