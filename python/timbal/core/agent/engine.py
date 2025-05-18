import asyncio
import inspect
import time
import traceback
from collections.abc import AsyncGenerator, Callable
from typing import Any

import structlog
from anthropic.types import (
    Message as AnthropicMessage,
)
from openai.types.chat import (
    ChatCompletion as OpenAICompletion,
)
from openai.types.chat import (
    ChatCompletionMessage as OpenAIMessage,
)
from pydantic import BaseModel, ConfigDict, TypeAdapter
from timbal.types.events.chunk import ChunkEvent
from uuid_extensions import uuid7

from ...errors import AgentError, EarlyExit
from ...state.context import RunContext, run_context_var
from ...state.savers.base import BaseSaver
from ...state.snapshot import Snapshot
from ...steps.llms.router import llm_router
from ...types.chat.content import ToolResultContent, ToolUseContent
from ...types.events import OutputEvent, StartEvent
from ...types.field import Field
from ...types.llms.usage import acc_usage
from ...types.message import Message
from ...types.models import dump
from ..base import BaseStep
from ..flow.engine import Flow
from ..step import Step
from ..stream import AsyncGenState, handle_event, sync_to_async_gen
from .types.llm_chunk import LLMChunk
from .types.llm_result import LLMResult
from .types.tool_result import ToolResult
from .types.tool import Tool

logger = structlog.get_logger("timbal.core.agent.engine")


class AgentParamsModel(BaseModel):
    """Fixed parameter model for Agents."""
    model_config = ConfigDict(extra="ignore")

    prompt: Message = Field(description="The prompt to use for the agent.")
    system_prompt: str = Field(
        default=None,
        description="The system prompt to use for the agent."
    )
    model: str = Field(
        default="gpt-4.1-nano",
        description="The model to use for the agent."
    )
    max_tokens: int = Field(
        default=None,
        description="The maximum number of tokens to generate."
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the output."
    )


# These are costly to generate. Pre-compute them and store them.
agent_params_model_schema = AgentParamsModel.model_json_schema()
message_model_schema = TypeAdapter(Message).json_schema()


class Agent(BaseStep):
    """A powerful LLM-powered agent that can execute complex tasks using tools.
    
    The Agent class implements an autonomous system that combines Large Language Models (LLMs) 
    with a set of tools to solve complex tasks. It operates in an iterative loop where:

    1. The LLM receives a prompt/query and available tools
    2. The LLM decides which tools to use and how to use them
    3. The tools are executed and their results are fed back to the LLM
    4. This continues until the task is complete or max iterations reached

    Key Features:
        - Multi-turn conversations with memory.
        - Parallel tool execution.
        - Support for both sync and async tools.
        - Automatic state persistence (optional).
        - Usage tracking and detailed execution traces.
        - Compatible with multiple LLM providers (OpenAI, Anthropic).

    Attributes:
        id (str): Unique identifier for the agent, defaults to "agent".
                  This is useful when adding the agent as a substep of a flow or as a tool of another agent.
        path (str): Hierarchical path identifier for nested flows/agents.
        metadata (dict): Custom metadata for the agent instance.
        tools (list): Available tools, which can be:
            - Callable functions (automatically wrapped).
            - BaseStep instances (direct tool implementations).
            - Dicts with {"tool": callable, "description": str} for custom tool configs.
        max_iter (int): Maximum number of tool-use iterations, defaults to 10.
        state_saver (BaseSaver): Optional component for persisting conversation state.
        **kwargs (dict): Configuration for the underlying LLM (model, system_prompt, temperature, etc.).

    Example:
        ```python
        agent = Agent(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            tools=[
                get_datetime,
                {
                    "runnable": search,
                    "description": "Search the internet.",
                }
            ]
        )
        ```
    """

    # TODO before tools callbacks. after tools callbacks.
    def __init__(
        self,
        id: str = "agent",
        path: str | None = None,
        metadata: dict[str, Any] = {},
        tools: list[Callable | BaseStep | dict[str, Any] | Tool] = [],
        max_iter: int = 10,
        state_saver: BaseSaver | None = None,
        # ? Should these be RunnableLike
        before_agent_callback: Callable | None = None,
        after_agent_callback: Callable | None = None,
        **kwargs: Any, # These are the LLM specific kwargs.
    ) -> None:
        if path is None:
            path = id
        super().__init__(id=id, path=path, metadata=metadata)

        self.state_saver = state_saver
        if self.state_saver is not None:
            self._is_state_saver_get_async = inspect.iscoroutinefunction(self.state_saver.get_last)
            self._is_state_saver_put_async = inspect.iscoroutinefunction(self.state_saver.put)

        self._load_tools(tools)
        self.max_iter = max_iter
        # TODO Validate these.
        self.before_agent_callback = before_agent_callback
        self.after_agent_callback = after_agent_callback

        self.llm_kwargs = kwargs

        # ? These params are used to enable passing this as a substep of a flow
        # ? (might remove them in the future).
        self.is_llm = False
        self.is_coroutine = False
        self.is_async_gen = True


    def prefix_path(self, prefix: str) -> None:
        """Prefix the step's path with a given path."""
        self.path = f"{prefix}.{self.id}"
        for tool in self.tools:
            tool.prefix_path(self.path)


    def _load_tools(
        self, 
        tools: list[Callable | BaseStep | dict[str, Any] | Tool],
    ) -> None:
        """Store the tools as BaseStep instances.
        This enables us to automatically generate params models and schemas for the tools using pydantic.
        """
        self.tools = []
        self.tools_lookup = {}

        for i, tool_config in enumerate(tools):
            # Ensure we have a Tool instance. It's better to rely on pydantic for validation.
            if isinstance(tool_config, Tool):
                tool = tool_config
            else:
                if not isinstance(tool_config, dict):
                    tool = Tool(runnable=tool_config)
                else:
                    tool = Tool(**tool_config)
                
            tool_step = tool.runnable
            if callable(tool_step):
                tool_step = Step(
                    id=tool_step.__name__,
                    handler_fn=tool_step,
                )

            if tool_step.id in self.tools_lookup:
                raise ValueError(f"Cannot add tool {tool_step.id} twice.")

            # We can use the step instance to store any additional properties we need.
            if tool.description is not None:
                tool_step.tool_description = tool.description
            tool_step.tool_force_exit = tool.force_exit
            tool_step.tool_params_mode = tool.params_mode
            tool_step.tool_include_params = tool.include_params
            tool_step.tool_exclude_params = tool.exclude_params

            tool_step.prefix_path(self.path)

            self.tools.append(tool_step)
            self.tools_lookup[tool_step.id] = i


    def params_model(self) -> BaseModel:
        """Returns the Pydantic model defining the expected parameters for this step."""
        return AgentParamsModel
        

    def params_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the step's parameter model."""
        return agent_params_model_schema


    def to_openai_tool(self) -> dict[str, Any]:
        """Overwrite the BaseStep.to_openai_tool method.
        For the moment we just expose the input message.
        """
        tool_description = ""
        if hasattr(self, "tool_description"):
            tool_description = self.tool_description or ""

        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {
                        "type": "string",
                        "description": "The input message to send to the agent.",
                    }},
                    "required": ["prompt"],
                }
            }
        }

    
    def to_anthropic_tool(self) -> dict[str, Any]:
        """Overwrite the BaseStep.to_anthropic_tool method.
        For the moment we just expose the input message.
        """
        tool_description = ""
        if hasattr(self, "tool_description"):
            tool_description = self.tool_description or ""

        return {
            "name": self.id,
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": {"prompt": {
                    "type": "string",
                    "description": "The input message to send to the agent.",
                }},
                "required": ["prompt"],
            }
        }
    

    def return_model(self) -> Any:
        """Returns the expected return type for this step."""
        return Message


    def return_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the step's return value model."""
        return message_model_schema


    async def _run_llm(
        self,
        messages: list[Message],
        tools: list[BaseStep],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Agent LLM execution wrapper.

        This method handles the core LLM interaction, including:
            - Model routing
            - Message formatting according to the LLM sdk being used.
            - Tool preparation according to the LLM sdk being used.
            - Handles both streaming and non-streaming calls.
            - Accumulates token usage for billing/monitoring.
            - Preserves full request/response context for tracing and debugging.
            - Non-recoverable errors are propagated up for agent error handling.

        Args:
            messages (list[Message]): The list of messages to send to the LLM.
            tools (list[BaseStep]): The list of tools to use.
            **kwargs: Additional keyword arguments for the LLM.

        Returns:
            LLMResult
        """
        t0 = int(time.time() * 1000)

        llm_error = None
        llm_message = None
        llm_usage = {}
        tools_dump = []
        try:
            async_gen_state = AsyncGenState()

            llm_sdk = None
            llm_output = await llm_router(
                messages=messages,
                tools=tools,
                **kwargs,
            )

            # If tool_output is an async generator, collect it.
            if inspect.isasyncgen(llm_output):
                async_gen_state = AsyncGenState()
                async for llm_output_chunk in llm_output:
                    llm_output_chunk = handle_event(llm_output_chunk, async_gen_state)
                    if llm_output_chunk:
                        yield LLMChunk(output=llm_output_chunk)
                llm_usage = async_gen_state.usage
                llm_message = Message.validate({
                    "role": "assistant",
                    "content": async_gen_state.collect(),
                })
                llm_sdk = async_gen_state.events_source
            else:
                llm_usage = acc_usage(
                    acc={},
                    model=kwargs.get("model"),
                    llm_output=llm_output,
                )
                if isinstance(llm_output, (OpenAICompletion, OpenAIMessage)):
                    llm_sdk = "openai"
                elif isinstance(llm_output, AnthropicMessage):
                    llm_sdk = "anthropic"
                llm_message = Message.validate(llm_output)

            # Properly format/dump the tools for tracing and debugging (this is what we send to the LLM).
            if llm_sdk == "openai":
                tools_dump = [tool.to_openai_tool() for tool in tools]
            elif llm_sdk == "anthropic":
                tools_dump = [tool.to_anthropic_tool() for tool in tools]
            else:
                raise ValueError("Unsupported LLM sdk!")

        except Exception as err:
            # We don't raise an error here. We want the agent to be able to recover from this.
            # e.g. the LLM is passing a badly formatted parameter to the tool.
            llm_error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }

        t1 = int(time.time() * 1000)

        llm_input = {
            "messages": messages,
            "tools": tools_dump,
            **kwargs,
        }

        yield LLMResult(
            input=llm_input,
            output=llm_message,
            error=llm_error,
            t0=t0,
            t1=t1,
            usage=llm_usage,
        )


    async def _run_tool(
        self, 
        tool: BaseStep,
        tool_input: dict[str, Any],
        tool_use_id: str,
        context: RunContext,
    ) -> ToolResult:
        """Agent tool execution wrapper.
        
        This method handles:
            - Input validation against the tool's params model.
            - Sync and async tools.
            - Collects and formats generator outputs (both sync and async).
            - All outputs are prepared for LLM consumption.
            - Tool execution is traced and can be monitored/logged.

        Args:
            tool (BaseStep): The tool to execute.
            tool_input (dict[str, Any]): The input to the tool.
            tool_use_id (str): The id of the tool use.
            context (RunContext): The run context. Needed when tools are nested agents or flows.

        Returns:
            ToolResult
        """
        t0 = int(time.time() * 1000)

        # There's no need to dump the tool input here. It will be generated by an LLM. No existing references to any objects in place.

        tool_output = None
        tool_error = None
        tool_usage = {}
        try:
            # Flow and Agent inputs are validated within their own .run() method. 
            # That is because we want to validate the first initial call to the parent flow as well.
            if not isinstance(tool, (Flow, Agent)): # noqa: UP038
                tool_input = dict(tool.params_model().model_validate(tool_input))

            # If we're dealing with a regular sync function, we need to run it in an executor to 
            # avoid blocking the event loop.
            if not tool.is_coroutine and not tool.is_async_gen:
                loop = asyncio.get_running_loop()
                tool_output = await loop.run_in_executor(None, lambda: tool.run(
                    context=context,
                    **tool_input
                ))
                # Convert to async generator.
                if inspect.isgenerator(tool_output):
                    tool_output = sync_to_async_gen(tool_output, loop)
            
            else:
                tool_output = tool.run(
                    context=context,
                    **tool_input
                )

                if tool.is_coroutine:
                    tool_output = await tool_output

            # If tool_output is an async generator, collect it.
            if inspect.isasyncgen(tool_output):
                async_gen_state = AsyncGenState()
                async for event in tool_output:
                    event = handle_event(event, async_gen_state)
                tool_output = async_gen_state.collect()
                tool_usage = async_gen_state.usage
                # TODO Collect usage when it's not a generator.

            # Handle the case where the tool already returns an LLM message. We need to modify it so it represents the result of a tool call.
            if not isinstance(tool_output, Message):
                tool_output = Message.validate({
                    "role": "user",
                    "content": tool_output,
                })
            
        except Exception as err:
            # We don't raise an error here. We want the agent to be able to recover from this.
            # e.g. the LLM is passing a badly formatted parameter to the tool.
            tool_error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }
            tool_output = Message.validate({
                "role": "user",
                "content": f"There was an error while running the tool. Error: {tool_error}",
            })

        t1 = int(time.time() * 1000)

        return ToolResult(
            id=tool_use_id,
            t0=t0,
            t1=t1,
            input=tool_input,
            output=tool_output,
            # We pass the error so that we're able to properly identify errors in the traces.
            error=tool_error,
            usage=tool_usage,
            force_exit=tool.tool_force_exit,
        )


    async def run(
        self, 
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the agent's main processing loop with tool usage, state management, and event streaming.
        
        This method implements the core agent execution cycle, including:
            - State loading and conversation history retrieval
            - Initial LLM execution with provided prompt
            - Tool execution based on LLM decisions
            - Result feeding back to LLM for next decisions
            - State persistence and usage tracking
            - Event streaming for monitoring/logging

        Args:
            context (RunContext | None): Execution context containing:
                - Run identifiers (auto-generated if None)
                - Parent context for nested executions
                - Session tracking for state persistence
                - Additional execution metadata
            **kwargs: Agent input parameters.

        Yields:
            Event objects in sequence:
                StartEvent:
                    - Marks beginning of agent/tool executions
                    - Contains run identifiers and paths
                OutputEvent:
                    - Contains execution results or errors
                    - Includes usage statistics
                    - Marks completion of execution steps

        State Management:
            - Loads previous conversation if state_saver configured
            - Maintains message history across iterations
            - Persists state after each significant step
            - Handles state loading/saving errors gracefully

        Error Handling:
            - Non-recoverable LLM errors propagate up
            - Tool errors are captured and fed back to LLM
            - State persistence errors are logged but non-fatal
            - Input validation errors raise AgentError

        Usage Tracking:
            - Accumulates token usage across all LLM calls
            - Tracks tool-specific resource usage
            - Maintains execution timing information
            - Preserves full execution traces

        Notes:
            - Implements parallel tool execution when possible
            - Respects max_iter limit for tool usage
            - Automatically formats all messages for LLM consumption
            - Provides detailed tracing for debugging/monitoring
            - Can be used as both standalone agent or subagent
            - Supports continuation of previous conversations
            - Handles both streaming and non-streaming LLMs
        """
        t0 = int(time.time() * 1000)

        if context is None:
            context = RunContext(id=uuid7(as_type="str"))
        elif context.id is None:
            context.id = uuid7(as_type="str")

        # Set the run context for the duration of the agent run.
        run_context_var.set(context)

        # Copy the input as is, so we save the traces without validated data and defaults.
        agent_input = dump(kwargs, context=context)

        # Load the memory.
        # Always do this first to ensure even if validation fails we can carry the memory to the next run.
        # Add this to the context. All modifications of messages will affect by reference the context data.
        messages = []
        context.data["memory"] = messages
        if self.state_saver is not None and context.parent_id is not None:
            try:
                if self._is_state_saver_get_async:
                    last_snapshot = await self.state_saver.get_last(path=self.path, context=context)
                else:
                    last_snapshot = self.state_saver.get_last(path=self.path, context=context)
            except Exception as err:
                logger.error("get_memory_error", err=err)
                last_snapshot = None
            
            # TODO Window sizes.
            # Ensure all the messages in the memory are actual Message instances.
            # (when loading from InMemorySaver, this will be already true)
            if last_snapshot is not None and "memory" in last_snapshot.data:
                messages.extend([
                    Message.validate(message) 
                    for message in last_snapshot.data["memory"].resolve()
                ])

        # Add all input kwargs to the run context data.
        # This way we can access them from any part of the agent run.
        for k, v in kwargs.items():
            context.data[k] = v

        agent_start_event = StartEvent(
            run_id=context.id,
            path=self.path,
            status_text="Starting...",
        )

        logger.info("start_event", start_event=agent_start_event)
        yield agent_start_event

        # Aggregated traces and usage for the entire run.
        run_steps = {}
        run_usage = {}

        try:
            # TODO Review this. Will it make sense to pass the kwargs as well (with before agent callback)
            if self.before_agent_callback is not None:
                self.before_agent_callback(context)

            has_prompt = False
            if "prompt" in kwargs:
                # We pre-validate the prompt field as a message. Frontend expects this to be a Message instance.
                kwargs["prompt"] = Message.validate(kwargs["prompt"])
                agent_input = dump(kwargs, context=context)
                has_prompt = True
            elif len(messages):
                # Hack to pass the validation.
                # TODO We should handle this better. Perhaps exposing a json schema and validating another model.
                kwargs["prompt"] = Message.validate("hack")
            else:
                raise ValueError("No prompt or message history found!")

            kwargs = {**self.llm_kwargs, **kwargs}
            kwargs = dict(self.params_model().model_validate(kwargs))

            # Add the prompt to the messages. The LLM router expects everything inside the messages list.
            prompt = kwargs.pop("prompt")
            if has_prompt:
                messages.append(prompt)

        except EarlyExit:
            return

        except Exception as err:
            error = {
                "type": type(err).__name__,
                "message": str(err),
                "traceback": traceback.format_exc(),
            }

            if self.state_saver is not None:
                t1 = int(time.time() * 1000)
                snapshot = Snapshot(
                    v="0.2.0",
                    id=context.id,
                    parent_id=context.parent_id,
                    path=self.path,
                    input=agent_input,
                    output=None,
                    error=error,
                    t0=t0,
                    t1=t1,
                    data=context.data.as_dict(),
                )
                
                # We don't want to cancel the execution if this errors. 
                try:
                    if self._is_state_saver_put_async:
                        await self.state_saver.put(snapshot=snapshot, context=context)
                    else:
                        self.state_saver.put(snapshot=snapshot, context=context)
                except Exception as err:
                    logger.error("put_memory_error", err=err)

            raise AgentError(error) from None

        # Run an llm first (that is, the handler_fn for the llm gateway step).
        llm_i_path = f"{self.path}.llm-0"
        llm_start_event = StartEvent(
            run_id=context.id,
            path=llm_i_path,
            status_text="Thinking...",
        )

        logger.info("start_event", start_event=llm_start_event)
        yield llm_start_event

        async for llm_output in self._run_llm(
            messages=messages,
            tools=self.tools,
            **kwargs,
        ):
            if isinstance(llm_output, LLMResult):
                llm_result = llm_output
            else:
                llm_chunk = llm_output

                llm_chunk_event = ChunkEvent(
                    run_id=context.id,
                    path=llm_i_path,
                    chunk=llm_chunk.output,
                )

                logger.info("chunk_event", chunk_event=llm_chunk_event)
                yield llm_chunk_event

                # # If the LLM is returning a stream, that indicates it's not going to use a tool.
                # # We're safe streaming the chunks as the final agent response.
                # agent_chunk_event = ChunkEvent(
                #     run_id=context.id,
                #     path=self.path,
                #     chunk=llm_chunk.output,
                # )

                # logger.info("chunk_event", chunk_event=agent_chunk_event)
                # yield agent_chunk_event

        llm_output_event = OutputEvent(
            run_id=context.id,
            path=llm_i_path,
            input=llm_result.input,
            output=llm_result.output,
            error=llm_result.error,
            t0=llm_result.t0,
            t1=llm_result.t1,
            usage=llm_result.usage,
        )

        logger.info("output_event", output_event=llm_output_event)
        yield llm_output_event

        # Store the trace of the LLM step.
        run_steps[llm_i_path] = dump(llm_output_event, context=context)

        # Aggregate the usage of the LLM step.
        for k, v in llm_result.usage.items():
            current_kv = run_usage.get(k, 0)
            run_usage[k] = current_kv + v

        # An LLM error is non-recoverable for the agent (after retries and all).
        # We raise the error upwards so others can catch it if this is a subagent.
        if llm_result.error is not None:
            if self.state_saver is not None:
                t1 = int(time.time() * 1000)
                snapshot = Snapshot(
                    v="0.2.0",
                    id=context.id,
                    parent_id=context.parent_id,
                    path=self.path,
                    input=agent_input,
                    output=None,
                    error=llm_result.error,
                    t0=t0,
                    t1=t1,
                    data=context.data.as_dict(),
                    steps=run_steps,
                    usage=run_usage,
                )
                
                # We don't want to cancel the execution if this errors. 
                try:
                    if self._is_state_saver_put_async:
                        await self.state_saver.put(snapshot=snapshot, context=context)
                    else:
                        self.state_saver.put(snapshot=snapshot, context=context)
                except Exception as err:
                    logger.error("put_memory_error", err=err)

            raise AgentError(llm_result.error) from None

        last_message = llm_result.output
        messages.append(last_message)

        # Check if we have tool calls in the response.
        tool_calls = [
            content for content in last_message.content 
            if isinstance(content, ToolUseContent)
        ]

        i = 0
        while tool_calls:
            i += 1

            # Run the tools in parallel.
            tool_tasks = []
            for tool_call in tool_calls:
                # Run some assertions. This should never happen when this function is called internally.
                assert tool_call.name in self.tools_lookup, f"Tool {tool_call.name} not found."
                tool_idx = self.tools_lookup[tool_call.name]
                assert len(self.tools) > tool_idx, f"Tool {tool_call.name} not found at index {tool_idx}."
                tool = self.tools[tool_idx]

                tool_start_event = StartEvent(
                    run_id=context.id,
                    path=f"{tool.path}-{tool_call.id}",
                    # TODO Review this one.
                    status_text=f"Running tool: {tool.path}...",
                )

                logger.info("start_event", start_event=tool_start_event)
                yield tool_start_event

                tool_task = asyncio.create_task(
                    self._run_tool(
                        tool=tool,
                        tool_input=tool_call.input,
                        tool_use_id=tool_call.id,
                        context=context,
                    ),
                    name=tool_call.id,
                )
                tool_tasks.append(tool_task)

            tool_exit = False

            # Await for tool completions.
            for tool_task in asyncio.as_completed(tool_tasks): # ? We could use this for timeouts.
                tool_result = await tool_task
                tool_result_message = Message.validate({
                    "role": "user",
                    "content": ToolResultContent(
                        id=tool_result.id,
                        content=tool_result.output.content,
                    ) 
                })
                messages.append(tool_result_message)
                
                tool_output_event = OutputEvent(
                    run_id=context.id,
                    path=f"{tool.path}-{tool_call.id}",
                    input=tool_result.input,
                    output=tool_result.output,
                    error=None,
                    t0=tool_result.t0,
                    t1=tool_result.t1,
                    usage=tool_result.usage,
                )

                logger.info("output_event", output_event=tool_output_event)
                yield tool_output_event

                # Store the trace of the LLM step.
                run_steps[f"{tool.path}-{tool_call.id}"] = dump(tool_output_event, context=context)

                # Aggregate the usage of the tool step.
                for k, v in tool_result.usage.items():
                    current_kv = run_usage.get(k, 0)
                    run_usage[k] = current_kv + v

                if tool_result.force_exit:
                    last_message = tool_result.output
                    tool_exit = True

            if tool_exit:
                break

            # Run the llm again.
            llm_i_path = f"{self.path}.llm-{i}"

            llm_start_event = StartEvent(
                run_id=context.id,
                path=llm_i_path,
                status_text="Thinking...",
            )

            logger.info("start_event", start_event=llm_start_event)
            yield llm_start_event

            async for llm_output in self._run_llm(
                messages=messages,
                # We don't pass tools to the LLM so it can't choose to call them and perform another iteration.
                tools=self.tools if i < self.max_iter else [],
                **kwargs,
            ):
                if isinstance(llm_output, LLMResult):
                    llm_result = llm_output
                else:
                    llm_chunk = llm_output

                    llm_chunk_event = ChunkEvent(
                        run_id=context.id,
                        path=llm_i_path,
                        chunk=llm_chunk.output,
                    )

                    logger.info("chunk_event", chunk_event=llm_chunk_event)
                    yield llm_chunk_event

                    # # If the LLM is returning a stream, that indicates it's not going to use a tool.
                    # # We're safe streaming the chunks as the final agent response.
                    # agent_chunk_event = ChunkEvent(
                    #     run_id=context.id,
                    #     path=self.path,
                    #     chunk=llm_chunk.output,
                    # )

                    # logger.info("chunk_event", chunk_event=agent_chunk_event)
                    # yield agent_chunk_event

            llm_output_event = OutputEvent(
                run_id=context.id,
                path=llm_i_path,
                input=llm_result.input,
                output=llm_result.output,
                error=llm_result.error,
                t0=llm_result.t0,
                t1=llm_result.t1,
                usage=llm_result.usage,
            )

            logger.info("output_event", output_event=llm_output_event)
            yield llm_output_event

            # Store the trace of the LLM step.
            run_steps[llm_i_path] = dump(llm_output_event, context=context)

            # Aggregate the usage of the LLM step.
            for k, v in llm_result.usage.items():
                current_kv = run_usage.get(k, 0)
                run_usage[k] = current_kv + v

            # An LLM error is non-recoverable for the agent (after retries and all).
            # We raise the error upwards so others can catch it if this is a subagent.
            if llm_result.error is not None:
                if self.state_saver is not None:
                    t1 = int(time.time() * 1000)
                    snapshot = Snapshot(
                        v="0.2.0",
                        id=context.id,
                        parent_id=context.parent_id,
                        path=self.path,
                        input=agent_input,
                        output=None,
                        error=llm_result.error,
                        t0=t0,
                        t1=t1,
                        data=context.data.as_dict(),
                        steps=run_steps,
                        usage=run_usage,
                    )
                    
                    # We don't want to cancel the execution if this errors. 
                    try:
                        if self._is_state_saver_put_async:
                            await self.state_saver.put(snapshot=snapshot, context=context)
                        else:
                            self.state_saver.put(snapshot=snapshot, context=context)
                    except Exception as err:
                        logger.error("put_memory_error", err=err)

                raise AgentError(llm_result.error)

            last_message = llm_result.output
            messages.append(last_message)

            # Check if we have tool calls in the response.
            tool_calls = [
                content for content in last_message.content 
                if isinstance(content, ToolUseContent)
            ]

        t1 = int(time.time() * 1000)

        if self.state_saver is not None:
            snapshot = Snapshot(
                v="0.2.0",
                id=context.id,
                parent_id=context.parent_id,
                path=self.path,
                input=agent_input,
                output=last_message,
                error=None,
                t0=t0,
                t1=t1,
                data=context.data.as_dict(),
                steps=run_steps,
                usage=run_usage,
            )

            # We don't want to cancel the execution if this errors. 
            try:
                if self._is_state_saver_put_async:
                    await self.state_saver.put(snapshot=snapshot, context=context)
                else:
                    self.state_saver.put(snapshot=snapshot, context=context)
            except Exception as err:
                logger.error("put_memory_error", err=err)

        agent_output_event = OutputEvent(
            run_id=context.id,
            path=self.path,
            input=agent_input,
            output=last_message,
            error=None, # If it reaches this point, the agent has completed successfully.
            t0=t0,
            t1=t1,
            usage=run_usage,
        )

        logger.info("output_event", output_event=agent_output_event)
        yield agent_output_event

        # TODO Make this async.
        if self.after_agent_callback is not None:
            try:
                self.after_agent_callback(context)
            except Exception as err:
                logger.error("after_agent_callback_error", err=err)

    
    async def complete(
        self,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> OutputEvent:
        """run() wrapper method that completes the flow execution.
        
        Args: 
            context: RunContext
            **kwargs: Additional keyword arguments required for step execution.
        
        Returns:
            OutputEvent: The agent's selected outputs.
        """
        agent_output_event = None
        async for event in self.run(context=context, **kwargs):
            if isinstance(event, OutputEvent) and event.path == self.path:
                agent_output_event = event
        return agent_output_event
