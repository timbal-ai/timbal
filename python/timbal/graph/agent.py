import asyncio
import copy
import inspect
import time
import traceback
from collections.abc import Callable
from typing import Any

import structlog
from pydantic import BaseModel, TypeAdapter
from uuid_extensions import uuid7

from ..graph.stream import AsyncGenState, handle_event, sync_to_async_gen
from ..state.data import DataValue
from ..graph.step import Step
from ..graph.flow import Flow
from ..state.data import BaseData
from ..state.context import RunContext
from ..state.savers.base import BaseSaver
from ..state.snapshot import Snapshot
from ..types.models import dump
from ..types.field import Field
from ..types.message import Message
from ..types.chat.content import TextContent, ToolResultContent, ToolUseContent
from .base import BaseStep

from ..steps.llms.router import llm_router

from ..types.events import StartEvent, OutputEvent

logger = structlog.get_logger("timbal.graph.agent")


class LLMResult(BaseModel):
    input: dict[str, Any]
    output: Message
    t0: int 
    t1: int
    usage: dict[str, int] = {}


class ToolResult(BaseModel):
    id: str # This will be used to create the ToolResultContent (with matching ToolUseContent).
    input: dict[str, Any] # This will be used for tracing and debugging.
    output: str # ! We always stringify any result. Even errors. We want the agents to be able to correct their actions.
    t0: int
    t1: int
    usage: dict[str, int] = {}

    def to_content(self) -> ToolResultContent:
        return ToolResultContent(
            id=self.id,
            content=[TextContent(text=self.output)],
        )


class AgentParamsModel(BaseModel):
    prompt: str = Field(description="The prompt to use for the agent.")
    system_prompt: str | None = Field(
        default=None,
        description="The system prompt to use for the agent."
    )
    model: str | None = Field(
        default="gpt-4o-mini",
        description="The model to use for the agent."
    )


agent_params_model_schema = AgentParamsModel.model_json_schema()
message_model_schema = TypeAdapter(Message).json_schema()


class Agent(BaseStep):
    """Subclass of Flow that implements an LLM agent with tool-use capabilities.
    
    An Agent is a specialized Flow that creates a chain of LLM steps and tools, allowing
    the LLM to use tools multiple times in a conversation. Each LLM step can call any
    available tool, and the results are fed back to the next LLM step.

    Attributes:
        id (str): Unique identifier for the agent, defaults to "agent"
        tools (list): List of callable functions, BaseSteps, or dicts with tool configs
        max_iter (int): Maximum number of tool use iterations allowed, defaults to 1
        **kwargs: Additional keyword arguments for the LLMs

    Example:
        ```python
        agent = Agent(
            tools=[search_tool, calculator],
            max_iter=3,
            system_prompt="You are a helpful assistant"
        )
        ```
    """

    def __init__(
        self,
        id: str = "agent",
        path: str | None = None,
        metadata: dict[str, Any] = {},
        tools: list[Callable | BaseStep | dict[str, Any]] = [],
        max_iter: int = 10,
        state_saver: BaseSaver | None = None,
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
        tools: list[Callable | BaseStep | dict[str, Any]]
    ) -> None:
        """"""
        self.tools = []
        self.tools_lookup = {}
        for i, tool_config in enumerate(tools):
            if isinstance(tool_config, dict):
                if "tool" not in tool_config:
                    raise ValueError("You must specify a 'tool' key when passinga tool as a dict.")
                tool = tool_config["tool"]
                tool_description = tool_config.get("description", None)
                # TODO Enable passing 'fixed' values for some params (or perhaps which params to expose).
            else: 
                tool = tool_config
                tool_description = None
                
            if callable(tool):
                tool = Step(
                    id=tool.__name__,
                    handler_fn=tool,
                )

            if not isinstance(tool, BaseStep):
                raise ValueError(f"Tool needs to be an instance of BaseStep or a callable.")

            if tool.id in self.tools_lookup:
                raise ValueError(f"Cannot add tool {tool.id} twice.")

            # TODO Think. We could enable passing multiple tool descriptions. Depending on the agent that's calling the tool.
            if tool_description is not None:
                tool.tool_description = tool_description

            tool.prefix_path(self.path)

            self.tools.append(tool)
            self.tools_lookup[tool.id] = i


    def params_model(self) -> BaseModel:
        """Returns the Pydantic model defining the expected parameters for this step."""
        return AgentParamsModel
        

    def params_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the step's parameter model."""
        return agent_params_model_schema
    

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
    ) -> Message:
        """"""
        t0 = int(time.time() * 1000)

        async_gen_state = AsyncGenState()

        async for chunk in llm_router(
            messages=messages,
            tools=tools,
            **kwargs,
        ):
            chunk = handle_event(chunk, async_gen_state)

        # Add the message to the memory.
        message = Message.validate({
            "role": "assistant",
            "content": async_gen_state.collect(),
        })

        t1 = int(time.time() * 1000)

        llm_input = {
            "messages": messages,
            "tools": tools, # TODO We might need to dump this (as openai or anthropic).
            **kwargs,
        }

        return LLMResult(
            input=llm_input,
            output=message,
            t0=t0,
            t1=t1,
            usage=async_gen_state.usage,
        )


    async def _run_tool(
        self, 
        tool: BaseStep,
        tool_input: dict[str, Any],
        tool_use_id: str,
        context: RunContext,
    ) -> ToolResult:
        """"""
        t0 = int(time.time() * 1000)

        # There's no need to dump the tool input here. It will be generated by an LLM. No existing references to any objects in place.

        try:
            # Flow and Agent inputs are validated within their own .run() method. 
            # That is because we want to validate the first initial call to the parent flow as well.
            if isinstance(tool, (Flow, Agent)):
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

            tool_output = str(tool_output)
            
        except Exception as e:
            tool_output = f"""There was an error while running the tool. Error: {e}"""

        t1 = int(time.time() * 1000)

        return ToolResult(
            id=tool_use_id,
            t0=t0,
            t1=t1,
            input=tool_input,
            output=tool_output,
            # TODO Think what to do about the (recoverable) error.
        )


    async def run(
        self, 
        context: RunContext | None = None, # noqa: ARG002
        **kwargs: Any,
    ) -> Any:
        """Executes the step's processing logic."""
        # TODO Find a way to encapsulate all this logic. I am imagining a class that implements the collect method, wraps the run method, etc. 
        if context is None:
            context = RunContext(id=uuid7(as_type="str"))
        elif context.id is None:
            context.id = uuid7(as_type="str")

        # This one is global to the agent.
        t0 = int(time.time() * 1000)
        yield StartEvent(
            run_id=context.id,
            path=self.path,
        )

        agent_input = dump(kwargs)

        # # Validate kwargs to store the validated inputs in the snapshot.
        # try:
        #     kwargs = {**self.llm_kwargs, **kwargs}
        #     kwargs = dict(self.params_model().model_validate(kwargs))
        # except Exception as e:
        #     error = {
        #         "type": type(e).__name__,
        #         "message": str(e),
        #         "traceback": traceback.format_exc(),
        #     }
        #     t1 = int(time.time() * 1000)

        #     if self.state_saver is not None:
        #         snapshot = Snapshot(
        #             v="0.2.0",
        #             id=context.id,
        #             parent_id=context.parent_id,
        #             path=self.path,
        #             input=kwargs,
        #             output=None,
        #             error=error,
        #             t0=t0,
        #             t1=t1,
        #             # We allow for a direct reference to the data dictionary, because we know for 
        #             # sure that the PUT operation will not modify the data dictionary in any way.
        #             data=self.data,
        #         )
                
        #         if self._is_state_saver_put_async:
        #             await self.state_saver.put(snapshot=snapshot, context=context)
        #         else:
        #             self.state_saver.put(snapshot=snapshot, context=context)

        #     raise FlowExecutionError(f"Error validating kwargs for step {self.path}.") from e

        # Load the memory.
        # The data stored in the state saver is just the passed messages.
        # TODO Window sizes.
        messages = []
        if self.state_saver is not None and context.parent_id is not None:
            try:
                if self._is_state_saver_get_async:
                    last_snapshot = await self.state_saver.get_last(path=self.path, context=context)
                else:
                    last_snapshot = self.state_saver.get_last(path=self.path, context=context)
            except Exception as err:
                logger.error("get_memory_error", err=err)
                last_snapshot = None
            
            # Ensure all the messages in the memory are actual Message instances.
            # (when loading from InMemorySaver, this will be already true)
            if last_snapshot is not None and "memory" in last_snapshot.data:
                messages = [
                    Message.validate(message) 
                    for message in last_snapshot.data["memory"].resolve()
                ]

        prompt = kwargs.pop("prompt") # ? 'prompt' is like the content of a message.
        messages.append(Message.validate({
            "role": "user", 
            "content": prompt,
        }))

        # Run an llm first (that is, the handler_fn for the llm gateway step).
        llm_i_path = f"{self.path}.llm-0"
        yield StartEvent(
            run_id=context.id,
            path=llm_i_path,
            # ? This one could grab properties from the tool to customize a little bit more the event.
        )

        llm_result = await self._run_llm(
            messages=messages,
            tools=self.tools,
        ) 
        last_message = llm_result.output
        messages.append(last_message)

        yield OutputEvent(
            run_id=context.id,
            path=llm_i_path,
            input=llm_result.input,
            output=last_message,
            error=None, # TODO We should try catch the LLM call.
            t0=llm_result.t0,
            t1=llm_result.t1,
            usage=llm_result.usage,
        )

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

                yield StartEvent(
                    run_id=context.id,
                    path=f"{tool.path}-{tool_call.id}",
                    # ? This one could grab properties from the tool to customize a little bit more the event.
                )

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

            # Await for tool completions.
            for tool_task in asyncio.as_completed(tool_tasks): # ? We could use this for timeouts.
                tool_result = await tool_task
                tool_result_message = Message.validate({
                    "role": "user",
                    "content": tool_result.to_content(),
                })
                messages.append(tool_result_message)
                
                # TODO Store this in a 'steps' dictionary.
                yield OutputEvent(
                    run_id=context.id,
                    path=f"{tool.path}-{tool_call.id}",
                    input=tool_result.input,
                    output=tool_result.output,
                    error=None,
                    t0=tool_result.t0,
                    t1=tool_result.t1,
                    usage=tool_result.usage,
                )

            # Run the llm again.
            llm_i_path = f"{self.path}.llm-{i}"
            yield StartEvent(
                run_id=context.id,
                path=llm_i_path,
                # ? This one could grab properties from the tool to customize a little bit more the event.
            )

            llm_result = await self._run_llm(
                messages=messages,
                # We don't pass tools to the LLM so it can't choose to call them and perform another iteration.
                tools=self.tools if i < self.max_iter else [],
                **kwargs,
            ) 
            last_message = llm_result.output
            messages.append(last_message)

            # TODO Store this in a 'steps' dictionary.
            yield OutputEvent(
                run_id=context.id,
                path=llm_i_path,
                input=llm_result.input,
                output=last_message,
                error=None, # TODO We should try catch the LLM call.
                t0=llm_result.t0,
                t1=llm_result.t1,
                usage=llm_result.usage,
            )

            # Check if we have tool calls in the response.
            tool_calls = [
                content for content in last_message.content 
                if isinstance(content, ToolUseContent)
            ]

        t1 = int(time.time() * 1000)

        if self.state_saver is not None:
            data = {"memory": DataValue(value=messages)}
            snapshot = Snapshot(
                v="0.2.0",
                id=context.id,
                parent_id=context.parent_id,
                path=self.path,
                input=agent_input,
                output=last_message,
                error=None, # TODO
                t0=t0,
                t1=t1,
                steps={}, # TODO
                # We allow for a direct reference to the data dictionary, because we know for 
                # sure that the PUT operation will not modify the data dictionary in any way.
                data=data,
                usage={}, # TODO Aggregate the usage from all the steps.
            )

            # We don't want to cancel the execution if this errors. 
            try:
                if self._is_state_saver_put_async:
                    await self.state_saver.put(snapshot=snapshot, context=context)
                else:
                    self.state_saver.put(snapshot=snapshot, context=context)
            except Exception as err:
                logger.error("put_memory_error", err=err)

        yield OutputEvent(
            run_id=context.id,
            path=self.path,
            input=agent_input,
            output=last_message,
            error=None, # TODO We should try catch the LLM call.
            t0=t0,
            t1=t1,
            usage={}, # TODO Aggregate the usage from all the steps.
        )

    
    async def complete(
        self,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> OutputEvent:
        """Flow.run() wrapper method that completes the flow execution.
        
        Args: 
            context: RunContext
            **kwargs: Additional keyword arguments required for step execution.
        
        Returns:
            dict[str, Any]: The flow's selected outputs.
        """
        async for event in self.run(context=context, **kwargs):
            if isinstance(event, OutputEvent) and event.path == self.path:
                return event
