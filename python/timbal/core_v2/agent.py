import asyncio
from collections.abc import AsyncGenerator, Callable
from functools import cached_property
from typing import Any, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    computed_field,
)

from ..types.chat.content import ToolResultContent, ToolUseContent
from ..types.events import OutputEvent
from ..types.message import Message
from .handlers import llm_router
from .runnable import Runnable
from .tool import Tool

ToolLike = Runnable | dict[str, Any] | Callable[..., Any]


# TODO Add more params here
class AgentParams(BaseModel):
    """"""
    model_config = ConfigDict(extra="ignore")

    prompt: Message = Field(
        ...,
        description="Input message to send to the agent.",
    )


class Agent(Runnable):
    """"""

    model: str
    """"""
    instructions: str | None = None
    """"""
    tools: list[SkipValidation[ToolLike]] = []
    """"""
    max_iter: int = 10
    """"""

    _llm: Tool = PrivateAttr()
    """"""
    _tools_by_name: dict[str, Tool] = PrivateAttr()
    """"""


    # NOTE: No need to add @override since pydantic doesn't have `model_post_init` as an abstract method.
    def model_post_init(self, __context: Any) -> None:
        """"""
        self._path = self.name
        
        self._llm = Tool(
            name="llm",
            handler=llm_router,
        )
        self._llm.nest(self._path)

        # Normalized tools and tools_by_name will hold references to the same Tool instances.
        # Modifying either will modify the other.
        normalized_tools = []
        tools_by_name = {}
        for tool in self.tools:
            if not isinstance(tool, Runnable):
                if isinstance(tool, dict):
                    tool = Tool(**tool)
                else:
                    # This will error if tool is not a proper callable
                    tool = Tool(handler=tool)
            
            if tool.name in tools_by_name:
                raise ValueError(
                    f"Tool {tool.name} already exists. "
                    "You can only add a tool once."
                )
            
            tool.nest(self._path)
            normalized_tools.append(tool)
            tools_by_name[tool.name] = tool
        
        self.tools = normalized_tools
        self._tools_by_name = tools_by_name 

        # The handler for the agent is always an async generator
        self._is_coroutine = False
        self._is_gen = False
        self._is_async_gen = True


    @override
    def nest(self, parent_path: str) -> None:
        """"""
        self._path = f"{parent_path}.{self.name}"
        self._llm.nest(self._path)
        for tool in self.tools:
            tool.nest(self._path)
    

    @override 
    @computed_field
    @cached_property
    def params_model(self) -> BaseModel:
        """"""
        return AgentParams


    @override 
    @computed_field
    @cached_property
    def return_model(self) -> Any:
        """"""
        return Message

    
    async def _enqueue_tool_events(self, tool_call: ToolUseContent, queue: asyncio.Queue) -> None:
        """"""
        tool = self._tools_by_name[tool_call.name]
        async for event in tool(**tool_call.input):
            await queue.put((tool_call, event))

        # Signal completion with a sentinel value (None)
        await queue.put((tool_call, None))


    async def _multiplex_tools(self, tool_calls: list[ToolUseContent]) -> AsyncGenerator[Any, None]:
        """"""
        queue = asyncio.Queue()
        tasks = [
            asyncio.create_task(self._enqueue_tool_events(tc, queue))
            for tc in tool_calls
        ]

        remaining = len(tasks)
        while remaining > 0:
            tool_call, event = await queue.get()
            if event is None:
                remaining -= 1
            else:
                yield tool_call, event
        
        await asyncio.gather(*tasks)


    async def handler(
        self, 
        **kwargs: Any,
    ) -> Any:
        """"""
        
        # TODO Think how to refactor memory from previous version
        messages = [kwargs.pop("prompt")]

        while True:
            async for event in self._llm(
                model=self.model,
                messages=messages,
                tools=self.tools,
                **kwargs,
            ):
                if isinstance(event, OutputEvent):
                    assert isinstance(event.output, Message), f"Expected event.output to be a Message, got {type(event.output)}"
                    messages.append(event.output)
                yield event

            tool_calls = [
                content for content in messages[-1].content
                if isinstance(content, ToolUseContent)
            ]
            if not tool_calls:
                break

            async for tool_call, event in self._multiplex_tools(tool_calls):
                # We'll receive a bunch of upwards streaming events from nested agents
                # We only need to process the ones that are immediate children of this very agent
                if isinstance(event, OutputEvent) and event.path.count(".") == self._path.count(".") + 1:
                    # We need to convert the tool output to a tool result message, for the next LLM to consume and match 
                    # ? Can we optimize this double validate?
                    event_output = event.output
                    if not isinstance(event_output, Message):
                        event_output = Message.validate({
                            "role": "user",
                            "content": str(event_output),
                        })
                    message = Message(
                        role="tool",
                        content=[ToolResultContent(
                            id=tool_call.id,
                            content=event_output.content,
                        )]
                    )
                    messages.append(message)

                yield event
            