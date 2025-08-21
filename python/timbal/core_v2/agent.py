import asyncio
from collections.abc import AsyncGenerator
from functools import cached_property
from typing import Any, override

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    computed_field,
)

from ..state import get_run_context
from ..types.chat.content import ToolResultContent, ToolUseContent
from ..types.events import OutputEvent
from ..types.message import Message
from .handlers import llm_router
from .runnable import Runnable, RunnableLike
from .tool import Tool

logger = structlog.get_logger("timbal.core_v2.agent")


# TODO Add more params here
class AgentParams(BaseModel):
    """Parameter model for Agent execution.
    
    Defines the input parameters that agents accept when called.
    Supports flexible system prompt composition through templates.
    """
    model_config = ConfigDict(extra="ignore")

    prompt: Message = Field(
        ...,
        description="Input message to send to the agent.",
    )
    system_context: str | None = Field(
        None,
        description=(
            "System prompt context. Can be used to provide additional context to the agent. "
            "You can inject the agent instructions using the {instructions} placeholder. "
            "This design enables:\n"
            "- Platform Flexibility: Send different prompt structures without changing agent code\n"
            "- Reusable Components: Agent instructions slot into any template\n"
            "- Context Adaptation: Same agent, different contexts, different structures\n"
            "- Clean Separation: Agent logic in code, presentation logic from platform\n\n"
            "If not provided, agent.instructions will be used directly as system prompt."
        ),
    )


class Agent(Runnable):
    """An Agent is a Runnable that orchestrates LLM interactions with tool calling.
    
    Agents implement an autonomous execution pattern where an LLM can:
    1. Receive a prompt and generate a response
    2. Decide to call available tools based on the context
    3. Process tool results and continue the conversation
    4. Repeat until no more tool calls are needed or max_iter is reached
    
    Agents support:
    - Multi-turn conversations with memory across iterations
    - Concurrent tool execution for efficiency
    - Flexible tool definition (functions, dicts, or Runnable objects)
    - Integration with multiple LLM providers via llm_router
    
    Example:
        def get_weather(location: str) -> str:
            return f"Weather in {location}: sunny"
            
        agent = Agent(
            name="weather_agent",
            model="claude-3-sonnet",
            tools=[get_weather]
        )
        
        result = await agent(prompt=Message(role="user", content="What's the weather in Paris?")).collect()
    """

    model: str
    """The LLM model identifier (e.g., 'claude-3-sonnet', 'gpt-4')."""
    instructions: str | None = None
    """Optional system instructions to provide context for the agent."""
    tools: list[SkipValidation[RunnableLike]] = []
    """List of tools available to the agent. Can be functions, dicts, or Runnable objects."""
    max_iter: int = 10
    """Maximum number of LLM->tool call iterations before stopping."""

    _llm: Tool = PrivateAttr()
    """Internal LLM tool instance for making model calls."""
    _tools_by_name: dict[str, Tool] = PrivateAttr()
    """Dictionary mapping tool names to Tool instances for fast lookup."""


    # NOTE: No need to add @override since pydantic doesn't have `model_post_init` as an abstract method.
    def model_post_init(self, __context: Any) -> None:
        """Initialize agent-specific attributes after Pydantic model creation.
        
        This method sets up the agent's internal tools and execution characteristics:
        1. Creates an internal LLM tool for model interactions
        2. Normalizes user-provided tools into Tool instances
        3. Sets up tool name mapping for fast lookup during execution
        4. Configures execution characteristics as an orchestrator
        """
        self._path = self.name

        # Make sure the system_context is not exposed a stool param when using the agent as a tool
        if self.exclude_params is None:
            self.exclude_params = ["system_context"]
        else:
            self.exclude_params.append("system_context")
        
        # Create internal LLM tool for model interactions
        self._llm = Tool(
            name="llm",
            handler=llm_router,
        )
        self._llm.nest(self._path)

        # Normalize tools: convert functions/dicts to Tool instances
        # tools and _tools_by_name will hold references to the same Tool instances
        normalized_tools = []
        tools_by_name = {}
        for tool in self.tools:
            # Convert non-Runnable objects to Tool instances
            if not isinstance(tool, Runnable):
                if isinstance(tool, dict):
                    tool = Tool(**tool)  # Create from dict config
                else:
                    # Assume it's a callable and wrap it
                    tool = Tool(handler=tool)
            
            if tool.name in tools_by_name:
                raise ValueError(f"Tool {tool.name} already exists. You can only add a tool once.")
            
            tool.nest(self._path)
            normalized_tools.append(tool)
            tools_by_name[tool.name] = tool
        
        self.tools = normalized_tools
        self._tools_by_name = tools_by_name 

        # Agents are always orchestrators with async generator handlers
        self._is_orchestrator = True
        self._is_coroutine = False
        self._is_gen = False
        self._is_async_gen = True


    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"
        # Update paths for internal LLM and all tools
        self._llm.nest(self._path)
        for tool in self.tools:
            tool.nest(self._path)
    

    @override 
    @computed_field
    @cached_property
    def params_model(self) -> BaseModel:
        """See base class."""
        return AgentParams


    @override 
    @computed_field
    @cached_property
    def return_model(self) -> Any:
        """See base class."""
        return Message

    
    def _build_system_prompt(self, **kwargs) -> str | None:
        """Build system prompt from template and instructions.
        
        This method implements the flexible system prompt composition pattern:
        1. If system_context template is provided, format it with instructions
        2. If no template but instructions exist, use instructions directly  
        3. If neither exist, return None (no system prompt)
        
        Args:
            **kwargs: Execution parameters, may contain system_context
            
        Returns:
            Formatted system prompt string or None
        """
        system_context = kwargs.pop("system_context", None)
        if system_context:
            return system_context.format(instructions=self.instructions or "")
        elif self.instructions:
            return self.instructions
        else:
            return None


    async def _resolve_memory(self) -> list[Message]:
        """Resolve conversation memory from parent agent context.
        
        When agents are nested (e.g., subagents called by parent agents),
        this method retrieves the conversation history from the parent's
        tracing data to maintain context across agent calls.
        
        Returns:
            List of Messages representing the conversation history,
            or empty list if no parent context exists
        """
        memory = []
        run_context = get_run_context()
        if run_context.parent_id:
            # Try to get tracing data from parent execution
            parent_tracing = await run_context.get_parent_tracing()
            if parent_tracing is None:
                logger.error("Parent tracing not found. Continuing without memory...", parent_id=run_context.parent_id, run_id=run_context.id)
            else:
                # Extract conversation history from parent's LLM calls
                # TODO: Handle multiple call_ids for this subagent
                llm_tracing = parent_tracing.get_path(self._llm._path)
                assert len(llm_tracing) >= 1, f"Agent trace does not have any records for path {self._llm._path}"
                # Get the most recent LLM interaction
                llm_input_messages = llm_tracing[-1].input.get("messages", []) # TODO Put an assertion in here
                llm_output_message = llm_tracing[-1].output
                # Reconstruct conversation: input messages + LLM response
                memory = [
                    *[Message.validate(m) for m in llm_input_messages], 
                    Message.validate(llm_output_message)
                ]
        return memory

    
    async def _enqueue_tool_events(self, tool_call: ToolUseContent, queue: asyncio.Queue) -> None:
        """Execute a single tool call and enqueue its events.
        
        This method runs a tool call asynchronously and puts all generated
        events into a shared queue for multiplexed processing. Used by
        _multiplex_tools to handle concurrent tool execution.
        
        Args:
            tool_call: The tool call content with name and input parameters
            queue: Async queue to put events into
        """
        tool = self._tools_by_name[tool_call.name]
        # Execute tool and enqueue all events
        async for event in tool(**tool_call.input):
            await queue.put((tool_call, event))
        # Signal completion with a sentinel value (None)
        await queue.put((tool_call, None))


    async def _multiplex_tools(self, tool_calls: list[ToolUseContent]) -> AsyncGenerator[Any, None]:
        """Execute multiple tool calls concurrently and multiplex their events.
        
        This method enables concurrent execution of multiple tool calls requested
        by the LLM, significantly improving performance when tools can run in parallel.
        Events from all tools are multiplexed and yielded as they become available.
        
        Args:
            tool_calls: List of tool calls to execute concurrently
            
        Yields:
            Tuples of (tool_call, event) as events are generated by tools
        """
        queue = asyncio.Queue()
        # Start all tool executions concurrently
        tasks = [asyncio.create_task(self._enqueue_tool_events(tc, queue)) for tc in tool_calls]
        # Process events as they arrive from any tool
        remaining = len(tasks)
        while remaining > 0:
            tool_call, event = await queue.get()
            if event is None:
                # Sentinel value indicates a tool completed
                remaining -= 1
            else:
                # Yield event along with which tool call generated it
                yield tool_call, event
        # Ensure all tasks complete cleanly
        await asyncio.gather(*tasks)


    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Main agent execution handler implementing the autonomous agent loop.
        
        This is the core agent logic that implements the autonomous execution pattern:
        1. Load conversation memory from parent context (if nested)
        2. Add the user prompt to the conversation
        3. Loop until no more tool calls or max_iter reached:
           a. Call LLM with current conversation and available tools
           b. Add LLM response to conversation
           c. If LLM made tool calls, execute them concurrently
           d. Add tool results to conversation and continue
        
        Args:
            **kwargs: Execution parameters including:
                - prompt: The input Message to process
                - Other parameters passed through to LLM
                
        Yields:
            Events from LLM calls and tool executions
        """
        # Build system prompt from template and instructions
        system_prompt = self._build_system_prompt(**kwargs)
        # Initialize conversation with memory + user prompt
        messages = await self._resolve_memory()
        messages.append(kwargs.pop("prompt"))

        i = 0
        while True:
            # Call LLM with current conversation
            async for event in self._llm(
                model=self.model,
                messages=messages,
                system_prompt=system_prompt,
                # Only provide tools if we haven't hit max iterations
                tools=self.tools if i < self.max_iter else [],
                **kwargs, 
            ):
                if isinstance(event, OutputEvent):
                    # If the LLM call fails, we want to propagate the error upwards
                    if event.error is not None:
                        raise RuntimeError(event.error)
                    assert isinstance(event.output, Message), f"Expected event.output to be a Message, got {type(event.output)}"
                    # Add LLM response to conversation for next iteration
                    messages.append(event.output)
                yield event

            tool_calls = [
                content for content in messages[-1].content
                if isinstance(content, ToolUseContent)
            ]
            
            if not tool_calls:
                break

            async for tool_call, event in self._multiplex_tools(tool_calls):
                # Only process events from immediate children (not nested subagents)
                if isinstance(event, OutputEvent) and event.path.count(".") == self._path.count(".") + 1:
                    # Convert tool output to Message format if needed
                    # ? Can we optimize this double validate
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
            i += 1
            