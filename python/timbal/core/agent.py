import asyncio
import importlib
import inspect
import json
import re
import sys
from collections.abc import AsyncGenerator
from functools import cached_property
from pathlib import Path
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    computed_field,
    model_validator,
)
from uuid_extensions import uuid7

from ..errors import InterruptError, bail
from ..state import get_run_context
from ..types.content import TextContent, ToolUseContent
from ..types.events import BaseEvent, OutputEvent
from ..types.message import Message
from ..utils import dump
from .llm_router import Model, _llm_router
from .runnable import Runnable, RunnableLike
from .skill import ReadSkill, Skill
from .tool import Tool
from .tool_set import ToolSet

logger = structlog.get_logger("timbal.core.agent")


SYSTEM_PROMPT_FN_PATTERN = re.compile(r"\{[a-zA-Z0-9_]*::[a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+)*\}")


class AgentParams(BaseModel):
    """Parameter model for Agent execution.
    
    Defines the input parameters that agents accept when called.
    Use either 'prompt' or 'messages', not both:
    - 'prompt': Single message input. The framework will automatically resolve 
      and include memory from previous runs.
    - 'messages': Explicit list of messages. No automatic memory resolution occurs;
      you have full control over the message history.
    """
    model_config = ConfigDict(extra="allow")

    prompt: Message | None = Field(
        None,
        description="Single input message. Framework automatically resolves memory from previous runs.",
    )
    messages: list[Message] | None = Field(
        None,
        description="Explicit list of messages. No automatic memory resolution; full manual control.",
    )

    @model_validator(mode="after")
    def validate_prompt_or_messages(self) -> "AgentParams":
        """Ensure exactly one of prompt or messages is set."""
        if self.prompt is not None and self.messages is not None:
            logger.warning("Calling agent with both 'prompt' and 'messages'. Using 'messages'.")
        if self.prompt is None and self.messages is None:
            raise ValueError("Must specify either 'prompt' or 'messages'.")
        return self

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:
        """Override model_json_schema to return a custom schema so that agents can be used as tools more easily."""
        return {
            "type": "object",
            "properties": {"prompt": {
                "type": "object",
                "title": "TimbalMessage",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["user"],
                    },
                    "content": {
                        "type": "array",
                        "items": {},
                    }
                },
            }},
            "required": ["prompt"],
        }


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
    - Integration with multiple LLM providers via _llm_router
    """

    model: Model | str
    """The LLM model identifier (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-haiku-4-5')."""
    system_prompt: str | None = None
    """System prompt to provide context for the agent."""
    tools: list[SkipValidation[RunnableLike]] = []
    """List of tools available to the agent. Can be functions, dicts, or Runnable objects."""
    skills_path: str | Path | None = None
    """Path to the skills directory."""
    max_iter: int = 10
    """Maximum number of LLM->tool call iterations before stopping."""
    model_params: dict[str, Any] = {}
    """Model parameters to pass to the agent."""
    output_model: type[BaseModel] | None = None
    """BaseModel to generate a structured output."""

    _llm: Tool = PrivateAttr()
    """Internal LLM tool instance for making model calls."""
    _system_prompt_callables: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Dictionary mapping template patterns to their callable functions and metadata."""


    def model_post_init(self, __context: Any) -> None:
        """Initialize agent-specific attributes after Pydantic model creation.
        
        This method sets up the agent's internal tools and execution characteristics:
        1. Parses and loads system prompt template functions if present
        2. Creates an internal LLM tool for model interactions
        3. Normalizes user-provided tools into Tool instances
        4. Sets up tool name mapping for fast lookup during execution
        5. Configures execution characteristics as an orchestrator
        
        System prompt template functions are discovered by parsing the system_prompt
        for patterns like {namespace::function} and dynamically importing the callable
        from either packages or files relative to the Agent constructor's caller.
        """
        super().model_post_init(__context)
        self._path = self.name

        if self.system_prompt:
            for match in SYSTEM_PROMPT_FN_PATTERN.finditer(self.system_prompt):
                text = match.group()
                path = text[1:-1]  # Remove { and }
                path_parts = path.split("::")
                assert len(path_parts) >= 2, f"Invalid path format for system prompt: {path}. Review the SYSTEM_PROMPT_FN_PATTERN regex."
                module, fn_i = None, None
                if path_parts[0] == "":
                    # File-relative import: look in caller's globals first
                    frame = inspect.currentframe()
                    caller_globals = None
                    while frame:
                        frame = frame.f_back
                        if frame is None:
                            break
                        frame_self = frame.f_locals.get("self")
                        if not isinstance(frame_self, Agent):
                            caller_globals = frame.f_globals
                            break
                    assert caller_globals is not None, "Could not determine caller globals for Agent constructor."
                    
                    # Try to resolve from caller's globals directly (handles same-file definitions)
                    fn = caller_globals
                    try:
                        for attr_name in path_parts[1:]:
                            fn = fn[attr_name] if isinstance(fn, dict) else getattr(fn, attr_name)
                        logger.info(f"Resolved callable '{text}' from caller's globals")
                    except (KeyError, AttributeError):
                        # If not in globals, try loading the module
                        caller_file = Path(caller_globals.get("__file__", "")).expanduser().resolve()
                        agent_path = caller_file
                        for fn_i in range(1, len(path_parts)):
                            module_path = agent_path / "/".join(path_parts[:-fn_i])
                            try:
                                # Use absolute path as module identifier
                                module_path_str = str(module_path.resolve())
                                # Check if module is already loaded in sys.modules by absolute path
                                if module_path_str in sys.modules:
                                    module = sys.modules[module_path_str]
                                    logger.info(f"Using already loaded module '{module_path}' for system prompt callable '{text}'")
                                    break
                                # Load the module if not already loaded
                                module_spec = importlib.util.spec_from_file_location(module_path_str, module_path.as_posix())
                                if not module_spec or not module_spec.loader:
                                    raise ValueError(f"Failed to load module {module_path}")
                                module = importlib.util.module_from_spec(module_spec)
                                # Register in sys.modules BEFORE executing to prevent re-entry
                                sys.modules[module_path_str] = module
                                module_spec.loader.exec_module(module)
                                logger.info(f"Loaded module '{module_path}' for system prompt callable '{text}'")
                                break
                            except Exception:
                                pass
                        else:
                            for fn_i in range(1, len(path_parts)):
                                module_path = ".".join(path_parts[:-fn_i])
                                try:
                                    module = importlib.import_module(module_path)
                                    logger.info(f"Loaded module '{module_path}' for system prompt callable '{text}'")
                                    break
                                except Exception:
                                    pass
                        fn = module
                        for j in path_parts[-fn_i:]:
                            fn = getattr(fn, j)
                    
                    inspect_result = self._inspect_callable(fn)
                    self._system_prompt_callables[text] = {
                        "start": match.start(),
                        "end": match.end(),
                        "callable": fn,
                        "is_coroutine": inspect_result["is_coroutine"],
                    }
                else:
                    # Package import (e.g., {os::getcwd})
                    for fn_i in range(1, len(path_parts)):
                        module_path = ".".join(path_parts[:-fn_i])
                        try:
                            module = importlib.import_module(module_path)
                            logger.info(f"Loaded module '{module_path}' for system prompt callable '{text}'")
                            break
                        except Exception:
                            pass
                    fn = module
                    for j in path_parts[-fn_i:]:
                        fn = getattr(fn, j)
                    inspect_result = self._inspect_callable(fn)
                    self._system_prompt_callables[text] = {
                        "start": match.start(),
                        "end": match.end(),
                        "callable": fn,
                        "is_coroutine": inspect_result["is_coroutine"],
                    }
        
        model_provider, model_name = self.model.split("/", 1)
        if model_provider == "anthropic":
            if not self.model_params.get("max_tokens"):
                raise ValueError("'max_tokens' is required for claude models.")
        
        # Create internal LLM tool for model interactions
        self._llm = Tool(
            name="llm",
            handler=_llm_router,
            default_params=self.model_params,
            metadata={
                "type": "LLM",
                "model_provider": model_provider,
                "model_name": model_name,
            },
        )
        self._llm.nest(self._path)

        # Load skills directory if provided
        if self.skills_path is not None:
            self.skills_path = Path(self.skills_path).expanduser().resolve()
            if not self.skills_path.exists() or not self.skills_path.is_dir():
                raise ValueError(f"Skills directory {self.skills_path} does not exist or is not a directory. Skipping...")
            for skill_path in self.skills_path.iterdir():
                skill = Skill(path=skill_path)
                self.tools.append(skill)

        # Add structured output tool if output_model is provided
        if self.output_model:
            output_model_tool = Tool(
                name="output_model_tool",
                description="Use it always before providing the final answer to give the structured output.",
                handler=lambda x: x
            )
            output_model_tool.params_model = self.output_model
            self.tools.append(output_model_tool)

        # Normalize the rest of the tools and prevent duplicate names
        names = set()
        skills_metadata = []
        for i, tool in enumerate(self.tools):
            # ToolSet instances are kept as-is and resolved later in _resolve_tools()
            # _resolve_tools() nests the tools with the orchestrator path - no need to do anything here
            if isinstance(tool, ToolSet):
                if isinstance(tool, Skill):
                    if tool.name in names:
                        raise ValueError(f"Skill '{tool.name}' already exists. You can only add a skill once.")
                    names.add(tool.name)
                    skills_metadata.append(f"- **{tool.name}**: {tool.description}")
                continue
            # Otherwise make sure they are runnable instances
            if not isinstance(tool, Runnable):
                if isinstance(tool, dict):
                    tool = Tool(**tool)
                else:
                    tool = Tool(handler=tool)
            if tool.name in names:
                raise ValueError(f"Tool {tool.name} already exists. You can only add a tool once.")
            names.add(tool.name)
            tool.nest(self._path)
            self.tools[i] = tool

        # If there are skills, we need to add the read_skill tool and indicate the agent about the skills in the system prompt
        if skills_metadata:
            read_skill_tool = ReadSkill()
            read_skill_tool.nest(self._path)
            self.tools.append(read_skill_tool)
            if not isinstance(self.system_prompt, str):
                self.system_prompt = ""
            self.system_prompt += f"""
<skills>
Skills provide additional knowledge of a specific topic. The following skills are available:
{'\n'.join(skills_metadata)}
In skills documentation, you will encounter references to additional files.
If the file is relevant for the user query, USE the `read_skill` tool to get its content.
</skills>"""

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


    async def _resolve_system_prompt(self) -> str | None:
        """Resolve system prompt by executing embedded template functions."""
        if not self.system_prompt:
            return None
        if not self._system_prompt_callables:
            return self.system_prompt

        system_prompt_tasks = []
        for _, v in self._system_prompt_callables.items():
            callable_fn = v["callable"]
            system_prompt_tasks.append(self._execute_runtime_callable(callable_fn, v["is_coroutine"]))
        results = await asyncio.gather(*system_prompt_tasks)
        
        # TODO: Optimize with single-pass substitution using stored positions
        system_prompt = self.system_prompt
        for (k, _), result in zip(self._system_prompt_callables.items(), results, strict=False):
            system_prompt = system_prompt.replace(k, str(result) if result is not None else "")
        
        return system_prompt


    async def _resolve_memory(self) -> list[Message]:
        """Resolve conversation memory from previous agent trace."""
        run_context = get_run_context()
        if not run_context.parent_id:
            return []
        current_span = run_context.current_span()
        # Try to get tracing data from parent execution
        parent_trace = await run_context._get_parent_trace()
        if parent_trace is None:
            logger.error("Parent trace not found. Continuing without memory...", parent_id=run_context.parent_id, run_id=run_context.id)
            return []

        self_spans = parent_trace.get_path(self._path)
        if not len(self_spans):
            return []
        if len(self_spans) > 1:
            # TODO Handle multiple call_ids for this agent (we could access step spans)
            raise NotImplementedError("Multiple spans for the same agent are not supported yet.")
        previous_span = self_spans[0]
        if hasattr(previous_span, "in_context_skills"):
            current_span.in_context_skills = previous_span.in_context_skills

        # >= 1.1.0 memory is stored in the agent span. This enables us to modify the memory directly without passing it to the LLM.
        if hasattr(previous_span, "memory"):
            memory = [Message.validate(m) for m in previous_span.memory]
        else:
            # < 1.1.0 Extract conversation history from parent's LLM calls
            # TODO Handle multiple call_ids for this agent (we could access step spans)
            llm_spans = parent_trace.get_path(self._llm._path)
            # In a subagent, this can be empty if the parent agent didn't call the LLM
            if not len(llm_spans):
                return []
            # Get the most recent LLM interaction
            llm_input_messages = llm_spans[-1].input.get("messages", [])
            llm_output_message = llm_spans[-1].output
            # Reconstruct conversation: input messages + LLM response
            memory = [
                *[Message.validate(m) for m in llm_input_messages], 
                Message.validate(llm_output_message)
            ]
        
        # Make sure interrupted tool calls have a corresponding tool result
        # We assume all previous messages but the last one have matching tool uses and tool results
        for content in memory[-1].content:
            if content.type != "tool_use" or content.is_server_tool_use:
                continue
            tool_result_path = f"{self._path}.{content.name}"
            tool_result_spans = parent_trace.get_path(tool_result_path)
            tool_result_content = None
            for tool_result_span in tool_result_spans:
                if tool_result_span.metadata.get("tool_call_id") == content.id:
                    tool_result_content = tool_result_span.output.content if isinstance(tool_result_span.output, Message) else tool_result_span.output
                    tool_result_status_code = tool_result_span.status.code if hasattr(tool_result_span.status, "code") else tool_result_span.status.get("code")
                    tool_result_status_reason = tool_result_span.status.reason if hasattr(tool_result_span.status, "reason") else tool_result_span.status.get("reason")
                    if tool_result_status_code == "cancelled" and tool_result_status_reason == "interrupted":
                        tool_result_content = "INTERRUPTED: Tool call was cancelled or interrupted by the user / system."
            # If there's no matching tool result, we create an empty one indicating there was an error
            if tool_result_content is None:
                tool_result_content = "ERROR: There was an unexpected error executing the tool."
            memory.append(Message.validate({
                "role": "tool",
                "content": [{
                    "type": "tool_result",
                    "id": content.id,
                    "content": tool_result_content,
                }]
            }))
        return memory


    async def _resolve_tools(self, i: int) -> tuple[list[Tool], dict[str, Tool]]:
        """Resolve the tools to be provided to the LLM."""
        if i >= self.max_iter:
            return [], {}
        tools = []
        tools_names = set()
        commands = {}
        for t in self.tools:
            if isinstance(t, ToolSet):
                resolved_toolset = await t.resolve()
                for tool in resolved_toolset:
                    if tool.name in tools_names:
                        logger.warning(f"Tool with name '{tool.name}' already exists. You can only add a tool once.")
                    else:
                        tool.nest(self._path)
                        tools.append(tool)
                        tools_names.add(tool.name)
                        if tool.command:
                            commands[tool.command] = tool
            else:
                if t.name in tools_names:
                    logger.warning(f"Tool with name '{t.name}' already exists. You can only add a tool once.")
                else:
                    tools.append(t)
                    tools_names.add(t.name)
                    if t.command:
                        commands[t.command] = t
        return tools, commands


    async def _multiplex_tools(self, tools: list[Tool], tool_calls: list[ToolUseContent]) -> AsyncGenerator[Any, None]:
        """Execute multiple tool calls concurrently and multiplex their events."""
        queue = asyncio.Queue()
        tasks = []
        
        async def consume_tool(tool_call: ToolUseContent):
            """Consume events from a single tool and put them in the queue."""
            tool = next((t for t in tools if t.name == tool_call.name), None)
            assert tool is not None, f"Tool {tool_call.name} not found"
            try:
                async for event in tool(**tool_call.input):
                    # We need to link the tool call id to the span so that we can later match when resolving memory
                    if event.type == "START":
                        tool_call_id = event.call_id 
                        tool_call_span = get_run_context()._trace[tool_call_id]
                        tool_call_span.metadata["tool_call_id"] = tool_call.id
                    await queue.put((tool_call, event))
            finally:
                await queue.put((tool_call, None))  # Sentinel
        
        try:
            # Start all tool tasks
            for tc in tool_calls:
                task = asyncio.create_task(consume_tool(tc))
                tasks.append(task)
            
            # Consume events as they arrive
            remaining = len(tool_calls)
            while remaining > 0:
                tool_call, event = await queue.get()
                if event is None:
                    remaining -= 1
                else:
                    yield tool_call, event
        except (asyncio.CancelledError, GeneratorExit, InterruptError):
            # Cancellation or generator closed - clean up gracefully
            raise
        finally:
            # Cancel all pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all cancellations to complete, suppressing errors
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Main agent execution handler implementing the autonomous agent loop.
        
        This is the core agent logic that implements the autonomous execution pattern:
        1. Load conversation memory from parent context (if nested)
        2. Add the user prompt to the conversation
        3. Check for slash commands (e.g., /command args):
           a. If command found, execute the corresponding tool directly
           b. Add tool use and result to memory, then return early
        4. Loop until no more tool calls or max_iter reached:
           a. Call LLM with current conversation and available tools
           b. Add LLM response to conversation
           c. If LLM made tool calls, execute them concurrently
           d. Add tool results to conversation and continue
        
        Args:
            **kwargs: Execution parameters including:
                - prompt: The input Message to process (or messages list)
                - messages: Optional explicit list of messages (bypasses memory resolution)
                - system_prompt: Optional system prompt override
                - Other parameters passed through to LLM
                
        Yields:
            Events from LLM calls and tool executions
        """
        current_span = get_run_context().current_span()
        # ? Do we want to allow the user to pass parameterized system prompts
        system_prompt = kwargs.pop("system_prompt", None)
        if not system_prompt:
            system_prompt = await self._resolve_system_prompt()
            
        # We allow the user to pass a 'hardcoded' list of messages
        # Or simply a prompt and we attempt to resolve memory
        current_span.memory = kwargs.pop("messages", [])
        if not current_span.memory:
            current_span.memory = await self._resolve_memory()
            current_span.memory.append(kwargs.pop("prompt"))
        # Span memory will also be modified with messages array modification (it's the same object)
        # current_span.memory = messages
        current_span._memory_dump = await dump(current_span.memory) # ? Can we optimize the dumping the llm already does next

        async def _process_tool_event(event: BaseEvent, tool_call_id: str, append_to_messages: bool = True):
            """Helper to process tool output events and create tool results."""
            if not isinstance(event, OutputEvent) or event.path.count(".") != self._path.count(".") + 1:
                return
            if event.status.code == "cancelled" and event.status.reason == "early_exit":
                bail(event.status.message)
            content = None
            if event.error is not None:
                content = event.error
            elif isinstance(event.output, Message):
                content = event.output.content
            else:
                content = event.output
            tool_result = Message.validate({
                "role": "tool",
                "content": [{
                    "type": "tool_result",
                    "id": tool_call_id,
                    "content": content,
                }]
            })
            if append_to_messages:
                current_span.memory.append(tool_result)
            tool_result_dump = await dump(tool_result)
            current_span._memory_dump.append(tool_result_dump)

        i = 0
        while True:
            # ? We could resolve the system prompt at each iteration
            tools, commands = await self._resolve_tools(i)
            if commands:
                # Commands will only be user messages with a single text content
                if len(current_span.memory[-1].content) == 1:
                    content = current_span.memory[-1].content[0]
                    if isinstance(content, TextContent) and content.text.startswith("/"):
                        import shlex
                        args = shlex.split(content.text)
                        command = args[0].strip("/")
                        args = args[1:]
                        # If no command is found, we'll simply let the message pass through the LLM
                        if command in commands:
                            tool = commands[command]
                            tool_input = {}
                            for i, field_name in enumerate(tool.params_model.model_fields.keys()):
                                # Params model preserves the ordering of the fields as they appear in the signature
                                # We grab as many arguments as we can. If there are too few arguments, we'll let the tool params model validator give a better error
                                if i >= len(args):
                                    break
                                tool_input[field_name] = args[i]
                            # Craft a fake tool_use so we can keep this interaction in the agent memory
                            tool_use_id = uuid7(as_type="str").replace("-", "")
                            current_span._memory_dump.append({
                                "role": "assistant",
                                "content": [{
                                    "type": "tool_use",
                                    "id": tool_use_id,
                                    "name": tool.name,
                                    "input": tool_input
                                }]
                            })
                            # Run the tool
                            async for event in tool(**tool_input):
                                await _process_tool_event(event, tool_use_id, append_to_messages=False)
                                yield event
                            return

            is_output_model = False
            async for event in self._llm(
                model=self.model,
                messages=current_span.memory,
                system_prompt=system_prompt,
                tools=tools,
                **kwargs, 
            ):
                if isinstance(event, OutputEvent):
                    # If the LLM call fails, we want to propagate the error upwards
                    if event.error is not None:
                        raise RuntimeError(event.error)
                    # If the LLM was interrupted, propagate the interruption
                    if event.status.code == "cancelled" and event.status.reason == "interrupted":
                        raise InterruptError(event.call_id)
                    assert isinstance(event.output, Message), f"Expected event.output to be a Message, got {type(event.output)}"


                    # Add LLM response to conversation for next iteration
                    current_span.memory.append(event.output)
                    # This breaks the reference to the original messages object. We need to manually append the output to the span memory.
                    current_span._memory_dump.append(event._output_dump)

                    for content in event.output.content:
                        if isinstance(content, TextContent):
                            try:
                                json_data = json.loads(content.text)
                                event.output = self.output_model(**json_data)
                                is_output_model = True
                                break
                            except (json.JSONDecodeError, ValueError):
                                continue
                
                yield event

            if is_output_model:
                break

            tool_calls = [
                content for content in current_span.memory[-1].content
                if isinstance(content, ToolUseContent) and not content.is_server_tool_use
            ]
            
            if not tool_calls:
                break           

            async for tool_call, event in self._multiplex_tools(tools, tool_calls):
                await _process_tool_event(event, tool_call.id, append_to_messages=True)
                yield event
            i += 1