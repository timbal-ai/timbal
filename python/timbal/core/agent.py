import asyncio
import importlib
import inspect
import json
import re
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine
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
    ValidationError,
    computed_field,
    model_validator,
)
from uuid_extensions import uuid7

from ..errors import InterruptError, bail
from ..state import get_run_context
from ..types.content import CustomContent, FileContent, TextContent, ToolUseContent
from ..types.events import BaseEvent, OutputEvent
from ..types.message import Message
from ..utils import coerce_to_dict, dump
from .llm_router import Model, _llm_router
from .runnable import Runnable, RunnableLike
from .skill import ReadSkill, Skill
from .tool import Tool
from .tool_set import ToolSet

logger = structlog.get_logger("timbal.core.agent")


SYSTEM_PROMPT_FN_PATTERN = re.compile(r"\{[a-zA-Z0-9_]*::[a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+)*\}")


class AgentParams(BaseModel):
    """Input parameters for Agent execution. Use either 'prompt' or 'messages', not both."""

    model_config = ConfigDict(extra="allow")

    prompt: Message | None = Field(
        None,
        description="Single input message. Memory is automatically resolved from previous runs.",
    )
    messages: list[Message] | None = Field(
        None,
        description="Explicit list of messages. No automatic memory resolution.",
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
        """Custom schema for using agents as tools."""
        return {
            "type": "object",
            "properties": {
                "prompt": {
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
                        },
                    },
                }
            },
            "required": ["prompt"],
        }


class Agent(Runnable):
    """Orchestrates LLM interactions with autonomous tool calling."""

    model: Model | str
    """LLM model identifier (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-haiku-4-5')."""
    system_prompt: str | Callable[[], str] | Callable[[], Coroutine[Any, Any, str]] | None = None
    """System prompt. Can be a string, sync callable, or async callable returning a string."""
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
    """Internal LLM tool instance."""
    _system_prompt_templates: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Template patterns {module::func} mapped to their callables."""
    _system_prompt_fn: Callable | None = PrivateAttr(default=None)
    """Callable passed as system_prompt."""
    _system_prompt_fn_is_async: bool = PrivateAttr(default=False)
    """Whether _system_prompt_fn is async."""

    def model_post_init(self, __context: Any) -> None:
        """Initialize agent after Pydantic model creation."""
        super().model_post_init(__context)
        self._path = self.name

        # Handle callable system_prompt
        if callable(self.system_prompt):
            self._system_prompt_fn = self.system_prompt
            self._system_prompt_fn_is_async = self._inspect_callable(self.system_prompt)["is_coroutine"]
            self.system_prompt = None

        if self.system_prompt:
            for match in SYSTEM_PROMPT_FN_PATTERN.finditer(self.system_prompt):
                text = match.group()
                path = text[1:-1]  # Remove { and }
                path_parts = path.split("::")
                assert len(path_parts) >= 2, (
                    f"Invalid path format for system prompt: {path}. Review the SYSTEM_PROMPT_FN_PATTERN regex."
                )
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
                                    logger.info(
                                        f"Using already loaded module '{module_path}' for system prompt callable '{text}'"
                                    )
                                    break
                                # Load the module if not already loaded
                                module_spec = importlib.util.spec_from_file_location(
                                    module_path_str, module_path.as_posix()
                                )
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
                    self._system_prompt_templates[text] = {
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
                    self._system_prompt_templates[text] = {
                        "start": match.start(),
                        "end": match.end(),
                        "callable": fn,
                        "is_coroutine": inspect_result["is_coroutine"],
                    }

        model_provider, model_name = self.model.split("/", 1)
        if model_provider == "anthropic":
            if not self.model_params.get("max_tokens"):
                raise ValueError("'max_tokens' is required for claude models.")
            if self.output_model is not None:
                logger.warning(
                    "Anthropic's structured output (output_model) is currently in beta. "
                    "The API may change in future releases. "
                    "See: https://docs.anthropic.com/en/docs/build-with-claude/structured-output"
                )

        if self.tools and self.output_model is not None:
            logger.warning(
                "Using both 'tools' and 'output_model' together may cause unexpected behavior. "
                "Consider using one or the other."
            )

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

        if self.skills_path is not None:
            self.skills_path = Path(self.skills_path).expanduser().resolve()
            if not self.skills_path.exists() or not self.skills_path.is_dir():
                raise ValueError(
                    f"Skills directory {self.skills_path} does not exist or is not a directory. Skipping..."
                )
            for skill_path in self.skills_path.iterdir():
                skill = Skill(path=skill_path)
                self.tools.append(skill)

        # Normalize tools and prevent duplicate names
        names = set()
        skills = []
        for i, tool in enumerate(self.tools):
            # ToolSet resolved later in _resolve_tools()
            if isinstance(tool, ToolSet):
                if isinstance(tool, Skill):
                    if tool.name in names:
                        raise ValueError(f"Skill '{tool.name}' already exists. You can only add a skill once.")
                    names.add(tool.name)
                    skills.append(tool)
                    # skills_metadata.append(f"- **{tool.name}**: {tool.description}")
                continue
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

        if skills:
            skills_in_system_prompt = "\n".join([f"- **{skill.name}**: {skill.description}" for skill in skills])
            read_skill_tool = ReadSkill(agent_path=self._path, skills=skills)
            read_skill_tool.nest(self._path)
            self.tools.append(read_skill_tool)
            if not isinstance(self.system_prompt, str):
                self.system_prompt = ""
            self.system_prompt += f"""
<skills>
Skills provide additional knowledge of a specific topic. The following skills are available:
{skills_in_system_prompt}
In skills documentation, you will encounter references to additional files.
If the file is relevant for the user query, USE the `read_skill` tool to get its content.
</skills>"""

        self._is_orchestrator = True
        self._is_coroutine = False
        self._is_gen = False
        self._is_async_gen = True

    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"
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
        """Resolve system prompt by executing callable or embedded template functions."""
        if self._system_prompt_fn is not None:
            return await self._execute_runtime_callable(self._system_prompt_fn, self._system_prompt_fn_is_async)

        if not self.system_prompt:
            return None
        if not self._system_prompt_templates:
            return self.system_prompt

        # Execute template functions in parallel
        system_prompt_tasks = []
        for _, v in self._system_prompt_templates.items():
            callable_fn = v["callable"]
            system_prompt_tasks.append(self._execute_runtime_callable(callable_fn, v["is_coroutine"]))
        results = await asyncio.gather(*system_prompt_tasks)

        # Substitute results into template
        system_prompt = self.system_prompt
        for (k, _), result in zip(self._system_prompt_templates.items(), results, strict=False):
            system_prompt = system_prompt.replace(k, str(result) if result is not None else "")

        return system_prompt

    async def resolve_memory(self) -> None:
        """Resolve conversation memory from previous agent trace."""
        run_context = get_run_context()
        assert run_context is not None, "Run context not found"

        current_span = run_context.current_span()
        if current_span.memory:
            return
        current_span.memory = [Message.validate(current_span.input.get("prompt", ""))]

        # Subagents have isolated context
        if current_span.parent_call_id is not None:
            return

        # User can override the message list
        input_messages = current_span.input.get("messages", [])
        if input_messages:
            current_span.memory = [Message.validate(m) for m in input_messages]
            return

        if not run_context.parent_id:
            return
        parent_trace = await run_context._get_parent_trace()
        if parent_trace is None:
            logger.error(
                "Parent trace not found. Continuing without memory...",
                parent_id=run_context.parent_id,
                run_id=run_context.id,
            )
            return

        self_spans = parent_trace.get_path(self._path)
        if not len(self_spans):
            return
        if len(self_spans) > 1:
            # TODO Handle multiple call_ids for this agent (we could access step spans)
            raise NotImplementedError("Multiple spans for the same agent are not supported yet.")

        previous_span = self_spans[0]
        # >= 1.1.0: memory stored in agent span
        if isinstance(previous_span.memory, list):
            memory = [Message.validate(m) for m in previous_span.memory]
        else:
            # < 1.1.0: extract from LLM calls
            llm_spans = parent_trace.get_path(self._llm._path)
            if not len(llm_spans):
                return
            llm_input_messages = llm_spans[-1].input.get("messages", [])
            llm_output_message = llm_spans[-1].output
            memory = [*[Message.validate(m) for m in llm_input_messages], Message.validate(llm_output_message)]

        # Ensure interrupted tool calls have corresponding results.
        # When resuming from memory, the last assistant message may contain tool_use blocks
        # that were interrupted before their results could be recorded. The LLM expects every
        # tool_use to have a corresponding tool_result, so we need to synthesize error results
        # for any missing ones.
        #
        # We iterate in reverse to detect if there's any non-tool_use content (text, etc.)
        # after a server_tool_use block. If there is, the server tool completed and the LLM
        # already continued, so no synthetic result is needed.
        #
        # For regular tool_use: append a tool_result message with an error.
        # For server_tool_use (e.g., web_search): append an inline error result in the same
        # message, using the provider's expected error format (currently Anthropic only).
        # TODO OpenAI & other server tool use providers
        has_followup_after_server_tool_use = False
        for content in memory[-1].content[::-1]:
            if content.type != "tool_use":
                has_followup_after_server_tool_use = True
                continue
            if content.is_server_tool_use:
                if has_followup_after_server_tool_use:
                    continue
                # Server tool use was interrupted before completion. Synthesize an error result.
                memory[-1].content.append(
                    CustomContent(
                        value={
                            "type": "web_search_tool_result",
                            "tool_use_id": content.id,
                            "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
                        },
                    )
                )
            else:
                # Regular tool use was interrupted. Append a tool_result message with error.
                memory.append(
                    Message.validate(
                        {
                            "role": "tool",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "id": content.id,
                                    "content": "ERROR: There was an unexpected error executing the tool.",
                                }
                            ],
                        }
                    )
                )
        current_span.memory = memory + current_span.memory

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

        if self._bg_tasks:
            get_background_task_tool = Tool(
                name="get_background_task",
                description="Get the status and events of a background task.",
                handler=self.get_background_task,
            )
            get_background_task_tool.nest(self._path)
            tools.append(get_background_task_tool)
            tools_names.add("get_background_task")
            # Add to commands dict if the tool has a command attribute
            # if get_background_task_tool.command:
            #     commands[get_background_task_tool.command] = get_background_task_tool

        return tools, commands

    async def _multiplex_tools(self, tools: list[Tool], tool_calls: list[ToolUseContent]) -> AsyncGenerator[Any, None]:
        """Execute multiple tool calls concurrently and multiplex their events."""
        queue = asyncio.Queue()
        tasks = []

        async def consume_tool(tool_call: ToolUseContent):
            tool = next((t for t in tools if t.name == tool_call.name), None)
            assert tool is not None, f"Tool {tool_call.name} not found"
            try:
                async for event in tool(**tool_call.input):
                    # Link tool call id to span for memory resolution
                    if event.type == "START":
                        tool_call_id = event.call_id
                        tool_call_span = get_run_context()._trace[tool_call_id]
                        tool_call_span.metadata["tool_call_id"] = tool_call.id
                    await queue.put((tool_call, event))
            finally:
                await queue.put((tool_call, None))

        try:
            for tc in tool_calls:
                task = asyncio.create_task(consume_tool(tc))
                tasks.append(task)

            remaining = len(tool_calls)
            while remaining > 0:
                tool_call, event = await queue.get()
                if event is None:
                    remaining -= 1
                else:
                    yield tool_call, event
        except (asyncio.CancelledError, GeneratorExit, InterruptError):
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Execute the autonomous agent loop."""
        run_context = get_run_context()
        assert run_context is not None, "Run context is not initialized"
        current_span = run_context.current_span()

        model = kwargs.pop("model", self.model)

        # ? Do we want to allow the user to pass parameterized system prompts
        system_prompt = kwargs.pop("system_prompt", None)
        if not system_prompt:
            system_prompt = await self._resolve_system_prompt()

        await self.resolve_memory()
        # Span memory will also be modified with messages array modification (it's the same object)
        # current_span.memory = messages
        current_span._memory_dump = await dump(
            current_span.memory
        )  # ? Can we optimize the dumping the llm already does next
        # Remove these so the llm runnable doesn't try to use/validate them again
        kwargs.pop("prompt", None)
        kwargs.pop("messages", None)

        async def _process_tool_event(event: BaseEvent, tool_call_id: str, append_to_messages: bool = True):
            """Helper to process tool output events and create tool results."""
            if not isinstance(event, OutputEvent) or event.path.count(".") != self._path.count(".") + 1:
                return
            if event.status.code == "cancelled" and event.status.reason == "early_exit":
                bail(event.status.message)
            content = None
            if event.status.code == "cancelled" and event.status.reason == "early_exit_local":
                msg = event.status.message or "The tool exited early."
                content = f"[Cancelled] {msg}"
            elif event.error is not None:
                content = event.error
            elif isinstance(event.output, Message):
                content = [c for c in event.output.content if isinstance(c, TextContent | FileContent)]
            else:
                content = event.output
            tool_result = Message.validate(
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "id": tool_call_id,
                            "content": content,
                        }
                    ],
                }
            )
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
                            current_span._memory_dump.append(
                                {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "tool_use", "id": tool_use_id, "name": tool.name, "input": tool_input}
                                    ],
                                }
                            )
                            # Run the tool
                            async for event in tool(**tool_input):
                                await _process_tool_event(event, tool_use_id, append_to_messages=False)
                                if isinstance(event, OutputEvent) and event.output is not None:
                                    current_span.memory.append(
                                        Message.validate(
                                            {
                                                "role": "assistant",
                                                "content": [TextContent(text=str(event.output))],
                                            }
                                        )
                                    )
                                yield event
                            return

            async for event in self._llm(
                model=model,
                messages=current_span.memory,
                system_prompt=system_prompt,
                tools=tools,
                output_model=self.output_model,
                **kwargs,
            ):
                if isinstance(event, OutputEvent):
                    # If the LLM call fails, we want to propagate the error upwards
                    if event.error is not None:
                        raise RuntimeError(event.error)

                    interrupted = event.status.code == "cancelled" and event.status.reason == "interrupted"

                    # Handle case where LLM was interrupted before any output was received
                    if event.output is None and interrupted:
                        raise InterruptError(event.call_id, output=None)

                    assert isinstance(event.output, Message), (
                        f"Expected event.output to be a Message, got {type(event.output)}"
                    )

                    # Add LLM response to conversation for next iteration
                    current_span.memory.append(event.output)
                    current_span._memory_dump.append(event._output_dump)

                    if self.output_model is not None:
                        validation_error_msg = None
                        validation_context = self.model_params.get("validation_context")
                        for content in event.output.content:
                            # TODO Refactor this. If the llm should return a fixed output model, we should error if it doesn't return it
                            if isinstance(content, TextContent):
                                try:
                                    output = coerce_to_dict(content.text)
                                    validated_output = self.output_model.model_validate(
                                        output, context=validation_context
                                    )
                                    event.output = validated_output
                                    break
                                except (json.JSONDecodeError, ValueError, ValidationError) as e:
                                    validation_error_msg = str(e)
                                    logger.warning(f"Output validation failed: {validation_error_msg}")
                                    continue

                        if validation_error_msg and i < self.max_iter - 1:
                            # Feed the error back to the LLM so it can correct itself
                            error_feedback = Message.validate(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                f"Your output failed validation:\n{validation_error_msg}\n\nTry again."
                                            ),
                                        }
                                    ],
                                }
                            )
                            current_span.memory.append(error_feedback)
                            current_span._memory_dump.append(await dump(error_feedback))
                            i += 1
                            break  # Break out of async for to retry in while loop

                    # Propagate the interruption with the processed output
                    if interrupted:
                        raise InterruptError(event.call_id, output=event.output)
                yield event

            if self.output_model is not None:
                break

            tool_calls = [
                content
                for content in current_span.memory[-1].content
                if isinstance(content, ToolUseContent) and not content.is_server_tool_use
            ]

            if not tool_calls:
                break

            async for tool_call, event in self._multiplex_tools(tools, tool_calls):
                await _process_tool_event(event, tool_call.id, append_to_messages=True)
                yield event
            i += 1
