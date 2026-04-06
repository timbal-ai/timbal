import asyncio
import contextvars
import json
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
    SecretStr,
    SkipValidation,
    ValidationError,
    computed_field,
    model_validator,
)
from uuid_extensions import uuid7

from ..errors import InterruptError, bail
from ..state import get_run_context
from ..types.content import CustomContent, FileContent, TextContent, ToolResultContent, ToolUseContent
from ..types.events import BaseEvent, OutputEvent
from ..types.message import Message
from ..utils import coerce_to_dict, dump
from .llm_router import _llm_router
from .memory_compaction import MemoryCompactor
from .models import Model, get_context_window
from .runnable import Runnable, RunnableLike
from .skill import ReadSkill, Skill
from .tool import Tool
from .tool_set import ToolSet

logger = structlog.get_logger("timbal.core.agent")


def _content_chars(c: TextContent | ToolUseContent | ToolResultContent) -> int:
    if isinstance(c, TextContent):
        return len(c.text)
    if isinstance(c, ToolUseContent):
        return len(c.name) + len(str(c.input))
    if isinstance(c, ToolResultContent):
        return sum(len(item.text) for item in c.content if isinstance(item, TextContent))
    return 0


def _estimate_tokens_from_memory(memory: list[Message]) -> int:
    """Rough token estimate from message content. 1 token ≈ 4 chars (standard approximation)."""
    return max(1, sum(_content_chars(c) for msg in memory for c in msg.content) // 4)


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

    model: SkipValidation[Model | str]
    """LLM model identifier (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-haiku-4-5') or a TestModel for offline testing."""
    system_prompt: str | Callable[[], str] | Callable[[], Coroutine[Any, Any, str]] | None = None
    """System prompt. Can be a string, sync callable, or async callable returning a string."""
    tools: list[SkipValidation[RunnableLike]] = []
    """List of tools available to the agent. Can be functions, dicts, or Runnable objects."""
    skills_path: str | Path | None = None
    """Path to the skills directory."""
    max_iter: int = 10
    """Maximum number of LLM->tool call iterations before stopping."""
    max_tokens: int | None = None
    """Maximum tokens for the LLM response. Required for Anthropic models."""
    memory_compaction: list[SkipValidation[MemoryCompactor]] | SkipValidation[MemoryCompactor] | None = None
    """Memory compaction strategies applied after resolve_memory. Reduces context window usage.
    Can be a single compactor or list (applied in order). Built-in strategies:
    compact_tool_results(keep_last_n=None, threshold=0, replacement=None),
    keep_last_n_messages(n), keep_last_n_turns(n),
    summarize(threshold, model=None, keep_last_n=4, max_summary_tokens=500).
    summarize uses the agent's model by default; override with a cheaper model if needed.
    Compaction is triggered automatically when context window utilization exceeds memory_compaction_ratio."""
    memory_compaction_ratio: float = 0.75
    """Context window utilization ratio that triggers compaction. Uses previous run's token
    usage from span data and the model's context window from models.yaml. Set to 0.0 to
    always compact, or 1.0 to effectively disable auto-triggering. Default: 0.75 (75%)."""
    temperature: float | None = None
    """Sampling temperature for the LLM response."""
    output_model: type[BaseModel] | None = None
    """BaseModel to generate a structured output."""
    output_model_context: Any = None
    """Validation context passed to Pydantic's model_validate when validating output_model responses."""
    model_params: dict[str, Any] = {}
    """Additional provider-specific LLM parameters (e.g. thinking, service_tier, logprobs, top_p, stop)."""
    base_url: str | None = None
    """Custom base URL for the LLM API."""
    api_key: SecretStr | None = None
    """Custom API key for the LLM API."""

    _llm: Tool = PrivateAttr()
    _system_prompt_fn: Callable | None = PrivateAttr(default=None)
    _system_prompt_fn_is_async: bool = PrivateAttr(default=False)
    _system_prompt_skills: str | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize agent after Pydantic model creation."""
        super().model_post_init(__context)
        self._path = self.name

        # Handle callable system_prompt
        if callable(self.system_prompt):
            self._system_prompt_fn = self.system_prompt
            self._system_prompt_fn_is_async = self._inspect_callable(self.system_prompt)["is_coroutine"]
            self.system_prompt = None

        # Backward compat: silently promote known keys from model_params to individual fields
        if self.model_params:
            _promote = {
                "max_tokens": "max_tokens",
                "temperature": "temperature",
                "base_url": "base_url",
                "api_key": "api_key",
                "validation_context": "output_model_context",
            }
            for old_key, new_key in _promote.items():
                if old_key in self.model_params:
                    if getattr(self, new_key) is None:
                        setattr(self, new_key, self.model_params.pop(old_key))
                    else:
                        self.model_params.pop(old_key)

        if not isinstance(self.model, str):
            model_provider = getattr(self.model, "provider", "test")
            model_name = getattr(self.model, "model_name", "model")
        else:
            model_provider, model_name = self.model.split("/", 1)
            if model_provider == "anthropic":
                if not self.max_tokens:
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

        # Build default params for the internal LLM tool from individual fields
        _llm_default_params = {
            k: v for k, v in [
                ("max_tokens", self.max_tokens),
                ("temperature", self.temperature),
                ("base_url", self.base_url),
                ("api_key", self.api_key),
            ] if v is not None
        }
        if self.model_params:
            _llm_default_params["provider_params"] = self.model_params

        self._llm = Tool(
            name="llm",
            handler=_llm_router,
            default_params=_llm_default_params,
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
                skill._agent_path = self._path
                self.tools.append(skill)

        # Normalize tools and prevent duplicate names
        names: set[str] = set()
        skills = []
        for i, tool in enumerate(self.tools):
            if isinstance(tool, Skill):
                if tool.name in names:
                    raise ValueError(f"Skill '{tool.name}' already exists. You can only add a skill once.")
                names.add(tool.name)
                skills.append(tool)
                continue
            if isinstance(tool, ToolSet):
                # Non-Skill ToolSets are resolved later in _resolve_tools()
                continue
            if not isinstance(tool, Runnable):
                tool = Tool(**tool) if isinstance(tool, dict) else Tool(handler=tool)
            if tool.name in names:
                raise ValueError(f"Tool {tool.name} already exists. You can only add a tool once.")
            names.add(tool.name)
            tool.nest(self._path)
            self.tools[i] = tool

        if skills:
            self._system_prompt_skills = f"""<skills>
Skills provide additional knowledge of a specific topic. The following skills are available:
{chr(10).join([f"- **{skill.name}**: {skill.description}" for skill in skills])}
In skills documentation, you will encounter references to additional files.
If the file is relevant for the user query, USE the `read_skill` tool to get its content.
</skills>"""
            read_skill_tool = ReadSkill(agent_path=self._path, skills=skills)
            read_skill_tool.nest(self._path)
            self.tools.append(read_skill_tool)

        self._is_orchestrator = True
        self._is_coroutine = False
        self._is_gen = False
        self._is_async_gen = True

    @override
    def get_config(self) -> dict[str, Any]:
        """See base class."""
        if self._system_prompt_fn is not None:
            system_prompt = f"<{self._system_prompt_fn.__name__}>"
        else:
            system_prompt = self.system_prompt

        config = self._annotate_config(
            {
                "model": self.model,
                "system_prompt": system_prompt,
                "max_iter": self.max_iter,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "base_url": self.base_url,
                "api_key": self.api_key,
            }
        )

        return {**super().get_config(), **config}

    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"
        self._llm.nest(self._path)
        for tool in self.tools:
            if isinstance(tool, Runnable):
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
        if self.output_model is not None:
            return self.output_model
        return Message

    async def _resolve_system_prompt(self) -> str | None:
        """Resolve system prompt by executing callable or embedded template functions."""
        system_prompt = self.system_prompt

        if self._system_prompt_fn is not None:
            system_prompt = await self._execute_runtime_callable(
                self._system_prompt_fn, self._system_prompt_fn_is_async
            )

        if self._system_prompt_skills:
            if not isinstance(system_prompt, str):
                system_prompt = self._system_prompt_skills
            else:
                system_prompt += "\n\n" + self._system_prompt_skills

        if system_prompt is not None and not isinstance(system_prompt, str):
            logger.warning(
                "Invalid system prompt type after resolution, ignoring...",
                system_prompt=system_prompt,
                system_prompt_type=type(system_prompt).__name__,
            )
            system_prompt = None

        return system_prompt

    def _synthesize_missing_tool_results(self, memory: list[Message]) -> None:
        """Append synthetic error results for any tool_use blocks that were interrupted.

        Mutates memory in-place. Every tool_use the LLM emitted must have a corresponding
        tool_result before the next LLM call, otherwise the API rejects the request.
        Server tool_use (e.g. web_search) gets an inline error block; regular tool_use
        gets a new tool-role message appended.
        """
        model_provider = str(self.model).split("/", 1)[0]
        # Iterate in reverse: if any non-tool_use content follows a server_tool_use,
        # the LLM already continued past it — no synthetic result needed.
        has_followup_after_server_tool_use = False
        for content in memory[-1].content[::-1]:
            if content.type != "tool_use":
                has_followup_after_server_tool_use = True
                continue
            if content.is_server_tool_use:
                if has_followup_after_server_tool_use:
                    continue
                if model_provider == "anthropic":
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
                    logger.warning(
                        "Interrupted server tool use on unsupported provider; skipping synthetic error result.",
                        provider=model_provider,
                        tool_use_id=content.id,
                    )
            else:
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

    async def resolve_memory(self) -> None:
        """Resolve conversation memory from previous agent trace."""
        run_context = get_run_context()
        assert run_context is not None, "Run context not found"

        current_span = run_context.current_span()
        if current_span.memory:
            return
        prompt = Message.validate(current_span.input.get("prompt", ""))
        if prompt.role != "user":
            prompt = Message(role="user", content=prompt.content)
        current_span.memory = [prompt]

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
        if not self_spans:
            return
        if len(self_spans) > 1:
            logger.warning(
                "Multiple spans found for agent path in parent trace; skipping memory resolution.",
                path=self._path,
                span_count=len(self_spans),
            )
            return

        previous_span = self_spans[0]

        if not isinstance(previous_span.memory, list):
            return
        memory = [Message.validate(m) for m in previous_span.memory]

        # Ensure interrupted tool calls have corresponding results before resuming.
        self._synthesize_missing_tool_results(memory)
        current_span.memory = memory + current_span.memory

        # Apply memory compaction if configured and context window utilization warrants it
        if self.memory_compaction is not None:
            should_compact = self.memory_compaction_ratio <= 0.0
            utilization = None
            if not should_compact:
                context_window = get_context_window(str(self.model))
                if context_window is None:
                    # Unknown model — context window not in models.yaml.
                    # Estimate token count from message content as a best effort.
                    estimated_tokens = _estimate_tokens_from_memory(current_span.memory)
                    logger.warning(
                        "Context window unknown for model; token usage estimated from message content (1 token ≈ 4 chars). Compacting as safe fallback.",
                        model=str(self.model),
                        estimated_tokens=estimated_tokens,
                    )
                    should_compact = True
                elif previous_span.usage:
                    prev_input_tokens = sum(
                        v for k, v in previous_span.usage.items() if ":input" in k and "token" in k
                    )
                    prev_output_tokens = sum(
                        v for k, v in previous_span.usage.items() if ":output" in k and "token" in k
                    )
                    utilization = (prev_input_tokens + prev_output_tokens) / context_window
                    should_compact = utilization >= self.memory_compaction_ratio
                else:
                    # Known model but no usage data from previous run.
                    # Estimate utilization from message content.
                    estimated_tokens = _estimate_tokens_from_memory(current_span.memory)
                    utilization = estimated_tokens / context_window
                    logger.warning(
                        "No token usage data from previous run; estimating utilization from message content (1 token ≈ 4 chars).",
                        model=str(self.model),
                        estimated_tokens=estimated_tokens,
                        estimated_utilization=round(utilization, 4),
                    )
                    should_compact = utilization >= self.memory_compaction_ratio

            if should_compact:
                compactors = (
                    [self.memory_compaction]
                    if not isinstance(self.memory_compaction, list)
                    else self.memory_compaction
                )
                compaction_steps = []
                for compactor in compactors:
                    # Set the agent's model on compactors that support it (e.g. summarize)
                    if hasattr(compactor, "_state"):
                        compactor._state["agent_model"] = str(self.model)
                    before = len(current_span.memory)
                    if asyncio.iscoroutinefunction(compactor):
                        current_span.memory = await compactor(current_span.memory)
                    else:
                        current_span.memory = compactor(current_span.memory)
                    compaction_steps.append({
                        "compactor": getattr(compactor, "__name__", repr(compactor)),
                        "before": before,
                        "after": len(current_span.memory),
                    })
                current_span.metadata["compaction"] = {
                    "triggered": True,
                    "utilization": round(utilization, 4) if utilization is not None else None,
                    "steps": compaction_steps,
                }

    async def _resolve_tools(self, i: int) -> tuple[list[Tool], dict[str, Tool]]:
        """Resolve the tools to be provided to the LLM."""
        if i >= self.max_iter:
            return [], {}

        tools = []
        tools_names: set[str] = set()
        commands: dict[str, Tool] = {}

        def _register(tool: Tool) -> None:
            if tool.name in tools_names:
                logger.warning(f"Tool with name '{tool.name}' already exists. You can only add a tool once.")
                return
            tools.append(tool)
            tools_names.add(tool.name)
            if tool.command:
                commands[tool.command] = tool

        for t in self.tools:
            if isinstance(t, ToolSet):
                for tool in await t.resolve():
                    tool.nest(self._path)
                    _register(tool)
            else:
                _register(t)

        if self._bg_tasks:
            bg_tool = Tool(
                name="get_background_task",
                description="Get the status and events of a background task.",
                handler=self.get_background_task,
            )
            bg_tool.nest(self._path)
            _register(bg_tool)

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
                task = asyncio.create_task(consume_tool(tc), context=contextvars.copy_context())
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

        current_span._memory_dump = await dump(current_span.memory)
        # Remove these so the llm runnable doesn't try to use/validate them again
        kwargs.pop("prompt", None)
        kwargs.pop("messages", None)

        async def _append_memory(message: Message, dump_value: Any = None) -> None:
            """Append a message to both memory and its serialized dump in lockstep."""
            current_span.memory.append(message)
            current_span._memory_dump.append(dump_value if dump_value is not None else await dump(message))

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
                    await _append_memory(event.output, dump_value=event._output_dump)

                    if self.output_model is not None:
                        validation_error_msg = None
                        validation_context = self.output_model_context
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
                            await _append_memory(error_feedback)
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
