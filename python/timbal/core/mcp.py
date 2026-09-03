import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, TypeVar

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import httpx
import structlog
from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from pydantic import Field, PrivateAttr, computed_field, model_validator

from ..errors import RunCancelled
from ..state import _suspension_id_for, get_run_context, suspend
from ..types.approval import Cancel
from ..types.content import FileContent, TextContent
from ..types.file import File
from ..types.message import Message
from .runnable import Runnable
from .tool import Tool
from .tool_set import ToolSet

logger = structlog.get_logger("timbal.core.mcp")


ELICITATION_KIND = "mcp_elicitation"
"""``InteractionEvent.kind`` used when an MCP server elicits input mid tool call."""

_ELICIT_ACTIONS = frozenset({"accept", "decline", "cancel"})

# The SDK's streamable HTTP client turns a 404 on a request (server no longer knows our
# Mcp-Session-Id — restarted, or the session expired) into this JSON-RPC error. Not a
# standard code: mcp.types.INVALID_REQUEST is -32600, this is the positive twin.
_SESSION_TERMINATED_CODE = 32600

T = TypeVar("T")

# MCP log levels (RFC 5424) -> structlog method names.
_MCP_LOG_LEVELS = {
    "debug": "debug",
    "info": "info",
    "notice": "info",
    "warning": "warning",
    "error": "error",
    "critical": "error",
    "alert": "error",
    "emergency": "error",
}

_JSON_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
}


def _convert_call_tool_result(tool_name: str, result: mcp_types.CallToolResult) -> Any:
    """Convert an MCP CallToolResult into a value Timbal can feed back to the LLM.

    Text-only results collapse to a plain string, structured results return the
    structured payload, and binary content (images, audio, blob resources) is
    wrapped in Timbal Files inside a Message so the agent forwards them as
    FileContent tool results.
    """
    texts: list[str] = []
    files: list[File] = []
    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            texts.append(block.text)
        elif isinstance(block, mcp_types.ImageContent | mcp_types.AudioContent):
            files.append(File(f"data:{block.mimeType};base64,{block.data}"))
        elif isinstance(block, mcp_types.EmbeddedResource):
            resource = block.resource
            if isinstance(resource, mcp_types.TextResourceContents):
                texts.append(resource.text)
            else:
                mime_type = resource.mimeType or "application/octet-stream"
                files.append(File(f"data:{mime_type};base64,{resource.blob}"))
        elif isinstance(block, mcp_types.ResourceLink):
            texts.append(str(block.uri))

    if result.isError:
        raise RuntimeError("\n".join(texts) or f"MCP tool '{tool_name}' call failed.")

    if files:
        content: list[Any] = [TextContent(text=text) for text in texts]
        content.extend(FileContent(file=file) for file in files)
        return Message(role="assistant", content=content)
    # Per the MCP spec, text blocks carry the canonical serialized result even when
    # structuredContent is present (e.g. FastMCP wraps returns as {"result": ...}),
    # so prefer text and only fall back to the structured payload.
    if len(texts) == 1:
        return texts[0]
    if texts:
        return texts
    return result.structuredContent


def _describe_error(error: BaseException) -> str:
    """``str(error)`` that looks through anyio's ExceptionGroups to the real causes."""
    if isinstance(error, BaseExceptionGroup):
        leaves: list[str] = []
        for sub in error.exceptions:
            leaves.append(_describe_error(sub))
        return "; ".join(leaves) or str(error)
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def _is_session_terminated(error: BaseException) -> bool:
    return (
        isinstance(error, McpError)
        and error.error is not None
        and (error.error.code == _SESSION_TERMINATED_CODE or error.error.message == "Session terminated")
    )


def _is_destructive(mcp_tool: mcp_types.Tool) -> bool:
    """Whether the server marked this tool as destructive.

    Follows the spec defaults: ``readOnlyHint`` wins; otherwise ``destructiveHint``
    defaults to true *when annotations are present*. A tool with no annotations
    at all is unknown, not destructive — gating it would mean gating every tool
    of every server that never bothered to annotate.
    """
    annotations = mcp_tool.annotations
    if annotations is None:
        return False
    if annotations.readOnlyHint:
        return False
    return annotations.destructiveHint is not False


def _validate_elicit_content(content: Any, schema: dict[str, Any] | None) -> str | None:
    """Light client-side check of accepted elicitation content against ``requestedSchema``.

    Elicitation schemas are restricted to flat objects of primitives, so a
    required-keys + primitive-type + enum check covers what the server will
    validate anyway; this just produces a readable error before the round trip.
    Returns an error message, or ``None`` when the content passes.
    """
    if not isinstance(content, dict):
        return f"accepted content must be an object, got {type(content).__name__}"
    if not schema:
        return None
    missing = [key for key in schema.get("required", []) if key not in content]
    if missing:
        return f"missing required field(s): {', '.join(missing)}"
    properties = schema.get("properties") or {}
    for key, value in content.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        enum = prop.get("enum")
        if enum is not None and value not in enum:
            return f"field '{key}' must be one of {enum!r}"
        check = _JSON_TYPE_CHECKS.get(prop.get("type", ""))
        if check is not None and not check(value):
            return f"field '{key}' must be of type {prop['type']}"
    return None


def _coerce_elicit_resume(
    value: Any, schema: dict[str, Any] | None
) -> tuple[mcp_types.ElicitResult | None, str | None]:
    """Turn a ``resume={interaction_id: value}`` value into an ``ElicitResult``.

    - ``None`` / ``False`` -> decline
    - ``True`` -> accept with empty content (pure confirmations)
    - ``{"action": "accept"|"decline"|"cancel", "content": {...}}`` -> explicit
    - any other dict -> accept with that dict as content

    Returns ``(result, error)``; exactly one is set.
    """
    if value is None or value is False:
        return mcp_types.ElicitResult(action="decline"), None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        return None, f"unsupported resume value type {type(value).__name__}; expected a dict, bool, None, or Cancel"

    if set(value) <= {"action", "content"} and value.get("action") in _ELICIT_ACTIONS:
        action = value["action"]
        if action != "accept":
            return mcp_types.ElicitResult(action=action), None
        content = value.get("content", {})
    else:
        content = value

    if schema is None:
        # URL mode: accept means "the user agreed to open the URL"; no content is sent.
        return mcp_types.ElicitResult(action="accept"), None
    error = _validate_elicit_content(content, schema)
    if error is not None:
        return None, error
    return mcp_types.ElicitResult(action="accept", content=content), None


@dataclass
class _InflightCall:
    """Per ``tools/call`` state shared between the handler task and the SDK receive loop.

    Server -> client requests (elicitation, sampling) arrive on the session's
    receive loop, which runs outside the calling tool's task and contextvars.
    The handler registers one of these before ``call_tool`` so the callbacks can
    find the run context / span to attribute the request to, and leave the
    outcome here for the handler to act on once ``call_tool`` returns.
    """

    key: int
    tool_name: str
    path: str | None
    tool_call_id: str | None
    run_context: Any
    ctx: contextvars.Context
    pending_payload: dict[str, Any] | None = None
    pending_schema: dict[str, Any] | None = None
    cancel: Cancel | None = None
    invalid_resume: str | None = None
    ambiguous_with: int = 0


class MCPTool(Tool):
    """A Tool whose parameter schema comes from an MCP server instead of handler introspection.

    The handler is a ``**kwargs`` passthrough that forwards the arguments to the
    MCP session, so ``params_model`` accepts anything; the schema the LLM sees is
    the server-declared JSON schema.
    """

    input_schema: dict[str, Any] = Field(default_factory=dict)
    """The JSON schema declared by the MCP server for this tool's arguments."""

    title: str | None = None
    """Human-readable title declared by the server (``Tool.title`` or ``annotations.title``)."""

    tool_annotations: dict[str, Any] | None = None
    """Server-declared ``ToolAnnotations`` (readOnlyHint, destructiveHint, ...) as a plain dict."""

    @override
    @computed_field
    @cached_property
    def params_model_schema(self) -> dict[str, Any]:
        """See base class."""
        schema = dict(self.input_schema)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema


class MCPServer(ToolSet):
    """A single MCP server connection that resolves its tools at runtime.

    Each MCPServer instance represents one MCP server. Configure the
    transport type and its required parameters:

        MCPServer(transport="stdio", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])
        MCPServer(transport="http", url="https://api.timbal.ai/mcp")

    Protocol features wired into the run:

    - **Elicitation** (``elicitation=True``): a server calling ``elicit()`` mid
      tool call pauses the run exactly like :func:`timbal.state.suspend` — the
      run ends ``input_required`` with an ``InteractionEvent`` of kind
      ``"mcp_elicitation"``; resume with ``{interaction_id: value}`` and the tool
      is re-invoked with the answer. The server's tool must be idempotent up to
      the elicit point and its message deterministic for identical input.
    - **Approval** (``approval=``): gate tools through Timbal's approval flow
      based on the server's ``ToolAnnotations`` — client-side policy a server
      cannot skip.
    - **tools/list_changed**: marks the cached tool list stale; it is refreshed
      on the *next run*, never mid-run (a mid-run swap would bust the prompt cache).
    - **Logging**: server ``notifications/message`` land in structlog.
    - **Sampling** (``sampling_model=``): ``sampling/createMessage`` requests are
      served with the given Timbal model string.
    """

    name: str | None = None
    """Optional identifier for this server.

    When set, each resolved tool is exposed to the agent as
    ``{name}__{tool}`` so two servers that declare the same bare tool name
    don't collide in the agent's flat registry. The bare name is still used
    for ``session.call_tool``. Also used by codegen (``remove-tool --name``).
    """

    transport: Literal["stdio", "http"]

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    timeout: float | None = None
    """Seconds to wait for the server to answer a request on an established session
    (``tools/list``, ``tools/call``). ``None`` (default) waits forever, which is what MCP
    tools that legitimately run for minutes need; set it for servers that may accept a
    request and never answer. A dead transport is detected independently of this and
    fails the call immediately. Does not cover connecting — see ``connect_timeout``."""

    connect_timeout: float | None = None
    """Seconds to wait for the transport to come up and ``initialize`` to complete. Kept
    separate from ``timeout`` because a stdio server's cold start (interpreter + imports)
    is routinely longer than a sensible per-request bound. ``None`` (default) waits forever."""

    elicitation: bool = True
    """Advertise the ``elicitation`` capability and bridge server ``elicit()`` calls to
    ``suspend()``. Set False to hide the capability (servers then must fail closed
    on tools that need confirmation)."""

    approval: Literal["destructive", "all"] | Callable[[mcp_types.Tool], bool] | None = None
    """Route tools through the human-approval gate (``requires_approval``).

    - ``"destructive"``: tools whose annotations say destructive (``destructiveHint``
      not false and ``readOnlyHint`` not true). Unannotated tools are not gated.
    - ``"all"``: every tool.
    - callable: receives the server's ``mcp.types.Tool``; return True to gate.
    - ``None`` (default): no gating.
    """

    sampling_model: Any = None
    """Timbal model string (e.g. ``"anthropic/claude-haiku-4-5"``) or a ``TestModel`` used
    to answer the server's ``sampling/createMessage`` requests. ``None`` leaves sampling
    unsupported. Note the SDK answers server requests inline on the session's receive
    loop, so other traffic on this session waits while the LLM call runs."""

    _session: Any | None = PrivateAttr(default=None)
    _session_task: asyncio.Task | None = PrivateAttr(default=None)
    _session_closed: asyncio.Event | None = PrivateAttr(default=None)
    _tools_cache: list[Runnable] | None = PrivateAttr(default=None)
    _tools_cache_run_id: str | None = PrivateAttr(default=None)
    _tools_stale: bool = PrivateAttr(default=False)
    _lock: asyncio.Lock | None = PrivateAttr(default=None)
    _tools_lock: asyncio.Lock | None = PrivateAttr(default=None)
    _inflight: dict[int, _InflightCall] = PrivateAttr(default_factory=dict)
    _inflight_seq: int = PrivateAttr(default=0)

    def _get_lock(self) -> asyncio.Lock:
        """Session lock: guards open/reset/close of the transport."""
        # Lazily create so model construction doesn't require a running loop.
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_tools_lock(self) -> asyncio.Lock:
        """Tool-cache lock: one ``tools/list`` per refresh. Always taken *before* the session lock."""
        if self._tools_lock is None:
            self._tools_lock = asyncio.Lock()
        return self._tools_lock

    @property
    def _label(self) -> str:
        return self.name or self.url or self.command or self.transport

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "MCPServer":
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("'command' is required for stdio transport")
        elif self.transport == "http":
            if not self.url:
                raise ValueError("'url' is required for http transport")
        return self

    # ------------------------------------------------------------------ session

    def _client_session(self, read: Any, write: Any) -> ClientSession:
        return ClientSession(
            read,
            write,
            elicitation_callback=self._on_elicitation if self.elicitation else None,
            sampling_callback=self._on_sampling if self.sampling_model else None,
            logging_callback=self._on_log,
            message_handler=self._on_message,
        )

    @asynccontextmanager
    async def _connect_stdio(self):
        assert self.command is not None
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env if self.env else None,
        )
        async with stdio_client(server_params) as (read, write):
            async with self._client_session(read, write) as session:
                await session.initialize()
                logger.info("Connected to MCP server via stdio", command=self.command)
                yield session

    @asynccontextmanager
    async def _connect_http(self):
        assert self.url is not None
        async with httpx.AsyncClient(headers=self.headers if self.headers else None) as http_client:
            async with streamable_http_client(self.url, http_client=http_client) as (read, write, _):
                async with self._client_session(read, write) as session:
                    await session.initialize()
                    logger.info("Connected to MCP server via http", url=self.url)
                    yield session

    async def _run_session(self, ready: asyncio.Future, closed: asyncio.Event) -> None:
        """Own the transport for its whole life.

        The SDK transports and ``ClientSession`` are anyio task groups / cancel
        scopes, which must be entered and exited by the *same* task, in LIFO
        order. Splitting ``__aenter__``/``__aexit__`` across whatever tasks happen
        to call ``_connect()``/``close()`` breaks as soon as two servers are opened
        in one task and closed in the other order, or closed from another task.
        Running the whole ``async with`` here makes ``close()`` order-independent.
        """
        connect = self._connect_stdio() if self.transport == "stdio" else self._connect_http()
        try:
            async with connect as session:
                if not ready.done():
                    ready.set_result(session)
                await closed.wait()
        except BaseException as e:
            if not ready.done():
                # Connect failed: hand the error to the opener and end quietly.
                if isinstance(e, asyncio.CancelledError):
                    ready.cancel()
                else:
                    # The raw failure is an anyio ExceptionGroup whose str() is
                    # "unhandled errors in a TaskGroup" — useless to the LLM and to logs.
                    wrapped = ConnectionError(f"MCP server '{self._label}' failed to connect: {_describe_error(e)}")
                    wrapped.__cause__ = e
                    ready.set_exception(wrapped)
                return
            raise

    def _forget_session(self) -> None:
        """Drop references to the current session without closing anything (the owner is already done)."""
        self._session = None
        self._session_task = None
        self._session_closed = None

    def _on_session_task_done(self, task: asyncio.Task) -> None:
        """Forget a session whose owner task ended on its own (transport died).

        Retrieves the exception so asyncio does not log it as never-retrieved,
        and resets ``_session`` so the next ``_connect()`` reopens the transport.
        The tool cache stays valid: tool handlers reconnect through ``self``.
        """
        if task is not self._session_task:
            return
        error = None if task.cancelled() else task.exception()
        if self._session_closed is not None and self._session_closed.is_set():
            return  # close() / _reset_session() drive the state reset on that path
        logger.warning(
            "MCP session ended unexpectedly; reconnecting on next use",
            server=self.name,
            error=_describe_error(error) if error else None,
        )
        self._forget_session()

    async def _shutdown_owner(self, task: asyncio.Task, closed: asyncio.Event) -> None:
        closed.set()
        # wait() rather than await: the owner task's own cancellation must not
        # surface here as if the caller itself had been cancelled.
        await asyncio.wait({task})
        error = None if task.cancelled() else task.exception()
        if error is not None:
            logger.error("Error closing MCP connection", server=self.name, error=_describe_error(error))

    async def _reset_session(self, task: asyncio.Task | None) -> None:
        """Tear down the session owned by ``task`` (if it is still current) so the next ``_connect()`` reopens.

        Used when the server tells us it no longer knows our session: the owner task is
        alive and healthy, the transport is fine, but every request would 404 forever.
        """
        async with self._get_lock():
            if task is None or self._session_task is not task:
                return  # somebody else already replaced or dropped it
            closed = self._session_closed
            self._forget_session()
            if closed is not None:
                await self._shutdown_owner(task, closed)

    async def _open_session(self) -> ClientSession:
        """Open the transport in its own owner task. Caller must hold ``_get_lock()`` and check ``_session`` first."""
        loop = asyncio.get_running_loop()
        ready: asyncio.Future = loop.create_future()
        closed = asyncio.Event()
        # A fresh context: the owner task (and the SDK receive loop inside it) must not
        # pin the opener's RunContext for the life of the connection. Callbacks find
        # the current run through ``_inflight``, not contextvars.
        task = loop.create_task(
            self._run_session(ready, closed),
            name=f"mcp-session:{self.name or self.transport}",
            context=contextvars.Context(),
        )
        try:
            if self.connect_timeout is not None:
                done, _ = await asyncio.wait({ready}, timeout=self.connect_timeout)
                if not done:
                    # The owner is still inside the transport / initialize(); only cancelling
                    # it can unwind that. _run_session() turns the cancel into ready.cancel().
                    task.cancel()
                    await asyncio.wait({task})
                    raise ConnectionError(
                        f"MCP server '{self._label}' failed to connect: timed out after {self.connect_timeout}s"
                    )
            session = await ready
        except BaseException:
            closed.set()
            await asyncio.wait({task})
            raise
        self._session = session
        self._session_task = task
        self._session_closed = closed
        task.add_done_callback(self._on_session_task_done)
        return session

    def _session_alive(self) -> bool:
        return self._session is not None and self._session_task is not None and not self._session_task.done()

    async def _connect(self) -> ClientSession:
        """Establish connection and store session for reuse.

        Synchronized so parallel tool calls (agent multiplexing) don't each
        open a duplicate stdio subprocess / HTTP session and orphan the first.
        A session whose owner task has already finished (transport died, done
        callback not yet run) counts as absent and is reopened.
        """
        if self._session_alive():
            return self._session  # type: ignore[return-value]

        async with self._get_lock():
            if self._session_alive():
                return self._session  # type: ignore[return-value]
            if self._session_task is not None and self._session_task.done():
                self._forget_session()
            return await self._open_session()

    async def _request(
        self,
        send: Callable[[ClientSession], Awaitable[T]],
        *,
        what: str,
        idempotent: bool = False,
    ) -> T:
        """Run one request against the live session, with the failure modes the SDK leaves to us.

        - **Dead transport mid-request**: the SDK never fails the pending future when the
          receive loop dies, so the request is raced against the owner task and turned
          into a ``ConnectionError`` instead of hanging forever.
        - **Stale session** (``Session terminated``: the server 404s our ``Mcp-Session-Id``
          after a restart or expiry): the session is reset and the request retried once.
          Safe for any request — a 404 means the server never saw it.
        - **No answer within** ``timeout`` (server alive, never replies): ``TimeoutError``.
          The session stays up; the server may still be running the tool.
        - ``idempotent=True`` also retries once after a mid-request connection loss
          (``tools/list``); ``tools/call`` never does, the tool may have run.
        """
        for attempt in (1, 2):
            session = await self._connect()
            owner = self._session_task
            request: asyncio.Future = asyncio.ensure_future(send(session))
            try:
                waiting = {request, owner} if owner is not None else {request}
                done, _ = await asyncio.wait(waiting, timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED)
            finally:
                if not request.done():
                    request.cancel()
            if request not in done:
                if not done:
                    raise TimeoutError(f"MCP server '{self._label}': {what} timed out after {self.timeout}s")
                if idempotent and attempt == 1:
                    logger.warning("MCP connection lost; retrying", server=self.name, what=what)
                    continue
                raise ConnectionError(f"MCP server '{self._label}': connection lost during {what}")
            try:
                return request.result()
            except McpError as e:
                if attempt == 1 and _is_session_terminated(e):
                    logger.warning(
                        "MCP server no longer knows our session; reconnecting",
                        server=self.name,
                        what=what,
                    )
                    await self._reset_session(owner)
                    continue
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    # ---------------------------------------------------------- server callbacks

    async def _on_message(
        self,
        message: Any,
    ) -> None:
        if isinstance(message, Exception):
            logger.warning("MCP session error", server=self.name, error=str(message))
            return
        if isinstance(message, mcp_types.ServerNotification) and isinstance(
            message.root, mcp_types.ToolListChangedNotification
        ):
            # Never swap tools mid-run: resolve() only refetches once the run id changes.
            self._tools_stale = True
            logger.info(
                "MCP server reported tools/list_changed; tool list refreshes on the next run",
                server=self.name,
            )

    async def _on_log(self, params: mcp_types.LoggingMessageNotificationParams) -> None:
        method = getattr(logger, _MCP_LOG_LEVELS.get(params.level, "info"))
        method("MCP server log", server=self.name, level=params.level, logger=params.logger, data=params.data)

    def _elicitation_payload(
        self, call: _InflightCall, params: mcp_types.ElicitRequestParams
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Build the ``InteractionEvent`` payload (and ``response_schema``) for an elicitation.

        The payload doubles as the suspension-id input, so it must be plain JSON and
        contain only what the server sent plus stable identifiers.
        """
        dumped = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        payload: dict[str, Any] = {
            "server": self.name,
            "tool": call.tool_name,
            "mode": dumped.get("mode") or "form",
            "message": dumped.get("message", ""),
        }
        if isinstance(params, mcp_types.ElicitRequestURLParams):
            payload["url"] = dumped.get("url")
            payload["elicitation_id"] = dumped.get("elicitationId")
            return payload, None
        requested_schema = dumped.get("requestedSchema") or {}
        payload["requested_schema"] = requested_schema
        return payload, requested_schema

    async def _on_elicitation(
        self,
        _context: Any,
        params: mcp_types.ElicitRequestParams,
    ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
        """Answer a server ``elicitation/create`` from resume values, or arrange a suspend.

        Runs on the session's receive loop, so it cannot raise ``Suspend`` itself:
        it records the request on the in-flight call and returns ``cancel`` so the
        server unwinds; the handler then raises ``Suspend`` once ``call_tool`` returns.
        """
        calls = list(self._inflight.values())
        if len(calls) != 1:
            # Elicitation requests carry nothing that ties them to a tools/call, so with
            # several calls in flight on one session we cannot know which one asked.
            for call in calls:
                call.ambiguous_with = len(calls)
            logger.warning(
                "MCP elicitation could not be attributed to a single tool call",
                server=self.name,
                inflight=len(calls),
            )
            return mcp_types.ElicitResult(action="cancel")

        call = calls[0]
        payload, response_schema = self._elicitation_payload(call, params)
        if call.run_context is None or call.path is None:
            logger.warning("MCP elicitation outside of a run context; declining", server=self.name, tool=call.tool_name)
            return mcp_types.ElicitResult(action="decline")

        suspension_id = _suspension_id_for(call.path, payload, call.tool_call_id)
        resume_values = call.run_context._resume_values
        if suspension_id in resume_values:
            call.run_context._used_resume_ids.add(suspension_id)
            value = resume_values[suspension_id]
            if isinstance(value, Cancel):
                call.cancel = value
                return mcp_types.ElicitResult(action="cancel")
            result, error = _coerce_elicit_resume(value, response_schema)
            if error is not None:
                call.invalid_resume = error
                return mcp_types.ElicitResult(action="cancel")
            assert result is not None
            return result

        call.pending_payload = payload
        call.pending_schema = response_schema
        return mcp_types.ElicitResult(action="cancel")

    async def _on_sampling(
        self,
        _context: Any,
        params: mcp_types.CreateMessageRequestParams,
    ) -> mcp_types.CreateMessageResult | mcp_types.ErrorData:
        """Serve ``sampling/createMessage`` with ``sampling_model`` via the LLM router.

        When exactly one tool call is in flight the LLM call runs inside that
        call's contextvars, so its span and usage land under the MCP tool in the trace.
        """
        if not self.sampling_model:
            return mcp_types.ErrorData(code=mcp_types.INVALID_REQUEST, message="Sampling not supported")
        calls = list(self._inflight.values())
        ctx = calls[0].ctx if len(calls) == 1 else None
        try:
            coro = self._sample(params)
            if ctx is not None:
                text = await asyncio.create_task(coro, context=ctx)
            else:
                text = await coro
        except Exception as e:  # noqa: BLE001 — the server must get a JSON-RPC error, not a dropped request
            logger.warning("MCP sampling request failed", server=self.name, error=str(e))
            return mcp_types.ErrorData(code=mcp_types.INTERNAL_ERROR, message=str(e))
        return mcp_types.CreateMessageResult(
            role="assistant",
            content=mcp_types.TextContent(type="text", text=text),
            model=self.sampling_model if isinstance(self.sampling_model, str) else type(self.sampling_model).__name__,
            stopReason="endTurn",
        )

    async def _sample(self, params: mcp_types.CreateMessageRequestParams) -> str:
        from .llm import _llm_router

        messages: list[Message] = []
        for sampling_message in params.messages:
            blocks = sampling_message.content
            if not isinstance(blocks, list):
                blocks = [blocks]
            content: list[Any] = []
            for block in blocks:
                if isinstance(block, mcp_types.TextContent):
                    content.append(TextContent(text=block.text))
                elif isinstance(block, mcp_types.ImageContent | mcp_types.AudioContent):
                    content.append(FileContent(file=File(f"data:{block.mimeType};base64,{block.data}")))
            if content:
                messages.append(Message(role=sampling_message.role, content=content))

        llm = Tool(
            name="mcp_sampling",
            handler=_llm_router,
            record_default_request_usage=False,
            metadata={"type": "LLM"},
        )
        result = await llm(
            model=self.sampling_model,
            system_prompt=params.systemPrompt,
            messages=messages,
            max_tokens=params.maxTokens,
            temperature=params.temperature,
        ).collect()
        if result.status.code != "success":
            message = (result.error or {}).get("message") if isinstance(result.error, dict) else None
            raise RuntimeError(message or f"sampling with {self.sampling_model} failed")
        output = result.output
        return output.collect_text() if isinstance(output, Message) else str(output)

    # -------------------------------------------------------------------- tools

    def _begin_call(self, tool_name: str) -> _InflightCall:
        run_context = get_run_context()
        path: str | None = None
        tool_call_id: str | None = None
        if run_context is not None:
            try:
                span = run_context.current_span()
                path = span.path
                tool_call_id = (span.metadata or {}).get("tool_call_id")
            except RuntimeError:
                pass
        self._inflight_seq += 1
        call = _InflightCall(
            key=self._inflight_seq,
            tool_name=tool_name,
            path=path,
            tool_call_id=tool_call_id,
            run_context=run_context,
            ctx=contextvars.copy_context(),
        )
        self._inflight[call.key] = call
        return call

    def _end_call(self, call: _InflightCall) -> None:
        self._inflight.pop(call.key, None)

    def _qualified_tool_name(self, tool_name: str) -> str:
        """Name exposed to the agent/LLM for an MCP tool.

        Prefix with ``{server}__`` when this server has a ``name``, so multiple
        MCPServer instances can coexist without their tools clobbering each
        other in the agent's flat registry. Without a server name the bare
        MCP tool name is kept (fine for a single unnamed server).
        """
        if self.name:
            return f"{self.name}__{tool_name}"
        return tool_name

    def _needs_approval(self, mcp_tool: mcp_types.Tool) -> bool:
        if self.approval is None:
            return False
        if self.approval == "all":
            return True
        if self.approval == "destructive":
            return _is_destructive(mcp_tool)
        return bool(self.approval(mcp_tool))

    def _make_tool(self, mcp_tool: mcp_types.Tool) -> MCPTool:
        # Bare name for the wire call; qualified name for the agent registry.
        bare_name = mcp_tool.name
        exposed_name = self._qualified_tool_name(bare_name)
        description = mcp_tool.description or ""
        if self.name and description:
            description = f"[{self.name}] {description}"
        elif self.name:
            description = f"[{self.name}] {bare_name}"

        annotations = mcp_tool.annotations.model_dump(exclude_none=True) if mcp_tool.annotations else None
        title = mcp_tool.title or (annotations or {}).get("title")

        async def _handler(**kwargs: Any) -> Any:
            call = self._begin_call(bare_name)
            try:
                result = await self._request(
                    lambda session: session.call_tool(bare_name, arguments=kwargs),
                    what=f"tools/call {bare_name}",
                )
            finally:
                self._end_call(call)

            if call.cancel is not None:
                raise RunCancelled(call.cancel.reason or "Run cancelled by user.")
            if call.invalid_resume is not None:
                raise ValueError(f"Invalid resume value for MCP elicitation on '{exposed_name}': {call.invalid_resume}")
            if call.pending_payload is not None:
                # Raises Suspend: the callback already checked the same id against the resume values.
                suspend(call.pending_payload, kind=ELICITATION_KIND, response_schema=call.pending_schema)
            if call.ambiguous_with and result.isError:
                raise RuntimeError(
                    f"MCP tool '{exposed_name}' asked for user input, but {call.ambiguous_with} tool calls were in "
                    f"flight on this server so the request could not be attributed to one of them. "
                    "Call this tool on its own (not in parallel with other tools) and try again."
                )
            return _convert_call_tool_result(bare_name, result)

        requires_approval = self._needs_approval(mcp_tool)
        return MCPTool(
            name=exposed_name,
            description=description,
            handler=_handler,
            input_schema=mcp_tool.inputSchema or {},
            title=title,
            tool_annotations=annotations,
            requires_approval=requires_approval,
            approval_prompt=f"Run {title or exposed_name}?" if requires_approval else None,
            approval_description=(mcp_tool.description or None) if requires_approval else None,
            approval_kind="mcp_tool" if requires_approval else None,
            approval_ui=(
                {"server": self.name, "tool": bare_name, "title": title, "annotations": annotations}
                if requires_approval
                else None
            ),
        )

    def _tools_refresh_due(self, run_id: str | None) -> bool:
        """A stale list is only refetched at a run boundary, so a run never sees the tool set change."""
        if self._tools_cache is None:
            return True
        if not self._tools_stale:
            return False
        return run_id is None or run_id != self._tools_cache_run_id

    async def resolve(self) -> list[Runnable]:
        """See base class.

        Lists the server's tools once and caches them until ``close()``. A
        ``tools/list_changed`` notification marks the cache stale, but the refetch
        waits for the next run (different run id) so the tool set stays fixed for
        the whole run (prompt-cache friendly, no mid-run surprises). Tools are
        sorted by name so a server that lists in nondeterministic order does not
        churn the LLM's tools prefix.
        """
        run_context = get_run_context()
        run_id = run_context.id if run_context is not None else None
        if not self._tools_refresh_due(run_id):
            assert self._tools_cache is not None
            return self._tools_cache

        async with self._get_tools_lock():
            if not self._tools_refresh_due(run_id):
                assert self._tools_cache is not None
                return self._tools_cache

            result = await self._request(lambda session: session.list_tools(), what="tools/list", idempotent=True)
            mcp_tools = sorted(result.tools, key=lambda t: t.name)
            tools: list[Runnable] = [self._make_tool(mcp_tool) for mcp_tool in mcp_tools]
            logger.info("Resolved MCP tools", server=self.name, tools=[t.name for t in tools])
            self._tools_cache = tools
            self._tools_cache_run_id = run_id
            self._tools_stale = False
            return tools

    async def close(self) -> None:
        """Close the MCP server connection. Safe from any task, in any order."""
        async with self._get_tools_lock(), self._get_lock():
            task, closed = self._session_task, self._session_closed
            self._forget_session()
            if task is not None and closed is not None:
                await self._shutdown_owner(task, closed)
            self._tools_cache = None
            self._tools_cache_run_id = None
            self._tools_stale = False
            self._inflight.clear()

    def __del__(self):
        session = getattr(self, "_session", None)
        if session:
            logger.warning("MCPServer deleted without calling close()")
