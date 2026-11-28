import asyncio
import contextlib
import contextvars
import functools
import hashlib
import hmac
import inspect
import itertools
import json
import os
import re
import secrets
import string
import types
import unicodedata
import weakref
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Literal, TypeVar
from urllib.parse import urlsplit

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
from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_validator

from ..errors import RunCancelled
from ..state import _suspension_id_for, get_call_id, get_run_context, suspend
from ..state.background import current_background_task_id
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

# ``tools/call`` ``_meta`` keys naming the calling run and the tool's span (``MCPServer.provenance``).
PROVENANCE_RUN_KEY = "timbal.ai/run_id"
PROVENANCE_CALL_KEY = "timbal.ai/call_id"
PROVENANCE_PARENT_KEY = "timbal.ai/parent_run_id"
PROVENANCE_SESSION_KEY = "timbal.ai/session_id"
# The platform's server only: the app and revision the run belongs to, and the calling tool span's path.
PROVENANCE_APP_KEY = "timbal.ai/app_id"
PROVENANCE_REV_KEY = "timbal.ai/rev"
PROVENANCE_PATH_KEY = "timbal.ai/path"

# What the platform's ``_meta`` reader takes (monolith ``src/mcp/meta.rs``). It refuses the whole tool call (a
# 400) for a claim it cannot read and for any value shaped like a credential, so a value that would not pass is
# left out here instead: a call never fails because of its provenance, it goes through unlinked.
_CLAIM_ID_MAX = 128
_CLAIM_ID_CHARS = frozenset(string.ascii_letters + string.digits + "-_.:")
_CLAIM_REV_MAX = 255
_CLAIM_PATH_MAX = 512
_CLAIM_APP_ID_MAX = 2**63 - 1  # the platform reads it as an i64
_API_KEY_PREFIX = re.compile(r"t\+?[0-9]+_")
_JWT_CHARS = frozenset(string.ascii_letters + string.digits + "-_=")

_ELICIT_ACTIONS = frozenset({"accept", "decline", "cancel"})

# The SDK's streamable HTTP client turns a 404 on a request (server no longer knows our
# Mcp-Session-Id — restarted, or the session expired) into this JSON-RPC error. Not a
# standard code: mcp.types.INVALID_REQUEST is -32600, this is the positive twin.
_SESSION_TERMINATED_CODE = 32600

DEFAULT_TIMEOUT_SECONDS = 600.0
"""Default per-request bound (``tools/list``, ``tools/call``). Ten minutes is past any tool
that legitimately runs long (a full test run, a document ingest) while still turning a
server that accepted a request and will never answer into an error instead of a run that
sits open until someone restarts the process. ``timeout=None`` opts out explicitly."""

T = TypeVar("T")

_USE_SERVER_TIMEOUT: Any = object()
"""Sentinel for ``_request(timeout=...)``: fall back to ``MCPServer.timeout``."""

_AMBIGUOUS_ELICITATION_NOTE = (
    "[timbal] While this call was in flight, the MCP server asked for user input that could not be "
    "attributed to one of the {n} tool calls running in parallel on this server, so the question was "
    "answered with `cancel` on the tool's behalf. If this result reflects a declined or cancelled action, "
    "call this tool on its own (not in parallel with other tools) and try again."
)

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


def _looks_like_a_credential(value: str) -> bool:
    """A value the platform takes for a credential in ``_meta``: an ``Authorization`` value (``Bearer`` /
    ``Basic``), a Timbal API key (``t<n>_`` with a random tail of 32 or more letters and digits) or a JWT."""
    value = value.strip()
    if value[:7].lower() == "bearer " or value[:6].lower() == "basic ":
        return True
    tail = value.rsplit("_", 1)[-1]
    if len(tail) >= 32 and tail.isascii() and tail.isalnum() and _API_KEY_PREFIX.match(value):
        return True
    parts = value.split(".")
    return len(parts) == 3 and value.startswith("eyJ") and all(part and set(part) <= _JWT_CHARS for part in parts)


def _claim_id(value: Any) -> str | None:
    """``value`` if the platform reads it as an id (1 to 128 letters, digits, ``-``, ``_``, ``.``, ``:``)."""
    if (
        isinstance(value, str)
        and 0 < len(value) <= _CLAIM_ID_MAX
        and set(value) <= _CLAIM_ID_CHARS
        and not _looks_like_a_credential(value)
    ):
        return value
    return None


def _claim_text(value: Any, limit: int) -> str | None:
    """``value`` if the platform reads it as free text (``rev``, ``path``): not blank, at most ``limit``
    characters, no control character. Spaces and any other Unicode are fine: agent names are free text. Not a
    lone surrogate (a ``str`` can hold one, read with ``surrogateescape``): no UTF-8 request can carry it."""
    if (
        isinstance(value, str)
        and value.strip()
        and len(value) <= limit
        and not any(unicodedata.category(c) in ("Cc", "Cs") for c in value)
        and not _looks_like_a_credential(value)
    ):
        return value
    return None


def _claim_app_id(value: Any) -> int | None:
    """``value`` as the positive integer the platform reads for ``app_id`` (sent as a JSON number)."""
    if isinstance(value, str):
        digits = value.lstrip("0")
        if not (value.isascii() and value.isdigit()) or len(digits) > 19:
            return None  # (and int() parses no more than 4,300 digits from text)
        value = int(digits or "0")
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= _CLAIM_APP_ID_MAX:
        return value
    return None


def _deployed_rev(subject: Any) -> Any:
    """The revision a run's code is: the subject's, else ``TIMBAL_PROJECT_REV``, which the platform's runtime sets
    (the config loader reads ``TIMBAL_REV`` only), for a run of the app this process was deployed for alone
    (``TIMBAL_APP_ID``): another app's run is not this deployment's code. Both ids are read as the ``app_id``
    claim reads them (``_claim_app_id``: ``"001829"`` is app 1829), and neither may be one the claim would leave
    out."""
    rev = getattr(subject, "rev", None)
    if rev:
        return rev
    app = _claim_app_id(getattr(subject, "app_id", None))
    if app is not None and app == _claim_app_id(os.getenv("TIMBAL_APP_ID")):
        return os.getenv("TIMBAL_PROJECT_REV")
    return None


def _platform_claims(run_context: Any, call_id: str | None) -> dict[str, Any]:
    """The ``_meta`` claims for the platform's server only: the app and revision the run belongs to
    (``platform_config.subject``), and the calling tool span's path (the agent that made the call is the
    path without its last segment). Each is None when the platform would refuse its value."""
    subject = getattr(run_context.platform_config, "subject", None)
    span = run_context._trace.get(call_id) if call_id else None
    return {
        PROVENANCE_APP_KEY: _claim_app_id(getattr(subject, "app_id", None)),
        PROVENANCE_REV_KEY: _claim_text(_deployed_rev(subject), _CLAIM_REV_MAX),
        PROVENANCE_PATH_KEY: _claim_text(getattr(span, "path", None), _CLAIM_PATH_MAX),
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


def _with_ambiguity_note(converted: Any, inflight: int) -> Any:
    """Append the auto-cancelled-elicitation note to a text result; leave other shapes alone."""
    note = _AMBIGUOUS_ELICITATION_NOTE.format(n=inflight)
    if isinstance(converted, str):
        return f"{converted}\n\n{note}"
    if isinstance(converted, list) and all(isinstance(item, str) for item in converted):
        return [*converted, note]
    return converted


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


def _approval_policy_arity(policy: Callable[..., Any]) -> int:
    """How many positional arguments an ``approval=`` callable takes: 1 (``tool``) or 2 (``tool, arguments``).

    Anything that cannot be introspected (builtins, some callables with ``__call__``) is
    treated as the one-argument form, which is the documented original.
    """
    try:
        params = list(inspect.signature(policy).parameters.values())
    except (TypeError, ValueError):
        return 1
    positional = [
        p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return 2
    return 2 if len(positional) >= 2 else 1


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
    abort: asyncio.Future
    """Resolved with an exception by a callback that knows the call can never complete
    (e.g. the server asked us something on a connection that cannot carry the answer).
    The handler races ``call_tool`` against it instead of waiting forever."""
    pending_payload: dict[str, Any] | None = None
    pending_schema: dict[str, Any] | None = None
    cancel: Cancel | None = None
    invalid_resume: str | None = None
    ambiguous_with: int = 0


PROVENANCE_SECRET_ENV = "TIMBAL_PROVENANCE_SECRET"
_PROCESS_SECRET = secrets.token_bytes(32)


def _pseudonym(origin: str, value: str) -> str:
    """What a third-party server sees for one of our ids: stable for that server, unlinkable across servers.

    Keyed by ``TIMBAL_PROVENANCE_SECRET`` (set it for pseudonyms that hold across processes), else by a
    random per-process key."""
    secret = os.environ.get(PROVENANCE_SECRET_ENV)
    # surrogatepass: a command line read with surrogateescape still hashes, and the call never fails here.
    key = secret.encode("utf-8", "surrogatepass") if secret else _PROCESS_SECRET
    return hmac.new(key, f"{origin}\n{value}".encode("utf-8", "surrogatepass"), hashlib.sha256).hexdigest()[:32]


def _ambient_trace_context() -> dict[str, str]:
    """``traceparent`` / ``tracestate`` of the caller's active OpenTelemetry span, when there is one."""
    try:
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    except ImportError:
        return {}
    if not trace.get_current_span().get_span_context().is_valid:
        return {}
    carrier: dict[str, str] = {}
    # The W3C format itself, not the global propagator: one configured for B3 writes no traceparent.
    TraceContextTextMapPropagator().inject(carrier)
    return {key: carrier[key] for key in ("traceparent", "tracestate") if carrier.get(key)}


def _timbal_trace_context(run_id: str, call_id: str | None) -> dict[str, str]:
    """A ``traceparent`` naming the tool's span in the trace ``OTelExporter`` exports for the run."""
    if not call_id:
        return {}
    from ..state.tracing.exporters.otel import OTelExporter

    try:
        return {"traceparent": f"00-{OTelExporter._trace_id(run_id)}-{OTelExporter._span_id(call_id)}-01"}
    except UnicodeError:  # a run id given with a lone surrogate (``context.id``): no trace, never a failed call
        return {}


# W3C Trace Context as the platform keeps it (monolith ``src/mcp/meta.rs``): it drops a ``traceparent`` of another
# shape, with its ``tracestate``.
_TRACEPARENT = re.compile(r"00-(?!0{32})[0-9a-f]{32}-(?!0{16})[0-9a-f]{16}-[0-9a-f]{2}")
_TRACESTATE_MAX = 512


def _trace_claims(context: Mapping[str, Any]) -> dict[str, str]:
    """The ``traceparent`` / ``tracestate`` of ``context`` that are sent: a version-``00`` ``traceparent``, and with it
    a ``tracestate`` of at most 512 printable ASCII characters (or tabs). Neither may look like a credential: the
    platform answers a whole call with one anywhere in its ``_meta`` with 400 (``_looks_like_a_credential``)."""
    traceparent = context.get("traceparent")
    if not isinstance(traceparent, str) or not _TRACEPARENT.fullmatch(traceparent):
        return {}
    claims = {"traceparent": traceparent}
    tracestate = context.get("tracestate")
    if (
        isinstance(tracestate, str)
        and tracestate.strip()
        and len(tracestate) <= _TRACESTATE_MAX
        and all(c == "\t" or " " <= c <= "~" for c in tracestate)
        and not _looks_like_a_credential(tracestate)
    ):
        claims["tracestate"] = tracestate
    return claims


# A run's platform connection outlives its last call this long (background children after the run's end).
_RUN_CONNECTION_IDLE_S = 60.0
# The run's end waits this long at most for its connections to close (a session's teardown has no bound of its own:
# a platform that never answers it would hold the run past its ``timeout``); the closes go on past it.
_RUN_END_CLOSE_S = 10.0

# What close() and a run's end say when what they would close belongs to another event loop: how to close it there.
_CLOSE_ON_OWNER = (
    "Call close() on the loop that owns the connection: runner.run(server.close()) for an asyncio.Runner, or "
    "asyncio.run_coroutine_threadsafe(server.close(), loop).result() for a loop running in another thread "
    "(await asyncio.wrap_future(...) of it from async code)."
)


class _RunConnection:
    """One run's ``service="api"`` connection (``MCPServer._project_connection``)."""

    __slots__ = ("close_task", "ended", "fingerprint", "idle", "loop", "server", "users")

    def __init__(self, server: "MCPServer", fingerprint: str, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.server = server
        self.fingerprint = fingerprint
        # The run's loop, which owns the session: only a close() or the run's end on it closes it.
        self.loop = loop
        self.users = 0
        # The run ended (or ``MCPServer.close()`` let go of it): its last user closes it.
        self.ended = False
        self.close_task: asyncio.Task | None = None
        self.idle: asyncio.TimerHandle | None = None

    def cancel_idle(self) -> None:
        if self.idle is not None:
            self.idle.cancel()
            self.idle = None


def _log_close_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.warning("MCP run connection close failed", error=str(task.exception()))


def _elsewhere(owner: asyncio.AbstractEventLoop | None, loop: asyncio.AbstractEventLoop) -> bool:
    """Whether ``owner`` is another event loop than ``loop`` that has not closed: what it holds, only it can close
    (a loop that has closed took its sessions with it)."""
    return owner is not None and owner is not loop and not owner.is_closed()


def _gone(owner: asyncio.AbstractEventLoop | None, loop: asyncio.AbstractEventLoop) -> bool:
    """Whether ``owner`` is another event loop than ``loop`` that has closed: what it held went with it."""
    return owner is not None and owner is not loop and owner.is_closed()


@dataclass
class _Left:
    """What ``MCPServer.close()`` could not close: cut (or failed) before the session closed (``cut``), sessions of
    the server still open or shutting down (``sessions``), closes it waited for still going (``closing``)."""

    cut: int = 0
    sessions: int = 0
    closing: int = 0

    def raise_if_owed(self, label: str, failed: list[BaseException]) -> None:
        problems = []
        cause = failed[0] if failed else None
        why = f" ({cause!r})" if cause is not None else ""
        # A failure closing the server's own session, which no count covers: said as it is.
        if failed and not (self.sessions or self.closing):
            problems.append(
                f"Closing the server's session on the event loop that owns it failed ({cause!r}). Call close() again."
            )
        if self.cut:
            problems.append(
                f"{self.cut} platform connection(s) are still open: their close was cancelled or failed{why}. "
                "Call close() again."
            )
        if self.closing:
            problems.append(
                f"{self.closing} platform connection(s) are still closing: the wait for them was cancelled{why}. "
                "Call close() again."
            )
        if self.sessions:
            problems.append(
                f"{self.sessions} session(s) of this server are still open or shutting down: their close was "
                f"cancelled or failed{why}. Call close() again."
            )
        if problems:
            raise RuntimeError(f"MCP server {label!r}: " + " ".join(problems)) from cause


# How many times close() comes back to wait for closes still running before it counts what is owed.
_SETTLE_ROUNDS = 3


async def _wait_all_closed(tasks: set[asyncio.Task], label: str, timeout: float | None = None) -> None:
    """``_wait_closed`` for several closes at once, for at most ``timeout`` seconds: the closes go on after it."""
    waiting = asyncio.gather(*tasks, return_exceptions=True)
    # wait(), not wait_for(): neither its timeout nor a cut of this wait cancels the closes.
    done, _ = await asyncio.wait({waiting}, timeout=timeout)
    if not done:
        logger.warning("MCP run connection close still running at the run's end", server=label, seconds=timeout)


async def _wait_closed(task: asyncio.Task) -> None:
    """Wait for a connection's close; a cut ends the wait, not the close, and a failed close (logged by
    ``_log_close_failure``) does not fail the run or the call that waited."""
    with contextlib.suppress(Exception):
        await asyncio.shield(task)


def _run_connection_key(run_context: Any) -> str | None:
    """The run a platform connection is kept for, while one goes on in ``run_context`` and the caller is on its
    loop: its root call id, unique to the run (a caller may give two concurrent runs one ``context.id``).

    None (a call opens and closes its own connection, as outside a run) for no context, a context no run started
    (the platform resolver creates one when there is none), a run that has ended (unless the caller is one of its
    background children), and a call on another event loop than the run's: a handler offloaded to a thread that
    runs ``asyncio.run`` carries the run's context, and a session the run's loop owns cannot serve it."""
    if run_context is None:
        return None
    root = run_context.root_span()
    if root is None:
        return None
    if root.t1 is not None and current_background_task_id() is None:
        return None
    if run_context._loop is not asyncio.get_running_loop():
        return None
    return root.call_id


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


# How far _auth_identity looks into what an auth keeps, how many values it takes from there, and the largest set
# it reads (a set has no order every process agrees on: it is read whole, sorted, or not at all).
_AUTH_DEPTH = 4
_AUTH_VALUES = 64
_AUTH_SET = 256
# How many entries of a list it reads at most (in their order), and the largest mapping it reads (sorted).
_AUTH_SCAN = 4096
_PRIMITIVE_KEYS = (str, int, float, bool, bytes)


# Each auth object's identity, taken the first time it is seen (by id, with a weak reference that removes the
# entry when the auth goes): an auth keeps what it does as it runs (a digest's last challenge, an OAuth provider's
# tokens), which is not who it is. A new auth object is another identity.
_AUTH_IDENTITIES: dict[int, tuple[Any, tuple]] = {}


def _seen_auth_identity(auth: Any) -> tuple | None:
    """:func:`_auth_identity`, taken the first time this auth object is seen and kept for it."""
    if auth is None:
        return None
    entry = _AUTH_IDENTITIES.get(id(auth))
    if entry is not None and entry[0]() is auth:
        return entry[1]
    identity = _auth_identity(auth)
    try:
        ref = weakref.ref(auth, partial(_forget_auth_identity, id(auth)))
    except TypeError:  # not weakly referable: read each time
        return identity
    _AUTH_IDENTITIES[id(auth)] = (ref, identity)
    return identity


def _forget_auth_identity(auth_id: int, _ref: Any) -> None:
    _AUTH_IDENTITIES.pop(auth_id, None)


def _auth_identity(auth: Any) -> tuple | None:
    """Who an ``httpx.Auth`` reaches a server as (see ``MCPServer._provider_identity``).

    An auth that says so itself (a ``_provider_identity()`` of its own, as ``SnowflakeAuth`` has: what it was
    configured with, not what it resolved as it ran) is taken at its word. Any other: its class and the text it
    holds (a header, a token, a user and a password, a secret's value), also inside what it keeps (an object of
    its own, a dict, a list), at most ``_AUTH_DEPTH`` deep and ``_AUTH_VALUES`` values, in an order every process
    agrees on. Not its numbers: those are what it does as it runs (a digest's nonce count), not who it is. What
    cannot be read is left out (a proxy outside its context, a property that raises), and objects that answer
    attribute lookups themselves (proxies, lazy objects) are not looked into: a look would run their code.
    """
    if auth is None:
        return None
    own = _safe_get(type(auth), "_provider_identity")
    if callable(own):
        try:
            return (type(auth).__module__, type(auth).__qualname__, own(auth))
        except Exception:
            pass
    held: list[tuple[str, str]] = []
    _held_text(auth, "", 0, held, set())
    return (type(auth).__module__, type(auth).__qualname__, held)


def _safe_get(value: Any, name: str) -> Any:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _looked_into(value: Any) -> bool:
    """Whether the walk may read an object's attributes: not one that answers lookups itself (its own class chain
    defines ``__getattr__`` or another ``__getattribute__``; a metaclass's lookups are the class's, not its
    instances'). A pydantic model is read by its fields (:func:`_attributes`)."""
    if isinstance(value, BaseModel):
        return True
    mro = type(value).__mro__
    return not any(
        "__getattr__" in cls.__dict__ or ("__getattribute__" in cls.__dict__ and cls is not object) for cls in mro
    )


def _held_text(value: Any, path: str, depth: int, held: list[tuple[str, str]], seen: set[int]) -> None:
    if len(held) >= _AUTH_VALUES:
        return
    if isinstance(value, str):
        held.append((path, value))
        return
    if isinstance(value, bytes | bytearray):
        held.append((path, bytes(value).hex()))
        return
    if (
        depth >= _AUTH_DEPTH
        or value is None
        or isinstance(value, int | float | complex)
        or isinstance(
            value, type | types.ModuleType | types.FunctionType | types.MethodType | types.BuiltinFunctionType
        )
        or id(value) in seen
    ):
        return
    seen.add(id(value))
    try:
        if isinstance(value, Mapping):
            # Keys that are plain values name their entry (an object's key would name it by its address), in
            # sorted order: two equal mappings built in another order are one identity. A mapping too large to
            # sort cheaply is left out whole, which every process agrees on.
            if len(value) > _AUTH_SCAN:
                return
            items = sorted(
                ((f"{key!r}", item) for key, item in value.items() if isinstance(key, _PRIMITIVE_KEYS)),
                key=lambda entry: entry[0],
            )
        elif isinstance(value, list | tuple):
            items = ((str(index), item) for index, item in enumerate(itertools.islice(value, _AUTH_SCAN)))
        elif isinstance(value, set | frozenset):
            if len(value) > _AUTH_SET:
                return
            texts = sorted(
                item if isinstance(item, str) else bytes(item).hex()
                for item in value
                if isinstance(item, str | bytes | bytearray)
            )
            items = (("*", text) for text in texts)
        elif _looked_into(value):
            items = _attributes(value)
        else:
            return
        for name, item in items:
            if len(held) >= _AUTH_VALUES:
                return
            _held_text(item, f"{path}.{name}", depth + 1, held, seen)
    except Exception:
        return  # what cannot be read is left out


def _attributes(value: Any) -> list[tuple[str, Any]]:
    """An object's own attributes: its ``__dict__`` and its slots (a base class's too), each read on its own. A
    pydantic model's: its fields and extras only (its private attributes are what it does as it runs, and may be
    made per instance)."""
    if isinstance(value, BaseModel):
        try:
            fields = [(name, getattr(value, name)) for name in type(value).model_fields]
            return sorted(fields + list((value.__pydantic_extra__ or {}).items()), key=lambda entry: entry[0])
        except Exception:
            return []
    items = []
    try:
        own = vars(value)
    except Exception:
        own = {}
    items.extend(own.items())
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if not isinstance(slot, str) or slot in ("__dict__", "__weakref__") or slot in own:
                continue
            try:
                items.append((slot, getattr(value, slot)))
            except Exception:
                continue
    # By name: two equal objects whose attributes were set in another order are one identity.
    return sorted(items, key=lambda entry: entry[0])


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

    service: Literal["api"] | None = None
    """Resolve the project's API MCP endpoint with the current run's identity.

    ``MCPServer(transport="http", service="api")`` uses the same service routing
    as platform requests: local API port or deployed origin + /api/mcp. Each run
    gets its own connection, opened on its first call, reused by its other calls
    (and ``tools/list``), closed when the run ends; credentials and connections
    are never shared across runs. ``url`` and custom headers are not used in this
    mode. External MCP servers continue to use an explicit ``url``.
    """

    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    auth: httpx.Auth | None = Field(default=None, exclude=True)
    """``httpx.Auth`` applied to every HTTP request, for credentials that rotate during a
    session (``headers`` are fixed at connect). E.g. ``SnowflakeAuth`` re-mints its
    key-pair JWT before the one-hour expiry and retries once on 401.

    The auth flow runs in the transport's owner task, which has an empty context (no
    ``RunContext``). If the object defines ``async prepare()``, it is awaited in the
    caller's context before every request; resolve anything identity-bound there."""

    timeout: float | None = DEFAULT_TIMEOUT_SECONDS
    """Seconds to wait for the server to answer a request on an established session
    (``tools/list``, ``tools/call``). Defaults to ``DEFAULT_TIMEOUT_SECONDS`` (10 minutes);
    ``None`` waits forever. Per-tool overrides go in ``tool_timeouts``. A dead transport is
    detected independently of this and fails the call immediately. Does not cover
    connecting — see ``connect_timeout``."""

    tool_timeouts: dict[str, float | None] = Field(default_factory=dict)
    """Per-tool override of ``timeout`` for ``tools/call``, keyed by the server's bare tool
    name (``"codegen"``, not ``"timbal__codegen"``). ``None`` waits forever for that tool.
    One server usually mixes a few long-running tools with many that answer in
    milliseconds; this keeps the long bound on the tools that earn it."""

    connect_timeout: float | None = None
    """Seconds to wait for the transport to come up and ``initialize`` to complete. Kept
    separate from ``timeout`` because a stdio server's cold start (interpreter + imports)
    is routinely longer than a sensible per-request bound. ``None`` (default) waits forever."""

    elicitation: bool = True
    """Advertise the ``elicitation`` capability and bridge server ``elicit()`` calls to
    ``suspend()``. Set False to hide the capability (servers then must fail closed
    on tools that need confirmation)."""

    approval: Literal["destructive", "all"] | Callable[..., Any] | None = None
    """Route tools through the human-approval gate (``requires_approval``).

    - ``"destructive"``: tools whose annotations say destructive (``destructiveHint``
      not false and ``readOnlyHint`` not true). Unannotated tools are not gated.
    - ``"all"``: every tool.
    - callable ``(tool) -> bool``: receives the server's ``mcp.types.Tool`` once, at
      resolve time; return True to gate every call of that tool.
    - callable ``(tool, arguments) -> bool | dict | ApprovalPolicyDecision``: decided
      per call, with the LLM's arguments — for one tool that multiplexes subcommands
      (``codegen delete-workforce`` yes, ``codegen get-flow`` no). Same return contract
      as ``Runnable.requires_approval``; may be async.
    - ``None`` (default): no gating.
    """

    defer_loading: bool = False
    """Mark every tool of this server ``defer_loading``: the model finds them through tool
    search (``Agent.tool_search``) instead of carrying all their schemas on every call."""

    provenance: bool | None = None
    """Send the calling run's id and the tool span's call id in each ``tools/call``'s ``_meta``
    (``timbal.ai/run_id``, ``timbal.ai/call_id``), so the server can attribute what the call did
    to the exact span (monolith#565). Per-call ids cannot ride on HTTP headers: those are fixed
    per connection, which runs share. ``None`` (default): on for ``service="api"`` (the platform's
    own server) only, so third-party servers never see run ids.

    Also ``timbal.ai/parent_run_id`` and ``timbal.ai/session_id`` when known; for the platform's
    server only, ``timbal.ai/app_id`` (a JSON integer), ``timbal.ai/rev`` (``platform_config.subject``)
    and ``timbal.ai/path`` (the tool span's). A value the platform would refuse is left out, never
    sent, so a call never fails because of its provenance (see ``_claim_id``).

    Also W3C ``traceparent`` / ``tracestate``: the caller's active OpenTelemetry span when there is
    one; for the platform, otherwise the tool's span as ``OTelExporter`` exports it (see
    ``_trace_claims``). A third-party server that opts in (``True``) gets per-origin pseudonyms (HMAC,
    keyed by ``TIMBAL_PROVENANCE_SECRET`` or a per-process key) instead of our ids, and only an
    ambient trace context. It can correlate its own calls; our ids do not link them across servers
    (an ambient ``traceparent`` does, as any OpenTelemetry-instrumented request's headers would).
    Never any credential."""

    sampling_model: Any = None
    """Timbal model string (e.g. ``"anthropic/claude-haiku-4-5"``) or a ``TestModel`` used
    to answer the server's ``sampling/createMessage`` requests. ``None`` leaves sampling
    unsupported. Note the SDK answers server requests inline on the session's receive
    loop, so other traffic on this session waits while the LLM call runs."""

    _session: Any | None = PrivateAttr(default=None)
    # The per-run connection a ``service="api"`` server opens to the platform: it sends the platform's claims.
    _first_party: bool = PrivateAttr(default=False)
    _session_task: asyncio.Task | None = PrivateAttr(default=None)
    # The session's owner task, from the start of its handshake until the session is forgotten (a failed handshake,
    # close(), a reset, its owner ending): the one reference close() reads for it, closing it on its loop or refusing
    # when it is another's. Not one field for the handshake and another once published: a close() on another thread
    # could read the first before the publish and the second after the other was cleared, and close nothing.
    _owner_task: asyncio.Task | None = PrivateAttr(default=None)
    # Owner tasks a close() signalled to shut down and has not seen end (it was cut while waiting): the next close()
    # waits for them, its session refs being gone by then.
    _owners_closing: set[asyncio.Task] = PrivateAttr(default_factory=set)
    _session_closed: asyncio.Event | None = PrivateAttr(default=None)
    _get_session_id: Callable[[], str | None] | None = PrivateAttr(default=None)
    _tools_cache: list[Runnable] | None = PrivateAttr(default=None)
    _tools_cache_run_id: str | None = PrivateAttr(default=None)
    _tools_stale: bool = PrivateAttr(default=False)
    _lock: asyncio.Lock | None = PrivateAttr(default=None)
    _tools_lock: asyncio.Lock | None = PrivateAttr(default=None)
    _inflight: dict[int, _InflightCall] = PrivateAttr(default_factory=dict)
    _inflight_seq: int = PrivateAttr(default=0)
    # service="api": a run's root call id -> that run's platform connection (see _project_connection).
    _run_connections: dict[str, "_RunConnection"] = PrivateAttr(default_factory=dict)
    # Closes of run connections still going (``close()`` waits for them), and those of each run's (its end waits).
    _close_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)
    _run_closes: dict[str, set[asyncio.Task]] = PrivateAttr(default_factory=dict)
    # Run connections whose close was cut or failed before the session had closed: out of the pool (no call reuses
    # them), closed again by the next close() on their loop, the end of their run, or once idle (_retry_unclosed).
    # By id(entry), taken with an atomic pop: close() on two threads may both forget those of a loop that closed.
    _unclosed: dict[int, tuple[str, "_RunConnection"]] = PrivateAttr(default_factory=dict)
    # Runs whose end hook is registered, with the run's loop: one hook per run, which finds the run's connection
    # when the run ends (a connection let go of earlier is not kept alive by it), on that loop.
    _run_hooked: dict[str, asyncio.AbstractEventLoop] = PrivateAttr(default_factory=dict)

    def _provider_identity(self) -> tuple:
        # Its name, where it is reached and who it is reached as (headers, env, auth: two tenants of one endpoint are
        # two backends). A server that keeps its name and moves to another endpoint is another backend, and a call
        # that paused on the first one is not run on the second on the answer given then. Kept as a digest (the
        # agent's _provider_key): the credentials among these are never stored. A rotated token changes it too, and
        # a call paused before is then decided on again.
        return (
            "mcp",
            self.name,
            self.transport,
            self.service,
            self.url,
            self.command,
            tuple(self.args),
            sorted(self.headers.items()),
            sorted(self.env.items()),
            # Who it reaches the server as: read the first time this auth object is seen (what an auth keeps as it
            # runs, a challenge or a token it refreshed, is not who it is); a new auth object is another identity.
            _seen_auth_identity(self.auth),
        )

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
        if self.service is not None:
            if self.transport != "http" or self.url is not None or self.headers or self.auth is not None:
                raise ValueError("service='api' requires HTTP transport without url, headers or auth")
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("'command' is required for stdio transport")
        elif self.transport == "http":
            if not self.url and self.service is None:
                raise ValueError("'url' is required for http transport unless service='api'")
        return self

    def _new_project_server(self, url: str, headers: dict[str, str]) -> "MCPServer":
        server = MCPServer(
            name=self.name,
            transport="http",
            url=url,
            headers=headers,
            timeout=self.timeout,
            tool_timeouts=self.tool_timeouts,
            connect_timeout=self.connect_timeout,
            elicitation=self.elicitation,
            sampling_model=self.sampling_model,
            provenance=self._sends_provenance(),
        )
        server._first_party = True
        return server

    @asynccontextmanager
    async def _project_connection(self):
        """The platform connection of the calling run: opened on its first call, reused by the rest, closed when
        the run ends, or ``_RUN_CONNECTION_IDLE_S`` after a call when no other follows (between two calls of a run,
        or for background children that outlive it). Never shared across runs: the platform takes each call's
        principal and party from this connection's credential (monolith#565), so another run on it would act as
        the wrong caller. A run whose credential changes (a refreshed token) gets a new connection. A call outside
        a run that goes on, or on another loop than the run's (``_run_connection_key``), opens and closes its own."""
        # Read before resolving: the resolver creates and sets a run context when there is none.
        run_context = get_run_context()
        key = _run_connection_key(run_context)
        # Resolve BEFORE opening the transport's fresh-context owner task. It
        # must inherit the caller's identity, never a process service secret.
        from ..platform.utils import _resolve_url_and_headers

        url, headers = _resolve_url_and_headers("api", "mcp", {})
        if key is None:
            server = self._new_project_server(url, headers)
            try:
                yield server
            finally:
                await server.close()
            return
        fingerprint = hashlib.sha256(json.dumps([url, sorted(headers.items())]).encode()).hexdigest()
        entry = self._run_connections.get(key)
        if entry is not None and entry.fingerprint != fingerprint:
            self._retire_run_connection(key)
            entry = None
        if entry is None:
            entry = _RunConnection(self._new_project_server(url, headers), fingerprint, asyncio.get_running_loop())
            self._run_connections[key] = entry
            # Refused once the run's root has ended (a background child's call): the idle close closes it then.
            if key not in self._run_hooked and run_context.on_end(functools.partial(self._run_ended, key)):
                self._run_hooked[key] = entry.loop
        entry.users += 1
        entry.cancel_idle()
        try:
            yield entry.server
        finally:
            entry.users -= 1
            if entry.users == 0:
                if entry.ended or self._run_connections.get(key) is not entry:
                    # Its run ended, or it was replaced or let go of: closed now, as a call outside a run closes.
                    task = self._retire_run_connection(key, entry)
                    if task is not None:
                        await _wait_closed(task)
                else:
                    entry.idle = asyncio.get_running_loop().call_later(
                        _RUN_CONNECTION_IDLE_S, self._retire_run_connection, key, entry
                    )

    async def _wait_loop_closes(self) -> None:
        """Wait for every close of this loop still going. Shielded: a cut of this wait must not cut the closes, which
        would leave sessions open with nothing left to close them (one cut anyway is kept, ``_close_done``)."""
        loop = asyncio.get_running_loop()
        # A copy first: another loop's thread adds and drops its own closes (iterating the set itself can raise).
        closing = [task for task in list(self._close_tasks) if task.get_loop() is loop]
        if closing:
            await asyncio.shield(asyncio.gather(*closing, return_exceptions=True))

    def _run_ended(self, key: str) -> Awaitable[None] | None:
        """The run's end hook (one per run): its connection is let go of, and the run waits for every close of its
        connections still going, earlier ones included, so they are closed before the run returns (a loop that
        ends with the run, as under ``asyncio.run``, would cancel a close left to the background). One still in
        use is closed by its last user.

        Only on the run's loop, which owns the sessions. A run's loop that has closed took them with it: they are
        forgotten. Another loop (a stream closed from there) closes nothing: ``RuntimeError``, which the run logs
        (``run_end_hook_failed``); its connections are closed by their idle close (a cut one's too, ``_retry_unclosed``)
        or a ``close()`` on their loop."""
        owner = self._run_hooked.pop(key, None)  # the hook runs once, whatever it finds
        loop = asyncio.get_running_loop()
        if owner is None or owner is loop:
            return self._end_run_here(key)
        if owner.is_closed():
            self._drop_run(key)
            return None
        raise RuntimeError(
            f"MCP server {self._label!r}: the platform connection of a run that ended belongs to another event loop "
            "than the one ending the run, so it was not closed here. It closes on its own loop once idle "
            f"({_RUN_CONNECTION_IDLE_S:g} s after its last call or a close that failed). {_CLOSE_ON_OWNER}"
        )

    def _end_run_here(self, key: str) -> Awaitable[None] | None:
        entry = self._run_connections.get(key)
        if entry is not None:
            entry.ended = True
            self._retire_run_connection(key, entry)
        for ident, (kept_key, _) in list(self._unclosed.items()):
            if kept_key == key and (pair := self._unclosed.pop(ident, None)) is not None:
                self._retire_run_connection(*pair)
        closes = self._run_closes.pop(key, None)
        return _wait_all_closed(closes, self._label, _RUN_END_CLOSE_S) if closes else None

    def _retire_run_connection(self, key: str, entry: "_RunConnection | None" = None) -> asyncio.Task | None:
        """Forget a run's connection and, if nothing uses it, start closing it (``close()`` waits for the close;
        a caller that can wait gets the task). One still in use is closed by its last user."""
        if entry is None:
            entry = self._run_connections.get(key)
        if entry is None:
            return None
        self._unpool(key, entry)
        entry.cancel_idle()
        if entry.users:
            entry.ended = True
            return None
        if entry.close_task is None:
            entry.close_task = task = asyncio.get_running_loop().create_task(entry.server.close())
            self._close_tasks.add(task)
            self._run_closes.setdefault(key, set()).add(task)
            task.add_done_callback(functools.partial(self._close_done, key, entry))
            task.add_done_callback(_log_close_failure)
        return entry.close_task

    # A run's connections and close records are written by the run's loop alone (a run is keyed to its loop,
    # ``_run_connection_key``), except those of a loop that has closed, which close() on any thread forgets: the
    # writes below are single dict operations that tolerate a second taker, no check-then-delete.

    def _unpool(self, key: str, entry: "_RunConnection") -> None:
        """Take ``entry`` out of the pool if it is still the one under ``key`` (a replacement is never removed)."""
        if self._run_connections.get(key) is entry:
            self._run_connections.pop(key, None)

    def _drop_run(self, key: str) -> None:
        """Forget everything of the run ``key`` (its loop closed, taking its sessions): its connection, its closes,
        and its cut connections, which hold their credential headers."""
        self._run_connections.pop(key, None)
        self._run_closes.pop(key, None)
        for ident, (kept_key, _) in list(self._unclosed.items()):
            if kept_key == key:
                self._unclosed.pop(ident, None)

    def _forget_close(self, task: asyncio.Task) -> None:
        """Drop a close (its loop closed) from both collections, ``_close_done``'s callback never having run."""
        self._close_tasks.discard(task)
        for key in [key for key, closes in list(self._run_closes.items()) if task in closes]:
            self._discard_close(key, task)

    def _discard_close(self, key: str, task: asyncio.Task) -> None:
        """Take ``task`` off the run ``key``'s close records, and the record once empty."""
        closes = self._run_closes.get(key)
        if closes is None:
            return
        closes.discard(task)
        if not closes and self._run_closes.get(key) is closes:
            self._run_closes.pop(key, None)

    def _close_done(self, key: str, entry: "_RunConnection", task: asyncio.Task) -> None:
        self._close_tasks.discard(task)
        self._discard_close(key, task)
        if (task.cancelled() or task.exception() is not None) and not task.get_loop().is_closed():
            # Cut, or failed (the server's close() raised: what it could not close is still open), before the session
            # had closed: it may still be open. Kept out of the pool (a later call of the run must not reuse it and
            # open a second session on its server) for the next close() or end of the run, and closed again once
            # idle, as a pooled one is: a run's end on another loop is refused and closes nothing of it.
            entry.close_task = None
            entry.ended = True
            self._unclosed[id(entry)] = (key, entry)
            entry.cancel_idle()
            entry.idle = task.get_loop().call_later(_RUN_CONNECTION_IDLE_S, self._retry_unclosed, key, entry)

    def _retry_unclosed(self, key: str, entry: "_RunConnection") -> None:
        """A cut connection's idle close, on its loop: closed again unless a close() or its run's end took it first
        (they cancel this when they do)."""
        entry.idle = None
        if self._unclosed.pop(id(entry), None) is not None:
            self._retire_run_connection(key, entry)

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

    def _http_timeout(self) -> httpx.Timeout:
        """httpx timeouts for the streamable HTTP transport.

        A bare ``httpx.AsyncClient()`` applies its 5-second default to *reads*, and a
        ``tools/call`` answer arrives as the first bytes of a POST's SSE stream — so any
        tool that ran longer than 5s tripped ``ReadTimeout`` inside the SDK's SSE reader,
        which logs it at debug and returns without failing the pending request. The
        transport stayed up, the owner task stayed alive, and the call hung forever
        (Composer bootstrap on ``codegen test``, 2026-09-07). Reads are unbounded here:
        the wall-clock bound on a request is ``timeout`` in ``_request``, and the
        standalone GET stream for server notifications idles by design. Connect stays
        bounded so a black-holed host does not wedge the handshake even with
        ``connect_timeout=None``.
        """
        connect = self.connect_timeout if self.connect_timeout is not None else 30.0
        return httpx.Timeout(connect=connect, read=None, write=30.0, pool=30.0)

    @asynccontextmanager
    async def _connect_http(self):
        assert self.url is not None
        async with httpx.AsyncClient(
            headers=self.headers if self.headers else None,
            auth=self.auth,
            timeout=self._http_timeout(),
            follow_redirects=True,
        ) as http_client:
            async with streamable_http_client(self.url, http_client=http_client) as (read, write, get_session_id):
                async with self._client_session(read, write) as session:
                    await session.initialize()
                    self._get_session_id = get_session_id
                    logger.info("Connected to MCP server via http", url=self.url)
                    if self._stateless_http() and (self.elicitation or self.sampling_model):
                        logger.warning(
                            "MCP server is stateless (no Mcp-Session-Id): it cannot receive answers to "
                            "server-initiated requests (elicitation, sampling) on this protocol version. "
                            "Tools that use them will fail fast instead of hanging; use approval= for "
                            "confirmations or wait for MRTR (2026-07-28).",
                            server=self.name,
                            url=self.url,
                        )
                    yield session

    def _stateless_http(self) -> bool:
        """An HTTP server that issued no ``Mcp-Session-Id`` treats every POST as independent,
        so our reply to a server-initiated request lands on a transport that has no idea
        who asked. The server's ``elicit()`` / ``create_message()`` then waits forever."""
        return self.transport == "http" and self._get_session_id is not None and self._get_session_id() is None

    def _abort_inflight_for_stateless(self, calls: list["_InflightCall"], kind: str) -> None:
        """Fail every in-flight call fast: the server is blocked on an answer it can never get."""
        for call in calls:
            if not call.abort.done():
                call.abort.set_exception(
                    RuntimeError(
                        f"MCP server '{self._label}' asked the client for {kind} during '{call.tool_name}', "
                        "but the server is stateless (no Mcp-Session-Id) so the answer cannot be delivered "
                        "and the call would never complete. Use approval= for confirmations, or a "
                        "sessionful server."
                    )
                )

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

    def _forget_session(self, owner: asyncio.Task | None = None) -> None:
        """Drop references to the current session without closing anything (the owner is already done, or its
        shutdown is the caller's, which recorded it in ``_owners_closing`` first). ``owner``: the owner task let go
        of, when it is not the published session's (a handshake on a loop that closed). The owner reference goes
        only if it is that one: a handshake started since keeps its own."""
        owner = owner or self._session_task
        self._session = None
        self._session_task = None
        self._session_closed = None
        self._get_session_id = None
        if owner is not None and self._owner_task is owner:
            self._owner_task = None

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
            self._owners_closing.add(task)  # before the refs go, as in close()
            task.add_done_callback(self._owners_closing.discard)
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
        self._owner_task = task
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
            try:
                await asyncio.wait({task})
            finally:  # also when this wait is cut too (a connect cancelled twice)
                if not task.done():  # signalled and still shutting down: the next close() waits for it
                    self._owners_closing.add(task)
                    task.add_done_callback(self._owners_closing.discard)
                if self._owner_task is task:
                    self._owner_task = None
            raise
        # The owner reference stays as it was: the same task from the handshake on.
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
        abort: asyncio.Future | None = None,
        timeout: float | None = _USE_SERVER_TIMEOUT,
    ) -> T:
        """Run one request against the live session, with the failure modes the SDK leaves to us.

        - **Dead transport mid-request**: the SDK never fails the pending future when the
          receive loop dies, so the request is raced against the owner task and turned
          into a ``ConnectionError`` instead of hanging forever.
        - **Stale session** (``Session terminated``: the server 404s our ``Mcp-Session-Id``
          after a restart or expiry): the session is reset and the request retried once.
          Safe for any request — a 404 means the server never saw it.
        - **No answer within** ``timeout`` (server alive, never replies): ``TimeoutError``.
          The session stays up; the server may still be running the tool. Defaults to
          ``self.timeout``; ``tools/call`` passes the tool's ``tool_timeouts`` entry.
        - ``abort``: a future a callback may fail when it learns the call can never complete
          (stateless server asked us something); raised here instead of waiting.
        - ``idempotent=True`` also retries once after a mid-request connection loss
          (``tools/list``); ``tools/call`` never does, the tool may have run.
        """
        if timeout is _USE_SERVER_TIMEOUT:
            timeout = self.timeout
        prepare = getattr(self.auth, "prepare", None)
        if prepare is not None:
            await prepare()
        for attempt in (1, 2):
            session = await self._connect()
            owner = self._session_task
            request: asyncio.Future = asyncio.ensure_future(send(session))
            try:
                waiting: set[asyncio.Future] = {request}
                if owner is not None:
                    waiting.add(owner)
                if abort is not None:
                    waiting.add(abort)
                done, _ = await asyncio.wait(waiting, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            finally:
                if not request.done():
                    request.cancel()
            if request not in done:
                if abort is not None and abort in done:
                    raise abort.exception()  # type: ignore[misc]
                if not done:
                    raise TimeoutError(f"MCP server '{self._label}': {what} timed out after {timeout}s")
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
        if self._stateless_http():
            # Whatever we answer cannot reach the handler that asked; don't let it hang.
            self._abort_inflight_for_stateless(calls, "elicitation")
            return mcp_types.ErrorData(
                code=mcp_types.INVALID_REQUEST,
                message="Client cannot answer server-initiated requests on a stateless connection",
            )
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
        if self._stateless_http():
            self._abort_inflight_for_stateless(calls, "sampling")
            return mcp_types.ErrorData(
                code=mcp_types.INVALID_REQUEST,
                message="Client cannot answer server-initiated requests on a stateless connection",
            )
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
            abort=asyncio.get_running_loop().create_future(),
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

    def _tool_timeout(self, bare_name: str) -> float | None:
        if bare_name in self.tool_timeouts:
            return self.tool_timeouts[bare_name]
        return self.timeout

    def _approval_policy(self, mcp_tool: mcp_types.Tool) -> bool | Callable[..., Any]:
        """The ``requires_approval`` value for one tool: a static bool, or a per-call policy.

        The two-argument form of ``approval`` is wrapped into a ``**arguments`` callable so
        ``Runnable._execute_approval_callable`` hands it the whole validated input; its
        return (bool / dict / ``ApprovalPolicyDecision``) flows through unchanged.
        """
        if self.approval is None:
            return False
        if self.approval == "all":
            return True
        if self.approval == "destructive":
            return _is_destructive(mcp_tool)
        policy = self.approval
        if _approval_policy_arity(policy) == 1:
            return bool(policy(mcp_tool))

        if inspect.iscoroutinefunction(policy):

            async def _per_call_async(**arguments: Any) -> Any:
                return await policy(mcp_tool, arguments)

            return _per_call_async

        def _per_call(**arguments: Any) -> Any:
            return policy(mcp_tool, arguments)

        return _per_call

    def _sends_provenance(self) -> bool:
        return self.provenance if self.provenance is not None else self.service == "api"

    def _provenance_meta(self) -> dict[str, Any] | None:
        """``tools/call`` ``_meta`` naming the calling run and the tool's span, when enabled (see ``provenance``).

        Also the parent run (when chained: the server derives the thread from it before the run is stored) and
        the session (a hint, see ``ChatSession.root_run_id``); for the platform's server, the app, revision and
        tool span path. Claims for linking only: identity is the request's own credential, never anything in
        ``_meta``. A value the platform would refuse is left out (see ``_claim_id``), never sent.
        """
        if not self._sends_provenance():
            return None
        run_context = get_run_context()
        if run_context is None:
            return None
        call_id = get_call_id()
        ids = {
            PROVENANCE_RUN_KEY: run_context.id,
            PROVENANCE_CALL_KEY: call_id,
            PROVENANCE_PARENT_KEY: run_context.parent_id,
            PROVENANCE_SESSION_KEY: getattr(run_context, "_session_id", None),
        }
        if self._first_party or self.service == "api":
            claims = {key: _claim_id(value) for key, value in ids.items()}
            claims.update(_platform_claims(run_context, call_id))
            # The caller's span when it is one the platform keeps, else the tool's own.
            claims.update(
                _trace_claims(_ambient_trace_context()) or _trace_claims(_timbal_trace_context(run_context.id, call_id))
            )
        else:
            # A third party that opted in: ids it can correlate among its own calls, but not across servers.
            origin = self._origin()
            # Only ids the claims would carry (``_claim_id``), as for the platform: the same keys either way.
            claims = {key: _pseudonym(origin, value) for key, value in ids.items() if _claim_id(value) is not None}
            claims.update(_trace_claims(_ambient_trace_context()))
        return {key: value for key, value in claims.items() if value is not None} or None

    def _origin(self) -> str:
        """What a pseudonym is keyed to: ``scheme://host:port`` (default port filled in, credentials in the URL
        left out, so one server gets one set of pseudonyms however its URL is written), or for stdio the command,
        its arguments and its environment, encoded exactly."""
        if self.transport == "http" and self.url:
            parts = urlsplit(self.url)
            scheme = parts.scheme.lower()
            try:
                port = parts.port
            except ValueError:
                port = None
            port = port or {"http": 80, "https": 443}.get(scheme)
            host = parts.hostname or ""
            host = f"[{host}]" if ":" in host else host
            return f"{scheme}://{host}:{port}" if port else f"{scheme}://{host}"
        # Exact (an argument with a space is not two), and with the environment: one command reaching two
        # upstreams through env is two servers. Never sent: only ever hashed with the key (_pseudonym).
        return "stdio:" + json.dumps([self.command or "", list(self.args), sorted((self.env or {}).items())])

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
            if self.service is not None:
                async with self._project_connection() as server:
                    # Approval lives on this public tool; invoke the bound
                    # handler directly so it is not requested a second time.
                    return await server._make_tool(mcp_tool).handler(**kwargs)
            # Taken here, in the calling task: the session's owner task does not see the caller's context.
            meta = self._provenance_meta()
            call = self._begin_call(bare_name)
            try:
                result = await self._request(
                    lambda session: session.call_tool(bare_name, arguments=kwargs, meta=meta),
                    what=f"tools/call {bare_name}",
                    abort=call.abort,
                    timeout=self._tool_timeout(bare_name),
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
            converted = _convert_call_tool_result(bare_name, result)
            if call.ambiguous_with:
                # The server's question was answered `cancel` for it, and a server that treats
                # a cancelled confirmation as a normal answer ("not swapped") returns a result
                # that looks fine but is not what the user chose. Raising here would also
                # discard the innocent sibling's result, so the LLM is told instead. Which of
                # the in-flight calls asked is unknowable, so every one of them carries the note.
                converted = _with_ambiguity_note(converted, call.ambiguous_with)
                if not isinstance(converted, str | list):
                    logger.warning(
                        "MCP tool result may reflect an auto-cancelled elicitation",
                        server=self.name,
                        tool=bare_name,
                        inflight=call.ambiguous_with,
                    )
            return converted

        approval_policy = self._approval_policy(mcp_tool)
        requires_approval = bool(approval_policy)
        return MCPTool(
            name=exposed_name,
            description=description,
            handler=_handler,
            input_schema=mcp_tool.inputSchema or {},
            title=title,
            tool_annotations=annotations,
            read_only=bool(annotations and annotations.get("readOnlyHint") is True),
            defer_loading=self.defer_loading,
            search_group=self.name,
            requires_approval=approval_policy,
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

    def _reuse_tools_cache(self, run_id: str | None) -> list[Runnable]:
        """Hand out the cached list and stamp it with the run that is now using it.

        The stamp is what ``_tools_refresh_due`` compares against, so it must follow the
        *latest* run that saw the list, not the one that fetched it. A long-lived server
        reused across runs would otherwise treat a ``tools/list_changed`` arriving mid-run
        B as a boundary (``B != A``) and swap the tool set under the LLM's feet.
        """
        assert self._tools_cache is not None
        if run_id is not None:
            self._tools_cache_run_id = run_id
        return self._tools_cache

    async def resolve(self) -> list[Runnable]:
        """See base class.

        Lists the server's tools once and caches them until ``close()``. A
        ``tools/list_changed`` notification marks the cache stale, but the refetch
        waits for the next run (different run id) so the tool set stays fixed for
        the whole run (prompt-cache friendly, no mid-run surprises). Tools are
        sorted by name so a server that lists in nondeterministic order does not
        churn the LLM's tools prefix.
        """
        if self.service is not None:
            async with self._project_connection() as server:
                result = await server._request(lambda session: session.list_tools(), what="tools/list", idempotent=True)
            return [self._make_tool(tool) for tool in sorted(result.tools, key=lambda tool: tool.name)]

        run_context = get_run_context()
        run_id = run_context.id if run_context is not None else None
        if not self._tools_refresh_due(run_id):
            return self._reuse_tools_cache(run_id)

        async with self._get_tools_lock():
            if not self._tools_refresh_due(run_id):
                return self._reuse_tools_cache(run_id)

            result = await self._request(lambda session: session.list_tools(), what="tools/list", idempotent=True)
            mcp_tools = sorted(result.tools, key=lambda t: t.name)
            tools: list[Runnable] = [self._make_tool(mcp_tool) for mcp_tool in mcp_tools]
            logger.info("Resolved MCP tools", server=self.name, tools=[t.name for t in tools])
            self._tools_cache = tools
            self._tools_cache_run_id = run_id
            self._tools_stale = False
            return tools

    async def close(self) -> None:
        """Close the MCP server connection. Safe from any task of the loop that owns it, in any order.

        Acts only on what this event loop owns. The server's own session (a standalone server's) can only be closed
        on its loop: from another loop that has not closed, ``RuntimeError`` is raised before anything is touched
        (call close() on the loop that owns the connection). A ``service="api"`` server's run connections belong to
        their runs: this loop's are closed now when idle, one a call still uses is closed by that call's end (a call
        in flight is not cut), and another loop's are left to their run's end or idle close there. What a loop that
        has closed held went with it, and is forgotten. Then close() waits for every close still going on this loop,
        earlier ones included, and for the owner tasks an earlier close() signalled and was cut waiting for.

        Raises RuntimeError once it has done what it can when something it set out to close is still open: a close
        was cancelled or failed (call close() again)."""
        loop = asyncio.get_running_loop()
        self._refuse_other_loops(loop)
        # The server's session's owner task, at whichever stage it is (handshake or published): one read.
        session = self._owner_task
        failed: list[BaseException] = []
        # The closes this close() waits for: those of this loop going when it began, and those it starts. One still
        # going after its waits is owed; one a call starts later is that call's.
        ledger = {task for task in list(self._close_tasks) if not task.done() and task.get_loop() is loop}
        # The entries themselves, not only their ids: one freed meanwhile could leave its id to a new connection.
        entries = self._take_run_connections(loop)
        taken = {id(entry): entry for _, entry in entries}
        # Let go of before any wait, so a close() running beside this one waits for their closes too.
        for key, entry in entries:
            task = self._retire_run_connection(key, entry)
            if task is not None:
                ledger.add(task)
        # Shielded: a close() that is cancelled stops waiting, the closes go on (gather would cancel them).
        waits = asyncio.gather(self._wait_loop_closes(), self._close_own_session(loop), return_exceptions=True)
        for result in await asyncio.shield(waits):
            if isinstance(result, BaseException):
                failed.append(result)  # its cause is said; what it left is counted below, once
        await self._settle_closes(loop)
        await self._wait_owners_closing(loop)
        self._owed(loop, taken, session, ledger).raise_if_owed(self._label, failed)

    def _refuse_other_loops(self, loop: asyncio.AbstractEventLoop) -> None:
        """Raise, before close() touches anything, when the server's own session belongs to another event loop that
        has not closed (open, in its handshake, or still shutting down after a close() that was cut): its owner task,
        locks and transport are bound to that loop. The owner reference is read first: a close there moves the owner
        to ``_owners_closing`` before it lets go of the reference, so one of the two reads finds it."""
        owner = self._owner_task
        owners = [owner] if owner is not None else []
        owners += [task for task in list(self._owners_closing) if not task.done()]
        if any(_elsewhere(task.get_loop(), loop) for task in owners):
            raise RuntimeError(
                f"MCP server {self._label!r}: close() was called on another event loop than the one that owns its "
                f"session, which alone can close it. {_CLOSE_ON_OWNER}"
            )

    async def _settle_closes(self, loop: asyncio.AbstractEventLoop) -> None:
        """Wait for every close still going on this loop (shielded, ``_wait_loop_closes``), again while some run: one
        may have started during a wait."""
        for _ in range(_SETTLE_ROUNDS):
            if not any(task.get_loop() is loop and not task.done() for task in list(self._close_tasks)):
                return
            await self._wait_loop_closes()

    def _owed(
        self,
        loop: asyncio.AbstractEventLoop,
        taken: "dict[int, _RunConnection]",
        session: asyncio.Task | None,
        ledger: set[asyncio.Task],
    ) -> "_Left":
        """What is still owed once close() has done what it could, from the state it ends in: each connection,
        close or session once, whichever step left it. ``taken``: the connections this close() set out to close;
        ``session``: the server's own session when it began (not one a call opened since). Only this loop's."""
        left = _Left()
        for entry in list(self._run_connections.values()):
            if taken.get(id(entry)) is entry:
                left.cut += 1  # set out to close it, and it is still there
        for _, entry in list(self._unclosed.values()):
            if (entry.loop or loop) is loop:
                left.cut += 1
        # Any other close still running is not owed: one a call started since (a connection in use is left to its
        # call) or another close()'s.
        left.closing = sum(1 for task in ledger if not task.done())
        # The session close() began with, while it is still the server's (its close, or handshake, did not happen),
        # and the owners a close() signalled that are still shutting down (the wait for them was cut). The owner
        # reference is read first, as in _refuse_other_loops; each counted once. A handshake started since is a new
        # connection (a call reconnecting after this close() let go of the old one), not this close()'s to report.
        still = {session} if session is not None and session is self._owner_task else set()
        still.update(list(self._owners_closing))
        left.sessions = sum(1 for owner in still if not owner.done() and owner.get_loop() is loop)
        return left

    async def _close_own_session(self, loop: asyncio.AbstractEventLoop) -> None:
        """The server's own session (a standalone server's), closed on this loop, which owns it and its locks. One
        whose loop has closed went with it: only the references are dropped, and the caches cleared."""
        task = self._owner_task  # (a handshake under way, or the published session: one read)
        if task is None:
            # Its refs gone (an earlier close() forgot them and is still shutting it down, holding the locks): the
            # owner it signalled.
            closing = [owner for owner in list(self._owners_closing) if not owner.done()]
            task = closing[0] if closing else None
        if task is not None and _gone(task.get_loop(), loop):  # its calls in flight died with it: the caches go too
            self._forget_session(task)
            self._drop_caches()
            return
        if task is not None and task.get_loop() is not loop:
            return  # opened on another loop since close() began: a new connection, not this close()'s
        # Nothing open nor opening is closed here too, which clears the caches.
        await self._close_session_here()

    async def _close_session_here(self) -> None:
        async with self._get_tools_lock(), self._get_lock():
            task, closed = self._session_task, self._session_closed
            if task is not None:
                # Recorded before the refs go: a close() on another thread sees the session or its owner, never
                # neither (it would take the locks on its own loop instead of refusing).
                self._owners_closing.add(task)
                task.add_done_callback(self._owners_closing.discard)
            self._forget_session()
            if task is not None and closed is not None:
                await self._shutdown_owner(task, closed)
            self._drop_caches()

    def _drop_caches(self) -> None:
        """What every close() clears with the session: the tool list and the calls in flight on it."""
        self._tools_cache = None
        self._tools_cache_run_id = None
        self._tools_stale = False
        self._inflight.clear()

    def _take_run_connections(self, loop: asyncio.AbstractEventLoop) -> list[tuple[str, "_RunConnection"]]:
        """This loop's run connections, pooled or cut (those cut are taken out at once). What a loop that has closed
        held went with it and is forgotten (its cut connections hold their credential headers); another loop's is
        left to that loop."""
        taken: list[tuple[str, _RunConnection]] = []
        for key, entry in list(self._run_connections.items()):
            owner = entry.loop or loop
            if owner is loop:
                taken.append((key, entry))
            elif owner.is_closed():
                self._unpool(key, entry)  # (another close() on another thread may be forgetting it too)
        for ident, (key, entry) in list(self._unclosed.items()):
            owner = entry.loop or loop
            if owner is loop:
                if self._unclosed.pop(ident, None) is not None:
                    taken.append((key, entry))
            elif owner.is_closed():
                self._unclosed.pop(ident, None)
        for task in list(self._close_tasks):
            if _gone(task.get_loop(), loop):
                self._forget_close(task)
        return taken

    async def _wait_owners_closing(self, loop: asyncio.AbstractEventLoop) -> None:
        """Wait for the session owner tasks an earlier close() on this loop signalled and was cut waiting for (their
        refs are gone). Those of a loop that has closed went with it."""
        here = []
        for owner in [owner for owner in list(self._owners_closing) if not owner.done()]:
            if owner.get_loop() is loop:
                here.append(owner)
            elif owner.get_loop().is_closed():
                self._owners_closing.discard(owner)
        if here:
            await asyncio.wait(here)

    def __del__(self):
        session = getattr(self, "_session", None)
        if session:
            logger.warning("MCPServer deleted without calling close()")
