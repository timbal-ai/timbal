"""A text conversation that stays open: messages in at any time, one turn at a time, events out."""

import asyncio
import contextlib
import copy
import functools
import json
import math
import os
import re
import time
import weakref
from collections.abc import AsyncIterator, Callable, Collection
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import structlog
from uuid_extensions import uuid7

from ..state import RunContext, set_call_id, set_parent_call_id, set_run_context
from ..state.background import (
    DEFAULT_BG_LOG_MAX_BYTES,
    DEFAULT_BG_LOG_MAX_EVENTS,
    BackgroundEventLog,
    BackgroundTaskStore,
    SessionEvent,
    adopt_background_store,
    forget_idle_background_store,
    session_continues_from,
    store_for_run,
)
from ..state.chat_store import (
    ChatSessionStore,
    SessionLeased,
    SessionNotStored,
    SessionSnapshot,
    SnapshotConflict,
    process_owner,
    storable,
)
from ..types.approval import ApprovalResolution, Cancel
from ..types.content import TextContent
from ..types.events import OutputEvent
from ..types.message import RUNTIME_SOURCE, Message
from .runnable import ATTENDED_TRIGGER_KINDS, _own_copy
from .session_scheduler import Followup, SessionScheduler, is_parked, pending_user_batch
from .tool_policy import ReadOnlyWithin, read_only_within, restore_tool_policy
from .tool_policy import ToolPaths as _ToolPaths  # noqa: F401 — re-exported for callers and tests
from .tool_policy import allowed_tool_paths as _allowed_tool_paths  # noqa: F401
from .tool_policy import describe_tool_policy as _describe_policy

if TYPE_CHECKING:
    from .agent import Agent

__all__ = ["SEND_MODES", "SILENT_REPLY", "ChatSession", "OnDeadTurn", "SendMode", "ToolPolicy"]

logger = structlog.get_logger("timbal.core.chat_session")

SendMode = Literal["steer", "now", "followup", "collect", "interrupt"]
SEND_MODES: tuple[str, ...] = ("steer", "now", "followup", "collect", "interrupt")

ToolPolicy = Literal["read_only"] | Collection[str] | Callable[[Any], bool]
"""Which tools a turn offers its model: ``"read_only"`` (tools declared ``read_only``,
MCP tools the server marks ``readOnlyHint``, the agent's own readers), a collection of
tool names, or a callable deciding per tool."""

SILENT_REPLY = "NO_REPLY"
"""What a turn the runtime started answers to stay silent (``turn_finished.silent``)."""

WAKE_KIND = "wake"
RECOVERY_KIND = "recovery"

OnDeadTurn = Literal["continue", "notify", "drop"]
"""What a restored session does with a turn that died with its process (see :meth:`ChatSession.from_store`)."""

_SILENCE_NOTE = f"If nothing here needs the user's attention, reply with exactly {SILENT_REPLY} and nothing else."
_WAKE_HEADER = "Scheduled wake-up (the user did not write this):"
# A turn that fails before it starts: retried after 0.5 s, then 1 s; past that, only when new work arrives.
_MAX_TURN_FAILURES = 3
_TURN_RETRY_DELAY_S = 0.5
# Between attempts to learn whether the run a session continues is parked (its trace store is down).
_RESTORE_RETRY_MAX_S = 30.0
# How much of a lease before its end a session that could not renew it stops. Stopping takes a moment (the
# turn is cancelled, not cut off), and the store's clock is not this process's.
_LEASE_MARGIN = 0.1
_RECOVERY_HEADER = "Your previous turn was cut off by a server restart (the user did not write this)."
_RECOVERY_ADVICE = (
    "Continue where you left off. Do not repeat completed actions. Before redoing one that is not confirmed, "
    "check whether it already happened, or ask the user."
)
# A restored log continues this far past the saved cursor: further than any client of the old process read.
_LOG_RESTART_GAP = 1_000_000_000
_PERSISTED_KINDS = frozenset(
    {"message", "message_settled", "wake", "wake_superseded", "turn_started", "turn_finished", "interrupt", "closed"}
)
_READ_ONLY_NOTE = (
    "This turn can only read: tools that send, write, delete or buy are not available. If the task needs "
    "one, do not retry or look for a way around it; say what you would do and that it needs the user."
)
_SESSION_KWARGS = frozenset({"prompt", "messages", "parent_id", "resume"})


@dataclass(eq=False)
class _Wake:
    text: str
    key: str | None
    tool_policy: Any
    silent_ok: bool
    id: str = field(default_factory=lambda: uuid7(as_type="hex"))  # type: ignore[arg-type]
    kind: str = WAKE_KIND
    trigger: dict[str, Any] | None = None
    # The pending wakes of the same key this one replaced: put back if it is withdrawn. Not stored. Kept
    # until this wake is itself stored (``mark``, see ChatSession.stored_since): from then on it is what
    # a withdrawn replacement of it puts back.
    replaced: list["_Wake"] = field(default_factory=list)
    mark: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "key": self.key,
            "tool_policy": _store_policy(self.tool_policy),
            "silent_ok": self.silent_ok,
            "id": self.id,
            "kind": self.kind,
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_Wake":
        return cls(
            text=data["text"],
            key=data.get("key"),
            tool_policy=_load_policy(data.get("tool_policy")),
            silent_ok=bool(data.get("silent_ok", True)),
            id=data["id"],
            kind=data.get("kind", WAKE_KIND),
            trigger=data.get("trigger"),
        )


@dataclass(frozen=True)
class _TurnOptions:
    """What a turn runs under; a resume of it runs under the same."""

    tool_policy: Any = None
    silence: bool = False
    # Takes the session's lane="next" messages between steps. A wake does not: a user message would be
    # answered under the wake's tool policy and silence, so it waits for a turn of its own.
    steer: bool = True


def _check_tool_policy(policy: Any) -> Any:
    if policy is None or policy == "read_only" or callable(policy):
        return policy
    if isinstance(policy, str) or not isinstance(policy, Collection):
        raise ValueError("tool_policy must be 'read_only', a collection of tool names, a callable, or None.")
    names = frozenset(policy)
    if not all(isinstance(name, str) for name in names):
        raise ValueError("tool_policy names must be strings.")
    return names


def _with_note(prompt: Message, note: str) -> Message:
    return Message(
        role=prompt.role,
        content=[*prompt.content, TextContent(text=note)],
        metadata=dict(prompt.metadata) if prompt.metadata else None,
    )


def _store_policy(policy: Any) -> Any:
    """A tool policy as a snapshot holds it. A callable cannot be stored: it is ``"custom"`` (see _load_policy)."""
    if policy is None or policy == "read_only":
        return policy
    if isinstance(policy, ReadOnlyWithin):
        return policy.to_stored()
    if callable(policy):
        return "custom"
    return sorted(policy)


def _load_policy(stored: Any, trace: Any = None) -> Any:
    """A stored policy back. ``"custom"`` (a callable) never comes back wider than it was: with the trace of a
    run it governed, the tools its agents were offered there (by path); without one, no tools at all. Not
    ``"read_only"``: the callable may have denied read-only tools too."""
    if stored == "custom":
        if trace is None:
            logger.warning("chat_session_policy_restored_empty", detail="a callable tool_policy restores with no tools")
            return frozenset()
        return restore_tool_policy("custom", trace)
    if isinstance(stored, dict):
        return ReadOnlyWithin.from_stored(stored)
    return frozenset(stored) if isinstance(stored, list) else stored


def _stored_turn_policy(in_flight: dict[str, Any], last_options: dict[str, Any] | None) -> Any:
    """The policy the turn in flight ran under, as stored (for :func:`_load_policy`).

    Kept with the turn (``in_flight.tool_policy``). A snapshot written before it was holds the same policy
    in two other places: the wake the turn ran for, and the options saved for a resume of the turn
    (``last_options``, set and stored in the same step as the turn itself, so they are this turn's, also when
    it is the resume of a parked wake and has no wake of its own). Never "no policy" because the new field is
    missing: that would let the recovery turn reach every read-only tool, whatever the dead turn was allowed.
    """
    if "tool_policy" in in_flight:
        return in_flight["tool_policy"]
    wake = in_flight.get("wake")
    if isinstance(wake, dict) and "tool_policy" in wake:
        return wake["tool_policy"]
    return (last_options or {}).get("tool_policy")


def _dump_prompt(prompt: Any) -> Any:
    """A turn's prompt as a snapshot holds it: always JSON. One that JSON cannot carry whole (files, metadata
    from a caller's own run) is kept as its text: a resume only needs the turn to have a prompt, and the agent
    drops it. Never a reason for a save to fail."""
    from ..utils.serialization import _dump_sync, _NeedsAsync

    if prompt is None or isinstance(prompt, str):
        return prompt
    try:
        dumped = _dump_sync(prompt)
    except (_NeedsAsync, TypeError, ValueError):
        dumped = None
    # `storable`: the snapshot's own encoder, so what passes here cannot fail the write (NaN in a caller's
    # metadata passes a plain json.dumps and would then fail every save of the session, not only this one).
    if dumped is not None and storable(dumped):
        return dumped
    if isinstance(prompt, Message):
        role, text = prompt.role, prompt.collect_text()
    elif isinstance(prompt, dict):
        role = prompt.get("role") if isinstance(prompt.get("role"), str) else "user"
        text = "".join(block["text"] for block in _blocks(prompt, "text") if isinstance(block.get("text"), str))
    else:
        role, text = "user", str(prompt)
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _load_prompt(stored: Any) -> Any:
    return Message.validate(stored) if isinstance(stored, dict) else stored


def _stored_answer(value: Any) -> Any:
    """One resume value as a snapshot holds it: JSON that means the same to the gate that reads it back.

    ``Cancel`` and ``ApprovalResolution`` have a JSON form of their own (an approval's is the object the
    approval gate accepts over HTTP, every field kept: scope, input overrides, who decided and when). Anything
    else must be JSON already, and come back from it unchanged. TypeError otherwise: written as text (its
    ``str()``), an answer would come back from a restart as something the gate cannot read, and an approved
    call would silently not run.
    """
    if isinstance(value, Cancel):
        return {"$cancel": value.reason}
    if isinstance(value, ApprovalResolution):
        # Its fields as they are, checked like any other answer below. Not pydantic's JSON mode: that writes
        # NaN and Infinity as null (an input override silently changed) and a set as a list.
        value = value.model_dump()
    return _unchanged_by_json(value)


# A wake trigger's kind: what started the turn, shown to an approval reviewer (Runnable._review_gate).
_TRIGGER_KIND = re.compile(r"[a-z][a-z0-9_.:-]{0,63}")
# What the runtime writes on a turn's trigger (a wake's own fields, a session event's, a recovery's): an
# application's wake trigger may not say them (the playground reads event_ids to mark the messages a turn took).
_RESERVED_TRIGGER_KEYS = (
    "wake_id",
    "key",
    "tool_policy",
    "task_ids",
    "event_ids",
    "announce",
    "continuation_index",
    "dead_run_id",
    "cut",
)


def _unchanged_by_json(value: Any) -> Any:
    """What JSON gives back of ``value`` (a copy), if it gives it back unchanged. TypeError otherwise."""
    try:
        # allow_nan=False: NaN and Infinity are not JSON (Python writes and reads them all the same).
        carried = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as error:
        raise TypeError(str(error) or type(error).__name__) from None
    if carried != value:
        # JSON takes it and hands back something else (a tuple as a list, a number key as text).
        raise TypeError(f"{type(value).__name__} does not come back from JSON as it went in")
    return carried  # equal, and the caller's no more: what is kept from here cannot be changed under it


def _dump_resume(values: dict[str, Any] | None) -> dict[str, Any] | None:
    """Answers to a parked turn as a snapshot holds them. TypeError for one it cannot hold."""
    if values is None:
        return None
    stored = {}
    for key, value in values.items():
        try:
            stored[key] = _stored_answer(value)
        except TypeError as error:
            raise TypeError(
                f"The answer to {key!r} cannot be stored ({error}): a durable session takes what JSON can "
                "carry, an ApprovalResolution or a Cancel."
            ) from None
    return stored


def _abandon(call: asyncio.Future) -> None:
    """Stop waiting for ``call``: cancelled, and whatever it ends with is taken (never logged as unretrieved)."""
    call.cancel()
    call.add_done_callback(lambda done: done.cancelled() or done.exception())


async def _answered_within(call: Any, timeout: float) -> bool:
    """Wait for a store call for at most ``timeout`` s. One that has not answered is given up, not waited for
    (``asyncio.wait_for`` waits for it to end its cancellation, which a store that hangs may never do). Its
    error is raised; False when it did not answer."""
    attempt = asyncio.ensure_future(call)
    try:
        done, _ = await asyncio.wait({attempt}, timeout=max(timeout, 0.0))
    except BaseException:
        _abandon(attempt)
        raise
    if not done:
        _abandon(attempt)
        return False
    attempt.result()
    return True


def _load_resume(stored: dict[str, Any] | None) -> dict[str, Any] | None:
    if stored is None:
        return None
    return {
        key: Cancel(reason=value["$cancel"]) if isinstance(value, dict) and "$cancel" in value else value
        for key, value in stored.items()
    }


def _same_answer(one: Any, other: Any) -> bool:
    """The same answer, also when one of the two has been through the store (a typed approval comes back as
    its JSON object)."""
    if one == other:
        return True
    try:
        return _stored_answer(one) == _stored_answer(other)
    except TypeError:
        return False


def _approves(value: Any) -> bool:
    """Whether a resume value lets the gated call it answers run, as the approval gate reads it."""
    if value is True:
        return True
    if isinstance(value, ApprovalResolution):
        return value.approved
    if isinstance(value, dict) and not Cancel.matches(value):
        try:
            return ApprovalResolution.model_validate(value).approved
        except ValueError:  # not a decision the gate accepts: it runs nothing
            return False
    return False


def _status_dict(status: Any) -> dict[str, Any] | None:
    if status is None:
        return None
    if isinstance(status, dict):
        return {k: status.get(k) for k in ("code", "reason", "message")}
    return {"code": status.code, "reason": status.reason, "message": status.message}


def _shown_events(root: Any) -> set[str]:
    """The session events a stored run's memory shows it took, by id: every message built from events carries
    them (``metadata.event_ids``)."""
    memory = root.memory if isinstance(root.memory, list) else []
    return {
        event_id
        for message in memory
        if isinstance(message, dict)
        for event_id in (message.get("metadata") or {}).get("event_ids") or []
    }


def _was_cut(status: Any) -> bool:
    """A run that ended because it was cancelled from outside (a close, an interrupt), as opposed to one
    that ended by itself, parked on an approval or a question included."""
    data = _status_dict(status) or {}
    return data.get("code") == "cancelled" and data.get("reason") == "interrupted"


def _ended_by_itself(turn: asyncio.Task) -> bool:
    """A turn's task that produced its run's outcome, and not because it was cut."""
    if not turn.done() or turn.cancelled() or turn.exception() is not None:
        return False
    output = turn.result()
    return output is not None and not _was_cut(output.status)


def _ended_output(context: RunContext) -> OutputEvent | None:
    """The outcome of a run that ended without its OUTPUT reaching the caller: the root span is closed just
    before the run's last save, and the event is sent after it. None for a run that had not ended."""
    root = context.root_span()
    if root is None or root.t1 is None:
        return None
    output = OutputEvent(
        run_id=context.id,
        parent_run_id=context.parent_id,
        path=root.path,
        call_id=root.call_id,
        parent_call_id=root.parent_call_id,
        input=root.input,
        status=root.status,
        output=root.output,
        error=root.error,
        t0=root.t0,
        t1=root.t1,
        usage=dict(root.usage),
        metadata=root.metadata,
    )
    output._input_dump, output._output_dump = root._input_dump, root._output_dump
    return output


def _ended_parked(turn: asyncio.Task) -> bool:
    """A turn's task whose run ended waiting on an approval or a question."""
    return _ended_by_itself(turn) and is_parked(turn.result().status)


def _left_parked(turn: asyncio.Task | None) -> bool | None:
    """Whether the run a finished turn left ended waiting on an approval or a question.

    None for a turn that left no run (never started, failed before it did): the head is as it was, and so
    is whether it waits for an answer.
    """
    if turn is None or not turn.done() or turn.cancelled() or turn.exception() is not None:
        return None
    output = turn.result()
    return is_parked(output.status) if output is not None else None


def _brief(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _blocks(message: Any, kind: str) -> list[dict[str, Any]]:
    content = message.get("content") if isinstance(message, dict) else None
    return [block for block in content or [] if isinstance(block, dict) and block.get("type") == kind]


def _outcome(span: Any) -> str:
    return _brief(span.error.get("message") if isinstance(span.error, dict) else span.output)


def _tool_calls(context: RunContext, dispatched: list[dict[str, Any]] | tuple = ()) -> tuple[list[str], list[str]]:
    """A dead turn's tool calls: completed (an outcome stored) and unconfirmed (asked for or started, no outcome).

    The model's requests come from its stored calls' outputs as well as the agent's memory: a save can land
    between a model call ending and its tool spans (or the memory) being stored. ``dispatched``: the calls the
    session recorded before the agent ran them (``in_flight.dispatched``). One the trace does not show at all
    (saved before that step, or not since) is unconfirmed too: it may have run. A ``call_tool`` request
    (``tool_search="dispatch"``) ran as the tool it names, under that tool's span: the two are one call.
    """
    from .agent import is_call_tool_name

    root = context.root_span()
    llm_path = f"{root.path}.llm"
    children = [span for span in context._trace.as_records() if span.parent_call_id == root.call_id]
    memory = root.memory if isinstance(root.memory, list) else []
    requests: dict[str, tuple[str, Any]] = {}
    for message in [span.output for span in children if span.path == llm_path] + memory:
        for block in _blocks(message, "tool_use"):
            if block.get("id"):
                requests.setdefault(block["id"], (block.get("name") or "?", block.get("input")))
    for call in dispatched:
        if call.get("id"):
            requests.setdefault(call["id"], (call.get("name") or "?", call.get("input")))
    results = {block.get("id"): block for message in memory for block in _blocks(message, "tool_result")}
    # A span records the tool call it ran for (``metadata.tool_call_id``, set as the tool starts): that is
    # its request, whatever its name. Pairing by name alone gave an earlier dispatched call the span of a
    # later one that failed as ``call_tool`` (an unknown target), and listed its real span a second time.
    # By name only for a span without the id (a trace from before spans recorded it, a tool cut as it started).
    by_request: dict[str, Any] = {}
    spans: dict[str, list[Any]] = {}
    for span in children:
        if span.path == llm_path:
            continue
        request_id = span.metadata.get("tool_call_id") if isinstance(span.metadata, dict) else None
        if isinstance(request_id, str) and request_id in requests and request_id not in by_request:
            by_request[request_id] = span
        else:
            spans.setdefault(span.path.rsplit(".", 1)[-1], []).append(span)
    completed, unconfirmed = [], []
    for request_id, (name, args) in requests.items():
        call = f"{name}({_brief(args, 80)})"
        span = by_request.get(request_id)
        if span is None:
            ran = name
            if is_call_tool_name(name) and not spans.get(name) and isinstance(args, dict):
                target = args.get("name")
                if isinstance(target, str) and spans.get(target):
                    ran = target
            span = spans[ran].pop(0) if spans.get(ran) else None
        if request_id in results:
            text = " ".join(b.get("text", "") for b in _blocks(results[request_id], "text")) or "done"
            completed.append(f"{call} → {_brief(text)}")
        elif span is not None and span.t1 is not None:
            completed.append(f"{call} → {_outcome(span)}")
        else:
            unconfirmed.append(call)
    for name, rest in spans.items():  # started with no request stored
        for span in rest:
            call = f"{name}({_brief(span.input, 80)})"
            (completed if span.t1 is not None else unconfirmed).append(
                f"{call} → {_outcome(span)}" if span.t1 is not None else call
            )
    return completed, unconfirmed


def _recovery_notice(
    completed: list[str],
    unconfirmed: list[str],
    *,
    asked: list[str] | None = None,
    untraced: bool = False,
    answered: bool = False,
) -> str:
    """``asked``: what the dead turn was answering, when no stored run holds it (the model cannot read it in
    the history). ``untraced``: nothing of the turn itself was stored, only what the session recorded.
    ``answered``: it was acting on the user's answer to a parked question, which is closed now."""
    lines = [_RECOVERY_HEADER]
    if asked:
        lines.append("It was answering: " + "; ".join(json.dumps(_brief(text, 500)) for text in asked) + ".")
    if answered:
        lines.append(
            "It was acting on the user's answer to a pending question or approval. That question is closed now: "
            "ask again if it still matters."
        )
    if completed:
        lines.append("Completed before the restart: " + "; ".join(completed) + ".")
    if unconfirmed:
        lines.append("Started but not confirmed (each may or may not have run): " + "; ".join(unconfirmed) + ".")
    if untraced:
        lines.append("The rest of that turn was not recorded before the restart.")
    elif not completed and not unconfirmed:
        lines.append("No tool call had run yet.")
    lines.append(_RECOVERY_ADVICE)
    return "\n".join(lines)


def _stopped_notice(asked: list[str]) -> str:
    """For a dead turn its user had stopped (see :meth:`ChatSession._recover`): what it was answering, kept in
    the conversation, and not to be acted on."""
    return "\n".join(
        [
            _RECOVERY_HEADER,
            "It was answering: " + "; ".join(json.dumps(_brief(text, 500)) for text in asked) + ".",
            "The user stopped that turn before the restart. Do not act on it: answer what they send next, and "
            "ask before doing any of it.",
        ]
    )


def _interrupts_waiting(pending: list[dict[str, Any]] | None, taken: list[dict[str, Any]]) -> bool:
    """A user message that interrupts (``mode="interrupt"``) is stored and queued behind a turn that did not take
    it: sent while that turn ran, to stop it. Stored before it cuts anything, so a process that died in between
    left it as the only record of the stop."""
    taken_ids = {event.get("id") for event in taken}
    return any(
        event.get("source") == "user"
        and event.get("announce") == "interrupt"
        and not event.get("sent_by")
        and event.get("id") not in taken_ids
        for event in pending or ()
    )


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Turn arguments as a snapshot can hold them, and the names of those it cannot (not JSON: a callable,
    a model class): :meth:`ChatSession.from_store` asks for these again."""
    stored, unstored = {}, []
    for name, value in kwargs.items():
        if storable(value):
            stored[name] = value
        else:
            unstored.append(name)
    return stored, sorted(unstored)


class ChatSession:
    """One agent's conversation with a user, open for messages while its turns run.

    :meth:`send` takes a user message at any time. With no turn running it
    starts one. While a turn runs, ``mode`` decides what happens to it:

    - ``"steer"``: the running turn takes it after its current tool batch, as a
      user message, and does not end before answering it (``Agent.steering``,
      on for these runs unless the agent turns it off). Messages a turn ends
      without taking start the next turn together.
    - ``"now"``: like steer, but without waiting: the answer being streamed is
      cut (its text kept, marked interrupted) and running tools that allow it
      (``Tool.interruptible``, default: read-only ones) are cancelled with an
      interrupted result; the others finish first. Messages within a quarter
      second make one cut. Same turn — unlike ``"interrupt"``.
    - ``"followup"``: a turn of its own once the running one ends.
    - ``"collect"``: one turn for every collected message, once none has
      arrived for ``collect_window`` seconds.
    - ``"interrupt"``: cancels the running turn (it ends ``cancelled`` /
      ``interrupted``, keeping what it produced), then starts one with this
      message, ahead of anything else queued.

    ``busy_mode`` is the mode a message gets when none is given. Finished
    background tasks and posted session events start turns too (when their
    announce policy asks for one), as with :class:`AgentContinuation`, and
    :meth:`wake` starts one for the application (a routine, a heartbeat) with
    a narrowed tool set. Those turns may stay silent: told so in their prompt,
    one that answers exactly :data:`SILENT_REPLY` ends ``silent``
    (``allow_silence``). A turn that ends waiting on an approval or
    ``ask_user`` parks the session: new messages wait until :meth:`resume`
    answers it, or :meth:`interrupt` (a message sent with ``"interrupt"``)
    cancels what it waits for.

    :attr:`log` holds every event of every turn, in order, plus session records
    (``{"type": "SESSION", "kind": ...}``): ``message`` (a message was sent),
    ``message_settled`` (a turn took it, or it was superseded), ``wake`` /
    ``wake_superseded``, ``turn_started``, ``turn_finished`` (``status``,
    ``parked``, ``silent``, ``tool_search`` when it loaded deferred tools, ``error`` ``{type, message}``
    when it failed), ``interrupt``, ``closed``. Read it with :meth:`events`.

    Process-local, like the session store it owns. Call its methods on the loop
    it runs on.
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        parent_id: str | None = None,
        busy_mode: SendMode = "steer",
        collect_window: float = 1.0,
        allow_silence: bool = True,
        max_log_events: int | None = DEFAULT_BG_LOG_MAX_EVENTS,
        max_log_bytes: int | None = DEFAULT_BG_LOG_MAX_BYTES,
        durability: Literal["sync", "buffered", "end"] | None = None,
        session_id: str | None = None,
        session_store: ChatSessionStore | None = None,
        on_dead_turn: OnDeadTurn = "continue",
        **kwargs: Any,
    ) -> None:
        if durability not in (None, "sync", "buffered", "end"):
            raise ValueError(f"durability must be 'sync', 'buffered' or 'end', got {durability!r}")
        if on_dead_turn not in ("continue", "notify", "drop"):
            raise ValueError(f"on_dead_turn must be 'continue', 'notify' or 'drop', got {on_dead_turn!r}")
        if "." in agent._path or agent.background_mode != "never":
            raise RuntimeError("A chat session needs a foreground, top-level Agent.")
        if agent.tracing_provider is None:
            raise ValueError("A chat session needs a tracing provider: every turn reads the history of the last.")
        if busy_mode not in SEND_MODES:
            raise ValueError(f"busy_mode must be one of {list(SEND_MODES)}, got {busy_mode!r}")
        if not math.isfinite(collect_window) or collect_window < 0:
            raise ValueError("collect_window must be a finite, non-negative number of seconds.")
        passed = _SESSION_KWARGS.intersection(kwargs)
        if passed:
            raise TypeError(
                f"{sorted(passed)} are not turn arguments: send messages with send(), "
                "answer a parked turn with resume(), and continue a conversation with parent_id=."
            )
        self.id: str = session_id or uuid7(as_type="hex")  # type: ignore[assignment]
        # A caller's session_id is its own key and may be reused for the same conversation: running a turn under it
        # would collide with the earlier thread's root run. Only an id generated here doubles as a run id: in this
        # object, for its first attempt at a first turn (_run_turn). A restored session (from_store) never runs
        # under its id (its first turn may have started under it and died).
        self._generated_id = not session_id
        self.root_run_id: str | None = self.id if session_id and parent_id is None else None
        """The thread's root run as the platform knows it, sent with every MCP call of the session's turns as the
        session hint (``timbal.ai/session_id``, monolith#565). A session opened with its own ``session_id`` and
        no ``parent_id``: that id, which the platform's chat reserves as the thread's root run (no turn runs
        under it here: the first turn gets a run id of its own, stored with no parent). A generated id: the run
        that started the thread, once one has (the first turn runs under the session's id, so it is
        ``session.id``). None while unknown: continued from ``parent_id``, or restored from a snapshot written
        before the field existed (the first turn of one that had none sets it). Stored in the snapshot."""
        self.busy_mode: SendMode = busy_mode
        self.on_dead_turn: OnDeadTurn = on_dead_turn
        # As what they mean, whatever number or truth value the caller passed: a snapshot holds them.
        self.collect_window = float(collect_window)
        self._turn_failures = 0
        self._attempted: set[Any] = set()
        self.allow_silence = bool(allow_silence)
        self.log = BackgroundEventLog(max_events=max_log_events, max_bytes=max_log_bytes)
        self.closed = False
        self._parked = False
        self.turns = 0
        self.last_status: dict[str, Any] | None = None
        self.last_activity = time.monotonic()
        self._agent = agent
        self._kwargs = kwargs
        self._stored_kwargs, self._unstored_kwargs = _split_kwargs(kwargs)
        self._durability = durability
        self._providers: dict[tuple[type, str], type] = {}
        store = store_for_run(parent_id) if parent_id is not None else None
        self._store = store if store is not None else BackgroundTaskStore()
        if store is None and parent_id is not None:
            # The session continues from a run that had no store: a turn another channel chains onto that
            # run (a call joining, a plain client) is the session's, and asks this channel first.
            session_continues_from(self._store, parent_id)
        self._scheduler = SessionScheduler(
            channel=self, on_busy="enqueue", can_interrupt=True, on_work=self._work_arrived, lends=True
        )
        self._scheduler.join(self._store)
        self._scheduler.head = parent_id
        self._work = asyncio.Event()
        self._resume: dict[str, Any] | None = None
        self._wakes: list[_Wake] = []
        self._last_prompt: Any = None
        self._last_options = _TurnOptions()
        self._worker: asyncio.Task | None = None
        self._turn: asyncio.Task | None = None
        self._turn_context: RunContext | None = None
        self._last_context: RunContext | None = None
        self._restore_from = parent_id
        self._restoring: asyncio.Task | None = None
        self._interrupt_on_restore = False
        self._save_cut = False
        # Durable sessions (session_store=): a snapshot written ahead of every change.
        self._session_store = session_store
        self._version: int | None = None
        self._dirty = False
        self._saver: asyncio.Task | None = None
        self._leased = False
        self._lease_task: asyncio.Task | None = None
        # When the request that last confirmed the lease was sent (monotonic): the store counted the lease
        # from no earlier than that, so it holds at least until this plus `lease_ttl_s`.
        self._lease_from: float | None = None
        self._lost = False
        # Set once close() has finished: turn cancelled, scheduler and task store let go, the store settled.
        self._close_done = asyncio.Event()
        self._keep = False
        self._in_flight: dict[str, Any] | None = None
        # The latest failed snapshot write (cleared by the next good one): flush() reports it.
        self._save_error: Exception | None = None
        # Set by close() when the store could neither mark the session closed nor drop its row (a restore
        # would bring it back): the error.
        self.close_error: Exception | None = None
        # Changes made (every _persist) and how many of them the last good write covers: stored_since().
        self._changes = 0
        self._stored_changes = 0
        # How many changes the write that is out right now covers (None: none is): may_be_stored().
        self._saving: int | None = None
        # close(store_timeout=) gave up waiting for the store.
        self.close_timed_out = False
        # from_store() is still settling the stored state: no turn may start before it is done.
        self._hold_worker = False
        # Set by _recover: why the parked question a dead resume was acting on is to be abandoned
        # (from_store), and the events the dead run's stored memory shows (taken by it, whatever an older
        # snapshot says).
        self._abandon_parked: str | None = None
        self._shown_events: set[str] = set()
        # from_store(), in the process that closed the session: the events its task store still held.
        self._kept_events: dict[str, SessionEvent] = {}
        # Turns in a row the store would not record the start of: they did not run (see _write_ahead).
        self._store_failures = 0
        # Reads in a row that failed for the question a call left the session with (see _take_question).
        self._question_failures = 0
        # The head a call left parked whose stored trace held no question when read (_dropped_question_holds).
        self._dropped_question: str | None = None
        # close(keep=True): the ids of the events pending in the task store when it let go of it (the
        # user's messages among them marked SessionEvent.kept_for). From then on only those are this
        # session's (see _mine).
        self._left: frozenset[str] | None = None
        # from_store(), if the restore fails: the messages that were kept for this session when it took
        # them back (the only ones its closing marks again).
        self._only_keep: frozenset[str] | None = None
        if self._scheduler.lent and any(self._scheduler.takes(event) for event in self._store.pending_events()):
            # Opened on a live call that holds something its user wrote: this chat's to answer from now on
            # (the call leaves it to the chat), and nothing else would start the worker.
            self._work.set()
            with contextlib.suppress(RuntimeError):  # no loop running yet: the first send() starts it
                self._ensure_worker()
        # The restorable events last seen queued: one arriving through the task store is stored at once.
        self._queued_events: tuple[str, ...] = ()
        # App and client events still queued that the session follows until a turn takes them (_work_arrived).
        self._followed_events: set[str] = set()

    # -- Public API -------------------------------------------------------------

    @property
    def head(self) -> str | None:
        """The session's latest run: where the next turn continues from."""
        return self._scheduler.head

    @property
    def busy(self) -> bool:
        """A turn is running, or has just ended and is not settled yet: its status, and whether it left the
        session parked, are recorded in the same step that ends this (never "not busy" with the last
        turn's outcome still missing)."""
        return self._turn is not None

    @property
    def run_id(self) -> str | None:
        """The running turn's run id, or None."""
        return self._turn_context.id if self.busy and self._turn_context is not None else None

    @property
    def store(self) -> BackgroundTaskStore:
        return self._store

    def send(self, text: str, *, mode: SendMode | None = None, defer_cut: bool = False) -> SessionEvent:
        """Post a user message (see the class docstring for ``mode``); returns its receipt at once.

        ``defer_cut``: ``now`` and ``interrupt`` only queue the message; the caller cuts the running turn
        with :meth:`cut_for` once the message is stored (a route that may still answer "not accepted").
        """
        if self.closed:
            raise RuntimeError("The chat session is closed.")
        mode = mode or self.busy_mode
        if mode not in SEND_MODES:
            raise ValueError(f"mode must be one of {list(SEND_MODES)}, got {mode!r}")
        busy = self.busy
        if mode == "interrupt":
            event = self._store.post_event(text, source="user", mode="followup", announce="interrupt")
        else:
            # 'now' joins the running turn like steer, after cutting what is in flight.
            event = self._store.post_event(text, source="user", mode="steer" if mode == "now" else mode)
        self.last_activity = time.monotonic()
        self._record("message", event_id=event.id, text=event.text, mode=mode, busy=busy)
        self._watch(event)
        self._ensure_worker()
        if not defer_cut:
            self.cut_for(event, mode)
        return event

    def cut_for(self, event: SessionEvent, mode: SendMode | None = None) -> bool:
        """Cut the running turn for a message sent with ``defer_cut`` (``mode`` as it was sent): ``interrupt``
        cancels it, ``now`` cuts the answer being streamed. Nothing (False) for the other modes, and once a
        turn has taken the message: the cut was for what ran before it, not for its own answer."""
        mode = mode or self.busy_mode
        # Still queued: not taken by a turn that is about to start either (its receipt is pending until then).
        if self.closed or not any(pending.id == event.id for pending in self._store.pending_events()):
            return False
        if mode == "interrupt":
            return self.interrupt()
        if mode == "now" and self.busy and self._turn_context is not None and self._turn_context._cut is not None:
            self._turn_context._cut.set()
            return True
        return False

    def wake(
        self,
        text: str,
        *,
        key: str | None = None,
        tool_policy: ToolPolicy | None = "read_only",
        silent_ok: bool = True,
        trigger: dict[str, Any] | None = None,
    ) -> str:
        """Start a turn for the application (a routine, a heartbeat): returns the wake's id.

        It runs once nothing the user sent or the session owes is waiting, as
        a turn of its own (never folded into a user's), with only the tools
        ``tool_policy`` allows (default ``"read_only"``: proactive work that
        cannot send, write or buy; ``None`` for all tools). With ``silent_ok``
        (and ``allow_silence``) it may answer :data:`SILENT_REPLY` to say
        nothing. A pending wake with the same ``key`` is replaced. The text is
        trusted application text: input guardrails do not screen it.
        ``trigger`` is merged over the turn's root ``metadata.trigger``
        (``{kind: "wake", wake_id, key, tool_policy}``); a routine fire passes
        ``{"kind": "routine", "routine_id": ..., "fire_id": ..., "scheduled_at": ...}``.
        """
        if self.closed:
            raise RuntimeError("The chat session is closed.")
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            raise ValueError("A wake needs non-empty text.")
        if key is not None and not isinstance(key, str):
            raise TypeError("key must be a string or None")
        policy = _check_tool_policy(tool_policy)
        key = (key or "").strip() or None
        if trigger is not None and not isinstance(trigger, dict):
            raise TypeError("trigger must be a dict")
        if trigger:
            reserved = [name for name in _RESERVED_TRIGGER_KEYS if name in trigger]
            if reserved:
                raise ValueError(
                    f"A wake's trigger cannot set what the runtime says of the turn: {', '.join(reserved)}."
                )
            kind = trigger.get("kind", WAKE_KIND)
            if not isinstance(kind, str) or not _TRIGGER_KIND.fullmatch(kind) or kind in ATTENDED_TRIGGER_KINDS:
                # An approval reviewer reads it (Runnable._review_gate), and shows it to a model reviewer: a short
                # identifier of what started the turn, never a person's turn.
                raise ValueError(
                    "A wake's trigger kind is a short lowercase identifier of what started the turn, not a user's turn "
                    f"({_TRIGGER_KIND.pattern}): {repr(kind)[:80]}."
                )
        owned = None
        if trigger and self._session_store is not None:
            # Stored with the wake: one a snapshot cannot hold would fail every save from now on. Refused here,
            # to its caller, with nothing queued (as for resume()). Kept as JSON gave it back: the caller's dict
            # (and what is nested in it) may change after this returns, and the queued wake must not.
            try:
                owned = _unchanged_by_json(trigger)
                json.dumps(owned, ensure_ascii=False).encode("utf-8")  # a lone surrogate is no UTF-8
            except (TypeError, UnicodeEncodeError) as error:
                raise TypeError(
                    f"The wake's trigger cannot be stored ({error}): a durable session takes what JSON can carry."
                ) from None
        replaced = [wake for wake in self._wakes if wake.key == key] if key is not None else []
        for old in replaced:
            self._wakes.remove(old)
            if self.stored_since(old.mark):
                old.replaced = []  # stored: it is what comes back, and what it replaced is gone for good
            # Not stored yet (its sender may still be told it was not accepted): it keeps what it replaced,
            # so the last stored wake of the key survives however many replacements of it fail.
            self._record("wake_superseded", wake_id=old.id, key=key)
        wake = _Wake(text=text, key=key, tool_policy=policy, silent_ok=bool(silent_ok), replaced=replaced)
        if trigger:
            # The caller's fields, then the wake's own, which no trigger renames (its kind may be the caller's).
            # Its own copy (a session without a store: deep where it can be), never the caller's objects.
            wake.trigger = {
                "kind": WAKE_KIND,
                **(owned if owned is not None else _own_copy(trigger)),
                "wake_id": wake.id,
                "key": key,
                "tool_policy": _describe_policy(policy),
            }
        self._wakes.append(wake)
        self.last_activity = time.monotonic()
        # The log's own copy: whoever reads the log cannot change the queued wake.
        self._record(
            "wake",
            wake_id=wake.id,
            text=text,
            key=key,
            tool_policy=_describe_policy(policy),
            trigger=_own_copy(wake.trigger),
        )
        wake.mark = self._changes
        self._work.set()
        self._ensure_worker()
        return wake.id

    def message_task(self, task_id: str, text: str) -> SessionEvent:
        """Send one of the session's running background agents a message from the user.

        It reads it after its current step (:meth:`BackgroundTaskStore.message_task`
        raises for an unknown, finished or non-agent task). Background tasks are
        in-process: a durable session does not keep the message across a restart.
        """
        if self.closed:
            raise RuntimeError("The chat session is closed.")
        event = self._store.message_task(task_id, text, source="user")
        self.last_activity = time.monotonic()
        self._record("task_message", task_id=task_id, event_id=event.id, text=event.text)
        # What became of it, for the user's side: the parent is not shown a user's message the task
        # never read (nothing screened it), so this record is where the user learns it did not arrive.
        # Through a weak reference: the task holds the receipt until it reads the message or ends, and
        # it outlives the session (close() leaves detached children running). A callback that held
        # the session would keep every closed chat that wrote to a long-running task, log and all.
        session = weakref.ref(self)

        def settled(event: SessionEvent) -> None:
            live = session()
            if live is not None:
                live._task_message_settled(task_id, event)

        event.on_settle(settled)
        return event

    def _task_message_settled(self, task_id: str, event: SessionEvent) -> None:
        if self._close_done.is_set():
            return
        self._record("task_message_settled", task_id=task_id, event_id=event.id, state=event.state, run_id=event.run_id)

    def interrupt(self) -> bool:
        """Cut what the session is doing; queued messages then start the next turn.

        Cancels the running turn, or, when the session is parked, every pending
        approval and question not answered yet (``Cancel``: the gated tools never
        run, the run ends ``cancelled``; answers already given stand). False when
        there is neither. Before :meth:`restore` has learned whether the session is
        parked, it is applied once it has (True).
        """
        if self._restore_pending():
            # Whether the session is parked is not known yet: restore() applies the interrupt.
            self._interrupt_on_restore = True
            self._ensure_worker()
            return True
        return self._interrupt_now()

    def withdraw(self, work_id: str) -> bool:
        """Take back a message (event id) or wake (wake id) no turn has taken yet; False once one has.

        For a caller that could not get it stored (:meth:`flush` raised): withdrawn, a retry cannot run it twice.
        """
        if any(event.id == work_id and not self._mine(event) for event in self._store.pending_events()):
            return False  # another session's, kept for its restore
        withdrawn = self._store.withdraw_event(work_id)
        if not withdrawn:
            wake = next((wake for wake in self._wakes if wake.id == work_id), None)
            if wake is not None:
                # The wakes it replaced come back in its place: as if it had never been sent.
                at = self._wakes.index(wake)
                self._wakes[at : at + 1] = wake.replaced
                self._record("wake_withdrawn", wake_id=wake.id, restored=[old.id for old in wake.replaced])
                withdrawn = True
            else:
                # Replaced by a newer wake of its key before its sender knew it was stored: it must not come
                # back if that one is withdrawn too. What it had replaced takes its place.
                withdrawn = self._drop_replaced(self._wakes, work_id)
        if withdrawn:
            self._persist()
        return withdrawn

    def _drop_replaced(self, holders: list[_Wake], wake_id: str) -> bool:
        for holder in holders:
            for at, old in enumerate(holder.replaced):
                if old.id == wake_id:
                    holder.replaced[at : at + 1] = old.replaced
                    return True
            if self._drop_replaced(holder.replaced, wake_id):
                return True
        return False

    def mark(self) -> int:
        """A point in the session's changes, for :meth:`stored_since`: take it right after making one."""
        return self._changes

    def stored_since(self, mark: int) -> bool:
        """Whether the store holds the session as of ``mark`` or later (always, when it is not durable).

        After a :meth:`flush` that raised: True means the caller's own change is stored and a later one is
        what failed; False means the store holds nothing of it.
        """
        return self._session_store is None or self._stored_changes >= mark

    def may_be_stored(self, mark: int) -> bool:
        """Whether the store holds the session as of ``mark``, or a write that covers it is still out.

        For a caller that stopped waiting for :meth:`flush` (a store that hangs, not one that fails): False
        means no write holds the change, so taking it back leaves nothing of it anywhere. True means the
        write may land at any time, or already has: taking the change back and telling its sender to try
        again could run it twice.
        """
        return self.stored_since(mark) or (self._saving is not None and self._saving >= mark)

    def withdraw_answers(self, values: dict[str, Any]) -> bool:
        """Take back answers sent with :meth:`resume` that no turn has taken yet; False once one has."""
        queued = self._resume
        if queued is None or any(
            key not in queued or not _same_answer(queued[key], value) for key, value in values.items()
        ):
            return False
        rest = {key: value for key, value in queued.items() if key not in values}
        self._resume = rest or None
        self._persist()
        return True

    def _restore_pending(self) -> bool:
        return self._restore_from is not None or (self._restoring is not None and not self._restoring.done())

    def _interrupt_now(self) -> bool:
        turn = self._turn
        if turn is not None and not turn.done():
            self._record("interrupt", run_id=self.run_id)
            if self._in_flight is not None:
                # Kept with the turn: the cut may reach a run that has just ended on a question (during its
                # last save), or a resume that never got to use its answers. Either leaves the session
                # parked, and the stop then means that question (_run_turn, or _recover after a restart).
                self._in_flight["stopped"] = True
                self._persist()
            turn.cancel()
            return True
        reason = "The user moved on before answering."
        if turn is not None:
            # The turn has ended and is not settled yet (its trace is being stored): nothing of it is left to
            # cut. If it ended on a question, that is what the stop is about, as on a parked session: the
            # answer is stored now and taken once the turn is settled (resume() itself waits for that).
            # Dropped here, the session would park on a question its user has just walked away from, and a
            # later approval would run the gated tool.
            context = self._turn_context
            if context is None or not _ended_parked(turn):
                return False
            pending = self._cancel_pending(context, reason)
            if not pending:
                return False
            self._record("interrupt", run_id=context.id, abandoned=pending)
            self.last_activity = time.monotonic()
            self._persist()
            return True
        if self.parked and self._last_context is not None:
            pending = self._unanswered(self._last_context)
            if not pending:
                return False
            self._record("interrupt", run_id=self._last_context.id, abandoned=pending)
            self.resume({pending_id: Cancel(reason=reason) for pending_id in pending})
            return True
        return False

    def _unanswered(self, context: RunContext) -> list[str]:
        """What a run that ended on approvals / questions still waits for: the ids no queued answer covers."""
        answered = self._resume or {}
        pending = [a["approval_id"] for a in context.pending_approvals()]
        pending += [i["interaction_id"] for i in context.pending_interactions()]
        return [pending_id for pending_id in pending if pending_id not in answered]

    def _cancel_pending(self, context: RunContext, reason: str) -> list[str]:
        """Queue a cancel for everything ``context`` still waits for (answers already given stand): the gated
        tools never run, the run ends ``cancelled``. The ids it cancelled."""
        pending = self._unanswered(context)
        if pending:
            self._resume = {**(self._resume or {}), **{pending_id: Cancel(reason=reason) for pending_id in pending}}
        return pending

    @property
    def parked(self) -> bool:
        """The last turn ended waiting on an approval or ``ask_user``: :meth:`resume` answers it.

        False again once another client's run of the session answers it or goes past it (the head moved).
        """
        context = self._last_context
        # The session's head: while a call has the session, the call's, not the one this channel last saw.
        return self._parked and context is not None and self._scheduler.session_head == context.id

    def resume(self, values: dict[str, Any]) -> None:
        """Answer the parked turn: ``{approval_id: ...}`` / ``{interaction_id: value}``, as for ``resume=``.

        A durable session stores the answers before a turn takes them, so it takes what a snapshot can hold:
        JSON values (as over HTTP), :class:`~timbal.types.approval.ApprovalResolution` and
        :class:`~timbal.types.approval.Cancel`. TypeError for anything else, with nothing queued.
        """
        if self.closed:
            raise RuntimeError("The chat session is closed.")
        if self.busy:
            # Its answers, if it ends waiting for some, are taken once it does (it is busy until the turn
            # is settled: ended, and its trace stored).
            raise RuntimeError("A turn of this session is running; answer once it waits for an answer.")
        if not self.parked and not self._restore_pending():
            # Before restore() has finished, the answers wait for it: dropped if the run is not parked after all.
            raise RuntimeError("No turn of this session is waiting for an answer.")
        if self._session_store is not None:
            # Before anything is queued: an answer the store cannot hold is refused here, to its caller, not
            # accepted and then lost (or written as text) by a save in the background.
            _dump_resume(values)
        queued = self._resume or {}
        conflicts = sorted(
            key for key, value in values.items() if key in queued and not _same_answer(queued[key], value)
        )
        if conflicts:
            raise ValueError(f"Already answered differently: {', '.join(conflicts)}.")
        # Answers sent before the worker takes them go together (two approvals of one parked turn).
        self._resume = {**queued, **values}
        self.last_activity = time.monotonic()
        self._persist()
        self._work.set()
        self._ensure_worker()

    def events(self, after: int = 0, *, numbered: bool = False) -> AsyncIterator[Any]:
        """Replay the log from cursor ``after``, then follow it until :meth:`close`.

        ``numbered``: ``(cursor, item)`` pairs instead, see :meth:`BackgroundEventLog.subscribe`.
        """
        return self.log.subscribe(after, numbered=numbered)

    def _mine(self, event: SessionEvent) -> bool:
        """Whether a pending ``event`` is this session's to list, store, mark, withdraw or take back.

        Not a message another session keeps for its own restore (``SessionEvent.kept_for``): that one is in
        the other session's snapshot. And once this session has closed to be restored, only what was
        pending when it let go of the task store (``_left``): what arrives after that is another
        channel's (a call it shared with answers or speaks it), and stored here it would be delivered
        again when the session comes back. The snapshot of a closed session is written after that point.
        """
        if self._left is not None:
            return event.id in self._left
        return event.kept_for is None or event.kept_for == self.id

    def pending_messages(self) -> list[dict[str, Any]]:
        """Receipts of the user messages no turn has taken yet, oldest first."""
        return [
            {**event.receipt(), "text": event.text}
            for event in self._store.pending_events()
            if event.source == "user" and self._mine(event)
        ]

    def state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "session_id": self.id,
            "head": self.head,
            "busy": self.busy,
            "run_id": self.run_id,
            "parked": self.parked,
            "turns": self.turns,
            "busy_mode": self.busy_mode,
            "pending": self.pending_messages(),
            "wakes": [{"wake_id": wake.id, "text": wake.text, "key": wake.key} for wake in self._wakes],
            "last_status": self.last_status,
            "log_cursor": self.log.cursor_end,
            "durable": self._session_store is not None and not self._lost,
        }
        if self.parked and self._last_context is not None:
            state["pending_approvals"] = self._last_context.pending_approvals()
            state["pending_interactions"] = self._last_context.pending_interactions()
        return state

    @property
    def due(self) -> bool:
        """Work the session will start a turn for by itself: answers to its parked question, a wake, and
        whatever its scheduler would start one for — a message (one still collecting included), an event or
        a finished background task whose announce policy asks for a turn, and a background task still
        running whose result will.

        What only rides with the next turn (a silent event, the result of a silent task) is not due: it
        starts nothing. Neither is a parked question: it waits for its user, not for the session. False here
        means a caller may let the session go (a durable one is kept in its store) without leaving work
        nothing will run.
        """
        if self._resume is not None:
            return True
        if self.parked:
            return False
        # The scheduler's own test for "a turn may still come" (it ends the worker's wait on it): the same
        # announce policy decides here, so the two cannot disagree about a session that is still owed a turn.
        return bool(self._wakes) or self._scheduler.may_need_turn()

    @property
    def lost(self) -> bool:
        """Another process took the session (its lease, or a newer snapshot): this copy stopped without
        writing over it, and what it was asked last may not have been stored."""
        return self._lost

    @property
    def handed_off(self) -> bool:
        """This process no longer serves the session, but its store still holds it: another process took it,
        or it was closed with ``keep``. A server drops such an object and restores the session from the store."""
        return self._session_store is not None and (self._lost or (self.closed and self._keep))

    async def close(self, *, keep: bool = False, store_timeout: float | None = None) -> None:
        """Stop the session: the running turn is cancelled, the store let go. Detached children keep running.

        A durable session's snapshot is deleted, unless ``keep``: this process is going away (shutdown, an idle
        session let go), and :meth:`from_store` brings the session back. A turn this cuts is then recovered there
        like one that died with its process.

        ``store_timeout``: how long to wait for the session store (its last write, the delete, letting the
        lease go). A store that hangs would hold the close for as long. Past it the session is closed here
        all the same and the store is left as it stands (:attr:`close_timed_out`; without ``keep`` also
        :attr:`close_error`, since the row may still say the session is open). None waits.
        """
        if self.closed:
            return
        cut = self._in_flight if keep and self.busy else None
        self.closed = True
        self._keep = keep
        try:
            worker, self._worker = self._worker, None
            if worker is not None and not worker.done():
                worker.cancel()
                with contextlib.suppress(BaseException):
                    await worker
            if cut is not None and self._in_flight is cut:
                # Still the turn in flight: this close cut it. One that ended meanwhile was settled as it
                # stopped (its in-flight record is gone), and is not a cut turn.
                self._in_flight = {**cut, "cut": "shutdown"}
            if self._last_context is not None:
                self._scheduler.unmark(self._last_context)
            if keep and self._session_store is not None:
                # Kept in its store: restored later with what is pending now, which its snapshot holds.
                # What the user typed and it has not answered is marked as this session's (kept_for):
                # another channel on the session (a call it shares with, a chat opened meanwhile) must not
                # answer that too. What arrives from here on is not this session's (_left).
                left = []
                for event in self._store.pending_events():
                    if event.task_id is not None or not self._mine(event):
                        continue
                    if event.source == "user":
                        if self._only_keep is not None and event.id not in self._only_keep:
                            continue  # a restore that failed: a message sent to the call since is not its own
                        event.kept_for = self.id
                    left.append(event.id)
                self._left = frozenset(left)
            self._scheduler.release(self._store)
            forget_idle_background_store(self._store)
            self._record("closed", kept=keep)
            self.log.close()
            self._stop_lease()
            if store_timeout is None:
                await self._settle_store()
            else:
                try:
                    await asyncio.wait_for(self._settle_store(), max(store_timeout, 0))
                except TimeoutError:
                    self._store_did_not_answer(store_timeout)
        finally:
            self._close_done.set()

    def _store_did_not_answer(self, waited: float) -> None:
        """The close stopped waiting for the store: nothing more is written from here."""
        self.close_timed_out = True
        logger.error("chat_session_close_store_timeout", session_id=self.id, timeout_s=waited, kept=self._keep)
        saver = self._saver
        if saver is not None and not saver.done():
            # This object is done: a write that came back later would only be followed by more of them.
            saver.cancel()
        if not self._keep:
            # The row may still say the session is open (a restore would bring it back): its caller is told.
            self.close_error = TimeoutError(f"the session store did not answer within {waited:g} s")

    async def wait_closed(self) -> None:
        """Until :meth:`close` has finished (one already called, or the one losing ownership starts). Restore a
        :attr:`handed_off` session only after it: before, its turn may still run and its task store be bound."""
        await self._close_done.wait()

    @classmethod
    async def from_store(
        cls, agent: "Agent", session_store: ChatSessionStore, session_id: str, **kwargs: Any
    ) -> "ChatSession":
        """Bring back a durable session another process (or an earlier life of this one) left in ``session_store``.

        Queued messages and wakes come back with their ids; a parked turn can be answered with :meth:`resume`;
        clients reconnecting to :meth:`events` with an old cursor get a gap and continue from the new head
        (the log itself is not stored: the turns are, in the traces). A turn that was running when the old
        process died is settled by ``on_dead_turn``:

        - it had finished (the crash came between its trace and the snapshot): adopted as is;
        - it had saved nothing: its messages (or its wake) are queued again;
        - otherwise (cut mid-run) ``"continue"`` (default) runs a read-only recovery turn chained on it, told
          which tool calls completed and which were started but not confirmed, and to check before redoing
          those: nothing is re-run without the model deciding to. ``"notify"`` queues that notice for the
          next turn instead, ``"drop"`` forgets it.

        Raises LookupError (no such session), ValueError (another agent's) or
        :class:`~timbal.state.chat_store.SessionLeased` (another live process serves it).
        """
        # The lease first, then the row: read before it, a previous owner could still write (and let go)
        # in between, and this process would serve an older state than the store holds.
        session: ChatSession | None = None
        leased_at = time.monotonic()
        try:
            await session_store.acquire_lease(session_id, process_owner())
            snapshot = await session_store.load(session_id)
            if snapshot is None or snapshot.closed:
                raise LookupError(f"No stored chat session {session_id!r}.")
            if snapshot.agent != agent.name:
                raise ValueError(
                    f"Chat session {session_id!r} belongs to agent {snapshot.agent!r}, not {agent.name!r}."
                )
            config = dict(snapshot.config)
            turn_kwargs = config.pop("kwargs", None) or {}
            missing = sorted(set(config.pop("kwargs_unstored", None) or ()) - set(kwargs))
            if missing:
                raise ValueError(
                    f"Chat session {session_id!r} runs its turns with arguments a snapshot cannot hold "
                    f"({', '.join(missing)}): pass them to from_store() again."
                )
            config = {**turn_kwargs, **config, **kwargs}
            session = cls(agent, parent_id=snapshot.head, session_id=session_id, session_store=session_store, **config)
            # Queuing the stored events would start the worker: not before the dead turn is settled (its recovery
            # goes first, and a turn finishing meanwhile would store a snapshot without it).
            session._hold_worker = True
            worker, session._worker = session._worker, None
            if worker is not None:
                # Started by the constructor for a message the task store holds (a call has the session):
                # it has not run a step yet, and does not until the stored state is settled.
                worker.cancel()
            # Restored in the process that closed it, the run's task store may still be there (a background
            # task going, an event no turn took), holding the session's events: in the order they were left,
            # followed by the closed session. They are taken out and queued again below from the snapshot,
            # as in a process that never had them: what a dead turn took goes back ahead of what was queued
            # behind it (and loses to a newer event with its key, not the other way round), each one
            # followed by this session. The events themselves are kept, so their senders' receipts settle.
            session._kept_events = {
                event.id: event
                for event in session._store.drain_events(
                    settle=False, where=lambda event: event.task_id is None and session._mine(event)
                )
            }
            # If this restore fails: what its closing marks as kept again (restore_event clears the mark as
            # it queues them). Not a message sent to another channel since, which was never this session's.
            session._only_keep = frozenset(
                event.id for event in session._kept_events.values() if event.kept_for == session_id
            )
            session._version, session._leased, session._lease_from = snapshot.version, True, leased_at
            session.root_run_id = snapshot.root_run_id
            session.turns, session.last_status, session._parked = snapshot.turns, snapshot.last_status, snapshot.parked
            session._last_prompt = _load_prompt(snapshot.last_prompt)
            session.log.restart_at(snapshot.log_cursor + _LOG_RESTART_GAP)
            session._wakes = [_Wake.from_dict(data) for data in snapshot.wakes]
            session._resume = _load_resume(snapshot.resume)
            recovered = None
            if snapshot.in_flight is not None:
                # First: the messages a dead turn took were sent before anything still pending, and go back
                # ahead of it.
                recovered = await session._recover(snapshot.in_flight, snapshot.last_options, snapshot.pending)
            # Not the ones the dead run's stored memory shows: it took them, and the snapshot that would have
            # said so was never written.
            session._restore_events([event for event in snapshot.pending if event["id"] not in session._shown_events])
            session._requeue_kept_events(snapshot.in_flight["run_id"] if snapshot.in_flight is not None else None)
            if session._parked and session._last_context is None:
                session._last_context = await session._load_run(session.head)
                if session._last_context is None:
                    # The run that asked is not in the trace store (traces kept elsewhere and gone, a run
                    # the platform no longer has): nothing can answer it. Not parked: the session goes on
                    # from there, and answers still queued for it are dropped by the worker.
                    logger.warning("chat_session_parked_run_missing", session_id=session_id, run_id=session.head)
                    session._parked, session._abandon_parked = False, None
            if session._abandon_parked:
                session._abandon_question(session._abandon_parked)
            parked_trace = session._last_context._trace if session._parked and session._last_context else None
            last_policy = snapshot.last_options.get("tool_policy")
            session._last_options = _TurnOptions(
                # Only a resume of a parked turn uses them. A callable's resumes with the tools its agents
                # were offered (by path); not parked, nothing will run under it (no tools, no warning).
                tool_policy=_load_policy(last_policy, parked_trace)
                if session._parked or last_policy != "custom"
                else frozenset(),
                silence=bool(snapshot.last_options.get("silence")),
                # A parked wake or recovery turn is resumed without steering, as it ran.
                steer=snapshot.last_options.get("steer", True) is not False,
            )
        except BaseException:
            # Nothing was written: the stored state stays as it was, for the next attempt (or another process).
            # Cancelled included (a server shutting down): the lease is let go, also one still being taken.
            if session is not None:
                # What was taken out of the task store and not queued again yet: back where it was.
                session._store.requeue([], list(session._kept_events.values()))
                session._kept_events = {}
                session._lost = True  # stores nothing while it lets go of the task store
                with contextlib.suppress(BaseException):
                    await session.close(keep=True)
            with contextlib.suppress(Exception):
                # For as long as the lease lasts, no longer: once it has run out there is nothing to let go
                # of, and a store that does not answer would hold the caller (a cancelled restore, a server
                # shutting down) for good.
                left = leased_at + session_store.lease_ttl_s - time.monotonic()
                await _answered_within(session_store.release_lease(session_id, process_owner()), left)
            raise
        if session.parked:
            # Restored on a question: the session is not lent to a call while it waits (see SessionScheduler).
            session._scheduler.note_parked(session._last_context.id)
        session._only_keep = None  # restored: a later close keeps whatever is unanswered then
        session._record("restored", head=session.head, recovered=recovered, pending=len(snapshot.pending))
        session._persist()
        session._lease_task = asyncio.get_running_loop().create_task(session._renew_lease())
        session._hold_worker = False
        if session.parked and any(
            event.source == "user" and event.announce == "interrupt" and session._mine(event)
            for event in session._store.pending_events()
        ):
            # An 'interrupt' message is stored before it cuts anything: if the process died in between, the
            # question it was to abandon still waits. Abandon it now (nothing to do if that was stored too).
            session._interrupt_now()
        # _work: something arrived while the worker was held (a finished background task, an event).
        if session._wakes or session._resume is not None or session._store.pending_events() or session._work.is_set():
            session._work.set()
            session._ensure_worker()
        return session

    async def __aenter__(self) -> "ChatSession":
        await self.restore()
        return self

    async def restore(self) -> None:
        """Pick up the state of the run the session continues (``parent_id``): one parked on an approval or
        ``ask_user`` leaves the session parked, so messages wait and :meth:`resume` answers it (a turn chained
        onto it would resume it and drop its prompt). Once; ``async with`` and the first turn call it."""
        if self._restoring is None or (self._restoring.done() and self._restore_from is not None):
            self._restoring = asyncio.get_running_loop().create_task(self._restore_parked())
        await asyncio.shield(self._restoring)

    async def _restore_parked(self) -> None:
        try:
            await self._load_parked()
        finally:
            if self._interrupt_on_restore and self._restore_from is None:
                # interrupt() came before the session knew whether it was parked.
                self._interrupt_on_restore = False
                self._interrupt_now()

    async def _load_parked(self) -> None:
        run_id, self._restore_from = self._restore_from, None
        if run_id is None or self._last_context is not None:
            return
        try:
            # Not get() / _get_parent_trace: they read a failed lookup as "no parent", and an unparked session
            # would resume the parked run with the next message's turn and drop that message.
            found = await self._parked_run(run_id)
        except Exception as error:
            self._restore_from = run_id
            self._record("restore_failed", run_id=run_id, error=str(error))
            raise
        if found is None or self._last_context is not None:
            return
        self._park_on(*found)

    async def _parked_run(self, run_id: str) -> tuple[RunContext, Any] | None:
        """The stored run ``run_id`` if it ended on an approval or a question: a context holding its trace,
        and its root span. Raises when the run cannot be read."""
        # The context resolves the agent's setting (TRACING_UNSET: auto-detected) into the provider runs use.
        lookup = RunContext(parent_id=run_id, tracing_provider=self._agent.tracing_provider)
        provider = lookup._tracing_provider
        if provider is None:
            return None
        trace = await provider.get_strict(lookup)
        root = next((span for span in trace.values() if span.parent_call_id is None), None) if trace else None
        if root is None or not is_parked(root.status):
            return None
        context = RunContext(id=run_id, tracing_provider=self._agent.tracing_provider)
        context.__dict__["_trace"] = trace
        return context, root

    def _park_on(self, context: RunContext, root: Any, *, handed_over: bool = False) -> None:
        """The question ``context``'s run ended on is this session's to answer: :meth:`resume`, :meth:`interrupt`."""
        prompt = (root.input or {}).get("prompt") or ""
        self._last_context, self._parked, self._last_prompt = context, True, prompt
        # Its question is this session's to answer: not lent to a call while it waits (see SessionScheduler).
        self._scheduler.note_parked(context.id)
        meta = prompt.get("metadata") if isinstance(prompt, dict) else getattr(prompt, "metadata", None)
        if meta and meta.get("kind") in (WAKE_KIND, RECOVERY_KIND):
            self._last_options = self._wake_options(meta, context._trace)
        elif handed_over:
            # Another channel's turn: resumed as that ran, not under what this session's last turn was given.
            self._last_options = _TurnOptions()

    async def _take_question(self, run_id: str) -> bool:
        """Take over the question the session came back with: a call hung up on one of its own.

        Like the question of a run a session is opened on: ``state()`` lists it, :meth:`resume` answers
        it, :meth:`interrupt` abandons it. Left to the call's client alone, the chat would hold everything
        typed behind a question nobody here could answer.

        A run whose stored trace holds no question (not parked, or nothing stored of it where this
        session reads) is not waited for: nothing here could answer it, and no turn of this session
        can be taken for its answer (the turn reads the same trace). The session goes on from it,
        and reads the run once more before each turn it starts on it (:meth:`_dropped_question_holds`):
        the end of a run can be stored late (a save that failed and was sent again).

        False only when the run could not be read: that is not "no question" (a turn started now
        might resume it and lose its prompt), so the caller asks again later. True otherwise: taken,
        found not to wait, or no longer this session's to settle (the head moved, the session closed).
        """
        try:
            found = await self._parked_run(run_id)
        except Exception as error:
            logger.warning("chat_session_question_unreadable", session_id=self.id, run_id=run_id, error=str(error))
            return False
        scheduler = self._scheduler
        if self.closed or self._parked or scheduler.lent or scheduler.head != run_id:
            return True
        if found is None:
            logger.warning("chat_session_question_missing", session_id=self.id, run_id=run_id)
            scheduler.not_waiting(run_id)
            self._dropped_question = run_id
            return True
        self._dropped_question = None
        if self._last_context is not None:
            scheduler.unmark(self._last_context)
        self._park_on(*found, handed_over=True)
        self.last_activity = time.monotonic()
        self._record("parked", run_id=run_id)
        self._persist()
        if any(
            event.source == "user" and event.announce == "interrupt" and self._mine(event)
            for event in self._store.pending_events()
        ):
            # Sent while the question was still the call's: its user has already moved on from it.
            self._interrupt_now()
        return True

    async def _dropped_question_holds(self) -> bool:
        """Whether the turn about to start must wait for the run whose question was dropped.

        That run held no question when it was read (nothing stored of it), but its end can be stored
        late, and the turn reads the same trace: found parked by then, the turn would be taken for
        the answer and lose its prompt. So it is read once more right before a turn starts on it,
        also while a call has the session on that same head (the turn would chain onto it there).

        - Still no question: False, the turn starts.
        - Parked now: it is this session's question after all. True, and the run is handed back to
          the worker to take (with the session, if a call has it).
        - Unreadable: not known, which is neither of the above. True after a wait, so the turn does
          not start and the worker asks again; nothing else changes (the call keeps the session,
          nothing is marked as waiting for an answer).

        Forgotten once the head has moved on: nothing chains onto that run any more.
        """
        run_id, scheduler = self._dropped_question, self._scheduler
        if run_id is None:
            return False
        if scheduler.session_head != run_id:
            self._dropped_question = None
            return False
        try:
            found = await self._parked_run(run_id)
        except Exception as error:
            logger.warning("chat_session_question_unreadable", session_id=self.id, run_id=run_id, error=str(error))
            await self._wait_to_read_again()
            return True
        self._question_failures = 0
        if found is None:
            return False
        self._dropped_question = None
        if self.closed or scheduler.session_head != run_id:
            return False
        scheduler.note_parked(run_id)
        return True

    async def _wait_to_read_again(self) -> None:
        """After a read of a question's run failed: a while (0.5 s doubling, up to 30 s), or until something
        arrives or the session changes. Nothing else would bring the worker back to it.

        Only a wait that ran its whole time makes the next one longer. One cut short (a message, a wake)
        is followed by a read at once and is not counted: messages typed during an outage must not push
        the next try out to the longest wait, since nothing tells the worker when the store is back.
        """
        delay = _TURN_RETRY_DELAY_S * 2 ** min(self._question_failures, 8)
        try:
            await asyncio.wait_for(self._until_scheduler_or_work(), min(delay, _RESTORE_RETRY_MAX_S))
        except TimeoutError:
            self._question_failures += 1

    def _wake_options(self, meta: dict[str, Any], trace: Any) -> _TurnOptions:
        """The options of a parked wake, from its prompt; a wake never resumes with more than it had.

        A callable policy is not stored: the agents of the run noted, by path, what it let through, and
        that becomes the policy (an agent that never resolved its tools in the run gets none). Wakes from
        before the policy was stored resume read-only.
        """
        policy = restore_tool_policy(meta.get("tool_policy", "read_only"), trace)
        return _TurnOptions(tool_policy=policy, silence=self.allow_silence and bool(meta.get("silent")), steer=False)

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    # -- Turns ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._hold_worker:
            return
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._serve())

    def _work_arrived(self) -> None:
        self._work.set()
        if self.closed:
            return
        # An app or client event is queued by the task store (post_event), not by a method of the session.
        # Task progress is not restored, so it is left alone.
        restorable = [event for event in self._store.pending_events() if event.task_id is None and self._mine(event)]
        for event in restorable:
            if event.source != "user" and event.id not in self._followed_events:
                # Followed until a turn takes it: steered into a running one, it stays with that turn.
                self._followed_events.add(event.id)
                event.on_settle(self._event_settled)
        if not self._hold_worker:
            # Stored like a message, or a restart loses it before the turn it was for. Not while from_store()
            # settles the stored state (it stores once, at its end).
            queued = tuple(event.id for event in restorable)
            if queued != self._queued_events:
                self._queued_events = queued
                self._persist()
        # An event or a finished task can be the session's first work: nothing else started the worker.
        with contextlib.suppress(RuntimeError):
            self._ensure_worker()

    async def _serve(self) -> None:
        try:
            await self._serve_turns()
        except Exception as error:
            # The next send() starts a new worker; the session itself stays usable.
            logger.error("chat_session_worker_failed", session_id=self.id, error=str(error), exc_info=True)
            raise

    async def _serve_turns(self) -> None:
        delay = _TURN_RETRY_DELAY_S
        while not self.closed:
            # No message is admitted until the state of the run the session continues is known.
            try:
                await self.restore()
                break
            except Exception as error:
                logger.warning("chat_session_restore_failed", session_id=self.id, error=str(error))
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RESTORE_RETRY_MAX_S)
        scheduler = self._scheduler
        while not self.closed:
            self._work.clear()
            if self._store_failures:
                # The store would not record the last turn's start, so it did not run. Again once the store
                # may be back (it holds acknowledged work: not only when something new arrives), sooner when
                # something does (that got stored, or will be refused).
                delay = _TURN_RETRY_DELAY_S * 2 ** (min(self._store_failures, 8) - 1)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._work.wait(), min(delay, _RESTORE_RETRY_MAX_S))
                self._work.clear()
            if self._turn_failures >= _MAX_TURN_FAILURES:
                # The same work failed before it started, time after time (a broken model setup, a store
                # that is down): try again when new work arrives, not in a loop. The failed attempt handing
                # its work back sets _work too, so what counts is work it did not have.
                if self._resume is None and self._pending_work() <= self._attempted:
                    await self._work.wait()
                    continue
                self._turn_failures = _MAX_TURN_FAILURES - 1
            if self._turn_failures:
                await asyncio.sleep(_TURN_RETRY_DELAY_S * 2 ** (self._turn_failures - 1))
            if self._resume is not None:
                values, self._resume = self._resume, None
                if not self.parked:
                    # Another client's run answered or went past the parked turn first: these answers are for
                    # a run that no longer waits, and a resume now would re-send its prompt as a new message.
                    self._record("resume_dropped", run_id=self._last_context.id if self._last_context else None)
                    continue
                scheduler.turn_started()
                await self._run_turn(resume=values)
                continue
            if self._parked:
                if not self.parked:
                    # Another client's run answered the parked turn or went past it: go on from that run.
                    self._parked = False
                    parked_run = self._last_context.id if self._last_context is not None else None
                    self._record("unparked", run_id=parked_run, head=scheduler.session_head)
                    continue
                await self._until_scheduler_or_work()
                continue
            if scheduler.asking and not scheduler.lent:
                # The call hung up on a question of its own and the session came back with it.
                if await self._take_question(scheduler.head):
                    continue
                # The run that asked could not be read (the trace store is away). No turn starts on a run
                # that may be waiting for an answer. Asked again after a while, as restore() does, and
                # sooner when something arrives.
                await self._wait_to_read_again()
                continue
            if self._dropped_question is None:
                # No question left to read (taken, answered by another client, the head moved): the next
                # one that cannot be read starts its waits from the shortest again.
                self._question_failures = 0
            if scheduler.session_parked:
                # Another client's run of this session waits for its caller (an approval, a question): a
                # turn now would chain onto it, and that run's resume would drop the turn's prompt. Its
                # resume, or any newer head, lets the session go on; messages stay pending until then.
                # (The session's head: a call that has the session moves it, and may be the one asking.)
                await self._until_scheduler_or_work()
                continue
            # A recovery turn goes before anything queued: it is about the turn the user saw cut, and the turns
            # after it must find it settled (also while a call has the session: it is this session's own cut
            # turn, and what is typed must not go ahead of it). Other wakes wait for the user, and for the
            # session: while a call has it the chat answers only what is typed, and an application's wake
            # runs once it is the chat's again (the call hands it back with its head).
            if self._wakes and await self._dropped_question_holds():
                continue
            recovering = bool(self._wakes) and self._wakes[0].kind == RECOVERY_KIND
            if self._wakes and (recovering or (not scheduler.lent and scheduler.pending_turn() is None)):
                if scheduler.other_turn_first():
                    # Another turn is ahead (the call's, another client's): the wake goes right after it (a
                    # recovery turn too, not after whatever was typed meanwhile). Its end is a change of the
                    # scheduler, not new work (_turn_due waits for work alone once nothing pending asks for a
                    # turn of its own, and the wake would wait for the next message).
                    await self._until_scheduler_or_work()
                    continue
                wake = self._wakes.pop(0)
                scheduler.turn_started()
                await self._run_turn(wake=wake)
                continue
            if not await self._turn_due():
                continue
            if await self._dropped_question_holds():
                continue
            # Take and busy in one step: a foreign root turn either goes before it or meets a busy owner.
            followup = scheduler.take()
            if followup is None:
                continue
            scheduler.turn_started()
            await self._run_turn(followup=followup)

    def _pending_work(self, followup: Followup | None = None, wake: _Wake | None = None) -> set[Any]:
        """Ids of the work waiting for a turn, plus what ``followup`` and ``wake`` took of it."""
        store = self._store
        events = store.pending_events() + (followup.events if followup is not None else [])
        completions = store.pending_completions() + (followup.completions if followup is not None else [])
        work: set[Any] = {event.id for event in events}
        work.update(("task", completion.get("task_id")) for completion in completions)
        work.update(("wake", item.id) for item in (self._wakes if wake is None else [*self._wakes, wake]))
        return work

    async def _until_scheduler_or_work(self) -> None:
        # The scheduler's future is taken now, in the step that read its state: from a task of its own it was
        # taken a step later, and a turn of another client ending in between (its notice going to no future)
        # left the worker waiting for new work. Shared with other waiters: waited on, never cancelled.
        changed = self._scheduler.changed()
        work = asyncio.get_running_loop().create_task(self._work.wait())
        try:
            await asyncio.wait({changed, work}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not work.done():
                work.cancel()

    async def _turn_due(self) -> bool:
        """True once the scheduler says a turn is due; False when other work arrived first (a wake, a resume)."""
        loop = asyncio.get_running_loop()
        ready = loop.create_task(self._scheduler.wait_ready(self._admit))
        work = loop.create_task(self._work.wait())
        try:
            await asyncio.wait({ready, work}, return_when=asyncio.FIRST_COMPLETED)
            if ready.done() and ready.result() is not None:
                return True
            # Nothing pending or running could ask for a turn: wait for new work.
            if not work.done():
                await work
            return False
        finally:
            for task in (ready, work):
                if not task.done():
                    task.cancel()

    async def _admit(self) -> None:
        """Hold a turn for collected messages until none has arrived for ``collect_window`` seconds, and while
        another client's run of the session waits for its caller (a turn would chain onto it, see _serve_turns)."""
        while True:
            if self._scheduler.session_parked:
                await self._scheduler._wait_change()
                continue
            # The batch the turn will take (the same function): not a message that is not its to take.
            collected = [event.posted_at for event in pending_user_batch(self._store) if event.mode == "collect"]
            if not collected:
                return
            delay = max(collected) + self.collect_window - time.time()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    def _turn_inputs(
        self, followup: Followup | None, resume: dict[str, Any] | None, wake: _Wake | None
    ) -> tuple[dict[str, Any], dict[str, Any], _TurnOptions]:
        """The agent kwargs, the log's trigger, and the options of the turn about to run."""
        kwargs = dict(self._kwargs)
        if resume is not None:
            # The run a resume chains onto drops its prompt; the agent still requires one.
            kwargs["prompt"] = self._last_prompt
            kwargs["resume"] = resume
            return kwargs, {"kind": "resume"}, self._last_options
        if wake is not None:
            options = _TurnOptions(
                tool_policy=wake.tool_policy, silence=self.allow_silence and wake.silent_ok, steer=False
            )
            # A recovery notice carries its own header (it is the runtime's, not a scheduled wake).
            text = wake.text if wake.kind == RECOVERY_KIND else f"{_WAKE_HEADER}\n\n{wake.text}"
            if wake.tool_policy == "read_only" or isinstance(wake.tool_policy, ReadOnlyWithin):
                # Narrowed (a recovery turn after a restricted one) it is read-only all the same.
                text += f"\n\n{_READ_ONLY_NOTE}"
            prompt = Message(
                role="user",
                content=[TextContent(text=text)],
                metadata={
                    "source": RUNTIME_SOURCE,
                    "kind": wake.kind,
                    "wake_id": wake.id,
                    # Read back by restore(): a session continuing this run parked resumes it under the same.
                    "tool_policy": _describe_policy(wake.tool_policy),
                    "silent": options.silence,
                },
            )
            trigger = wake.trigger or {
                "kind": WAKE_KIND,
                "wake_id": wake.id,
                "key": wake.key,
                "tool_policy": _describe_policy(wake.tool_policy),
            }
        else:
            assert followup is not None
            trigger = followup.trigger
            options = _TurnOptions(silence=self.allow_silence and trigger["kind"] != "user_message")
            prompt = followup.prompt
        if options.silence:
            prompt = _with_note(prompt, _SILENCE_NOTE)
        kwargs["prompt"] = prompt
        return kwargs, trigger, options

    async def _run_turn(
        self,
        *,
        followup: Followup | None = None,
        resume: dict[str, Any] | None = None,
        wake: _Wake | None = None,
    ) -> None:
        """Run one turn the scheduler is already busy with (``turn_started``), and settle it."""
        scheduler = self._scheduler
        turn: asyncio.Task | None = None
        stopping = False
        try:
            kwargs, trigger, options = self._turn_inputs(followup, resume, wake)
            if scheduler.head is None:
                # A new thread: this turn is its root.
                if self._generated_id:
                    # Once: a turn cut or failed before it started hands its work back, and the turn that takes
                    # it is another run, with an id of its own (and the thread's root if it is the one to start).
                    self._generated_id = False
                    context = RunContext(id=self.id, tracing_provider=self._agent.tracing_provider)
                else:
                    context = RunContext(tracing_provider=self._agent.tracing_provider)
            else:
                context = RunContext(parent_id=scheduler.head, tracing_provider=self._agent.tracing_provider)
            # What started the turn is the run's from its start: stamped on the root span as it is made (before
            # any save), read by a tool as ``ctx.trigger`` and by an approval reviewer from the root. Its own copy:
            # the session stores ``trigger`` (in_flight, the log), and a tool's write to it must change neither (deep
            # where it can be: a session without a store takes any trigger).
            context.trigger = _own_copy(trigger)
            context._tracing_provider = self._turn_provider(context._tracing_provider)
            context._steering = options.steer
            # The session hint: the thread's root run, or this turn when it starts the thread and the session
            # has none (it becomes the root once it is the head, see _pump).
            context._session_id = self.root_run_id or (context.id if context.parent_id is None else None)
            context._tool_policy = options.tool_policy
            context._silence_token = SILENT_REPLY if options.silence else None
            context._cut = asyncio.Event()
            adopt_background_store(context, self._store)
            scheduler.mark_own(context)
            self._last_prompt, self._last_options = kwargs["prompt"], options
            self._attempted = self._pending_work(followup, wake)
            # What a restored session needs if this process dies mid-turn: stored from the turn's first event.
            in_flight = {
                "run_id": context.id,
                "parent_id": context.parent_id,
                # It starts the thread and the session knows no root: recovery may make it the root (_root_at).
                "starts_thread": context.parent_id is None and self.root_run_id is None,
                "trigger": trigger,
                "events": [event.to_dict() for event in followup.events] if followup is not None else [],
                "wake": wake.to_dict() if wake is not None else None,
                # What this turn was allowed: a recovery turn after it may read, and no further than it could.
                "tool_policy": _store_policy(options.tool_policy),
            }
            if resume is not None:
                # Only a durable session reads these back (after a restart). One without a store takes any
                # value its tools understand, storable or not.
                in_flight["resume"] = _dump_resume(resume) if self._session_store is not None else {}
                in_flight["approved"] = self._approved_calls(resume)
            # Stored from now, not from the turn's first event: what it took (messages, a wake, answers) is no
            # longer pending, and a restart before that event would lose it. A turn that never started is put
            # back by recovery (nothing of it stored), or here, below, if this process sees it end first.
            if self._session_store is not None:
                # The agent says which tool calls it is about to run, and waits for the store to hold them.
                context._before_tool_dispatch = functools.partial(self._note_dispatch, in_flight)
            self._in_flight = in_flight
            self._persist()
            turn = asyncio.get_running_loop().create_task(
                self._pump(context, kwargs, followup, trigger, in_flight, self.mark())
            )
            self._turn, self._turn_context = turn, context
            try:
                await asyncio.wait({turn})
                if not turn.cancelled() and turn.exception() is None and turn.result() is not None:
                    # Here, on the worker, and not on the turn's own task: an interrupt cancels that task,
                    # and this turn has ended. There is nothing left of it to cut, and the session is busy
                    # for as long as this takes, which is just when an interrupt is likely to come.
                    await self._trace_stored(context)
            except asyncio.CancelledError:
                # The worker is being stopped (a close). A turn still running is cut with it and stays the
                # turn in flight (the agent ends it as interrupted; close() marks it, and a restart
                # recovers it). One that had already ended, or that ends by itself all the same, is settled
                # below like any other: how it ended (parked, perhaps) must not be lost with the worker, nor
                # the turn be stored as one that was cut. Unless, on a durable session, its trace is not
                # stored: then it stays the turn in flight too, and a restart recovers it from what the
                # provider does hold. (Any other session has no restart to leave it to: it is settled.)
                stopping = True
                cut = not turn.done()
                if cut:
                    turn.cancel()
                    while not turn.done():
                        # Until it is over, whatever cancels this worker again meanwhile: what follows reads
                        # its outcome. wait(), not `await turn`: that would pass a second cancellation on to
                        # the turn and cut its cleanup short (and raise the turn's own error here).
                        with contextlib.suppress(asyncio.CancelledError):
                            await asyncio.wait({turn})
                unsaved = self._holds_turns_for_their_trace and context.__dict__.get("_trace_unsaved")
                if (cut and not _ended_by_itself(turn)) or unsaved:
                    raise
        finally:
            self._turn = None
            # With how its run ended: a session left on a question is not lent to a call (only the chat
            # can answer it), and the call's turns wait for the answer.
            scheduler.turn_finished(_left_parked(turn))
            if followup is not None:
                # Handed back unless a run started with it (then this does nothing). Here as well as in the
                # turn's own task: one cancelled before its first step never runs any of its code.
                followup.release()
        self._settle_turn(turn, context, resume, wake)
        if in_flight.get("stopped") and self.parked and self._last_context is not None:
            # The stop was taken while this turn ran, and the session is parked all the same: the run had
            # just ended on a question when the cut reached it, or it was a resume cut before it used its
            # answers. The stop is about that question now, as it is on a parked session.
            pending = self._cancel_pending(self._last_context, "The user moved on before answering.")
            if pending:
                self._record("interrupt", run_id=self._last_context.id, abandoned=pending)
                self._persist()
        if stopping:
            raise asyncio.CancelledError

    def _settle_turn(
        self, turn: asyncio.Task, context: RunContext, resume: dict[str, Any] | None, wake: _Wake | None
    ) -> None:
        """Record how a turn ended: its status and whether it left the session parked, or, for one that
        never started or failed before it did, its work handed back. One step, with nothing awaited."""
        scheduler = self._scheduler
        output = None
        if not turn.cancelled():
            error = turn.exception()
            if isinstance(error, SessionNotStored):
                # Its start could not be stored: it did not run (_write_ahead). Its work is handed back (below,
                # and by _pump), answers to a parked turn included, and the worker tries again.
                self._store_failures += 1
                logger.warning("chat_session_turn_not_stored", session_id=self.id, run_id=context.id, error=str(error))
                self._record(
                    "turn_failed",
                    run_id=context.id,
                    error=str(error),
                    reason="not_stored",
                    attempt=self._store_failures,
                )
                if resume is not None and self._resume is None and not self.closed:
                    self._resume = resume
            elif error is not None:
                logger.error("chat_session_turn_failed", session_id=self.id, run_id=context.id, error=str(error))
                self._turn_failures += 1
                self._record("turn_failed", run_id=context.id, error=str(error), attempt=self._turn_failures)
            else:
                output = turn.result()
        if output is not None:
            self._turn_failures = 0
        if output is None:
            # Never started (cut before its first event): the head stays on the last run that did.
            if wake is not None and not self.closed:
                self._wakes.insert(0, wake)
            if not self.closed:
                self._in_flight = None
                self._persist()
            return
        if self._last_context is not None and self._last_context is not context:
            scheduler.unmark(self._last_context)
        self._last_context = context
        status = output.status
        self._parked = is_parked(status)
        self.last_status = {"code": status.code, "reason": status.reason, "message": status.message}
        self.last_activity = time.monotonic()
        self._in_flight = None
        self._record(
            "turn_finished",
            run_id=context.id,
            status=self.last_status,
            parked=self.parked,
            silent=bool(output.metadata.get("silent")),
            **({"tool_search": output.metadata["tool_search"]} if output.metadata.get("tool_search") else {}),
            **({"error": {k: output.error.get(k) for k in ("type", "message")}} if output.error else {}),
        )

    async def _pump(
        self,
        context: RunContext,
        kwargs: dict[str, Any],
        followup: Followup | None,
        trigger: dict[str, Any],
        in_flight: dict[str, Any],
        mark: int,
    ) -> OutputEvent | None:
        """Run one turn on its own task (so it can be cut) and copy its events into the log.

        Written ahead, on a durable session: the turn does not start before the store holds that it took its
        input (``mark``), and no tool call runs before the store holds it (:meth:`_note_dispatch`). A restart
        can then tell a turn that never ran (its input is queued again) from one that may have acted (a
        read-only recovery turn), whatever the trace provider had saved by then.
        """
        set_run_context(context)
        set_call_id(None)
        set_parent_call_id(None)
        output: OutputEvent | None = None
        started = False
        try:
            await self._write_ahead(mark)
            self._store_failures = 0
            async with aclosing(self._agent(**kwargs)) as stream:
                async for event in stream:
                    if not started:
                        started = True
                        self._scheduler.note_run(context.id)
                        if self.root_run_id is None and context.parent_id is None:
                            self.root_run_id = context.id  # it started the thread
                        self.turns += 1
                        self._parked = False  # a resume took the answers: what is stored now is this turn
                        self._in_flight = in_flight
                        # (The log's own copy of the trigger: ``in_flight`` holds the one that is stored.)
                        self._record(
                            "turn_started", run_id=context.id, parent_id=context.parent_id, trigger=_own_copy(trigger)
                        )
                        if followup is not None:
                            followup.started(context)  # (the trigger is on the root already: context.trigger)
                    self.log.put_nowait(event)
                    if isinstance(event, OutputEvent) and event.run_id == context.id and event.parent_call_id is None:
                        output = event
        except asyncio.CancelledError:
            # Cut (an interrupt, a close) when the run was already over: right after its OUTPUT, or during
            # its last save, before the event was sent. It ended as its root span says, not as a turn that
            # never ran: taken for one, its run would stay the head with nobody having settled it, and the
            # next turn would pick up the tool calls it left waiting (a question its user had just stopped).
            if output is None and started:
                output = _ended_output(context)
                if output is not None:
                    self._save_cut = True
                    context.__dict__["_trace_unsaved"] = True  # until _trace_stored has saved it again
                    self.log.put_nowait(output)
            if output is None:
                raise
        finally:
            if not started and followup is not None:
                followup.release()
        return output

    @property
    def _holds_turns_for_their_trace(self) -> bool:
        """Whether a turn stays open until the provider holds its trace (:meth:`_trace_stored`).

        Only a durable session: it promised that the conversation comes back after a restart, and the
        conversation is in the traces. Any other session keeps its turns as every run keeps its trace, as well
        as the provider allows: a save it refuses (a platform that answers 403, a store that is down) is logged
        and counted (``tracing:save_failed``) and the conversation goes on. Held open there, one refused save
        would stop the session for good: busy, no ``turn_finished``, every later message queued behind it.
        """
        return self._session_store is not None

    async def _trace_stored(self, context: RunContext) -> None:
        """Return once the provider holds the trace of the turn that just ended, on a durable session; on any
        other, once it has been tried.

        The conversation is in the traces: the next turn reads this one from the provider. A turn whose last
        save failed (the run itself only logs that and goes on) would be a turn nobody remembers: its
        messages gone from the history, a question it parked on answered into nothing. So on a durable
        session the turn is not over, and the session starts nothing else, until a save lands. Only stopping
        the session ends the wait (it runs on the worker): an interrupt has nothing to cut in a turn that has
        ended.
        """
        delay = _TURN_RETRY_DELAY_S
        if self._save_cut:
            # The run's own last save was cut, not refused: again at once (every session: one try).
            self._save_cut = False
            await context._save_trace(barrier=True)
        if not self._holds_turns_for_their_trace:
            if context.__dict__.get("_trace_unsaved"):
                # Said once, here: the history the next turn reads may lack this one.
                logger.warning("chat_session_trace_not_stored", session_id=self.id, run_id=context.id, waiting=False)
            return
        while context.__dict__.get("_trace_unsaved"):
            logger.warning("chat_session_trace_not_stored", session_id=self.id, run_id=context.id, waiting=True)
            await asyncio.sleep(delay)
            await context._save_trace(barrier=True)
            delay = min(delay * 2, _RESTORE_RETRY_MAX_S)

    async def _note_dispatch(
        self, in_flight: dict[str, Any], agent_path: str, calls: list[tuple[str, str, Any]]
    ) -> None:
        """The agent is about to run these tool calls (``RunContext._before_tool_dispatch``: a model's, a
        slash command's, those a resume left pending): they are stored with the turn first, however long the
        store takes. Told by the agent itself, so it holds for every tool, whatever it is called and however
        it is started; what a restart finds recorded here and not in the trace may have run."""
        recorded = in_flight.setdefault("dispatched", [])
        if agent_path != self._agent._path and recorded:
            return  # a sub-agent's calls: the root agent's call that runs it is recorded, recovery reads that
        known = {call["id"] for call in recorded}
        new = [
            {"id": call_id, "name": name, "input": _brief(arguments, 200)}
            for call_id, name, arguments in calls
            if call_id not in known
        ]
        if not new:
            return
        recorded.extend(new)
        self._persist()
        await self._write_ahead(self.mark(), wait=True)

    async def _write_ahead(self, mark: int, *, wait: bool = False) -> None:
        """Return once the store holds the session as of ``mark`` (at once when it is not durable).

        When the store fails: raises :class:`SessionNotStored`, or with ``wait`` keeps trying (the caller is
        cancelled by an interrupt or a close). :class:`SessionLeased` when another process took the session.
        """
        delay = _TURN_RETRY_DELAY_S
        while self._session_store is not None:
            try:
                await self.flush()
                return
            except SessionNotStored:
                if self.stored_since(mark):
                    return
                if not wait:
                    raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RESTORE_RETRY_MAX_S)

    # -- Log ----------------------------------------------------------------------

    def _turn_provider(self, provider: Any) -> Any:
        """The provider a turn saves with: ``durability`` when given; else platform saves are buffered.

        A whole-trace save uploads the run's history again, so a turn that saved on every finished span
        would upload it dozens of times. With deltas a save is small, but each still costs a round trip the
        turn would wait for (about two per tool step). Buffered, mid-run saves go out at most once per
        flush interval without the turn waiting; with ``store_before_acting`` (set on the buffered
        providers configured here) the agent still stores a tool batch that is not read-only before
        running it, so recovery never misses a call whose effects may have happened.
        A provider or ``TIMBAL_TRACE_DURABILITY`` that already chose a mode keeps it.
        """
        from ..state.tracing import uploads
        from ..state.tracing.providers import PlatformTracingProvider

        if not isinstance(provider, type):
            return provider
        want = self._durability
        if want is None:
            chosen = provider.durability is not None or os.environ.get(uploads.DURABILITY_ENV)
            if not issubclass(provider, PlatformTracingProvider) or chosen:
                return provider
            want = "buffered"
        settings: dict[str, Any] = {}
        if uploads.durability(provider) != want:
            settings["durability"] = want
        if want == "buffered" and not getattr(provider, "store_before_acting", False):
            # Not with ``durability=`` when the provider is buffered already (the platform default): an
            # explicit mode also turns on the ``tracing:*`` usage counters, which a default must not add.
            settings["store_before_acting"] = True
        if not settings:
            return provider
        configured = self._providers.get((provider, want))
        if configured is None:
            configured = self._providers[(provider, want)] = provider.configured(**settings)
        return configured

    def _record(self, kind: str, **fields: Any) -> None:
        self.log.put_nowait({"type": "SESSION", "kind": kind, "t": int(time.time() * 1000), **fields})
        if kind in _PERSISTED_KINDS:
            self._persist()

    # -- Durability ---------------------------------------------------------------

    async def flush(self) -> None:
        """Wait until the session's latest state is stored (at once when it is not durable).

        The HTTP routes call it before answering: a message's ``202`` means it survives a restart. Raises
        :class:`~timbal.state.chat_store.SessionNotStored` when the store failed (once more after a retry here),
        and :class:`~timbal.state.chat_store.SessionLeased` when another process took the session: either
        way the latest change is not stored.
        """
        await self._saves_done()
        if self._session_store is None:
            return
        if not self._lost and self._dirty:
            self._persist()  # the write that failed, once more
            await self._saves_done()
        if self._lost:
            raise SessionLeased(f"Chat session {self.id!r} is served by another process now.")
        if self._dirty:
            raise SessionNotStored(f"Chat session {self.id!r} could not be stored: {self._save_error}")

    async def _saves_done(self) -> None:
        while self._saver is not None and not self._saver.done():
            await asyncio.shield(self._saver)

    def _snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=self.id,
            agent=self._agent.name,
            config={
                "busy_mode": self.busy_mode,
                "collect_window": self.collect_window,
                "allow_silence": self.allow_silence,
                "durability": self._durability,
                "on_dead_turn": self.on_dead_turn,
                # The arguments every turn runs with (max_tokens=...): a restored session keeps them. Those a
                # snapshot cannot hold are named, and from_store() asks for them again.
                "kwargs": self._stored_kwargs,
                "kwargs_unstored": self._unstored_kwargs,
            },
            head=self.head,
            root_run_id=self.root_run_id,
            turns=self.turns,
            last_status=self.last_status,
            parked=self.parked,
            last_prompt=_dump_prompt(self._last_prompt),
            last_options={
                "tool_policy": _store_policy(self._last_options.tool_policy),
                "silence": self._last_options.silence,
                "steer": self._last_options.steer,
            },
            # A copy: the turn goes on adding to it ("dispatched", "stopped") while a save is out, and a store
            # that serializes after its first await would write a later state than the one it was handed.
            in_flight=copy.deepcopy(self._in_flight),
            # Progress of a background task dies with the task (process-local): not worth restoring.
            pending=[
                event.to_dict() for event in self._store.pending_events() if event.task_id is None and self._mine(event)
            ],
            wakes=[wake.to_dict() for wake in self._wakes],
            resume=_dump_resume(self._resume),
            log_cursor=self.log.cursor_end,
            closed=self.closed and not self._keep,
        )

    def persist(self) -> None:
        """Store a durable session now (it stores itself on every change; a new one has had none yet)."""
        self._persist()

    def _persist(self) -> None:
        """Queue a snapshot save: saves run one at a time, in order, a burst coalescing into one."""
        if self._close_done.is_set():
            # Closed, and what it had to store is stored (or was given up): nothing is written from here
            # again. A session restored in this process holds the same lease owner, so the store could not
            # tell a late write of this one from the live session's, and would take it.
            return
        self._changes += 1
        if self._session_store is None or self._lost:
            return
        self._dirty = True
        if self._saver is None or self._saver.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._saver = loop.create_task(self._save_loop())

    async def _save_loop(self) -> None:
        store = self._session_store
        assert store is not None
        while self._dirty and not self._lost:
            self._dirty = False
            covers = self._changes
            created = self._version is None
            snapshot = self._snapshot()
            self._saving = covers
            sent = time.monotonic()
            try:
                self._version = await store.save(snapshot, expect_version=self._version, owner=process_owner())
            except SnapshotConflict as e:
                self._saving = None
                self._lose(str(e))
                return
            except Exception as e:
                # Kept dirty: the next change (or flush()) retries; flush() reports it, and does not hang.
                self._saving = None
                self._dirty, self._save_error = True, e
                logger.error("chat_session_save_failed", session_id=self.id, error=str(e))
                return
            except BaseException:
                # Cancelled with the write out (a close that stopped waiting for the store): whether it
                # lands is not known, and this session is not there to see it.
                self._saving = None
                raise
            if created:
                # Inserted with this process's lease, in the same write (ChatSessionStore.save): the row never
                # exists without an owner, and there is no second write that could fail and leave it so.
                self._leased, self._lease_from = True, sent
            elif not self._leased and not self.closed:
                sent = time.monotonic()
                try:
                    await store.acquire_lease(self.id, process_owner())
                except SessionLeased as e:
                    self._lose(str(e))
                    return
                except Exception as e:
                    # Like a failed write: not counted as stored, tried again with the next one. (The row
                    # is written all the same: `_saving` still says so.)
                    self._dirty, self._save_error = True, e
                    logger.error("chat_session_lease_failed", session_id=self.id, error=str(e))
                    return
                self._leased, self._lease_from = True, sent
            if self._leased and self._lease_task is None and not self.closed:
                self._lease_task = asyncio.get_running_loop().create_task(self._renew_lease())
            self._save_error, self._stored_changes, self._saving = None, covers, None
            for wake in self._wakes:
                if wake.replaced and wake.mark <= covers:
                    wake.replaced = []  # stored: withdrawing a later replacement puts this one back

    async def _renew_lease(self) -> None:
        """Renew the lease every third of it, and stop the session when it runs out unrenewed.

        The lease is counted from when the request that last confirmed it was *sent* (`_lease_from`): the
        store counted from no earlier, so another process can be given the session no sooner than a whole
        lease after that. This copy stops `_LEASE_MARGIN` of a lease before then, whatever a renewal still
        out is doing: checked only when a renewal came back, a store that hangs kept the old owner acting
        for a third of a lease after the new one had started.
        """
        store = self._session_store
        assert store is not None
        owner = process_owner()
        if self._lease_from is None:
            self._lease_from = time.monotonic()
        due = self._lease_from + store.lease_ttl_s / 3
        error = "no renewal was tried"
        while not self.closed and not self._lost:
            ttl = store.lease_ttl_s
            ends = self._lease_from + ttl * (1 - _LEASE_MARGIN)
            now = time.monotonic()
            if now < min(due, ends):
                await asyncio.sleep(min(due, ends) - now)
                if self.closed or self._lost:
                    return
                now = time.monotonic()  # (a timer may fire a clock tick early: what it waited for has come)
            if due >= ends or now >= ends:
                # It has (as good as) run out, and another process may hold the session without this one
                # being able to tell. Going on would be two owners acting.
                self._lose(
                    f"the {ttl:g} s lease ran out, {now - self._lease_from:.3g} s after its last renewal: {error}"
                )
                return
            # Not awaited past what is left of the lease, and not waited for once given up (`wait_for`
            # would wait for the call to end its cancellation: a store that does not, holds the deadline).
            attempt = asyncio.ensure_future(store.acquire_lease(self.id, owner))
            waited = max(min(ttl / 3, ends - now), 0.0)
            try:
                done, _ = await asyncio.wait({attempt}, timeout=waited)
            except BaseException:
                _abandon(attempt)
                raise
            if not done:
                _abandon(attempt)
                error = f"no answer in {waited:.3g} s"
            else:
                try:
                    attempt.result()
                except SessionLeased as e:
                    self._lose(str(e))
                    return
                except asyncio.CancelledError:
                    error = "cancelled by the store"
                except Exception as e:
                    error = str(e) or type(e).__name__
                else:
                    # An earlier failure is no longer why the lease may run out (a stalled loop would be).
                    self._lease_from, due = now, now + ttl / 3
                    error = "no renewal was tried"
                    continue
            logger.warning("chat_session_lease_renewal_failed", session_id=self.id, error=error)
            # Sooner than the next renewal would be: what is left of the lease is all the time there is.
            due = time.monotonic() + ttl / 10

    def _lose(self, reason: str) -> None:
        """Another process serves this session now: stop, without writing over its state."""
        if self._lost:
            return
        self._lost = True
        logger.error("chat_session_ownership_lost", session_id=self.id, reason=reason)
        if not self.closed:
            asyncio.get_running_loop().create_task(self.close(keep=True))

    def _stop_lease(self) -> None:
        lease, self._lease_task = self._lease_task, None
        if lease is not None and not lease.done() and lease is not asyncio.current_task():
            lease.cancel()

    async def _settle_store(self) -> None:
        """After close: the last state stored (kept) or the row gone, and the lease let go."""
        store = self._session_store
        if store is None:
            return
        self._stop_lease()
        if self._lost:
            # Another process serves it: nothing here is written, so there is nothing to wait for either
            # (a write still out is this copy's last, and is refused or superseded by the owner's).
            return
        closed_stored = True
        try:
            await self.flush()
        except SessionNotStored as e:
            closed_stored = False
            logger.error("chat_session_close_not_stored", session_id=self.id, error=str(e))
        except SessionLeased:
            pass
        if self._lost:
            return
        try:
            if not self._keep:
                # Only what this process last wrote, and only while nobody else holds it: a session another
                # process took meanwhile (this one stalled past its lease) is theirs, row included.
                await store.delete(self.id, expect_version=self._version, owner=process_owner())
            elif self._leased:
                await store.release_lease(self.id, process_owner())
        except SnapshotConflict as e:
            self._lost = True  # its caller is told the session lives on elsewhere (DELETE: 409)
            logger.error("chat_session_ownership_lost", session_id=self.id, reason=str(e))
        except Exception as e:
            logger.error("chat_session_close_store_failed", session_id=self.id, error=str(e))
            if not self._keep and not closed_stored:
                # The row is still there and does not say closed: a restore would bring the session back.
                # (Stored as closed, a row that could not be deleted is never restored.)
                self.close_error = e

    def _approved_calls(self, values: dict[str, Any]) -> list[str]:
        """The gated calls a resume lets run, as the recovery notice names them."""
        if self._last_context is None:
            return []
        calls = []
        for approval in self._last_context.pending_approvals():
            if _approves(values.get(approval["approval_id"])):
                name = approval.get("path", "?").rsplit(".", 1)[-1]
                calls.append(f"{name}({_brief(approval.get('input'), 80)})")
        return calls

    def _abandon_question(self, reason: str) -> None:
        """Answer what the parked turn still waits for with a cancel (answers already given stand): the gated
        tools never run, the run ends ``cancelled``."""
        if self._last_context is not None:
            self._cancel_pending(self._last_context, reason)

    def _restore_events(self, stored: list[dict[str, Any]]) -> None:
        # A snapshot can name an event twice: with the turn that took it and, handed back as that turn was
        # cut before it started, pending again. Queued once, where it comes first.
        queued = {event.id for event in self._store.pending_events()}
        for data in stored:
            if data["id"] in queued:
                continue
            queued.add(data["id"])
            # The event itself when this process still has it (from_store() took it out of the task store).
            event = self._store.restore_event(data, held=self._kept_events.pop(data["id"], None))
            if event.source == "user":
                self._watch(event)

    def _drop_stopped(self, events: list[dict[str, Any]], wake: dict[str, Any] | None, dead_run_id: str | None) -> None:
        """The input of a dead turn its user stopped, its messages or its wake: delivered to it (the messages'
        receipts settle), and named to the next turn in a silent notice, which rides with it (that turn answers
        the stop). Their ids join ``_shown_events``: a snapshot can list them as pending too (a turn cut before
        it started hands its messages back), and from there they would be queued again."""
        for data in events:
            self._shown_events.add(data.get("id"))
            held = self._kept_events.pop(data.get("id"), None)
            if held is not None:
                held._settle("delivered", dead_run_id)
        if self.on_dead_turn != "drop":
            asked = [event.get("text") or "" for event in events]
            if wake is not None:
                asked.append(wake.get("text") or "")
            self._store.post_event(
                _stopped_notice([text for text in asked if text]), key=RECOVERY_KIND, announce="silent"
            )

    def _requeue_kept_events(self, dead_run_id: str | None) -> None:
        """The events the task store held that the snapshot did not queue again: posted after its last
        write (they go back behind the stored ones, in their order), or taken by the dead run, as its
        stored memory shows (delivered there: queued again, they would be answered twice)."""
        kept, self._kept_events = self._kept_events, {}
        for event in kept.values():
            if event.id in self._shown_events:
                event._settle("delivered", dead_run_id)
                continue
            self._store.restore_event(event.to_dict(), held=event)
            if event.source == "user":
                self._watch(event)

    async def _load_run(self, run_id: str | None) -> RunContext | None:
        """A stored run of this session as a context (its trace loaded), or None when it was never stored."""
        if run_id is None:
            return None
        lookup = RunContext(parent_id=run_id, tracing_provider=self._agent.tracing_provider)
        provider = lookup._tracing_provider
        if provider is None:
            return None
        # Strict: an unreadable run raises (from_store fails, the snapshot stays for a retry). Taken for a
        # missing one, a dead turn that ran tools would have its input queued and run again.
        trace = await provider.get_strict(lookup)
        if trace is None or not trace.as_records():
            return None
        loaded = RunContext(id=run_id, parent_id=trace.parent_id, tracing_provider=self._agent.tracing_provider)
        loaded._trace = trace
        return loaded

    def _root_at(self, head: str | None, in_flight: dict[str, Any]) -> None:
        """Keep :attr:`root_run_id` right as recovery settles a dead turn: ``head``, the run it is left on (None:
        it goes back to the run before it, and is run again under another id)."""
        if in_flight.get("parent_id") is not None:
            return  # not the thread's first run
        if head is not None and self.root_run_id is None and in_flight.get("starts_thread"):
            # It started the thread. (Stored before the field existed, a first run is no proof of the root: for
            # the platform's chat that is its session id. No hint then, as before.)
            self.root_run_id = head
        elif head is None and self.root_run_id == in_flight["run_id"]:
            self.root_run_id = None  # it did not: the turn that runs its input again does

    async def _recover(
        self,
        in_flight: dict[str, Any],
        last_options: dict[str, Any] | None = None,
        pending: list[dict[str, Any]] | None = None,
    ) -> str:
        """Settle the turn that was running when the session was last stored (see :meth:`from_store`).
        ``last_options``: the snapshot's, which an older one needs to tell what the turn was allowed.
        ``pending``: the snapshot's queued events (an interrupting message among them stopped this turn)."""
        run_id = in_flight["run_id"]
        self._in_flight = None
        context = await self._load_run(run_id)
        root = context.root_span() if context is not None else None
        cut = in_flight.get("cut")
        if root is not None and root.t1 is not None and (cut is None or not _was_cut(root.status)):
            # It finished; the process died before the session stored that. Also when a shutdown marked it
            # as cut: a run that ended by itself (answered, parked on a question) was not cut, and treating
            # it as one would chain a recovery turn onto a run that waits for its user.
            self._scheduler.head = run_id
            self._root_at(run_id, in_flight)
            self._parked = is_parked(root.status)
            self.last_status = _status_dict(root.status)
            self._last_context = context if self._parked else None
            if self._parked and in_flight.get("stopped"):
                # Its user stopped it as it ended on this question (interrupt() said so): abandoned.
                self._cancel_pending(context, "The user moved on before answering.")
            # The snapshots that said it took them (a steered message, most of all) may never have been
            # written: what its memory shows is not pending any more.
            self._shown_events = _shown_events(root)
            return "finished"
        dispatched = in_flight.get("dispatched")
        calls = [call for call in dispatched if isinstance(call, dict)] if isinstance(dispatched, list) else []
        resumed = in_flight.get("resume") is not None
        events = in_flight.get("events") or []
        # Its user stopped it: the stop reached it (`stopped`), or only the message that was to stop it was stored
        # before the process died (it cuts nothing until it is). Live, the stop cut the turn and its input was
        # never acted on again.
        stopped = bool(in_flight.get("stopped")) or _interrupts_waiting(pending, events)
        if not dispatched:
            # It had run no tool call (each is stored before it runs, see _note_dispatch): whatever is stored
            # of it, it did nothing that cannot be done again. Its input goes back in the queue, the answers
            # to a parked turn included (none of the calls they were for had started), and the next turn
            # continues from the run before it. Not a recovery turn on top of it: a run cut during its
            # model call (a shutdown, most often) stores a trace whose memory need not hold its prompt,
            # and the message would be answered by nobody.
            self._scheduler.head = in_flight.get("parent_id")
            self._root_at(None, in_flight)
            wake = in_flight.get("wake")
            if stopped and not resumed and (events or wake is not None):
                # Not run again. Messages run after the stop would be acted on as the user's latest words (the
                # interrupting message goes first, and they would follow it): the gated tool asked for again and
                # approved. A wake (an application's turn) would redo what its user stopped. Kept in the
                # conversation instead, as something the user stopped.
                self._drop_stopped(events, wake, run_id)
            else:
                self._restore_events(events)
                if wake is not None:
                    self._wakes.insert(0, _Wake.from_dict(wake))
            if resumed:
                self._parked = True
                if stopped:
                    # Its user stopped this resume before any of its calls ran, and the process died before
                    # the session could turn that into a cancel (_run_turn does, live). The answers go with
                    # the turn and the question is abandoned, as a stop on a parked session abandons it.
                    self._abandon_parked = "The user moved on before answering."
                else:
                    self._resume = _load_resume(in_flight["resume"])
            return "requeued"
        if root is None:
            # It may have acted and no trace of it is stored (a provider that saves at the end of a run, or
            # in the background). Nothing is run again: the recovery turn is told what it was answering (no
            # stored run holds that) and which calls it had started.
            self._scheduler.head = in_flight.get("parent_id")
            self._root_at(None, in_flight)
            asked = [event.get("text") or "" for event in events]
            if in_flight.get("wake") is not None:
                asked.append(in_flight["wake"].get("text") or "")
            unconfirmed = [f"{call.get('name') or '?'}({_brief(call.get('input'), 80)})" for call in calls]
            if resumed:
                # It was acting on the user's answers to a parked turn. Answering it again would run again
                # whatever followed them: a claimed approval protects its own call, not the tool that
                # asked the question, nor the ones the model called next. The question is abandoned instead.
                self._parked = True
                self._abandon_parked = "The server restarted while the answer to this was being acted on."
                approved = in_flight.get("approved") or []
                unconfirmed = [*approved, *(call for call in unconfirmed if call not in approved)]
            notice = _recovery_notice(
                [], unconfirmed, asked=[text for text in asked if text], untraced=True, answered=resumed
            )
        else:
            self._scheduler.head = run_id
            self._root_at(run_id, in_flight)
            self._parked = False
            # What its stored memory shows it took, by event id (every message built from session events
            # carries them). A steered one it does not show would be lost with the run: back in the queue.
            self._shown_events = _shown_events(root)
            self._restore_events(
                [event for event in events if event.get("steered") and event.get("id") not in self._shown_events]
            )
            # Calls the session recorded and the trace does not show (saved before them) are unconfirmed.
            completed, unconfirmed = _tool_calls(context, calls)
            notice = _recovery_notice(completed, unconfirmed)
        if self.on_dead_turn == "drop":
            return "dropped"
        if self.on_dead_turn == "notify":
            # Rides with the next turn: it does not start one.
            self._store.post_event(notice, key=RECOVERY_KIND, announce="silent")
            return "notified"
        self._wakes.insert(
            0,
            _Wake(
                text=notice,
                key=RECOVERY_KIND,
                # Read-only, and no further than the dead turn itself was allowed: a turn that reports on a
                # run under a narrow policy must not reach past it, however read-only what it reaches is.
                tool_policy=read_only_within(
                    _load_policy(
                        _stored_turn_policy(in_flight, last_options),
                        context._trace if context is not None else None,
                    )
                ),
                silent_ok=True,
                kind=RECOVERY_KIND,
                trigger={"kind": RECOVERY_KIND, "dead_run_id": run_id, "cut": cut or "crash"},
            ),
        )
        return "continued"

    def _watch(self, event: SessionEvent) -> None:
        if self.closed:
            return
        event.on_settle(self._settled)

    def _settled(self, event: SessionEvent) -> None:
        if self._close_done.is_set():
            return  # the session restored after this one follows it now (same process, same task store)
        self._keep_with_turn(event)
        self._record("message_settled", event_id=event.id, state=event.state, run_id=event.run_id)

    def _event_settled(self, event: SessionEvent) -> None:
        if self._close_done.is_set():
            return
        self._followed_events.discard(event.id)
        if self._keep_with_turn(event) and not self.closed:
            self._persist()

    def _keep_with_turn(self, event: SessionEvent) -> bool:
        """An event steered into the running turn is no longer pending, and until that run's trace is stored
        it is nowhere else: kept with the turn (``in_flight.events``), so a restart puts it back or tells the
        recovery turn about it. False for one the turn started with (already there) or that went elsewhere."""
        flight = self._in_flight
        if (
            event.state != "delivered"
            or flight is None
            or event.run_id != flight.get("run_id")
            or any(taken.get("id") == event.id for taken in flight["events"])
        ):
            return False
        flight["events"].append({**event.to_dict(), "steered": True})
        return True
