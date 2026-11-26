"""Agents that message each other: a directory of addressable chat sessions and the ``send_to_agent`` tool.

Each :class:`~timbal.core.chat_session.ChatSession` that joins an :class:`AgentDirectory` under a name can
be messaged by the others, and its root agent gains ``send_to_agent``. A message lands in the recipient's
inbox as a session event (``source="agent"``) whose envelope the directory stamps, so the model can't
claim to be someone else: it starts a turn when the recipient is idle, or with ``priority="urgent"`` joins
its running turn after the current tool batch. The recipient answers with ``send_to_agent`` too (its
reply text goes to its own user), so every exchange is visible in both traces.

- **Dedupe:** a message id derived from the sender's run and tool call, so a tool call re-run on resume
  sends nothing new; ids are per sender. The recipient also drops a second identical message from the
  same sender while the first is still unread.
- **Chatter:** every message carries ``hops``, one more than the deepest agent message the sending run
  received (or the message it replies to); past ``max_hops`` the tool refuses and tells the model to stop.
- **Access:** ``accepts=`` on join limits who may message a member (a routing filter, not authentication:
  over HTTP, ``from`` is whatever the authenticated caller claims).
- **Screening:** the recipient's input guardrails screen the text on arrival: a block, or a redaction that
  leaves nothing, refuses the message; a redaction rewrites it and is recorded on the run that takes it.

Process-local, like the sessions; :class:`RemotePeer` reaches a session on another Timbal server through
``POST /chat/sessions/{id}/agent_messages``.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import os
import re
import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import structlog

from ..state import get_call_id, get_run_context
from .tool import Tool

if TYPE_CHECKING:
    from .agent import Agent
    from .chat_session import ChatSession

__all__ = [
    "DEFAULT_MAX_HOPS",
    "SEND_TO_AGENT_NAME",
    "AgentDirectory",
    "PeerMembership",
    "PeerPriority",
    "PeerRejected",
    "RemotePeer",
]

logger = structlog.get_logger("timbal.core.peers")

PeerPriority = Literal["normal", "urgent"]
SEND_TO_AGENT_NAME = "send_to_agent"
# Used when a tool set the agent has brings a tool of that name (an MCP server's own ``send_to_agent``).
_SEND_TO_AGENT_FALLBACKS = ("message_agent", "send_agent_message")
DEFAULT_MAX_HOPS = 6
"""Messages one chain of agent replies may run to before ``send_to_agent`` refuses."""

Accepts = Collection[str] | Callable[[str], bool] | None

# A message id, as the recipient keeps it (dedupe, ``reply_to``) and shows it to its model: what the directory
# makes (``am_`` and hex) and the usual id alphabets, not free text.
_MESSAGE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")

# Guards every directory's members: sessions are served on their own loops, and join and leave from their
# threads. One lock for all directories (held for a few dict operations, never while user code runs). Re-entrant:
# a membership's ``accepts`` is never called under it, but a directory method called from a callback may be.
_LOCK = threading.RLock()


def _new_lock() -> None:
    """In a forked child: the parent's lock may have been held by a thread that does not exist here."""
    global _LOCK
    _LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_lock)


class PeerRejected(Exception):
    """A message was not delivered. ``reason``: ``unknown`` | ``not_accepted`` | ``blocked`` | ``closed`` |
    ``hop_limit`` | ``invalid`` | ``unreachable`` (nothing reached the agent) | ``unconfirmed`` (a remote agent's
    server did not confirm it: it may have arrived)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _accepts(rule: Accepts, sender: str) -> bool:
    if rule is None:
        return True
    if callable(rule):
        return bool(rule(sender))
    return sender in rule


def _check_name(name: Any) -> str:
    name = name.strip() if isinstance(name, str) else ""
    if not name or len(name) > 64 or any(ch.isspace() or ch in "\"'<>&" or not ch.isprintable() for ch in name):
        raise ValueError("An agent's name must be 1-64 characters, without spaces, quotes or <>&.")
    return name


def _check_message_id(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _MESSAGE_ID.fullmatch(value):
        raise PeerRejected("invalid", f"{what} must be 1-128 characters of letters, digits and _.:-")
    return value


_REASONS = frozenset(
    {"unknown", "not_accepted", "blocked", "closed", "hop_limit", "invalid", "unreachable", "unconfirmed"}
)
_REASON_BY_STATUS = {400: "invalid", 403: "not_accepted", 404: "unknown", 409: "closed", 422: "invalid"}
_UNCONFIRMED = (
    "The agent's server did not confirm it: the message may have arrived. Do not send it again; tell your user "
    "if it matters."
)
_FALLBACK_MESSAGES = {
    "not_accepted": "The agent does not take messages from you.",
    "unknown": "The agent's session no longer exists.",
    "closed": "The agent's session is closed.",
    "invalid": "The agent refused the message.",
}


_RECEIPT_STATES = frozenset({"pending", "delivered", "superseded", "withdrawn", "undelivered"})


def _is_receipt(body: Any) -> bool:
    """What the route answers for a message it queued (``SessionEvent.receipt``, plus ``duplicate``)."""
    return (
        isinstance(body, dict)
        and isinstance(body.get("event_id"), str)
        and bool(body["event_id"])
        and body.get("state") in _RECEIPT_STATES
        and isinstance(body.get("duplicate"), bool)
    )


def _not_sent() -> tuple[type[BaseException], ...]:
    import httpx

    # Raised before any byte of the request went out: no connection, a bad URL or proxy, a request httpx refuses.
    return (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        httpx.ProxyError,
        httpx.UnsupportedProtocol,
        httpx.LocalProtocolError,
        httpx.InvalidURL,
    )


@dataclass
class RemotePeer:
    """A chat session served by another Timbal server: messages go to its ``agent_messages`` route.

    Only ``headers`` go out with a message (the run's platform credentials never do): give it what that
    server takes. They are left out of the peer's ``repr``.
    """

    url: str
    """The server's base URL (the chat routes are under ``{url}/chat``)."""
    session_id: str
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    description: str | None = None
    timeout: float = 30.0

    async def deliver(self, envelope: dict[str, Any]) -> dict[str, Any]:
        import httpx

        not_sent = _not_sent()
        url = f"{self.url.rstrip('/')}/chat/sessions/{quote(self.session_id, safe='')}/agent_messages"
        try:
            # No redirects: the headers are for this server, and a receipt comes from the route itself.
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                response = await client.post(url, json=envelope, headers=self.headers)
        except not_sent as e:
            # Nothing was sent: no connection was made, or the request could not be made at all.
            raise PeerRejected("unreachable", f"Could not reach the agent: {type(e).__name__}.") from e
        except httpx.HTTPError as e:
            # Sent, or partly: the server may have taken it before the answer was lost.
            raise PeerRejected("unconfirmed", _UNCONFIRMED) from e
        body: Any = None
        with contextlib.suppress(ValueError):
            body = response.json()
        if response.status_code in _REASON_BY_STATUS:
            said = body if isinstance(body, dict) else {}
            reason = said.get("reason") if said.get("reason") in _REASONS else _REASON_BY_STATUS[response.status_code]
            detail = said.get("detail")
            message = detail if isinstance(detail, str) and detail else _FALLBACK_MESSAGES.get(reason, "Refused.")
            raise PeerRejected(reason, message)
        if response.status_code >= 500 and not (response.status_code == 503 and "retry-after" in response.headers):
            # The route answers a 503 with Retry-After only when nothing was accepted: without it the message may be
            # stored (or a turn may have it), and any other server error (a gateway's 502 or 504 after it forwarded
            # the request) may have come after it was queued.
            raise PeerRejected("unconfirmed", _UNCONFIRMED)
        if not response.is_success:
            raise PeerRejected("unreachable", f"The agent's server answered {response.status_code}.")
        if not _is_receipt(body):
            # Taken by something, and nothing in its answer says a message was queued, nor which.
            raise PeerRejected("unconfirmed", _UNCONFIRMED)
        return body


class PeerMembership:
    """One session's place in a directory: its name, who may message it, and its ``send_to_agent`` tool."""

    def __init__(
        self,
        directory: AgentDirectory,
        name: str,
        session: ChatSession,
        *,
        description: str | None,
        accepts: Accepts,
    ) -> None:
        self.directory = directory
        self.name = name
        self.session = session
        self.description = description
        self.accepts = accepts
        # The tool by agent path and name, with the directory version it lists the peers of.
        self._tools: dict[tuple[str, str], tuple[int, Tool]] = {}

    @property
    def current(self) -> bool:
        """Whether the directory still has this session under its name (one that left may have been replaced)."""
        return self.directory._members.get(self.name) is self

    def accepts_from(self, sender: str) -> bool:
        return _accepts(self.accepts, sender)

    def leave(self) -> None:
        self.directory.leave(self.name, membership=self)

    def tool(self, agent: Agent, name: str = SEND_TO_AGENT_NAME) -> Tool:
        """``send_to_agent`` for ``agent``, under ``name`` (the agent picks it: see ``Agent._with_peer_tool``).

        Rebuilt only when the directory changes (its description lists the peers): the prompt cache holds.
        """
        key = (agent._path, name)
        version = self.directory.version
        cached = self._tools.get(key)
        if cached is not None and cached[0] == version:
            return cached[1]
        tool = self._build_tool(agent, name)
        self._tools[key] = (version, tool)
        return tool

    def _build_tool(self, agent: Agent, name: str) -> Tool:
        membership = self

        async def send_to_agent(
            to: str, message: str, reply_to: str | None = None, priority: PeerPriority = "normal"
        ) -> dict[str, Any]:
            """Send another agent a message.

            Args:
                to: The agent's name, from the list in this tool's description.
                message: What to tell or ask it, self-contained: it does not see your conversation.
                reply_to: The message_id of the agent message you are answering, if any.
                priority: "urgent" only when it must reach the agent before it finishes what it is doing.
            """
            if not membership.current:
                # Left the directory (the session closed, or the application took it out): another session may
                # have its name now, and a message stamped with it would claim to be that one.
                raise ValueError("Not sent: you can no longer message other agents.")
            run_context = get_run_context()
            hops = 0
            if run_context is not None and run_context._peer_hops is not None:
                hops = run_context._peer_hops + 1
            if reply_to:
                answered = membership.session._agent_message(to, reply_to)
                if answered is not None:
                    hops = max(hops, (answered.peer.get("hops") or 0) + 1)
            message_id = _sent_message_id(run_context)
            try:
                return await membership.directory.send(
                    membership.name,
                    to,
                    message,
                    message_id=message_id,
                    reply_to=reply_to,
                    priority=priority,
                    hops=hops,
                )
            except PeerRejected as e:
                raise ValueError(f"{'Not confirmed' if e.reason == 'unconfirmed' else 'Not sent'}: {e}") from None

        peers = self.directory.peers(for_name=self.name)
        listing = "\n".join(f"- {peer}" + (f": {description}" if description else "") for peer, description in peers)
        tool = Tool(
            name=name,
            description=(
                "Send a message to another agent. It arrives in that agent's inbox; it answers, if it needs to, "
                "with a message of its own that reaches you later. To answer a message an agent sent you, send to "
                "its from= name with its message_id as reply_to. Only message an agent when it can do or know "
                "something you need, or needs something from you; never just to acknowledge or thank.\n\n"
                + ("Agents you can message:\n" + listing if listing else "No other agent can be messaged right now.")
            ),
            handler=send_to_agent,
        )
        tool.nest(agent._path)
        # One of the agent's own (``_note_pause``): a call that waits keeps this tool, not one of the same name.
        tool._builtin = SEND_TO_AGENT_NAME
        # Its ``tool_args`` rails screen what the agent sends, as for any of its calls (a block, an approval).
        agent._wire_tool_guardrails(tool)
        return tool


def _leave_on_close(session: ChatSession) -> None:
    """A closing session (``closed`` set already) leaves its directory: its membership is read under the lock, so a
    join racing the close (from another thread) either is found here or finds the session closed."""
    with _LOCK:
        membership = session._peer
        if membership is not None:
            membership.leave()


#: The agent span's record of the ids its sends used, by tool call id (carried into the run that resumes a pause:
#: ``Agent.resolve_memory``).
PEER_SENT_KEY = "peer_sent"


def _message_id(run_id: str, call_id: str) -> str:
    return "am_" + hashlib.sha256(f"{run_id}:{call_id}".encode()).hexdigest()[:24]


def _sent_message_id(run_context: Any) -> str:
    """The id of the message this ``send_to_agent`` call sends: the one an earlier attempt at the same tool call
    used, else a new one, recorded on the agent's span before anything goes out.

    A call cut before its answer came back (its caller stopped reading at another call's approval card) runs again
    when the turn resumes, under another run and span: with the id it had, the peer takes it once."""
    if run_context is None:
        return _message_id("", get_call_id() or "")
    span = run_context.current_span()
    tool_call_id = span.metadata.get("tool_call_id") if span is not None else None
    message_id = _message_id(str(run_context.id), get_call_id() or "")
    agent_span = run_context.parent_span() if tool_call_id else None
    if agent_span is None:
        return message_id
    sent = agent_span.metadata.get(PEER_SENT_KEY)
    if isinstance(sent, dict) and isinstance(sent.get(tool_call_id), str):
        return sent[tool_call_id]
    # A copy, never written in place: a span's metadata starts as a shallow copy of the agent's own.
    agent_span.metadata[PEER_SENT_KEY] = {**(sent if isinstance(sent, dict) else {}), tool_call_id: message_id}
    return message_id


class AgentDirectory:
    """Names agents' chat sessions so they can message each other (``send_to_agent``)."""

    def __init__(self, *, max_hops: int = DEFAULT_MAX_HOPS) -> None:
        if max_hops < 1:
            raise ValueError("max_hops must be at least 1.")
        self.max_hops = max_hops
        self._versions = itertools.count(1)
        self.version = 0
        self._members: dict[str, PeerMembership | RemotePeer] = {}

    def join(
        self,
        session: ChatSession,
        name: str,
        *,
        description: str | None = None,
        accepts: Accepts = None,
    ) -> PeerMembership:
        """Make ``session`` reachable as ``name``; its root agent gains ``send_to_agent`` (another name if one of
        its tools has that one, see ``Agent._with_peer_tool``). Left on ``close()``."""
        name = _check_name(name)
        membership = PeerMembership(self, name, session, description=description, accepts=accepts)
        with _LOCK:
            # Under the lock: a close on the session's own thread reads its membership under it too
            # (_leave_on_close), so it either finds this one or this join finds the session closed.
            if session.closed:
                raise RuntimeError("The chat session is closed.")
            if name in self._members:
                raise ValueError(f"An agent named {name!r} is already in the directory.")
            if session._peer is not None:
                raise ValueError(f"The session is already in a directory as {session._peer.name!r}.")
            self._members[name] = membership
            session._peer = membership
            self.version = next(self._versions)
        return membership

    def add_remote(self, name: str, peer: RemotePeer) -> None:
        """Make a session on another server reachable as ``name``."""
        name = _check_name(name)
        with _LOCK:
            if name in self._members:
                raise ValueError(f"An agent named {name!r} is already in the directory.")
            self._members[name] = peer
            self.version = next(self._versions)

    def leave(self, name: str, *, membership: PeerMembership | None = None) -> None:
        with _LOCK:
            member = self._members.get(name)
            if member is None or (membership is not None and member is not membership):
                return
            del self._members[name]
            if isinstance(member, PeerMembership) and member.session._peer is member:
                member.session._peer = None
            self.version = next(self._versions)

    def peers(self, *, for_name: str | None = None) -> list[tuple[str, str | None]]:
        """``(name, description)`` of every agent ``for_name`` may message, sorted (a stable tool description)."""
        with _LOCK:
            members = sorted(self._members.items())
        found = []
        for name, member in members:
            if name == for_name:
                continue
            if isinstance(member, PeerMembership) and for_name is not None and not member.accepts_from(for_name):
                continue
            found.append((name, member.description))
        return found

    async def send(
        self,
        sender: str,
        to: str,
        text: str,
        *,
        message_id: str | None = None,
        reply_to: str | None = None,
        priority: PeerPriority = "normal",
        hops: int = 0,
    ) -> dict[str, Any]:
        """Deliver a message from ``sender`` to ``to``; returns the recipient's receipt.

        Raises :class:`PeerRejected`. The application can call it too (``sender`` is then whatever name it
        speaks for).
        """
        if to == sender:
            raise PeerRejected("invalid", "An agent cannot message itself.")
        member = self._members.get(to)
        if member is None:
            known = ", ".join(name for name, _ in self.peers(for_name=sender)) or "none"
            raise PeerRejected("unknown", f"No agent named {to!r} (agents you can message: {known}).")
        if hops >= self.max_hops:
            raise PeerRejected(
                "hop_limit",
                f"this exchange between agents already went back and forth {hops} times. Stop messaging "
                "and tell your user where things stand.",
            )
        envelope = {
            "from": sender,
            "text": text,
            "message_id": message_id,
            "reply_to": reply_to,
            "priority": priority,
            "hops": hops,
        }
        if isinstance(member, RemotePeer):
            receipt = await member.deliver(envelope)
        else:
            receipt = await _deliver_local(member, envelope)
        return {"status": "duplicate" if receipt.get("duplicate") else "sent", "to": to, **receipt}


async def _deliver_local(member: PeerMembership, envelope: dict[str, Any]) -> dict[str, Any]:
    # (The session runs it on the loop that serves it, whichever loop this is: ChatSession.receive_agent_message.)
    return await member.session.receive_agent_message(
        envelope["text"],
        sender=envelope["from"],
        message_id=envelope["message_id"],
        reply_to=envelope["reply_to"],
        priority=envelope["priority"],
        hops=envelope["hops"],
    )
