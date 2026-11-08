"""Agents that message each other: a directory of addressable chat sessions and the ``send_to_agent`` tool.

Each :class:`~timbal.core.chat_session.ChatSession` that joins an :class:`AgentDirectory` under a name can
be messaged by the others, and its root agent gains ``send_to_agent``. A message lands in the recipient's
inbox as a session event (``source="agent"``) whose envelope the directory stamps, so the model can't
claim to be someone else: it starts a turn when the recipient is idle, or with ``priority="urgent"`` joins
its running turn after the current tool batch. The recipient answers with ``send_to_agent`` too (its
reply text goes to its own user), so every exchange is visible in both traces.

- **Dedupe:** a message id derived from the sender's run and tool call, so a tool call re-run on resume
  sends nothing new; the recipient also drops a second identical message from the same sender while the
  first is still unread.
- **Chatter:** every message carries ``hops``, one more than the deepest agent message the sending run
  received (or the message it replies to); past ``max_hops`` the tool refuses and tells the model to stop.
- **Access:** ``accepts=`` on join limits who may message a member (a routing filter, not authentication:
  over HTTP, ``from`` is whatever the authenticated caller claims).
- **Screening:** the recipient's input guardrails screen the text (a block refuses the message).

Process-local, like the sessions; :class:`RemotePeer` reaches a session on another Timbal server through
``POST /chat/sessions/{id}/agent_messages``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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
DEFAULT_MAX_HOPS = 6
"""Messages one chain of agent replies may run to before ``send_to_agent`` refuses."""

Accepts = Collection[str] | Callable[[str], bool] | None


class PeerRejected(Exception):
    """A message was not delivered. ``reason``: ``unknown`` | ``not_accepted`` | ``blocked`` | ``closed`` |
    ``hop_limit`` | ``invalid`` | ``unreachable``."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _accepts(rule: Accepts, sender: str) -> bool:
    if rule is None:
        return True
    if callable(rule):
        return bool(rule(sender))
    return sender in rule


@dataclass
class RemotePeer:
    """A chat session served by another Timbal server: messages go to its ``agent_messages`` route."""

    url: str
    """The server's base URL (the chat routes are under ``{url}/chat``)."""
    session_id: str
    headers: dict[str, str] = field(default_factory=dict)
    description: str | None = None
    timeout: float = 30.0

    async def deliver(self, envelope: dict[str, Any]) -> dict[str, Any]:
        import httpx

        url = f"{self.url.rstrip('/')}/chat/sessions/{self.session_id}/agent_messages"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=envelope, headers=self.headers)
        except httpx.HTTPError as e:
            raise PeerRejected("unreachable", f"Could not reach the agent: {type(e).__name__}.") from e
        if response.status_code == 403:
            raise PeerRejected("not_accepted", _detail(response) or "The agent does not take messages from you.")
        if response.status_code == 404:
            raise PeerRejected("unknown", "The agent's session no longer exists.")
        if response.status_code == 409:
            raise PeerRejected("closed", "The agent's session is closed.")
        if response.status_code == 422:
            raise PeerRejected("invalid", _detail(response) or "The agent refused the message.")
        if response.status_code >= 400:
            raise PeerRejected("unreachable", f"The agent's server answered {response.status_code}.")
        return response.json()


def _detail(response: Any) -> str | None:
    try:
        detail = response.json().get("detail")
    except Exception:
        return None
    return detail if isinstance(detail, str) else None


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
        self._loop: asyncio.AbstractEventLoop | None = None
        with contextlib.suppress(RuntimeError):  # joined outside a loop: delivered on the sender's
            self._loop = asyncio.get_running_loop()
        self._tools: dict[str, tuple[int, Tool]] = {}

    def accepts_from(self, sender: str) -> bool:
        return _accepts(self.accepts, sender)

    def leave(self) -> None:
        self.directory.leave(self.name, membership=self)

    def tool(self, agent: Agent) -> Tool:
        """``send_to_agent`` for ``agent``: rebuilt only when the directory changes (its description lists peers)."""
        cached = self._tools.get(agent._path)
        if cached is not None and cached[0] == self.directory.version:
            return cached[1]
        tool = self._build_tool(agent)
        self._tools[agent._path] = (self.directory.version, tool)
        return tool

    def _build_tool(self, agent: Agent) -> Tool:
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
            run_context = get_run_context()
            hops = 0
            if run_context is not None and run_context._peer_hops is not None:
                hops = run_context._peer_hops + 1
            if reply_to:
                answered = membership.session._agent_message(reply_to)
                if answered is not None:
                    hops = max(hops, (answered.peer.get("hops") or 0) + 1)
            message_id = _message_id(run_context.id if run_context is not None else "", get_call_id() or "")
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
                raise ValueError(f"Not sent: {e}") from None

        peers = self.directory.peers(for_name=self.name)
        listing = "\n".join(f"- {name}" + (f": {description}" if description else "") for name, description in peers)
        tool = Tool(
            name=SEND_TO_AGENT_NAME,
            description=(
                "Send a message to another agent. It arrives in that agent's inbox; it answers, if it needs to, "
                "with a message of its own that reaches you later. Only message an agent when it can do or know "
                "something you need, or needs something from you; never just to acknowledge or thank.\n\n"
                + ("Agents you can message:\n" + listing if listing else "No other agent can be messaged right now.")
            ),
            handler=send_to_agent,
        )
        tool.nest(agent._path)
        return tool


def _message_id(run_id: str, call_id: str) -> str:
    # The same tool call (a re-run on resume) always sends the same id: the recipient drops the repeat.
    return "am_" + hashlib.sha256(f"{run_id}:{call_id}".encode()).hexdigest()[:24]


class AgentDirectory:
    """Names agents' chat sessions so they can message each other (``send_to_agent``)."""

    def __init__(self, *, max_hops: int = DEFAULT_MAX_HOPS) -> None:
        if max_hops < 1:
            raise ValueError("max_hops must be at least 1.")
        self.max_hops = max_hops
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
        """Make ``session`` reachable as ``name``; its root agent gains ``send_to_agent``. Left on ``close()``."""
        name = _check_name(name)
        if name in self._members:
            raise ValueError(f"An agent named {name!r} is already in the directory.")
        if session.closed:
            raise RuntimeError("The chat session is closed.")
        if session._peer is not None:
            raise ValueError(f"The session is already in a directory as {session._peer.name!r}.")
        membership = PeerMembership(self, name, session, description=description, accepts=accepts)
        self._members[name] = membership
        session._peer = membership
        self.version += 1
        return membership

    def add_remote(self, name: str, peer: RemotePeer) -> None:
        """Make a session on another server reachable as ``name``."""
        name = _check_name(name)
        if name in self._members:
            raise ValueError(f"An agent named {name!r} is already in the directory.")
        self._members[name] = peer
        self.version += 1

    def leave(self, name: str, *, membership: PeerMembership | None = None) -> None:
        member = self._members.get(name)
        if member is None or (membership is not None and member is not membership):
            return
        del self._members[name]
        if isinstance(member, PeerMembership) and member.session._peer is member:
            member.session._peer = None
        self.version += 1

    def peers(self, *, for_name: str | None = None) -> list[tuple[str, str | None]]:
        """``(name, description)`` of every agent ``for_name`` may message, sorted (a stable tool description)."""
        found = []
        for name, member in sorted(self._members.items()):
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
    session = member.session
    receive = session.receive_agent_message(
        envelope["text"],
        sender=envelope["from"],
        message_id=envelope["message_id"],
        reply_to=envelope["reply_to"],
        priority=envelope["priority"],
        hops=envelope["hops"],
    )
    loop = member._loop
    if loop is not None and not loop.is_closed() and loop is not asyncio.get_running_loop():
        # Sessions are served on their own loop: deliver there.
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(receive, loop))
    return await receive


def _check_name(name: str) -> str:
    name = name.strip() if isinstance(name, str) else ""
    if not name or len(name) > 64 or any(ch.isspace() or ch in "\"'<>&" for ch in name):
        raise ValueError("An agent's name must be 1-64 characters, without spaces, quotes or <>&.")
    return name
