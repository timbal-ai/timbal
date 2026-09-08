# ruff: noqa: ARG001, ARG005 — test handlers declare params for schema, do not consume them
"""Session-scoped approvals — "yes, and don't ask me again for this tool".

``resume={approval_id: {"approved": True, "scope": "session"}}`` approves the call and
leaves an ``ApprovalGrant`` in the run's session data under the decision's ``grant_key``
(default: the runnable path). Every later gate in the same session (``parent_id`` chain)
that resolves to that key is approved without a card, with a synthesized resolution
pointing back at the human decision.

Invariants:

- A grant covers *any input* under the key — that is the point. A tool that must not be
  remembered says so with ``grantable=False`` / ``approval_grantable=False``; the card
  then hides the option and a resolution that asks anyway is honoured for the call only.
- ``grant_key`` narrows a grant below "this tool": a subcommand-multiplexing tool keys
  per subcommand so remembering one does not wave the others through.
- An explicit resume value for a call beats a grant (declining one occurrence after
  remembering the tool means that occurrence).
- Grants live in the session, so they travel with the trace: the JSONL provider round
  trip is the same code path the platform provider uses.
- A decline is never remembered.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from timbal import Agent, Tool
from timbal.core.runnable import _approval_id_for
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.types.approval import APPROVAL_GRANTS_SESSION_KEY, ApprovalGrant, ApprovalResolution
from timbal.types.content import ToolResultContent, ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message


def _approval_event(events) -> ApprovalEvent | None:
    return next((e for e in events if isinstance(e, ApprovalEvent)), None)


def _final(events) -> OutputEvent:
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


def _tool_out(events, path: str) -> OutputEvent:
    return next(e for e in events if isinstance(e, OutputEvent) and e.path == path)


def _tool_call(name: str, inp: dict, cid: str = "c1") -> Message:
    return Message(role="assistant", content=[ToolUseContent(id=cid, name=name, input=inp)], stop_reason="tool_use")


REMEMBER = {"approved": True, "scope": "session"}


# ---------------------------------------------------------------------------
# Direct tool calls
# ---------------------------------------------------------------------------


class TestSessionGrantOnTool:
    async def test_session_scope_approves_later_calls_with_other_inputs(self):
        tool = Tool(name="rm", handler=lambda path: f"removed {path}", requires_approval=True)

        first = [e async for e in tool(path="/a")]
        card = _approval_event(first)
        assert card is not None
        assert card.grantable is True
        assert card.grant_key == "rm"  # "this tool" by default

        approved = await tool(path="/a", parent_id=_final(first).run_id, resume={card.approval_id: REMEMBER}).collect()
        assert approved.output == "removed /a"
        assert approved.metadata["approval"]["resolution"]["scope"] == "session"
        assert approved.usage.get("approvals:remembered") == 1

        # A different input, later in the session: no card.
        later_events = [e async for e in tool(path="/b", parent_id=approved.run_id)]
        assert _approval_event(later_events) is None
        later = _final(later_events)
        assert later.output == "removed /b"
        assert later.status.code == "success"
        granted = later.metadata["approval"]["granted"]
        assert granted["key"] == "rm"
        assert granted["approval_id"] == card.approval_id
        resolution = later.metadata["approval"]["resolution"]
        assert resolution["approved"] is True
        assert resolution["scope"] == "session"
        assert resolution["metadata"] == {"granted_by": card.approval_id, "grant_key": "rm"}
        assert later.usage.get("approvals:granted") == 1
        assert "approvals:required" not in later.usage

        # And it keeps holding one more hop down the chain.
        again = await tool(path="/c", parent_id=later.run_id).collect()
        assert again.output == "removed /c"

    async def test_call_scope_is_the_default_and_covers_one_input_only(self):
        tool = Tool(name="rm", handler=lambda path: f"removed {path}", requires_approval=True)
        first = [e async for e in tool(path="/a")]
        card = _approval_event(first)
        approved = await tool(path="/a", parent_id=_final(first).run_id, resume={card.approval_id: True}).collect()
        assert approved.metadata["approval"]["resolution"]["scope"] == "call"
        assert APPROVAL_GRANTS_SESSION_KEY not in (await _session_of(approved))

        later = [e async for e in tool(path="/b", parent_id=approved.run_id)]
        assert _approval_event(later) is not None

    async def test_explicit_resume_for_a_call_beats_the_grant(self):
        tool = Tool(name="rm", handler=lambda path: f"removed {path}", requires_approval=True)
        first = [e async for e in tool(path="/a")]
        card = _approval_event(first)
        approved = await tool(path="/a", parent_id=_final(first).run_id, resume={card.approval_id: REMEMBER}).collect()

        # The human declines the next occurrence explicitly (a client may still show the
        # card, or the decline may be a policy's doing). The explicit value wins.
        b_id = _approval_id_for("rm", {"path": "/b"})
        declined = await tool(path="/b", parent_id=approved.run_id, resume={b_id: False}).collect()
        assert declined.status.reason == "approval_denied"
        assert "granted" not in declined.metadata["approval"]

    async def test_decline_is_never_remembered(self):
        tool = Tool(name="rm", handler=lambda path: f"removed {path}", requires_approval=True)
        first = [e async for e in tool(path="/a")]
        card = _approval_event(first)
        declined = await tool(
            path="/a",
            parent_id=_final(first).run_id,
            resume={card.approval_id: {"approved": False, "scope": "session"}},
        ).collect()
        assert declined.status.reason == "approval_denied"
        assert APPROVAL_GRANTS_SESSION_KEY not in (await _session_of(declined))
        later = [e async for e in tool(path="/a", parent_id=declined.run_id)]
        assert _approval_event(later) is not None

    async def test_pending_approvals_carry_the_grant_fields(self):
        """A client that reloads a paused run reads `pending_approvals()`, not the event."""
        from timbal.state import get_run_context

        tool = Tool(name="wire", handler=lambda amount: "ok", requires_approval=True, approval_grantable=False)
        events = [e async for e in tool(amount=1)]
        card = _approval_event(events)
        assert card is not None
        entry = next(e for e in get_run_context().pending_approvals() if e["approval_id"] == card.approval_id)
        assert entry["grant_key"] == "wire"
        assert entry["grantable"] is False

    async def test_not_grantable_hides_the_option_and_refuses_the_memory(self):
        tool = Tool(
            name="wire",
            handler=lambda amount: f"sent {amount}",
            requires_approval=True,
            approval_grantable=False,
        )
        first = [e async for e in tool(amount=100)]
        card = _approval_event(first)
        assert card.grantable is False

        approved = await tool(amount=100, parent_id=_final(first).run_id, resume={card.approval_id: REMEMBER}).collect()
        assert approved.output == "sent 100"  # the call itself is approved
        assert approved.metadata["approval"]["grant_refused"] is True
        assert "approvals:remembered" not in approved.usage
        assert APPROVAL_GRANTS_SESSION_KEY not in (await _session_of(approved))

        later = [e async for e in tool(amount=200, parent_id=approved.run_id)]
        assert _approval_event(later) is not None

    async def test_grant_key_from_the_decision_narrows_the_memory(self):
        """One tool, several subcommands: remembering `delete` must not cover `drop`."""

        def policy(command: str, target: str):
            return {"required": command in {"delete", "drop"}, "grant_key": f"admin:{command}"}

        tool = Tool(name="admin", handler=lambda command, target: f"{command} {target}", requires_approval=policy)

        first = [e async for e in tool(command="delete", target="t1")]
        card = _approval_event(first)
        assert card.grant_key == "admin:delete"
        approved = await tool(
            command="delete", target="t1", parent_id=_final(first).run_id, resume={card.approval_id: REMEMBER}
        ).collect()
        assert approved.output == "delete t1"

        # Same subcommand, other target: covered.
        covered = [e async for e in tool(command="delete", target="t2", parent_id=approved.run_id)]
        assert _approval_event(covered) is None
        assert _final(covered).output == "delete t2"

        # Other gated subcommand: not covered.
        not_covered = [e async for e in tool(command="drop", target="t2", parent_id=approved.run_id)]
        assert _approval_event(not_covered) is not None
        assert _approval_event(not_covered).grant_key == "admin:drop"

        # Ungated subcommand never asked in the first place.
        free = await tool(command="list", target="t2", parent_id=approved.run_id).collect()
        assert free.output == "list t2"

    async def test_grant_key_from_the_tool_field_and_callable(self):
        static = Tool(name="a", handler=lambda x: x, requires_approval=True, approval_grant_key="shared-key")
        dynamic = Tool(name="b", handler=lambda x: x, requires_approval=True, approval_grant_key=lambda x: f"b:{x % 2}")
        assert _approval_event([e async for e in static(x=1)]).grant_key == "shared-key"
        assert _approval_event([e async for e in dynamic(x=3)]).grant_key == "b:1"
        assert _approval_event([e async for e in dynamic(x=4)]).grant_key == "b:0"

    async def test_expires_at_on_the_resolution_bounds_the_grant(self):
        tool = Tool(name="rm", handler=lambda path: f"removed {path}", requires_approval=True)
        first = [e async for e in tool(path="/a")]
        card = _approval_event(first)
        soon = int(time.time() * 1000) + 300
        approved = await tool(
            path="/a",
            parent_id=_final(first).run_id,
            resume={card.approval_id: ApprovalResolution(approved=True, scope="session", expires_at=soon)},
        ).collect()
        grant = ApprovalGrant.model_validate((await _session_of(approved))[APPROVAL_GRANTS_SESSION_KEY]["rm"])
        assert grant.expires_at == soon

        covered = [e async for e in tool(path="/b", parent_id=approved.run_id)]
        assert _approval_event(covered) is None

        time.sleep(0.35)
        expired = [e async for e in tool(path="/c", parent_id=_final(covered).run_id)]
        assert _approval_event(expired) is not None

    async def test_grant_carries_the_audit_fields(self):
        tool = Tool(name="rm", handler=lambda path: "ok", requires_approval=True)
        first = [e async for e in tool(path="/a")]
        card = _approval_event(first)
        approved = await tool(
            path="/a",
            parent_id=_final(first).run_id,
            resume={
                card.approval_id: {
                    "approved": True,
                    "scope": "session",
                    "approver_id": "user_42",
                    "comment": "trusted",
                    "decided_at": 4242,
                }
            },
        ).collect()
        grant = ApprovalGrant.model_validate((await _session_of(approved))[APPROVAL_GRANTS_SESSION_KEY]["rm"])
        assert grant.approver_id == "user_42"
        assert grant.comment == "trusted"
        assert grant.granted_at == 4242
        assert grant.run_id == approved.run_id
        assert grant.runnable_path == "rm"

        later = await tool(path="/b", parent_id=approved.run_id).collect()
        resolution = later.metadata["approval"]["resolution"]
        assert resolution["approver_id"] == "user_42"
        assert resolution["comment"] == "trusted"
        assert resolution["decided_at"] == 4242


async def _session_of(out: OutputEvent) -> dict:
    """The session data a child of ``out`` would inherit."""
    from timbal.state import RunContext

    child = RunContext(parent_id=out.run_id)
    return await child.get_session()


# ---------------------------------------------------------------------------
# Inside an agent
# ---------------------------------------------------------------------------


class TestSessionGrantInAgent:
    async def test_agent_stops_carding_the_tool_after_remember(self):
        def rm(path: str) -> str:
            """Remove a path."""
            return f"removed {path}"

        tool = Tool(name="rm", handler=rm, requires_approval=True)

        def wants(path: str):
            # Memory-independent: call the tool unless the last message is its result.
            def handler(messages):
                last = messages[-1] if messages else None
                if last is not None and any(isinstance(c, ToolResultContent) for c in last.content):
                    return "done"
                return _tool_call("rm", {"path": path}, cid=f"c-{path}")

            return handler

        model = TestModel(handler=wants("/a"))
        agent = Agent(name="a", model=model, tools=[tool], max_iter=3)

        first = [e async for e in agent(prompt="rm /a")]
        card = _approval_event(first)
        assert card is not None
        assert card.grant_key == "a.rm"
        assert _final(first).status.reason == "approval_required"

        resumed = [
            e async for e in agent(prompt="rm /a", parent_id=_final(first).run_id, resume={card.approval_id: REMEMBER})
        ]
        assert _tool_out(resumed, "a.rm").output == "removed /a"
        assert _final(resumed).status.code == "success"

        # Next turn, other path: the tool just runs.
        agent2 = Agent(name="a", model=TestModel(handler=wants("/b")), tools=[tool], max_iter=3)
        second = [e async for e in agent2(prompt="rm /b", parent_id=_final(resumed).run_id)]
        assert _approval_event(second) is None
        assert _tool_out(second, "a.rm").output == "removed /b"
        assert _final(second).status.code == "success"
        assert _final(second).usage.get("approvals:granted") == 1


# ---------------------------------------------------------------------------
# Durable provider round trip
# ---------------------------------------------------------------------------


class TestGrantSurvivesDurableProvider:
    async def test_jsonl_round_trip(self, tmp_path: Path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
        tool = Tool(
            name="rm", handler=lambda path: f"removed {path}", requires_approval=True, tracing_provider=provider
        )

        first = [e async for e in tool(path="/a")]
        card = _approval_event(first)
        approved = await tool(path="/a", parent_id=_final(first).run_id, resume={card.approval_id: REMEMBER}).collect()
        assert approved.output == "removed /a"

        # A "new process": a fresh tool object with the same provider and the parent id.
        tool2 = Tool(
            name="rm", handler=lambda path: f"removed {path}", requires_approval=True, tracing_provider=provider
        )
        later = [e async for e in tool2(path="/b", parent_id=approved.run_id)]
        assert _approval_event(later) is None
        assert _final(later).output == "removed /b"
        assert _final(later).metadata["approval"]["granted"]["approval_id"] == card.approval_id

        # The grant is on disk in the root span's session.
        lines = (tmp_path / "traces.jsonl").read_text().splitlines()
        assert any(APPROVAL_GRANTS_SESSION_KEY in line and card.approval_id in line for line in lines)


@pytest.mark.parametrize("scope", ["call", "session"])
def test_resolution_scope_is_validated(scope: str):
    assert ApprovalResolution(approved=True, scope=scope).scope == scope


def test_resolution_rejects_unknown_scope():
    with pytest.raises(ValueError):
        ApprovalResolution(approved=True, scope="galaxy")
