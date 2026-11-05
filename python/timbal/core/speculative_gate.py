"""Hold tool calls until a speculative voice turn is confirmed.

The voice session installs one :class:`SpeculativeGate` for an attempt that
started before EndOfTurn. ``confirm`` lets every waiter run. ``discard``
cancels the wait so the handler never starts.

Pipecat's ``SpeculationGate`` holds a speculative reply until the turn
commits, and drops it on ``UserTurnResponseCancelFrame`` (BSD-2-Clause,
``pipecat.turns.speculation_gate``). Deepgram's Flux guide says to cancel on
``TurnResumed`` before committing tools. This is that wait, shared by every
tool call while the gate is installed. Sessions that are not speculating
never install a gate, so :func:`wait_for_turn_confirmation` returns
immediately and tool timing is unchanged.

The class lives here, not under ``timbal.voice``, so a tool call can see the
gate without importing the voice package.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from typing import Any


class SpeculativeGate:
    """Blocks tool calls until the voice turn is confirmed.

    ``confirm`` lets every waiter run the handler. ``discard`` cancels them
    so the handler never starts. Outside a speculative turn there is no gate
    and :func:`wait_for_turn_confirmation` returns immediately.
    """

    def __init__(self) -> None:
        self._confirmed = asyncio.Event()
        self._discarded = asyncio.Event()

    @property
    def confirmed(self) -> bool:
        return self._confirmed.is_set()

    def confirm(self) -> None:
        self._confirmed.set()

    def discard(self) -> None:
        self._discarded.set()

    async def wait(self) -> None:
        if self._confirmed.is_set():
            return
        if self._discarded.is_set():
            raise asyncio.CancelledError
        confirm_task = asyncio.create_task(self._confirmed.wait())
        discard_task = asyncio.create_task(self._discarded.wait())
        try:
            await asyncio.wait(
                {confirm_task, discard_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (confirm_task, discard_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(confirm_task, discard_task, return_exceptions=True)
        if self._discarded.is_set() and not self._confirmed.is_set():
            raise asyncio.CancelledError


_GATE: ContextVar[SpeculativeGate | None] = ContextVar("timbal_speculative_gate", default=None)


def install_speculative_gate(gate: SpeculativeGate) -> Token[SpeculativeGate | None]:
    return _GATE.set(gate)


def reset_speculative_gate(token: Token[SpeculativeGate | None]) -> None:
    _GATE.reset(token)


def speculative_tool_hold_active() -> bool:
    """True while an unconfirmed speculative turn is holding tool calls."""
    gate = _GATE.get()
    return gate is not None and not gate.confirmed


def holds_for_unconfirmed_turn(runnable: Any, parent_call_id: str | None) -> bool:
    """Whether this call must wait out an unconfirmed speculative voice turn.

    The root agent and its internal LLM tool are the early reply, so they
    run. Every other runnable is a tool call: a plain tool, an MCP tool, or
    a nested agent or workflow used as one. Their handler, ``pre_hook``,
    approval gate, argument guardrails, and default-parameter callables all
    sit behind :func:`wait_for_turn_confirmation`.

    No gate means speculative generation is off (or this is not a voice
    turn). ``defer_until_eot`` does not open a hold by itself.
    """
    metadata = getattr(runnable, "metadata", None) or {}
    if metadata.get("type") == "LLM":
        return False
    if getattr(runnable, "_is_orchestrator", False) and parent_call_id is None:
        return False
    return speculative_tool_hold_active()


async def wait_for_turn_confirmation() -> None:
    """No-op without a speculative gate; otherwise block until confirm or discard."""
    gate = _GATE.get()
    if gate is None:
        return
    await gate.wait()
