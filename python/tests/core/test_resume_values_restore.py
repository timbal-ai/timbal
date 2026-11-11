"""Resume values stay on a RunContext only while the call that brought them runs.

``Runnable._stream`` installs the ``resume=`` values of a call on the shared
RunContext and takes them off when the call ends. A value may be an approval
decision, so one left behind — or put back — after its call is a decision that
outlives the call it was given to. Three ways that used to happen (issue #279):

- something writing into the mapping in place: the "previous" mapping a call
  restores is the live one, so the write would survive the call
- a call that brought no values but outlived the one that did (a background
  child) writing back the mapping it saw when it started
- a call failing between the install and the ``try`` whose ``finally`` restores
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.state import get_run_context, set_run_context
from timbal.state.context import RunContext
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

RESUME_ID = "some-approval-id"


def _tool_call(tool_name: str, input: dict | None = None, *, id: str = "c1") -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=id, name=tool_name, input=input or {})],
        stop_reason="tool_use",
    )


async def _wait_until(condition, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


class TestReadOnly:
    def test_fresh_context_rejects_in_place_writes(self):
        ctx = RunContext()
        assert dict(ctx._resume_values) == {}
        with pytest.raises(TypeError):
            ctx._resume_values[RESUME_ID] = True
        with pytest.raises(AttributeError):
            ctx._resume_values.update({RESUME_ID: True})

    def test_contexts_do_not_share_writable_state(self):
        # The empty mapping is one shared object; that is only safe because it cannot be written.
        assert RunContext()._resume_values is RunContext()._resume_values

    async def test_installed_values_reject_in_place_writes(self):
        seen: dict = {}

        def handler() -> str:
            ctx = get_run_context()
            seen["ctx"] = ctx
            seen["during"] = dict(ctx._resume_values)
            with pytest.raises(TypeError):
                ctx._resume_values["another-id"] = True
            return "ok"

        result = await Tool(name="reader", handler=handler)(resume={RESUME_ID: True}).collect()

        assert result.status.code == "success", result.error
        assert RESUME_ID in seen["during"]
        assert dict(seen["ctx"]._resume_values) == {}


class TestBackgroundChild:
    async def test_child_outliving_the_run_does_not_put_resume_values_back(self):
        """A background sub-agent's inner call starts under the root's resume values and ends after it."""
        seen: dict = {}
        slow_started = asyncio.Event()
        release_slow = asyncio.Event()

        async def slow() -> str:
            seen["slow_saw"] = dict(get_run_context()._resume_values)
            slow_started.set()
            await release_slow.wait()
            return "slow done"

        async def wait_for_slow() -> str:
            seen["ctx"] = get_run_context()
            await slow_started.wait()
            return "waited"

        sub = Agent(
            name="sub",
            description="Works in the background.",
            model=TestModel(responses=[_tool_call("slow", id="s1"), "sub done"]),
            tools=[slow],
            background_mode="always",
        )
        root = Agent(
            name="root",
            model=TestModel(
                responses=[
                    _tool_call("sub", {"prompt": "work"}, id="r1"),
                    _tool_call("wait_for_slow", id="r2"),
                    "root done",
                ]
            ),
            tools=[sub, wait_for_slow],
        )

        result = await root(prompt="go", resume={RESUME_ID: True}).collect()
        ctx = seen["ctx"]

        assert result.status.code == "success", result.error
        assert ctx.id == result.run_id
        # The child's call did start while the values were installed, and is still running.
        assert RESUME_ID in seen["slow_saw"]
        assert ctx._live_background == 1
        assert dict(ctx._resume_values) == {}

        release_slow.set()
        await _wait_until(lambda: ctx._live_background == 0)

        assert dict(ctx._resume_values) == {}


class TestFailedSetup:
    async def test_call_failing_before_it_starts_leaves_nothing_on_the_context(self):
        ctx = RunContext()
        agent = Agent(name="agent", model=TestModel(responses=["hi"]), tools=[])

        async def run() -> None:
            # An empty caller-provided context is kept by ``_stream``, so what it leaves is visible here.
            # In a task of its own: the ambient ids the failed call set end with it.
            set_run_context(ctx)
            await agent(prompt="go", resume={RESUME_ID: True}).collect()

        with patch("timbal.core.runnable.Span", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                await asyncio.create_task(run())

        assert dict(ctx._resume_values) == {}
        assert getattr(ctx, "_background_handoff_control", None) is None
