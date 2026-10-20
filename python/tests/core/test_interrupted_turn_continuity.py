"""A turn interrupted before it assembled its memory must not cost the next turn its history.

The voice case: a completed turn, then a turn cancelled by a barge-in while it was still
resolving its system prompt or loading the previous turn, then a third turn. That second turn
stores no memory. Whatever the cancellation hit, the third turn continues from the same
conversation it would have had if the second turn had been cut right after assembling its
memory: the earlier turns, then the interrupted prompt. A turn cut later keeps what it reached.

Every case runs on each provider, since the in-memory one reads live spans and the others read
what was serialized, and on the platform provider with and without its recent-trace cache.
"""

import asyncio
import json
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.core.tool import Tool
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing.providers import (
    InMemoryTracingProvider,
    JsonlTracingProvider,
    PlatformTracingProvider,
    SqliteTracingProvider,
)
from timbal.state.tracing.providers.platform import _parse_cache_limits
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

CONFIG = PlatformConfig(
    host="api.example.test",
    auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_test"),
    subject=PlatformSubject(org_id="1", app_id="2"),
)

PROVIDERS = ["in_memory", "jsonl", "sqlite", "platform", "platform_uncached"]


class Gate:
    """Parks the first call made while armed, until the test cancels it."""

    def __init__(self) -> None:
        self.armed = False
        self.entered = asyncio.Event()

    async def hit(self) -> None:
        if self.armed:
            self.armed = False
            self.entered.set()
            await asyncio.Event().wait()


class GatedModel(TestModel):
    """TestModel that can park before producing its reply, or after streaming it."""

    __test__ = False

    def __init__(self, handler, before: Gate, after: Gate) -> None:
        super().__init__(handler=handler)
        self.before = before
        self.after = after

    async def stream(self, messages, **kwargs):
        await self.before.hit()
        async for item in super().stream(messages, **kwargs):
            yield item
        await self.after.hit()


@pytest.fixture
def platform():
    """The platform's run store, behind the provider's HTTP calls."""
    runs: dict[str, dict] = {}

    async def request(method, path, headers=None, json=None, content=None, **_):
        run_id = path.rsplit("/", 1)[-1]
        res = MagicMock()
        res.raise_for_status = MagicMock()
        if method == "PATCH":
            runs[run_id] = _json.loads(content)
        elif run_id in runs:
            res.json = MagicMock(return_value=runs[run_id])
        else:
            raise RuntimeError(f"404 {path}")
        return res

    _json = json
    with patch("timbal.platform.utils._request", request):
        yield runs


@pytest.fixture
def provider(request, tmp_path, platform, monkeypatch):
    monkeypatch.delenv("TIMBAL_TRACE_CACHE_MAX_ENTRIES", raising=False)
    _parse_cache_limits.cache_clear()
    kind = request.param
    if kind == "in_memory":
        return InMemoryTracingProvider
    if kind == "jsonl":
        return JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
    if kind == "sqlite":
        return SqliteTracingProvider.configured(_path=tmp_path / "traces.db")
    if kind == "platform_uncached":
        monkeypatch.setenv("TIMBAL_TRACE_CACHE_MAX_ENTRIES", "0")
    return PlatformTracingProvider.configured(platform_config=CONFIG)


class Conversation:
    """One agent, turns chained by run id, and a gate at each point a turn can be cut."""

    def __init__(self, provider, monkeypatch, replies=None) -> None:
        self.gates = {name: Gate() for name in ("system_prompt", "parent_load", "model_call", "after_reply")}
        self.inputs: list[list[Message]] = []
        replies = iter(replies or [])

        async def system_prompt() -> str:
            await self.gates["system_prompt"].hit()
            return "Be brief."

        def handler(messages):
            self.inputs.append(list(messages))
            return next(replies, f"reply {len(self.inputs)}")

        load_parent = RunContext._get_parent_trace

        async def gated_load(ctx, *args):
            await self.gates["parent_load"].hit()
            return await load_parent(ctx, *args)

        monkeypatch.setattr(RunContext, "_get_parent_trace", gated_load)
        self.agent = Agent(
            name="voice_agent",
            model=GatedModel(handler, before=self.gates["model_call"], after=self.gates["after_reply"]),
            system_prompt=system_prompt,
            tools=[Tool(name="charge_card", handler=lambda amount: "charged", requires_approval=True)],
            tracing_provider=provider,
        )

    @staticmethod
    def _fresh_task_context() -> None:
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

    async def turn(self, prompt: str, parent_id: str | None = None):
        self._fresh_task_context()
        return await self.agent(prompt=prompt, parent_id=parent_id).collect()

    async def interrupted_turn(self, prompt: str, parent_id: str | None, at: str) -> str:
        """Run a turn and cancel it once it reaches ``at``, like a barge-in cancels a voice turn."""
        self._fresh_task_context()
        gate = self.gates[at]
        gate.armed = True
        gate.entered.clear()
        run_id = None

        async def consume() -> None:
            nonlocal run_id
            async for event in self.agent(prompt=prompt, parent_id=parent_id):
                run_id = run_id or event.run_id

        task = asyncio.create_task(consume())
        await asyncio.wait_for(gate.entered.wait(), timeout=5)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        assert run_id is not None
        return run_id

    def last_input(self) -> list[str]:
        return [m.collect_text() for m in self.inputs[-1]]


@pytest.mark.parametrize("provider", PROVIDERS, indirect=True)
@pytest.mark.parametrize("cut_at", ["system_prompt", "parent_load", "model_call"])
async def test_where_the_cut_lands_does_not_change_the_next_turn(provider, monkeypatch, cut_at):
    chat = Conversation(provider, monkeypatch, replies=["Nice to meet you, Ada."])
    first = await chat.turn("My name is Ada.")
    cut = await chat.interrupted_turn("I live in Paris.", parent_id=first.run_id, at=cut_at)

    await chat.turn("What is my name?", parent_id=cut)

    assert chat.last_input() == ["My name is Ada.", "Nice to meet you, Ada.", "I live in Paris.", "What is my name?"]


@pytest.mark.parametrize("provider", PROVIDERS, indirect=True)
async def test_a_reply_cut_short_stays_in_the_conversation(provider, monkeypatch):
    chat = Conversation(provider, monkeypatch, replies=["Nice to meet you, Ada.", "Paris is lovely"])
    first = await chat.turn("My name is Ada.")
    cut = await chat.interrupted_turn("I live in Paris.", parent_id=first.run_id, at="after_reply")

    await chat.turn("What is my name?", parent_id=cut)

    assert chat.last_input() == [
        "My name is Ada.",
        "Nice to meet you, Ada.",
        "I live in Paris.",
        "Paris is lovely",
        "What is my name?",
    ]


@pytest.mark.parametrize("provider", PROVIDERS, indirect=True)
async def test_consecutive_interrupted_turns_are_all_kept(provider, monkeypatch):
    chat = Conversation(provider, monkeypatch, replies=["Nice to meet you, Ada."])
    first = await chat.turn("My name is Ada.")
    cut = await chat.interrupted_turn("I live in Paris.", parent_id=first.run_id, at="parent_load")
    cut_again = await chat.interrupted_turn("Well, near Paris.", parent_id=cut, at="system_prompt")

    await chat.turn("What is my name?", parent_id=cut_again)

    assert chat.last_input() == [
        "My name is Ada.",
        "Nice to meet you, Ada.",
        "I live in Paris.",
        "Well, near Paris.",
        "What is my name?",
    ]


@pytest.mark.parametrize("provider", ["in_memory"], indirect=True)
async def test_an_interrupted_first_turn_is_kept(provider, monkeypatch):
    chat = Conversation(provider, monkeypatch)
    cut = await chat.interrupted_turn("My name is Ada.", parent_id=None, at="system_prompt")

    await chat.turn("What is my name?", parent_id=cut)

    assert chat.last_input() == ["My name is Ada.", "What is my name?"]


@pytest.mark.parametrize("provider", ["in_memory", "platform"], indirect=True)
async def test_a_turn_cut_while_resuming_a_paused_turn_is_rebuilt_as_that_resume(provider, monkeypatch):
    """The cut turn would have resumed the approval its parent paused on, adding no prompt of its own.
    The next turn then finds that call unanswered, and closes it with an error result."""
    charge = Message(
        role="assistant",
        content=[ToolUseContent(id="call_1", name="charge_card", input={"amount": 20})],
        stop_reason="tool_use",
    )
    chat = Conversation(provider, monkeypatch, replies=[charge])
    paused = await chat.turn("Charge my card 20 euros.")
    assert (paused.status.code, paused.status.reason) == ("cancelled", "approval_required")
    cut = await chat.interrupted_turn("Yes, go ahead.", parent_id=paused.run_id, at="system_prompt")

    await chat.turn("Actually, don't.", parent_id=cut)

    sent = chat.inputs[-1]
    assert [m.role for m in sent] == ["user", "assistant", "tool", "user"]
    assert sent[0].collect_text() == "Charge my card 20 euros."
    assert sent[2].content[0].id == "call_1"
    assert sent[-1].collect_text() == "Actually, don't."
