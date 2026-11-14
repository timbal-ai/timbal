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
import collections
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
from timbal.types.content import ToolResultContent, ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
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


class FakePlatform:
    """The platform's runs API as monolith serves it (``apps/runs/edit.rs``, ``single.rs``).

    Runs get a numeric id; the SDK's run id is the idempotency key. Either one looks a run up.
    PATCH replaces the stored trace; creating a run whose ``parent_id`` matches no run is 403.
    GET returns ``parent_id`` as the parent's numeric id, serialized as a string.
    """

    def __init__(self) -> None:
        self.runs: dict[int, dict] = {}
        self.reads: list[str] = []

    def find(self, run_id: str | None) -> dict | None:
        if run_id is None:
            return None
        if run_id.isdigit():
            return self.runs.get(int(run_id))
        return next((run for run in self.runs.values() if run["idempotency_key"] == run_id), None)

    async def request(self, method, path, content=None, **_):
        run_id = path.rsplit("/", 1)[-1]
        run = self.find(run_id)
        res = MagicMock()
        res.raise_for_status = MagicMock()
        if method == "PATCH":
            body = json.loads(content)
            if run is None:
                parent = self.find(body.get("parent_id"))
                if body.get("parent_id") is not None and parent is None:
                    raise RuntimeError(f"403 {path}: parent not found")
                run = {"id": len(self.runs) + 1, "idempotency_key": run_id, "parent_pk": parent and parent["id"]}
                self.runs[run["id"]] = run
            run["trace"] = body["trace"]
            return res
        self.reads.append(run_id)
        if run is None:
            raise RuntimeError(f"404 {path}")
        parent_pk = run["parent_pk"]
        res.json = MagicMock(
            return_value={
                "id": str(run["id"]),
                "parent_id": None if parent_pk is None else str(parent_pk),
                "idempotency_key": run["idempotency_key"],
                "trace": run["trace"],
            }
        )
        return res


@pytest.fixture(autouse=True)
def platform():
    fake = FakePlatform()
    with patch("timbal.platform.utils._request", fake.request):
        yield fake


@pytest.fixture
def provider(request, tmp_path, monkeypatch):
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
        self.gates = {
            name: Gate() for name in ("session_load", "system_prompt", "parent_load", "model_call", "after_reply")
        }
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

        def charge_card(amount: int) -> str:
            return f"charged {amount}"

        load_session = RunContext.get_session

        async def gated_session(ctx, *args, **kwargs):
            # Before the run has resolved its input: the span holds it only as it was given.
            await self.gates["session_load"].hit()
            return await load_session(ctx, *args, **kwargs)

        monkeypatch.setattr(RunContext, "_get_parent_trace", gated_load)
        monkeypatch.setattr(RunContext, "get_session", gated_session)
        self.agent = Agent(
            name="voice_agent",
            model=GatedModel(handler, before=self.gates["model_call"], after=self.gates["after_reply"]),
            system_prompt=system_prompt,
            tools=[Tool(name="charge_card", handler=charge_card, requires_approval=True)],
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
@pytest.mark.parametrize("cut_at", ["session_load", "system_prompt", "parent_load", "model_call"])
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


@pytest.mark.parametrize("provider", ["platform_uncached"], indirect=True)
async def test_the_walk_back_follows_the_platforms_numeric_parent_id(provider, monkeypatch, platform):
    chat = Conversation(provider, monkeypatch, replies=["Nice to meet you, Ada."])
    first = await chat.turn("My name is Ada.")
    cut = await chat.interrupted_turn("I live in Paris.", parent_id=first.run_id, at="system_prompt")

    await chat.turn("What is my name?", parent_id=cut)

    assert chat.last_input() == ["My name is Ada.", "Nice to meet you, Ada.", "I live in Paris.", "What is my name?"]
    assert any(read.isdigit() for read in platform.reads)


@pytest.mark.parametrize("provider", ["in_memory", "jsonl", "sqlite"], indirect=True)
async def test_a_turn_whose_parent_cannot_be_loaded_is_counted(provider, monkeypatch):
    chat = Conversation(provider, monkeypatch)

    result = await chat.turn("What is my name?", parent_id="run-that-was-never-stored")

    assert chat.last_input() == ["What is my name?"]
    assert result.usage["memory:parent_missing"] == 1


@pytest.mark.parametrize("provider", ["platform", "platform_uncached"], indirect=True)
async def test_a_turn_the_platform_refuses_to_store_still_answers(provider, monkeypatch, platform):
    """The platform answers 403 to a new run whose parent it has no row for. The turn still
    answers; only its trace is lost, which is logged and counted."""
    chat = Conversation(provider, monkeypatch, replies=["Hello, who is calling?"])

    result = await chat.turn("Hi there.", parent_id="run-that-was-never-stored")

    assert result.status.code == "success"
    assert result.output.collect_text() == "Hello, who is calling?"
    assert result.usage["memory:parent_missing"] == 1
    assert result.usage["tracing:save_failed"] >= 1
    assert platform.runs == {}


@pytest.mark.parametrize("provider", ["in_memory"], indirect=True)
async def test_a_chain_broken_behind_an_interrupted_turn_is_counted(provider, monkeypatch):
    chat = Conversation(provider, monkeypatch, replies=["Nice to meet you, Ada."])
    first = await chat.turn("My name is Ada.")
    cut = await chat.interrupted_turn("I live in Paris.", parent_id=first.run_id, at="system_prompt")
    del InMemoryTracingProvider._storage[first.run_id]

    result = await chat.turn("What is my name?", parent_id=cut)

    assert chat.last_input() == ["I live in Paris.", "What is my name?"]
    assert result.usage["memory:parent_missing"] == 1


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


@pytest.mark.parametrize("recorded", ["one result", "no result"])
@pytest.mark.parametrize("provider", ["in_memory", "jsonl", "sqlite"], indirect=True)
async def test_calls_of_a_cut_batch_that_have_no_result_are_answered_not_run_again(provider, recorded):
    """A batch resumed after an approval records each result as it comes. Cut while one of its calls is still
    running, the turn leaves a tool_use message answered in part (or not at all, when the cut comes before
    the first result). The next turn must not take the rest for calls waiting to be run: the ones that were
    running would run a second time."""
    ran: collections.Counter = collections.Counter()
    entered = {name: asyncio.Event() for name in ("quick", "slow", "slower")}
    quick_recorded = asyncio.Event()
    inputs: list[list[Message]] = []

    async def quick() -> str:
        ran["quick"] += 1
        entered["quick"].set()
        if recorded == "no result":
            await asyncio.Event().wait()
        return "quick done"

    async def slow() -> str:
        ran["slow"] += 1
        entered["slow"].set()
        await asyncio.Event().wait()
        return "never"

    async def slower() -> str:
        ran["slower"] += 1
        entered["slower"].set()
        await asyncio.Event().wait()
        return "never"

    def handler(messages):
        inputs.append(list(messages))
        if len(inputs) == 1:
            return Message(
                role="assistant",
                content=[
                    ToolUseContent(id="call_quick", name="quick", input={}),
                    ToolUseContent(id="call_slow", name="slow", input={}),
                    ToolUseContent(id="call_slower", name="slower", input={}),
                ],
                stop_reason="tool_use",
            )
        return "ok"

    agent = Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[
            Tool(name="quick", handler=quick, requires_approval=True),
            Tool(name="slow", handler=slow, requires_approval=True),
            Tool(name="slower", handler=slower, requires_approval=True),
        ],
        tracing_provider=provider,
    )
    Conversation._fresh_task_context()
    events = [event async for event in agent(prompt="Do all three.")]
    paused = events[-1]
    assert paused.status.reason == "approval_required"
    approvals = {event.approval_id: True for event in events if isinstance(event, ApprovalEvent)}
    assert len(approvals) == 3

    Conversation._fresh_task_context()
    cut_run = None

    async def resumed() -> None:
        nonlocal cut_run
        async for event in agent(prompt="Do all three.", parent_id=paused.run_id, resume=approvals):
            cut_run = cut_run or event.run_id
            if isinstance(event, OutputEvent) and event.path == "agent.quick":
                quick_recorded.set()  # the agent records a result before it passes the event on

    # The calls start in no fixed order (each claims its approval first, through the provider), and how
    # long a result takes to be recorded depends on the machine: wait for the state itself, not for ticks.
    task = asyncio.create_task(resumed())
    state = [*entered.values()] + ([quick_recorded] if recorded == "one result" else [])
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in state)), timeout=30)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert dict(ran) == {"quick": 1, "slow": 1, "slower": 1}

    Conversation._fresh_task_context()
    after = await agent(prompt="What happened?", parent_id=cut_run).collect()

    assert after.status.code == "success"
    assert dict(ran) == {"quick": 1, "slow": 1, "slower": 1}  # nothing ran again
    sent = inputs[-1]
    results = {c.id: c for m in sent for c in m.content if isinstance(c, ToolResultContent)}
    assert set(results) == {"call_quick", "call_slow", "call_slower"}
    # One result per tool message, right after the calls: Chat Completions sends one result per message
    # (a second one in the same message is dropped), and every provider wants each call answered.
    assert [m.role for m in sent[-5:]] == ["assistant", "tool", "tool", "tool", "user"]
    assert all(len(m.content) == 1 for m in sent if m.role == "tool")
    as_chat_completions = [m.to_openai_chat_completions_input() for m in sent]
    assert {m["tool_call_id"] for m in as_chat_completions if m["role"] == "tool"} == set(results)
    if recorded == "one result":
        assert "quick done" in str(results["call_quick"].content)
        assert "[Interrupted]" in str(results["call_slow"].content)
        assert "[Interrupted]" in str(results["call_slower"].content)
    else:
        assert "quick done" not in str(results["call_quick"].content)  # cut too: no result of its own
    assert [m.role for m in sent][-1] == "user" and sent[-1].collect_text() == "What happened?"
