"""Loop controls: per-run tool policy, silent turns, the denial circuit breaker, the stream idle watchdog."""

import asyncio
import time

import pytest
from timbal import Agent, Tool
from timbal.core.agent import DEFAULT_DENIAL_NOTICE
from timbal.core.llm import router as router_module
from timbal.core.llm.retry import _retry_on_error
from timbal.core.test_model import TestModel
from timbal.errors import ModelStreamStalled
from timbal.guardrails import Verdict, guardrail
from timbal.state import get_run_context, set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message


@pytest.fixture(autouse=True)
def isolated_context():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


def call(name, call_id="c1", **input):
    return Message(
        role="assistant", content=[ToolUseContent(id=call_id, name=name, input=input)], stop_reason="tool_use"
    )


def context(**attrs):
    ctx = RunContext(tracing_provider=InMemoryTracingProvider)
    for name, value in attrs.items():
        setattr(ctx, f"_{name}", value)
    set_run_context(ctx)
    return ctx


def tool_results(output_messages):
    return [m for m in output_messages if m.role == "tool"]


async def memory_of(output):
    trace = await RunContext(tracing_provider=InMemoryTracingProvider)._get_parent_trace(output.run_id)
    return [Message.validate(message) for message in trace[output.call_id].memory]


# -- Tool policy ----------------------------------------------------------------


def policy_agent(model=None, sent=None):
    sent = sent if sent is not None else []

    def send_email(to: str) -> str:
        sent.append(to)
        return "sent"

    def lookup(q: str) -> str:
        return f"found {q}"

    return Agent(
        name="agent",
        model=model or TestModel(responses=["ok"]),
        tools=[Tool(name="lookup", handler=lookup, read_only=True), Tool(name="send_email", handler=send_email)],
        tracing_provider=InMemoryTracingProvider,
    )


class TestToolPolicy:
    async def test_no_policy_offers_everything(self):
        context()
        tools, _ = await policy_agent()._resolve_tools(0)
        assert {t.name for t in tools} == {"lookup", "send_email"}

    async def test_read_only_keeps_declared_readers(self):
        context(tool_policy="read_only")
        tools, _ = await policy_agent()._resolve_tools(0)
        assert [t.name for t in tools] == ["lookup"]

    async def test_names_and_callables(self):
        context(tool_policy=frozenset({"send_email"}))
        tools, _ = await policy_agent()._resolve_tools(0)
        assert [t.name for t in tools] == ["send_email"]
        context(tool_policy=lambda tool: tool.name.startswith("look"))
        tools, _ = await policy_agent()._resolve_tools(0)
        assert [t.name for t in tools] == ["lookup"]

    async def test_builtin_readers_count_as_read_only(self):
        agent = policy_agent()
        assert agent._tool_allowed(Tool(name="read_offloaded", handler=lambda: 1), "read_only")
        assert agent._tool_allowed(Tool(name="get_background_task", handler=lambda: 1), "read_only")
        assert not agent._tool_allowed(Tool(name="cancel_background_task", handler=lambda: 1), "read_only")

    async def test_a_hidden_tool_cannot_be_called(self):
        """Enforced in code: the model calling a tool the policy hides gets 'not found', and nothing runs."""
        sent = []

        def handler(messages):
            if messages[-1].role == "tool":
                return "gave up"
            return call("send_email", to="x@y.z")

        agent = policy_agent(TestModel(handler=handler), sent)
        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        ctx._tool_policy = "read_only"
        set_run_context(ctx)
        output = await agent(prompt="email them").collect()
        assert sent == []
        (result,) = tool_results(await memory_of(output))
        assert "Tool 'send_email' not found" in result.content[0].content[0].text

    async def test_policy_applies_to_subagents_in_the_run(self):
        seen = []

        def write() -> str:
            return "written"

        inner = Agent(
            name="inner",
            model=TestModel(responses=["inner done"]),
            tools=[Tool(name="write", handler=write)],
        )

        async def probe() -> str:
            tools, _ = await inner._resolve_tools(0)
            seen.extend(t.name for t in tools)
            return "probed"

        outer = Agent(
            name="outer",
            model=TestModel(handler=lambda m: "done" if m[-1].role == "tool" else call("probe")),
            tools=[Tool(name="probe", handler=probe, read_only=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        ctx._tool_policy = "read_only"
        set_run_context(ctx)
        await outer(prompt="go").collect()
        assert seen == []


# -- Silent turns -----------------------------------------------------------------


class TestSilence:
    @pytest.mark.parametrize(
        ("token", "reply", "silent"),
        [
            ("NO_REPLY", "NO_REPLY", True),
            ("NO_REPLY", "  NO_REPLY \n", True),
            ("NO_REPLY", "Hello", False),
            (None, "NO_REPLY", False),
        ],
    )
    async def test_exact_token_marks_the_turn(self, token, reply, silent):
        agent = Agent(name="agent", model=TestModel(responses=[reply]), tracing_provider=InMemoryTracingProvider)
        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        ctx._silence_token = token
        set_run_context(ctx)
        output = await agent(prompt="anything new?").collect()
        assert bool(output.metadata.get("silent")) is silent

    async def test_only_the_final_answer_counts(self):
        """A tool-calling step never ends a turn, so only the last message can make it silent."""
        responses = [call("lookup", q="x"), "Found it."]
        agent = policy_agent(TestModel(responses=responses))
        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        ctx._silence_token = "Found it."
        set_run_context(ctx)
        output = await agent(prompt="go").collect()
        assert output.metadata.get("silent") is True


# -- Denial circuit breaker ----------------------------------------------------------


def blocking_agent(*, limit=3, allow=lambda _n: False, final="I need your help."):
    calls = {"tool": 0, "model": []}

    def act(n: int) -> str:
        calls["tool"] += 1
        return f"did {n}"

    rail = guardrail(
        lambda text: True if allow(int(text.split('"n": ')[1].rstrip("}") or 0)) else Verdict.block("nope"),
        stages=["tool_args"],
    )

    def handler(messages):
        calls["model"].append(messages)
        if messages[-1].is_runtime() and messages[-1].metadata.get("kind") == "denial_limit":
            return final
        attempts = sum(1 for m in messages if m.role == "assistant")
        if attempts >= 8:
            return "giving up"
        return call("act", call_id=f"c{attempts}", n=attempts)

    agent = Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[Tool(name="act", handler=act, guardrails=[rail])],
        max_iter=12,
        max_consecutive_denials=limit,
        tracing_provider=InMemoryTracingProvider,
    )
    return agent, calls


class TestDenialBreaker:
    async def test_stops_after_the_limit_and_answers_without_tools(self):
        agent, calls = blocking_agent()
        output = await agent(prompt="do it").collect()
        assert output.status.code == "success"
        assert output.output.collect_text() == "I need your help."
        assert output.metadata["denial_limit"] == [{"denials": 3}]
        assert calls["tool"] == 0
        # Three blocked attempts, then the notice, then one tool-less answer.
        assert len(calls["model"]) == 4
        notice = calls["model"][-1][-1]
        assert notice.collect_text() == DEFAULT_DENIAL_NOTICE.format(count=3)
        llm_spans = [s for s in get_run_context()._trace.values() if s.path == "agent.llm"]
        assert not llm_spans[-1].input.get("tools")
        assert llm_spans[0].input.get("tools")

    async def test_a_success_resets_the_count(self):
        agent, calls = blocking_agent(allow=lambda n: n == 2)
        output = await agent(prompt="do it").collect()
        assert calls["tool"] == 1  # attempt 2 went through
        assert output.metadata["denial_limit"] == [{"denials": 3}]
        # 0, 1 blocked; 2 ran; 3, 4, 5 blocked -> limit after attempt 5.
        assert len(calls["model"]) == 7

    async def test_denied_approvals_count_across_resumes(self):
        """Each denied approval ends its run; the streak goes on in the run that resumes it."""

        def charge(n: int) -> str:
            return f"charged {n}"

        def handler(messages):
            last = messages[-1]
            if last.is_runtime() and last.metadata.get("kind") == "denial_limit":
                return "I will stop and ask how to proceed."
            attempts = sum(1 for m in messages if m.role == "assistant")
            return call("charge", call_id=f"c{attempts}", n=attempts)

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=charge, requires_approval=True)],
            max_iter=20,
            tracing_provider=InMemoryTracingProvider,
        )
        events = [e async for e in agent(prompt="pay")]
        denied = 0
        while denied < 5:
            denied += 1
            approval = next(e for e in events if isinstance(e, ApprovalEvent))
            run_id = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == "agent").run_id
            events = [e async for e in agent(prompt="pay", parent_id=run_id, resume={approval.approval_id: False})]
            final = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == "agent")
            if not any(isinstance(e, ApprovalEvent) for e in events):
                break
        assert denied == 3
        assert final.status.code == "success"
        assert final.metadata["denial_limit"] == [{"denials": 3}]
        assert final.output.collect_text() == "I will stop and ask how to proceed."

    async def test_a_new_turn_starts_a_new_streak(self):
        """A streak a paused run left applies to its resume only, not to a turn chained after another one."""

        def charge(n: int) -> str:
            return f"charged {n}"

        def handler(messages):
            last = messages[-1]
            if last.role == "user" and not last.is_runtime() and last.collect_text() == "hello":
                return "hi"
            if last.is_runtime() and last.metadata.get("kind") == "denial_limit":
                return "stopping"
            attempts = sum(1 for m in messages if m.role == "assistant")
            return call("charge", call_id=f"c{attempts}", n=attempts)

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=charge, requires_approval=True)],
            max_iter=20,
            tracing_provider=InMemoryTracingProvider,
        )

        def root(events):
            return next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == "agent")

        events = [e async for e in agent(prompt="pay")]
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        events = [
            e async for e in agent(prompt="pay", parent_id=root(events).run_id, resume={approval.approval_id: False})
        ]
        paused = root(events)  # one denial so far, paused on the next approval
        chat = await agent(prompt="hello", parent_id=paused.run_id).collect()
        events = [e async for e in agent(prompt="pay", parent_id=chat.run_id)]
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        events = [
            e async for e in agent(prompt="pay", parent_id=root(events).run_id, resume={approval.approval_id: False})
        ]
        assert "denial_limit" not in root(events).metadata  # one denial in this streak, not two
        assert any(isinstance(e, ApprovalEvent) for e in events)

    async def test_none_turns_it_off(self):
        agent, calls = blocking_agent(limit=None)
        output = await agent(prompt="do it").collect()
        assert "denial_limit" not in output.metadata
        assert output.output.collect_text() == "giving up"


# -- Stream idle watchdog ----------------------------------------------------------------


def scripted_stream(*delays, record=None):
    """A stream factory: each call yields chunks after the given delays (a list per attempt)."""
    attempts = iter(delays)
    record = record if record is not None else []

    def factory():
        plan = next(attempts)

        async def gen():
            record.append("start")
            try:
                for i, delay in enumerate(plan):
                    await asyncio.sleep(delay)
                    yield f"chunk{i}"
            finally:
                record.append("closed")

        return gen()

    return factory, record


async def collect(gen):
    return [item async for item in gen]


class TestIdleWatchdog:
    async def test_stall_before_output_is_retried(self):
        factory, record = scripted_stream([1.0], [0.0, 0.0])
        chunks = await collect(_retry_on_error(factory, 1, 0.0, "test", idle_timeout=0.05))
        assert chunks == ["chunk0", "chunk1"]
        assert record == ["start", "closed", "start", "closed"]

    async def test_stall_before_output_exhausts_retries(self):
        factory, _ = scripted_stream([1.0], [1.0])
        with pytest.raises(ModelStreamStalled) as raised:
            await collect(_retry_on_error(factory, 1, 0.0, "test", idle_timeout=0.05))
        assert raised.value.after_output is False and isinstance(raised.value, TimeoutError)

    async def test_stall_after_output_fails_without_retrying(self):
        factory, record = scripted_stream([0.0, 1.0], [0.0])
        got = []
        with pytest.raises(ModelStreamStalled) as raised:
            async for chunk in _retry_on_error(factory, 3, 0.0, "test", idle_timeout=0.05):
                got.append(chunk)
        assert raised.value.after_output is True
        assert got == ["chunk0"] and record == ["start", "closed"]

    async def test_slow_but_steady_streams_pass(self):
        """The timeout is per gap between chunks, not for the whole stream: 5 gaps of 0.1 s pass a 0.4 s
        timeout though the stream takes longer than it. 4x apart, so a coarse clock (15.6 ms ticks on
        Windows before 3.13) or a loaded runner cannot turn a gap into a stall."""
        factory, _ = scripted_stream([0.1] * 5)
        started = time.monotonic()
        chunks = await collect(_retry_on_error(factory, 0, 0.0, "test", idle_timeout=0.4))
        assert len(chunks) == 5
        assert time.monotonic() - started > 0.4

    async def test_no_timeout_waits(self):
        factory, _ = scripted_stream([0.1])
        assert await collect(_retry_on_error(factory, 0, 0.0, "test", idle_timeout=None)) == ["chunk0"]

    def test_a_stall_before_output_is_transient_for_fallback(self):
        from timbal.core.fallback_model import is_retryable_provider_error

        assert is_retryable_provider_error(ModelStreamStalled(1.0)) is True
        assert is_retryable_provider_error(ModelStreamStalled(1.0, after_output=True)) is False

    @pytest.mark.parametrize("after_output", [False, True])
    async def test_transient_only_fallback_moves_on_after_a_stall_before_output(self, after_output):
        from timbal.core.fallback_model import FallbackModel, is_retryable_provider_error

        model = FallbackModel("openai/primary", "openai/backup", fallback_on=is_retryable_provider_error)
        seen = []

        async def router(**kwargs):
            seen.append(kwargs["model"])
            if kwargs["model"] == "openai/primary":
                factory, _ = scripted_stream([0.0, 1.0] if after_output else [1.0])
                async for chunk in _retry_on_error(factory, 0, 0.0, "test", idle_timeout=0.05):
                    yield chunk
            else:
                yield "backup"

        if after_output:
            got = []
            with pytest.raises(ModelStreamStalled):
                async for chunk in model.route(router):
                    got.append(chunk)
            assert seen == ["openai/primary"] and got == ["chunk0"]
        else:
            assert [chunk async for chunk in model.route(router)] == ["backup"]
            assert seen == ["openai/primary", "openai/backup"]

    async def test_router_passes_it_through(self, monkeypatch):
        factory, _ = scripted_stream([1.0])
        monkeypatch.setattr(router_module, "prepare_messages_request", lambda **_: (factory, "fake"))
        monkeypatch.setattr(router_module, "_resolve_client", lambda *_a, **_k: (None, None))
        context()
        with pytest.raises(ModelStreamStalled):
            await collect(
                router_module._llm_router(
                    model="anthropic/claude-test", max_tokens=10, messages=[], stream_idle_timeout=0.05
                )
            )

    def test_agent_default_and_opt_out(self):
        agent = Agent(name="a", model="openai/gpt-4o-mini")
        assert agent._llm.default_params["stream_idle_timeout"] == 300.0
        agent = Agent(name="a", model="openai/gpt-4o-mini", stream_idle_timeout=None)
        assert "stream_idle_timeout" not in agent._llm.default_params
