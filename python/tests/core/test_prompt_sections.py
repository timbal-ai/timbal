"""Pinned system-prompt sections: the prompt bytes stay put; changes arrive as notes before the user message."""

import re

from timbal import Agent
from timbal.core.agent import PROMPT_UPDATE_KIND
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


def fresh():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


async def trace_of(out):
    return await RunContext(tracing_provider=InMemoryTracingProvider)._get_parent_trace(out.run_id)


async def sent_prompts(out) -> list[str | None]:
    trace = await trace_of(out)
    spans = sorted((s for s in trace.values() if s.path.endswith(".llm")), key=lambda s: s.t0)
    return [s.input.get("system_prompt") for s in spans]


def notes(messages) -> list[str]:
    return [m.collect_text() for m in messages if m.metadata.get("kind") == PROMPT_UPDATE_KIND]


class Sections:
    def __init__(self, **sections):
        self.sections = dict(sections)

    def __call__(self) -> dict[str, str]:
        return dict(self.sections)


def make_agent(prompt, seen, **kwargs) -> Agent:
    def handler(messages):
        seen.append(list(messages))
        return "ok"

    return Agent(
        name="bot",
        model=TestModel(handler=handler),
        system_prompt=prompt,
        tracing_provider=InMemoryTracingProvider,
        **kwargs,
    )


async def turn(agent, prompt, parent=None):
    out = await agent(prompt=prompt, parent_id=parent.run_id if parent else None).collect()
    assert out.status.code == "success", out.error
    return out


class TestPinnedSections:
    async def test_a_changed_section_is_a_note_and_the_prompt_stays_the_same(self):
        fresh()
        seen = []
        sections = Sections(role="You are a helpful bot.", today="Today is Monday.")
        agent = make_agent(sections, seen)
        first = await turn(agent, "one")
        sections.sections["today"] = "Today is Tuesday."
        second = await turn(agent, "two", first)
        third = await turn(agent, "three", second)

        prompts = [*(await sent_prompts(first)), *(await sent_prompts(second)), *(await sent_prompts(third))]
        assert len(set(prompts)) == 1  # identical bytes every turn
        assert '<section name="role">\nYou are a helpful bot.\n</section>' in prompts[0]
        assert '<section name="today">\nToday is Monday.\n</section>' in prompts[0]
        key = re.search(r'key="([0-9a-f]+)"', prompts[0]).group(1)
        assert notes(seen[0]) == []
        (note,) = notes(seen[1])
        assert f'<system_update section="today" key="{key}">\nToday is Tuesday.\n</system_update>' in note
        assert seen[1][-1].collect_text() == "two" and seen[1][-2].metadata["kind"] == PROMPT_UPDATE_KIND
        assert len(notes(seen[2])) == 1  # the note stays in history, and no new one was added

    async def test_a_removed_and_a_new_section(self):
        fresh()
        seen = []
        sections = Sections(role="Bot.", tasks="Tasks: none.")
        agent = make_agent(sections, seen)
        first = await turn(agent, "one")
        sections.sections = {"role": "Bot.", "mood": "Be upbeat."}
        await turn(agent, "two", first)
        (note,) = notes(seen[1])
        assert re.search(r'<system_update section="tasks" key="[0-9a-f]+" removed="true"/>', note)
        assert re.search(r'<system_update section="mood" key="[0-9a-f]+">\nBe upbeat.\n</system_update>', note)

    async def test_compaction_re_pins(self):
        fresh()
        seen = []
        sections = Sections(role="Bot.", today="Monday.")
        agent = make_agent(sections, seen)
        first = await turn(agent, "one")
        sections.sections["today"] = "Tuesday."
        original = Agent.resolve_memory

        async def compacted(self, *args, **kwargs):
            await original(self, *args, **kwargs)
            from timbal.state import get_run_context

            get_run_context().current_span().metadata["compaction"] = {"strategy": "test"}

        Agent.resolve_memory = compacted
        try:
            second = await turn(agent, "two", first)
        finally:
            Agent.resolve_memory = original
        (prompt,) = await sent_prompts(second)
        assert '<section name="today">\nTuesday.\n</section>' in prompt  # refreshed with the history rewrite
        assert notes(seen[1]) == []
        (old,) = await sent_prompts(first)
        assert re.search(r'key="([0-9a-f]+)"', old).group(1) != re.search(r'key="([0-9a-f]+)"', prompt).group(1)

    async def test_strings_are_unchanged(self):
        fresh()
        seen = []
        day = {"v": "Monday"}
        agent = make_agent(lambda: f"Bot. Today is {day['v']}.", seen)
        first = await turn(agent, "one")
        day["v"] = "Tuesday"
        second = await turn(agent, "two", first)
        assert await sent_prompts(second) == ["Bot. Today is Tuesday."] and notes(seen[1]) == []

    async def test_a_static_dict_and_a_subagent(self):
        fresh()
        inner_seen = []

        def inner(messages):
            inner_seen.append(messages)
            return "inner"

        helper = Agent(name="helper", model=TestModel(handler=inner), system_prompt={"a": "Helper.", "b": "Be terse."})
        call = Message(
            role="assistant",
            content=[ToolUseContent(id="h", name="helper", input={"prompt": "x"})],
            stop_reason="tool_use",
        )
        outer = Agent(
            name="bot",
            model=TestModel(responses=[call, "done"]),
            tools=[helper],
            system_prompt={"role": "Bot."},
            tracing_provider=InMemoryTracingProvider,
        )
        out = await turn(outer, "go")
        trace = await trace_of(out)
        prompts = {s.path: s.input.get("system_prompt") for s in trace.values() if s.path.endswith(".llm")}
        assert prompts["bot.llm"].startswith('<section name="role">\nBot.\n</section>')
        assert prompts["bot.helper.llm"] == "Helper.\n\nBe terse."  # subagents: the sections as they are


class TestWithOtherTurns:
    async def test_a_changed_section_is_not_put_into_a_resumed_tool_batch(self):
        """A resume runs the calls the paused run left: a note between a call and its result is refused by
        the providers. The change is said on the turn after."""
        from timbal import Tool
        from timbal.types.events import ApprovalEvent

        fresh()
        seen = []
        sections = Sections(role="You are a bot.", today="Today is Monday.")

        def handler(messages):
            seen.append(list(messages))
            if not any(m.role == "tool" for m in messages):
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="c1", name="charge", input={})],
                    stop_reason="tool_use",
                )
            return "Paid."

        agent = Agent(
            name="bot",
            model=TestModel(handler=handler),
            system_prompt=sections,
            tools=[Tool(name="charge", handler=lambda: "charged", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
        )
        events = [event async for event in agent(prompt="pay")]
        parked = events[-1]
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        sections.sections["today"] = "Today is Tuesday."
        set_run_context(RunContext(parent_id=parked.run_id, tracing_provider=InMemoryTracingProvider))
        resumed = await agent(prompt="pay", resume={approval.approval_id: True}).collect()
        assert resumed.status.code == "success", resumed.error
        assert [m.role for m in seen[-1]] == ["user", "assistant", "tool"]  # the result right after its call
        assert notes(seen[-1]) == []
        await turn(agent, "and now?", resumed)
        (note,) = notes(seen[-1])
        assert "Today is Tuesday." in note
        assert seen[-1][-1].collect_text() == "and now?" and seen[-1][-2].metadata["kind"] == PROMPT_UPDATE_KIND

    async def test_another_turn_continuing_the_same_run_is_told_too(self):
        """What a turn announced is its own: a second turn from the run it continued (an answer asked for
        again) has not been told."""
        fresh()
        seen = []
        sections = Sections(role="You are a bot.", today="Today is Monday.")
        agent = make_agent(sections, seen)
        first = await turn(agent, "one")
        sections.sections["today"] = "Today is Tuesday."
        await turn(agent, "two", first)
        await turn(agent, "two, asked again", first)
        assert len(notes(seen[1])) == 1
        assert len(notes(seen[2])) == 1
        trace = await trace_of(first)
        root = next(span for span in trace.values() if span.parent_call_id is None)
        assert root.session["__prompt_sections"]["bot"]["pinned"]["today"] == "Today is Monday."  # as it left it


def told(messages, system_prompt: str, name: str, text: str) -> bool:
    """Whether a model call was told that section ``name`` reads ``text``: in the pinned prompt, or in an
    update note that carries the prompt's key and comes after any other note for the section."""
    key = re.search(r'key="([0-9a-f]+)"', system_prompt).group(1)
    current = re.search(rf'<section name="{name}">\n(.*?)\n</section>', system_prompt, re.S)
    current = current.group(1) if current else None
    for note in notes(messages):
        for found in re.finditer(rf'<system_update section="{name}" key="{key}">\n(.*?)\n</system_update>', note, re.S):
            current = found.group(1)
    return current == text


async def llm_calls(out) -> list[tuple[str, list]]:
    """Each model call of a run, in order: its system prompt and the messages it was given."""
    trace = await trace_of(out)
    return [(s.input.get("system_prompt"), s.input.get("messages")) for s in trace.values() if s.path == "bot.llm"]


class TestTheModelIsToldWhateverHappensToTheHistory:
    """The pin is right only together with the update notes in the history the model gets. Wherever the
    history loses them, the model must still be told the current sections."""

    async def test_compaction_inside_a_turn_that_drops_the_note(self, monkeypatch):
        from timbal import Tool
        from timbal.core.memory_compaction import keep_last_n_messages

        # A small window: the fetched payload crosses the ratio, so the turn compacts before its next call.
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 1_000)
        fresh()
        sections = Sections(role="You are a bot.", today="Today is Monday.")

        def handler(messages):
            if messages[-1].role != "tool":
                asked = [m for m in messages if m.role == "user" and not m.is_runtime()]
                if asked[-1].collect_text() == "two":
                    return Message(
                        role="assistant",
                        content=[ToolUseContent(id="f1", name="fetch", input={})],
                        stop_reason="tool_use",
                    )
            return "ok"

        agent = Agent(
            name="bot",
            model=TestModel(handler=handler),
            system_prompt=sections,
            tools=[Tool(name="fetch", handler=lambda: "payload: " + "x" * 4_000)],
            memory_compaction=keep_last_n_messages(1),
            memory_compaction_ratio=0.75,
            tracing_provider=InMemoryTracingProvider,
        )
        first = await turn(agent, "one")
        sections.sections["today"] = "Today is Tuesday."
        second = await turn(agent, "two", first)
        calls_made = await llm_calls(second)
        assert len(calls_made) == 2
        trace = await trace_of(second)
        root = next(span for span in trace.values() if span.parent_call_id is None)
        assert root.metadata["compaction"]["triggered"] is True  # between the two calls
        for prompt, messages in calls_made:
            assert told(messages, prompt, "today", "Today is Tuesday.")
        assert notes(calls_made[0][1]) and not notes(calls_made[1][1])  # the note went with the compaction

        # A turn that does not compact as it starts (which pins again by itself) keeps that pin.
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 1_000_000)
        third = await turn(agent, "three", second)
        ((prompt, messages),) = await llm_calls(third)
        assert told(messages, prompt, "today", "Today is Tuesday.")
        assert prompt == calls_made[1][0]  # the same bytes as the call after the compaction: nothing to repeat

    async def test_compaction_inside_a_turn_that_keeps_the_note_changes_nothing(self, monkeypatch):
        from timbal import Tool
        from timbal.core.memory_compaction import compact_tool_results

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 1_000)
        fresh()
        sections = Sections(role="You are a bot.", today="Today is Monday.")
        asked = []

        def handler(_messages):
            asked.append(1)
            if len(asked) in (2, 3):  # the second turn: two fetches, so a pass has something to drop
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id=f"f{len(asked)}", name="fetch", input={})],
                    stop_reason="tool_use",
                )
            return "ok"

        agent = Agent(
            name="bot",
            model=TestModel(handler=handler),
            system_prompt=sections,
            tools=[Tool(name="fetch", handler=lambda: "payload: " + "x" * 4_000)],
            memory_compaction=compact_tool_results(),
            memory_compaction_ratio=0.75,
            tracing_provider=InMemoryTracingProvider,
        )
        first = await turn(agent, "one")
        sections.sections["today"] = "Today is Tuesday."
        second = await turn(agent, "two", first)
        calls_made = await llm_calls(second)
        trace = await trace_of(second)
        root = next(span for span in trace.values() if span.parent_call_id is None)
        assert root.metadata["compaction"]["triggered"] is True
        assert len({prompt for prompt, _ in calls_made}) == 1  # the pinned bytes did not move
        for prompt, messages in calls_made:
            assert told(messages, prompt, "today", "Today is Tuesday.")
        # Nor on the next turn (one that does not compact as it starts, which pins again by itself).
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 1_000_000)
        third = await turn(agent, "three", second)
        ((prompt, messages),) = await llm_calls(third)
        assert prompt == calls_made[0][0] and told(messages, prompt, "today", "Today is Tuesday.")
        assert len(notes(messages)) == 1  # the one note, not said again

    async def test_compaction_as_a_turn_starts_that_loses_nothing_changes_nothing(self, monkeypatch):
        """A history that is compacted as turns start: the pin moves only when an update went with it."""
        from timbal.core.memory_compaction import keep_last_n_messages

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 200)
        fresh()
        sections = Sections(role="You are a bot.", today="Today is Monday.")
        agent = Agent(
            name="bot",
            model=TestModel(responses=["an answer long enough to fill the small window " * 6]),
            system_prompt=sections,
            memory_compaction=keep_last_n_messages(1),
            memory_compaction_ratio=0.5,
            tracing_provider=InMemoryTracingProvider,
        )
        out, prompts, compacted = None, [], []
        for number in range(6):
            out = await turn(agent, f"turn {number}", out)
            ((prompt, messages),) = await llm_calls(out)
            prompts.append(prompt)
            trace = await trace_of(out)
            root = next(span for span in trace.values() if span.parent_call_id is None)
            compacted.append("compaction" in root.metadata)
            assert told(messages, prompt, "today", "Today is Monday.")
        assert compacted.count(True) >= 2  # several turns compacted as they started
        assert len(set(prompts)) == 1  # and the pinned bytes never moved

        # An update is told on every turn after it, whatever the compaction takes, and pinned at most once.
        sections.sections["today"] = "Today is Tuesday."
        later = []
        for number in range(6, 12):
            out = await turn(agent, f"turn {number}", out)
            ((prompt, messages),) = await llm_calls(out)
            later.append(prompt)
            assert told(messages, prompt, "today", "Today is Tuesday.")
        assert len(set(later)) <= 2 and later[-1] == later[-2]

    async def test_a_history_the_caller_supplies(self):
        """``messages=`` replaces the history: the notes earlier turns left are not in it."""
        fresh()
        seen = []
        sections = Sections(role="You are a bot.", today="Today is Monday.")
        agent = make_agent(sections, seen)
        first = await turn(agent, "one")
        sections.sections["today"] = "Today is Tuesday."
        second = await turn(agent, "two", first)
        supplied = [Message.validate({"role": "user", "content": "three, with a history of my own"})]
        third = await agent(messages=supplied, parent_id=second.run_id).collect()
        assert third.status.code == "success", third.error
        ((prompt, messages),) = await llm_calls(third)
        assert told(messages, prompt, "today", "Today is Tuesday.")

        # And a supplied history that does hold the note is not told twice.
        ((_, second_messages),) = await llm_calls(second)
        fourth = await agent(messages=[*second_messages, *supplied], parent_id=third.run_id).collect()
        ((prompt, messages),) = await llm_calls(fourth)
        assert told(messages, prompt, "today", "Today is Tuesday.") and len(notes(messages)) == 1

    async def test_a_history_that_does_not_continue_the_agents_own(self):
        """The run it continues has no turn of this agent (another agent answered it): no history, so no note."""
        fresh()
        seen = []
        sections = Sections(role="You are a bot.", today="Today is Monday.")
        agent = make_agent(sections, seen)
        first = await turn(agent, "one")
        sections.sections["today"] = "Today is Tuesday."
        second = await turn(agent, "two", first)
        other = Agent(name="other", model=TestModel(responses=["hi"]), tracing_provider=InMemoryTracingProvider)
        between = await other(prompt="x", parent_id=second.run_id).collect()
        assert between.status.code == "success", between.error
        third = await turn(agent, "three", between)
        ((prompt, messages),) = await llm_calls(third)
        assert told(messages, prompt, "today", "Today is Tuesday.")
