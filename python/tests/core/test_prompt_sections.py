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
