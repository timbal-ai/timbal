"""The rubric! eval validator: YAML forms, target resolution, failure reporting."""

import json

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.evals.validators import parse_validator
from timbal.evals.validators.context import ValidationContext
from timbal.evals.validators.rubric import RubricValidator
from timbal.state import get_run_context


def _judge(verdict_map: dict[str, str]):
    def handler(msgs):
        # Match keywords against the criterion section only — the graded text may
        # coincidentally contain a keyword.
        criterion_part = msgs[-1].collect_text().split("Text to grade:")[0]
        for keyword, verdict in verdict_map.items():
            if keyword in criterion_part:
                return json.dumps({"verdict": verdict, "reason": f"judged {keyword}"})
        return json.dumps({"verdict": "unknown", "reason": "no rule matched"})

    return handler


async def _trace_for(response: str) -> ValidationContext:
    """Run a TestModel agent and wrap its trace in a ValidationContext."""
    agent = Agent(name="writer", model=TestModel(responses=[response]), tools=[])
    await agent(prompt="go").collect()
    ctx = get_run_context()
    assert ctx is not None
    return ValidationContext(trace=ctx._trace)


class TestParsing:
    def test_list_form(self):
        v = parse_validator({"name": "rubric!", "target": "writer.output", "value": ["a", "b"]})
        assert isinstance(v, RubricValidator)
        assert v.value == ["a", "b"]
        assert v.pass_threshold == 1.0

    def test_dict_form_hoists_options(self):
        v = parse_validator(
            {
                "name": "rubric!",
                "target": "writer.output",
                "value": {
                    "criteria": ["a", {"criterion": "b", "weight": 2}],
                    "pass_threshold": 0.5,
                    "model": "openai/gpt-5.4-nano",
                    "context": "a report",
                },
            }
        )
        assert v.pass_threshold == 0.5
        assert v.context == "a report"
        assert len(v.value) == 2

    def test_markdown_form(self):
        v = parse_validator({"name": "rubric!", "target": "writer.output", "value": "- a\n- b"})
        assert isinstance(v.value, str)


class TestValidation:
    @pytest.mark.asyncio
    async def test_passes_when_all_criteria_pass(self):
        ctx = await _trace_for("hello, best regards")
        v = RubricValidator(
            target="writer.output",
            value=["Has a greeting", "Has a sign-off"],
            model=TestModel(handler=_judge({"": "pass"})),
        )
        await v(ctx)  # must not raise

    @pytest.mark.asyncio
    async def test_failure_lists_failing_criteria_with_reasons(self):
        ctx = await _trace_for("no greeting here")
        v = RubricValidator(
            target="writer.output",
            value=["Has a greeting", "Has a body"],
            model=TestModel(handler=_judge({"greeting": "fail", "body": "pass"})),
        )
        with pytest.raises(AssertionError) as err:
            await v(ctx)
        message = str(err.value)
        assert "score 0.50" in message
        assert "[fail] Has a greeting — judged greeting" in message
        assert "Has a body" not in message.split("\n")[0]  # only failures listed as lines

    @pytest.mark.asyncio
    async def test_unknown_counts_as_not_passing(self):
        ctx = await _trace_for("some text")
        v = RubricValidator(
            target="writer.output",
            value=["Something unverifiable"],
            model=TestModel(handler=_judge({})),  # always unknown
        )
        with pytest.raises(AssertionError, match=r"\[unknown\]"):
            await v(ctx)

    @pytest.mark.asyncio
    async def test_pass_threshold(self):
        ctx = await _trace_for("hello")
        v = RubricValidator(
            target="writer.output",
            value=["Has a greeting", "Has a sign-off"],
            pass_threshold=0.5,
            model=TestModel(handler=_judge({"greeting": "pass", "sign": "fail"})),
        )
        await v(ctx)  # 0.5 >= 0.5 — must not raise

    @pytest.mark.asyncio
    async def test_negate(self):
        ctx = await _trace_for("hello")
        v = RubricValidator(
            target="writer.output",
            value=["Has a greeting"],
            negate=True,
            model=TestModel(handler=_judge({"": "pass"})),
        )
        with pytest.raises(AssertionError, match="should have failed"):
            await v(ctx)
