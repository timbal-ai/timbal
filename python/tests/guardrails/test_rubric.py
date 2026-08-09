"""Rubric grading: parsing, per-criterion judging, and the runtime quality-gate loop."""

import json

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.guardrails import Criterion, LLMJudge, grade_rubric, parse_rubric
from timbal.types.events import GuardrailEvent, OutputEvent


def _judge(verdict_map: dict[str, str]):
    """A TestModel judge handler: matches criterion keywords to verdicts.

    Keywords match against the criterion section only — the graded text may
    coincidentally contain a keyword.
    """

    def handler(msgs):
        criterion_part = msgs[-1].collect_text().split("Text to grade:")[0]
        for keyword, verdict in verdict_map.items():
            if keyword in criterion_part:
                return json.dumps({"verdict": verdict, "reason": f"judged {keyword}"})
        return json.dumps({"verdict": "unknown", "reason": "no rule matched"})

    return handler


class TestParseRubric:
    def test_markdown_bullets_and_numbers(self):
        criteria = parse_rubric(
            """
# Quality rubric
Some prose that is not a criterion.
- Includes a comparison table
* Cites every source
2) Ends with a recommendation
"""
        )
        assert [c.criterion for c in criteria] == [
            "Includes a comparison table",
            "Cites every source",
            "Ends with a recommendation",
        ]

    def test_single_line_string_is_one_criterion(self):
        [c] = parse_rubric("Mentions the refund policy")
        assert c.criterion == "Mentions the refund policy"

    def test_list_mixing_strings_dicts_and_instances(self):
        criteria = parse_rubric(
            [
                "plain string",
                {"criterion": "weighted one", "name": "big", "weight": 3},
                Criterion(criterion="instance"),
            ]
        )
        assert criteria[1].name == "big" and criteria[1].weight == 3

    def test_names_are_slugged_and_deduped(self):
        criteria = parse_rubric(["Same thing!", "Same thing?"])
        assert criteria[0].name == "same_thing"
        assert criteria[1].name == "same_thing_2"

    def test_empty_rubric_rejected(self):
        with pytest.raises(ValueError, match="Empty rubric"):
            parse_rubric("# just a heading\n\nprose only")
        with pytest.raises(ValueError, match="Empty rubric"):
            parse_rubric([])

    def test_invalid_entry_rejected(self):
        with pytest.raises(ValueError, match="Invalid rubric entry"):
            parse_rubric([42])


class TestGradeRubric:
    @pytest.mark.asyncio
    async def test_per_criterion_verdicts_and_score(self):
        model = TestModel(handler=_judge({"table": "pass", "source": "fail", "accurate": "unknown"}))
        result = await grade_rubric(
            ["Has a table", "Cites the source", "Is accurate"], "text", model=model
        )
        assert [r.verdict for r in result.results] == ["pass", "fail", "unknown"]
        assert result.score == pytest.approx(1 / 3)
        assert not result.passed
        assert len(result.failing) == 2

    @pytest.mark.asyncio
    async def test_weights_shape_the_score(self):
        model = TestModel(handler=_judge({"heavy": "pass", "light": "fail"}))
        result = await grade_rubric(
            [{"criterion": "heavy one", "weight": 3}, {"criterion": "light one", "weight": 1}],
            "text",
            model=model,
            pass_threshold=0.75,
        )
        assert result.score == pytest.approx(0.75)
        assert result.passed

    @pytest.mark.asyncio
    async def test_all_pass(self):
        model = TestModel(handler=_judge({"": "pass"}))
        result = await grade_rubric(["a thing", "another thing"], "text", model=model)
        assert result.passed and result.score == 1.0

    @pytest.mark.asyncio
    async def test_judge_crash_fails_the_criterion_not_the_run(self):
        def broken(msgs):  # noqa: ARG001
            raise RuntimeError("judge down")

        result = await grade_rubric(["a thing"], "text", model=TestModel(handler=broken))
        [r] = result.results
        assert r.verdict == "error" and "judge down" in r.reason
        assert not result.passed, "a broken judge must never silently pass a criterion"

    @pytest.mark.asyncio
    async def test_feedback_lists_failing_criteria_with_reasons(self):
        model = TestModel(handler=_judge({"table": "pass", "source": "fail"}))
        result = await grade_rubric(["Has a table", "Cites the source"], "text", model=model)
        feedback = result.format_feedback()
        assert "Cites the source" in feedback and "judged source" in feedback
        assert "Has a table" not in feedback  # passing criteria are not re-litigated

    @pytest.mark.asyncio
    async def test_context_reaches_the_judge(self):
        seen = []

        def handler(msgs):
            seen.append(msgs[-1].collect_text())
            return json.dumps({"verdict": "pass", "reason": "ok"})

        await grade_rubric(["a thing"], "text", model=TestModel(handler=handler), context="a price report")
        assert "a price report" in seen[0]


class TestLLMJudgeRubricMode:
    def test_requires_criteria_or_rubric(self):
        with pytest.raises(ValueError, match="criteria or rubric"):
            LLMJudge()
        with pytest.raises(ValueError, match="not both"):
            LLMJudge("single", rubric=["a"])

    def test_invalid_rubric_fails_at_construction(self):
        with pytest.raises(ValueError, match="Empty rubric"):
            LLMJudge(rubric=[])

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError, match="pass_threshold"):
            LLMJudge(rubric=["a"], pass_threshold=0.0)

    @pytest.mark.asyncio
    async def test_outcomes_loop_grade_revise_regrade(self):
        """The Outcomes pattern: draft fails the rubric, failing criteria feed the
        revision, the revised draft passes."""

        def judge(msgs):
            prompt = msgs[-1].collect_text()
            text = prompt.split("Text to grade:")[-1].lower()
            if "greeting" in prompt.lower():
                ok = "hello" in text
            else:
                ok = "regards" in text
            return json.dumps({"verdict": "pass" if ok else "fail", "reason": "checked"})

        main_model = TestModel(responses=["hello, here is the answer", "hello, here is the answer. regards"])
        agent = Agent(
            name="writer",
            model=main_model,
            tools=[],
            guardrails=[
                LLMJudge(
                    rubric=["Starts with a greeting", "Ends with a sign-off (regards)"],
                    model=TestModel(handler=judge),
                    action="retry",
                )
            ],
        )
        events = [e async for e in agent(prompt="write it")]
        final = next(e for e in reversed(events) if isinstance(e, OutputEvent))
        assert final.status.code == "success"
        assert final.output.collect_text().endswith("regards")
        assert main_model.call_count == 2

        # the revision feedback the model received names the failing criterion
        retry_event = next(e for e in events if isinstance(e, GuardrailEvent))
        assert retry_event.action == "retry"
        assert retry_event.metadata["rubric"]["score"] == 0.5

    @pytest.mark.asyncio
    async def test_per_criterion_results_land_in_the_run_report(self):
        judge = TestModel(handler=_judge({"greeting": "fail"}))
        agent = Agent(
            name="writer",
            model=TestModel(responses=["draft"]),
            tools=[],
            max_guardrail_retries=0,
            guardrails=[LLMJudge(rubric=["Has a greeting"], model=judge, action="retry")],
        )
        result = await agent(prompt="go").collect()
        # retry budget of 0 → block after the first failed grade
        assert result.status.code == "blocked"
        [entry] = result.metadata["guardrails"]["triggered"]
        criteria = entry["metadata"]["rubric"]["criteria"]
        assert criteria[0]["verdict"] == "fail"
        assert criteria[0]["criterion"] == "Has a greeting"

    @pytest.mark.asyncio
    async def test_pass_threshold_allows_partial_rubrics(self):
        judge = TestModel(handler=_judge({"greeting": "pass", "sign": "fail"}))
        agent = Agent(
            name="writer",
            model=TestModel(responses=["draft"]),
            tools=[],
            guardrails=[
                LLMJudge(rubric=["Has a greeting", "Has a sign-off"], model=judge, pass_threshold=0.5)
            ],
        )
        result = await agent(prompt="go").collect()
        assert result.status.code == "success"
