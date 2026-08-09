"""LLM judge: one-line criteria rail — or a full rubric quality gate."""

from typing import Any

from pydantic import Field

from ..types import Guardrail, GuardrailContext, GuardrailStage, Verdict

__all__ = ["LLMJudge"]

_SYSTEM_PROMPT = """You are a strict content judge. Evaluate the text against this criteria:
{criteria}
Answer on the first line with exactly PASS or FAIL.
If FAIL, explain why in one short sentence on the second line."""


class LLMJudge(Guardrail):
    """Judge content against free-form criteria — or a structured rubric.

    Single criteria (one judge call, PASS/FAIL):

    ```python
    LLMJudge("Response must not give medical advice", action="retry")
    ```

    Rubric mode (one **isolated** judge call per criterion, structured verdicts —
    the grade → revise → re-grade loop popularized by rubric-based agent grading):

    ```python
    LLMJudge(
        rubric=[
            "Includes a comparison table",
            "Every price is attributed to a source",
            {"criterion": "At least 3 actionable recommendations", "weight": 2},
        ],
        pass_threshold=1.0,   # weighted fraction of criteria that must pass
        action="retry",       # failing criteria feed the agent's revision loop
    )
    ```

    With ``action="retry"`` the failing criteria (with the judges' reasons) are fed back
    and the response is re-generated, bounded by ``Agent.max_guardrail_retries``.
    Per-criterion results land in the verdict metadata → GuardrailEvent + run report.

    Write rubric criteria around verifiable structure ("prices are formatted and
    attributed"), not facts the judge cannot check ("prices are accurate").
    """

    name: str = "llm_judge"
    stages: set[GuardrailStage] = Field(default_factory=lambda: {GuardrailStage.MODEL_OUTPUT})
    action: str = "retry"
    criteria: str = ""
    """Free-form criteria for single-call mode. Mutually exclusive with rubric."""
    rubric: Any = None
    """Structured rubric: a markdown string (bullets become criteria) or a list of
    strings / dicts ({"criterion", "name", "weight"}). See timbal.guardrails.rubric."""
    pass_threshold: float = 1.0
    """Rubric mode: weighted fraction of criteria that must pass (1.0 = all)."""
    rubric_context: str | None = None
    """Optional task description shown to every rubric judge."""
    model: Any = "openai/gpt-5.4-nano"
    max_chars: int = 16_000

    def __init__(self, criteria: str | None = None, **kwargs: Any) -> None:
        # Positional sugar: LLMJudge("must not give medical advice", action="retry")
        if criteria is not None:
            kwargs["criteria"] = criteria
        super().__init__(**kwargs)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not self.criteria and self.rubric is None:
            raise ValueError("LLMJudge requires criteria or rubric=.")
        if self.criteria and self.rubric is not None:
            raise ValueError("LLMJudge takes criteria OR rubric=, not both.")
        if self.rubric is not None:
            from ..rubric import parse_rubric

            # Parse eagerly so an invalid rubric fails at construction, not mid-run.
            self._criteria_parsed = parse_rubric(self.rubric)
        else:
            self._criteria_parsed = None
        if not (0.0 < self.pass_threshold <= 1.0):
            raise ValueError("pass_threshold must be in (0, 1].")

    async def check(self, text: str, ctx: GuardrailContext) -> Any:
        if not text.strip():
            return Verdict.allow()
        if self._criteria_parsed is not None:
            return await self._check_rubric(text, ctx)
        return await self._check_single(text, ctx)

    async def _check_rubric(self, text: str, ctx: GuardrailContext) -> Verdict:
        from ..rubric import grade_rubric

        result = await grade_rubric(
            self._criteria_parsed,
            text[: self.max_chars],
            model=self.model,
            context=self.rubric_context,
            pass_threshold=self.pass_threshold,
        )
        metadata = {
            "rubric": {
                "score": round(result.score, 4),
                "passed": result.passed,
                "criteria": [r.model_dump() for r in result.results],
            }
        }
        if result.passed:
            verdict = Verdict.allow()
            verdict.metadata.update(metadata)
            return verdict
        failing_names = ", ".join(r.name for r in result.failing)
        reason = f"{self.name}: rubric score {result.score:.2f} < {self.pass_threshold} (failing: {failing_names})"
        verdict = self._verdict_for_action(ctx, reason=reason, feedback=result.format_feedback())
        verdict.metadata.update(metadata)
        return verdict

    async def _check_single(self, text: str, ctx: GuardrailContext) -> Verdict:
        from ..judge_llm import call_judge

        answer = await call_judge(
            model=self.model,
            system_prompt=_SYSTEM_PROMPT.format(criteria=self.criteria),
            prompt=text[: self.max_chars],
            max_tokens=128,
        )
        if answer is None:
            return Verdict.allow()
        first_line, _, rest = answer.strip().partition("\n")
        if "FAIL" not in first_line.upper():
            return Verdict.allow()
        critique = rest.strip() or f"failed criteria: {self.criteria}"
        reason = f"{self.name}: {critique}"
        return self._verdict_for_action(
            ctx,
            reason=reason,
            feedback=f"Your response was rejected by a quality check: {critique}. Rewrite it to comply.",
        )

    def _verdict_for_action(self, ctx: GuardrailContext, *, reason: str, feedback: str) -> Verdict:
        action = self.action_for(ctx.stage)
        if action == "retry":
            return Verdict.retry(feedback, reason=reason)
        if action == "warn":
            return Verdict.warn(reason)
        if action == "escalate":
            return Verdict.escalate(reason=reason)
        return Verdict.block(reason, blocked_message=self.blocked_message_for(ctx.stage))
