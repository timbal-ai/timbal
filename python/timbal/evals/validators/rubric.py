from typing import Any, Literal

from pydantic import SkipValidation, model_validator

from .base import BaseValidator
from .context import ValidationContext


class RubricValidator(BaseValidator):
    """Rubric validator — grades the target against structured criteria, one isolated
    LLM judge call per criterion (per-dimension judging grades more reliably than one
    judge scoring everything at once).

    Each criterion returns pass / fail / unknown with a reason; 'unknown' is the judge's
    escape hatch when the text gives no way to verify (it counts as not passing). The
    eval fails when the weighted pass fraction is below ``pass_threshold`` (default:
    all criteria must pass), and the failure message lists every failing criterion with
    the judge's reason.

    Usage in YAML:

        output:
            rubric!:
                - "Includes a comparison table"
                - "Every price is attributed to a source"

        # Full form with options and weighted criteria:
        output:
            rubric!:
                criteria:
                    - "Includes a comparison table"
                    - criterion: "At least 3 actionable recommendations"
                      name: recommendations
                      weight: 2
                pass_threshold: 0.75
                model: openai/gpt-5.4-nano
                context: "The agent produced a price-comparison report."

        # Markdown rubric (bullet lines become criteria):
        output:
            rubric!: |
                - Mentions the refund policy
                - Ends by offering further help

    Write criteria around verifiable structure ("prices are formatted and attributed"),
    not facts the judge cannot check ("prices are accurate").
    """

    name: Literal["rubric!"] = "rubric!"  # type: ignore
    model: SkipValidation[Any] = "openai/gpt-5.4-nano"
    """Judge model. Use a small, cheap model."""
    pass_threshold: float = 1.0
    """Weighted fraction of criteria that must pass (1.0 = all)."""
    context: str | None = None
    """Optional task description shown to every judge."""

    @model_validator(mode="before")
    @classmethod
    def extract_rubric_options(cls, data: Any) -> Any:
        """Support the dict form: ``rubric!: {criteria: [...], model: ..., ...}``."""
        if not isinstance(data, dict):
            return data
        value = data.get("value")
        if isinstance(value, dict) and "criteria" in value:
            extra = {k: v for k, v in value.items() if k != "criteria"}
            return {**data, "value": value["criteria"], **extra}
        return data

    async def __call__(self, ctx: ValidationContext) -> None:
        from ...guardrails.rubric import grade_rubric, parse_rubric
        from ...types.message import Message
        from ..utils import resolve_target

        criteria = parse_rubric(self.value)

        _, actual_value = resolve_target(ctx.trace, self.target, self.path_key)
        if isinstance(actual_value, Message):
            actual_value = actual_value.collect_text()
        if not isinstance(actual_value, str):
            actual_value = str(actual_value)
        actual_value = self.apply_transform(actual_value)

        result = await grade_rubric(
            criteria,
            actual_value,
            model=self.model,
            context=self.context,
            pass_threshold=self.pass_threshold,
        )

        if self.negate:
            if result.passed:
                raise AssertionError(
                    f"rubric! should have failed but passed (score {result.score:.2f})"
                )
            return

        if not result.passed:
            lines = [
                f"rubric! failed: score {result.score:.2f} < threshold {self.pass_threshold} "
                f"({len(result.failing)}/{len(result.results)} criteria not passing)"
            ]
            for r in result.failing:
                lines.append(f"  - [{r.verdict}] {r.criterion} — {r.reason}")
            raise AssertionError("\n".join(lines))
