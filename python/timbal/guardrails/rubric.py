"""Rubric grading: structured, per-criterion LLM judgment.

The pattern the industry converged on (Anthropic's agent-evals guidance and the Managed
Agents "Outcomes" loop): success criteria written down as a rubric, each criterion graded
by an **isolated** judge call in its own context, per-criterion pass/fail with a reason,
and an explicit UNKNOWN escape hatch so the judge never guesses.

This module is the shared core for both consumers:

- the ``rubric!`` eval validator (``timbal.evals``) — CI-able quality regression
- ``LLMJudge(rubric=...)`` (``timbal.guardrails``) — a runtime quality gate whose failing
  criteria feed the agent's retry loop (grade → revise → re-grade, bounded by
  ``max_guardrail_retries``)

Write criteria around **verifiable structure**, not facts the judge cannot check:
"prices are formatted and attributed to a source" grades reliably; "prices are accurate"
does not.
"""

import asyncio
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["Criterion", "CriterionResult", "RubricResult", "grade_rubric", "parse_rubric"]


class Criterion(BaseModel):
    """One rubric line: something the judge can verify against the text."""

    name: str = ""
    """Short identifier used in reports/feedback. Defaults to a slug of the criterion."""
    criterion: str
    """The requirement, phrased so it is verifiable from the text alone."""
    weight: float = Field(default=1.0, gt=0)
    """Relative weight in the aggregate score."""

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            slug = re.sub(r"[^a-z0-9]+", "_", self.criterion.lower()).strip("_")
            self.name = slug[:40] or "criterion"


class CriterionJudgment(BaseModel):
    """Structured output the judge must produce for one criterion."""

    verdict: Literal["pass", "fail", "unknown"] = Field(
        description=(
            "'pass' if the text verifiably satisfies the criterion, 'fail' if it does not, "
            "'unknown' ONLY when the text gives no way to verify it either way."
        )
    )
    reason: str = Field(description="One or two sentences explaining the verdict, citing the text.")


class CriterionResult(BaseModel):
    """One criterion's graded outcome."""

    name: str
    criterion: str
    weight: float
    verdict: str  # pass | fail | unknown | error
    reason: str


class RubricResult(BaseModel):
    """Aggregate of one rubric pass over one text."""

    results: list[CriterionResult]
    score: float
    """Weighted fraction of passing criteria in [0, 1]. 'unknown' and 'error' count as
    not passing (strict by default — the judge could not verify)."""
    passed: bool

    @property
    def failing(self) -> list[CriterionResult]:
        return [r for r in self.results if r.verdict != "pass"]

    def format_feedback(self) -> str:
        """Failing criteria as revision guidance (what the agent sees on retry)."""
        lines = ["Your output did not satisfy these criteria:"]
        for r in self.failing:
            lines.append(f"- {r.criterion} — {r.reason}")
        lines.append("Revise your output to satisfy every criterion. Keep what already passes.")
        return "\n".join(lines)


_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")


def parse_rubric(spec: Any) -> list[Criterion]:
    """Normalize a rubric spec into criteria.

    Accepts:

    - a markdown string — bullet/numbered lines become criteria (Outcomes-style rubric
      documents); headings and prose are ignored;
    - a list mixing plain strings and dicts (``{"criterion": ..., "name": ..., "weight": ...}``);
    - ``Criterion`` instances.
    """
    if isinstance(spec, str):
        criteria = [Criterion(criterion=m.group(1)) for line in spec.splitlines() if (m := _BULLET_RE.match(line))]
        if not criteria and spec.strip() and "\n" not in spec.strip():
            # A single-line rubric with no bullets is one criterion. Multi-line prose
            # without bullets is ambiguous — reject it rather than grading a blob.
            criteria = [Criterion(criterion=spec.strip())]
        if not criteria:
            raise ValueError("Empty rubric: provide bullet lines or a list of criteria.")
        return criteria
    if isinstance(spec, list | tuple):
        out: list[Criterion] = []
        for item in spec:
            if isinstance(item, Criterion):
                out.append(item)
            elif isinstance(item, str):
                out.append(Criterion(criterion=item))
            elif isinstance(item, dict):
                out.append(Criterion(**item))
            else:
                raise ValueError(f"Invalid rubric entry {item!r}. Expected str, dict, or Criterion.")
        if not out:
            raise ValueError("Empty rubric: provide at least one criterion.")
        _ensure_unique_names(out)
        return out
    raise ValueError(f"Invalid rubric spec {type(spec).__name__}. Expected a markdown string or a list.")


def _ensure_unique_names(criteria: list[Criterion]) -> None:
    seen: dict[str, int] = {}
    for c in criteria:
        if c.name in seen:
            seen[c.name] += 1
            c.name = f"{c.name}_{seen[c.name]}"
        else:
            seen[c.name] = 1


_JUDGE_SYSTEM_PROMPT = """You are a strict grader evaluating a piece of text against ONE criterion.

Rules:
- Judge ONLY from the text provided. Do not use outside knowledge to fill gaps.
- 'pass' requires the text to verifiably satisfy the criterion.
- 'fail' when the text does not satisfy it, or satisfies it only partially.
- 'unknown' ONLY when the text gives you no way to verify the criterion either way.
  Never guess: if you cannot verify, answer 'unknown'.
- Cite the text in your reason."""


def _judge_user_prompt(criterion: Criterion, text: str, context: str | None) -> str:
    parts = []
    if context:
        parts.append(f"Task context:\n{context}\n")
    parts.append(f"Criterion:\n{criterion.criterion}\n")
    parts.append(f"Text to grade:\n{text}")
    return "\n".join(parts)


async def _grade_one(criterion: Criterion, text: str, *, model: Any, context: str | None) -> CriterionResult:
    # Local import: keeps timbal.guardrails importable from timbal.core without a cycle.
    from ..core.agent import Agent

    judge = Agent(
        name=f"rubric_judge_{criterion.name}"[:60],
        model=model,
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        output_model=CriterionJudgment,
        max_tokens=1024,
        temperature=0.0,
    )
    try:
        output_event = await judge(prompt=_judge_user_prompt(criterion, text, context)).collect()
        if output_event.error is not None:
            raise RuntimeError(str(output_event.error))
        judgment: CriterionJudgment = output_event.output
        return CriterionResult(
            name=criterion.name,
            criterion=criterion.criterion,
            weight=criterion.weight,
            verdict=judgment.verdict,
            reason=judgment.reason,
        )
    except Exception as e:
        # A broken judge must not silently pass the criterion.
        return CriterionResult(
            name=criterion.name,
            criterion=criterion.criterion,
            weight=criterion.weight,
            verdict="error",
            reason=f"judge failed: {type(e).__name__}: {e}",
        )


async def grade_rubric(
    rubric: Any,
    text: str,
    *,
    model: Any = "openai/gpt-5.4-nano",
    context: str | None = None,
    pass_threshold: float = 1.0,
    max_concurrency: int = 8,
) -> RubricResult:
    """Grade ``text`` against a rubric, one isolated judge call per criterion.

    Each criterion gets its own judge with its own context window — per-dimension
    isolation grades more reliably than one judge scoring everything at once, and the
    grader is never influenced by the producer's reasoning.

    Args:
        rubric: Anything ``parse_rubric`` accepts.
        text: The artifact to grade.
        model: Judge model (string or a TestModel for offline tests). Use a cheap model.
        context: Optional task description shown to every judge.
        pass_threshold: Weighted fraction of criteria that must pass (1.0 = all).
        max_concurrency: Parallel judge calls cap.
    """
    criteria = parse_rubric(rubric)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(criterion: Criterion) -> CriterionResult:
        async with semaphore:
            return await _grade_one(criterion, text, model=model, context=context)

    results = list(await asyncio.gather(*(_bounded(c) for c in criteria)))
    total_weight = sum(r.weight for r in results)
    passed_weight = sum(r.weight for r in results if r.verdict == "pass")
    score = passed_weight / total_weight if total_weight else 0.0
    return RubricResult(results=results, score=score, passed=score >= pass_threshold)
