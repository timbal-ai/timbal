"""Testing helpers: run guardrails against text without touching an LLM or an agent loop.

```python
report = await check_guardrails(agent, "my ssn is 123-45-6789")
assert report.triggered("detect_pii").action == "replace"
```
"""

from dataclasses import dataclass, field
from typing import Any

from .presets import build_guardrail_runner
from .runner import GuardrailRunner, TriggerRecord
from .types import GuardrailContext, GuardrailStage

__all__ = ["GuardrailReport", "check_guardrails"]


@dataclass
class GuardrailReport:
    """Per-rail verdicts from one :func:`check_guardrails` pass."""

    stage: str
    text: str
    """The text after any replace/redact verdicts were applied."""
    records: list[TriggerRecord] = field(default_factory=list)
    blocked: bool = False
    blocking_rail: str | None = None

    def triggered(self, rail: str) -> TriggerRecord | None:
        """The trigger record for ``rail``, or None if it did not fire."""
        return next((r for r in self.records if r.rail == rail), None)

    @property
    def triggered_rails(self) -> list[str]:
        return [r.rail for r in self.records]


def _resolve_runner(target: Any) -> GuardrailRunner:
    if hasattr(target, "_guardrail_runner"):
        # An Agent — use its compiled runner (never wrap the agent itself as a rail).
        runner = target._guardrail_runner
        if isinstance(runner, GuardrailRunner):
            return runner
        raise ValueError("No guardrails configured on the given agent / spec.")
    runner = build_guardrail_runner(target)
    if runner is None:
        raise ValueError("No guardrails configured on the given agent / spec.")
    return runner


async def check_guardrails(
    target: Any,
    text: str,
    *,
    stage: str = "input",
) -> GuardrailReport:
    """Run only the guardrails (no LLM loop) of ``target`` against ``text``.

    ``target`` can be an Agent with ``guardrails=`` configured, a list of rails, a
    single rail, or a shorthand spec — anything ``Agent(guardrails=...)`` accepts.
    LLM-backed rails (Moderate, TopicGuard, LLMJudge) do make their classifier calls.
    """
    runner = _resolve_runner(target)
    resolved_stage = GuardrailStage(stage)
    ctx = GuardrailContext(stage=resolved_stage)
    outcome = await runner.run_stage(resolved_stage, text, ctx)
    return GuardrailReport(
        stage=resolved_stage.value,
        text=outcome.text,
        records=outcome.triggered,
        blocked=outcome.verdict is not None and outcome.verdict.action == "block",
        blocking_rail=outcome.rail.name if outcome.rail is not None else None,
    )
