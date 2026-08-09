"""Timbal guardrails: four-edge content policy for agents.

Rails intercept content at four edges — user input, model output, tool args, and tool
results — with rich verdicts (block / redact / retry / escalate / warn), shadow mode for
zero-risk rollout, stream-safe enforcement, and first-class events and reporting.

```python
from timbal import Agent

Agent(..., guardrails="default")
Agent(..., guardrails=["pii:redact", "injection:block"])

from timbal.guardrails import DetectPII, LLMJudge, guardrail

Agent(..., guardrails=[
    DetectPII(on_input="redact", on_output="block"),
    LLMJudge("Must not give medical advice", action="retry"),
    guardrail(lambda text: "acme" not in text.lower(), stages=["model_output"]),
])
```

Built-in rails are imported lazily so this package never drags provider SDKs (or
``timbal.core``) into module load.
"""

from typing import Any

from .presets import build_guardrail_runner, default_safety
from .rubric import Criterion, CriterionResult, RubricResult, grade_rubric, parse_rubric
from .runner import GuardrailRunner, StageOutcome, StreamScrubber, TriggerRecord
from .testing import GuardrailReport, check_guardrails
from .trace import trace_redactor
from .types import (
    Guardrail,
    GuardrailContext,
    GuardrailMatch,
    GuardrailStage,
    Verdict,
    coerce_verdict,
    guardrail,
)

_BUILTINS = {
    "DetectPII": "pii",
    "RedactSecrets": "secrets",
    "PromptInjection": "injection",
    "KeywordGuard": "keywords",
    "MaxLength": "length",
    "Moderate": "moderate",
    "TopicGuard": "topic",
    "LLMJudge": "judge",
}

__all__ = [
    "Criterion",
    "CriterionResult",
    "Guardrail",
    "GuardrailContext",
    "GuardrailMatch",
    "GuardrailReport",
    "GuardrailRunner",
    "GuardrailStage",
    "RubricResult",
    "StageOutcome",
    "StreamScrubber",
    "TriggerRecord",
    "Verdict",
    "build_guardrail_runner",
    "check_guardrails",
    "coerce_verdict",
    "default_safety",
    "grade_rubric",
    "guardrail",
    "parse_rubric",
    "trace_redactor",
    *sorted(_BUILTINS),
]


def __getattr__(name: str) -> Any:
    module_name = _BUILTINS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".builtins.{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value
