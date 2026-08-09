"""Shorthand parsing and presets: the one-string / string-list onboarding surface.

```python
Agent(..., guardrails="default")
Agent(..., guardrails=["pii:redact", "injection:block", "secrets", "moderation:warn"])
```
"""

from typing import Any

from .runner import GuardrailRunner
from .types import FunctionGuardrail, Guardrail

__all__ = ["DEFAULT_SHORTHANDS", "build_guardrail_runner", "coerce_rail", "default_safety"]

_VALID_ACTIONS = ("block", "redact", "warn", "retry", "escalate")

DEFAULT_SHORTHANDS = ("pii:redact", "secrets", "injection:block")
"""The shorthand expansion of the "default" preset — kept in sync with default_safety()
(codegen expands guardrails="default" through this when editing the list)."""


def _builtin_registry() -> dict[str, Any]:
    # Imported lazily so `import timbal.guardrails.presets` stays dependency-light.
    from . import builtins as b

    return {
        "pii": b.DetectPII,
        "secrets": b.RedactSecrets,
        "injection": b.PromptInjection,
        "keywords": b.KeywordGuard,
        "moderation": b.Moderate,
        "length": b.MaxLength,
        "topic": b.TopicGuard,
        "judge": b.LLMJudge,
    }


def default_safety() -> list[Guardrail]:
    """The ``guardrails="default"`` preset: sane, cheap, fully deterministic.

    - PII redacted on input, output, and tool results
    - secrets redacted on output and tool results
    - prompt injection blocked on input

    Equivalent to ``[coerce_rail(s) for s in DEFAULT_SHORTHANDS]``.
    """
    return [_parse_shorthand(s) for s in DEFAULT_SHORTHANDS]


def _parse_shorthand(spec: str) -> Guardrail:
    name, _, action = spec.partition(":")
    name = name.strip().lower()
    action = action.strip().lower()
    registry = _builtin_registry()
    if name not in registry:
        raise ValueError(
            f"Unknown guardrail shorthand {name!r}. Valid names: {sorted(registry)} "
            "(optionally suffixed with an action, e.g. 'pii:redact')."
        )
    if action and action not in _VALID_ACTIONS:
        raise ValueError(
            f"Unknown guardrail action {action!r} in {spec!r}. Valid actions: {list(_VALID_ACTIONS)}."
        )
    kwargs: dict[str, Any] = {"action": action} if action else {}
    return registry[name](**kwargs)


def coerce_rail(item: Any) -> Guardrail:
    """Coerce one entry of ``Agent(guardrails=[...])`` into a Guardrail instance."""
    if isinstance(item, Guardrail):
        return item
    if isinstance(item, str):
        return _parse_shorthand(item)
    if callable(item):
        return FunctionGuardrail(fn=item)
    raise ValueError(
        f"Invalid guardrail entry {item!r}. Expected a Guardrail, a shorthand string "
        "(e.g. 'pii:redact'), or a callable."
    )


def build_guardrail_runner(
    spec: Any,
    *,
    mode: str = "enforce",
    max_retries: int = 2,
) -> GuardrailRunner | None:
    """Build a runner from the ``Agent(guardrails=...)`` value.

    Accepts ``None``, the string ``"default"``, a single rail/shorthand/callable, or a
    list mixing all of the above.
    """
    if spec is None:
        return None
    if isinstance(spec, GuardrailRunner):
        return spec
    if isinstance(spec, str) and spec.strip().lower() == "default":
        rails = default_safety()
    elif isinstance(spec, list | tuple):
        rails = [coerce_rail(item) for item in spec]
    else:
        rails = [coerce_rail(spec)]
    if not rails:
        return None
    return GuardrailRunner(rails, mode=mode, max_retries=max_retries)
