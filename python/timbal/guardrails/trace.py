"""Trace-boundary redaction: scrub persisted/exported traces without touching the run.

The v1 guardrails redact agent memory and outputs, but the inner LLM child span still
carries the raw text into traces. Nobody selectively redacts inside spans — the industry
answer (LangSmith ``hide_inputs``, OpenAI's ``trace_include_sensitive_data``) is to
transform or omit at the observability boundary. This module implements that boundary
for Timbal: a redactor callable attached to any tracing provider via ``configured()``:

```python
from timbal.guardrails import trace_redactor
from timbal.state.tracing.providers import JsonlTracingProvider

provider = JsonlTracingProvider.configured(
    _path=Path("traces.jsonl"),
    _trace_redactor=trace_redactor(),          # PII + secrets, deterministic
    # _trace_redactor=trace_redactor("pii:redact", DetectPII(types=["ssn"])),
)
agent = Agent(..., tracing_provider=provider, guardrails="default")
```

The redactor runs inside ``TracingProvider.put()`` on a **copied view** of every span —
the live run (memory, dumps, outputs) is never mutated, only what gets stored and
exported. All spans are covered, including the inner LLM span that in-run guardrails
cannot reach.
"""

from collections.abc import Callable
from typing import Any

from .presets import coerce_rail
from .types import Guardrail

__all__ = ["trace_redactor"]

_DEFAULT_SPECS = ("pii", "secrets")


def trace_redactor(*specs: Any) -> Callable[[Any], Any]:
    """Build a redactor for ``TracingProvider.configured(_trace_redactor=...)``.

    Accepts the same values as ``Agent(guardrails=[...])`` — shorthand strings,
    ``Guardrail`` instances — but only **deterministic** rails (those implementing
    ``detect``/``scrub``): trace redaction runs on every span store, so LLM-backed
    rails are rejected loudly. With no arguments, defaults to PII + secret redaction.

    The returned callable walks any JSON-ish value (dicts, lists, strings) and scrubs
    every string through the rails. Non-string leaves and unknown objects pass through
    untouched.
    """
    rails: list[Guardrail] = [coerce_rail(s) for s in (specs or _DEFAULT_SPECS)]
    for rail in rails:
        if not rail.streamable:
            raise ValueError(
                f"trace_redactor only accepts deterministic rails (detect/scrub); "
                f"'{rail.name}' ({type(rail).__name__}) is judgment-based. "
                "Trace redaction runs on every span store — an LLM call there is a footgun."
            )

    def _scrub(text: str) -> str:
        for rail in rails:
            text = rail.scrub(text)
        return text

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return _scrub(value)
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_walk(v) for v in value)
        return value

    return _walk
