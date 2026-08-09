"""Core guardrail types: stages, verdicts, the Guardrail base class, and callable wrapping.

This module is dependency-light on purpose: it must be importable from
``timbal.core.agent`` and ``timbal.core.runnable`` without creating import cycles, so it
never imports from ``timbal.core``.
"""

import asyncio
import hashlib
import inspect
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Guardrail",
    "GuardrailAction",
    "GuardrailContext",
    "GuardrailMatch",
    "GuardrailStage",
    "Verdict",
    "coerce_verdict",
    "guardrail",
]


class GuardrailStage(StrEnum):
    """The edges of an agent run where guardrails can intercept content."""

    INPUT = "input"
    MODEL_OUTPUT = "model_output"
    MODEL_STEP = "model_step"
    """Every assistant message, including intermediate tool-calling steps — not just the
    final response. Opt-in: judging every step with an LLM rail multiplies classifier
    calls per turn (deterministic rails are free)."""
    TOOL_ARGS = "tool_args"
    TOOL_RESULT = "tool_result"


GuardrailAction = Literal["block", "redact", "warn", "retry", "escalate"]
"""Enforcement strategy configured on a rail. ``redact`` produces replace verdicts using
the rail's ``scrub``; the other values map to the verdict action of the same name."""

_VERDICT_ACTIONS = frozenset({"allow", "block", "replace", "retry", "escalate", "warn"})

_DEFAULT_BLOCKED_INPUT = "This request was blocked by a content policy."
_DEFAULT_BLOCKED_OUTPUT = "The response was withheld by a content policy."


class Verdict(BaseModel):
    """The outcome of one guardrail check.

    ``action`` semantics:

    - ``allow`` — pass through untouched.
    - ``block`` — stop: the run ends with ``status.code="blocked"`` (input/output) or a
      ``[Blocked by guardrail]`` tool result is fed back to the LLM (tool stages).
    - ``replace`` — swap the content for ``replacement`` and continue (redaction is a
      replace verdict produced from the rail's ``scrub``).
    - ``retry`` — reject the model output and re-generate with ``feedback`` appended
      (model_output stage only; bounded by ``Agent.max_guardrail_retries``).
    - ``escalate`` — convert into a human approval gate (tool_args stage only).
    - ``warn`` — allow, but record the violation in events and the run report.
    """

    action: str = "allow"
    reason: str | None = None
    """Dev-facing explanation. Recorded in events/traces, never shown to end users."""
    replacement: Any = None
    """For ``replace``: the new text (str) or new tool args (dict)."""
    feedback: str | None = None
    """For ``retry``: critique injected as a user message before re-generating."""
    blocked_message: str | None = None
    """User-safe text returned as the assistant reply when this verdict blocks."""
    approval_prompt: str | None = None
    """For ``escalate``: the prompt shown on the resulting approval gate."""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.action not in _VERDICT_ACTIONS:
            raise ValueError(f"Invalid verdict action {self.action!r}. Must be one of {sorted(_VERDICT_ACTIONS)}.")

    # -- constructors ------------------------------------------------------------

    @classmethod
    def allow(cls) -> "Verdict":
        return cls(action="allow")

    @classmethod
    def block(cls, reason: str | None = None, *, blocked_message: str | None = None) -> "Verdict":
        return cls(action="block", reason=reason, blocked_message=blocked_message)

    @classmethod
    def redact(cls, replacement: Any, *, reason: str | None = None) -> "Verdict":
        return cls(action="replace", replacement=replacement, reason=reason)

    @classmethod
    def replace(cls, replacement: Any, *, reason: str | None = None) -> "Verdict":
        return cls(action="replace", replacement=replacement, reason=reason)

    @classmethod
    def retry(cls, feedback: str, *, reason: str | None = None) -> "Verdict":
        return cls(action="retry", feedback=feedback, reason=reason)

    @classmethod
    def escalate(cls, approval_prompt: str | None = None, *, reason: str | None = None) -> "Verdict":
        return cls(action="escalate", approval_prompt=approval_prompt, reason=reason)

    @classmethod
    def warn(cls, reason: str | None = None) -> "Verdict":
        return cls(action="warn", reason=reason)

    @property
    def triggered(self) -> bool:
        return self.action != "allow"


def coerce_verdict(raw: Any) -> Verdict:
    """Coerce a guard callable's return value into a :class:`Verdict`.

    ``True``/``None`` → allow, ``False`` → block, ``str`` → replace with that string,
    ``dict`` → replace (tool args), ``Verdict`` → as-is. Anything else is a loud error so
    a buggy guard never silently allows.
    """
    if raw is None or raw is True:
        return Verdict.allow()
    if raw is False:
        return Verdict.block()
    if isinstance(raw, Verdict):
        return raw
    if isinstance(raw, str):
        return Verdict.replace(raw)
    if isinstance(raw, dict):
        return Verdict.replace(raw)
    raise ValueError(
        f"Guardrail returned {type(raw).__name__!r}; expected bool, None, str, dict, or Verdict."
    )


@dataclass
class GuardrailContext:
    """Execution context passed to :meth:`Guardrail.check`."""

    stage: GuardrailStage
    agent_path: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    payload: Any = None
    """The raw object under check: the Message for output stages, the validated input
    dict for tool_args, the ToolResultContent for tool_result."""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailMatch:
    """One detection produced by a deterministic rail's :meth:`Guardrail.detect`."""

    kind: str
    start: int
    end: int
    text: str


class Guardrail(BaseModel):
    """Base class for all guardrails.

    Two implementation styles:

    - **Deterministic rails** implement :meth:`detect` (and get redaction via the base
      :meth:`scrub`); the base :meth:`check` turns matches into verdicts according to the
      configured action.
    - **Judgment rails** (LLM classifiers, external APIs, custom logic) override
      :meth:`check` directly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    """Rail identifier used in events, reports, and status reasons."""
    stages: set[GuardrailStage] = Field(default_factory=lambda: {GuardrailStage.INPUT, GuardrailStage.MODEL_OUTPUT})
    """Which edges this rail runs on."""
    action: str = "block"
    """Default enforcement when the rail triggers: block | redact | warn | retry | escalate."""
    on_input: str | None = None
    """Per-stage action override for the input stage."""
    on_output: str | None = None
    """Per-stage action override for the model_output stage."""
    on_step: str | None = None
    """Per-stage action override for the model_step stage (every assistant message)."""
    on_tool_args: str | None = None
    """Per-stage action override for the tool_args stage."""
    on_tool_result: str | None = None
    """Per-stage action override for the tool_result stage."""
    shadow: bool = False
    """Record verdicts in events/reports without enforcing them."""
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    """Fraction of checks this rail actually runs on (1.0 = every check, 0.1 = ~10%).
    Sampled-out checks record nothing. Built for cost-bounded online monitoring —
    LLM judges in shadow/warn mode grading a slice of production traffic. Sampling an
    *enforcing* rail creates nondeterministic enforcement gaps (and, on the output
    stages, the stream still buffers every run because the buffering decision precedes
    the sampling roll) — you will be warned."""
    strict: bool = False
    """If the rail itself crashes: True fails closed (block), False fails open (allow)."""
    blocked_message: str | None = None
    """User-safe text shown when this rail blocks. Defaults per stage."""
    scrub_window: int = 256
    """Holdback window (chars) for in-flight stream scrubbing. Must cover the longest
    pattern this rail can match across chunk boundaries."""

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            self.name = _snake_case(type(self).__name__)
        # Normalize stages given as strings.
        self.stages = {GuardrailStage(s) for s in self.stages}
        for attr in ("action", "on_input", "on_output", "on_step", "on_tool_args", "on_tool_result"):
            value = getattr(self, attr)
            if value is not None and value not in ("block", "redact", "warn", "retry", "escalate"):
                raise ValueError(
                    f"Invalid guardrail action {value!r} for {attr!r}. "
                    "Must be one of: block, redact, warn, retry, escalate."
                )

    # -- configuration resolution --------------------------------------------------

    _STAGE_OVERRIDES = {
        GuardrailStage.INPUT: "on_input",
        GuardrailStage.MODEL_OUTPUT: "on_output",
        GuardrailStage.MODEL_STEP: "on_step",
        GuardrailStage.TOOL_ARGS: "on_tool_args",
        GuardrailStage.TOOL_RESULT: "on_tool_result",
    }

    def action_for(self, stage: GuardrailStage) -> str:
        override = getattr(self, self._STAGE_OVERRIDES[stage])
        return override or self.action

    def runs_on(self, stage: GuardrailStage) -> bool:
        if stage in self.stages:
            return True
        # A per-stage action override implicitly opts the rail into that stage.
        return getattr(self, self._STAGE_OVERRIDES[stage]) is not None

    @property
    def streamable(self) -> bool:
        """Whether this rail can transform a stream in flight (deterministic redaction)."""
        return type(self).detect is not Guardrail.detect

    def blocked_message_for(self, stage: GuardrailStage) -> str:
        if self.blocked_message:
            return self.blocked_message
        return _DEFAULT_BLOCKED_INPUT if stage == GuardrailStage.INPUT else _DEFAULT_BLOCKED_OUTPUT

    # -- detection / transformation --------------------------------------------------

    def detect(self, text: str) -> list[GuardrailMatch]:
        """Deterministic detection. Override in pattern-based rails."""
        raise NotImplementedError

    def redact_match(self, match: GuardrailMatch) -> str:
        """Replacement text for one match. Override to customize (mask, hash, ...)."""
        return f"[REDACTED_{match.kind.upper()}]"

    def scrub(self, text: str) -> str:
        """Replace every detection in ``text``. Used by redact actions and stream transforms."""
        matches = self.detect(text)
        if not matches:
            return text
        out: list[str] = []
        cursor = 0
        for m in sorted(matches, key=lambda m: (m.start, -m.end)):
            if m.start < cursor:
                continue  # overlapping match already covered
            out.append(text[cursor : m.start])
            out.append(self.redact_match(m))
            cursor = m.end
        out.append(text[cursor:])
        return "".join(out)

    async def check(self, text: str, ctx: GuardrailContext) -> Any:
        """Run the rail against ``text``. Default: detect() + configured action."""
        matches = self.detect(text)
        if not matches:
            return Verdict.allow()
        kinds = sorted({m.kind for m in matches})
        reason = f"{self.name} detected {len(matches)} match(es): {', '.join(kinds)}"
        action = self.action_for(ctx.stage)
        if action == "redact":
            return Verdict.replace(self.scrub(text), reason=reason)
        if action == "warn":
            return Verdict.warn(reason)
        if action == "retry":
            return Verdict.retry(
                f"Your response was rejected: {reason}. Rewrite it without that content.", reason=reason
            )
        if action == "escalate":
            return Verdict.escalate(reason=reason)
        return Verdict.block(reason, blocked_message=self.blocked_message_for(ctx.stage))


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


class FunctionGuardrail(Guardrail):
    """Wraps a plain callable as a guardrail.

    The callable receives ``(text)`` or ``(text, ctx)`` (sync or async) and returns
    ``bool | None | str | dict | Verdict`` — see :func:`coerce_verdict`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fn: Any = None

    def model_post_init(self, __context: Any) -> None:
        if self.fn is None:
            raise ValueError("FunctionGuardrail requires a callable 'fn'.")
        if not self.name:
            self.name = getattr(self.fn, "__name__", "") or "custom_guardrail"
            if self.name == "<lambda>":
                self.name = f"guardrail_{_hash_text(repr(self.fn))}"
        super().model_post_init(__context)
        try:
            sig = inspect.signature(self.fn)
            self._wants_ctx = len(sig.parameters) >= 2
        except (TypeError, ValueError):
            self._wants_ctx = False

    async def check(self, text: str, ctx: GuardrailContext) -> Any:
        args = (text, ctx) if self._wants_ctx else (text,)
        result = self.fn(*args)
        if asyncio.iscoroutine(result) or inspect.isawaitable(result):
            result = await result
        return result


def guardrail(
    fn: Any = None,
    *,
    stages: list[str] | set[str] | None = None,
    name: str | None = None,
    action: str = "block",
    shadow: bool = False,
    sample_rate: float = 1.0,
    strict: bool = False,
    blocked_message: str | None = None,
) -> Any:
    """Wrap a plain callable as a guardrail, or use as a decorator.

    ```python
    def no_competitors(text: str):
        return Verdict.block("competitor mention") if "acme" in text.lower() else True

    agent = Agent(..., guardrails=[guardrail(no_competitors, stages=["model_output"])])

    @guardrail(stages=["input"])
    def clean_input(text: str): ...
    ```
    """

    def _wrap(f: Any) -> FunctionGuardrail:
        resolved_stages = (
            {GuardrailStage(s) for s in stages}
            if stages is not None
            else {GuardrailStage.INPUT, GuardrailStage.MODEL_OUTPUT}
        )
        return FunctionGuardrail(
            fn=f,
            name=name or "",
            stages=resolved_stages,
            action=action,
            shadow=shadow,
            sample_rate=sample_rate,
            strict=strict,
            blocked_message=blocked_message,
        )

    if fn is None:
        return _wrap
    return _wrap(fn)
