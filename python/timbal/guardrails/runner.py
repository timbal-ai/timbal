"""Guardrail execution engine.

The :class:`GuardrailRunner` owns an ordered list of rails and executes the ones
registered for a given stage. Rails are checked against the text as of their position in
the list, so ``replace`` verdicts chain and everything after a redactor sees the redacted
text. Adjacent rails that cannot rewrite the text observe the same input and are checked
concurrently; a mutating rail is a barrier. The first non-allow verdict in list order
decides the stage outcome.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from .types import Guardrail, GuardrailContext, GuardrailStage, Verdict, coerce_verdict

logger = structlog.get_logger("timbal.guardrails.runner")

__all__ = ["GuardrailRunner", "StageOutcome", "StreamScrubber", "TriggerRecord"]

_MUTATING_ACTIONS = frozenset({"redact", "retry"})


@dataclass
class TriggerRecord:
    """One rail's verdict on one stage pass — the unit of the run report and events."""

    rail: str
    stage: str
    action: str
    reason: str | None
    latency_ms: int
    shadow: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "rail": self.rail,
            "stage": self.stage,
            "action": self.action,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "shadow": self.shadow,
        }
        if self.error is not None:
            out["error"] = self.error
        if self.metadata:
            out["metadata"] = self.metadata
        return out


@dataclass
class StageOutcome:
    """Result of running all of a stage's rails against one piece of content."""

    text: str
    verdict: Verdict | None = None
    """The controlling non-allow verdict (block/retry/escalate), or None."""
    rail: Guardrail | None = None
    """The rail that produced the controlling verdict."""
    triggered: list[TriggerRecord] = field(default_factory=list)
    """Every triggered (or errored) rail, including shadowed ones."""
    replaced: bool = False
    """Whether any replace verdict rewrote the text (or tool args)."""
    replacement_args: dict[str, Any] | None = None
    """For tool_args: the rewritten args dict when a replace verdict fired."""


class StreamScrubber:
    """Windowed in-flight redaction over streamed text deltas.

    Holds back a tail window so patterns spanning chunk boundaries are still caught;
    ``flush()`` releases the held-back tail at end of stream.
    """

    def __init__(self, rails: list[Guardrail]) -> None:
        self._rails = rails
        self._window = max((r.scrub_window for r in rails), default=256)
        self._pending = ""

    def _scrub(self, text: str) -> str:
        for rail in self._rails:
            if not rail.shadow:
                text = rail.scrub(text)
        return text

    def feed(self, chunk: str) -> str:
        """Add a chunk; return the scrubbed stable prefix that is safe to emit."""
        self._pending += chunk
        if len(self._pending) <= self._window:
            return ""
        stable, self._pending = self._pending[: -self._window], self._pending[-self._window :]
        return self._scrub(stable)

    def flush(self) -> str:
        out = self._scrub(self._pending)
        self._pending = ""
        return out


class GuardrailRunner:
    """Executes an ordered list of rails for the stages they register on."""

    def __init__(
        self,
        rails: list[Guardrail],
        *,
        mode: str = "enforce",
        max_retries: int = 2,
    ) -> None:
        if mode not in ("enforce", "shadow"):
            raise ValueError(f"Invalid guardrail mode {mode!r}. Must be 'enforce' or 'shadow'.")
        self.rails = list(rails)
        self.mode = mode
        self.max_retries = max_retries
        seen: set[str] = set()
        for rail in self.rails:
            if rail.name in seen:
                raise ValueError(f"Duplicate guardrail name {rail.name!r}. Give each rail a unique name.")
            seen.add(rail.name)
            if rail.sample_rate < 1.0 and not (rail.shadow or mode == "shadow"):
                enforcing = {rail.action_for(s) for s in GuardrailStage if rail.runs_on(s)} - {"warn"}
                if enforcing:
                    logger.warning(
                        "Sampled enforcement: this rail enforces on only a fraction of checks. "
                        "For monitoring, combine sample_rate with shadow=True or action='warn'.",
                        rail=rail.name,
                        sample_rate=rail.sample_rate,
                        actions=sorted(enforcing),
                    )

    # -- introspection ---------------------------------------------------------------

    def stage_rails(self, stage: GuardrailStage) -> list[Guardrail]:
        return [r for r in self.rails if r.runs_on(stage)]

    def has_stage(self, stage: GuardrailStage) -> bool:
        return any(r.runs_on(stage) for r in self.rails)

    def needs_buffering(self, *stages: GuardrailStage) -> bool:
        """True when any enforced rail on these stages requires a full-text verdict before
        content may be released (block/retry/escalate, or a replace that is not a
        deterministic scrub)."""
        for stage in stages:
            for rail in self.stage_rails(stage):
                if rail.shadow or self.mode == "shadow":
                    continue
                action = rail.action_for(stage)
                if action in ("block", "retry", "escalate"):
                    return True
                if action == "redact" and not rail.streamable:
                    return True
        return False

    def scrub_rails(self, *stages: GuardrailStage) -> list[Guardrail]:
        """The enforced deterministic redact rails across these stages (deduped, in order)."""
        if self.mode == "shadow":
            return []
        out: list[Guardrail] = []
        seen: set[str] = set()
        for rail in self.rails:
            if rail.shadow or rail.name in seen:
                continue
            if any(rail.runs_on(s) and rail.action_for(s) == "redact" and rail.streamable for s in stages):
                out.append(rail)
                seen.add(rail.name)
        return out

    def scrub_text(self, text: str, *stages: GuardrailStage) -> str:
        """Apply every enforced deterministic redact rail of these stages to ``text``."""
        for rail in self.scrub_rails(*stages):
            text = rail.scrub(text)
        return text

    def stream_scrubber(self, *stages: GuardrailStage) -> StreamScrubber | None:
        """A scrubber over these stages' enforced deterministic redact rails, or None."""
        rails = self.scrub_rails(*stages)
        return StreamScrubber(rails) if rails else None

    def merged_with(self, extra: "GuardrailRunner | list[Guardrail] | None") -> "GuardrailRunner":
        """A runner over ``self.rails + extra`` (agent-level + tool-local rails)."""
        if not extra:
            return self
        extra_rails = extra.rails if isinstance(extra, GuardrailRunner) else list(extra)
        if not extra_rails:
            return self
        return GuardrailRunner(self.rails + extra_rails, mode=self.mode, max_retries=self.max_retries)

    # -- execution ---------------------------------------------------------------

    def _is_shadowed(self, rail: Guardrail) -> bool:
        return self.mode == "shadow" or rail.shadow

    async def _check_one(self, rail: Guardrail, text: str, ctx: GuardrailContext) -> tuple[Verdict, TriggerRecord | None]:
        t0 = time.perf_counter()
        try:
            verdict = coerce_verdict(await rail.check(text, ctx))
        except Exception as e:
            latency = int((time.perf_counter() - t0) * 1000)
            logger.exception("Guardrail crashed.", rail=rail.name, stage=ctx.stage.value, strict=rail.strict)
            record = TriggerRecord(
                rail=rail.name,
                stage=ctx.stage.value,
                action="error",
                reason=f"{type(e).__name__}: {e}",
                latency_ms=latency,
                shadow=self._is_shadowed(rail),
                error=type(e).__name__,
            )
            if rail.strict and not self._is_shadowed(rail):
                # Fail closed: a broken security rail must not silently allow.
                verdict = Verdict.block(
                    f"guardrail '{rail.name}' failed (strict mode)",
                    blocked_message=rail.blocked_message_for(ctx.stage),
                )
                return verdict, record
            return Verdict.allow(), record
        latency = int((time.perf_counter() - t0) * 1000)
        if not verdict.triggered:
            return verdict, None
        record = TriggerRecord(
            rail=rail.name,
            stage=ctx.stage.value,
            action=verdict.action,
            reason=verdict.reason,
            latency_ms=latency,
            shadow=self._is_shadowed(rail),
            metadata=verdict.metadata,
        )
        return verdict, record

    async def run_stage(self, stage: GuardrailStage, text: str, ctx: GuardrailContext) -> StageOutcome:
        """Run every rail registered for ``stage`` against ``text``.

        Every rail sees the text as of **its own position in the list**: a rail placed
        after a redactor is checked against the redacted text, never the raw original.
        That is what makes the documented "normalize first, judge second" ordering mean
        what it says — and it keeps raw PII out of the LLM judges and moderation APIs
        that a redactor was put in front of to protect.

        Concurrency is preserved where it is safe: a run of adjacent rails that cannot
        rewrite the text all observe the same input, so they are checked together. A
        mutating rail is a barrier that must resolve before the rails behind it run.
        Batching is decided from each rail's configured action, so a rail that returns
        replacement text without declaring ``action="redact"`` is a loud error rather
        than a silent ordering bug.

        Shadowed rails are always evaluated (their verdicts are recorded) but never
        enforced, so a shadowed redactor does not alter what later rails see.
        """
        outcome = StageOutcome(text=text)
        rails = [
            r
            for r in self.stage_rails(stage)
            if r.sample_rate >= 1.0 or random.random() < r.sample_rate
        ]
        if not rails:
            return outcome

        controlling: tuple[Guardrail, Verdict] | None = None
        current = text

        def apply(rail: Guardrail, verdict: Verdict, record: TriggerRecord | None) -> None:
            nonlocal controlling, current
            if record is not None:
                outcome.triggered.append(record)
            if not verdict.triggered or self._is_shadowed(rail):
                return
            if verdict.action == "replace":
                if isinstance(verdict.replacement, dict):
                    # Rewrites tool args, not the text — no effect on what later rails see.
                    outcome.replacement_args = verdict.replacement
                    outcome.replaced = True
                elif isinstance(verdict.replacement, str):
                    if rail.action_for(stage) not in _MUTATING_ACTIONS:
                        # Batching is decided from the configured action, so a rail that
                        # rewrites the text without declaring it would be checked
                        # alongside rails that should have seen its output.
                        raise ValueError(
                            f"Guardrail '{rail.name}' rewrote the text at stage '{stage.value}' but is "
                            f"configured with action={rail.action_for(stage)!r}. A rail that returns "
                            "replacement text must declare action='redact' so rails after it are "
                            "checked against the rewritten text."
                        )
                    current = verdict.replacement
                    outcome.replaced = True
                return
            if verdict.action == "warn":
                return
            # block / retry / escalate: first one in list order controls the stage.
            if controlling is None:
                controlling = (rail, verdict)

        i = 0
        while i < len(rails):
            # Widest run of rails from i that cannot rewrite the text — they all see
            # `current`, so one gather covers them.
            j = i
            while j < len(rails) and rails[j].action_for(stage) not in _MUTATING_ACTIONS:
                j += 1

            if j > i:
                batch = rails[i:j]
                checked = await asyncio.gather(*(self._check_one(r, current, ctx) for r in batch))
                for rail, (verdict, record) in zip(batch, checked, strict=True):
                    apply(rail, verdict, record)
                i = j

            if i < len(rails):
                rail = rails[i]
                verdict, record = await self._check_one(rail, current, ctx)
                apply(rail, verdict, record)
                i += 1

        outcome.text = current
        if controlling is not None:
            outcome.rail, outcome.verdict = controlling[0], controlling[1]
        return outcome

    def describe(self) -> list[dict[str, Any]]:
        """Introspection rows for ``Agent.explain_guardrails()``."""
        rows = []
        for rail in self.rails:
            stages = sorted(s.value for s in GuardrailStage if rail.runs_on(s))
            actions = {s: rail.action_for(GuardrailStage(s)) for s in stages}
            row = {
                "name": rail.name,
                "type": type(rail).__name__,
                "stages": stages,
                "actions": actions,
                "shadow": self.mode == "shadow" or rail.shadow,
                "strict": rail.strict,
            }
            if rail.sample_rate < 1.0:
                row["sample_rate"] = rail.sample_rate
            rows.append(row)
        return rows
