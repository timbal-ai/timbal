"""Length bounds on input and output."""

from typing import Any

from pydantic import Field

from ..types import Guardrail, GuardrailContext, GuardrailStage, Verdict

__all__ = ["MaxLength"]


class MaxLength(Guardrail):
    """Block content outside a character budget.

    ```python
    MaxLength(max_chars=20_000, stages=["input"])   # cap prompt size
    MaxLength(min_chars=1, max_chars=5_000, stages=["model_output"])
    ```
    """

    name: str = "max_length"
    stages: set[GuardrailStage] = Field(default_factory=lambda: {GuardrailStage.INPUT})
    action: str = "block"
    max_chars: int | None = None
    min_chars: int | None = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.max_chars is None and self.min_chars is None:
            raise ValueError("MaxLength requires max_chars and/or min_chars.")

    async def check(self, text: str, ctx: GuardrailContext) -> Any:
        n = len(text)
        if self.max_chars is not None and n > self.max_chars:
            return Verdict.block(
                f"{self.name}: {n} chars exceeds the limit of {self.max_chars}",
                blocked_message=self.blocked_message_for(ctx.stage),
            )
        if self.min_chars is not None and n < self.min_chars:
            return Verdict.block(
                f"{self.name}: {n} chars is under the minimum of {self.min_chars}",
                blocked_message=self.blocked_message_for(ctx.stage),
            )
        return Verdict.allow()
