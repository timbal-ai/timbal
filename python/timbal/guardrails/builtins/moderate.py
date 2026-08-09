"""Content moderation adapters: OpenAI Moderation API and Llama-Guard-style endpoints."""

import os
from typing import Any

from pydantic import Field

from ..types import Guardrail, GuardrailContext, GuardrailStage, Verdict

__all__ = ["Moderate"]

_OPENAI_MODERATION_URL = "https://api.openai.com/v1/moderations"

_LLAMA_GUARD_SYSTEM_PROMPT = (
    "Task: Check if the following message contains unsafe content according to standard "
    "safety policy (violence, hate, sexual content involving minors, self-harm, criminal "
    "planning, weapons). Answer with exactly 'safe' or 'unsafe' on the first line."
)


class Moderate(Guardrail):
    """Moderate content through an external classifier.

    ``provider="openai"`` calls the OpenAI Moderation API (free, needs OPENAI_API_KEY).
    ``provider="llama_guard"`` prompts any OpenAI-schema-compatible model (specified via
    ``model=``) with a Llama-Guard-style safe/unsafe template.

    ```python
    Moderate()                                        # openai, block on any flag
    Moderate(categories=["hate", "violence"], threshold=0.7, action="warn")
    Moderate(provider="llama_guard", model="groq/llama-3.3-70b-versatile")
    ```
    """

    name: str = "moderate"
    stages: set[GuardrailStage] = Field(default_factory=lambda: {GuardrailStage.INPUT, GuardrailStage.MODEL_OUTPUT})
    action: str = "block"
    provider: str = "openai"
    categories: list[str] | None = None
    """OpenAI moderation categories to consider (None = all). E.g. hate, harassment,
    violence, sexual, self-harm."""
    threshold: float | None = None
    """Minimum category score to trigger (None = trust the API's boolean flags)."""
    model: Any = None
    """For provider='llama_guard': the model to prompt — a model string, or any model
    instance the router accepts (FallbackModel, TestModel)."""
    api_key: str | None = None
    """Overrides OPENAI_API_KEY for provider='openai'."""
    max_chars: int = 16_000

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.provider not in ("openai", "llama_guard"):
            raise ValueError(f"Invalid Moderate provider {self.provider!r}. Must be 'openai' or 'llama_guard'.")
        if self.provider == "llama_guard" and not self.model:
            raise ValueError("Moderate(provider='llama_guard') requires model=.")

    async def check(self, text: str, ctx: GuardrailContext) -> Any:
        text = text[: self.max_chars]
        if not text.strip():
            return Verdict.allow()
        if self.provider == "openai":
            return await self._check_openai(text, ctx)
        return await self._check_llama_guard(text, ctx)

    async def _check_openai(self, text: str, ctx: GuardrailContext) -> Verdict:
        import httpx

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Moderate(provider='openai') requires OPENAI_API_KEY (or api_key=).")
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                _OPENAI_MODERATION_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "omni-moderation-latest", "input": text},
            )
            response.raise_for_status()
            result = response.json()["results"][0]

        flagged_categories: list[str] = []
        scores = result.get("category_scores", {})
        flags = result.get("categories", {})
        considered = self.categories if self.categories is not None else list(flags)
        for category in considered:
            score = scores.get(category)
            if self.threshold is not None:
                if score is not None and score >= self.threshold:
                    flagged_categories.append(category)
            elif flags.get(category):
                flagged_categories.append(category)

        if not flagged_categories:
            return Verdict.allow()
        reason = f"{self.name}: flagged for {', '.join(sorted(flagged_categories))}"
        return self._verdict_for(reason, ctx, metadata={"categories": sorted(flagged_categories)})

    async def _check_llama_guard(self, text: str, ctx: GuardrailContext) -> Verdict:
        from ..judge_llm import call_judge

        answer = await call_judge(
            model=self.model,
            system_prompt=_LLAMA_GUARD_SYSTEM_PROMPT,
            prompt=text,
            max_tokens=16,
        )
        if answer is None or "unsafe" not in answer.lower():
            return Verdict.allow()
        return self._verdict_for(f"{self.name}: classifier answered unsafe", ctx, metadata={"answer": answer})

    def _verdict_for(self, reason: str, ctx: GuardrailContext, *, metadata: dict[str, Any]) -> Verdict:
        action = self.action_for(ctx.stage)
        if action == "warn":
            verdict = Verdict.warn(reason)
        elif action == "retry":
            verdict = Verdict.retry(f"Your response was rejected by moderation: {reason}. Rewrite it.", reason=reason)
        else:
            verdict = Verdict.block(reason, blocked_message=self.blocked_message_for(ctx.stage))
        verdict.metadata.update(metadata)
        return verdict
