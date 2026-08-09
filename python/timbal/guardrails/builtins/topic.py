"""Topic control: keep the agent inside (or away from) declared topics."""

from typing import Any

from pydantic import Field

from ..types import Guardrail, GuardrailContext, GuardrailStage, Verdict

__all__ = ["TopicGuard"]

_SYSTEM_PROMPT = """You are a topic classifier for a scoped assistant.
{scope}
Decide whether the user's message is within scope. Small talk and greetings are in scope.
Answer with exactly one word: ON_TOPIC or OFF_TOPIC."""


class TopicGuard(Guardrail):
    """Refuse off-topic requests using a cheap LLM classifier — NeMo topical rails
    without the dialog-flow runtime.

    ```python
    TopicGuard(allow=["billing", "shipping"],
               blocked_message="I can only help with billing and shipping.")
    TopicGuard(deny=["medical advice", "legal advice"], model="openai/gpt-5.4-nano")
    ```
    """

    name: str = "topic_guard"
    stages: set[GuardrailStage] = Field(default_factory=lambda: {GuardrailStage.INPUT})
    action: str = "block"
    allow: list[str] = Field(default_factory=list)
    """Topics the agent may discuss. Anything else is off-topic."""
    deny: list[str] = Field(default_factory=list)
    """Topics that are always off-topic (checked in addition to allow)."""
    model: Any = "openai/gpt-5.4-nano"
    """Classifier model: a model string, or any model instance the router accepts."""
    max_chars: int = 8_000

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not self.allow and not self.deny:
            raise ValueError("TopicGuard requires allow= and/or deny= topics.")

    def _scope(self) -> str:
        parts = []
        if self.allow:
            parts.append("The assistant may ONLY discuss these topics: " + ", ".join(self.allow) + ".")
        if self.deny:
            parts.append("The assistant must NEVER discuss these topics: " + ", ".join(self.deny) + ".")
        return " ".join(parts)

    async def check(self, text: str, ctx: GuardrailContext) -> Any:
        if not text.strip():
            return Verdict.allow()
        from ..judge_llm import call_judge

        answer = await call_judge(
            model=self.model,
            system_prompt=_SYSTEM_PROMPT.format(scope=self._scope()),
            prompt=text[: self.max_chars],
            max_tokens=8,
        )
        if answer is None or "OFF_TOPIC" not in answer.upper():
            return Verdict.allow()
        reason = f"{self.name}: message classified off-topic"
        action = self.action_for(ctx.stage)
        if action == "warn":
            return Verdict.warn(reason)
        return Verdict.block(reason, blocked_message=self.blocked_message_for(ctx.stage))
