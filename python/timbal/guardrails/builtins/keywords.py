"""Keyword allow/block lists — the simplest deterministic rail."""

import re
from typing import Any

from pydantic import Field

from ..types import Guardrail, GuardrailMatch, GuardrailStage

__all__ = ["KeywordGuard"]


class KeywordGuard(Guardrail):
    """Trigger on banned words or phrases (literal or regex).

    ```python
    KeywordGuard(banned=["acme corp", r"project\\s+titan"], action="block")
    KeywordGuard(banned=["internal codename"], action="redact")
    ```
    """

    name: str = "keyword_guard"
    stages: set[GuardrailStage] = Field(default_factory=lambda: {GuardrailStage.INPUT, GuardrailStage.MODEL_OUTPUT})
    action: str = "block"
    banned: list[str] = Field(default_factory=list)
    """Banned terms. Each entry is a case-insensitive regex; plain words work as-is."""
    case_sensitive: bool = False

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not self.banned:
            raise ValueError("KeywordGuard requires at least one banned term.")
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._compiled = [re.compile(term, flags) for term in self.banned]

    def detect(self, text: str) -> list[GuardrailMatch]:
        matches: list[GuardrailMatch] = []
        for pattern in self._compiled:
            for m in pattern.finditer(text):
                matches.append(GuardrailMatch(kind="keyword", start=m.start(), end=m.end(), text=m.group()))
        return matches

    def redact_match(self, match: GuardrailMatch) -> str:  # noqa: ARG002
        return "[REDACTED]"
