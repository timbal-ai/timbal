"""Deterministic PII detection: regex + validation (Luhn), zero dependencies, zero LLM cost."""

import hashlib
import re
from typing import Any

from pydantic import Field

from ..types import Guardrail, GuardrailMatch, GuardrailStage

__all__ = ["DetectPII"]

_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # 13-19 digits with optional spaces/dashes between groups; Luhn-validated below.
    "credit_card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(
        r"(?<![\d.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3}[ .-]\d{3,4}[ .-]?\d{0,4}(?![\d.])"
    ),
    "ip": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "url": re.compile(r"\bhttps?://[^\s<>\"']+", re.IGNORECASE),
}

PII_TYPES = tuple(_PATTERNS)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class DetectPII(Guardrail):
    """Detect (and redact/mask/hash/block) personally identifiable information.

    Deterministic — regex plus Luhn validation for card numbers — so it costs nothing
    and streams safely. Default: redact on input, model output, and tool results.

    ```python
    DetectPII()                                   # redact everywhere it runs
    DetectPII(types=["email", "ssn"], action="block")
    DetectPII(on_input="redact", on_output="block", redaction="mask")
    ```
    """

    name: str = "detect_pii"
    stages: set[GuardrailStage] = Field(
        default_factory=lambda: {GuardrailStage.INPUT, GuardrailStage.MODEL_OUTPUT, GuardrailStage.TOOL_RESULT}
    )
    action: str = "redact"
    types: list[str] = Field(default_factory=lambda: list(PII_TYPES))
    """Which PII kinds to detect: email, credit_card, ssn, phone, ip, url."""
    redaction: str = "placeholder"
    """How redaction renders: 'placeholder' ([REDACTED_EMAIL]), 'mask' (keep last 4),
    'hash' (<email_hash:a1b2c3d4> — pseudonymous, joinable for analytics)."""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        unknown = set(self.types) - set(PII_TYPES)
        if unknown:
            raise ValueError(f"Unknown PII types {sorted(unknown)}. Valid: {list(PII_TYPES)}.")
        if self.redaction not in ("placeholder", "mask", "hash"):
            raise ValueError(f"Invalid redaction {self.redaction!r}. Must be placeholder, mask, or hash.")

    def detect(self, text: str) -> list[GuardrailMatch]:
        matches: list[GuardrailMatch] = []
        for kind in self.types:
            for m in _PATTERNS[kind].finditer(text):
                if kind == "credit_card":
                    digits = re.sub(r"\D", "", m.group())
                    if not (13 <= len(digits) <= 19 and _luhn_valid(digits)):
                        continue
                matches.append(GuardrailMatch(kind=kind, start=m.start(), end=m.end(), text=m.group()))
        return matches

    def redact_match(self, match: GuardrailMatch) -> str:
        if self.redaction == "mask":
            tail = match.text[-4:] if len(match.text) > 4 else match.text
            return f"{'*' * max(len(match.text) - len(tail), 4)}{tail}"
        if self.redaction == "hash":
            digest = hashlib.sha256(match.text.encode()).hexdigest()[:8]
            return f"<{match.kind}_hash:{digest}>"
        return f"[REDACTED_{match.kind.upper()}]"
