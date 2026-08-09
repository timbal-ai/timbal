"""Deterministic secret/credential detection for model output and tool results."""

import re

from pydantic import Field

from ..types import Guardrail, GuardrailMatch, GuardrailStage

__all__ = ["RedactSecrets"]

_PATTERNS: dict[str, re.Pattern] = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "stripe_key": re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "bearer_token": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"),
    # key=value / key: value assignments with a high-entropy-looking literal
    "credential_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)\b"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9_/+.~-]{12,}['\"]?"
    ),
}


class RedactSecrets(Guardrail):
    """Redact API keys, tokens, private keys, and credential assignments.

    Defaults to model output and tool results — the two places secrets leak from
    (a tool reads an .env file; the model echoes a key it saw in context).
    """

    name: str = "redact_secrets"
    stages: set[GuardrailStage] = Field(
        default_factory=lambda: {GuardrailStage.MODEL_OUTPUT, GuardrailStage.TOOL_RESULT}
    )
    action: str = "redact"
    # PEM blocks and JWTs can be long; widen the stream holdback so they never split.
    scrub_window: int = 4096

    def detect(self, text: str) -> list[GuardrailMatch]:
        matches: list[GuardrailMatch] = []
        for kind, pattern in _PATTERNS.items():
            for m in pattern.finditer(text):
                matches.append(GuardrailMatch(kind=kind, start=m.start(), end=m.end(), text=m.group()))
        return matches

    def redact_match(self, match: GuardrailMatch) -> str:
        return f"[REDACTED_{match.kind.upper()}]"
