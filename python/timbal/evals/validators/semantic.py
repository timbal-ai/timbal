from typing import Any, Literal

from .llm_base import LLMValidator


class SemanticValidator(LLMValidator):
    """Semantic validator - checks if text semantically matches expected meaning using LLM.

    With negate=True, checks that text does NOT semantically match the expected meaning.
    """

    name: Literal["semantic!"] = "semantic!"  # type: ignore

    system_prompt: str = """You are evaluating whether a piece of text semantically matches a given requirement or meaning.

Your task is to determine if the actual text conveys the same semantic meaning as the expected requirement,
even if the wording is different. Focus on the meaning and intent, not exact word matches."""

    def get_user_prompt(self, actual_value: Any) -> str:
        return f"""Expected semantic meaning: {self.value}

Actual text to evaluate:
{actual_value}

Does the actual text semantically match the expected meaning?"""
