from typing import Any, Literal

from .llm_base import LLMValidator


class LanguageValidator(LLMValidator):
    """Language validator - checks if text is in the expected language using LLM."""

    name: Literal["language!"] = "language!"  # type: ignore

    system_prompt: str = """You are evaluating whether a piece of text is written in a specific language.

Analyze the text and determine what language it is written in.
Compare it to the expected language."""

    def get_user_prompt(self, actual_value: Any) -> str:
        return f"""Expected language: {self.value}

Text to evaluate:
{actual_value}

Is this text written in {self.value}?"""
