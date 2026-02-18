from typing import Any, Literal

from .llm_base import LLMValidator


class PromptValidator(LLMValidator):
    """Prompt validator - checks if a statement/claim is true in the actual value using LLM.

    This validator asks the LLM to determine whether what is stated in `self.value` 
    (a claim or statement) is actually true based on the `actual_value` (the actual output/text).

    With negate=True, checks that the statement is NOT true in the actual value.

    Usage in YAML:
        output:
            prompt!: "The agent asked about the customer's typology"
    """

    name: Literal["prompt!"] = "prompt!"  # type: ignore

    system_prompt: str = """You are a factual validator assistant.
You receive a statement or claim and an actual text to validate against.

Your task: determine whether the statement is TRUE based on the actual text.

Guidelines:
- The statement describes something that should be present or true in the actual text
- You must verify that the actual text contains evidence supporting the statement
- Be precise: the statement must be verifiably true based on the actual text
- Consider the meaning and intent, not just exact word matches
- If the statement makes a claim about what happened, what was said, or what was asked, 
  verify that this is actually reflected in the actual text

Return:
- passes: true if the statement is verifiably true in the actual text
- passes: false if the statement cannot be verified or is contradicted by the actual text
"""

    def get_user_prompt(self, actual_value: Any) -> str:
        return f"""Statement to validate: {self.value}

Actual text to check:
{actual_value}

Is the statement true based on the actual text?"""
