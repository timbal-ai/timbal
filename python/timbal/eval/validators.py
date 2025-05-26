import functools
import json
import re

from ..errors import EvalError
from ..steps.llms.router import llm_router
from ..types.message import Message


class Validator:
    """Wraps a validator function to provide a better __repr__ and unified interface.
    This class is intended only for internal use.
    No validation is performed on the arguments.
    """

    def __init__(self, func, name, ref):
        functools.update_wrapper(self, func)
        self.func = func
        self.name = name
        self.ref = ref

    
    def __call__(self, message):
        """The caller is responsible for awaiting if the function is async."""
        return self.func(message)

    
    def __repr__(self):
        """Return a string representation of the validator."""
        ref_repr = repr(self.ref)
        # Truncate long reference strings.
        if len(ref_repr) > 50:
            ref_repr = ref_repr[:47] + "..."
        return f"<Validator(name={self.name}, ref={ref_repr})>"


def contains(ref: list[str]):
    """Validate that the message contains the given substrings."""
    if not isinstance(ref, list):
        raise ValueError(f"Invalid contains validator: expected list, got {type(ref)}")

    ref = [str(v) for v in ref]

    def validator(message: Message):
        message_text = message.collect_text()
        if not message_text:
            raise EvalError("Message does not contain any text to validate.")

        for v in ref:
            if v not in message_text:
                raise EvalError(f"Message does not contain '{v}'.")

    return Validator(validator, "contains", ref)


def regex(ref: str):
    """Validate that the message matches the given regex pattern."""
    try:
        compiled = re.compile(ref)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{ref}': {e}") from e

    def validator(message: Message):
        message_text = message.collect_text()
        if not message_text:
            raise EvalError("Message does not contain any text to validate.")

        if not compiled.search(message_text):
            raise EvalError(f"Message does not match regex '{ref}'.")

    return Validator(validator, "regex", ref)


def semantic_output(ref: str | list[str]):
    """Validate that the message matches the given semantic content."""
    if not isinstance(ref, list):
        ref = [ref]

    ref = [str(v) for v in ref]

    async def validator(message: Message):
        message_text = message.collect_text()
        if not message_text:
            raise EvalError("Message does not contain any text to validate.")

        system_prompt = """You are a helpful assistant that evaluates if an output is semantically correct w.r.t. a set of valid responses.
The output should be considered correct if it is a helpful, relevant, and contextually appropriate reply to the user's request, and if it covers the key information or question present in the reference answer.
Do not penalize for paraphrasing, extra detail, or reasonable conversational steps if they help address the user's need.
Only mark as incorrect if the response is irrelevant, unhelpful, or fails to address the user's request.
"""

        prompt = f"""<output>
{message_text}
</output>

{"\n".join([f"<reference>\n{v}\n</reference>\n" for v in ref])}
"""
        messages = [Message.validate(prompt)]

        json_schema = {
            "name": "SemanticEval",
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"}, 
                    "is_valid": {"type": "boolean"}
                }, 
            }
        }

        res = await llm_router(
            model="gpt-4.1",
            system_prompt=system_prompt,
            messages=messages,
            json_schema=json_schema,
        )
        res = res.choices[0].message.content
        res = json.loads(res)

        if not res["is_valid"]:
            raise EvalError(res["explanation"])

    return Validator(validator, "semantic", ref)


def semantic_steps(ref: str | list[str]):
    """Validate that the message matches the given semantic content."""
    if not isinstance(ref, list):
        ref = [ref]

    ref = [str(v) for v in ref]

    async def validator(message: Message):
        message_text = message.collect_text()
        if not message_text:
            raise EvalError("Message does not contain any text to validate.")

        system_prompt = """You are a helpful assistant that evaluates if a sequence of steps (actions or tool uses) is semantically correct with respect to a set of reference steps.
The steps should be considered correct if they are relevant, logically ordered, and cover the key actions or information present in the reference steps, even if the wording or exact details differ.
Do not penalize for reasonable variations, extra helpful steps, or minor differences in order, as long as the essential actions are present and the user's need is addressed.
Only mark as incorrect if the steps are missing key actions, are irrelevant, or fail to address the user's request as described in the reference.
"""

        prompt = f"""<steps>
{message_text}
</steps>

{"\n".join([f"<reference>\n{v}\n</reference>\n" for v in ref])}
"""
        messages = [Message.validate(prompt)]

        json_schema = {
            "name": "SemanticEval",
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"}, 
                    "is_valid": {"type": "boolean"}
                }, 
            }
        }

        res = await llm_router(
            model="gpt-4.1",
            system_prompt=system_prompt,
            messages=messages,
            json_schema=json_schema,
        )
        res = res.choices[0].message.content
        res = json.loads(res)

        if not res["is_valid"]:
            raise EvalError(res["explanation"])

    return Validator(validator, "semantic", ref)


