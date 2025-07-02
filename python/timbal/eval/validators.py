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


def contains_steps(ref: list[dict]):
    """
    Validate that the steps contain the given tool names and, optionally, input key-value substrings.
    Each ref item should be a dict with 'name' (tool name) and optionally 'input' (a dict of expected input substrings).
    """
    if not isinstance(ref, list):
        raise ValueError(f"Invalid contains_steps validator: expected list, got {type(ref)}")

    def validator(steps: list[dict]):
        for expected in ref:
            tool_name = expected.get("name")
            input_dict = expected.get("input", {})
            found = False
            for step in steps:
                if step.get("tool") == tool_name:
                    # If no input dict specified, just match tool
                    if not input_dict:
                        found = True
                        break
                    # Otherwise, check all input keys/values
                    step_input = step.get("input", {})
                    all_match = True
                    for k, v in input_dict.items():
                        step_val = step_input.get(k)
                        if step_val is None or str(v) not in str(step_val):
                            all_match = False
                            break
                    if all_match:
                        found = True
                        break
            if not found:
                if input_dict:
                    raise EvalError(f"No step found with tool '{tool_name}' and input containing {input_dict}.")
                else:
                    raise EvalError(f"No step found with tool '{tool_name}'.")
    return Validator(validator, "contains_steps", ref)


def not_contains_steps(ref: list[dict]):
    """
    Validate that the steps do not contain the given tool names and, optionally, input key-value substrings.
    Each ref item should be a dict with 'name' (tool name) and optionally 'input' (a dict of expected input substrings).
    """
    if not isinstance(ref, list):
        raise ValueError(f"Invalid not_contains_steps validator: expected list, got {type(ref)}")

    def validator(steps: list[dict]):
        for expected in ref:
            tool_name = expected.get("name")
            # Handle input parameters the same way as contains_steps
            input_dict = expected.get("input", {})
            # If no explicit input dict, treat all other keys as input parameters
            if not input_dict:
                input_dict = {k: v for k, v in expected.items() if k != "name"}
            
            found = False
            for step in steps:
                if step.get("tool") == tool_name:
                    # If no input dict specified, just match tool
                    if not input_dict:
                        found = True
                        break
                    # Otherwise, check all input keys/values
                    step_input = step.get("input", {})
                    all_match = True
                    for k, v in input_dict.items():
                        step_val = step_input.get(k)
                        if step_val is None or str(v) not in str(step_val):
                            all_match = False
                            break
                    if all_match:
                        found = True
                        break
            if found:
                if input_dict:
                    raise EvalError(f"Step found with tool '{tool_name}' and input containing {input_dict}.")
                else:
                    raise EvalError(f"Step found with tool '{tool_name}'.")
    return Validator(validator, "not_contains_steps", ref)


def contains_output(ref: list[str]):
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


def not_contains_output(ref: list[str]):
    """Validate that the message contains the given substrings."""
    if not isinstance(ref, list):
        raise ValueError(f"Invalid contains validator: expected list, got {type(ref)}")

    ref = [str(v) for v in ref]

    def validator(message: Message):
        message_text = message.collect_text()
        if not message_text:
            raise EvalError("Message does not contain any text to validate.")

        for v in ref:
            if v in message_text:
                raise EvalError(f"Message contains '{v}'.")

    return Validator(validator, "not_contains", ref)


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
    """Validate that the steps sequence matches the given semantic content."""
    if not isinstance(ref, list):
        ref = [ref]

    ref = [str(v) for v in ref]

    async def validator(steps: list[dict]):
        # Convert steps list to text representation
        if not steps:
            steps_text = "No steps were executed."
        else:
            steps_text = "\n".join([f"- Tool: {step.get('tool', 'unknown')}, Input: {step.get('input', {})}" for step in steps])

        system_prompt = """You are a helpful assistant that evaluates if a sequence of steps (actions or tool uses) is semantically correct with respect to a set of reference steps.
The steps should be considered correct if they are relevant, logically ordered, and cover the key actions or information present in the reference steps, even if the wording or exact details differ.
Do not penalize for reasonable variations, extra helpful steps, or minor differences in order, as long as the essential actions are present and the user's need is addressed.
Note that no steps can be correct if the reference indicates no tools should be used.
Only mark as incorrect if the steps are missing key actions, are irrelevant, or fail to address the user's request as described in the reference.
"""

        prompt = f"""<steps>
{steps_text}
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


