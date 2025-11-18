import functools
import json
import re

from ..core.agent import Agent
from ..errors import EvalError
from ..state import RunContext, set_run_context
from ..types.message import Message


def _extract_json_from_text(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks if present."""
    text = text.strip()
    
    # Try to find JSON in markdown code blocks (```json or ```)
    # Match everything between ```json or ``` and the closing ```
    json_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        # Extract the content inside the code block and try to parse it
        content = match.group(1).strip()
        # If it looks like JSON (starts with {), return it
        if content.startswith('{'):
            return content
    
    # Try to find JSON object directly (simple pattern for objects)
    # Match from first { to matching closing }
    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        if brace_count == 0:
            return text[start_idx:end_idx]
    
    # If no JSON found, return original text
    return text


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
    Input values can be:
    - Direct values: will check if the value is contained in the step input
    - Dicts with 'validators': will apply validators to the step input value
    If an input value is a list, it will match if any of the values in the list is found (OR logic).
    """
    if not isinstance(ref, list):
        raise ValueError(f"Invalid contains_steps validator: expected list, got {type(ref)}")

    def _check_value_match(step_val, expected_val):
        """Check if step_val contains expected_val. If expected_val is a list, match if any value matches."""
        if step_val is None:
            return False
        if isinstance(expected_val, list):
            # OR logic: match if any value in the list is found
            return any(str(v) in str(step_val) for v in expected_val)
        else:
            # Single value: check if it's contained
            return str(expected_val) in str(step_val)
    
    def _apply_validators_to_value(step_val, validators_spec):
        """Apply validators to a step input value. Returns True if all validators pass."""
        from ..types.content import TextContent
        from ..types.message import Message
        
        if not isinstance(validators_spec, dict):
            return False
        
        # Create a message with the step value for validation
        validation_message = Message(role="user", content=[TextContent(text=str(step_val) if step_val is not None else "")])
        
        # Process each validator (using functions defined in this module)
        for validator_name, validator_arg in validators_spec.items():
            if validator_name == "contains":
                # Apply contains validator
                validator = contains_output(validator_arg)
                try:
                    validator(validation_message)
                except EvalError:
                    return False
            elif validator_name == "not_contains":
                validator = not_contains_output(validator_arg)
                try:
                    validator(validation_message)
                except EvalError:
                    return False
            elif validator_name == "equals":
                validator = equals(validator_arg)
                try:
                    validator(validation_message)
                except EvalError:
                    return False
            elif validator_name == "regex":
                validator = regex(validator_arg)
                try:
                    validator(validation_message)
                except EvalError:
                    return False
            # Add more validators as needed
            else:
                # Unknown validator, skip
                continue
        
        return True

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
                        
                        # Check if v is a dict with validators
                        if isinstance(v, dict) and "validators" in v:
                            # Apply validators to the step value
                            if not _apply_validators_to_value(step_val, v["validators"]):
                                all_match = False
                                break
                        else:
                            # Direct value matching
                            if not _check_value_match(step_val, v):
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
    If an input value is a list, it will match if any of the values in the list is found (OR logic).
    """
    if not isinstance(ref, list):
        raise ValueError(f"Invalid not_contains_steps validator: expected list, got {type(ref)}")

    def _check_value_match(step_val, expected_val):
        """Check if step_val contains expected_val. If expected_val is a list, match if any value matches."""
        if step_val is None:
            return False
        if isinstance(expected_val, list):
            # OR logic: match if any value in the list is found
            return any(str(v) in str(step_val) for v in expected_val)
        else:
            # Single value: check if it's contained
            return str(expected_val) in str(step_val)

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
                        if not _check_value_match(step_val, v):
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


def contains_output(ref: str | list[str]):
    """Validate that the message contains the given substrings (case-insensitive)."""
    if not isinstance(ref, list):
        ref = [ref]
    
    ref = [str(v) for v in ref]

    def validator(message: Message):
        message_text = message.collect_text()
        if not message_text:
            raise EvalError("Message does not contain any text to validate.")

        # Case-insensitive comparison
        message_text_lower = message_text.lower()
        for v in ref:
            if v.lower() not in message_text_lower:
                raise EvalError(f"Message does not contain '{v}'.")

    return Validator(validator, "contains", ref)


def not_contains_output(ref: str | list[str]):
    """Validate that the message does not contain the given substrings."""
    if not isinstance(ref, list):
        ref = [ref]
    
    ref = [str(v) for v in ref]

    def validator(message: Message):
        message_text = message.collect_text()
        if not message_text:
            raise EvalError("Message does not contain any text to validate.")

        for v in ref:
            if v in message_text:
                raise EvalError(f"Message contains '{v}'.")

    return Validator(validator, "not_contains", ref)


def equals(ref: str):
    """Validate that the message exactly matches the given text."""
    if not isinstance(ref, str):
        raise ValueError(f"Invalid equals validator: expected str, got {type(ref)}")

    def validator(message: Message):
        message_text = message.collect_text()
        if not message_text:
            raise EvalError("Message does not contain any text to validate.")
        
        if message_text.strip() != ref.strip():
            raise EvalError(f"Message does not exactly match expected output. Expected: '{ref}', Got: '{message_text}'")

    return Validator(validator, "equals", ref)


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
Be generous in your evaluation - if the output reasonably addresses the reference, even if it's not a perfect match, consider it valid.
Do not penalize for paraphrasing, extra detail, or reasonable conversational steps if they help address the user's need.
Only mark as incorrect if the response is clearly irrelevant, unhelpful, or fails to address the user's request.

You must respond with a valid JSON object containing:
- "is_valid": boolean indicating if the output is correct
- "explanation": string explaining your reasoning
"""

        # Newlines in f-strings were introduced in python >= 3.12
        prompt = "<output>\n" + str(message_text) + "\n</output>\n\n"
        prompt += "\n".join(["<reference>\n" + str(v) + "\n</reference>\n" for v in ref])

        # Create agent in fresh context without parent
        set_run_context(RunContext())
        agent = Agent(
            name="SemanticOutputValidator",
            model="openai/gpt-4.1-mini",
            system_prompt=system_prompt,
        )
        res = await agent(prompt=Message.validate(prompt)).collect()
        if not res.output or not res.output.content or not res.output.content[0].text:
            raise EvalError("Semantic validator agent failed to generate a response. This may be due to missing API key or other configuration issues.")
        
        res_text = res.output.content[0].text
        try:
            # Extract JSON from markdown code blocks if present
            json_text = _extract_json_from_text(res_text)
            res = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise EvalError(f"Semantic validator agent returned invalid JSON: {res_text}. Error: {str(e)}")

        if not res["is_valid"]:
            raise EvalError(res["explanation"])

    return Validator(validator, "semantic", ref)


def contains_any_output(ref: str | list[str]):
    """Validate that the message contains any of the given substrings (OR logic)."""
    if not isinstance(ref, list):
        ref = [ref]
    
    ref = [str(v) for v in ref]

    def validator(message: Message):
        message_text = message.collect_text()
        if not message_text:
            raise EvalError("Message does not contain any text to validate.")

        for v in ref:
            if v in message_text:
                return  # Found at least one, validation passes

        # If we get here, none were found
        ref_str = "', '".join(ref)
        raise EvalError(f"Message does not contain any of: '{ref_str}'.")

    return Validator(validator, "contains_any", ref)


def contains_any_steps(ref: list[dict]):
    """
    Validate that the steps contain any of the given tool names and, optionally, input key-value substrings (OR logic).
    Each ref item should be a dict with 'name' (tool name) and optionally 'input' (a dict of expected input substrings).
    If an input value is a list, it will match if any of the values in the list is found (OR logic).
    """
    if not isinstance(ref, list):
        raise ValueError(f"Invalid contains_any_steps validator: expected list, got {type(ref)}")

    def _check_value_match(step_val, expected_val):
        """Check if step_val contains expected_val. If expected_val is a list, match if any value matches."""
        if step_val is None:
            return False
        if isinstance(expected_val, list):
            # OR logic: match if any value in the list is found
            return any(str(v) in str(step_val) for v in expected_val)
        else:
            # Single value: check if it's contained
            return str(expected_val) in str(step_val)

    def validator(steps: list[dict]):
        for expected in ref:
            tool_name = expected.get("name")
            input_dict = expected.get("input", {})
            
            for step in steps:
                if step.get("tool") == tool_name:
                    # If no input dict specified, just match tool
                    if not input_dict:
                        return  # Found a match, validation passes
                    # Otherwise, check all input keys/values
                    step_input = step.get("input", {})
                    all_match = True
                    for k, v in input_dict.items():
                        step_val = step_input.get(k)
                        if not _check_value_match(step_val, v):
                            all_match = False
                            break
                    if all_match:
                        return  # Found a match, validation passes
        
        # If we get here, none were found
        tool_names = [expected.get("name") for expected in ref]
        tools_str = "', '".join(tool_names)
        raise EvalError(f"No step found with any of the tools: '{tools_str}'.")
    
    return Validator(validator, "contains_any_steps", ref)


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

You must respond with a valid JSON object containing:
- "is_valid": boolean indicating if the steps are correct
- "explanation": string explaining your reasoning
"""

        # Newlines in f-strings were introduced in python >= 3.12
        prompt = "<steps>\n" + str(steps_text) + "\n</steps>\n\n"
        prompt += "\n".join(["<reference>\n" + str(v) + "\n</reference>\n" for v in ref])

        # Create agent in fresh context without parent
        set_run_context(RunContext())
        agent = Agent(
            name="SemanticStepsValidator",
            model="openai/gpt-4.1-mini",
            system_prompt=system_prompt,
        )
        res = await agent(prompt=Message.validate(prompt)).collect()
        if not res.output or not res.output.content or not res.output.content[0].text:
            raise EvalError("Semantic validator agent failed to generate a response. This may be due to missing API key or other configuration issues.")
        
        res_text = res.output.content[0].text
        try:
            # Extract JSON from markdown code blocks if present
            json_text = _extract_json_from_text(res_text)
            res = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise EvalError(f"Semantic validator agent returned invalid JSON: {res_text}. Error: {str(e)}")

        if not res["is_valid"]:
            raise EvalError(res["explanation"])

    return Validator(validator, "semantic", ref)


