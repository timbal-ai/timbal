import functools
import json
import re

from ..core.agent import Agent
from ..errors import EvalError
from ..state import RunContext, get_run_context, set_run_context
from ..types.content import TextContent
from ..types.message import Message


def _extract_json_from_text(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks if present."""
    text = text.strip()
    
    json_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content.startswith('{'):
            return content
    
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
    
    return text


def _find_matching_model_name(model_prefix: str, model_names: set[str]) -> tuple[str | None, str | None]:
    """Find matching model name from a set using exact matching or prefix matching for versioned models.
    
    Returns (matched_model_name, error_message). If error_message is not None, it indicates
    an ambiguous match (multiple unrelated models).
    """
    if model_prefix in model_names:
        return model_prefix, None
    
    if model_prefix.endswith("-latest"):
        base_prefix = model_prefix[:-7]
        matching_models = [m for m in model_names if m.startswith(base_prefix + "-") or m == base_prefix]
        if not matching_models:
            return None, None
        if len(matching_models) == 1:
            return matching_models[0], None
        matching_models.sort(key=lambda x: (-len(x), x))
        return matching_models[0], None
    
    matching_models = []
    for model in model_names:
        if model.startswith(model_prefix + "-"):
            matching_models.append(model)
    
    if not matching_models:
        return None, None
    if len(matching_models) == 1:
        return matching_models[0], None
    
    unrelated_models = []
    for m1 in matching_models:
        is_related = False
        for m2 in matching_models:
            if m1 != m2:
                if (m1.startswith(m2 + "-") or m2.startswith(m1 + "-")):
                    is_related = True
                    break
        if not is_related:
            unrelated_models.append(m1)
    
    def get_priority(model: str) -> tuple[int, int]:
        suffix = model[len(model_prefix) + 1:]
        component_count = suffix.count("-") + 1
        return (component_count, -len(model))
    
    if len(unrelated_models) > 1:
        unrelated_models.sort(key=get_priority)
        min_components = get_priority(unrelated_models[0])[0]
        most_direct = [m for m in unrelated_models if get_priority(m)[0] == min_components]
        if len(most_direct) > 1:
            return None, f"Multiple models match prefix '{model_prefix}': {', '.join(most_direct)}"
        return most_direct[0], None
    
    matching_models.sort(key=get_priority)
    return matching_models[0], None


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
            step_validators = expected.get("validators", {})
            found = False
            matched_step = None
            for step in steps:
                if step.get("tool") == tool_name:
                    # If no input dict specified, just match tool
                    if not input_dict:
                        found = True
                        matched_step = step
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
                        matched_step = step
                        break
            if not found:
                if input_dict:
                    raise EvalError(f"No step found with tool '{tool_name}' and input containing {input_dict}.")
                else:
                    raise EvalError(f"No step found with tool '{tool_name}'.")
            
            if step_validators and matched_step:
                run_context = get_run_context()
                if run_context and run_context._trace:
                    # Find the step's span in the trace to get its execution time
                    step_tool_name = matched_step.get("tool")
                    step_execution_time = None
                    
                    for call_id, span in run_context._trace.items():
                        if span.path and "." in span.path:
                            path_parts = span.path.split(".")
                            if len(path_parts) >= 2 and path_parts[1] == step_tool_name:
                                # Found the step's span, calculate its execution time
                                if span.t0 is not None and span.t1 is not None:
                                    step_execution_time = (span.t1 - span.t0) / 1000.0  # Convert to seconds
                                    break
                    
                    for validator_name, validator_arg in step_validators.items():
                        if validator_name == "time":
                            max_value = validator_arg.get("max")
                            min_value = validator_arg.get("min")
                            
                            if step_execution_time is not None:
                                if max_value is not None:
                                    try:
                                        max_value = float(max_value)
                                    except (ValueError, TypeError):
                                        raise ValueError(f"time.max must be a number, got {type(max_value)}")
                                    if step_execution_time >= max_value:
                                        raise EvalError(f"Step '{tool_name}' execution time {step_execution_time:.3f}s is greater than or equal to max value {max_value}s.")
                                
                                if min_value is not None:
                                    try:
                                        min_value = float(min_value)
                                    except (ValueError, TypeError):
                                        raise ValueError(f"time.min must be a number, got {type(min_value)}")
                                    if step_execution_time <= min_value:
                                        raise EvalError(f"Step '{tool_name}' execution time {step_execution_time:.3f}s is less than or equal to min value {min_value}s.")
                            else:
                                raise EvalError(f"Cannot validate time for step '{tool_name}': step execution time not available in trace.")
                        
                        elif validator_name == "usage":
                            step_usage = None
                            for call_id, span in run_context._trace.items():
                                if span.path and "." in span.path:
                                    path_parts = span.path.split(".")
                                    if len(path_parts) >= 2 and path_parts[1] == step_tool_name:
                                        # Found the step's span, get its usage
                                        step_usage = span.usage or {}
                                        break
                            
                            # Tools like web_search register their usage in the agent/llm span via update_usage.
                            if step_usage is None or not step_usage:
                                usage_keys_to_find = list(validator_arg.keys())
                                for call_id, span in run_context._trace.items():
                                    if span.usage:
                                        for usage_key in usage_keys_to_find:
                                            found = False
                                            if usage_key in span.usage:
                                                found = True
                                            else:
                                                # Try prefix matching - check if any key in span.usage matches
                                                # e.g., "gpt-4.1-mini:web_search_requests" matches
                                                if ":" in usage_key:
                                                    prefix, usage_type = usage_key.split(":", 1)
                                                    matching_keys = [k for k in span.usage.keys() if k.endswith(":" + usage_type) or k == usage_type]
                                                    if matching_keys:
                                                        found = True
                                            if found:
                                                # Found a span with at least one matching usage key - use it
                                                step_usage = span.usage or {}
                                                break
                                        if step_usage:
                                            break
                            
                            if step_usage is None or not step_usage:
                                raise EvalError(f"Cannot validate usage for step '{tool_name}': step usage not available in trace.")
                            
                            # Validate each usage constraint
                            for usage_key, limits in validator_arg.items():
                                if not isinstance(limits, dict):
                                    raise ValueError(f"Invalid usage limits for '{usage_key}': expected dict, got {type(limits)}")
                                
                                max_value = limits.get("max")
                                min_value = limits.get("min")
                                equals_value = limits.get("equals")
                                
                                if max_value is not None:
                                    try:
                                        max_value = float(max_value)
                                    except (ValueError, TypeError):
                                        raise ValueError(f"usage.max must be a number for '{usage_key}', got {type(max_value)}")
                                
                                if min_value is not None:
                                    try:
                                        min_value = float(min_value)
                                    except (ValueError, TypeError):
                                        raise ValueError(f"usage.min must be a number for '{usage_key}', got {type(min_value)}")
                                
                                if equals_value is not None:
                                    try:
                                        equals_value = float(equals_value)
                                    except (ValueError, TypeError):
                                        raise ValueError(f"usage.equals must be a number for '{usage_key}', got {type(equals_value)}")
                                
                                if max_value is None and min_value is None and equals_value is None:
                                    raise ValueError(f"usage constraint for '{usage_key}' must have at least one of 'max', 'min', or 'equals' keys")
                                
                                # Parse usage_key and find matching model/tool
                                # For step-specific validation, we support:
                                # 1. "usage_type" - just the usage type (e.g., "api_calls")
                                # 2. "model:usage_type" - with model prefix (e.g., "gpt-4.1-mini:input_text_tokens")
                                # 3. "tool_name:usage_type" - with tool prefix (e.g., "my_tool:api_calls")
                                if ":" in usage_key:
                                    prefix, usage_type = usage_key.split(":", 1)
                                    full_usage_key = usage_key
                                    actual_value = step_usage.get(full_usage_key)
                                    
                                    if actual_value is None:
                                        model_names = set()
                                        for k in step_usage.keys():
                                            if ":" in k and k.endswith(":" + usage_type):
                                                model_names.add(k.split(":")[0])
                                        
                                        matched_model, match_error = _find_matching_model_name(prefix, model_names)
                                        if match_error:
                                            raise EvalError(f"Cannot validate usage for '{usage_key}': {match_error}")
                                        if matched_model:
                                            full_usage_key = f"{matched_model}:{usage_type}"
                                            actual_value = step_usage.get(full_usage_key)
                                    
                                    if actual_value is None:
                                        available_models_str = ", ".join(sorted(model_names)) if model_names else "none"
                                        raise EvalError(f"Cannot validate usage for '{usage_key}': no matching usage found in step usage data. Expected model prefix '{prefix}'. Available models: {available_models_str}")
                                    
                                    if "+" in usage_type and actual_value is None:
                                        matched_model, match_error = _find_matching_model_name(prefix, model_names)
                                        if match_error:
                                            raise EvalError(f"Cannot validate usage for '{usage_key}': {match_error}")
                                        if matched_model:
                                            keys = usage_type.split("+")
                                            actual_value = sum(step_usage.get(f"{matched_model}:{k}", 0) for k in keys)
                                            full_usage_key = f"{matched_model}:{usage_type}"
                                        else:
                                            raise EvalError(f"Cannot validate usage for '{usage_key}': cannot aggregate without matching prefix.")
                                else:
                                    raise EvalError(f"Invalid usage key '{usage_key}': must be in format 'model:usage_type' (e.g., 'gpt-4o-mini:web_search_requests')")
                                
                                if actual_value is None:
                                    raise EvalError(f"Cannot validate usage for '{usage_key}': usage value not found in step usage data.")
                                
                                # Validate constraints
                                if equals_value is not None:
                                    if actual_value != equals_value:
                                        raise EvalError(f"Step '{tool_name}' usage {full_usage_key} value {actual_value} is not equal to expected value {equals_value}.")
                                
                                if max_value is not None and actual_value > max_value:
                                    raise EvalError(f"Step '{tool_name}' usage {full_usage_key} value {actual_value} is greater than max value {max_value}.")
                                
                                if min_value is not None and actual_value < min_value:
                                    raise EvalError(f"Step '{tool_name}' usage {full_usage_key} value {actual_value} is less than min value {min_value}.")
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


def time(ref: dict):
    """Validate that the execution time meets the specified criteria.
    
    Works for input, output, and steps validators. The execution time is measured
    for the entire turn and stored in run_context.
    
    Args:
        ref: Dictionary with validation criteria. Supported keys:
            - "max": float - maximum execution time in seconds
            - "min": float - minimum execution time in seconds
    
    Examples:
        time:
          max: 5.0  # execution time must be < 5 seconds
        time:
          min: 1.0  # execution time must be > 1 second
        time:
          max: 5.0
          min: 1.0  # execution time must be between 1 and 5 seconds
    """
    from ..state import get_run_context
    
    if not isinstance(ref, dict):
        raise ValueError(f"Invalid time validator: expected dict, got {type(ref)}")
    
    max_value = ref.get("max")
    min_value = ref.get("min")
    
    if max_value is not None:
        try:
            max_value = float(max_value)
        except (ValueError, TypeError):
            raise ValueError(f"time.max must be a number, got {type(max_value)}")
    
    if min_value is not None:
        try:
            min_value = float(min_value)
        except (ValueError, TypeError):
            raise ValueError(f"time.min must be a number, got {type(min_value)}")
    
    if max_value is None and min_value is None:
        raise ValueError("time validator must have at least one of 'max' or 'min' keys")
    
    def validator(value):
        # Get execution time from run_context
        # value can be Message (for input/output) or list (for steps)
        run_context = get_run_context()
        if not run_context:
            raise EvalError("Cannot validate time: no run context available.")
        
        # If value is a list, it's being used in steps validation - use steps time
        # Otherwise, use the total turn execution time
        if isinstance(value, list):
            execution_time = getattr(run_context, "_last_steps_execution_time", None)
            if execution_time is None:
                raise EvalError("Cannot validate time: steps execution time not available in run context.")
        else:
            execution_time = getattr(run_context, "_last_execution_time", None)
            if execution_time is None:
                raise EvalError("Cannot validate time: execution time not available in run context.")
        
        if max_value is not None and execution_time >= max_value:
            raise EvalError(f"Execution time {execution_time:.3f}s is greater than or equal to max value {max_value}s.")
        
        if min_value is not None and execution_time <= min_value:
            raise EvalError(f"Execution time {execution_time:.3f}s is less than or equal to min value {min_value}s.")
    
    return Validator(validator, "time", ref)


def usage(ref: dict):
    """Validate that the usage (tokens, API calls, tool metrics, etc.) meets the specified criteria.
    
    Works for input, output, and step-specific validators. The usage is retrieved from run_context
    or from a specific step's span in the trace.
    
    Args:
        ref: Dictionary with usage constraints. Keys are usage identifiers like:
            - "model:input_text_tokens" - text tokens in input
            - "model:output_text_tokens" - text tokens in output
            - "model:input_audio_tokens" - audio tokens in input
            - "model:output_audio_tokens" - audio tokens in output
            - "model:input_cached_tokens" - cached tokens in input
            - "model:web_search_requests" - number of web search requests (not tokens, but requests)
            - "model:tool_name_requests" - custom tool usage metrics (if tool calls update_usage)
            - "model:usage_type" (supports prefix matching for model names)
            Values are dicts with "max", "min", and/or "equals" keys.
    
    Note: Some usage types like "web_search_requests" track requests/calls, not tokens.
    
    Note: Files (images, documents) are converted to tokens, so there's no separate
    "files" usage type. File usage is included in the token counts.
    
    Note: Usage validation at `steps.validators` level (top-level) is not recommended
    because steps can include different types of usage (LLM tokens, tool requests, etc.)
    and it's unclear what should be included in the total steps usage. Instead, validate
    usage for specific steps within `contains_steps` validators.
    
    Tools can track their own usage by calling `run_context.update_usage(key, value)`.
    The usage will be stored in the tool's span and can be validated for specific steps.
    
    Examples:
        # Input usage validation
        input:
          validators:
            usage:
              "gpt-4.1-mini:input_text_tokens":
                max: 1000
        
        # Output usage validation
        output:
          validators:
            usage:
              "gpt-4.1-mini:output_text_tokens":
                max: 500
        
        # Step-specific usage validation (recommended)
        steps:
          validators:
            contains:
              - name: web_search
                validators:
                  usage:
                    # Must use format "model:usage_type"
                    "gpt-4o-mini:web_search_requests":
                      equals: 1  # Exactly 1 request
                    # For LLM token usage
                    "gpt-4o-mini:input_text_tokens":
                      max: 50
    """
    if not isinstance(ref, dict):
        raise ValueError(f"Invalid usage validator: expected dict, got {type(ref)}")
    
    for usage_key, limits in ref.items():
        if not isinstance(limits, dict):
            raise ValueError(f"Invalid usage limits for '{usage_key}': expected dict, got {type(limits)}")
        
        # Require format "model:usage_type" - must contain a colon
        if ":" not in usage_key:
            raise ValueError(f"Invalid usage key '{usage_key}': must be in format 'model:usage_type' (e.g., 'gpt-4o-mini:web_search_requests')")
        
        max_value = limits.get("max")
        min_value = limits.get("min")
        equals_value = limits.get("equals")
        
        if max_value is None and min_value is None and equals_value is None:
            raise ValueError(f"usage constraint for '{usage_key}' must have at least one of 'max', 'min', or 'equals' keys")
    
    def _find_matching_model(model_prefix: str, actual_usage: dict) -> tuple[str | None, str | None]:
        """Find matching model in actual_usage using exact matching or prefix matching for versioned models."""
        if not actual_usage:
            return None, None
        
        model_names = set()
        for key in actual_usage.keys():
            if ":" in key:
                model_names.add(key.split(":")[0])
        
        matched_model, match_error = _find_matching_model_name(model_prefix, model_names)
        return matched_model, match_error
    
    def validator(value):
        run_context = get_run_context()
        if not run_context:
            raise EvalError("Cannot validate usage: no run context available.")
        
        if isinstance(value, list):
            actual_usage = getattr(run_context, "_last_steps_usage", None)
            if actual_usage is None:
                raise EvalError("Cannot validate usage: steps usage data not available in run context.")
        else:
            actual_usage = getattr(run_context, "_last_usage", {})
            if not actual_usage:
                raise EvalError("Cannot validate usage: usage data not available in run context.")
        
        for usage_key, limits in ref.items():
            max_value = limits.get("max")
            min_value = limits.get("min")
            equals_value = limits.get("equals")
            
            if max_value is not None:
                try:
                    max_value = float(max_value)
                except (ValueError, TypeError):
                    raise ValueError(f"usage.max must be a number for '{usage_key}', got {type(max_value)}")
            
            if min_value is not None:
                try:
                    min_value = float(min_value)
                except (ValueError, TypeError):
                    raise ValueError(f"usage.min must be a number for '{usage_key}', got {type(min_value)}")
            
            if equals_value is not None:
                try:
                    equals_value = float(equals_value)
                except (ValueError, TypeError):
                    raise ValueError(f"usage.equals must be a number for '{usage_key}', got {type(equals_value)}")
            
            # Must be in format "model:usage_type"
            if ":" not in usage_key:
                raise EvalError(f"Invalid usage key '{usage_key}': must be in format 'model:usage_type' (e.g., 'gpt-4o-mini:web_search_requests')")
            
            model_prefix, usage_type = usage_key.split(":", 1)
            # Find matching model using exact matching
            matched_model, prefix_error = _find_matching_model(model_prefix, actual_usage)
            
            if prefix_error:
                raise EvalError(f"Cannot validate usage for '{usage_key}': {prefix_error}")
            
            if matched_model is None:
                raise EvalError(f"Cannot validate usage for '{usage_key}': no matching model found in usage data. Expected model prefix '{model_prefix}'.")
            
            # Use the matched model name
            full_usage_key = f"{matched_model}:{usage_type}"
            
            # Get actual value
            if "+" in usage_type:
                # Sum multiple usage types
                keys = usage_type.split("+")
                actual_value = sum(actual_usage.get(f"{matched_model}:{k}", 0) for k in keys)
            else:
                actual_value = actual_usage.get(full_usage_key)
            
            if actual_value is None:
                raise EvalError(f"Cannot validate usage for '{usage_key}': usage value not found in usage data.")
            
            # Validate constraints
            if equals_value is not None:
                if actual_value != equals_value:
                    raise EvalError(f"Usage {full_usage_key} value {actual_value} is not equal to expected value {equals_value}.")
            
            if max_value is not None and actual_value > max_value:
                raise EvalError(f"Usage {full_usage_key} value {actual_value} is greater than max value {max_value}.")
            
            if min_value is not None and actual_value < min_value:
                raise EvalError(f"Usage {full_usage_key} value {actual_value} is less than min value {min_value}.")
    
    return Validator(validator, "usage", ref)


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


