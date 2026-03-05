import inspect
from typing import Any, Literal, cast

import pydantic

# Type definitions from anthropic SDK for schema transformation
SupportedTypes = Literal["object", "array", "string", "number", "integer", "boolean", "null"]
SupportedStringFormats = Literal["date", "date-time", "email", "hostname", "ipv4", "ipv6", "uri", "uuid"]


def is_list(obj: object) -> bool:
    """Helper to check if an object is a list."""
    return isinstance(obj, list)


def assert_never(value: object) -> None:
    """Helper for exhaustiveness checking."""
    raise AssertionError(f"Unexpected value: {value}")


def transform_schema(
    json_schema: type[pydantic.BaseModel] | dict[str, Any],
) -> dict[str, Any]:
    """
    Transforms a JSON schema to ensure compatibility with OpenAI strict mode.

    This is a modified version of anthropic.transform_schema that enforces
    OpenAI's structured output requirements:
    - All object properties are marked as required (strict mode requirement)
    - additionalProperties is set to false
    - Converts oneOf to anyOf (OpenAI doesn't support allOf but supports anyOf)
    - Rejects unsupported composition keywords (allOf, not, etc.)

    Modified from: anthropic.transform_schema
    Original: https://github.com/anthropics/anthropic-sdk-python

    OpenAI Structured Outputs Requirements:
    - All fields must be in the 'required' array when strict=true
    - additionalProperties must be false
    - Root object cannot use anyOf
    - allOf, not, dependentRequired, dependentSchemas, if/then/else not supported

    References:
    - https://platform.openai.com/docs/guides/structured-outputs
    - https://docs.anthropic.com/en/docs/build-with-claude/structured-output

    Args:
        json_schema: A Pydantic BaseModel class or a JSON schema dictionary.

    Returns:
        The transformed JSON schema compatible with both OpenAI and Anthropic.

    Examples:
        >>> class MyModel(BaseModel):
        ...     name: str
        ...     age: Optional[int] = None
        >>> transform_schema(MyModel)
        {'type': 'object', 'properties': {...}, 'required': ['name', 'age'], ...}
    """
    if inspect.isclass(json_schema) and issubclass(json_schema, pydantic.BaseModel):
        json_schema = json_schema.model_json_schema()

    strict_schema: dict[str, Any] = {}
    json_schema = {**json_schema}

    ref = json_schema.pop("$ref", None)
    if ref is not None:
        strict_schema["$ref"] = ref
        return strict_schema

    defs = json_schema.pop("$defs", None)
    if defs is not None:
        strict_defs: dict[str, Any] = {}
        strict_schema["$defs"] = strict_defs

        for name, schema in defs.items():
            strict_defs[name] = transform_schema(schema)

    type_: SupportedTypes | None = json_schema.pop("type", None)
    any_of = json_schema.pop("anyOf", None)
    one_of = json_schema.pop("oneOf", None)
    all_of = json_schema.pop("allOf", None)

    # Reject unsupported composition keywords per OpenAI strict mode
    unsupported_keywords = ["not", "dependentRequired", "dependentSchemas", "if", "then", "else"]
    for keyword in unsupported_keywords:
        if keyword in json_schema:
            raise ValueError(
                f"Unsupported JSON Schema keyword '{keyword}' for OpenAI strict mode. "
                f"See: https://platform.openai.com/docs/guides/structured-outputs"
            )

    if is_list(any_of):
        strict_schema["anyOf"] = [transform_schema(cast("dict[str, Any]", variant)) for variant in any_of]
    elif is_list(one_of):
        # Convert oneOf to anyOf for compatibility (OpenAI supports anyOf but not oneOf in strict mode)
        strict_schema["anyOf"] = [transform_schema(cast("dict[str, Any]", variant)) for variant in one_of]
    elif is_list(all_of):
        # allOf is not supported in OpenAI strict mode
        raise ValueError(
            "allOf is not supported in OpenAI strict mode. "
            "Consider flattening the schema or using anyOf instead. "
            "See: https://platform.openai.com/docs/guides/structured-outputs"
        )
    else:
        if type_ is None:
            raise ValueError("Schema must have a 'type', 'anyOf', 'oneOf', or 'allOf' field.")

        strict_schema["type"] = type_

    description = json_schema.pop("description", None)
    if description is not None:
        strict_schema["description"] = description

    title = json_schema.pop("title", None)
    if title is not None:
        strict_schema["title"] = title

    if type_ == "object":
        strict_schema["properties"] = {
            key: transform_schema(prop_schema) for key, prop_schema in json_schema.pop("properties", {}).items()
        }
        json_schema.pop("additionalProperties", None)
        strict_schema["additionalProperties"] = False

        # MODIFICATION: Force all properties to be required
        # Original code: required = json_schema.pop("required", None)
        # This modification extracts all property keys and marks them as required
        properties = strict_schema.get("properties", {})
        if properties:
            strict_schema["required"] = list(properties.keys())

    elif type_ == "string":
        format = json_schema.pop("format", None)
        if format and format in SupportedStringFormats:
            strict_schema["format"] = format
        elif format:
            # add it back so its treated as an extra property and appended to the description
            json_schema["format"] = format
    elif type_ == "array":
        items = json_schema.pop("items", None)
        if items is not None:
            strict_schema["items"] = transform_schema(items)

        min_items = json_schema.pop("minItems", None)
        if min_items is not None and (min_items == 0 or min_items == 1):
            strict_schema["minItems"] = min_items
        elif min_items is not None:
            # add it back so its treated as an extra property and appended to the description
            json_schema["minItems"] = min_items

    elif type_ == "boolean" or type_ == "integer" or type_ == "number" or type_ == "null" or type_ is None:
        pass
    else:
        assert_never(type_)

    # if there are any props leftover then they aren't supported, so we add them to the description
    # so that the model *might* follow them.
    if json_schema:
        description = strict_schema.get("description")
        strict_schema["description"] = (
            (description + "\n\n" if description is not None else "")
            + "{"
            + ", ".join(f"{key}: {value}" for key, value in json_schema.items())
            + "}"
        )

    return strict_schema
