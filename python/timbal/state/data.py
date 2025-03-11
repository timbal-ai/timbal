"""Data handling system for Timbal.

This module provides a type-safe way to handle data flow between steps. It supports
three main types of data:

Usage:

1. Direct values:
   >>> data = TypeAdapter(Data).validate_python({
   ...     "type": "value",
   ...     "value": "test"
   ... })
   >>> assert data.resolve() == "test"

2. Error values:
   >>> data = TypeAdapter(Data).validate_python({
   ...     "type": "error",
   ...     "error": "error message"
   ... })
   >>> assert data.resolve() == "error message"

3. Mapped values (referencing other steps):
   >>> data = TypeAdapter(Data).validate_python({
   ...     "type": "map",
   ...     "ancestor_id": "step_1",
   ...     "ancestor_key": "return",
   ... })
   >>> context = {
   ...     "step_1.return": TypeAdapter(Data).validate_python({
   ...         "type": "value",
   ...         "value": "test"
   ...     })
   ... }
   >>> assert data.resolve(context_data=context) == "test"

All data types are validated using Pydantic and can be created from Python dictionaries
using TypeAdapter(Data).validate_python(). The resolve() method is used to get the
final value, with mapped data requiring context from other steps.
"""

import re
from abc import ABC, abstractmethod
from typing import (
    Annotated,
    Any,
    Literal,
)

import structlog
from pydantic import BaseModel, ConfigDict, Discriminator

from ..errors import DataKeyError

DATA_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_.]{0,128}$")
DATA_KEY_INTERPOLATION_PATTERN = re.compile(r"{{([a-zA-Z0-9_.]{0,128})}}")

logger = structlog.get_logger("timbal.state.data")


class BaseData(BaseModel, ABC):
    """Abstract base class for all data types in Timbal.

    It uses Pydantic for validation and supports extra fields and assignment validation.
    """
    # Allow storing extra fields in the model.
    # Validate dynamic assignments to the model.
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    type: str
    """The type identifier for the data class."""

    @abstractmethod 
    def resolve(self, context_data: dict[str, "BaseData"] | None = None) -> Any:
        pass


def get_data_key(
    data: dict[str, BaseData], 
    data_key: str,
    default: Any | None = "__NO_DEFAULT__", # We don't use an object() sentinel value here to avoid issues with serialization.
) -> Any:
    """Retrieves the value associated with a given data key from the flow's data dictionary during execution.

    This method supports nested keys using dot notation and allows indexing into lists. 

    Args:
        data: The data dictionary containing all flow execution data.
        data_key: The key to look up in the data dictionary, which can include 
                  nested keys separated by dots and list indices.
        default: The default value to return if the key is not found.
                 If not provided, a DataKeyError will be raised.

    Returns:
        Any: The resolved value associated with the specified data key.
    """
    if not DATA_KEY_PATTERN.match(data_key):
        raise ValueError(f"Invalid data key: {data_key}. Expected a string of 0-128 alphanumeric characters, underscores, or dots.")

    data_key_parts = data_key.split(".")
    data_key_builder = ""

    data_key_found = False
    data_value = None
    for i, data_key_part in enumerate(data_key_parts): # noqa: B007
        if len(data_key_builder):
            data_key_builder = f"{data_key_builder}.{data_key_part}"
        else:
            data_key_builder = data_key_part

        if data_key_builder in data:
            data_instance = data[data_key_builder]
            assert isinstance(data_instance, BaseData), "Data value should be an instance of BaseData."
            data_value = data_instance.resolve(context_data=data)
            data_key_found = True
            break
    
    if not data_key_found:
        if default == "__NO_DEFAULT__":
            raise DataKeyError(f"Cannot find data key '{data_key}'.")
        else:
            return default
    
    if data_value is None and default != "__NO_DEFAULT__":
        return default

    # When a data key is found, we need to check if we need to grab nested items or properties.
    if len(data_key_parts) > i + 1:
        remaining_key_parts = data_key_parts[i+1:]
        for remaining_key_part in remaining_key_parts:

            # Allowing indexing into lists.
            if remaining_key_part.isdigit():
                remaining_key_part = int(remaining_key_part)
                if not isinstance(data_value, list) or len(data_value) <= remaining_key_part:
                    raise KeyError(f"Cannot get {remaining_key_part} from {data_key_builder}: {data_value}.")
                data_key_builder = f"{data_key_builder}.{remaining_key_part}"
                data_value = data_value[remaining_key_part]
                continue

            # Else we assume we're working with a string key.

            elif isinstance(data_value, dict) and remaining_key_part in data_value:
                data_key_builder = f"{data_key_builder}.{remaining_key_part}"
                data_value = data_value[remaining_key_part]
                continue

            elif hasattr(data_value, remaining_key_part):
                data_key_builder = f"{data_key_builder}.{remaining_key_part}"
                data_value = getattr(data_value, remaining_key_part)
                continue

            raise KeyError(f"Cannot get data key '{data_key}'. " \
                           f"Cannot get '{remaining_key_part}' from '{data_key_builder}': {data_value}.")

    return data_value


class DataValue(BaseData):
    """Represents a direct value in the data system.

    This class handles direct values and supports string interpolation using {{key}} syntax
    when the value is a string.
    """

    type: Literal["value"] = "value"
    value: Any
    """The actual value to be stored."""

    def resolve(self, context_data: dict[str, BaseData] | None = None) -> Any:
        data_value = self.value

        # TODO Should we perform interpolation here for nested structures, e.g. list[str] ? 
        if isinstance(self.value, str):
            replacement_keys = DATA_KEY_INTERPOLATION_PATTERN.findall(self.value)
            if replacement_keys:
                assert isinstance(context_data, dict), \
                    f"Cannot resolve data key '{replacement_keys}' in string '{self.value}', " \
                    f"because context_data is not defined."
                for replacement_key in replacement_keys:
                    replacement_value = get_data_key(context_data, replacement_key)
                    # TODO If this is an LLM message, we need to dump the pydantic model.
                    # Directly add the value as a string, no need to care for python evals.
                    data_value = data_value.replace(f"{{{{{replacement_key}}}}}", str(replacement_value))

        return data_value


class DataError(BaseData):
    """Represents an error value in the data system.

    This class is used to store and propagate error messages or error states.
    """

    type: Literal["error"] = "error"
    error: Any
    """The error value or message to be stored."""

    def resolve(self, context_data: dict[str, BaseData] | None = None) -> Any: # noqa: ARG002
        # We return the error as is, because we're matching on isisntance of the class.
        return self


class DataMap(BaseData):
    """Represents a reference to another data value in the system.

    This class allows referencing values from other steps using a key path.
    The key can use dot notation to access nested values.
    """

    type: Literal["map"] = "map"
    key: str
    """The key path to the referenced data."""
    default: Any | None = "__NO_DEFAULT__"
    """The default value to return if the key is not found."""

    def resolve(self, context_data: dict[str, BaseData] | None = None) -> Any:
        assert isinstance(context_data, dict), \
            f"Cannot resolve data key '{self.key}' because context_data is not defined."
        return get_data_key(context_data, self.key, default=self.default)


Data = Annotated[
    DataValue | DataError | DataMap,
    Discriminator("type"),
]
