import inspect
from collections.abc import Callable
from functools import cached_property
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from pydantic import BaseModel, SkipValidation, computed_field, model_validator

from ..utils import create_model_from_argspec
from .runnable import Runnable


class Tool(Runnable):
    """A Tool is a Runnable that wraps a callable function or method.
    
    Tools automatically introspect the handler function to:
    - Generate parameter models from function signatures
    - Determine execution characteristics (sync/async, generator, etc.)
    - Extract return type annotations
    - Auto-generate names from function names
    
    Tools are the basic building blocks that can be used standalone or
    composed into more complex Agents and Workflows.
    """

    handler: SkipValidation[Callable[..., Any]]
    """The callable function or method that this tool wraps."""


    @model_validator(mode="before")
    @classmethod
    def validate_handler_and_name(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate handler and auto-generate tool name if not provided.
        
        This validator runs before Pydantic model creation and:
        1. Ensures the handler is a proper function, not a Runnable instance
        2. Automatically extracts the function name to use as the tool name when no explicit name is provided
        
        Args:
            values: Raw input values for tool creation
            
        Returns:
            Updated values dict with name field populated
            
        Raises:
            ValueError: If handler is missing, is a Runnable, has no __name__, or is a lambda without explicit name
        """
        handler = values.get("handler", None)
        if handler is None:
            raise ValueError("You must provide a handler when creating a tool.")
        
        # Check if handler is a Runnable instance
        if isinstance(handler, Runnable):
            raise ValueError(
                "Handler cannot be a Runnable instance. Tools should wrap any other python callable. "
                "If you want to compose Runnables, use an Agent or Workflow instead, or modify the properties of the Runnable itself."
            )
        
        if "name" not in values:
            handler_name = getattr(handler, "__name__", None)
            if handler_name is None:
                raise ValueError(
                    "Handler must be a function or lambda with a __name__ attribute. "
                    "If you are using a callable object or functools.partial, please provide a 'name' explicitly."
                )
            if handler_name == "<lambda>":
                raise ValueError("A name must be specified when using a lambda function as a tool.")
            values["name"] = handler_name

        return values

    
    def model_post_init(self, __context: Any) -> None:
        """Initialize tool-specific attributes after Pydantic model creation.
        
        This method introspects the handler function to determine its execution
        characteristics, which are used by the base Runnable class to determine
        how to execute the handler.
        """
        super().model_post_init(__context)
        self._path = self.name
        
        inspect_result = self._inspect_callable(
            self.handler, 
            allow_required_params=True,
            allow_gen=True,
            allow_async_gen=True,
        )

        self._is_orchestrator = False
        self._is_coroutine = inspect_result["is_coroutine"]
        self._is_gen = inspect_result["is_gen"]
        self._is_async_gen = inspect_result["is_async_gen"]
        self._dependencies = inspect_result["dependencies"]

    
    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"
    
    
    @override
    @computed_field
    @cached_property
    def params_model(self) -> BaseModel:
        """See base class."""
        handler_argspec = inspect.getfullargspec(self.handler)
        params_model_name = self.name.title().replace("_", "") + "Params"
        params_model = create_model_from_argspec(
            name=params_model_name,
            argspec=handler_argspec,
        )
        return params_model


    @override
    @computed_field 
    @cached_property 
    def return_model(self) -> Any:
        """See base class."""
        handler_argspec = inspect.getfullargspec(self.handler)
        handler_return_annotation = handler_argspec.annotations.get("return", Any)
        return handler_return_annotation
