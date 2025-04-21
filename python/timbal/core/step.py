import inspect
from collections.abc import Callable
from typing import Any

import structlog
from pydantic import BaseModel, TypeAdapter

from ..state import RunContext
from ..types.models import create_model_from_argspec
from .base import BaseStep

logger = structlog.get_logger("timbal.core.step")


class Step(BaseStep):
    """A class representing a processing step in a workflow.

    The Step class encapsulates a handler function and provides functionality to:
    - Validate input parameters using Pydantic models
    - Validate return values using Pydantic models
    - Execute the handler function with proper parameter passing
    - Support both synchronous and asynchronous execution

    The handler function's signature is automatically analyzed to create Pydantic models
    for parameter validation and return type checking. These models are accessible
    through the params_model() and return_model() methods.

    Attributes:
        handler_fn: The function that implements the step's processing logic
        handler_params_model: Pydantic model for validating input parameters
        handler_return_model: Pydantic model for validating return values
        is_llm: Flag indicating if this step involves LLM processing
        is_coroutine: Flag indicating if the handler is a coroutine function
        is_async_gen: Flag indicating if the handler is an async generator function
    """


    def __init__(
        self, 
        id: str,
        path: str | None = None,
        handler_fn: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Overrides the __init__ method that is inherited from BaseStep, i.e. a pydantic BaseModel.

        Args:
            id: The unique identifier for the step (within a flow or subflow).
            path: Optional path for the flow. Use when directly adding this to a flow or subflow.
            handler_fn: The handler function to be used.
            **kwargs: Additional keyword arguments will be passed to the BaseStep's __init__ method.
        """
        if path is None:
            path = id

        super().__init__(id=id, path=path, **kwargs)
        
        if not inspect.isfunction(handler_fn):
            raise TypeError(f"handler_fn must be a function, got {type(handler_fn)}")

        self.handler_fn = handler_fn
        handler_argspec = inspect.getfullargspec(handler_fn)
        self.handler_params_model = create_model_from_argspec(
            name=f"Step_{self.id}_params", 
            argspec=handler_argspec
        )
        self.handler_params_model_schema = self.handler_params_model.model_json_schema()

        self.handler_return_model = handler_argspec.annotations.get("return", Any)
        self.handler_return_model_schema = TypeAdapter(self.handler_return_model).json_schema()

        # Initialize as False, it will be set to True in the Flow.add_llm method.
        self.is_llm = False
        # These are used to determine if we need to run the step in an executor.
        self.is_coroutine = inspect.iscoroutinefunction(handler_fn)
        self.is_async_gen = inspect.isasyncgenfunction(handler_fn)

    
    def prefix_path(self, prefix: str) -> None:
        """Prefix the step's path with a given path."""
        self.path = f"{prefix}.{self.id}"


    def params_model(self) -> BaseModel:
        """Returns the Pydantic model defining the expected parameters for this step."""
        return self.handler_params_model
        

    def params_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the step's parameter model."""
        return self.handler_params_model_schema
    

    def return_model(self) -> Any:
        """Returns the expected return type for this step."""
        return self.handler_return_model


    def return_model_schema(self) -> dict[str, Any]:
        """Returns the JSON schema for the step's return value model."""
        return self.handler_return_model_schema


    def run(
        self, 
        context: RunContext | None = None, # noqa: ARG002
        **kwargs: Any,
    ) -> Any:
        """Executes the step's processing logic."""
        return self.handler_fn(**kwargs)
    