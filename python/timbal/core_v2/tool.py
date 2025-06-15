import inspect
from collections.abc import Callable
from functools import cached_property
from typing import Any, override

from pydantic import BaseModel, computed_field, model_validator

from ..types.models import create_model_from_argspec
from .runnable import Runnable


class Tool(Runnable):
    """"""

    handler: Callable[..., Any]
    """"""


    @model_validator(mode="before")
    @classmethod
    def validate_name(cls, values: dict[str, Any]) -> dict[str, Any]:
        """"""
        if "name" not in values:
            handler = values.get("handler", None)
            if handler is None:
                raise ValueError("You must provide a handler when creating a tool.")
            
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

    
    # NOTE: No need to add @override since pydantic doesn't have `model_post_init` as an abstract method.
    def model_post_init(self, __context: Any) -> None:
        """"""
        self._path = self.name

        self._is_coroutine = inspect.iscoroutinefunction(self.handler)
        self._is_gen = inspect.isgeneratorfunction(self.handler)
        self._is_async_gen = inspect.isasyncgenfunction(self.handler)

    
    @override
    def nest(self, parent_path: str) -> None:
        """"""
        self._path = f"{parent_path}.{self.name}"
    
    
    @override
    @computed_field
    @cached_property
    def params_model(self) -> BaseModel:
        """"""
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
        """"""
        handler_argspec = inspect.getfullargspec(self.handler)
        handler_return_annotation = handler_argspec.annotations.get("return", Any)
        return handler_return_annotation
