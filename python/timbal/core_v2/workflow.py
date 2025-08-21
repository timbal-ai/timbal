from collections.abc import AsyncGenerator
from functools import cached_property
from typing import Any, override

import structlog
from pydantic import (
    BaseModel,
    PrivateAttr,
    computed_field,
)

from .runnable import Runnable, RunnableLike
from .tool import Tool

logger = structlog.get_logger("timbal.core_v2.workflow")


class Workflow(Runnable):
    """"""

    _steps: list[Runnable] = PrivateAttr(default_factory=list)
    """List of steps to execute in the workflow."""


    # NOTE: No need to add @override since pydantic doesn't have `model_post_init` as an abstract method.
    def model_post_init(self, __context: Any) -> None:
        """"""
        self._path = self.name

        # Workflows are always orchestrators with async generator handlers
        self._is_orchestrator = True
        self._is_coroutine = False
        self._is_gen = False
        self._is_async_gen = True


    @override
    def nest(self, parent_path: str) -> None:
        """See base class."""
        self._path = f"{parent_path}.{self.name}"
        # Update paths for internal LLM and all tools
        for step in self._steps:
            step.nest(self._path)
    

    @override 
    @computed_field
    @cached_property
    def params_model(self) -> BaseModel:
        """See base class."""
        pass


    @override 
    @computed_field
    @cached_property
    def return_model(self) -> Any:
        """See base class."""
        pass


    # TODO Add kwargs -> will be the data maps and whatnot
    # TODO Add a condition param
    def add_step(self, step: RunnableLike, **kwargs: Any) -> "Workflow":
        """"""
        if not isinstance(step, Runnable):
            if isinstance(step, dict):
                step = Tool(**step)
            else:
                step = Tool(handler=step)
            
        if any(step.name == s.name for s in self._steps):
            raise ValueError(f"Step {step.name} already exists in the workflow.")
            
        # TODO Add more stuff to the add_step() method
        step.default_params.update(kwargs)

        step.nest(self._path)
        self._steps.append(step)

        return self


    async def handler(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """"""
        # Extract parent call ID for tracing nested execution
        _parent_call_id = kwargs.pop("_parent_call_id", None)
        # TODO
