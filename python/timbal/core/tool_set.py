from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from .runnable import Runnable


class ToolSet(ABC, BaseModel):
    """Abstract base class for dynamic tool resolution.

    ToolSet enables agents to work with tools that are resolved at runtime
    rather than being statically defined. This pattern is useful for:

    - **Context-dependent tools**: Tools that should only be available under
      certain conditions (e.g., user permissions, environment state)
    - **Lazy loading**: Deferring tool initialization until they're actually needed
    - **Dynamic configuration**: Tools that need runtime parameters or state
    - **Conditional availability**: Tools that may or may not be available based
      on iteration count or other execution context
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @abstractmethod
    async def resolve(self) -> list[Runnable]:
        """Resolve and return the list of tools to make available.

        This method is called by the agent during execution to dynamically
        determine which tools should be provided to the LLM. It is invoked
        before each LLM call (up to max_iter iterations).

        Returns:
            A list of Tool instances that should be available to the agent.
            Can return an empty list if no tools should be available in the
            current context.
        """
        pass
