import json
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

    def _provider_identity(self) -> tuple:
        """What makes this set the same provider of tools in another process (a paused call is resumed only on
        a tool of the provider it was made for, see ``Agent._note_pause``): its class, and its configuration,
        the fields (declared or extra, ``name`` among them) whose values are plain JSON. Kept as a digest, so a
        secret among them is not stored. A field holding anything else (a client, a list of tools) is left out:
        its ``str()`` may differ in every process. A subclass whose instances differ otherwise says how.
        """
        fields = {}
        for name, value in (
            *((name, getattr(self, name, None)) for name in type(self).model_fields),
            *(self.__pydantic_extra__ or {}).items(),
        ):
            try:
                fields[name] = json.loads(json.dumps(value, sort_keys=True))
            except (TypeError, ValueError, RecursionError):
                continue
        return (type(self).__module__, type(self).__qualname__, fields)
