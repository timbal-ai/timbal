from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import AsyncGenerator, Any, Optional
from timbal.state.context import RunContext


class BaseAdapter(BaseModel, ABC):
    """
    An abstract base class for all adapters.

    Adapters are responsible for connecting the agent to external services,
    such as messaging platforms (Twilio, Slack), databases, or APIs.
    This class defines a common interface and provides basic functionality.
    """
    type: str

    class Config:
        arbitrary_types_allowed = True

    async def before_agent(self, context: RunContext) -> None:
        """
        Called before the agent processes each turn.
        Can be used to set up context, listen for input, etc.
        This default implementation does nothing. Concrete classes can override it.
        
        Args:
            context: The run context for this agent execution
        """
        pass

    async def after_agent(self, context: RunContext) -> None:
        """
        Called after the agent completes each turn.
        Can be used to send responses, clean up, etc.
        This default implementation does nothing. Concrete classes can override it.
        
        Args:
            context: The run context for this agent execution
        """
        pass

    async def on_error(self, context: RunContext, error: dict[str, Any]) -> None:
        """
        Called when an error occurs during agent execution.
        This default implementation does nothing. Concrete classes can override it.
        
        Args:
            context: The run context for this agent execution
            error: The error information
        """
        pass
