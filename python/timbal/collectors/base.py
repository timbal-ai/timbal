import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from functools import wraps
from typing import Any


class BaseCollector(ABC):
    """Base abstract class for all event collectors with internal state management."""
    
    def __init__(self, async_gen: AsyncGenerator[Any, None], **kwargs: Any): # noqa: ARG002
        self._async_gen = async_gen
        self._collected = False

    def __aiter__(self):
        """Return the async iterator object (self).
        
        In Python's async iterator protocol:
        - __aiter__() is called when you do `async for item in obj:`
        - It should return an object that implements __anext__()
        - We return `self` because this class implements __anext__()
        
        This is equivalent to how regular iterators work:
        - __iter__() returns an iterator object with __next__()
        
        Returns:
            Self as the async iterator
        """
        return self

    async def __anext__(self):
        """Get the next item from the async iterator.
        
        This is called by `async for` loops and is the core of the async iterator protocol.
        When the generator is exhausted, it raises StopAsyncIteration to signal completion.
        
        We intercept each event and cache it in self._events for later use by collect().
        
        Returns:
            The next Event from the underlying async generator
            
        Raises:
            StopAsyncIteration: When the generator is exhausted
        """
        try:
            # Get the next event from the wrapped generator
            event = await self._async_gen.__anext__()
            # Cache OutputEvent directly when we encounter it
            processed_event = self.process(event)
            if asyncio.iscoroutine(processed_event):
                processed_event = await processed_event
            return processed_event
        except StopAsyncIteration:
            # The generator is exhausted - mark as collected and re-raise
            self._collected = True
            raise  # This stops the `async for` loop
    
    async def aclose(self):
        """Close the generator gracefully.
        
        This is called when the generator needs to be cleaned up,
        either explicitly or when the generator is garbage collected.
        """
        await self._async_gen.aclose()
        self._collected = True
    
    async def collect(self) -> Any:
        """Collect the final output by consuming the entire stream.
        
        This method consumes all remaining events from the async generator
        and returns the final OutputEvent. It can be called multiple times
        safely - subsequent calls return the cached result.
        
        How this works:
        1. If we already have the OutputEvent cached, return it immediately
        2. Otherwise, consume remaining events using `async for event in self:`
           - This calls our __anext__() method which caches the OutputEvent when found
        3. Return the cached OutputEvent
        
        Returns:
            The final OutputEvent from the stream, or None if no OutputEvent was yielded
        """
        # If we already found and cached the OutputEvent, return it
        if self._collected:
            return self.result()
        # Generator not fully consumed yet - consume remaining events
        try:
            # This calls our __aiter__() and __anext__() methods
            # __anext__() will cache the OutputEvent when it encounters it
            async for _ in self:
                # We could break early if we found the OutputEvent, but typically
                # the OutputEvent is the last event, so we consume everything
                pass
        except StopAsyncIteration:
            pass  # Expected when generator is exhausted
        # Return the cached OutputEvent (will be None if no OutputEvent was yielded)
        return self.result()

    @classmethod
    def wrap(cls, func):
        """Decorator that wraps async generator return with BaseCollector."""
        @wraps(func)
        def wrapper(self, **kwargs) -> cls:
            return cls(async_gen=func(self, **kwargs))
        return wrapper
    
    @classmethod
    @abstractmethod
    def can_handle(cls, event: Any) -> bool:
        """Check if this collector can handle the given event type."""
        pass
    
    @abstractmethod
    def process(self, event: Any) -> Any:
        """Process the event and update internal state.
        
        Args:
            event: The event to process
            
        Returns:
            Processed content if available for streaming, None otherwise
        """
        pass

    @abstractmethod
    def result(self) -> Any:
        """Return the final result."""
        pass
