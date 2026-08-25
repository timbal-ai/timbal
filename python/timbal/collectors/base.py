import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from functools import wraps
from typing import Any


class BackgroundRequest:
    """Awaitable result of a cooperative foreground-to-background request."""

    def __init__(self, future: asyncio.Future[dict[str, Any]], consumer_task: asyncio.Task | None) -> None:
        self._future = future
        self._consumer_task = consumer_task
        future.add_done_callback(self._consume_unretrieved_exception)

    @staticmethod
    def _consume_unretrieved_exception(future: asyncio.Future[dict[str, Any]]) -> None:
        if not future.cancelled():
            future.exception()

    def __await__(self):
        if not self._future.done() and asyncio.current_task() is self._consumer_task:
            raise RuntimeError(
                "Cannot await background() from the task consuming this stream. "
                "Keep consuming, then await the request after the placeholder is emitted."
            )
        return self._future.__await__()

    def done(self) -> bool:
        return self._future.done()

    def result(self) -> dict[str, Any]:
        return self._future.result()


class BaseCollector(ABC):
    """Base abstract class for all event collectors with internal state management."""
    
    def __init__(
        self,
        async_gen: AsyncGenerator[Any, None],
        background_handoff: Any = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        self._async_gen = async_gen
        self._collected = False
        self._background_handoff = background_handoff
        self._consumer_task: asyncio.Task | None = None

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
        self._consumer_task = asyncio.current_task()
        return self

    async def __anext__(self):
        """Get the next item from the async iterator.
        
        This is called by `async for` loops and is the core of the async iterator protocol.
        When the generator is exhausted, it raises StopAsyncIteration to signal completion.
        
        We intercept each event and process it through the process() method.
        
        Returns:
            The next processed event from the underlying async generator
            
        Raises:
            StopAsyncIteration: When the generator is exhausted
        """
        try:
            while True:
                event = await self._async_gen.__anext__()
                processed = self.process(event)
                if asyncio.iscoroutine(processed):
                    processed = await processed
                if processed is None:
                    continue
                return processed
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

    def background(self, call_id: str | None = None) -> BackgroundRequest:
        """Move one active eligible foreground child to the background.

        This is a cooperative handoff: a streaming child switches ownership at
        its next emitted event. The returned request resolves to the usual
        ``{"task_id", "status": "running"}`` placeholder after cutover.
        Omit ``call_id`` when exactly one ``background_mode="auto"`` child is
        in flight; pass it (from that child's ``START`` / ``DELTA``) when
        several are. Keep consuming this collector while awaiting the future
        from another task; awaiting it inline pauses the producer and
        therefore the handoff.
        """
        if self._collected:
            raise RuntimeError("This run has already finished.")
        if self._background_handoff is None:
            raise RuntimeError("This collector does not support foreground-to-background handoff.")
        future = self._background_handoff.request(call_id=call_id)
        consumer_task = self._consumer_task or asyncio.current_task()
        return BackgroundRequest(future, consumer_task)
    
    async def collect(self) -> Any:
        """Collect the final result by consuming the entire stream.

        This method consumes all remaining events from the async generator
        and returns the final result as determined by the result() method.
        It can be called multiple times safely - subsequent calls return the cached result.

        How this works:
        1. If the stream is already collected, return the result immediately
        2. Otherwise, consume remaining events using `async for event in self:`
           - This calls our __anext__() method which processes events via process()
        3. Return the final result via result()

        Returns:
            The final result from the stream as determined by the result() method
        """
        # If we already collected the stream, return the result
        if self._collected:
            return self.result()
        # Generator not fully consumed yet - consume remaining events
        try:
            # This calls our __aiter__() and __anext__() methods
            # __anext__() will process events and update internal state
            async for _ in self:
                # Continue consuming all events to ensure complete processing
                pass
        except StopAsyncIteration:
            pass  # Expected when generator is exhausted
        # Return the final result from the collector's internal state
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
