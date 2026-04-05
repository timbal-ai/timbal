from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..trace import Trace

if TYPE_CHECKING:
    from ...context import RunContext


class TracingProvider(ABC):
    """Abstract base class for tracing providers.

    Providers are responsible for persisting and retrieving run traces. Timbal
    calls ``put()`` at the end of every run to store the completed trace, and
    ``get()`` at the start of a run to retrieve the parent's trace when session
    chaining is active (i.e. when ``run_context.parent_id`` is set).

    **Why class-based, not instance-based?**

    Providers are passed as types, not instances::

        agent = Agent(..., tracing_provider=MyProvider)

    This keeps the call-site ergonomic and lets the framework call provider
    methods without managing provider lifecycle. It also means shared state
    (e.g. an in-memory store) is naturally available across all runs without
    the caller explicitly threading an instance through every call.

    **Configuration**

    Use ``configured()`` to create a provider subclass with specific class-level
    attributes set, avoiding global state conflicts between providers::

        provider = MyProvider.configured(endpoint="http://...", api_key="...")
        agent = Agent(..., tracing_provider=provider)

    Each subclass returned by ``configured()`` has its own independent state.

    **Implementing a custom provider**

    Subclass ``TracingProvider``, declare any class-level attributes your
    implementation needs, and implement ``get()`` and ``put()``::

        class MyProvider(TracingProvider):
            endpoint: str = ""

            @classmethod
            async def get(cls, run_context):
                # Return the parent run's Trace, or None
                ...

            @classmethod
            async def put(cls, run_context):
                # Persist run_context._trace
                ...

    See ``JsonlTracingProvider`` for a reference implementation.
    """

    @classmethod
    def configured(cls, **kwargs) -> type["TracingProvider"]:
        """Return a configured subclass with the given class-level attributes.

        Creates a new subclass with the specified attributes set at the class
        level. The original class is never mutated, so multiple independent
        configurations can coexist safely.

        Args:
            **kwargs: Class-level attributes to set on the new subclass.
                      What attributes are valid depends on the provider — see
                      the provider's class docstring for the full list.

        Returns:
            A new subclass of this provider with the given attributes applied.
        """
        return type(cls.__name__, (cls,), kwargs)

    @classmethod
    @abstractmethod
    async def get(cls, run_context: "RunContext") -> Trace | None:
        """Retrieve the parent run's trace for session chaining.

        Called at the start of a run when ``run_context.parent_id`` is set.
        Should return the ``Trace`` from the parent run, or ``None`` if not
        found or if session chaining is not supported by this provider.

        Args:
            run_context: The current run context. Use ``run_context.parent_id``
                         to look up the parent run.

        Returns:
            The parent run's Trace, or None.
        """
        pass

    @classmethod
    @abstractmethod
    async def put(cls, run_context: "RunContext") -> None:
        """Persist the completed run's trace.

        Called at the end of every run (in a finally block — always executes).
        The full trace is available as ``run_context._trace``.

        Args:
            run_context: The current run context. Persist ``run_context._trace``
                         keyed by ``run_context.id``.
        """
        pass
