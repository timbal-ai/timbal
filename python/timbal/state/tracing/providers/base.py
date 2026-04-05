from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..trace import Trace

if TYPE_CHECKING:
    from ...context import RunContext


class Exporter(ABC):
    """Abstract base class for trace exporters.

    Exporters are write-only sinks attached to a ``TracingProvider``. After
    every run, ``TracingProvider.put()`` calls each exporter in
    ``cls._exporters`` once the primary storage has been written.

    Exporters are attached via ``configured()``::

        provider = JsonlTracingProvider.configured(
            _path=Path("traces.jsonl"),
            _exporters=[OTelExporter(endpoint="http://...")],
        )

    **Why not a full TracingProvider?**

    Some backends (OpenTelemetry, Datadog, Langfuse) are write-only — they
    have no retrieval API for session chaining. Forcing them into the
    ``TracingProvider`` interface would require a fake ``get()`` that always
    returns ``None``. Exporters are the right abstraction: they fire after
    every run and never participate in session chaining.

    **Implementing a custom exporter**

    Subclass ``Exporter`` and implement ``export()``::

        class MyExporter(Exporter):
            async def export(self, run_context):
                # Forward run_context._trace to your backend
                ...
    """

    @abstractmethod
    async def export(self, run_context: "RunContext") -> None:
        """Forward the completed run's trace to an external backend.

        Called after ``TracingProvider._store()`` completes. Exceptions are
        silently caught by the framework to avoid breaking the run.

        Args:
            run_context: The current run context. Read ``run_context._trace``
                         for spans and ``run_context.id`` for the run id.
        """
        pass


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

    **Attaching exporters**

    Pass one or more ``Exporter`` instances via ``_exporters`` to forward
    traces to write-only sinks (OTel, Langfuse, Datadog, etc.) after the
    primary storage has been written::

        provider = JsonlTracingProvider.configured(
            _path=Path("traces.jsonl"),
            _exporters=[OTelExporter(endpoint="http://...")],
        )

    Exporter exceptions are silenced so they never break a run.

    **Implementing a custom provider**

    Subclass ``TracingProvider``, declare any class-level attributes your
    implementation needs, and implement ``get()`` and ``_store()``::

        class MyProvider(TracingProvider):
            endpoint: str = ""

            @classmethod
            async def get(cls, run_context):
                # Return the parent run's Trace, or None
                ...

            @classmethod
            async def _store(cls, run_context):
                # Persist run_context._trace
                ...

    See ``JsonlTracingProvider`` for a reference implementation.
    """

    _exporters: list[Exporter] = []

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
                      Pass ``_exporters=[...]`` to attach write-only sinks.

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
    async def put(cls, run_context: "RunContext") -> None:
        """Persist the completed run's trace and fire all attached exporters.

        Called at the end of every run (in a finally block — always executes).
        Calls ``_store()`` first, then calls each exporter in ``_exporters``
        in order. Exporter exceptions are silenced so they never break a run.

        Args:
            run_context: The current run context. Persist ``run_context._trace``
                         keyed by ``run_context.id``.
        """
        await cls._store(run_context)
        for exporter in cls._exporters:
            try:
                await exporter.export(run_context)
            except Exception:
                pass

    @classmethod
    @abstractmethod
    async def _store(cls, run_context: "RunContext") -> None:
        """Persist the completed run's trace (provider-specific storage).

        Implement this in your provider subclass. Called by ``put()`` before
        exporters are fired.

        Args:
            run_context: The current run context. Persist ``run_context._trace``
                         keyed by ``run_context.id``.
        """
        pass
