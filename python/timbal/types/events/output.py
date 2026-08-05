from typing import Any

from ...types.run_status import RunStatus
from .base import BaseEvent


class OutputEvent(BaseEvent):
    """Event emitted when a step completes with its full output."""

    __slots__ = ("input", "status", "output", "error", "t0", "t1", "usage", "metadata", "_input_dump", "_output_dump")

    type = "OUTPUT"

    _FIELDS = BaseEvent._FIELDS + ("input", "status", "output", "error", "t0", "t1", "usage", "metadata")

    def __init__(
        self,
        *,
        run_id: str,
        path: str,
        call_id: str,
        status: RunStatus | dict[str, Any],
        t0: int,
        t1: int,
        parent_run_id: str | None = None,
        parent_call_id: str | None = None,
        input: Any = None,
        output: Any = None,
        error: Any = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(
            run_id=run_id,
            path=path,
            call_id=call_id,
            parent_run_id=parent_run_id,
            parent_call_id=parent_call_id,
        )
        # Preserve the pydantic-era invariant that a status is required: several
        # regression tests pin that an OutputEvent can never be built without one.
        if isinstance(status, dict):
            status = RunStatus(**status)
        elif not isinstance(status, RunStatus):
            raise ValueError(f"OutputEvent status is required and must be a RunStatus or dict, got {status!r}.")
        self.input = input
        """The input arguments passed to the runnable."""
        self.status = status
        """The status summary of the runnable after it completed."""
        self.output = output
        """The result of the runnable."""
        self.error = error
        """The error that occurred during the runnable."""
        self.t0 = t0
        """The start time of the runnable in milliseconds."""
        self.t1 = t1
        """The end time of the runnable in milliseconds."""
        self.usage = usage if usage is not None else {}
        """The usage of the runnable."""
        self.metadata = metadata if metadata is not None else {}
        """Additional metadata about the runnable."""
        # _input_dump / _output_dump are assigned post-construction by the
        # framework (dumped/serialized versions of input/output for internal use).

    def model_dump(self, mode: str = "python", **kwargs: Any) -> dict[str, Any]:
        """Override model_dump to use dumped versions of input and output during serialization."""
        data = super().model_dump(mode=mode, **kwargs)
        # Use dumped versions if available, otherwise fall back to originals
        if hasattr(self, "_input_dump"):
            data["input"] = self._input_dump
        if hasattr(self, "_output_dump"):
            data["output"] = self._output_dump
        return data
