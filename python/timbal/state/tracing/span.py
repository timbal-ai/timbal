from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field


class Span(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    path: str = Field(
        ...,
        description="The path of the runnable.",
    )
    call_id: str = Field(
        ...,
        description="The call id of the runnable.",
    )
    parent_call_id: str | None = Field(
        None,
        description="The parent call id of the runnable.",
    )
    t0: int = Field(
        ...,
        description="The start time of the runnable.",
    )
    t1: int | None = Field(
        None,
        description="The end time of the runnable. Will be None if the runnable has not yet completed.",
    )

    @computed_field
    @property
    def elapsed(self) -> int | None:
        """The elapsed time in milliseconds (t1 - t0). None if span is not yet completed."""
        if self.t1 is None:
            return None
        return self.t1 - self.t0

    input: Any = Field(
        None,
        description="The input of the runnable. Will be None if the runnable has not yet started or if there was an error gathering the input.",
    )
    status: Any | None = Field(  # Any to prevent circular import
        None,
        description="The status of the runnable.",
    )
    output: Any = Field(
        None,
        description="The output of the runnable. Will be None if the runnable has not yet completed or if there was an error.",
    )
    error: Any = Field(
        None,
        description="The error of the runnable. Will be None if the runnable has not yet completed or if there was no error.",
    )
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="The usage of the runnable.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata storage for run-specific metrics and data.",
    )

    runnable: Any = Field(
        None,
        description=(
            "A reference to the runnable being executed. "
            "Can be used to access runnable properties, background tasks, and other runtime attributes. "
            "Will be None when initializing traces from serialized data."
        ),
        exclude=True,
    )
    memory: Any = Field(
        None,
        description=(
            "This field is used by Agent to retrieve message histories between runs. "
            "It can also be used to overwrite what an llm can see and whatnot. "
        ),
        exclude=True,
    )
    # INTERNAL: Exposed as a field instead of PrivateAttr for deserialization support.
    # Do not access directly; use RunContext.get_session() instead.
    session: Any = Field(None, exclude=True)

    _input_dump: Any = PrivateAttr()
    """The dumped/serialized version of input for internal use."""
    _output_dump: Any = PrivateAttr()
    """The dumped/serialized version of output for internal use."""
    _memory_dump: Any = PrivateAttr()
    """The dumped/serialized version of memory for internal use."""
    _session_dump: Any = PrivateAttr()
    """The dumped/serialized version of session for internal use."""

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to use dumped versions of input and output during serialization."""
        data = super().model_dump(**kwargs)  # Pydantic ignores private attributes by default
        # Use dumped versions if available, otherwise fall back to originals
        if hasattr(self, "_input_dump"):
            data["input"] = self._input_dump
        if hasattr(self, "_output_dump"):
            data["output"] = self._output_dump
        if hasattr(self, "_memory_dump"):
            data["memory"] = self._memory_dump
        if hasattr(self, "_session_dump"):
            data["session"] = self._session_dump
        return data
