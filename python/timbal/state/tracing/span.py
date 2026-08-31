from typing import Any, ClassVar

from ..._slots import SlotModel, dump_value


class Span(SlotModel):
    """A single runnable invocation inside a run's trace.

    Plain ``__slots__`` class (constructed once per runnable call — hot path).
    Preserves the old pydantic surface:

    - ``model_dump()`` with ``elapsed`` included and ``input``/``output``/
      ``memory``/``session`` replaced by their dumped versions when available.
    - ``runnable``/``memory``/``session`` excluded from dumps (``memory`` and
      ``session`` only appear via their ``_*_dump`` counterparts).
    - Tolerant construction from serialized records: unknown keys are kept and
      re-emitted on dump (old ``extra="allow"``), ``elapsed`` is recomputed.
    - ``status`` is stored as-is: a live span holds a ``RunStatus``; a span
      reloaded from a provider keeps the plain dict (intentional).
    """

    __slots__ = (
        "path",
        "call_id",
        "parent_call_id",
        "t0",
        "t1",
        "input",
        "status",
        "output",
        "error",
        "usage",
        "metadata",
        "runnable",
        "memory",
        "session",
        "_input_dump",
        "_output_dump",
        "_memory_dump",
        "_prev_memory_dump",
        "_session_dump",
        "_emit_sink",
        "_extra",
    )

    _FIELDS = ("path", "call_id", "parent_call_id", "t0", "t1", "input", "status", "output", "error", "usage", "metadata")

    # pydantic-compat: evals introspect Span.model_fields.keys() to know which
    # names address span properties (includes non-serialized fields).
    model_fields: ClassVar[dict[str, None]] = dict.fromkeys(_FIELDS + ("runnable", "memory", "session"))

    def __init__(
        self,
        *,
        path: str,
        call_id: str,
        t0: int,
        parent_call_id: str | None = None,
        t1: int | None = None,
        input: Any = None,
        status: Any = None,
        output: Any = None,
        error: Any = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
        runnable: Any = None,
        memory: Any = None,
        session: Any = None,
        **extra: Any,
    ) -> None:
        # _extra must be set first: __getattr__ reads it for unknown-attribute
        # lookups and would otherwise recurse while the slot is still unset.
        extra.pop("elapsed", None)  # Computed field in dumps; always recomputed from t0/t1.
        object.__setattr__(self, "_extra", extra or None)
        self.path = path
        """The path of the runnable."""
        self.call_id = call_id
        """The call id of the runnable."""
        self.parent_call_id = parent_call_id
        """The parent call id of the runnable."""
        self.t0 = t0
        """The start time of the runnable."""
        self.t1 = t1
        """The end time of the runnable. Will be None if the runnable has not yet completed."""
        self.input = input
        """The input of the runnable."""
        self.status = status
        """The status of the runnable."""
        self.output = output
        """The output of the runnable."""
        self.error = error
        """The error of the runnable."""
        self.usage = usage if usage is not None else {}
        """The usage of the runnable."""
        self.metadata = metadata if metadata is not None else {}
        """Flexible metadata storage for run-specific metrics and data."""
        self.runnable = runnable
        """A reference to the runnable being executed. Excluded from dumps.
        Will be None when initializing traces from serialized data."""
        self.memory = memory
        """Used by Agent to retrieve message histories between runs. Excluded from dumps."""
        # INTERNAL: accepted on construction for deserialization support.
        # Do not access directly; use RunContext.get_session() instead.
        self.session = session
        # Per-call emit sink. None until the first RunContext.emit() (or a
        # background spawn rebinds it). Must be initialized so the streaming
        # hot path can do `span._emit_sink is not None` without AttributeError.
        self._emit_sink = None

    @property
    def elapsed(self) -> int | None:
        """The elapsed time in milliseconds (t1 - t0). None if span is not yet completed."""
        if self.t1 is None:
            return None
        return self.t1 - self.t0

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails: unset _*_dump slots and
        # extra keys from deserialized records land here.
        if not name.startswith("_"):
            extra = object.__getattribute__(self, "_extra")
            if extra and name in extra:
                return extra[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        # Old pydantic config was extra="allow": handlers may stash arbitrary
        # attributes on their span (read by siblings via step_span()). Unknown
        # public names go to _extra and are included in model_dump().
        try:
            object.__setattr__(self, name, value)
        except AttributeError:
            if name.startswith("_"):
                raise
            extra = object.__getattribute__(self, "_extra")
            if extra is None:
                extra = {}
                object.__setattr__(self, "_extra", extra)
            extra[name] = value

    def model_dump(self, mode: str = "python", **_kwargs: Any) -> dict[str, Any]:
        """Serialize the span, preferring the dumped versions of input/output/memory/session."""
        data: dict[str, Any] = {}
        for field in self._FIELDS:
            data[field] = dump_value(getattr(self, field), mode)
            if field == "t1":
                data["elapsed"] = self.elapsed
        extra = self._extra
        if extra:
            data.update(extra)
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
