# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ...state.tracing.trace import Trace
from .base import BaseValidator


class AnyValidator(BaseValidator):
    # TODO contains
    # TODO not_contains
    def __init__(self, min, max, contains=[], not_contains=[], **kwargs) -> None:
        super().__init__(type="any", **kwargs)

        if min == ".":
            min = None
        if min is not None:
            min = int(min)
            if min < 0:
                raise ValueError("any! validator min must be a non-negative integer")
        self.min = min

        if max == ".":
            max = None
        if max is not None:
            max = int(max)
            if max < 1:
                raise ValueError("any! validator max must be greater than 1")
        self.max = max

    def __repr__(self) -> str:
        """Return a readable representation of the validator."""
        params = []
        if self.min is not None:
            params.append(f"min={self.min}")
        if self.max is not None:
            params.append(f"max={self.max}")

        params_str = ", ".join(params) if params else ""
        return f"{self.type}!({params_str})"

    @override
    async def run(self, trace: Trace, **kwargs) -> bool:
        spans = trace.get_level(self.path)
        if min is not None and len(spans) < min:  # type: ignore
            return False
        if max is not None and len(spans) > max:  # type: ignore
            return False
        return True
