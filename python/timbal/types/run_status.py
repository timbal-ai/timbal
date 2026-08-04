from typing import Any

from .._slots import SlotModel

VALID_STATUS_CODES = frozenset({"success", "error", "cancelled", "timeout"})


class RunStatus(SlotModel):
    __slots__ = ("code", "reason", "message")

    _FIELDS = ("code", "reason", "message")

    def __init__(self, *, code: str, reason: str | None = None, message: str | None = None, **_ignored: Any) -> None:
        if code not in VALID_STATUS_CODES:
            raise ValueError(f"Invalid run status code {code!r}. Must be one of {sorted(VALID_STATUS_CODES)}.")
        self.code = code
        """The code associated with the run status."""
        self.reason = reason
        """The reason for the run status."""
        self.message = message
        """The message associated with the run status."""
