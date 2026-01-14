from typing import Any


class TimbalError(Exception):
    """Base class for all Timbal errors."""


class APIKeyNotFoundError(TimbalError):
    """Error raised when an API key is not found."""


class EarlyExit(TimbalError):
    """Error raised when an early exit is requested.

    Args:
        message: Optional message describing the early exit reason.
        propagate: If True, the early exit propagates to parent orchestrators.
                   If False, it only exits the current runnable. Defaults to True.
    """

    def __init__(self, message: str = "", propagate: bool = True):
        super().__init__(message)
        self.message = message
        self.propagate = propagate


class EvalError(TimbalError):
    """Error raised when an eval test fails."""


class FileStateError(TimbalError):
    """Base exception for file state tracking errors."""


class FileModifiedError(FileStateError):
    """File changed since last read."""


class FileNotReadError(FileStateError):
    """File must be read before editing."""


class InterruptError(TimbalError):
    """Error raised when an interrupt is requested.

    Args:
        call_id: The ID of the call that was interrupted.
        output: Optional partial output collected before interruption.
        message: Optional message describing the interruption reason.
    """

    def __init__(self, call_id: str, output: Any = None, message: str = "") -> None:
        super().__init__(message)
        self.call_id = call_id
        self.output = output
        self.message = message


class ImageProcessingError(TimbalError):
    """Error raised when an image file cannot be processed."""


class PDFProcessingError(TimbalError):
    """Error raised when a PDF file cannot be processed."""


class PlatformError(TimbalError):
    """Error raised when a platform API call fails."""


def bail(message: str | None = None, propagate: bool = True) -> None:
    """Raise an EarlyExit error with the given message.

    Args:
        message: Optional message describing the early exit reason.
        propagate: If True, the early exit propagates to parent orchestrators.
                   If False, it only exits the current runnable. Defaults to True.
    """
    raise EarlyExit(message or "", propagate=propagate)
