class TimbalError(Exception):
    """Base class for all Timbal errors."""

class APIKeyNotFoundError(TimbalError):
    """Error raised when an API key is not found."""

class EvalError(TimbalError):
    """Error raised when an eval test fails."""

class PlatformError(TimbalError):
    """Error raised when a platform API call fails."""

class InterruptError(TimbalError):
    """Error raised when an interrupt is requested."""

class EarlyExit(TimbalError):
    """Error raised when an early exit is requested."""

def bail(message: str | None = None) -> None:
    """Raise an EarlyExit error with the given message."""
    raise EarlyExit(message or "")

class FileStateError(TimbalError):
    """Base exception for file state tracking errors."""
    pass

class FileNotReadError(FileStateError):
    """File must be read before editing."""
    pass

class FileModifiedError(FileStateError):
    """File changed since last read."""
    pass
