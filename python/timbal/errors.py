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
