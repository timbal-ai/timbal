class TimbalError(Exception):
    """Base class for all Timbal errors."""

class APIKeyNotFoundError(TimbalError):
    """Error raised when an API key is not found."""

class EarlyExit(TimbalError):
    """Error raised when an early exit is requested."""

class EvalError(TimbalError):
    """Error raised when an eval test fails."""

class FileStateError(TimbalError):
    """Base exception for file state tracking errors."""

class FileModifiedError(FileStateError):
    """File changed since last read."""

class FileNotReadError(FileStateError):
    """File must be read before editing."""

class InterruptError(TimbalError):
    """Error raised when an interrupt is requested."""

class ImageProcessingError(TimbalError):
    """Error raised when an image file cannot be processed."""

class PDFProcessingError(TimbalError):
    """Error raised when a PDF file cannot be processed."""

class PlatformError(TimbalError):
    """Error raised when a platform API call fails."""

def bail(message: str | None = None) -> None:
    """Raise an EarlyExit error with the given message."""
    raise EarlyExit(message or "")
