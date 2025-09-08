class TimbalError(Exception):
    """Base class for all Timbal errors."""


class APIKeyNotFoundError(TimbalError):
    """Error raised when an API key is not found."""


class EvalError(TimbalError):
    """Error raised when an eval test fails."""
    

class PlatformError(TimbalError):
    """Error raised when a platform API call fails."""
