class TimbalError(Exception):
    """Base class for all Timbal errors."""


class DataKeyError(TimbalError):
    """Error raised when a data key is not found."""


class StepKeyError(TimbalError):
    """Error raised when trying to set a step param that does not exist."""


class InvalidLinkError(TimbalError):
    """Error raised when trying to add a link between two steps that is invalid 
    (for some reason specified in the message)."""
