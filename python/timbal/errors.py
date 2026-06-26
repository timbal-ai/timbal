from typing import Any


class TimbalError(Exception):
    """Base class for all Timbal errors."""


class APIKeyNotFoundError(TimbalError):
    """Error raised when an API key is not found."""


class CredentialNotAvailable(APIKeyNotFoundError, ValueError):
    """Raised when a tool credential can't be resolved from any source.

    Subclasses ValueError for backward compat with tool code/tests. Carries
    structured fields (provider_name, missing, env_vars) so callers (e.g. a
    remote service-account executor) can react without parsing the message.
    """

    def __init__(
        self,
        provider_name: str,
        *,
        missing: list[str] | None = None,
        env_vars: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.missing = list(missing or [])
        self.env_vars = list(env_vars or [])
        super().__init__(message or self._default_message())

    def _default_message(self) -> str:
        parts = [f"{self.provider_name} credentials not found."]
        if self.missing:
            parts.append(f"Missing: {', '.join(self.missing)}.")
        hints = ["pass them on the tool", "configure an Integration"]
        if self.env_vars:
            hints.append(f"set the {', '.join(self.env_vars)} environment variable(s)")
        parts.append("To fix: " + ", or ".join(hints) + ".")
        return " ".join(parts)


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


class PauseRequired(TimbalError):
    """Internal signal used to propagate a paused child run up the call stack.

    One rail for both pause kinds: an approval gate (reason
    ``approval_required``/``approval_denied``) and a ``suspend()`` call (reason
    ``input_required``). Lets an agent/workflow stop its loop and bubble the
    child's paused ``OutputEvent`` up to the top-level run, which then persists
    for resume. The carried ``output_event.status.reason`` is the only thing
    that distinguishes the kinds — handling is identical.
    """

    def __init__(self, output_event: Any) -> None:
        super().__init__("Paused")
        self.output_event = output_event


class Suspend(TimbalError):
    """Control-flow signal raised by ``suspend()`` from inside a handler.

    Pauses the run and surfaces ``payload`` to the caller (rendered as an
    ``InteractionEvent``). On resume — same ``parent_id`` plus a ``resume`` map
    carrying ``{suspension_id: value}`` — the handler re-executes from the top
    and ``suspend()`` returns the supplied value instead of raising.

    This is the general form of the approval gate: approval resumes with a
    ``bool``, ``suspend`` resumes with an arbitrary value.
    """

    def __init__(
        self,
        suspension_id: str,
        payload: dict[str, Any],
        kind: str = "suspend",
        response_schema: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("Suspended")
        self.suspension_id = suspension_id
        self.payload = payload
        self.kind = kind
        self.response_schema = response_schema


class RunCancelled(TimbalError):
    """Control-flow signal: a human cancelled the run while resolving a pause.

    Raised when a ``Cancel`` value is supplied on the ``resume=`` channel for a
    pending approval gate or ``suspend()`` call. Unwinds the handler and marks
    the run ``cancelled`` / reason ``cancelled``. Unlike a denial, nothing is
    fed back to the model — the whole run stops.
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(message or "Run cancelled.")
        self.message = message or "Run cancelled."


class ApprovalPolicyError(TimbalError):
    """Error raised when an approval policy callable (requires_approval or
    approval_prompt) raises an exception. Surfaces as a span with status
    ``error`` and reason ``approval_policy_error`` so operators can distinguish
    policy bugs from runnable handler errors."""

    def __init__(self, runnable_path: str, original: BaseException) -> None:
        super().__init__(f"Approval policy raised in {runnable_path}: {original!r}")
        self.runnable_path = runnable_path
        self.original = original


class ImageProcessingError(TimbalError):
    """Error raised when an image file cannot be processed."""


class PDFProcessingError(TimbalError):
    """Error raised when a PDF file cannot be processed."""


class PlatformError(TimbalError):
    """Error raised when a platform API call fails.

    Carries the HTTP ``status_code`` (when the failure came from an HTTP
    response) so callers can branch on it — e.g. treat 404/501 as "endpoint
    not implemented" rather than a hard failure.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FallbackExhausted(TimbalError):
    """Error raised when all models in a fallback chain fail."""

    def __init__(self, errors: list[tuple[str, BaseException]]) -> None:
        self.errors = errors
        models = ", ".join(model for model, _ in errors)
        super().__init__(f"All {len(errors)} fallback models failed: {models}")


class WorkflowStepError(TimbalError):
    """Raised when a workflow step fails and the workflow should report error status.

    Carries the original step error dict (when available) so orchestrators can
    surface it on the workflow span without losing the child traceback.
    """

    def __init__(self, step_name: str, step_error: dict[str, Any] | None = None) -> None:
        self.step_name = step_name
        self.step_error = step_error
        message = step_error.get("message") if step_error else f"Step '{step_name}' failed"
        super().__init__(message or f"Step '{step_name}' failed")


class SpanNotFound(TimbalError):
    """Error raised when trying to access a span that doesn't exist in the trace.

    This is raised by step_span() when the requested step has no span,
    meaning it was skipped or never ran. Used in pull-based workflow skip
    logic to determine if a step can execute based on the availability of
    its dependencies' spans/outputs.

    Args:
        step_name: The name of the step whose span was not found.
        message: Optional message describing the error.
    """

    def __init__(self, step_name: str, message: str = "") -> None:
        super().__init__(message or f"Span not found for step '{step_name}'")
        self.step_name = step_name


def bail(message: str | None = None, propagate: bool = True) -> None:
    """Raise an EarlyExit error with the given message.

    Args:
        message: Optional message describing the early exit reason.
        propagate: If True, the early exit propagates to parent orchestrators.
                   If False, it only exits the current runnable. Defaults to True.
    """
    raise EarlyExit(message or "", propagate=propagate)
