from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from agent import CallHandlerAgent

async def hang_up_call() -> str:
    """
    Signals that the conversation should end and the call should be terminated.
    Use this tool AFTER providing the final response to the user (e.g., "Goodbye!").
    The system will handle the actual termination of the call after this tool is used.
    """
    # This function is a signal. The implementation is handled by the calling logic.
    return "Hang up signal received. The call will be terminated." 