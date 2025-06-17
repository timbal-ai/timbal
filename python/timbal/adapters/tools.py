"""
Tools for the agent.
"""

from timbal.state import RunContext
import structlog

logger = structlog.get_logger("timbal.adapters.tools")

async def hang_up_call(call_sid: str) -> str:
    """
    Hangs up a specific call.

    This tool should be used to terminate the conversation after the final
    goodbye message has been delivered.
    
    Args:
        call_sid: The unique identifier of the call to hang up
        
    Returns:
        str: Confirmation message about the call termination
    """
    logger.info(f"Executing hang_up_call tool for call_sid: {call_sid}")
    try:
        # Import here to avoid circular imports
        from timbal.state.context import run_context_var
        
        # Get the current context to find the adapter
        context = run_context_var.get()
        
        # Look for twilio adapter in the agent's adapters
        # The adapter should be accessible through the context or agent
        adapter = None
        if hasattr(context, 'data') and 'adapter' in context.data:
            adapter = context.data['adapter']
        
        if adapter and hasattr(adapter, 'hang_up_call'):
            await adapter.hang_up_call(call_sid)
            logger.info(f"Hang up signal sent for call {call_sid}")
            return "Adiós, que tengas un buen día. La llamada se ha terminado."
        else:
            logger.error("Could not find a compatible adapter to hang up the call.")
            return "Error: No se pudo terminar la llamada."
            
    except Exception as e:
        logger.error(f"Error while trying to hang up call {call_sid}: {e}", exc_info=True)
        return "Error al terminar la llamada."