"""
Tools for the agent.
"""

# from timbal.state import RunContext
import structlog

logger = structlog.get_logger("timbal.adapters.tools")

async def hang_up_call():
    """
    Hangs up the call.

    This tool should be used to terminate the conversation after the final
    goodbye message has been delivered.
    """
    logger.info("Executing hang_up_call tool.")
    # try:
    #    adapter = context.data.get('adapter')
    #    if adapter and hasattr(adapter, 'stop'):
    #        # The stop method in the adapter should handle the call termination
    #        await adapter.stop()
    #        logger.info("Hang up signal sent via adapter's stop() method.")
    #    else:
    #        logger.error("Could not find a compatible adapter in the context to hang up the call.")
    #except Exception as e:
    #    logger.error(f"Error while trying to hang up call: {e}", exc_info=True) 
    return "Call ended."