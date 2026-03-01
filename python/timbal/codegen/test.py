from ..state import RunContext, set_run_context
from ..utils import ImportSpec


async def run_test(
    import_spec: ImportSpec,
    params: dict,
    run_context: RunContext | None = None,
) -> tuple[list[dict], dict | None]:
    """Execute a single run of a Timbal runnable.

    Args:
        import_spec: Resolved ImportSpec pointing to the runnable.
        params: Input parameters to pass to the runnable.
        run_context: Optional RunContext to set before execution.

    Returns:
        Tuple of (all_events, output_event_dict_or_None).
    """
    if run_context is not None:
        set_run_context(run_context)

    all_events: list[dict] = []
    output_event: dict | None = None
    runnable = import_spec.load()
    async for event in runnable(**params):
        event_dict = event.model_dump()
        all_events.append(event_dict)
        if event.type == "OUTPUT":
            output_event = event_dict

    return all_events, output_event
