import contextlib
import json
import sys

from ..state import RunContext, set_run_context
from ..utils import ImportSpec


async def run_test(
    import_spec: ImportSpec,
    params: dict,
    run_context: RunContext | None = None,
    stream: bool = False,
) -> None:
    """Execute a single run of a Timbal runnable.

    Args:
        import_spec: Resolved ImportSpec pointing to the runnable.
        params: Input parameters to pass to the runnable.
        run_context: Optional RunContext to set before execution.
        stream: Whether to stream events as they are produced.
    """
    if run_context is not None:
        set_run_context(run_context)

    output_event = None
    protocol_stdout = sys.stdout

    with contextlib.redirect_stdout(sys.stderr):
        runnable = import_spec.load()
        async for event in runnable(**params):
            if stream:
                print(json.dumps(event.model_dump()), file=protocol_stdout, flush=True)
            elif event.type == "OUTPUT":
                output_event = event

    if not stream and output_event is not None:
        print(json.dumps(output_event.model_dump()), file=protocol_stdout)
