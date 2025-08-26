import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .. import __version__
from ..core.runnable import Runnable
from ..utils import ImportSpec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timbal probe script.")
    parser.add_argument(
        "-v", 
        "--version", 
        action="store_true", 
        help="Show version and exit."
    )
    parser.add_argument(
        "--import_spec",
        dest="import_spec",
        type=str,
        help="Path to a python module and optional object (format: path/to/file.py::object_name)",
    )
    args = parser.parse_args()

    if args.version:
        print(f"timbal.server.http {__version__}", file=sys.stderr) # noqa: T201
        sys.exit(0)

    load_dotenv()
    # We can overwrite the env configuration with the --import_spec flag
    import_spec = args.import_spec
    if not import_spec:
        import_spec = os.getenv("TIMBAL_RUNNABLE")
        if not import_spec:
            import_spec = os.getenv("TIMBAL_FLOW") # Legacy
            if import_spec:
                print("TIMBAL_FLOW environment variable is deprecated. Please use TIMBAL_RUNNABLE instead.", file=sys.stderr) # noqa: T201

    if not import_spec:
        print("No import spec provided. Set TIMBAL_RUNNABLE env variable or use --import_spec to specify a module to load.", file=sys.stderr) # noqa: T201
        sys.exit(1)

    import_parts = import_spec.split("::")
    if len(import_parts) != 2:
        print("Invalid import spec format. Use 'path/to/file.py::object_name' or 'path/to/file.py'", file=sys.stderr) # noqa: T201
        sys.exit(1)
    import_path, import_target = import_parts
    import_spec = ImportSpec(
        path=Path(import_path).expanduser().resolve(), 
        target=import_target,
    )

    redirect = io.StringIO()
    with contextlib.redirect_stdout(redirect):
        runnable = import_spec.load()

    if not isinstance(runnable, Runnable):
        raise ValueError("The loaded module is not a valid Runnable instance.")

    output = {
        "version": __version__,
        "type": runnable.__class__.__name__,
        "params_model_schema": runnable.params_model_schema,
        "return_model_schema": runnable.return_model_schema,
    }
    print(json.dumps(output)) # noqa: T201
