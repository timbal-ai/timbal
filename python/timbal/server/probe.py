import argparse
import json
import os
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv

from .. import Agent, Flow, __version__
from ..logs import setup_logging
from .utils import ModuleSpec, load_module

logger = structlog.get_logger("timbal.server.probe")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timbal probe script.")
    parser.add_argument(
        "-v", 
        "--version", 
        action="store_true", 
        help="Show version and exit."
    )
    parser.add_argument(
        "--module_spec",
        dest="module_spec",
        type=str,
        help="Path to a python module and optional object (format: path/to/file.py::object_name)",
    )
    args = parser.parse_args()

    if args.version:
        print(f"timbal.servers.http {__version__}") # noqa: T201
        sys.exit(0)

    # We can overwrite the env TIMBAL_FLOW variable with the --module_spec flag.
    module_spec = args.module_spec
    if not module_spec:
        module_spec = os.getenv("TIMBAL_FLOW")

    if not module_spec:
        print("No module spec provided. Set TIMBAL_FLOW env variable or use --module_spec to specify a module to load.") # noqa: T201
        sys.exit(1)

    module_parts = module_spec.split(":")
    if len(module_parts) > 2:
        print("Invalid module spec format. Use 'path/to/file.py:object_name' or 'path/to/file.py'") # noqa: T201
        sys.exit(1)
    elif len(module_parts) == 2:
        module_path, module_name = module_parts
        module_spec = ModuleSpec(
            path=Path(module_path).expanduser().resolve(), 
            object_name=module_name,
        )
    else:
        module_spec = ModuleSpec(
            path=Path(module_parts[0]).expanduser().resolve(), 
            object_name=None,
        )

    load_dotenv()
    setup_logging()

    flow = load_module(module_spec)

    if not isinstance(flow, (Agent, Flow)):
        raise ValueError("The loaded module is not a valid Agent or Flow instance.")

    params_model_schema = flow.params_model_schema()
    return_model_schema = flow.return_model_schema()

    output = {
        "params_model_schema": params_model_schema,
        "return_model_schema": return_model_schema
    }
    print(json.dumps(output)) # noqa: T201
