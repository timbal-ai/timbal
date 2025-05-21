import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import structlog
from dotenv import find_dotenv, load_dotenv

from .. import __version__
from ..logs import setup_logging
from ..state import RunContext
from ..types.models import dump
from ..server.utils import ModuleSpec, load_module

logger = structlog.get_logger("timbal.evals")


def discover_evals(evals_path: Path) -> list:
    evals_files = []
    if evals_path.is_dir():
        for file in evals_path.rglob("eval*.yaml"):
            evals_files.append(file)
    else:
        if not evals_path.name.endswith(".yaml"):
            raise ValueError(f"Invalid evals path: {evals_path}")
        evals_files.append(evals_path)

    return evals_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timbal evals.")
    parser.add_argument(
        "-v", 
        "--version", 
        action="store_true", 
        help="Show version and exit."
    )
    parser.add_argument(
        "--fqn",
        dest="fqn",
        type=str,
        help="Fully qualified name of the module to be run (format: path/to/file.py::object_name)",
    )
    parser.add_argument(
        "--evals",
        dest="evals",
        type=str,
        help="Path to the evals directory to be run.",
    )
    args = parser.parse_args()

    if args.version:
        print(f"timbal.evals {__version__}") # noqa: T201
        sys.exit(0)

    fqn = args.fqn
    if not fqn:
        print("No timbal app Fully Qualified Name (FQN) provided.") # noqa: T201
        sys.exit(1)

    fqn_parts = fqn.split(":")
    if len(fqn_parts) > 2:
        print("Invalid timbal app Fully Qualified Name (FQN) format. Use 'path/to/file.py:object_name' or 'path/to/file.py'") # noqa: T201
        sys.exit(1)
    elif len(fqn_parts) == 2:
        module_path, module_name = fqn_parts
        module_spec = ModuleSpec(
            path=Path(module_path).expanduser().resolve(), 
            object_name=module_name,
        )
    else:
        module_spec = ModuleSpec(
            path=Path(fqn_parts[0]).expanduser().resolve(), 
            object_name=None,
        )

    evals_path = args.evals
    if not evals_path:
        print("No evals path provided.") # noqa: T201
        sys.exit(1)
    evals_path = Path(evals_path).expanduser().resolve()
    if not evals_path.exists():
        print(f"Evals path {evals_path} does not exist.") # noqa: T201
        sys.exit(1)

    logger.info("loading_dotenv", path=find_dotenv())
    load_dotenv(override=True)
    setup_logging()

    discover_evals(evals_path)
