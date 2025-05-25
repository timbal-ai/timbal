import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import structlog
import yaml
from dotenv import find_dotenv, load_dotenv

from .. import __version__
from ..core.agent import Agent
from ..logs import setup_logging

# TODO This shouldn't be in just the server 'module'.
from ..server.utils import ModuleSpec, load_module
from .types.test_suite import TestSuite
from .utils import discover_files

logger = structlog.get_logger("timbal.eval")


async def eval_file(
    path: Path,
    _agent: Agent
) -> Any:
    """Parse and run all the tests in the given file."""
    with open(path) as f:
        test_suite = yaml.safe_load(f)

    test_suite = TestSuite.model_validate(test_suite)
    
    # TODO Run tests.


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
        "--tests",
        dest="tests",
        type=str,
        help="Path to the tests file or directory to be run.",
    )
    args = parser.parse_args()

    if args.version:
        print(f"timbal.eval {__version__}") # noqa: T201
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

    agent = load_module(module_spec)

    # from timbal.state.savers import JSONLSaver
    # agent.state_saver = JSONLSaver(path=Path("state.jsonl"))

    tests_path = args.tests
    if not tests_path:
        print("No tests path provided.") # noqa: T201
        sys.exit(1)
    tests_path = Path(tests_path).expanduser().resolve()
    if not tests_path.exists():
        print(f"Tests path {tests_path} does not exist.") # noqa: T201
        sys.exit(1)

    logger.info("loading_dotenv", path=find_dotenv())
    load_dotenv(override=True)
    setup_logging()

    files = discover_files(tests_path)
    logger.info("tests_files", files=files)

    # TODO Improve this.
    for file in files:
        asyncio.run(eval_file(file, agent))
