import argparse
import asyncio
import json
import sys
from pathlib import Path

import structlog
from dotenv import find_dotenv, load_dotenv

from .. import __version__
from ..core.agent import Agent
from ..logs import setup_logging
from ..utils import ImportSpec, dump
from .engine import eval_file
from .types.result import EvalTestSuiteResult
from .utils import discover_files

logger = structlog.get_logger("timbal.eval")


async def run_evals(files, agent, test_results, test_name=None):
    for file in files:
        await eval_file(file, agent, test_results, test_name=test_name)
    return await dump(test_results)

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

    if not args.fqn:
        print("No timbal app Fully Qualified Name (FQN) provided.") # noqa: T201
        sys.exit(1)

    fqn_parts = args.fqn.split("::")
    if len(fqn_parts) != 2:
        print("Invalid timbal app Fully Qualified Name (FQN) format. Use 'path/to/file.py::object_name' or 'path/to/file.py'") # noqa: T201
        sys.exit(1)
    import_path, import_target = fqn_parts
    import_spec = ImportSpec(
        path=Path(import_path).expanduser().resolve(), 
        target=import_target,
    )

    runnable = import_spec.load()

    if not isinstance(runnable, Agent):
        raise ValueError("Only Agents are supported for evaluation.")

    tests_path = args.tests
    if not tests_path:
        print("No tests path provided.") # noqa: T201
        sys.exit(1)
    
    if "::" in tests_path:
        tests_path, test_name = tests_path.split("::", 1)
    else:
        test_name = None
    
    tests_path = Path(tests_path).expanduser().resolve()

    if not tests_path.exists():
        print(f"Tests path {tests_path} does not exist.") # noqa: T201
        sys.exit(1)

    logger.info("loading_dotenv", path=find_dotenv())
    load_dotenv(override=True)
    setup_logging()

    files = discover_files(tests_path)
    logger.info("tests_files", files=files)

    test_results = EvalTestSuiteResult()
    dumped = asyncio.run(run_evals(files, runnable, test_results, test_name))
    with open("summary.json", "w") as f:
        json.dump(dumped, f, indent=2, ensure_ascii=False)