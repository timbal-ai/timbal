import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import structlog
import yaml
from dotenv import find_dotenv, load_dotenv

from .. import __version__
from ..logs import setup_logging
from ..state import RunContext
from ..types.models import dump
from ..server.utils import ModuleSpec, load_module

logger = structlog.get_logger("timbal.evals")


from pydantic import BaseModel, field_validator

from ..types.message import Message
from ..types.file import File

from collections.abc import Callable


Validator = Callable[[Message], bool]


class Turn(BaseModel):
    """"""

    input: Message
    steps: str | dict[str, Any] | None = None
    output: Message | list[Validator]

    @field_validator("input", mode="before")
    def validate_input(cls, v):
        if isinstance(v, str):
            return Message.validate(v)
        elif isinstance(v, dict):
            content = []
            if "files" in v:
                if isinstance(v["files"], list):
                    for file in v["files"]:
                        content.append(File.validate(file))
                else:
                    raise ValueError("files must be a list")
            elif "text" in v:
                content.append(v["text"])
            return Message.validate(content)
        raise ValueError("input must be a str or dict")
    
    @field_validator("output", mode="before")
    def validate_output(cls, v):
        
        if isinstance(v, str):
            return Message.validate(v)
        elif isinstance(v, dict):
            validators = []
            for validator, ref in v.items():
                if validator == "contains":
                    if not isinstance(ref, list):
                        raise ValueError("contains must be a list")
                    validators.append(lambda output: all([x in output for x in ref]))
                # TODO
                else:
                    logger.error("unknown_validator", validator=validator)

            return validators

        raise ValueError("output must be a fixed string or a dict of validators")


class Test(BaseModel):
    """"""

    name: str
    description: str | None = None
    turns: list[Turn]


class Config(BaseModel):
    """"""

    tests: list[Test]


def discover_files(path: Path) -> list[Path]:
    """"""
    files = []
    if path.is_dir():
        for file in path.rglob("eval*.yaml"):
            files.append(file)
    else:
        if not path.name.endswith(".yaml"):
            raise ValueError(f"Invalid evals path: {path}")
        files.append(path)

    return files


def eval_file(path: Path) -> Any:
    """"""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, list):
        config = {"tests": config}
    config = Config.model_validate(config)
    print(config)


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

    files = discover_files(evals_path)
    print(files)

    for file in files:
        eval_file(file)
