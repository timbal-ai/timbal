from pathlib import Path
from typing import Any

import yaml

from ..core.runnable import Runnable
from .models import Eval

CONFIG_FILENAME = "evalconf.yaml"


def discover_config(path: Path) -> dict[str, Any]:
    """Discover evalconf.yaml by walking up the directory tree.

    Searches from the given path upward to find evalconf.yaml.
    If path is a file, starts from its parent directory.
    """
    current = path if path.is_dir() else path.parent

    while current != current.parent:
        config_file = current / CONFIG_FILENAME
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f) or {}
        current = current.parent

    # Check root
    config_file = current / CONFIG_FILENAME
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f) or {}

    return {}


def discover_eval_files(path: Path) -> list[Path]:
    """Discover all eval files given a path.
    If the path is a directory, it will search recursively for files matching the pattern "eval*.yaml".
    If the path is a file, it'll simply check the file is .yaml and return it.
    """
    eval_files = []

    # If path doesn't exist, return empty list
    if not path.exists():
        return eval_files

    if path.is_dir():
        eval_file_set = set(path.rglob("eval*.yaml")) | set(path.rglob("*eval.yaml"))
        eval_files = list(eval_file_set)
    else:
        if not path.name.endswith(".yaml"):
            raise ValueError(f"Invalid evals path: {path}")
        eval_files.append(path)

    return eval_files


def parse_eval_file(path: Path, runnable: Runnable) -> list[Eval]:
    """Parse an eval file and return a list of Eval objects."""
    with open(path) as f:
        evals = yaml.safe_load(f)

    if not isinstance(evals, list):
        raise ValueError(f"Invalid eval file: {path}")

    evals = [Eval.model_validate({"path": path, "runnable": runnable, **eval}) for eval in evals]
    return evals
