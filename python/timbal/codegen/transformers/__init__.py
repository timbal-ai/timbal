import importlib
import pkgutil
from pathlib import Path


def load_modules() -> dict:
    modules = {}
    for info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        mod = importlib.import_module(f"timbal.codegen.transformers.{info.name}")
        modules[info.name] = mod
    return modules
