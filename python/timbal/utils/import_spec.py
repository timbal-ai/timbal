import importlib.util
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ImportSpec(BaseModel):
    """Specification for importing an object from a Python module."""

    path: Path
    target: str | None = None

    @classmethod
    def from_fqn(cls, fqn: str, base_path: Path | None = None) -> "ImportSpec":
        """Parse a 'path/to/file.py::object_name' string into an ImportSpec.

        Args:
            fqn: FQN string in the format 'path/to/file.py::object_name'.
            base_path: Optional base directory to resolve relative paths against.
                       If omitted, paths are resolved relative to cwd.
        """
        parts = fqn.split("::")
        if len(parts) != 2:
            raise ValueError(f"Invalid FQN {fqn!r}. Expected format: path/to/file.py::object_name")
        path = Path(parts[0])
        if base_path is not None:
            path = base_path / path
        return cls(path=path.expanduser().resolve(), target=parts[1])

    def load(self) -> Any:
        """Load and return the target object from the module."""
        spec = importlib.util.spec_from_file_location(self.path.stem, self.path.as_posix())
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            module_dir = str(self.path.parent)
            added = module_dir not in sys.path
            if added:
                sys.path.insert(0, module_dir)
            try:
                spec.loader.exec_module(module)
            finally:
                if added:
                    sys.path.remove(module_dir)

            if self.target:
                if hasattr(module, self.target):
                    obj = getattr(module, self.target)
                    return obj
                else:
                    raise ValueError(f"Module {self.path} has no target {self.target}")
            else:
                raise NotImplementedError("Does not support loading entire module")
        else:
            raise ValueError(f"Failed to load module {self.path}")
