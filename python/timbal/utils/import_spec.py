import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel

# Modules ``load()`` has registered in ``sys.modules``. Lets a re-load of the
# same entry replace its own earlier registration while never displacing a
# module somebody else imported (an entry file named ``types.py`` or
# ``timbal.py`` must not clobber the real one).
_owned_modules: "weakref.WeakSet[ModuleType]" = weakref.WeakSet()


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
        """Load and return the target object from the module.

        The module's directory is added to ``sys.path`` and left there. It
        used to be removed once the module finished executing, which broke
        any sibling import that runs later than module top-level — e.g. a
        ``from helper import x`` inside a handler, only reached at request
        time. Top-level imports were already cached in ``sys.modules`` so
        they kept working, which made this look like a deploy problem
        ("existing files update, new files never arrive") when it was
        really the import path vanishing under the running code. Locally
        the cwd masked it; in a sandbox with a different cwd it didn't.

        The module is also registered in ``sys.modules`` under its stem
        before it runs, exactly like a regular import. Without that, a
        sibling doing ``from main import CONST`` finds nothing cached and
        re-executes the entry file — a second ``Agent`` built, tools and
        tracing set up twice, and ``isinstance`` failing between the two
        copies of every class it defines. Registration is skipped when the
        name is already taken by a module we didn't create.
        """
        spec = importlib.util.spec_from_file_location(self.path.stem, self.path.as_posix())
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            module_dir = str(self.path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            existing = sys.modules.get(spec.name)
            registered = existing is None or existing in _owned_modules
            if registered:
                sys.modules[spec.name] = module
                _owned_modules.add(module)
            try:
                spec.loader.exec_module(module)
            except BaseException:
                if registered:
                    if existing is None:
                        sys.modules.pop(spec.name, None)
                    else:
                        sys.modules[spec.name] = existing
                raise

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
