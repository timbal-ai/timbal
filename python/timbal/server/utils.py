import importlib.util
import socket
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ModuleSpec(BaseModel):
    path: Path
    object_name: str | None = None


def load_module(module_spec: ModuleSpec) -> Any:
    """Load an object from a module."""
    path = module_spec.path 
    object_name = module_spec.object_name

    spec = importlib.util.spec_from_file_location(path.stem, path.as_posix())
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if object_name:
            if hasattr(module, object_name):
                obj = getattr(module, object_name)
                return obj
            else:
                raise ValueError(f"Module {path} has no object {object_name}")
        else:
            raise NotImplementedError("? support loading entire module")
    else:
        raise ValueError(f"Failed to load module {path}")


def is_port_in_use(port: int) -> bool:
    """Check if a TCP port is currently in use on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0
