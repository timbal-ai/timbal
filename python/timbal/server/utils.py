import socket
from pathlib import Path

from pydantic import BaseModel


class ModuleSpec(BaseModel):
    path: Path
    object_name: str | None = None


def is_port_in_use(port: int) -> bool:
    """Check if a TCP port is currently in use on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0
