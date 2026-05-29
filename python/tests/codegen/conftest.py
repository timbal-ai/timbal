import sys


def codegen_cmd(*args: str) -> list[str]:
    """Build argv for ``python -m timbal.codegen ...`` using the active interpreter."""
    return [sys.executable, "-m", "timbal.codegen", *args]
