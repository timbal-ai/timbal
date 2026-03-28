# pyright: reportUnsupportedDunderAll=false

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .mcp import MCPServer
    from .memory_compaction import MemoryCompactor  # noqa: F401 - type alias
    from .skill import Skill
    from .tool import Tool
    from .tool_set import ToolSet
    from .workflow import Workflow

__all__ = [
    "Agent",
    "MCPServer",
    "Skill",
    "Tool",
    "ToolSet",
    "Workflow",
]

_LAZY_IMPORTS = {
    "Agent": ".agent",
    "MCPServer": ".mcp",
    "Skill": ".skill",
    "Tool": ".tool",
    "ToolSet": ".tool_set",
    "Workflow": ".workflow",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache to bypass __getattr__ on subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
