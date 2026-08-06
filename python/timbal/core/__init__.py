# pyright: reportUnsupportedDunderAll=false

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .fallback_model import FallbackModel, ModelEntry
    from .mcp import MCPServer
    from .memory_compaction import MemoryCompactor  # noqa: F401 - type alias
    from .skill import Skill
    from .test_model import TestModel
    from .tool import Tool
    from .tool_result_offload import LocalOffloadStore, OffloadStore, Spill, ToolResultLimit, Truncate
    from .tool_set import ToolSet
    from .workflow import Workflow

__all__ = [
    "Agent",
    "FallbackModel",
    "LocalOffloadStore",
    "MCPServer",
    "ModelEntry",
    "OffloadStore",
    "Skill",
    "Spill",
    "TestModel",
    "Tool",
    "ToolResultLimit",
    "ToolSet",
    "Truncate",
    "Workflow",
]

_LAZY_IMPORTS = {
    "Agent": ".agent",
    "FallbackModel": ".fallback_model",
    "LocalOffloadStore": ".tool_result_offload",
    "MCPServer": ".mcp",
    "ModelEntry": ".fallback_model",
    "OffloadStore": ".tool_result_offload",
    "Skill": ".skill",
    "Spill": ".tool_result_offload",
    "TestModel": ".test_model",
    "Tool": ".tool",
    "ToolResultLimit": ".tool_result_offload",
    "ToolSet": ".tool_set",
    "Truncate": ".tool_result_offload",
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
