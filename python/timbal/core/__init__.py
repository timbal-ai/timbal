# pyright: reportUnsupportedDunderAll=false

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .mcp import MCPServer
    from .memory_compaction import MemoryCompactor
    from .skill import Skill
    from .tool import Tool
    from .tool_set import ToolSet
    from .workflow import Workflow

__all__ = [
    "Agent",
    "drop_tool_use_and_results",
    "keep_last_n_messages",
    "keep_last_n_tokens",
    "keep_last_n_turns",
    "MCPServer",
    "MemoryCompactor",
    "replace_tool_results_with_placeholder",
    "Skill",
    "summarize_old_messages",
    "Tool",
    "ToolSet",
    "truncate_message_tokens",
    "Workflow",
]

_LAZY_IMPORTS = {
    "Agent": ".agent",
    "drop_tool_use_and_results": ".memory_compaction",
    "keep_last_n_messages": ".memory_compaction",
    "keep_last_n_tokens": ".memory_compaction",
    "keep_last_n_turns": ".memory_compaction",
    "MCPServer": ".mcp",
    "MemoryCompactor": ".memory_compaction",
    "replace_tool_results_with_placeholder": ".memory_compaction",
    "Skill": ".skill",
    "summarize_old_messages": ".memory_compaction",
    "Tool": ".tool",
    "ToolSet": ".tool_set",
    "truncate_message_tokens": ".memory_compaction",
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
